import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
import numpy as np
from jolto import config
from jolto.core.analytics import (
    build_rapisarda_correlation_matrix,
    generate_sobol_bridge_increments,
    generate_prng_increments,
    build_fractional_kernel,
    build_hybrid_rough_driver,
    rough_vol_compensator,
)


class RoughSABRFMM(nn.Module):
    """Rough SABR Forward Market Model on an annual tenor grid, simulated under the terminal measure.

    tenors: grid T_0..T_N in years (N forwards); F0: the N initial forward rates (decimal);
    alpha_f(T) and rho_f(T): ATM normal vol (decimal) and spot-vol correlation per forward, called
    at the fixing times T_0..T_{N-1}; nu: one global vol-of-vol; H: Hurst exponent. Σ comes from
    omega_matrix (Rapisarda angles) or, when None, from an exponential rate-rate decay with rate
    beta_decay. beta_sabr and shift are the CEV exponent and shift; only beta_sabr = 0 (Bachelier)
    is supported. F0 and alphas are the autograd leaves of the PV01 and Vega ladders.
    """

    def __init__(
        self,
        tenors,
        F0,
        alpha_f,
        rho_f,
        nu,
        H,
        beta_decay=0.05,
        beta_sabr=0.0,
        shift=0.0,
        omega_matrix=None,
        device="cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch.float64

        self.register_buffer("T", torch.tensor(tenors, device=device, dtype=self.dtype))
        self.register_buffer("tau", torch.tensor(np.diff(tenors), device=device, dtype=self.dtype))
        self.N = len(F0)

        self.F0 = nn.Parameter(torch.tensor(F0, device=device, dtype=self.dtype))
        self.register_buffer("beta_sabr", torch.tensor(beta_sabr, device=device, dtype=self.dtype))
        self.register_buffer("shift", torch.tensor(shift, device=device, dtype=self.dtype))

        # alpha_f gives ATM normal vols; SABR alpha = normal vol / |F0+δ|^β
        # shortcut: |F0+δ| here but clamp(F0+δ, 0) in the simulation; they agree only for β = 0 or
        # F0+δ ≥ 0, the only cases the library supports. Unify both before enabling β > 0.
        base_normal_vols = torch.tensor(alpha_f(tenors[:-1]), device=device, dtype=self.dtype)
        eta_F0 = torch.pow(torch.abs(self.F0 + self.shift), self.beta_sabr)
        self.alphas = nn.Parameter(base_normal_vols / eta_F0)

        self.register_buffer(
            "rhos", torch.tensor(rho_f(tenors[:-1]), device=device, dtype=self.dtype)
        )
        # One global vol-of-vol: a single vol driver scales every forward.
        self.register_buffer("nu", torch.tensor(float(nu), device=device, dtype=self.dtype))
        self.register_buffer("H", torch.tensor(H, device=device, dtype=self.dtype))

        self.n_factors = self.N + 1  # full rank: 1 vol driver + N forward rates

        if omega_matrix is not None:
            omega_tensor = torch.tensor(omega_matrix, device=device, dtype=self.dtype)
            spatial_corr = build_rapisarda_correlation_matrix(omega_tensor)
        else:
            # fallback: exponential decay when no Rapisarda angles provided
            T_mat_i, T_mat_j = self.T[:-1].unsqueeze(1), self.T[:-1].unsqueeze(0)
            rate_rate_corr = torch.exp(-beta_decay * torch.abs(T_mat_i - T_mat_j))
            spatial_corr = torch.eye(self.N + 1, device=device, dtype=self.dtype)
            spatial_corr[1:, 1:] = rate_rate_corr

        # diagonal jitter keeps the Cholesky stable
        jitter = torch.eye(self.N + 1, device=device, dtype=self.dtype) * 1e-10
        spatial_cov = spatial_corr + jitter
        diag_scale = torch.sqrt(torch.clamp(torch.diagonal(spatial_cov), min=1e-16))
        spatial_corr_pd = spatial_cov / (diag_scale.unsqueeze(1) * diag_scale.unsqueeze(0))
        loadings = torch.linalg.cholesky(spatial_corr_pd)

        # Keep raw Cholesky loadings: L @ L^T must equal the target correlation.
        # Row-normalizing L would alter off-diagonal correlations.
        self.register_buffer("loadings", loadings)

        # strict upper triangle of the rate correlation (eq:drift)
        corr_rates = self.loadings[1:, :] @ self.loadings[1:, :].T
        self.register_buffer("Lambda_upper", torch.triu(corr_rates, diagonal=1))

    def get_terminal_bond(self):
        return torch.prod(1.0 / (1.0 + self.tau * self.F0))

    def differentiable_parameters(self):
        # live autograd leaves (not detached): backward() populates .grad on these
        return {"F0": self.F0, "alphas": self.alphas}

    def read_greeks(self, names=("PV01", "Vega")):
        # maps each leaf's populated .grad to its named ladder; the display ×1e4 stays in the caller
        ladders = {
            "PV01": (self.F0, 1e-4),
            "Vega": (self.alphas, 1e-4),
        }
        out = {}
        for name in names:
            if name not in ladders:
                raise KeyError(f"read_greeks: unknown greek {name!r}; known: {tuple(ladders)}")
            param, scale = ladders[name]
            if param.grad is None:
                raise RuntimeError(
                    f"read_greeks: .grad for {name!r} is None — run price.backward() "
                    f"after requires_grad_(True) on differentiable_parameters()."
                )
            out[name] = param.grad.detach().cpu().numpy() * scale
        return out

    def simulate_forward_curve(
        self,
        n_paths,
        time_grid,
        freeze_drift=True,
        use_checkpoint=False,
        driver_type="sobol",
        seed=42,
        dz_override=None,
        steps=None,
    ):
        """Simulate the forward curve on `time_grid` under the terminal measure.

        `steps` (default None ⇒ full (n_paths, n_steps+1, N) paths): time-grid indices to return, as
        (n_paths, len(steps), N). With frozen drift the path-sized sums are built tape-free and only
        these slices are materialised: same law and draws, roundoff-level differences only.

        `freeze_drift=True` substitutes F₀ for the live path in drift and vol-scale
        (frozen-coefficient predictor); standard BGM approximation (Glasserman–Zhao 2000,
        Hunter–Jäckel–Joshi 2001). `freeze_drift=False` is state-dependent Euler.

        `dz_override` (diagnostic seam, default None): when given, the base increments dz_all are
        taken from it instead of being generated, so callers can drive the sim from an
        externally-coupled Brownian path (W-coupled dt-convergence; the hybrid near-lag noise stays
        internal/uncoupled). Must be (n_paths, n_steps, n_factors).
        """
        n_steps = len(time_grid) - 1
        dt = float(time_grid[1] - time_grid[0])

        # BLP hybrid scheme is gated on config.HYBRID_KAPPA, read at call time (not an import-frozen
        # default arg) so a runtime flip of the flag is honoured. κ=0 is the Volterra-Euler scheme;
        # κ>0 routes branch (ii) through the hybrid near correction. The flag never touches
        # branch (i) → the plain increment driving the rate Euler steps stays identical across κ.
        kappa = config.HYBRID_KAPPA

        n_factors = self.n_factors

        if kappa == 0:
            kernel = build_fractional_kernel(self.H, n_steps, dt, self.device, self.dtype)
            # discrete martingale compensator (Euler ½ν²·ΣK²·dt), shape (1, 1, n_steps)
            var_comp = rough_vol_compensator(self.nu, kernel, None, dt, kappa)
        else:
            # κ>0: the K_tail convolution and the construction-variance compensator are built inside
            # _simulate from the hybrid driver, so they survive checkpoint recompute
            kernel = None
            var_comp = None

        if dz_override is not None:
            if tuple(dz_override.shape) != (n_paths, n_steps, n_factors):
                raise ValueError(
                    f"dz_override shape {tuple(dz_override.shape)} != "
                    f"(n_paths, n_steps, n_factors)=({n_paths}, {n_steps}, {n_factors})"
                )
            dz_all = dz_override
        elif driver_type == "sobol":
            dz_all = generate_sobol_bridge_increments(
                n_paths, n_steps, n_factors, dt, self.device, self.dtype, seed=seed
            )
        elif driver_type == "prng":
            dz_all = generate_prng_increments(
                n_paths, n_steps, n_factors, dt, self.device, self.dtype
            )
        else:
            raise ValueError("driver_type must be 'sobol' or 'prng'")

        def _simulate(dz_all, kernel, var_comp, F0, alphas):
            # branch (i), full Adachi N+1 matrix: the plain increment driving the rate Euler steps.
            # It stays byte-identical across κ; the near correction never reaches dW_all/dW_rates.
            dW_all = torch.matmul(dz_all, self.loadings.T)
            dW_rates = dW_all[..., 1:]  # rate drivers are factors 1..N

            # branch (ii): rough-vol convolution then the L^T projection. Only row 0 of Z̃·L^T
            # (the vol driver) is used, and L is lower-triangular (Cholesky), so row 0 is
            # L[0,0]·(factor-0 convolution): the other n_factors−1 convolutions never reach the
            # price.
            assert not self.loadings[0, 1:].any(), "vol row of loadings must be [L00, 0, ..., 0]"
            dz_vol = dz_all[..., 0]
            if kappa == 0:
                fbm_vol = torch.matmul(dz_vol, kernel.T)
            else:
                # near_gen is created and seeded inside the checkpointed closure so a backward
                # recompute redraws the identical ε; a generator only consumed here would
                # re-advance and desync. seed+7919 mirrors the 1D engine/pricer near-noise offset.
                near_gen = torch.Generator(device=self.device).manual_seed(seed + 7919)
                # antithetic only when the driver carries the half-split: the Sobol path is
                # [dZ_half; −dZ_half] → dz_vol[B/2:] == −dz_vol[:B/2], so ε negates in lockstep.
                # The "prng" path is not antithetic → False (no gain to recover, law unaffected).
                fbm_vol, var_vec, _ = build_hybrid_rough_driver(
                    dz_vol,
                    float(self.H),
                    n_steps,
                    dt,
                    kappa,
                    self.device,
                    self.dtype,
                    near_generator=near_gen,
                    antithetic=(driver_type == "sobol"),
                )
                # construction-variance compensator: var_vec already carries every dt power, so the
                # Euler form's trailing ·dt is dropped
                var_comp = rough_vol_compensator(self.nu, None, var_vec, dt, kappa)

            fbm = (self.loadings[0, 0] * fbm_vol).unsqueeze(-1)  # vol driver Z̃ is factor 0

            # (n_paths, n_steps, 1): one vol path, broadcast across forwards downstream
            unit_vols = torch.exp(self.nu * fbm - var_comp.view(1, -1, 1))

            # predictable (left-point) vol in the Euler updates, V(t₀)=1
            unit_vols_pred = torch.cat(
                [
                    torch.ones(
                        n_paths, 1, unit_vols.shape[-1], device=self.device, dtype=self.dtype
                    ),
                    unit_vols[:, :-1, :],
                ],
                dim=1,
            )
            v2_dt = unit_vols_pred**2 * dt

            if freeze_drift:
                if self.beta_sabr == 0.0:
                    vol_scale = alphas
                else:
                    eta_f0 = torch.pow(torch.clamp(F0 + self.shift, min=0.0), self.beta_sabr)
                    vol_scale = alphas * eta_f0

                omega_0 = (self.tau * vol_scale) / (1.0 + self.tau * F0)
                mu_0 = -vol_scale * torch.matmul(self.Lambda_upper, omega_0)

                if steps is not None:
                    # μ₀ and vol_scale are time-constant, so
                    # F(t_s) = F0 + μ₀·ΣV²dt + vol_scale·ΣV·dW.
                    # The path-sized sums depend on no grad leaf (only F0/α are), so build them in
                    # place without a tape and keep only the requested slices.
                    with torch.no_grad():
                        s_idx = torch.as_tensor(steps, device=self.device, dtype=torch.long)
                        c_idx = (s_idx - 1).clamp(min=0)  # F(t_s) uses increments 0..s−1
                        live = (s_idx > 0).to(self.dtype).view(1, -1, 1)
                        dW_all.mul_(unit_vols_pred).cumsum_(dim=1)  # dW_all is local to this call
                        sum_vdw = dW_all[:, c_idx, 1:] * live
                        sum_v2dt = torch.cumsum(v2_dt, dim=1)[:, c_idx, :] * live
                    return F0 + mu_0.view(1, 1, -1) * sum_v2dt + vol_scale.view(1, 1, -1) * sum_vdw

                dF = mu_0.view(1, 1, -1) * v2_dt
                dF.add_(dW_rates * (unit_vols_pred * vol_scale.view(1, 1, -1)))
                return torch.cat([F0.expand(n_paths, 1, -1), F0 + torch.cumsum(dF, dim=1)], dim=1)

            F_t = F0.expand(n_paths, self.N).clone()
            path_steps = [F_t.unsqueeze(1)]
            for i in range(n_steps):
                if self.beta_sabr == 0.0:
                    vol_scale = alphas
                else:
                    eta_F = torch.pow(torch.clamp(F_t + self.shift, min=0.0), self.beta_sabr)
                    vol_scale = alphas * eta_F

                omega = (self.tau * vol_scale) / (1.0 + self.tau * F_t)
                drift_weights = torch.einsum("ij,pj->pi", self.Lambda_upper, omega)
                drift = -vol_scale * drift_weights * v2_dt[:, i]

                F_t = F_t + drift + dW_rates[:, i] * (unit_vols_pred[:, i] * vol_scale)
                path_steps.append(F_t.unsqueeze(1))
            F_all = torch.cat(path_steps, dim=1)
            return F_all if steps is None else F_all[:, list(steps)]

        if use_checkpoint and torch.is_grad_enabled():
            return cp.checkpoint(
                _simulate, dz_all, kernel, var_comp, self.F0, self.alphas, use_reentrant=False
            )
        return _simulate(dz_all, kernel, var_comp, self.F0, self.alphas)
