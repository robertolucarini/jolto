"""Stage 2: the Rapisarda correlation calibrator (CorrelationCalibrator)."""

import numpy as np
import time
import math
import torch
from scipy.optimize import least_squares, minimize
from jolto.calibration.pricers import (
    _core_fukasawa_ode,
    mapped_smm_mc_pricer,
    precompute_smm_weights,
    apply_sigma_to_smm,
)
from jolto.core.analytics import (
    build_rapisarda_correlation_matrix,
    generate_sobol_bridge_increments,
    rough_vol_compensator,
    find_l_curve_corner,
    compute_sigma_structural_metrics,
    select_lambda_monotone,
)
from jolto.calibration.corr_target import FAMILIES, historical_target
from jolto import config
from jolto.core.utils import DualLogger
import sys
import os

from jolto.calibration.pricers import INV_SQRT_2, INV_SQRT_2PI, _hybrid_driver


def _lambda_grid():
    """The λ-sweep grid: LAMBDA_GRID_N log-spaced points from LAMBDA_GRID_LO to LAMBDA_GRID_HI."""
    return np.logspace(
        np.log10(config.LAMBDA_GRID_LO), np.log10(config.LAMBDA_GRID_HI), config.LAMBDA_GRID_N
    )


def _check_decay_form(form):
    """Raise on an unknown STAGE2_DECAY_FORM: 'linear', 'exp' or 'hist_<FAMILIES key>'."""
    allowed = ("linear", "exp") + tuple(f"hist_{k}" for k in FAMILIES)
    if form not in allowed:
        raise ValueError(f"unknown STAGE2_DECAY_FORM {form!r}; expected one of {allowed}")


class CorrelationCalibrator:
    """Stage 2: fits the Rapisarda angles of the (N+1)×(N+1) correlation matrix Σ.

    atm_vol_matrix: DataFrame of ATM normal vols (decimal), index = expiry, columns = underlying
    tenor, both in years; only quotes whose expiry + tenor falls on the forward grid are fitted.
    model: a RoughSABRFMM built from the Stage-1 marginals, whose ρ_i fix the first column of Σ.
    """

    def __init__(self, atm_vol_matrix, model, smile_data=None):
        self.model = model
        self.device = model.device
        self.N = model.N
        self.grid_T = model.T.cpu().numpy()

        # targets: ATM (strike = 0) + optional off-ATM strikes
        expiries, tenors, strikes, vols = [], [], [], []

        for exp in atm_vol_matrix.index:
            for ten in atm_vol_matrix.columns:
                vol = atm_vol_matrix.loc[exp, ten]
                if not np.isnan(vol):
                    expiries.append(float(exp))
                    tenors.append(float(ten))
                    strikes.append(0.0)
                    vols.append(float(vol))

        if smile_data is not None:
            for strike_offset, vol_matrix in smile_data.items():
                for exp in vol_matrix.index:
                    for ten in vol_matrix.columns:
                        vol = vol_matrix.loc[exp, ten]
                        if not np.isnan(vol):
                            expiries.append(float(exp))
                            tenors.append(float(ten))
                            strikes.append(float(strike_offset))
                            vols.append(float(vol))

        self.expiries = np.array(expiries)
        self.tenors = np.array(tenors)
        self.strikes = np.array(strikes)  # strike offset per target (0 = ATM)
        self.market_vols = np.array(vols)

        self.omega = np.zeros((self.N + 1, self.N + 1))

        # first column anchors to Stage-1 marginal ρ values
        rhos_np = np.clip(self.model.rhos.detach().cpu().numpy(), -0.9999, 0.9999)
        self.omega[1:, 0] = np.arccos(rhos_np)

        # structural PSD constraints
        if self.N > 1:
            self.omega[2, 1] = 0.0
        if self.N > 2:
            self.omega[3:, 2] = np.pi / 2.0

        # Jacobian MC cache persisted across AMMO iterations (fixed Sobol seed -> identical tensor)
        self._jac_cache = None
        # optional driver store shared across calibrations of one model (lambda_rule fold fits)
        self.mc_store = None
        self._pen_plan = None  # _compute_tikhonov_penalties gather plan, keyed by its config

        self._hist_target, self.hist_target_info = None, None
        if config.STAGE2_DECAY_FORM.startswith("hist_"):
            _check_decay_form(config.STAGE2_DECAY_FORM)  # before the history is loaded
            C, self.hist_target_info = historical_target(
                config.STAGE2_HIST_CURVE_DIR,
                config.STAGE2_HIST_END,
                config.STAGE2_HIST_YEARS,
                config.STAGE2_HIST_STEP,
                config.STAGE2_DECAY_FORM[len("hist_") :],
                rhos_np,
                config.GRID_STEP_SIZE,
                config.MAX_MATURITY,
            )
            self._hist_target = torch.tensor(C, device=self.device, dtype=self.model.dtype)

    def _generate_mc_cache(self, n_steps, dt_val, n_paths_val):
        kappa = config.HYBRID_KAPPA
        # the drivers are fixed by these keys for a given model (seed 42) and are only read
        key = (n_steps, dt_val, n_paths_val, kappa)
        if self.mc_store is not None and key in self.mc_store:
            return self.mc_store[key]
        with torch.no_grad():
            dZ_all = generate_sobol_bridge_increments(
                n_paths_val, n_steps, 1, dt_val, self.device, self.model.dtype
            )
            dz_vol = dZ_all[..., 0]
            fbm, var_vec, kernel = _hybrid_driver(
                dz_vol, self.model.H, n_steps, dt_val, kappa, self.device, self.model.dtype
            )
        cache = {
            "fbm": fbm.to(torch.float32),
            "kernel": kernel.to(torch.float32),
            "dz_vol": dz_vol.to(torch.float32),
            "var_vec": var_vec.to(torch.float32) if var_vec is not None else None,
            "hybrid_kappa": kappa,
            # Σ-independent memos, filled by their users; a driver cache belongs to one model.
            # mc_rough_sabr_pricer: vol integrals per expiry step; _compute_mapped_smm_parameters:
            # π-weights per target set; _get_global_hf_vols_and_jac: Jacobian-path integrals;
            # _run_ammo_tr_loop: the HF value and Jacobian at the cold-start anchor
            "vol_integrals": {},
            "smm_plans": {},
            "jac_integrals": {},
            "anchor_hf": {},
        }
        if self.mc_store is not None:
            self.mc_store[key] = cache
        return cache

    def _build_correlation_target_grid(self):
        global_exp, global_ten, global_vol, global_str = [], [], [], []
        r_idx_list, c_idx_list = [], []

        for i in range(2, self.N):
            target_maturity = self.grid_T[i + 1]
            mask = np.abs((self.expiries + self.tenors) - target_maturity) < 1e-4
            exp_targets = self.expiries[mask]

            idx = i + 1
            free_indices = [1] + list(range(3, idx))

            if len(exp_targets) > 0:
                global_exp.extend(exp_targets)
                global_ten.extend(self.tenors[mask])
                global_vol.extend(self.market_vols[mask])
                global_str.extend(self.strikes[mask])
                r_idx_list.extend([idx] * len(free_indices))
                c_idx_list.extend(free_indices)

        if not r_idx_list:
            print("No free indices available for calibration.")
            return None

        global_exp = np.array(global_exp)
        global_ten = np.array(global_ten)
        global_vol = np.array(global_vol)
        global_str = np.array(global_str)
        vol_targets_t = torch.tensor(global_vol, device=self.device, dtype=self.model.dtype)

        r_idx = torch.tensor(r_idx_list, device=self.device, dtype=torch.long)
        c_idx = torch.tensor(c_idx_list, device=self.device, dtype=torch.long)

        n_angles = len(r_idx_list)
        n_opts = len(global_exp)
        print(
            f"-> Global Setup: Optimizing {n_angles} free angles against {n_opts} swaptions simultaneously."
        )

        # SMM cache on unique (expiry, tenor) pairs: (α_smm, ρ_smm) are strike-independent,
        # so deduplication keeps the Rapisarda Jacobian O(n_unique_pairs) regardless of
        # how many strikes per swaption are included.
        pair_keys = np.column_stack([global_exp, global_ten])
        unique_pairs, expand_idx = np.unique(pair_keys, axis=0, return_inverse=True)
        n_unique = len(unique_pairs)
        n_strikes_avg = n_opts / max(n_unique, 1)
        print(
            f"-> Pre-computing SMM pi-weights cache: "
            f"{n_unique} unique (expiry, tenor) pairs "
            f"({n_opts} targets, ~{n_strikes_avg:.1f} strikes/pair)..."
        )
        global_smm_cache = precompute_smm_weights(
            self.model, unique_pairs[:, 0], unique_pairs[:, 1]
        )
        global_smm_cache["expand_idx"] = torch.tensor(
            expand_idx, device=self.device, dtype=torch.long
        )

        current_p = np.full(n_angles, 0.8)
        bounds = [(1e-4, np.pi - 1e-4)] * n_angles

        dt_val = config.CALI_DT
        n_paths_val = config.CALI_PATHS
        # non-empty: a free row was added only when it had targets, and r_idx_list is non-empty
        max_T = float(np.max(global_exp))
        n_steps = int(math.ceil(max_T / dt_val))

        print(f"-> Generating Global MC Cache (Steps: {n_steps}, Paths: {n_paths_val})...")
        global_cached_drivers = self._generate_mc_cache(n_steps, dt_val, n_paths_val)

        current_lambda = config.LAMBDA_CURVATURE
        use_tikhonov = config.USE_TIKHONOV
        is_first_order = config.STAGE_2_FIRST_ORDER
        o_base = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)

        return {
            "global_exp": global_exp,
            "global_ten": global_ten,
            "global_vol": global_vol,
            "global_str": global_str,
            "r_idx": r_idx,
            "c_idx": c_idx,
            "vol_targets_t": vol_targets_t,
            "global_smm_cache": global_smm_cache,
            "current_p": current_p,
            "bounds": bounds,
            "dt_val": dt_val,
            "n_paths_val": n_paths_val,
            "global_cached_drivers": global_cached_drivers,
            "current_lambda": current_lambda,
            "use_tikhonov": use_tikhonov,
            "is_first_order": is_first_order,
            "o_base": o_base,
        }

    def _stage2_decay_target(self, idx):
        """Correlation decay target for free row `idx`, shared by the Tikhonov decay penalty
        and the target-distance diagnostic so the two stay consistent. STAGE2_DECAY_FORM:
            'linear':  clamp(1 − s·d, min=ρ∞), slope s, floor ρ∞
            'exp':     ρ∞ + (1−ρ∞)·exp(−β·d)
            'hist_*':  row of the historical target built in __init__ (the default)
        Both parametric forms equal 1 at d=0."""
        if self._hist_target is not None:
            return self._hist_target[idx - 1, : idx - 1]
        rho_inf = config.STAGE2_DECAY_RHO_INF
        times = torch.tensor(self.grid_T[1:idx], device=self.device, dtype=self.model.dtype)
        distances = self.grid_T[idx] - times
        form = config.STAGE2_DECAY_FORM
        if form == "linear":
            s = config.STAGE2_DECAY_SLOPE
            return torch.clamp(1.0 - s * distances, min=rho_inf)
        if form != "exp":
            _check_decay_form(form)
            raise ValueError(
                f"STAGE2_DECAY_FORM {form!r} needs the historical target, which was not built"
            )
        beta = config.STAGE2_DECAY_BETA
        return rho_inf + (1.0 - rho_inf) * torch.exp(-beta * distances)

    def _decay_target_distance(self, Sigma):
        """RMS of (Σ_row − decay_target) over the free forward rows (correlation units,
        pre-×1e4). Diagnostic only: it separates 'λ too strong' from a mis-specified target.
        Returns None when the decay target is disabled (metric is meaningless there)."""
        if not config.STAGE2_USE_DECAY_TARGET:
            return None
        c0 = config.STAGE2_PENALTY_FIRST_FWD
        diffs = [
            Sigma[i + 1, c0 : i + 1] - self._stage2_decay_target(i + 1)[c0 - 1 :]
            for i in range(2, self.N)
        ]
        if not diffs:
            return 0.0
        return float(torch.sqrt(torch.mean(torch.cat(diffs) ** 2)).item())

    def _compute_tikhonov_penalties(self, Sigma):
        # Smoothness-only fallback: STAGE2_USE_DECAY_TARGET=False drops the decay term (Rv),
        # keeping only the curvature term (Rh), the market-agnostic regularisation path.
        use_decay = config.STAGE2_USE_DECAY_TARGET
        # columns before c0 are left out: the Stage-1 ρ and the RBM anchors fix forwards 1–2
        c0 = config.STAGE2_PENALTY_FIRST_FWD
        # gather indices and targets depend only on these settings; built once, reused per call
        form = (config.STAGE2_DECAY_FORM, config.STAGE2_DECAY_SLOPE, config.STAGE2_DECAY_BETA)
        key = (c0, use_decay, form, config.STAGE2_DECAY_RHO_INF)
        if self._pen_plan is None or self._pen_plan[0] != key:
            self._pen_plan = (key, *self._build_pen_plan(c0, use_decay))
        _, r_s, c_s, r_d, c_d, target, order = self._pen_plan

        pieces = []
        if len(r_s) > 0:
            pieces.append(Sigma[r_s, c_s + 2] - 2.0 * Sigma[r_s, c_s + 1] + Sigma[r_s, c_s])
        if target is not None:
            pieces.append(Sigma[r_d, c_d] - target)
        if pieces:
            return torch.cat(pieces)[order] * 10000.0
        return torch.tensor([], device=self.device, dtype=self.model.dtype)

    def _build_pen_plan(self, c0, use_decay):
        """Index plan for _compute_tikhonov_penalties: per free row, the D2 terms of Σ[idx, c0:idx]
        then its decay residuals, in that order; `order` interleaves the two gathered blocks back
        into it."""
        rows = range(3, self.N + 1)
        n_smooth = sum(max(idx - c0 - 2, 0) for idx in rows)  # decay terms sit after these
        r_s, c_s, r_d, c_d, targets, order = [], [], [], [], [], []
        for idx in rows:
            n = max(idx - c0, 0)
            if n >= 3:
                order += range(len(r_s), len(r_s) + n - 2)
                r_s += [idx] * (n - 2)
                c_s += range(c0, idx - 2)
            if use_decay:
                order += range(n_smooth + len(r_d), n_smooth + len(r_d) + n)
                r_d += [idx] * n
                c_d += range(c0, idx)
                targets.append(self._stage2_decay_target(idx)[c0 - 1 :])

        def _long(x):
            return torch.tensor(list(x), device=self.device, dtype=torch.long)

        target = torch.cat(targets) if targets else None
        return _long(r_s), _long(c_s), _long(r_d), _long(c_d), target, _long(order)

    def _sigma_from_angles(self, o_base, free_indices, p_np):
        """Σ with the free angles set to p_np (numpy) and the others fixed at o_base."""
        p_t = torch.tensor(p_np, device=self.device, dtype=self.model.dtype)
        return build_rapisarda_correlation_matrix(o_base.index_put(free_indices, p_t))

    def _log_penalty_norm(self, Sigma):
        """log‖Tikhonov penalty‖ at Σ (the L-curve ρ); −20 when the penalty vector is empty."""
        with torch.no_grad():
            raw_pen_t = self._compute_tikhonov_penalties(Sigma)
        if len(raw_pen_t) == 0:
            return -20.0
        return np.log(np.linalg.norm(raw_pen_t.cpu().numpy()) + 1e-12)

    def _get_global_hf_vols_and_jac(
        self,
        p_global,
        free_indices,
        exp_targets,
        ten_targets,
        strikes,
        dt_val,
        n_paths_val,
        cached_drivers,
        calculate_jac=False,
        *,
        smm_cache,
    ):
        r_idx, c_idx = free_indices
        device = self.device
        dtype = self.model.dtype

        with torch.no_grad():
            o_base = torch.tensor(self.omega, device=device, dtype=dtype)
            p_t = torch.tensor(p_global, device=device, dtype=dtype)
            o_mod = o_base.index_put((r_idx, c_idx), p_t)
            S_mat = build_rapisarda_correlation_matrix(o_mod)

            v_hf_base_raw = mapped_smm_mc_pricer(
                self.model,
                S_mat,
                exp_targets,
                ten_targets,
                strikes,
                dt=dt_val,
                n_paths=n_paths_val,
                cached_drivers=cached_drivers,
            )
            # mapped_smm_mc_pricer returns a numpy array
            v_hf_base = np.array(v_hf_base_raw, dtype=np.float64)

        J_hf = None

        if calculate_jac:
            n_paths_jac = config.CALI_PATHS_JAC
            n_steps = cached_drivers["dz_vol"].shape[1]

            if self._jac_cache is None or self._jac_cache.get("n_steps") != n_steps:
                self._jac_cache = self._generate_mc_cache(n_steps, dt_val, n_paths_jac)
                self._jac_cache["n_steps"] = n_steps
            jac_drivers = self._jac_cache

            with torch.no_grad():
                dt_t = torch.tensor(dt_val, device=device, dtype=dtype)

                T_t = torch.tensor(exp_targets, device=device, dtype=dtype)
                K_t = torch.tensor(strikes, device=device, dtype=dtype)
                step_idx = torch.clamp(torch.ceil(T_t / dt_val).long() - 1, min=0)
                max_step = int(step_idx.max().item()) + 1

                nu_val = float(self.model.nu)  # ν_SMM = ν, one global scalar

                # Σ-independent path integrals, kept per expiry step in the Jacobian driver cache;
                # a cumsum column does not depend on the truncation length, so the bits are the same
                cols = jac_drivers["jac_integrals"].setdefault((nu_val, float(dt_val), dtype), {})
                steps = step_idx.tolist()
                if any(s not in cols for s in steps):
                    fbm_j = jac_drivers["fbm"].to(dtype).to(device)
                    dz_j = jac_drivers["dz_vol"].to(dtype).to(device)
                    kernel_j = jac_drivers["kernel"].to(dtype).to(device)
                    n_paths_j = fbm_j.shape[0]
                    var_vec_j = (
                        jac_drivers["var_vec"].to(dtype).to(device)
                        if jac_drivers.get("var_vec") is not None
                        else None
                    )
                    hk = jac_drivers.get("hybrid_kappa", 0)
                    # (1, 1, n_steps) -> (n_steps,)
                    var_comp = rough_vol_compensator(nu_val, kernel_j, var_vec_j, dt_t, hk).view(-1)
                    fbm_exp = torch.exp(
                        nu_val * fbm_j[:, :max_step] - var_comp[:max_step].unsqueeze(0)
                    )  # (n_paths, max_step)
                    unit_vols = torch.cat(
                        [torch.ones(n_paths_j, 1, device=device, dtype=dtype), fbm_exp[:, :-1]],
                        dim=-1,
                    )  # (n_paths, max_step)
                    dz_trunc = dz_j[:, :max_step]
                    computed = (
                        torch.cumsum(unit_vols * dz_trunc, dim=-1)[
                            :, step_idx
                        ],  # (n_paths, n_opts)
                        torch.cumsum(unit_vols**2 * dt_t, dim=-1)[:, step_idx],
                        torch.cumsum(dz_trunc, dim=-1)[:, step_idx],
                    )
                    for j, s in enumerate(steps):
                        cols.setdefault(s, tuple(c[:, j].clone() for c in computed))
                I_v_T, I_V2_T, I_cv_T = (
                    torch.stack([cols[s][k] for s in steps], dim=1) for k in range(3)
                )

                a_b, r_b = apply_sigma_to_smm(S_mat, smm_cache)

                K_exp = K_t.unsqueeze(0)  # (1, n_opts)
                is_call = K_t >= 0
                ic_exp = is_call.unsqueeze(0)
                T_exp_u = T_t.unsqueeze(0)
                sign_c = torch.ones(1, device=device, dtype=dtype)
                sign_p = -torch.ones(1, device=device, dtype=dtype)

                def _cmc_payoff(a_e, r_e, iv, iv2, cv_v):
                    """Inline Bachelier CMC payoff (call or put by parity)."""
                    r_comp = torch.sqrt(torch.clamp(1.0 - r_e**2, min=1e-12))
                    F = a_e * r_e * iv
                    Std = a_e * r_comp * torch.sqrt(torch.clamp(iv2, min=1e-12))
                    d = (F - K_exp) / torch.clamp(Std, min=1e-12)
                    d_s = torch.where(ic_exp, d, -d)
                    sgn = torch.where(ic_exp, sign_c, sign_p)
                    pay = sgn * (F - K_exp) * (
                        0.5 * (1 + torch.erf(d_s * INV_SQRT_2))
                    ) + Std * INV_SQRT_2PI * torch.exp(-0.5 * d**2)

                    Std_cv = a_e * r_comp * torch.sqrt(T_exp_u)
                    F_cv = a_e * r_e * cv_v
                    d_cv = (F_cv - K_exp) / torch.clamp(Std_cv, min=1e-12)
                    d_cv_s = torch.where(ic_exp, d_cv, -d_cv)
                    pay_cv = sgn * (F_cv - K_exp) * (
                        0.5 * (1 + torch.erf(d_cv_s * INV_SQRT_2))
                    ) + Std_cv * INV_SQRT_2PI * torch.exp(-0.5 * d_cv**2)
                    return pay, pay_cv

                pay_b, pay_cv_b = _cmc_payoff(
                    a_b.unsqueeze(0), r_b.unsqueeze(0), I_v_T, I_V2_T, I_cv_T
                )

                # c_cv frozen at the base point (eq::pmc)
                cov_b = torch.mean((pay_b - pay_b.mean(0)) * (pay_cv_b - pay_cv_b.mean(0)), dim=0)
                beta_cv = (cov_b / torch.clamp(torch.var(pay_cv_b, dim=0), min=1e-12)).detach()

                # vega at base IV: chain rule denominator d(IV)/d(price) = 1/vega
                v_hf_t = torch.tensor(v_hf_base, device=device, dtype=dtype)
                std_base = torch.clamp(v_hf_t * torch.sqrt(T_t), min=1e-12)
                vega_b = torch.clamp(
                    torch.sqrt(T_t) * INV_SQRT_2PI * torch.exp(-0.5 * (K_t / std_base) ** 2),
                    min=1e-12,
                )

            # chain-rule Jacobian: path tensors held fixed, only (α_smm, ρ_smm) carry grad
            with torch.enable_grad():
                a_g1 = a_b.detach().clone().requires_grad_(True)
                r_g1 = r_b.detach().clone().requires_grad_(True)

                pay_g1, pay_cv_g1 = _cmc_payoff(
                    a_g1.unsqueeze(0), r_g1.unsqueeze(0), I_v_T, I_V2_T, I_cv_T
                )
                std_true1 = a_g1 * torch.sqrt(T_t)
                d_true1 = -K_t / torch.clamp(std_true1, min=1e-12)
                pdf_t1 = INV_SQRT_2PI * torch.exp(-0.5 * d_true1**2)
                cdf_c1 = 0.5 * (1 + torch.erf(d_true1 * INV_SQRT_2))
                cdf_p1 = 0.5 * (1 + torch.erf(-d_true1 * INV_SQRT_2))
                cv_true1 = torch.where(
                    K_t >= 0,
                    std_true1 * (cdf_c1 * d_true1 + pdf_t1),
                    std_true1 * (cdf_p1 * (-d_true1) + pdf_t1),
                )
                prices_g1 = torch.mean(
                    pay_g1 - beta_cv.unsqueeze(0) * (pay_cv_g1 - cv_true1.unsqueeze(0)),
                    dim=0,
                )
                prices_g1 = torch.where(is_call, prices_g1, prices_g1 - K_t)
                prices_g1.sum().backward()

                d_price_d_alpha = a_g1.grad.detach().clone()  # (n_opts,)
                d_price_d_rho = r_g1.grad.detach().clone()  # (n_opts,)

            # d(α_smm, ρ_smm)/dp: pure (N+1)x(N+1) ops, no path tensors
            p_t_g = torch.tensor(p_global, device=device, dtype=dtype)

            def smm_params_from_angles(p_tensor):
                """Free angles -> [α_smm; ρ_smm] per unique (expiry, tenor) pair."""
                o_mod_g = o_base.index_put((r_idx, c_idx), p_tensor)
                S_g = build_rapisarda_correlation_matrix(o_mod_g)
                # expand=False -> returns unique-pair sized (n_unique,) tensors
                a_g, r_g = apply_sigma_to_smm(S_g, smm_cache, expand=False)
                return torch.cat([a_g, r_g])  # (2*n_unique,)

            # vectorize: one batched backward instead of one per output; same values
            J_smm_t = torch.autograd.functional.jacobian(
                smm_params_from_angles, p_t_g, vectorize=True
            )
            # shape: (2*n_unique, n_params); expand_idx maps unique pairs back to targets
            n_unique_j = len(smm_cache["pi_ext"])
            exp_idx_j = smm_cache["expand_idx"].to(device)
            J_alpha_to_p = J_smm_t[:n_unique_j][exp_idx_j]  # (n_opts, n_params)
            J_rho_to_p = J_smm_t[n_unique_j:][exp_idx_j]  # (n_opts, n_params)

            J_price_to_p = (
                d_price_d_alpha.unsqueeze(1) * J_alpha_to_p
                + d_price_d_rho.unsqueeze(1) * J_rho_to_p
            )  # (n_opts, n_params)

            # d(IV)/d(p) = d(price)/d(p) / vega
            J_hf_t = J_price_to_p / vega_b.unsqueeze(1)
            J_hf = J_hf_t.detach().cpu().numpy().astype(np.float64)

            if not np.isfinite(J_hf).all():
                J_hf = np.nan_to_num(J_hf, nan=0.0, posinf=0.0, neginf=0.0)

        return v_hf_base, J_hf

    def _get_global_lf_vols_and_jac(
        self,
        p_global,
        free_indices,
        exp_targets,
        strikes,
        calculate_jac=False,
        *,
        smm_cache,
    ):
        """LF ODE surrogate evaluation; exact autograd Jacobian when calculate_jac=True."""
        r_idx, c_idx = free_indices
        device = self.device
        dtype = self.model.dtype
        o_base = torch.tensor(self.omega, device=device, dtype=dtype)

        k_t_const = torch.tensor(strikes, device=device, dtype=dtype)
        T_t_const = torch.clamp(torch.tensor(exp_targets, device=device, dtype=dtype), min=1e-4)

        def lf_obj_for_jac(p_tensor):
            o_mod = o_base.index_put((r_idx, c_idx), p_tensor)
            S_mat = build_rapisarda_correlation_matrix(o_mod)

            alpha_smm, rho_smm = apply_sigma_to_smm(S_mat, smm_cache)

            return _core_fukasawa_ode(
                k_t_const, T_t_const, alpha_smm, rho_smm, self.model.nu, self.model.H
            )

        with torch.no_grad():
            p_t = torch.tensor(p_global, device=device, dtype=dtype)
            v_lf_base_raw = lf_obj_for_jac(p_t)
            v_lf_base = v_lf_base_raw.detach().cpu().numpy().astype(np.float64)

        J_lf = None

        if calculate_jac:
            p_t_grad = torch.tensor(p_global, device=device, dtype=dtype)
            J_lf_t = torch.autograd.functional.jacobian(lf_obj_for_jac, p_t_grad, vectorize=True)
            J_lf = J_lf_t.detach().cpu().numpy().astype(np.float64)

            if not np.isfinite(J_lf).all():
                J_lf = np.nan_to_num(J_lf, nan=0.0, posinf=0.0, neginf=0.0)

        return v_lf_base, J_lf

    def _run_ammo_tr_loop(self, p_init, lam, setup):
        """AMMO trust-region loop (Alexandrov 2001) for one fixed λ value.

        Returns {"p_opt", "eta", "rho", "final_rmse", "mean_bias", "timing", "v_hf"}; v_hf is the
        HF vols at p_opt.
        """
        t0_loop = time.time()

        global_exp = setup["global_exp"]
        global_ten = setup["global_ten"]
        global_vol = setup["global_vol"]
        global_str = setup["global_str"]
        r_idx = setup["r_idx"]
        c_idx = setup["c_idx"]
        vol_targets_t = setup["vol_targets_t"]
        global_smm_cache = setup["global_smm_cache"]
        bounds = setup["bounds"]
        dt_val = setup["dt_val"]
        n_paths_val = setup["n_paths_val"]
        global_cached_drivers = setup["global_cached_drivers"]
        use_tikhonov = setup["use_tikhonov"]
        is_first_order = setup["is_first_order"]
        o_base = setup["o_base"]

        delta = config.TR_DELTA_0
        delta_max = config.TR_DELTA_MAX
        r1, r2 = config.TR_R1, config.TR_R2
        c1, c2 = config.TR_C1, config.TR_C2
        max_iters = config.STAGE2_AMMO_MAX_ITER

        lo_bounds = np.array([b[0] for b in bounds])
        hi_bounds = np.array([b[1] for b in bounds])

        # s_norm is reset at the anchor below when the penalty is on; the closures read it at
        # call time, after the reset
        s_norm = 1.0

        def _data_rmse(v_np):
            """Data RMSE in bps, without the penalty term."""
            v_t = torch.tensor(v_np, device=self.device, dtype=self.model.dtype)
            return float(torch.sqrt(torch.mean(((v_t - vol_targets_t) * 1e4) ** 2)).item())

        def _merit(v_np, pen_t):
            """Full regularised merit E = ||res||^2 + lam*s²*||pen||^2 (squared-bps).
            s_norm makes lam=1 the natural balance point at the cold-start anchor.
            Used only for the actual/predicted reductions in psi_k, never for RMSE."""
            v_t = torch.tensor(v_np, device=self.device, dtype=self.model.dtype)
            val = float(torch.sum(((v_t - vol_targets_t) * 1e4) ** 2).item())
            if use_tikhonov and len(pen_t) > 0:
                val += lam * (s_norm**2) * float(torch.sum(pen_t**2).item())
            return val

        def _pen_vec(p_np):
            """Tikhonov penalty vector at p_np (no_grad)."""
            if not use_tikhonov:
                return torch.zeros(0, device=self.device, dtype=self.model.dtype)
            with torch.no_grad():
                S_m = self._sigma_from_angles(o_base, (r_idx, c_idx), p_np)
                return self._compute_tikhonov_penalties(S_m)

        mode_str = "FIRST-ORDER" if is_first_order else "ZERO-ORDER"
        print(f"\n{'AMMO TRUST-REGION LOOP (Alexandrov Alg. 2) | MODE: ' + mode_str:^110}")
        print("=" * 110)
        print(
            f"{'ITER':<8} | {'PSI_K':>7} | {'STATUS':<6} | "
            f"{'RMSE: anchor -> trial':^23} | {'DELTA':>8} | "
            f"{'HF':>6} | {'LF':>6} | {'SUB':>6}"
        )
        print("=" * 110)

        current_p = p_init.copy()

        # Pre-loop: single HF eval at initial anchor. It does not depend on λ, so the driver cache
        # keeps it per target set and the λ grid of a cv_1se fold computes it once.
        t0_pre = time.time()
        anchor_memo = global_cached_drivers["anchor_hf"]
        anchor_key = (
            current_p.tobytes(),
            global_exp.tobytes(),
            global_ten.tobytes(),
            global_str.tobytes(),
            self.omega.tobytes(),
            is_first_order,
            config.CALI_PATHS_JAC,
        )
        if anchor_key not in anchor_memo:
            anchor_memo[anchor_key] = self._get_global_hf_vols_and_jac(
                current_p,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=is_first_order,
                smm_cache=global_smm_cache,
            )
        v_hf_current, J_hf_current = (
            None if x is None else x.copy() for x in anchor_memo[anchor_key]
        )
        t_pre_hf = time.time() - t0_pre

        pen_anchor = _pen_vec(current_p)

        # penalty normalization: s_norm = ||data_res|| / ||pen_anchor||
        # Both norms computed from the exact operands _merit squares (pre-λ, pre-s).
        # At λ=1 the two squared-norm terms are equal at the anchor by construction.
        if use_tikhonov and len(pen_anchor) > 0:
            _v_anchor_t = torch.tensor(v_hf_current, device=self.device, dtype=self.model.dtype)
            _data_res_anchor = (_v_anchor_t - vol_targets_t) * 1e4  # exact _merit data operand
            _data_norm = float(torch.linalg.norm(_data_res_anchor).item())
            _pen_norm = float(torch.linalg.norm(pen_anchor).item())
            s_norm = _data_norm / _pen_norm if _pen_norm > 1e-12 else 1.0

        merit_current = _merit(v_hf_current, pen_anchor)
        current_rmse = _data_rmse(v_hf_current)

        # Best-merit incumbent over {anchor, accepted trials}: one notion of "best"
        # (regularised merit) drives acceptance, selection and reporting. Seeded at the
        # anchor so a loop that never lowers merit returns the cold start.
        cold_start_rmse = current_rmse
        best_p = current_p.copy()
        best_merit = merit_current
        best_rmse = current_rmse
        best_v_hf = v_hf_current.copy()

        consecutive_rejects = 0

        print(
            f"Pre-loop HF eval: {t_pre_hf:.2f}s | anchor RMSE = {current_rmse:.3f} bps",
            flush=True,
        )
        print("-" * 110, flush=True)

        # loop-invariant ODE inputs of the surrogate; never mutated
        k_surr_t = torch.tensor(global_str, device=self.device, dtype=self.model.dtype)
        T_surr_t = torch.clamp(
            torch.tensor(global_exp, device=self.device, dtype=self.model.dtype),
            min=1e-4,
        )

        for ammo_iter in range(max_iters):
            t0_lf = time.time()
            v_lf_base, J_lf = self._get_global_lf_vols_and_jac(
                current_p,
                (r_idx, c_idx),
                global_exp,
                global_str,
                calculate_jac=is_first_order,
                smm_cache=global_smm_cache,
            )
            t_lf_eval = time.time() - t0_lf

            beta_k = v_hf_current / v_lf_base
            beta_k_t = torch.tensor(beta_k, device=self.device, dtype=self.model.dtype)
            current_p_anchor_t = torch.tensor(current_p, device=self.device, dtype=self.model.dtype)

            grad_beta_k_t = None
            if is_first_order:
                v_hf_col = v_hf_current[:, np.newaxis]
                v_lf_col = v_lf_base[:, np.newaxis]
                # eq::grad_beta, quotient rule: dβ/dp = (J_hf*v_lf − v_hf*J_lf) / v_lf²
                grad_beta_k = (J_hf_current * v_lf_col - v_hf_col * J_lf) / (v_lf_col**2)
                grad_beta_k_t = torch.tensor(
                    grad_beta_k, device=self.device, dtype=self.model.dtype
                )

            # eq:subproblem: L-inf ball around the anchor, clipped to the global bounds
            tr_bounds = list(
                zip(
                    np.maximum(lo_bounds, current_p - delta),
                    np.minimum(hi_bounds, current_p + delta),
                )
            )

            def _surrogate_vols(p_t):
                """s_k(p) (eq::beta_ammo): β-Taylor x ODE on the mapped SMM params; returns (s_k, Σ)."""
                o_m = o_base.index_put((r_idx, c_idx), p_t)
                S_m = build_rapisarda_correlation_matrix(o_m)

                alpha_smm_t, rho_smm_t = apply_sigma_to_smm(S_m, global_smm_cache)
                v_lf_trial = _core_fukasawa_ode(
                    k_surr_t, T_surr_t, alpha_smm_t, rho_smm_t, self.model.nu, self.model.H
                )

                if is_first_order:
                    step_t = p_t - current_p_anchor_t
                    beta_trial_t = beta_k_t + torch.mv(grad_beta_k_t, step_t)
                else:
                    beta_trial_t = beta_k_t

                return beta_trial_t * v_lf_trial, S_m

            def global_objective_and_grad(p_np):
                p_t = torch.tensor(
                    p_np, device=self.device, dtype=self.model.dtype, requires_grad=True
                )
                v_surr, S_m = _surrogate_vols(p_t)

                res_data = (v_surr - vol_targets_t) * 10000.0
                loss = torch.sum(res_data**2)

                if use_tikhonov:
                    pen = self._compute_tikhonov_penalties(S_m)
                    if len(pen) > 0:
                        loss = loss + lam * (s_norm**2) * torch.sum(pen**2)

                loss.backward()

                grad_np = p_t.grad.detach().cpu().numpy().astype(np.float64)
                if not np.isfinite(grad_np).all():
                    grad_np = np.nan_to_num(grad_np, nan=0.0, posinf=0.0, neginf=0.0)

                return loss.detach().item(), grad_np

            t0_sub = time.time()
            res = minimize(
                global_objective_and_grad,
                current_p,
                method="L-BFGS-B",
                jac=True,
                bounds=tr_bounds,
                options={
                    "ftol": 1e-6,
                    "gtol": 1e-5,
                    "maxiter": config.STAGE2_LBFGSB_MAXITER,
                    "maxfun": config.STAGE2_LBFGSB_MAXFUN,
                },
            )
            t_sub_total = time.time() - t0_sub
            x_trial = res.x

            t0_hf = time.time()
            v_hf_trial, J_hf_trial = self._get_global_hf_vols_and_jac(
                x_trial,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=is_first_order,
                smm_cache=global_smm_cache,
            )
            t_hf_eval = time.time() - t0_hf

            pen_trial = _pen_vec(x_trial)

            # surrogate vol at x_trial, the same model the subproblem minimised
            with torch.no_grad():
                x_trial_t = torch.tensor(x_trial, device=self.device, dtype=self.model.dtype)
                v_surr_trial = _surrogate_vols(x_trial_t)[0].cpu().numpy()

            # full-merit reductions, used only for ψ_k (eq:psi)
            merit_trial_hf = _merit(v_hf_trial, pen_trial)
            merit_surr_trial = _merit(v_surr_trial, pen_trial)
            actual_reduction = merit_current - merit_trial_hf
            predicted_reduction = merit_current - merit_surr_trial
            pred_safe = predicted_reduction if abs(predicted_reduction) >= 1e-12 else 1e-12
            psi_k = actual_reduction / pred_safe

            # ‖p_k‖_∞, taken before the accept rebinds current_p
            step_norm = float(np.max(np.abs(x_trial - current_p)))

            trial_rmse = _data_rmse(v_hf_trial)

            status = "ACCEPT" if actual_reduction > 0 else "REJECT"
            print(
                f"[{ammo_iter + 1:>2}/{max_iters}] "
                f"psi={psi_k:+.3f} | {status:<6} | "
                f"RMSE: {current_rmse:.3f} -> {trial_rmse:.3f} bps | "
                f"delta={delta:.4f} | "
                f"HF {t_hf_eval:.1f}s | LF {t_lf_eval:.1f}s | sub {t_sub_total:.1f}s",
                flush=True,
            )

            # pre-accept merit, for the relative-reduction stop below
            merit_before = merit_current

            if actual_reduction > 0:
                current_p = x_trial
                v_hf_current = v_hf_trial
                J_hf_current = J_hf_trial
                pen_anchor = pen_trial
                merit_current = merit_trial_hf
                current_rmse = trial_rmse
                consecutive_rejects = 0
                if merit_trial_hf < best_merit:
                    best_p = x_trial.copy()
                    best_merit = merit_trial_hf
                    best_rmse = trial_rmse
                    best_v_hf = v_hf_trial.copy()
            else:
                consecutive_rejects += 1

            if psi_k < r1:
                delta = max(1e-5, c1 * step_norm)
            elif psi_k > r2:
                delta = min(c2 * delta, delta_max)

            # merit-based terminations
            if delta <= 1e-5:
                break
            if consecutive_rejects >= 2:
                break
            if (
                actual_reduction > 0
                and actual_reduction / (merit_before + 1e-12) < config.STAGE2_MERIT_RTOL
            ):
                break

        print("-" * 110, flush=True)

        # Return the best-merit incumbent (may be the anchor) and its data RMSE, not the
        # last-accepted iterate's. eta/rho use the incumbent's stored HF vols, bit-identical
        # to a fresh eval because the drivers are fixed.
        if best_rmse > cold_start_rmse + 1e-9:
            print(
                f"WARNING: calibrated Σ fits worse than cold-start anchor "
                f"({best_rmse:.2f} > {cold_start_rmse:.2f} bps) — λ likely too strong.",
                flush=True,
            )
        p_opt = best_p
        res_data_final = (best_v_hf - global_vol) * 10000.0
        eta = np.log(np.linalg.norm(res_data_final) + 1e-12)
        final_rmse = best_rmse

        if use_tikhonov:
            rho = self._log_penalty_norm(self._sigma_from_angles(o_base, (r_idx, c_idx), p_opt))
        else:
            rho = -20.0

        return {
            "p_opt": p_opt,
            "eta": eta,
            "rho": rho,
            "final_rmse": final_rmse,
            "mean_bias": float(np.mean(res_data_final)),
            "timing": {"total_s": time.time() - t0_loop},
            "v_hf": best_v_hf,
        }

    def calibrate(
        self,
        method="AMMO_MC",
        log_file="",
        stage1_csv=None,
    ):
        """Fit Σ at λ = config.LAMBDA_CURVATURE, or run the λ sweep when config.AUTO_LAMBDA is True.

        log_file: path that receives a copy of the console output; "" writes none. stage1_csv: the
        Stage-1 table stamped into the log (default results/stage1_parameters.csv).
        Returns a dict with "omega_matrix", "Sigma_matrix" (numpy, (N+1)×(N+1)), "final_rmse_bps",
        "final_max_err_bps" and "final_mean_bias_bps". A sweep returns "Sigma_matrix" None,
        "sweep_only" True, the table under "lambda_sweep_data" and the λ→0 Σ under "Sigma_floor".
        """
        if method not in ("PURE_MC", "AMMO_MC"):
            raise ValueError(f"unknown Stage-2 method {method!r}")
        # fail before the provenance stamp, which would otherwise log an unknown form as exp
        _check_decay_form(config.STAGE2_DECAY_FORM)
        original_stdout = sys.stdout
        logger = None
        if log_file:
            logger = DualLogger(log_file)
            sys.stdout = logger

        try:
            print("\n" + "=" * 80)
            print(f"{('GLOBAL SPATIAL CORRELATION CALIBRATION | MODE: ' + method):^80}")
            print("=" * 80)

            # Provenance stamp: pins the exact Stage-1 marginals + regularisation config so
            # accumulated logs can't be mistaken for run-to-run nondeterminism (input drift).
            s1_path = stage1_csv or os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)
            try:
                _s1_mtime = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(s1_path))
                )
            except OSError:
                _s1_mtime = "MISSING"
            _form = config.STAGE2_DECAY_FORM
            if self.hist_target_info is not None:
                _h = self.hist_target_info
                _shape = (
                    f"params={[round(p, 4) for p in _h['params']]} window={_h['window']} "
                    f"step={_h['step']} fit_rmse={_h['rmse']:.3f}"
                )
            else:
                _shape = (
                    f"s={config.STAGE2_DECAY_SLOPE}"
                    if _form == "linear"
                    else f"β={config.STAGE2_DECAY_BETA}"
                )
            if config.STAGE2_PENALTY_FIRST_FWD != 1:
                _shape += f" penalty_first_fwd={config.STAGE2_PENALTY_FIRST_FWD}"
            if config.STAGE2_LAMBDA_RULE != "fixed":
                _shape += f" λ_rule={config.STAGE2_LAMBDA_RULE}"
            print(
                f"[PROVENANCE] stage1={s1_path} (mtime {_s1_mtime}) | method={method} | "
                f"λ={config.LAMBDA_CURVATURE} "
                f"form={_form} {_shape} "
                f"ρ∞={config.STAGE2_DECAY_RHO_INF} "
                f"HYBRID_KAPPA={config.HYBRID_KAPPA}",
                flush=True,
            )

            start_time = time.time()

            setup = self._build_correlation_target_grid()
            if setup is None:
                return {
                    "omega_matrix": self.omega,
                    "Sigma_matrix": build_rapisarda_correlation_matrix(
                        torch.tensor(self.omega, device=self.device)
                    )
                    .cpu()
                    .numpy(),
                }
            global_exp = setup["global_exp"]
            global_ten = setup["global_ten"]
            global_vol = setup["global_vol"]
            global_str = setup["global_str"]
            r_idx = setup["r_idx"]
            c_idx = setup["c_idx"]
            dt_val = setup["dt_val"]
            n_paths_val = setup["n_paths_val"]
            global_cached_drivers = setup["global_cached_drivers"]
            o_base = setup["o_base"]

            if method == "PURE_MC":
                current_p, _sweep_df = self._calibrate_pure_mc(setup)
                v_final_known = None
            else:
                current_p, v_final_known, sweep_result = self._calibrate_ammo_mc(setup)
                if sweep_result is not None:
                    return sweep_result
                _sweep_df = None

            self.omega = (
                o_base.index_put(
                    (r_idx, c_idx),
                    torch.tensor(current_p, device=self.device, dtype=self.model.dtype),
                )
                .cpu()
                .numpy()
            )
            omega_tensor = torch.tensor(self.omega, device=self.device, dtype=self.model.dtype)
            Sigma_final = build_rapisarda_correlation_matrix(omega_tensor)

            print("\n-> Verifying Final Calibrated Matrix against High-Fidelity MC...")
            if v_final_known is not None:
                # the loop priced this Σ on these drivers and targets already; same numbers
                v_final = np.array(v_final_known)
            else:
                with torch.no_grad():
                    v_final_raw = mapped_smm_mc_pricer(
                        self.model,
                        Sigma_final,
                        global_exp,
                        global_ten,
                        global_str,
                        dt=dt_val,
                        n_paths=n_paths_val,
                        cached_drivers=global_cached_drivers,
                    )
                    v_final = np.array(v_final_raw)  # numpy from mapped_smm_mc_pricer

            final_res = (v_final - global_vol) * 10000.0
            final_rmse = float(np.sqrt(np.mean(final_res**2)))
            final_max_err = float(np.max(np.abs(final_res)))
            final_mean_err = float(np.mean(final_res))

            end_time = time.time()
            print("-" * 80)
            print(f"Status : SUCCESS | Global Matrix AMMO Total Time: {end_time - start_time:.2f}s")
            print(f"         Final Target RMSE : {final_rmse:.2f} bps")
            print(f"         Final Max Error   : {final_max_err:.2f} bps")
            print(f"         Final Mean Bias   : {final_mean_err:.2f} bps")
            print("=" * 80 + "\n")

            return {
                "omega_matrix": self.omega,
                "Sigma_matrix": Sigma_final.cpu().numpy(),
                "lambda_sweep_data": _sweep_df,
                "final_rmse_bps": final_rmse,
                "final_max_err_bps": final_max_err,
                "final_mean_bias_bps": final_mean_err,
            }

        finally:
            if logger:
                sys.stdout.close()
                sys.stdout = original_stdout

    def _calibrate_pure_mc(self, setup):
        """PURE_MC: least squares on the HF MC over the λ grid; returns (angles, sweep table or None)."""
        global_exp = setup["global_exp"]
        global_ten = setup["global_ten"]
        global_vol = setup["global_vol"]
        global_str = setup["global_str"]
        r_idx = setup["r_idx"]
        c_idx = setup["c_idx"]
        global_smm_cache = setup["global_smm_cache"]
        current_p = setup["current_p"]
        dt_val = setup["dt_val"]
        n_paths_val = setup["n_paths_val"]
        global_cached_drivers = setup["global_cached_drivers"]
        current_lambda = setup["current_lambda"]
        use_tikhonov = setup["use_tikhonov"]
        o_base = setup["o_base"]
        _sweep_df = None

        print("\n" + "=" * 110)
        print(f"{'PURE MC GLOBAL CALIBRATION (ALIGNED L-CURVE)':^110}")
        print("=" * 110)

        t_cumulative_total = 0.0
        n_obj_evals = 0
        n_jac_evals = 0

        def global_mc_obj(p_np):
            nonlocal t_cumulative_total, n_obj_evals
            n_obj_evals += 1
            t0_step = time.time()

            t0_hf = time.time()
            v_hf_base, _ = self._get_global_hf_vols_and_jac(
                p_np,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=False,
                smm_cache=global_smm_cache,
            )
            t_hf = time.time() - t0_hf

            res_data = (v_hf_base - global_vol) * 10000.0
            rmse_current = np.sqrt(np.mean(res_data**2))
            max_resid = np.max(np.abs(res_data))

            t0_pen = time.time()
            res_full = res_data.astype(np.float64)

            if use_tikhonov:
                S_m = self._sigma_from_angles(o_base, (r_idx, c_idx), p_np)

                with torch.no_grad():
                    pen_t = self._compute_tikhonov_penalties(S_m)
                    if len(pen_t) > 0:
                        res_pen = pen_t.cpu().numpy() * math.sqrt(current_lambda) * s_norm_mc
                        res_full = np.concatenate([res_data, res_pen]).astype(np.float64)
            t_pen = time.time() - t0_pen

            t_step = time.time() - t0_step
            t_cumulative_total += t_step

            if n_obj_evals == 1 or n_obj_evals % 10 == 0:
                print(
                    f"   [Obj Eval {n_obj_evals:3d}] "
                    f"[MC Pricing: {t_hf:5.2f}s | Tikhonov Penalty: {t_pen:5.2f}s] "
                    f"-> Step: {t_step:5.2f}s | Cumul: {t_cumulative_total:6.1f}s || "
                    f"RMSE: {rmse_current:6.2f} bps | Max Err: {max_resid:6.2f} bps",
                    flush=True,
                )

            return res_full

        def global_mc_jac(p_np):
            nonlocal t_cumulative_total, n_jac_evals
            n_jac_evals += 1
            t0_step = time.time()

            t0_jac = time.time()
            _, J_hf_raw = self._get_global_hf_vols_and_jac(
                p_np,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=True,
                smm_cache=global_smm_cache,
            )

            J_data = J_hf_raw * 10000.0
            if not np.isfinite(J_data).all():
                J_data = np.nan_to_num(J_data, nan=0.0, posinf=0.0, neginf=0.0)
            t_jac = time.time() - t0_jac

            t0_pen_jac = time.time()
            jac_full = J_data.astype(np.float64)

            if use_tikhonov:
                p_t = torch.tensor(
                    p_np, device=self.device, dtype=self.model.dtype, requires_grad=True
                )

                def pen_func(p_tensor):
                    o_m = o_base.index_put((r_idx, c_idx), p_tensor)
                    S_m = build_rapisarda_correlation_matrix(o_m)
                    return self._compute_tikhonov_penalties(S_m)

                J_pen_t = torch.autograd.functional.jacobian(pen_func, p_t)
                J_pen = J_pen_t.detach().cpu().numpy()

                if J_pen.size > 0:
                    J_pen = J_pen * math.sqrt(current_lambda) * s_norm_mc
                    jac_full = np.vstack([J_data, J_pen]).astype(np.float64)
                    if not np.isfinite(jac_full).all():
                        jac_full = np.nan_to_num(jac_full, nan=0.0, posinf=0.0, neginf=0.0)
            t_pen_jac = time.time() - t0_pen_jac

            t_step = time.time() - t0_step
            t_cumulative_total += t_step

            if n_jac_evals == 1 or n_jac_evals % 10 == 0:
                print(
                    f"   [Jac Eval {n_jac_evals:3d}] "
                    f"[MC Jacobian: {t_jac:5.2f}s | Tikhonov Jacobian: {t_pen_jac:5.2f}s] "
                    f"-> Step: {t_step:5.2f}s | Cumul: {t_cumulative_total:6.1f}s",
                    flush=True,
                )

            return jac_full

        is_auto_lambda = config.AUTO_LAMBDA
        if is_auto_lambda and use_tikhonov:
            lambda_grid = _lambda_grid()
        else:
            lambda_grid = [current_lambda]

        eta_list, penalty_norms, rmse_list = [], [], []
        struct_metrics_list = []
        opt_p_dict = {}
        initial_p_guess = current_p.copy()
        ls_bounds = (1e-4, np.pi - 1e-4)

        # fix s_norm at the cold-start anchor, as AMMO does, so both minimise one merit
        s_norm_mc = 1.0
        if use_tikhonov:
            v_anchor_mc, _ = self._get_global_hf_vols_and_jac(
                initial_p_guess,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=False,
                smm_cache=global_smm_cache,
            )
            _data_res_a = (v_anchor_mc - global_vol) * 1e4
            _S_m_a = self._sigma_from_angles(o_base, (r_idx, c_idx), initial_p_guess)
            with torch.no_grad():
                _pen_a = self._compute_tikhonov_penalties(_S_m_a)
            _dn = float(np.linalg.norm(_data_res_a))
            _pn = float(torch.linalg.norm(_pen_a).item()) if len(_pen_a) > 0 else 0.0
            s_norm_mc = _dn / _pn if _pn > 1e-12 else 1.0

        for lam_val in lambda_grid:
            current_lambda = lam_val
            p_guess = initial_p_guess.copy()

            t_cumulative_total = 0.0
            n_obj_evals = 0
            n_jac_evals = 0

            print(f"\n{'=' * 90}")
            print(f">>> STARTING OPTIMIZATION: λ (Tikhonov Lambda) = {lam_val:.4e} <<<")
            print(f"{'=' * 90}")

            t0_opt = time.time()
            res = least_squares(
                global_mc_obj,
                p_guess,
                jac=global_mc_jac,
                bounds=ls_bounds,
                method="trf",
                ftol=1e-5,
                xtol=1e-5,
                max_nfev=150,
            )
            t_opt_total = time.time() - t0_opt
            p_opt = res.x

            t0_final = time.time()
            v_hf_final, _ = self._get_global_hf_vols_and_jac(
                p_opt,
                (r_idx, c_idx),
                global_exp,
                global_ten,
                global_str,
                dt_val,
                n_paths_val,
                global_cached_drivers,
                calculate_jac=False,
                smm_cache=global_smm_cache,
            )
            t_final_eval = time.time() - t0_final

            res_data_final = (v_hf_final - global_vol) * 10000.0
            eta = np.log(np.linalg.norm(res_data_final) + 1e-12)
            rmse_pred = np.sqrt(np.mean(res_data_final**2))
            max_resid = np.max(np.abs(res_data_final))

            if use_tikhonov:
                S_m = self._sigma_from_angles(o_base, (r_idx, c_idx), p_opt)
                rho = self._log_penalty_norm(S_m)
                struct_metrics_list.append(
                    compute_sigma_structural_metrics(S_m.detach().cpu().numpy(), self.N)
                )
            else:
                rho = -20.0

            eta_list.append(eta)
            penalty_norms.append(rho)
            rmse_list.append(rmse_pred)
            opt_p_dict[lam_val] = p_opt.copy()

            t_scipy_overhead = max(0.0, t_opt_total - t_cumulative_total)

            print(
                f"\n   [λ = {lam_val:.2e} Complete] Final RMSE: {rmse_pred:.2f} bps | Max Err: {max_resid:.2f} bps"
            )
            print(f"   [L-Curve Norms] ETA (log Res): {eta:.4f} | RHO (log Pen): {rho:.4f}")
            print("   [Optimization Timers]")
            print(f"      ├─ Objective Evals : {n_obj_evals}")
            print(f"      ├─ Jacobian Evals  : {n_jac_evals}")
            print(f"      ├─ Logic Execution : {t_cumulative_total:.2f}s")
            print(f"      ├─ SciPy Overhead  : {t_scipy_overhead:.2f}s")
            print(f"      ├─ Final Clean Eval: {t_final_eval:.2f}s")
            print(f"      └─ Total Step Time : {t_opt_total + t_final_eval:.2f}s\n")

        if len(lambda_grid) > 1:
            print(f"\n{'=' * 90}")
            print(">>> LAMBDA SELECTION: MONOTONE REGIME (PURE_MC) <<<")
            print(f"{'=' * 90}")
            best_lambda, chosen_idx, fallback_used = select_lambda_monotone(
                lambda_grid, struct_metrics_list
            )
            lcurve_lambda, _ = find_l_curve_corner(lambda_grid, eta_list, penalty_norms)

            for idx_lam, lam_v in enumerate(lambda_grid):
                m = struct_metrics_list[idx_lam]
                marker = "<-- SELECTED (monotone lower edge)" if lam_v == best_lambda else ""
                print(
                    f"   Lam: {lam_v:.4e} | mono_viol: {m['mono_violations']:4d}"
                    f" | sign_flips: {m['sign_flips']:3d} {marker}"
                )

            if fallback_used:
                print(
                    f"\n   [WARNING] No >=2-consecutive mono_viol=0 run found."
                    f" Fallback: argmin(mono_viol) at λ={best_lambda:.4e}."
                    f" Consider raising LAMBDA_GRID_HI."
                )
            elif chosen_idx == 0:
                print(
                    f"\n   [WARNING] Monotone band starts at grid lower bound"
                    f" λ={best_lambda:.4e}. Consider lowering LAMBDA_GRID_LO."
                )

            print(f"\n   [L-CURVE CORNER (diagnostic only)]: λ = {lcurve_lambda:.4e}")
            current_lambda = best_lambda
            current_p = opt_p_dict[best_lambda]
            _sweep_df = {
                "lambda": list(lambda_grid),
                "rmse_bps": rmse_list,
                "mono_viol": [m["mono_violations"] for m in struct_metrics_list],
                "sign_flips": [m["sign_flips"] for m in struct_metrics_list],
                "eta": eta_list,
                "rho": penalty_norms,
                "selected": [lv == best_lambda for lv in lambda_grid],
            }
        else:
            current_p = opt_p_dict[lambda_grid[0]]

        print(
            f"\n-> PURE_MC Global Optimization Finished. Final Selected λ = {current_lambda:.2e}\n"
        )
        return current_p, _sweep_df

    def _calibrate_ammo_mc(self, setup):
        """AMMO_MC over the λ grid; returns (angles, HF vols, None), or (None, None, sweep result) in a sweep."""
        current_p = setup["current_p"]
        current_lambda = setup["current_lambda"]
        use_tikhonov = setup["use_tikhonov"]
        o_base = setup["o_base"]
        r_idx = setup["r_idx"]
        c_idx = setup["c_idx"]

        is_auto_lambda = config.AUTO_LAMBDA
        sweep_mode = is_auto_lambda and use_tikhonov
        if sweep_mode:
            _grid = _lambda_grid()
            # λ→0 floor row: the unregularised RMSE floor, through the same loop.
            _floor = config.STAGE2_LAMBDA_FLOOR
            lambda_grid_ammo = np.concatenate([[_floor], _grid])
        else:
            lambda_grid_ammo = [current_lambda]

        ammo_results = {}
        eta_list_ammo = []
        rho_list_ammo = []
        struct_metrics_ammo = []
        sigma_ammo = []  # per-λ Σ; only the λ→0 floor row is consumed
        target_dist_ammo = []
        mean_bias_ammo = []
        initial_p_ammo = current_p.copy()

        for lam_val in lambda_grid_ammo:
            result = self._run_ammo_tr_loop(initial_p_ammo.copy(), lam_val, setup)
            ammo_results[lam_val] = result
            eta_list_ammo.append(result["eta"])
            rho_list_ammo.append(result["rho"])
            mean_bias_ammo.append(result["mean_bias"])
            with torch.no_grad():
                S_m = self._sigma_from_angles(o_base, (r_idx, c_idx), result["p_opt"])
            struct_metrics_ammo.append(compute_sigma_structural_metrics(S_m.cpu().numpy(), self.N))
            sigma_ammo.append(S_m.cpu().numpy())
            target_dist_ammo.append(self._decay_target_distance(S_m))

        if sweep_mode:
            # the selectors are advisory: they neither commit a λ nor write Σ
            best_lambda, chosen_idx, fallback_used = select_lambda_monotone(
                lambda_grid_ammo, struct_metrics_ammo
            )
            lcurve_lambda, _ = find_l_curve_corner(lambda_grid_ammo, eta_list_ammo, rho_list_ammo)

            print(f"\n{'=' * 110}")
            print(f"{'>>> AMMO_MC λ-SWEEP (ADVISORY — diagnostic only, NOT committed) <<<':^110}")
            print(f"{'=' * 110}")
            print(
                f"   {'λ':>10} | {'RMSE':>7} | {'bias':>8} | {'tgt_dist':>8} | "
                f"{'mono_v':>6} | {'sgn_f':>5} | {'eta':>7} | {'rho':>7} | note"
            )
            print("   " + "-" * 104)
            for idx_lam, lam_v in enumerate(lambda_grid_ammo):
                m = struct_metrics_ammo[idx_lam]
                note = "FLOOR(λ→0)" if idx_lam == 0 else ""
                if lam_v == best_lambda:
                    note = (note + " ADVISORY←").strip()
                _td = target_dist_ammo[idx_lam]
                _td_str = f"{_td:>8.4f}" if _td is not None else f"{'—':>8}"
                print(
                    f"   {lam_v:>10.3e} | {ammo_results[lam_v]['final_rmse']:>7.3f} | "
                    f"{mean_bias_ammo[idx_lam]:>+8.3f} | {_td_str} | "
                    f"{m['mono_violations']:>6d} | {m['sign_flips']:>5d} | "
                    f"{eta_list_ammo[idx_lam]:>7.3f} | {rho_list_ammo[idx_lam]:>7.3f} | {note}"
                )

            if fallback_used:
                print(
                    f"\n   [ADVISORY] No >=2-consecutive mono_viol=0 run;"
                    f" argmin(mono_viol) at λ={best_lambda:.4e}."
                )
            print(f"   [ADVISORY] L-curve corner: λ = {lcurve_lambda:.4e}")

            _sweep_df = {
                "lambda": list(lambda_grid_ammo),
                "is_floor": [True] + [False] * (len(lambda_grid_ammo) - 1),
                "rmse_bps": [ammo_results[lv]["final_rmse"] for lv in lambda_grid_ammo],
                "mean_bias_bps": mean_bias_ammo,
                "target_dist": target_dist_ammo,
                "mono_viol": [m["mono_violations"] for m in struct_metrics_ammo],
                "sign_flips": [m["sign_flips"] for m in struct_metrics_ammo],
                "eta": eta_list_ammo,
                "rho": rho_list_ammo,
                "advisory_selected": [lv == best_lambda for lv in lambda_grid_ammo],
            }
            print(
                "\n-> SWEEP ONLY — no production λ committed and no Σ written. To fit at a "
                'λ from the table, set config.STAGE2_LAMBDA_RULE = "fixed" and '
                "config.LAMBDA_CURVATURE = <λ>, then re-run without the sweep.\n",
                flush=True,
            )
            return (
                None,
                None,
                {
                    "omega_matrix": None,
                    "Sigma_matrix": None,
                    "lambda_sweep_data": _sweep_df,
                    "sweep_only": True,
                    "Sigma_floor": sigma_ammo[0],
                    "lambda_floor": float(lambda_grid_ammo[0]),
                },
            )
        res0 = ammo_results[lambda_grid_ammo[0]]
        return res0["p_opt"], res0["v_hf"], None
