import torch
from jolto.core.utils import safe_cpu_compile
from jolto.core.analytics import (
    generate_sobol_bridge_increments,
    generate_prng_increments,
    build_hybrid_rough_driver,
    rough_vol_compensator,
)
import numpy as np
from jolto import config
import math

INV_SQRT_2 = 0.7071067811865476
INV_SQRT_2PI = 0.3989422804014327


def _hybrid_driver(dz_vol, H, n_steps, dt, kappa, device, dtype):
    """BLP hybrid rough driver on the Sobol-bridge increments; both stages' MC caches use it."""
    # default Sobol seed is 42 (generate_sobol_bridge_increments); +7919 keeps the
    # near-noise stream clear of it. κ=0 draws no near-noise → bit-identical cache.
    near_gen = torch.Generator(device=device).manual_seed(42 + 7919) if kappa > 0 else None
    # Sobol-bridge driver is antithetic ([dZ, −dZ]) → negate the near-noise too.
    return build_hybrid_rough_driver(
        dz_vol, H, n_steps, dt, kappa, device, dtype, near_generator=near_gen, antithetic=True
    )


def torch_bachelier(F, K, T, vol):
    """Differentiable Bachelier call price."""
    std_dev = vol * torch.sqrt(T) + 1e-12
    d = (F - K) / std_dev

    cdf = 0.5 * (1.0 + torch.erf(d * INV_SQRT_2))
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * d * d)

    return (F - K) * cdf + std_dev * pdf


def bachelier_iv_newton(
    target_prices, K_tensor, T_tensor, initial_guess_vol, max_iter=20, tol=1e-8
):
    """Newton Bachelier IV inversion in PyTorch, so root-finding stays on device."""
    intrinsic = torch.clamp(-K_tensor, min=0.0)
    price_floored = torch.maximum(target_prices, intrinsic)

    sigma = initial_guess_vol.clone()
    sqrt_t = torch.sqrt(T_tensor)

    for _ in range(max_iter):
        std = sigma * sqrt_t
        std = torch.clamp(std, min=1e-12)
        d = -K_tensor / std

        pdf = INV_SQRT_2PI * torch.exp(-0.5 * d * d)
        cdf = 0.5 * (1.0 + torch.erf(d * INV_SQRT_2))

        price_current = std * (d * cdf + pdf)

        vega = sqrt_t * pdf
        vega = torch.clamp(vega, min=1e-12)

        residual = price_current - price_floored
        newton_step = residual / vega
        sigma = sigma - newton_step

        if torch.max(torch.abs(residual)) < tol:
            break

    return torch.abs(sigma)


def _cond_bachelier_otm(F, Std, K_exp, is_call_exp, sign):
    """Per-path OTM Bachelier payoff (call for K ≥ 0, put otherwise) from the conditional moments."""
    d = (F - K_exp) / Std
    d_sign = torch.where(is_call_exp, d, -d)

    pdf = INV_SQRT_2PI * torch.exp(-0.5 * d * d)
    cdf = 0.5 * (1.0 + torch.erf(d_sign * INV_SQRT_2))
    return sign * (F - K_exp) * cdf + Std * pdf


def _cmc_vol_integrals(
    dz_vol, fbm, var_vec, kernel, nu, T_tensor, n_paths, n_steps, dt, hybrid_kappa, device, dtype
):
    """Left-point unit vol V on the driver; returns dZ, the expiry step indices, ∫V dZ and ∫V² dt."""
    dt_t = torch.tensor(dt, device=device, dtype=dtype)
    # ν is one global scalar; .item() accepts any size-1 shape and rejects a vector.
    nu_val = float(nu.item()) if torch.is_tensor(nu) else float(np.asarray(nu).item())
    # Itô compensator (κ=0) / hybrid construction variance (κ>0)
    var_comp = rough_vol_compensator(nu_val, kernel, var_vec, dt_t, hybrid_kappa)
    # prepend V(0)=1 and drop V(T): left-point (predictable) volatility
    unit_vols = torch.cat(
        [
            torch.ones(1, n_paths, 1, device=device, dtype=dtype),
            torch.exp(nu_val * fbm - var_comp)[..., :-1],
        ],
        dim=-1,
    )
    unit_vols_trunc = unit_vols.squeeze(0)[:, :n_steps]
    dz_vol_trunc = dz_vol.squeeze(0)[:, :n_steps] if dz_vol.dim() == 3 else dz_vol[:, :n_steps]

    I_v = torch.cumsum(unit_vols_trunc * dz_vol_trunc, dim=-1)  # \int V_t dZ_t
    V2_dt = (unit_vols_trunc**2) * dt_t
    I_V2 = torch.cumsum(V2_dt, dim=-1)  # \int V_t^2 dt

    step_indices = torch.clamp(torch.ceil(T_tensor / dt).long() - 1, min=0)
    return dz_vol_trunc, step_indices, I_v[:, step_indices], I_V2[:, step_indices]


def _vol_and_cv_integrals(*args):
    """∫V dZ, ∫V² dt and the control-variate ∫dZ at each option's expiry step (paths × options)."""
    dz_vol_trunc, step_indices, I_v_T, I_V2_T = _cmc_vol_integrals(*args)
    return I_v_T, I_V2_T, torch.cumsum(dz_vol_trunc, dim=-1)[:, step_indices]  # CV: \int dZ_t


# Stacked (paths × options) blocks kept per driver set. Every HF call of one Stage-2 fit prices
# the same target set, and lambda_rule prices the full set once per fold fit, so two blocks cover
# every repeat; a miss only re-stacks the columns, as every call did before. A block is
# 3 × paths × options float32 (12 bytes per path and option).
_BLOCK_CAP = 2


def _memo_vol_integrals(memo, *args):
    """_vol_and_cv_integrals through `memo`, the dict a Stage-2 driver cache carries: the integrals
    depend on the drivers, ν, dt and the expiry step, not on α or ρ, so each step's columns are
    computed once per driver set. Same bits as recomputing: a cumsum column does not depend on the
    truncation length it was computed under. The returned tensors are shared; callers only read."""
    nu, T_tensor, dt, dtype = args[4], args[5], args[8], args[11]
    nu_val = float(nu.item()) if torch.is_tensor(nu) else float(np.asarray(nu).item())
    entry = memo.setdefault((nu_val, float(dt), dtype), {"cols": {}, "blocks": {}})
    cols, blocks = entry["cols"], entry["blocks"]
    steps = tuple(torch.clamp(torch.ceil(T_tensor / dt).long() - 1, min=0).tolist())
    if steps in blocks:
        blocks[steps] = blocks.pop(steps)  # most recently used last
        return blocks[steps]
    if any(s not in cols for s in steps):
        computed = _vol_and_cv_integrals(*args)
        for j, s in enumerate(steps):
            cols.setdefault(s, tuple(integral[:, j].clone() for integral in computed))
    block = tuple(torch.stack([cols[s][k] for s in steps], dim=1) for k in range(3))
    blocks[steps] = block
    if len(blocks) > _BLOCK_CAP:
        blocks.pop(next(iter(blocks)))
    return block


@safe_cpu_compile
def mc_rough_sabr_pricer(
    K_flat,
    T_flat,
    alpha_flat,
    rho_flat,
    nu,
    H,
    n_paths=config.CALI_PATHS,
    dt=config.CALI_DT,
    device="cpu",
    cached_drivers=None,
    precomputed_beta=None,
    return_beta=False,
    return_std=False,
    seed=42,
    dtype=torch.float32,
    # default is import-frozen (evaluated once at def time): set HYBRID_KAPPA in config.py
    # before import (static-config workflow), or pass hybrid_kappa= explicitly to override.
    hybrid_kappa=config.HYBRID_KAPPA,
):
    """CMC pricer for the 1D rough SABR model (Fukasawa & Gatheral 2022)."""

    K_tensor = torch.as_tensor(K_flat, device=device, dtype=dtype)
    T_tensor = torch.as_tensor(T_flat, device=device, dtype=dtype)
    alpha_tensor = torch.as_tensor(alpha_flat, device=device, dtype=dtype)
    rho_tensor = torch.as_tensor(rho_flat, device=device, dtype=dtype)

    # steps to the longest expiry in this batch: simulation length, or truncation of a cache
    max_steps = int(math.ceil(float(T_tensor.max()) / dt))

    if cached_drivers is not None:
        fbm, kernel = cached_drivers["fbm"], cached_drivers["kernel"]
        dz_vol = cached_drivers["dz_vol"]
        var_vec = cached_drivers.get("var_vec")
        # a cache built under a different κ must not pair its driver with this call's compensator
        # (a κ=0 vs κ≥1 mismatch corrupts the martingale), so fail loudly
        assert cached_drivers.get("hybrid_kappa", 0) == hybrid_kappa, (
            f"cache HYBRID_KAPPA={cached_drivers.get('hybrid_kappa', 0)} ≠ pricer hybrid_kappa={hybrid_kappa}"
        )
        n_paths = fbm.shape[0]
    else:
        n_steps = max_steps

        dZ_all = generate_sobol_bridge_increments(n_paths, n_steps, 1, dt, device, dtype, seed=seed)
        dz_vol = dZ_all[..., 0]
        # 7919 (prime, far from the +999/+1000 increment-stream offsets) keeps the near-noise PRNG
        # stream from aliasing the Sobol increments; κ=0 draws no near-noise, so it is unaffected.
        near_gen = (
            torch.Generator(device=device).manual_seed(seed + 7919) if hybrid_kappa > 0 else None
        )
        # Sobol-bridge driver is antithetic ([dZ, −dZ]) → negate the near-noise too.
        fbm, var_vec, kernel = build_hybrid_rough_driver(
            dz_vol,
            H,
            n_steps,
            dt,
            hybrid_kappa,
            device,
            dtype,
            near_generator=near_gen,
            antithetic=True,
        )

    integral_args = (
        dz_vol,
        fbm,
        var_vec,
        kernel,
        nu,
        T_tensor,
        n_paths,
        max_steps,
        dt,
        hybrid_kappa,
        device,
        dtype,
    )
    memo = cached_drivers.get("vol_integrals") if cached_drivers is not None else None
    if memo is None:
        I_v_T, I_V2_T, I_cv_v_T = _vol_and_cv_integrals(*integral_args)
    else:
        I_v_T, I_V2_T, I_cv_v_T = _memo_vol_integrals(memo, *integral_args)

    alpha_exp = alpha_tensor.unsqueeze(0)
    rho_exp = rho_tensor.unsqueeze(0)
    rho_comp = torch.sqrt(torch.clamp(1.0 - rho_exp**2, min=1e-12))

    K_exp = K_tensor.unsqueeze(0)
    is_call_exp = (K_tensor >= 0).unsqueeze(0)
    sign = torch.where(
        is_call_exp,
        torch.tensor(1.0, device=device, dtype=dtype),
        torch.tensor(-1.0, device=device, dtype=dtype),
    )

    # conditional Bachelier payoff per path, given the vol path
    F_cond = alpha_exp * rho_exp * I_v_T
    Std_cond = alpha_exp * rho_comp * torch.sqrt(torch.clamp(I_V2_T, min=1e-12))
    pay_cond = _cond_bachelier_otm(F_cond, Std_cond, K_exp, is_call_exp, sign)

    # control-variate twin: same dZ, V≡1
    F_cv_cond = alpha_exp * rho_exp * I_cv_v_T
    Std_cv_cond = alpha_exp * rho_comp * torch.sqrt(T_tensor.unsqueeze(0))
    pay_cv_cond = _cond_bachelier_otm(F_cv_cond, Std_cv_cond, K_exp, is_call_exp, sign)

    std_cv_true = alpha_tensor * torch.sqrt(T_tensor)
    d_cv_true = -K_tensor / std_cv_true

    pdf_cv_true = INV_SQRT_2PI * torch.exp(-0.5 * d_cv_true * d_cv_true)
    cdf_true_call = 0.5 * (1.0 + torch.erf(d_cv_true * INV_SQRT_2))
    cdf_true_put = 0.5 * (1.0 + torch.erf(-d_cv_true * INV_SQRT_2))

    true_cv_call = std_cv_true * (cdf_true_call * d_cv_true + pdf_cv_true)
    true_cv_put = std_cv_true * (cdf_true_put * (-d_cv_true) + pdf_cv_true)
    true_cv_otm = torch.where(K_tensor >= 0, true_cv_call, true_cv_put)

    if precomputed_beta is None:
        mean_pay_cond = pay_cond.mean(0, keepdim=True)
        mean_pay_cv_cond = pay_cv_cond.mean(0, keepdim=True)

        cov = torch.mean((pay_cond - mean_pay_cond) * (pay_cv_cond - mean_pay_cv_cond), dim=0)
        var = torch.var(pay_cv_cond, dim=0)
        beta = cov / torch.clamp(var, min=1e-12)
    else:
        # frozen OOS beta: severs in-sample correlation
        beta = precomputed_beta

    vr_pay = pay_cond - beta * (pay_cv_cond - true_cv_otm.unsqueeze(0))
    mc_otm_price = vr_pay.mean(0)
    # put-call parity: return call-side price
    mc_call_price = torch.where(K_tensor >= 0, mc_otm_price, mc_otm_price - K_tensor)

    if return_beta:
        return mc_call_price.to(torch.float64), beta.detach()
    if return_std:
        return mc_call_price.to(torch.float64), vr_pay.std(0).to(torch.float64)
    return mc_call_price.to(torch.float64)


def oos_mc_rough_sabr_pricer(
    K_flat,
    T_flat,
    alpha_flat,
    rho_flat,
    nu,
    H,
    n_paths_main=config.CALI_PATHS,
    n_paths_pilot=4096,
    dt=config.CALI_DT,
    device="cpu",
    cached_drivers_main=None,
    cached_drivers_pilot=None,
    seed=42,
    dtype=torch.float32,
    return_std=False,
):
    """Pilot-β CV: β from an independent pilot (seed+999), applied frozen to the main run."""
    _, beta_pilot = mc_rough_sabr_pricer(
        K_flat,
        T_flat,
        alpha_flat,
        rho_flat,
        nu,
        H,
        n_paths=n_paths_pilot,
        dt=dt,
        device=device,
        cached_drivers=cached_drivers_pilot,
        return_beta=True,
        seed=seed + 999,
        dtype=dtype,
    )

    return mc_rough_sabr_pricer(
        K_flat,
        T_flat,
        alpha_flat,
        rho_flat,
        nu,
        H,
        n_paths=n_paths_main,
        dt=dt,
        device=device,
        cached_drivers=cached_drivers_main,
        precomputed_beta=beta_pilot,
        return_std=return_std,
        seed=seed,
        dtype=dtype,
    )


def _smm_plan(model, expiries, tenors):
    """Σ-independent part of _compute_mapped_smm_parameters: per swaption (start, end, π-weights),
    or None for a degenerate swap."""
    device = model.device
    dtype = model.dtype
    n_options = len(expiries)

    tau = model.tau
    F0 = model.F0
    P0 = torch.cumprod(
        torch.cat([torch.tensor([1.0], device=device, dtype=dtype), 1.0 / (1.0 + tau * F0)]), dim=0
    )

    # η(F0)=max(F0+δ,0)^β; β=0 ⇒ 1
    eta_F0 = torch.pow(torch.clamp(F0 + model.shift, min=0.0), model.beta_sabr)
    alpha_normal = model.alphas * eta_F0

    plan = []
    for i in range(n_options):
        T_exp = expiries[i]
        T_und = tenors[i]

        start_idx = torch.argmin(torch.abs(model.T - T_exp)).item()
        end_idx = torch.argmin(torch.abs(model.T - (T_exp + T_und))).item()

        # terminal tenor off the grid or zero-length swap: degenerate placeholder
        if end_idx <= start_idx:
            plan.append(None)
            continue

        P_I = P0[start_idx]
        P_J = P0[end_idx]

        taus_slice = tau[start_idx:end_idx]
        P0_slice = P0[start_idx + 1 : end_idx + 1]
        P0_j = P0[start_idx:end_idx]

        A0 = torch.sum(taus_slice * P0_slice)
        S0 = (P_I - P_J) / A0

        # reverse-cumsum for remaining annuity: sum_{k=j}^{N} (tau_k * P_k)
        sum_P_array = torch.flip(
            torch.cumsum(torch.flip(taus_slice * P0_slice, dims=[0]), dim=0), dims=[0]
        )
        pi_weights = (
            (taus_slice * P0_slice)
            / (A0 * P0_j)
            * (P_J + S0 * sum_P_array)
            * alpha_normal[start_idx:end_idx]
        )
        plan.append((start_idx, end_idx, pi_weights))
    return plan


def _compute_mapped_smm_parameters(model, Sigma_matrix, expiries, tenors, plans=None):
    """Maps the NxN FMM into 1D effective rough SABR parameters (α, ρ) per swaption; ν_SMM = model.nu.
    `plans`, a dict a Stage-2 driver cache carries, keeps the Σ-independent π-weights per target set."""
    device = model.device
    dtype = model.dtype
    if plans is None:
        plan = _smm_plan(model, expiries, tenors)
    else:
        key = (np.asarray(expiries).tobytes(), np.asarray(tenors).tobytes())
        if key not in plans:
            plans[key] = _smm_plan(model, expiries, tenors)
        plan = plans[key]

    alpha_smm_list, rho_smm_list = [], []
    for item in plan:
        if item is None:
            alpha_smm_list.append(torch.tensor(1e-5, device=device, dtype=dtype))
            rho_smm_list.append(torch.tensor(0.0, device=device, dtype=dtype))
            continue
        start_idx, end_idx, pi_weights = item

        Sigma_slice = Sigma_matrix[start_idx + 1 : end_idx + 1, start_idx + 1 : end_idx + 1]
        SigmaPi_local = torch.mv(Sigma_slice, pi_weights)
        v_0 = torch.dot(pi_weights, SigmaPi_local)
        alpha_smm_i = torch.sqrt(torch.clamp(v_0, min=1e-14))
        alpha_smm_list.append(alpha_smm_i)

        rho_slice = Sigma_matrix[start_idx + 1 : end_idx + 1, 0]
        rho_mapped = torch.sum(rho_slice * pi_weights) / alpha_smm_i
        rho_smm_list.append(torch.clamp(rho_mapped, -0.999, 0.999))

    return torch.stack(alpha_smm_list), torch.stack(rho_smm_list)


def precompute_smm_weights(model, expiries, tenors):
    """Pre-computes the Sigma-independent SMM components (pi-weights, indices) once per grid."""
    device = model.device
    dtype = model.dtype
    N1 = model.N + 1  # Sigma is (N+1) x (N+1)
    n_options = len(expiries)

    tau = model.tau
    F0 = model.F0
    eta_F0 = torch.pow(torch.clamp(F0 + model.shift, min=0.0), model.beta_sabr)
    alpha_normal = (model.alphas * eta_F0).detach()

    P0 = torch.cumprod(
        torch.cat(
            [torch.tensor([1.0], device=device, dtype=dtype), 1.0 / (1.0 + tau * F0.detach())]
        ),
        dim=0,
    )

    pi_rows, start_list, end_list = [], [], []

    for i in range(n_options):
        T_exp = float(expiries[i])
        T_und = float(tenors[i])

        start_i = torch.argmin(torch.abs(model.T - T_exp)).item()
        end_i = torch.argmin(torch.abs(model.T - (T_exp + T_und))).item()

        if end_i <= start_i:
            pi_rows.append(torch.zeros(N1, device=device, dtype=dtype))
            start_list.append(start_i + 1)
            end_list.append(start_i + 1)
            continue

        taus_slice = tau[start_i:end_i]
        P0_slice = P0[start_i + 1 : end_i + 1]
        P0_j_slice = P0[start_i:end_i]
        P_I = P0[start_i]
        P_J = P0[end_i]
        A0 = torch.sum(taus_slice * P0_slice)
        S0 = (P_I - P_J) / A0

        sum_P = torch.flip(torch.cumsum(torch.flip(taus_slice * P0_slice, [0]), 0), [0])
        pi_weights = (
            (taus_slice * P0_slice)
            / (A0 * P0_j_slice)
            * (P_J + S0 * sum_P)
            * alpha_normal[start_i:end_i]
        )

        pi_full = torch.zeros(N1, device=device, dtype=dtype)
        pi_full[start_i + 1 : end_i + 1] = pi_weights.detach()

        pi_rows.append(pi_full)
        start_list.append(start_i + 1)
        end_list.append(end_i + 1)

    return {
        "pi_ext": torch.stack(pi_rows),  # (n_options, N+1)
        "start_idx": start_list,
        "end_idx": end_list,
    }


def apply_sigma_to_smm(Sigma_matrix, smm_cache, expand=True):
    """
    Differentiable SMM mapping using pre-computed pi-weights.

    Replaces _compute_mapped_smm_parameters in gradient-hot paths.
    expand=False returns unique-pair outputs (n_unique,), used in the
    Rapisarda Jacobian step to avoid duplicate backward passes.

    Returns (α_SMM, ρ_SMM). ν is one global scalar: ν_SMM = model.nu, independent of Σ.
    """
    device = Sigma_matrix.device
    dtype = Sigma_matrix.dtype

    pi_ext = smm_cache["pi_ext"].to(device=device, dtype=dtype)  # (n_unique, N+1)

    SigmaPi = torch.mm(pi_ext, Sigma_matrix)  # (n_unique, N+1)
    v_0 = (SigmaPi * pi_ext).sum(dim=1)  # (n_unique,)
    alpha_unique = torch.sqrt(torch.clamp(v_0, min=1e-14))

    rho_num = torch.mv(pi_ext, Sigma_matrix[:, 0])  # (n_unique,)
    rho_unique = torch.clamp(rho_num / alpha_unique, -0.999, 0.999)

    if expand and "expand_idx" in smm_cache:
        expand_idx = smm_cache["expand_idx"].to(device=device)
        return alpha_unique[expand_idx], rho_unique[expand_idx]
    return alpha_unique, rho_unique


def mapped_smm_mc_pricer(
    model,
    Sigma_matrix,
    expiries,
    tenors,
    strike_offsets,
    dt=1.0 / 24.0,
    n_paths=config.CALI_PATHS,
    cached_drivers=None,
):
    """Prices swaptions via the mapped SMM approximation (Adachi et al. 2025) using the 1D MC engine."""
    device = model.device
    dtype = model.dtype

    plans = cached_drivers.get("smm_plans") if cached_drivers is not None else None
    alpha_smm, rho_smm = _compute_mapped_smm_parameters(
        model, Sigma_matrix, expiries, tenors, plans=plans
    )

    K_flat = torch.tensor(strike_offsets, device=device, dtype=dtype)
    T_flat = torch.tensor(expiries, device=device, dtype=dtype)

    mc_prices_t = mc_rough_sabr_pricer(
        K_flat,
        T_flat,
        alpha_smm,
        rho_smm,
        model.nu,
        model.H.item(),
        n_paths=n_paths,
        dt=dt,
        device=device,
        cached_drivers=cached_drivers,
    )

    ivs_t = bachelier_iv_newton(mc_prices_t, K_flat, T_flat, initial_guess_vol=alpha_smm)
    return ivs_t.detach().cpu().numpy()


@safe_cpu_compile
def _core_fukasawa_ode(k, T, alpha_smm, rho_smm, nu, H):
    """Fukasawa & Gatheral (2022) rough SABR IV formula on plain tensors (no model object)."""
    gamma_H = torch.exp(torch.lgamma(H + 0.5))
    nu_eff = nu / (torch.sqrt(2.0 * H) * gamma_H)  # nu_eff^2 T^{2H} = nu^2 Var Z~(T), the level

    # kappa drives xi = alpha^2: twice the log-alpha kernel (see rough_sabr_vol_ode)
    kappa_T = 2.0 * nu / gamma_H * (T ** (H - 0.5))
    y = kappa_T * k / alpha_smm

    rho_comp = torch.sqrt(torch.clamp(1.0 - rho_smm**2, min=1e-8))
    rho_ratio = rho_smm / rho_comp
    atan_1 = torch.atan(rho_ratio)
    rho_coeff = 2.0 * rho_ratio

    h_term1 = 2.0 * H + 1.0
    h_term2 = 2.0 * H + 3.0
    w_0 = 3.0 * (1.0 - 2.0 * H) / h_term2
    w_half = 2.0 * H / h_term2

    x_0 = y / h_term1
    x_half = 2.0 * x_0

    # G_0: rough limit H -> 0
    term1_0 = torch.log(torch.clamp(1.0 + 2.0 * rho_smm * x_0 + x_0**2, min=1e-10))
    atan_2 = torch.atan((x_0 + rho_smm) / rho_comp)
    g_0_eval = term1_0 + rho_coeff * (atan_1 - atan_2)

    # G_{1/2}: classical SABR limit H = 1/2
    sq_arg = torch.clamp(1.0 + rho_smm * x_half + (x_half**2) / 4.0, min=1e-10)
    inner = torch.sqrt(sq_arg) - rho_smm - x_half / 2.0
    log_arg = torch.clamp(inner / (1.0 - rho_smm), min=1e-10)
    g_half_eval = 4.0 * (torch.log(log_arg)) ** 2

    G_A = (h_term1**2) * (w_0 * g_0_eval + w_half * g_half_eval)
    G_A = torch.clamp(G_A, min=1e-12)

    # shape multiplier |y|/sqrt(G_A), =1 at ATM (y=0)
    shape_multiplier = torch.where(
        torch.abs(y) < 1e-8,
        torch.ones_like(y),
        torch.abs(y) / torch.sqrt(G_A),
    )

    drift = (2.0 - 3.0 * rho_smm**2) / 24.0 * (nu_eff**2) * (T ** (2.0 * H))

    return alpha_smm * shape_multiplier * (1.0 + drift)


def mapped_smm_ode_pricer(model, Sigma_matrix, expiries, tenors, strike_offsets):
    """Low-fidelity ODE surrogate for Stage 2 AMMO (Fukasawa & Gatheral 2022)."""
    device = model.device
    dtype = model.dtype

    alpha_smm, rho_smm = _compute_mapped_smm_parameters(model, Sigma_matrix, expiries, tenors)

    k_t = torch.tensor(strike_offsets, device=device, dtype=dtype)
    T_t = torch.clamp(torch.tensor(expiries, device=device, dtype=dtype), min=1e-4)

    return _core_fukasawa_ode(k_t, T_t, alpha_smm, rho_smm, model.nu, model.H)


@safe_cpu_compile
def oracle_rough_sabr_pricer(
    K_flat,
    T_flat,
    alpha_flat,
    rho_flat,
    nu,
    H,
    n_paths=1_000_000,
    dt=config.CALI_DT,
    device="cpu",
    hybrid_kappa=config.HYBRID_KAPPA,
):
    """PRNG oracle pricer: pseudo-random drivers, no CV. CLT benchmark for bias tests."""
    dtype = torch.float32

    K_tensor = torch.as_tensor(K_flat, device=device, dtype=dtype)
    T_tensor = torch.as_tensor(T_flat, device=device, dtype=dtype)
    alpha_tensor = torch.as_tensor(alpha_flat, device=device, dtype=dtype)
    rho_tensor = torch.as_tensor(rho_flat, device=device, dtype=dtype)

    n_steps = int(math.ceil(float(T_tensor.max()) / dt))

    dZ_all = generate_prng_increments(n_paths, n_steps, 1, dt, device, dtype)
    dz_vol = dZ_all[..., 0]
    # near-noise uses the global RNG (None), matching this oracle's unseeded PRNG increments.
    # generate_prng_increments is not antithetic, so antithetic=False (default): no half-split.
    fbm, var_vec, kernel = build_hybrid_rough_driver(
        dz_vol, H, n_steps, dt, hybrid_kappa, device, dtype, near_generator=None
    )

    _, _, I_v_T, I_V2_T = _cmc_vol_integrals(
        dz_vol,
        fbm,
        var_vec,
        kernel,
        nu,
        T_tensor,
        n_paths,
        n_steps,
        dt,
        hybrid_kappa,
        device,
        dtype,
    )

    alpha_exp = alpha_tensor.unsqueeze(0)
    rho_exp = rho_tensor.unsqueeze(0)
    rho_comp = torch.sqrt(torch.clamp(1.0 - rho_exp**2, min=1e-12))

    K_exp = K_tensor.unsqueeze(0)
    is_call_exp = (K_tensor >= 0).unsqueeze(0)
    sign = torch.where(
        is_call_exp,
        torch.tensor(1.0, device=device, dtype=dtype),
        torch.tensor(-1.0, device=device, dtype=dtype),
    )

    F_cond = alpha_exp * rho_exp * I_v_T
    Std_cond = alpha_exp * rho_comp * torch.sqrt(torch.clamp(I_V2_T, min=1e-12))
    pay_cond = _cond_bachelier_otm(F_cond, Std_cond, K_exp, is_call_exp, sign)

    # pure CMC mean, no control variate
    mc_otm_price = torch.mean(pay_cond, dim=0)
    # put-call parity
    mc_call_price = torch.where(K_tensor >= 0, mc_otm_price, mc_otm_price - K_tensor)

    return mc_call_price.to(torch.float64)
