import numpy as np
import torch
import math


def build_rapisarda_correlation_matrix(omega):
    """NxN correlation matrix from lower-triangular angles via Rapisarda et al. (2007)."""
    M = omega.shape[0]
    device = omega.device
    dtype = omega.dtype

    C = torch.cos(omega)
    S = torch.sin(omega)

    # shift S right by one col, padding first col with 1: prod_{k<0} = 1
    S_shifted = torch.cat([torch.ones(M, 1, dtype=dtype, device=device), S[:, :-1]], dim=1)
    S_cumprod = torch.cumprod(S_shifted, dim=1)

    # diagonal of C must be 1 so B_{i,i} = S_cumprod_{i,i}
    C_mod = C.clone()
    C_mod.diagonal().fill_(1.0)

    B = torch.tril(C_mod * S_cumprod)
    Sigma = torch.matmul(B, B.T)

    # clamp floating-point noise at boundaries
    Sigma = torch.clamp(Sigma, -1.0, 1.0)
    Sigma.diagonal().fill_(1.0)

    return Sigma


def find_l_curve_corner(lambda_grid, eta_list, penalty_norms):
    """2D curvature of the L-curve (axis-normalized, Hansen-pruned); returns λ at maximum curvature.

    Both axes are mapped to [0,1] before differentiation so scale differences do not bias
    curvature.  The argmax is restricted to the Hansen monotone corner region: points where
    the penalty norm is decreasing (dpen < 0) and the residual norm is increasing (dres > 0).
    This prevents denominator-collapse on the flat arms, where one or both first derivatives
    vanish.  Falls back to interior indices [1:-1] when the corner
    region is degenerate (fewer than 2 qualifying points).
    """
    penalty_norms = np.array(penalty_norms, dtype=float)
    residual_norms = np.array(eta_list, dtype=float)

    # parameterise by log10(λ) for uniform spacing on geometric grid
    t = np.log10(lambda_grid)

    # axis-normalize to [0,1]; guard zero-range axes against NaN
    _EPS_RANGE = 1e-12
    pen_norm = (penalty_norms - penalty_norms.min()) / max(
        penalty_norms.max() - penalty_norms.min(), _EPS_RANGE
    )
    res_norm = (residual_norms - residual_norms.min()) / max(
        residual_norms.max() - residual_norms.min(), _EPS_RANGE
    )

    dpen_dt = np.gradient(pen_norm, t)
    dres_dt = np.gradient(res_norm, t)
    d2pen_dt2 = np.gradient(dpen_dt, t)
    d2res_dt2 = np.gradient(dres_dt, t)

    numerator = np.abs(dpen_dt * d2res_dt2 - dres_dt * d2pen_dt2)
    denominator = (dpen_dt**2 + dres_dt**2) ** 1.5
    denominator = np.where(denominator < 1e-12, 1e-12, denominator)
    curvature = numerator / denominator

    # Hansen monotone corner: both arms actively trading off (pen ↓, res ↑)
    corner_mask = (dpen_dt < 0) & (dres_dt > 0)
    if corner_mask.sum() >= 2:
        search_idx = np.where(corner_mask)[0]
    elif len(lambda_grid) >= 5:
        search_idx = np.arange(1, len(lambda_grid) - 1)
    else:
        search_idx = np.arange(len(lambda_grid))

    best_idx = int(search_idx[np.argmax(curvature[search_idx])])
    return lambda_grid[best_idx], curvature


def compute_sigma_structural_metrics(Sigma, N):
    """Structural diagnostics on the calibrated Sigma from Stage 2.

    Examines free rows Sigma[idx, 1:idx] for idx = i+1, i in range(2, N).
    Col-0 is excluded (anchor-fixed by ω_{i,0} = arccos ρ; constant across λ).
    Sigma has shape (N+1, N+1); max idx = N < N+1, so no out-of-bounds access.
    """
    sign_flips = 0
    mono_violations = 0
    for i in range(2, N):
        idx = i + 1
        row = Sigma[idx, 1:idx]
        sign_flips += int(np.sum(row < 0))
        mono_violations += int(np.sum(np.diff(row) < 0))
    return {"mono_violations": mono_violations, "sign_flips": sign_flips}


def select_lambda_monotone(lambda_grid, struct_metrics_list):
    """λ selector for Stage 2: smallest λ beginning a sustained mono_viol=0 run.

    Requires ≥2 consecutive mono_viol=0 to guard against spurious isolated zeros: mono_viol
    can oscillate at low λ, so first-hit is unreliable.
    Returns the lower edge of the first such run (least over-smoothed monotone pick).

    Fallback when no ≥2-consecutive run exists: argmin(mono_viol), fallback_used=True.

    sign_flips are logged by the caller but are not a selection constraint.

    Returns (best_lambda, chosen_idx, fallback_used).
    """
    lambda_grid = np.asarray(lambda_grid)
    mono_viol = [m["mono_violations"] for m in struct_metrics_list]
    n = len(mono_viol)

    run_start = None
    for i in range(n - 1):
        if mono_viol[i] == 0 and mono_viol[i + 1] == 0:
            run_start = i
            break

    if run_start is None:
        chosen_idx = int(np.argmin(mono_viol))
        return float(lambda_grid[chosen_idx]), chosen_idx, True

    return float(lambda_grid[run_start]), run_start, False


def build_brownian_bridge_matrix(n_steps):
    """Brownian Bridge transformation matrix: maps N Sobol draws to N bridge-ordered increments."""
    t = np.linspace(0, 1, n_steps + 1)
    A = np.zeros((n_steps + 1, n_steps))

    # BFS bisection: macro-structure in the lowest Sobol dimensions
    intervals = [(0, n_steps)]
    # dimension 0 sets the terminal point W_N
    A[n_steps, 0] = np.sqrt(t[n_steps] - t[0])
    z_idx = 1

    while intervals:
        left, right = intervals.pop(0)
        if right - left <= 1:
            continue

        mid = (left + right) // 2

        weight_left = (t[right] - t[mid]) / (t[right] - t[left])
        weight_right = (t[mid] - t[left]) / (t[right] - t[left])
        variance = (t[mid] - t[left]) * (t[right] - t[mid]) / (t[right] - t[left])

        A[mid, :] = weight_left * A[left, :] + weight_right * A[right, :]
        A[mid, z_idx] = np.sqrt(variance)
        z_idx += 1

        # BFS: sub-intervals appended so macro-structure uses lowest Sobol dimensions
        intervals.append((left, mid))
        intervals.append((mid, right))

    dW_matrix = A[1:, :] - A[:-1, :]
    # normalise to N(0,1) so callers apply dt-scaling as usual
    dt = 1.0 / n_steps
    transformation_matrix = dW_matrix / np.sqrt(dt)

    return torch.tensor(transformation_matrix, dtype=torch.float64)


def generate_sobol_bridge_increments(n_paths, n_steps, n_factors, dt, device, dtype, seed=42):
    """Variance-reduced Brownian increments: Sobol + Brownian Bridge + Antithetic Variates."""
    assert n_paths % 2 == 0, "n_paths must be even (antithetic variates)."

    half_paths = n_paths // 2
    dim = n_steps * n_factors
    if dim > 21201:
        # above the SobolEngine cap: seeded PRNG, no bridge; the [z; -z] split mirrors
        # the Sobol branch so the downstream antithetic near-noise pairing still holds
        gen = torch.Generator(device=device).manual_seed(seed)
        z_half = torch.randn(
            (half_paths, n_steps, n_factors), device=device, dtype=dtype, generator=gen
        ) * math.sqrt(dt)
        return torch.cat([z_half, -z_half], dim=0)

    dt_t = torch.tensor(dt, device=device, dtype=dtype)

    sobol_gen = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=seed)
    sobol_draws = sobol_gen.draw(half_paths).to(dtype=dtype, device=device)
    # clamp to prevent ±∞ at boundaries during ICDF mapping
    sobol_draws = torch.clamp(sobol_draws, min=1e-7, max=1.0 - 1e-7)

    Z_base = 1.4142135623730951 * torch.erfinv(2.0 * sobol_draws - 1.0)
    Z_base = Z_base.view(half_paths, n_factors, n_steps)

    B_t = build_brownian_bridge_matrix(n_steps).to(device=device, dtype=dtype)
    Z_bridge = torch.matmul(Z_base, B_t.T)
    Z_bridge = Z_bridge.transpose(1, 2)

    dZ_half = Z_bridge * torch.sqrt(dt_t)
    # antithetic mirror
    return torch.cat([dZ_half, -dZ_half], dim=0)


def build_fractional_kernel(H, n_steps, dt, device, dtype):
    """Lower-triangular κ=0 Volterra-Euler kernel (eq::wk), weights (k^(H+½) − (k−1)^(H+½))/(H+½).

    At κ ≥ 1 build_hybrid_rough_driver zeroes its first κ lags and replaces them with exact
    near integrals; this function always returns the full kernel.
    """
    dt_t = torch.tensor(dt, device=device, dtype=dtype)
    H_t = (
        torch.tensor(H, device=device, dtype=dtype) if not torch.is_tensor(H) else H.to(dtype=dtype)
    )

    alpha_H = H_t - 0.5
    # log-gamma for numerical stability of Γ(H+0.5)
    gamma_const = torch.exp(torch.lgamma(H_t + 0.5))

    steps = torch.arange(n_steps, device=device, dtype=dtype)
    k_mat = torch.clamp(steps[:, None] - steps[None, :] + 1, min=0.0)

    # unified weights: (k^(H+0.5) - (k-1)^(H+0.5)) / (H+0.5)
    k_minus_1 = torch.clamp(k_mat - 1.0, min=0.0)
    w = (torch.pow(k_mat, alpha_H + 1.0) - torch.pow(k_minus_1, alpha_H + 1.0)) / (alpha_H + 1.0)

    # scale by dt^(H-0.5)/Γ(H+0.5); tril enforces non-anticipativity
    return torch.tril(w * torch.pow(dt_t, alpha_H) / gamma_const)


def build_hybrid_cell_covariance(H, dt, kappa, device="cpu", dtype=torch.float64):
    """Per-cell covariance objects for the Bennedsen–Lunde–Pakkanen (2017) hybrid scheme.

    Returns the constants Σ, B, Σ_cond used to draw the per-cell Gaussian vector
    W_j = (ΔW_j, W_{j,1}, …, W_{j,κ}) ~ N(0, Σ) of the hybrid driver, where
    W_{j,k} = ∫_{cell j} ((j+k)·dt − s)^α dW(s) is the k-th near power-weighted Wiener
    integral (α = H − ½), per BLP Eq 3.5 specialized to jolto's constant L_g = 1/Γ(H+½).
    Σ here is the *pure* power-integral
    covariance; the 1/Γ factor is applied once when the driver is assembled.

    Σ, B, Σ_cond depend only on (H, dt): they are constants w.r.t. every calibrated
    parameter, built outside the autograd graph as plain tensors (never nn.Parameter).

    Parameters
    ----------
    H : float          Hurst exponent (H < ½ is the rough regime).
    dt : float         Uniform time step.
    kappa : int        Number of near (exact) lags, ∈ {1, 2}.

    Returns
    -------
    dict with float64 tensors on `device`:
        "Sigma"      : (κ+1, κ+1) symmetric PSD, joint cov of (ΔW, W_{·,1..κ}).
                       0-indexed: index 0 = ΔW, index k = lag-k integral.
        "B"          : (κ,)        regression coeffs Σ_{[1:κ],0} / Σ_{0,0}.
        "Sigma_cond" : (κ, κ)      Schur complement Σ_{[1:κ],[1:κ]} − Σ_{[1:κ],0}Σ_{0,0}⁻¹Σ_{0,[1:κ]}.

    Raises
    ------
    ValueError if kappa ∉ {1, 2}, or if a variance/eigenvalue is meaningfully negative
    (beyond roundoff).
    """
    if kappa not in (1, 2):
        raise ValueError(f"build_hybrid_cell_covariance supports kappa ∈ {{1, 2}}, got {kappa}")

    alpha = H - 0.5
    # independent of all calibrated params, so plain float
    s_aa = dt ** (2.0 * alpha + 1.0)  # dt^(2α+1): variance scale for near integrals
    s_a = dt ** (alpha + 1.0)  # dt^(α+1): ΔW × near-integral cross scale

    # Σ entries (BLP Eq 3.5). 0=ΔW, 1=lag-1, 2=lag-2.
    Sig = np.zeros((kappa + 1, kappa + 1), dtype=np.float64)
    Sig[0, 0] = dt
    Sig[0, 1] = s_a / (alpha + 1.0)
    Sig[1, 1] = s_aa / (2.0 * alpha + 1.0)
    if kappa == 2:
        from scipy.special import hyp2f1

        Sig[0, 2] = s_a * (2.0 ** (alpha + 1.0) - 1.0) / (alpha + 1.0)
        Sig[2, 2] = s_aa * (2.0 ** (2.0 * alpha + 1.0) - 1.0) / (2.0 * alpha + 1.0)
        Sig[1, 2] = s_aa / (alpha + 1.0) * 2.0**alpha * float(hyp2f1(-alpha, 1.0, alpha + 2.0, 0.5))
    Sig = Sig + np.triu(Sig, 1).T  # mirror upper → symmetric

    B_np = Sig[1:, 0] / Sig[0, 0]  # (κ,)
    cond_np = Sig[1:, 1:] - np.outer(B_np, Sig[0, 1:])  # (κ, κ) Schur complement

    # PSD guards. σ²_cond → 0 as α → 0; clamp roundoff-negative, raise if material.
    var_floor = -1e-10 * s_aa  # roundoff tolerance scaled to the variance magnitude
    if kappa == 1:
        # Numerically stable σ²_cond = dt^(2α+1)·α²/[(2α+1)(α+1)²]: the algebraic identity
        # (α+1)² − (2α+1) = α² removes the 1/(2α+1) − 1/(α+1)² cancellation near α=0.
        sig2 = s_aa * alpha**2 / ((2.0 * alpha + 1.0) * (alpha + 1.0) ** 2)
        if sig2 < var_floor:
            raise ValueError(f"σ²_cond = {sig2:.3e} meaningfully negative at H={H}, dt={dt}")
        cond_np[0, 0] = max(sig2, 0.0)
    else:
        eigvals = np.linalg.eigvalsh(0.5 * (cond_np + cond_np.T))
        if eigvals.min() < var_floor:
            raise ValueError(
                f"Σ_cond min eigenvalue {eigvals.min():.3e} meaningfully negative at H={H}, dt={dt}"
            )
        if eigvals.min() < 0.0:  # clamp roundoff-negative eigenvalue to 0, reconstruct
            evals, evecs = np.linalg.eigh(0.5 * (cond_np + cond_np.T))
            cond_np = (evecs * np.clip(evals, 0.0, None)) @ evecs.T

    return {
        "Sigma": torch.tensor(Sig, dtype=dtype, device=device),
        "B": torch.tensor(B_np, dtype=dtype, device=device),
        "Sigma_cond": torch.tensor(cond_np, dtype=dtype, device=device),
    }


def build_hybrid_rough_driver(
    dz_vol, H, n_steps, dt, kappa, device, dtype, near_generator=None, antithetic=False
):
    """Bennedsen–Lunde–Pakkanen (2017) hybrid rough-vol driver Z̃; kappa=0 returns the Euler line.

    Replaces the first ``kappa`` near-singular lags of the Volterra-Euler convolution with their
    exact power-weighted Wiener integrals, drawn from the per-cell conditional law given the same
    increment ΔZ.

    Parameters
    ----------
    dz_vol : (n_paths, n_steps) tensor   the plain increments ΔZ (Sobol/PRNG), the same tensor
                                         that drives the I_v integral downstream; never modified.
    H, dt : float                        Hurst exponent and time step.
    kappa : int                          number of exact near lags, ∈ {0, 1, 2}.
    near_generator : torch.Generator or None
                                         PRNG for the conditional near-noise ε (independent of
                                         ΔZ). None → global RNG (matches the unseeded oracle).
    antithetic : bool                    True when dz_vol is a half-split
                                         (dz_vol[B/2:] == −dz_vol[:B/2]): ε is drawn for the first
                                         half and negated, so Z̃ is antisymmetric across the pair.
                                         Explicit because an even unpaired batch looks the same;
                                         negation leaves the per-cell conditional law unchanged.

    Returns
    -------
    (fbm, var_vec, kernel):
        fbm     : (n_paths, n_steps)  the driver Z̃ (= ΔZ @ K_tailᵀ + 1/Γ · aligned near integrals).
        var_vec : (n_steps,) or None  deterministic Var(Z̃(t_i)) compensator vector; None for
                                       kappa=0 (signal: caller uses the ½ν²·ΣK²·dt line).
        kernel  : (n_steps, n_steps)  the full Volterra-Euler kernel K (for the kappa=0 line).
    """
    kernel = build_fractional_kernel(H, n_steps, dt, device, dtype)
    if kappa == 0:
        return torch.matmul(dz_vol, kernel.T), None, kernel

    n_paths = dz_vol.shape[0]
    gamma_const = torch.exp(torch.lgamma(torch.tensor(H + 0.5, device=device, dtype=dtype)))
    inv_g = 1.0 / gamma_const

    # K_tail: zero the first kappa diagonals (lags 1..kappa, i.e. 0 <= i-j < kappa).
    steps = torch.arange(n_steps, device=device)
    lag_m1 = steps[:, None] - steps[None, :]  # i - j (lag = i-j+1)
    K_tail = kernel.clone()
    K_tail[(lag_m1 >= 0) & (lag_m1 < kappa)] = 0.0
    fbm = torch.matmul(dz_vol, K_tail.T)  # same matmul, same cost

    # per-cell conditional near draw; Σ, B, Σ_cond are grad-free (H is a float here)
    with torch.no_grad():
        cell = build_hybrid_cell_covariance(H, dt, kappa, device=device, dtype=dtype)
        Sigma, B = cell["Sigma"], cell["B"]
        # L_cond: eigh-based PSD square-root, robust to the H=½ zero eigenvalue (kept out of any
        # autograd graph so its degenerate backward is never exercised).
        evals, evecs = torch.linalg.eigh(cell["Sigma_cond"])
        L_cond = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))

    if antithetic:
        # draw the first half, negate it. The merged FMM batch keeps
        # the half-split (p is the major reshape index → dz_vol[B/2:] == −dz_vol[:B/2]), so the
        # same rule serves both the 1D and FMM callers. Pairs with the negated ΔW so the per-cell
        # draw negates in full.
        assert n_paths % 2 == 0, "antithetic near-noise requires an even batch (half-split)."
        half = n_paths // 2
        eps_half = torch.randn(
            half, n_steps, kappa, device=device, dtype=dtype, generator=near_generator
        )
        eps = torch.cat([eps_half, -eps_half], dim=0)
    else:
        eps = torch.randn(
            n_paths, n_steps, kappa, device=device, dtype=dtype, generator=near_generator
        )
    # W_near[:, m, k] = lag-(k+1) pure power integral of cell m  (1/Γ applied at assembly).
    W_near = dz_vol.unsqueeze(-1) * B + eps @ L_cond.T  # (n_paths, n_steps, kappa)

    # the lag-(k+1) integral of cell m replaces the zeroed K[i,m] at i = m + k, so it
    # lands on row m+k (shift down by k). lag-1 unshifted; lag-2 shifted by one (row 0 gets none).
    near_total = torch.zeros(n_paths, n_steps, device=device, dtype=dtype)
    for k in range(kappa):
        near_total[:, k:] = near_total[:, k:] + W_near[:, : n_steps - k, k]
    fbm = fbm + inv_g * near_total

    # compensator from the construction, not Euler ΣK²·dt or the closed form: near cells and
    # tail cells are disjoint and independent, so Var is the sum of per-cell variances. The near
    # part already carries dt^(2α+1) inside Σ_kk → no trailing ·dt; only the tail part gets ·dt.
    diag_cumsum = torch.cumsum(torch.diagonal(Sigma)[1:], dim=0)  # Σ_11, Σ_11+Σ_22, ...
    near_counts = torch.clamp(steps + 1, max=kappa)  # min(i+1, kappa) per row
    near_var = inv_g * inv_g * diag_cumsum[near_counts - 1]
    tail_var = dt * torch.sum(K_tail**2, dim=1)
    var_vec = near_var + tail_var

    return fbm, var_vec, kernel


def rough_vol_compensator(nu_scalar, kernel, var_vec, dt_t, kappa):
    """½·ν²·Var(Z̃(t_i)) Itô compensator, shared by every 1D driver consumer.

    kappa=0 is the ½ν²·Σ_j K²_{ij}·dt expression (eq::var_comp). The hybrid (kappa>0) uses the
    construction variance ``var_vec``, which already carries every dt power; a hybrid Z̃ paired
    with the Euler compensator breaks the martingale, so that pairing fails loudly.
    """
    if kappa == 0:
        return 0.5 * (nu_scalar**2) * torch.sum(kernel**2, dim=1).view(1, 1, -1) * dt_t
    assert var_vec is not None, (
        "HYBRID_KAPPA>0 but var_vec is None — hybrid Z̃ would pair with the Euler compensator"
    )
    return 0.5 * (nu_scalar**2) * var_vec.view(1, 1, -1)


def generate_prng_increments(n_paths, n_steps, n_factors, dt, device, dtype):
    """Plain PRNG Brownian increments for the CLT oracle (no Sobol, bridge, or antithetics)."""
    return torch.randn((n_paths, n_steps, n_factors), device=device, dtype=dtype) * math.sqrt(dt)
