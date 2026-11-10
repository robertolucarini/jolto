"""
Kernel-discretization bias diagnostic: Volterra-Euler and BLP hybrid vs an exact reference.

Four arms share one draw of PRNG increments ΔZ: the pure-Riemann Volterra-Euler kernel (κ=0,
build_fractional_kernel), the BLP hybrid driver at κ=1 and κ=2 (build_hybrid_rough_driver), and an
exact conditional-correction reference (κ=∞). Each non-exact arm's bias is measured against the
exact arm at the production operating point (H=0.10, dt=1/50), with H=0.05 retained as the
small-H hybrid-necessity anchor.

The exact arm takes the Euler arm's FBM on the same ΔZ and adds an independent
Gaussian correction calibrated to the exact residual covariance C_resid = C_exact − C_euler,
where C_euler = Δt·K·Kᵀ is the Euler FBM covariance and C_exact is the RL-FBM exact covariance.

Protocol
--------
- H=0.5 sanity gate runs first: C_resid=0 there, so Δ must be ≈0 within paired SE.
- Two full-assert arms follow only if the gate passes: H=0.10 (production) and H=0.05 (anchor).
- Production-κ decision pass at H=0.10 over both grids: dt=1/50 (matches calibration) and
  dt=1/24 (the production Bermudan pricing grid that gates κ).
- dt-sweep at H=0.10 over dt=1/24, 1/50 and 1/100 at fixed N.
- Optional H=0.20 BLP-consistency check (bias shrinks as H rises).
- Plain PRNG on every arm (no Sobol/bridge/antithetics): isolates discretisation bias.
- Paired (CRN) SE: std(ΔP^m)/√N converted to bps, far smaller than per-arm SE.

Usage
-----
    python tests/test_kernel_discretization_bias.py

Outputs
-------
    results/kernel_bias_H<H>.csv                  : per (H, T, K) bias table
    results/kernel_kappa_decision_dt<g>_T<T>.csv  : per-grid decision-pass cells
    results/kernel_dt_sweep_H0.10.csv             : ATM mean|Δ| and paired SE per (dt, κ)
    Console headline: mean/max |Δ| vs paired SE vs Stage-1 RMSE, per-grid production-κ read-off.
"""

import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

from jolto import config, val_config
from jolto.core.analytics import (
    build_fractional_kernel,
    build_hybrid_cell_covariance,
    build_hybrid_rough_driver,
    rough_vol_compensator,
)
from jolto.calibration.pricers import INV_SQRT_2, INV_SQRT_2PI, bachelier_iv_newton
from jolto.core.market_data import load_swaption_vol_surface
from _common import _grid_label, _stage1_rmse

DEVICE = "cpu"
DTYPE = torch.float64
DT_CALIB = config.CALI_DT  # Stage-1 calibration MC grid (1/50)
DT_PRICE = val_config.PRICE_DT  # production Bermudan pricing grid (1/24)
DT_SWEEP_FINE = 1.0 / 100.0  # diagnostic-only fine reference for the dt-sweep, not a config grid
N_PATHS = config.CALI_PATHS

H_SANITY = 0.5
H_PRODUCTION = val_config.VAL_H_TARGET
H_FULL_ASSERT = [H_PRODUCTION, 0.05]  # both carry the full battery; 0.05 = κ1≪κ0 anchor
H_COMPARISON = [0.20]  # BLP sanity: gap shrinks as H → 0.5

# Quadrature
N_QUAD_GJ = 64  # Gauss-Jacobi nodes for well-separated pairs
N_QUAD_GL = 128  # Gauss-Legendre nodes for log-scale COV piece
COV_THRESHOLD = 0.10  # r-1 below this → use COV; above → plain GJ

# Paths
STAGE1_CSV = os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)
RESULTS_DIR = config.RESULTS_DIR

ABS_TOL_BPS = 0.05  # absolute floor for H=0.5 gate (far above roundoff, far below any bias)

# Hybrid arm
HYBRID_KAPPAS = (1, 2)  # near-lag truncations to validate against the κ=∞ exact arm
NEAR_SEED_OFF = 7919  # near-noise PRNG offset (independent of ΔZ/eps)
EV_Z_TOL = 5.0  # E[V] martingality: max |E[V]−1| within this many SE of 1
KAPPA2_SE_MULT = 5.0  # κ=2 "within a few paired SE of exact" bound (max|Δ| / SE)
# Structural anti-second-matmul guard. The build is accounted for by one matmul + the kernel
# build + the near draw; an accidental second N·n² matmul (the eps@L_residᵀ failure mode) would
# add ≈ 1.0·t_matmul to the residual. The legitimate residual is sub-matmul: it absorbs build's
# post-matmul O(N·n) assembly (the fbm+near add, var_vec) that no timed component covers, about
# 0.3·t_matmul at about a thousand steps and shrinking as a fraction as n grows. The 0.5 bound clears that while
# still tripping on a second matmul (~1.3·t_matmul). It is not a build/κ0 ratio bound: near-noise
# randn is wall-clock-comparable to the matmul, so the build ratio is ~1.6–2.6× (≈2× the
# near-noise at κ=2) with no second matmul.
MATMUL_RESIDUAL_FRAC = 0.5
ARMS_NONEXACT = ["euler", "hybrid_k1", "hybrid_k2"]
KAPPA_ARM = {0: "euler", 1: "hybrid_k1", 2: "hybrid_k2"}  # κ → bias-table arm name
# A failed Bachelier IV inversion (deep-OTM × very short T → vega≈0, price≈0) is not a bias
# measurement: it maps near-identical prices to garbage normal vols (1e3–1e7 bps) while the
# paired SE stays ~0. Stats/asserts use only cells whose inverted vol lands in this sane band.
IV_FLOOR_BPS = 1.0  # below ⇒ degenerate/failed inversion
IV_CAP_BPS = 1000.0  # above ⇒ failed inversion (10% normal vol ≫ any real swaption)
# The wings (|K|≳0.01) carry residual IV-inversion noise
# even after the band filter, so the success-criteria asserts use the near-ATM strikes only
# (the full smile is still reported and saved). The κ-convergence claim is an ATM statement.
ATM_BAND_K = 0.005

# Production-κ decision pass (additive; leaves the full-assert arms and the gate untouched).
# Settling κ needs a paired SE small enough to resolve the sub-bps κ=1/κ=2 residual. The
# paired difference carries the hybrid's independent near-ε and the exact arm's independent
# eps@L_residᵀ, so SE only shrinks as 1/√N → we raise N. But the pricing H runs to the
# longest expiry (n_steps = T_max/dt); a single (N×n) float64 tensor at N=262144, n=1500 is ~3.1 GB (working
# set ~16 GB → OOM). So the decision pass runs a representative ATM expiry subset (production-relevant
# 1Y/5Y/10Y) and sizes N per expiry to a fixed memory budget: short expiries (small n) afford a far
# larger N, so their SE is tightest where production lives.
DECISION_EXPIRIES = (1.0, 5.0, 10.0)  # n_steps = 50/250/500 (dt=1/50) or 24/120/240 (dt=1/24)
DECISION_MEM_BUDGET_GB = 6.0  # per-pass working-set budget for N sizing
DECISION_TENSOR_MULT = 6  # ≈ live (N×n) float64 tensors at peak (4 unit_vols + dZ + transients)
DECISION_N_CEIL = 1_048_576  # cap N for short expiries (1Y would otherwise size to millions)
DECISION_N_FLOOR = N_PATHS  # never resolve worse than the existing run
# Read-off: production κ is the smallest κ whose ATM residual is negligible in absolute bps vs the
# Stage-1 calibration RMSE; this absolute floor alone is the materiality gate.
# An SE gate (mean|Δ| ≤ pairedSE) is not used: SE is path-count-dependent (→0 as N grows),
# so it would reject any fixed nonzero residual regardless of materiality. ≲SE stays a printed
# diagnostic ("is the residual even resolvable at this N?"), not a criterion.
DECISION_NEGLIGIBLE_BPS = 1.0  # immaterial vs the calibration error


def _rl_diag(t_grid, H):
    """Exact Var(Z̃(t_i)) = t_i^{2H} / (2H·Γ(H+½)²) for all steps."""
    gH = math.gamma(H + 0.5)
    return t_grid ** (2.0 * H) / (2.0 * H * gH**2)


def _quad_nodes(alpha):
    """Quadrature shared by _build_C_exact and its self-check: Gauss-Jacobi (weight u^α) and
    Gauss-Legendre nodes/weights mapped to [0,1], and CONST_H = ∫_0^1 s^α(1+s)^α ds.

    Singular endpoint is u=0, so the Jacobi weight must be (1+x)^α → roots_jacobi(n, 0, α).
    Map u=(1+x)/2, du=dx/2: ∫_0^1 u^α g du = 2^{-(α+1)} Σ w_i g(u_i).
    """
    xi_raw, wi_raw = roots_jacobi(N_QUAD_GJ, 0.0, alpha)
    xi_gj = 0.5 * (xi_raw + 1.0)
    wi_gj = (0.5 ** (alpha + 1.0)) * wi_raw

    # GL nodes/weights for ∫_0^1 f(x) dx, mapped [-1,1] → [0,1]
    xi_raw_gl, wi_raw_gl = leggauss(N_QUAD_GL)
    xi_gl = 0.5 * (xi_raw_gl + 1.0)
    wi_gl = 0.5 * wi_raw_gl

    # CONST_H = ∫_0^1 s^α(1+s)^α ds via GJ (weight s^α, f=(1+s)^α smooth ✓)
    CONST_H = float(np.dot(wi_gj, (1.0 + xi_gj) ** alpha))
    return xi_gj, wi_gj, xi_gl, wi_gl, CONST_H


def _build_C_exact(t_grid, H):
    """
    Build the exact n×n RL-FBM covariance matrix.

    C[i,j] = t_i^{2H}/Γ(H+½)² · I(r)    (WLOG t_i ≤ t_j, r = t_j/t_i ≥ 1)

    where I(r) = ∫_0^1 u^α·(r-1+u)^α du,  α = H − ½.

    Strategy
    --------
    Diagonal: closed-form t_i^{2H}/(2H·Γ(H+½)²).

    Well-separated pairs (r−1 ≥ COV_THRESHOLD):
        Vectorised Gauss-Jacobi with weight u^α; remaining factor (r-1+u)^α is smooth.

    Near-diagonal pairs (r−1 < COV_THRESHOLD):
        Change of variables u = (r-1)·s throughout [0,1] → [0, M], M = 1/(r-1):

            I(r) = (r-1)^{2H} · ∫_0^M s^α(1+s)^α ds
                 = (r-1)^{2H} · [CONST_H  +  log-piece]

        CONST_H = ∫_0^1 s^α(1+s)^α ds  (precomputed once with GJ, smooth integrand ✓)
        log-piece = ∫_1^M s^α(1+s)^α ds
                  = ∫_0^{ln M} e^{(α+1)t}(1+e^t)^α dt   [sub s=e^t, smooth on [0,ln M] ✓]
                  computed with standard Gauss-Legendre.
    """
    n = len(t_grid)
    alpha = H - 0.5
    gH = math.gamma(H + 0.5)
    gH_sq = gH**2

    xi_gj, wi_gj, xi_gl, wi_gl, CONST_H = _quad_nodes(alpha)

    C = np.zeros((n, n), dtype=np.float64)
    np.fill_diagonal(C, _rl_diag(t_grid, H))

    for i in range(n):
        t_i = t_grid[i]
        scale = t_i ** (2.0 * H) / gH_sq

        # Index of first well-separated j: t_j / t_i − 1 ≥ COV_THRESHOLD
        j_sep = int(np.searchsorted(t_grid, t_i * (1.0 + COV_THRESHOLD), side="left"))

        # Near-diagonal band: j in [i+1, j_sep)
        for j in range(i + 1, min(j_sep, n)):
            r_m1 = t_grid[j] / t_i - 1.0  # r − 1, small
            M = 1.0 / r_m1
            ln_M = math.log(M)
            # log-piece: ∫_0^{ln_M} e^{(α+1)t}(1+e^t)^α dt via GL on [0, ln_M]
            t_nodes = ln_M * xi_gl
            log_intgrd = (alpha + 1.0) * t_nodes + alpha * np.log1p(np.exp(t_nodes))
            log_piece = ln_M * float(np.dot(wi_gl, np.exp(log_intgrd)))
            I_r = (r_m1 ** (2.0 * H)) * (CONST_H + log_piece)
            C[i, j] = scale * I_r
            C[j, i] = C[i, j]

        # Well-separated: vectorised GJ
        if j_sep < n:
            r_m1_vals = t_grid[j_sep:] / t_i - 1.0  # (n-j_sep,), all ≥ threshold
            f_xi = (r_m1_vals[:, None] + xi_gj[None, :]) ** alpha  # broadcast
            I_vals = f_xi @ wi_gj  # (n-j_sep,)
            C[i, j_sep:] = scale * I_vals
            C[j_sep:, i] = scale * I_vals

    return C


def _self_check_quadrature(t_grid, H):
    """
    Compare our quadrature against scipy adaptive quad on 5 representative pairs.
    Returns max relative error; aborts with a warning if it exceeds 1e-7.
    """
    from scipy.integrate import quad as scipy_quad

    alpha = H - 0.5
    gH = math.gamma(H + 0.5)
    gH_sq = gH**2
    n = len(t_grid)

    xi_gj, wi_gj, xi_gl, wi_gl, CONST_H = _quad_nodes(alpha)

    test_pairs = [
        (0, n - 1),  # r-1 = n-1 (trivial)
        (0, 1),  # r-1 = 1.0  (mild)
        (n // 10, n // 10 + 1),  # r-1 ≈ 0.01 (moderate)
        (n // 2, n // 2 + 1),  # r-1 ≈ dt/t (mid near-diag)
        (n - 2, n - 1),  # r-1 = dt/t_{n-2} (worst case)
    ]

    print(f"\n[Self-check] Off-diagonal quadrature, H={H:.2f}")
    print(f"  {'Pair':>12}  {'r-1':>14}  {'Ours':>18}  {'Adaptive':>18}  {'RelErr':>10}")
    max_rel_err = 0.0

    for i, j in test_pairs:
        t_i, t_j = t_grid[i], t_grid[j]
        r_m1 = t_j / t_i - 1.0
        scale = t_i ** (2.0 * H) / gH_sq

        # Reference: scipy adaptive quad on ∫_0^1 u^α (r-1+u)^α du
        ref_I, _ = scipy_quad(
            lambda u, a=alpha, r=r_m1: (u**a) * ((r + u) ** a),
            0.0,
            1.0,
            limit=500,
            epsabs=1e-13,
            epsrel=1e-13,
        )
        ref_val = scale * ref_I

        # Our method
        if r_m1 < COV_THRESHOLD:
            M = 1.0 / r_m1
            ln_M = math.log(M)
            t_nodes = ln_M * xi_gl
            log_intgrd = (alpha + 1.0) * t_nodes + alpha * np.log1p(np.exp(t_nodes))
            log_piece = ln_M * float(np.dot(wi_gl, np.exp(log_intgrd)))
            our_I = (r_m1 ** (2.0 * H)) * (CONST_H + log_piece)
        else:
            our_I = float(np.dot(wi_gj, (r_m1 + xi_gj) ** alpha))
        our_val = scale * our_I

        rel_err = abs(our_val - ref_val) / max(abs(ref_val), 1e-20)
        max_rel_err = max(max_rel_err, rel_err)
        print(
            f"  ({i:4d},{j:4d})  r-1={r_m1:14.4g}  {our_val:18.10e}  {ref_val:18.10e}  {rel_err:10.2e}"
        )

    print(
        f"  Max relative error: {max_rel_err:.2e}  (threshold {COV_THRESHOLD}, n_gj={N_QUAD_GJ}, n_gl={N_QUAD_GL})"
    )
    if max_rel_err > 1e-7:
        print(
            f"  WARNING: quadrature error {max_rel_err:.2e} exceeds 1e-7 — COV recombination may have a bug."
        )
    return max_rel_err


def _atm_call_price(I_v, I_V2, alpha_v, rho_v, rho_comp_v, K_val, K_t):
    """Per-path conditional-MC call price, shape (N,)."""
    is_call = K_val >= 0.0
    sign_v = 1.0 if is_call else -1.0
    F = alpha_v * rho_v * I_v
    Std = alpha_v * rho_comp_v * torch.sqrt(I_V2.clamp(min=1e-14))
    d = (F - K_t) / Std
    ds = d if is_call else -d
    pdf = INV_SQRT_2PI * torch.exp(-0.5 * d * d)
    cdf = 0.5 * (1.0 + torch.erf(ds * INV_SQRT_2))
    p = sign_v * (F - K_t) * cdf + Std * pdf
    return p if is_call else p - K_t  # put-call parity → call price


def _arm_unit_vols_and_EV(Z, comp_vec, nu_val):
    """Left-point predictable vol + empirical E[V] for one arm.

    comp_vec is the FULL ½ν²·Var(Z̃) compensator vector (n_steps,); identical construction for
    every arm (Euler/hybrid/exact differ only through (Z, comp_vec)). E[V] is read from V_full
    before the left-point shift, so it tests martingality on the pricing path itself.
    """
    V_full = torch.exp(nu_val * Z - comp_vec.unsqueeze(0))  # (N, n_steps)
    EV = V_full.mean(dim=0)
    EV_se = V_full.std(dim=0) / math.sqrt(Z.shape[0])
    unit_vols = torch.cat(
        [
            torch.ones(Z.shape[0], 1, dtype=Z.dtype, device=Z.device),
            V_full[:, :-1],
        ],
        dim=1,
    )
    return unit_vols, EV, EV_se


def run_bias_diagnostic(
    H, params_df, K_vals, rng_seed=42, n_paths=N_PATHS, expiry_subset=None, dt=DT_CALIB
):
    """
    Four-arm kernel-discretization bias diagnostic for a single H.

    n_paths overrides the path count for this run (default module N_PATHS, which the gate and the
    full-assert arms use). dt selects the
    simulation grid (default DT_CALIB=1/50; the decision pass also calls it at DT_PRICE=1/24).
    expiry_subset (iterable of expiries) restricts both the priced expiries and n_steps to that
    subset; the production-κ decision pass uses it to cap n_steps (hence memory) while raising
    n_paths.

    Arms (all sharing the same ΔZ): Euler (κ=0), hybrid(κ=1), hybrid(κ=2), and the exact
    κ=∞ reference (Euler + eps@L_residᵀ). The hybrid arms use build_hybrid_rough_driver with
    their own construction compensator rough_vol_compensator(ν, kernel, var_vec, dt, κ), not
    diag(C_exact), so each is a martingale-consistent price and Δ measures that scheme's
    price-level discretization bias vs the exact scheme.

    Returns (df_bias, df_ev):
        df_bias : long format with H, T, K, arm, IV_bps, IV_exact_bps, delta_bps, SE_paired_bps
                  (arm ∈ {euler, hybrid_k1, hybrid_k2}; exact carried as IV_exact_bps).
        df_ev   : per arm (incl. exact): H, arm, max_abs_EV_dev, worst_z.
    """
    print(f"\n{'─' * 70}")
    print(f"  H = {H:.3f}   (dt = {dt:.5f})")
    print(f"{'─' * 70}")

    df_H = params_df[np.abs(params_df["H"] - H) < 1e-6].copy()
    if df_H.empty:
        raise ValueError(f"No Stage-1 parameters found for H={H}")

    if expiry_subset is not None:
        keep = np.array([np.any(np.abs(df_H["Expiry"].values - e) < 1e-6) for e in expiry_subset])
        sub = [float(e) for e, k in zip(expiry_subset, keep) if k]
        df_H = df_H[df_H["Expiry"].isin(sub)].copy()
        if df_H.empty:
            raise ValueError(f"expiry_subset {expiry_subset} not present for H={H}")

    expiries = df_H["Expiry"].values.astype(float)
    alpha_arr = df_H["Alpha"].values.astype(float)
    rho_arr = df_H["Rho"].values.astype(float)
    nu_val = float(df_H["Nu"].iloc[0])

    T_max = float(expiries.max())
    n_steps = int(math.ceil(T_max / dt))
    t_grid = np.arange(1, n_steps + 1, dtype=np.float64) * dt  # t_k = (k+1)·Δt

    # Production step indices (0-indexed), matching pricers.py
    step_e = np.array([int(math.ceil(T / dt)) - 1 for T in expiries])

    print(f"  n_steps={n_steps}, n_paths={n_paths}, dt={dt:.4f}, nu={nu_val:.5f}")

    t0 = time.time()
    print("  Building C_exact ...", end="", flush=True)
    C_exact = _build_C_exact(t_grid, H)
    print(f" {time.time() - t0:.1f}s")

    K_torch = build_fractional_kernel(H, n_steps, dt, DEVICE, DTYPE)
    K_np = K_torch.cpu().numpy()
    C_euler = dt * (K_np @ K_np.T)  # matches pricers.py at the matrix level

    # residual covariance and its Cholesky factor, used by the exact arm only
    C_resid = C_exact - C_euler
    C_resid = 0.5 * (C_resid + C_resid.T)  # symmetrize

    frob = np.linalg.norm(C_resid, "fro")
    print(f"  ‖C_resid‖_F = {frob:.4e}   (expect ≈0 at H=0.5)")

    if frob < 1e-10 * n_steps:
        # C_resid is numerically zero (H=0.5 case): no correction
        L_resid_np = np.zeros((n_steps, n_steps), dtype=np.float64)
        print("  C_resid ≈ 0 — skipping Cholesky (exact arm = Euler arm)")
    else:
        # Regularise if needed, then Cholesky
        min_eig = np.linalg.eigvalsh(C_resid).min()
        if min_eig < 0.0:
            nugget = max(-min_eig * 1.01, 1e-14)
            print(f"  Adding nugget {nugget:.2e} to C_resid (λ_min={min_eig:.2e})")
            C_resid += nugget * np.eye(n_steps)
        t0 = time.time()
        print("  Cholesky of C_resid ...", end="", flush=True)
        L_resid_np = np.linalg.cholesky(C_resid)
        print(f" {time.time() - t0:.1f}s")

    L_resid = torch.tensor(L_resid_np, dtype=DTYPE, device=DEVICE)

    g_dz = torch.Generator(device=DEVICE)
    g_dz.manual_seed(rng_seed)
    g_eps = torch.Generator(device=DEVICE)
    g_eps.manual_seed(rng_seed + 1000)

    print("  Drawing PRNG paths ...", end="", flush=True)
    t0 = time.time()
    dZ = torch.randn(n_paths, n_steps, generator=g_dz, device=DEVICE, dtype=DTYPE) * math.sqrt(dt)
    dt_t = torch.tensor(dt, dtype=DTYPE, device=DEVICE)

    # build the four arms (each → unit_vols + E[V]); free Z tensors promptly
    unit_vols = {}
    ev_rows = []

    def _add_arm(name, Z, comp_vec):
        uv, EV, EV_se = _arm_unit_vols_and_EV(Z, comp_vec, nu_val)
        unit_vols[name] = uv
        dev = (EV - 1.0).abs()
        zsc = dev / EV_se.clamp(min=1e-30)
        ev_rows.append(
            {
                "H": H,
                "arm": name,
                "max_abs_EV_dev": float(dev.max().item()),
                "worst_z": float(zsc.max().item()),
            }
        )

    # Euler arm (κ=0): compensator ½·ν²·Δt·Σ_j K²_{ij}, as in pricers.py.
    Z_euler = dZ @ K_torch.T
    comp_euler = (0.5 * nu_val**2 * dt) * (K_torch**2).sum(dim=1)
    # Object-identity guard: the κ=0 hybrid driver must consume the same ΔZ ⇒ bit-identical Z̃.
    fbm_k0, _, _ = build_hybrid_rough_driver(dZ, H, n_steps, dt, 0, DEVICE, DTYPE)
    assert torch.equal(fbm_k0, Z_euler), "hybrid arm not sharing the Euler ΔZ"
    del fbm_k0
    _add_arm("euler", Z_euler, comp_euler)
    del Z_euler

    # Hybrid arms (κ ∈ {1,2}): same ΔZ, near-ε at rng_seed+7919 (independent of ΔZ and eps).
    for kappa in HYBRID_KAPPAS:
        g_near = torch.Generator(device=DEVICE).manual_seed(rng_seed + NEAR_SEED_OFF)
        fbm, var_vec, kernel = build_hybrid_rough_driver(
            dZ, H, n_steps, dt, kappa, DEVICE, DTYPE, near_generator=g_near
        )
        comp_h = rough_vol_compensator(nu_val, kernel, var_vec, dt_t, kappa).reshape(n_steps)
        _add_arm(f"hybrid_k{kappa}", fbm, comp_h)
        del fbm

    # Exact arm (κ=∞): Euler + independent residual correction; compensator ½·ν²·diag(C_exact).
    eps = torch.randn(n_paths, n_steps, generator=g_eps, device=DEVICE, dtype=DTYPE)
    Z_exact = (dZ @ K_torch.T) + eps @ L_resid.T
    comp_exact = torch.tensor(0.5 * nu_val**2 * _rl_diag(t_grid, H), dtype=DTYPE, device=DEVICE)
    _add_arm("exact", Z_exact, comp_exact)
    del Z_exact, eps
    print(f" {time.time() - t0:.1f}s")

    # CMC payoffs, one block shared by all arms
    all_arms = ARMS_NONEXACT + ["exact"]
    rows = []
    print("  Pricing ...", end="", flush=True)
    t0 = time.time()
    for T_exp, s_e, alpha_v, rho_v in zip(expiries, step_e, alpha_arr, rho_arr):
        rho_comp_v = math.sqrt(max(1.0 - rho_v**2, 1e-12))
        dZ_t = dZ[:, : s_e + 1]

        # Cumulative integrals up to step s_e only (avoids full-grid cumsum tensor)
        I_v = {}
        I_V2 = {}
        for name in all_arms:
            uv = unit_vols[name][:, : s_e + 1]
            I_v[name] = (uv * dZ_t).sum(dim=1)
            I_V2[name] = (uv**2).sum(dim=1) * dt

        T_t = torch.tensor(T_exp, dtype=DTYPE, device=DEVICE)
        alpha_t = torch.tensor(alpha_v, dtype=DTYPE, device=DEVICE)
        vega_approx = math.sqrt(T_exp) * INV_SQRT_2PI

        for K_val in K_vals:
            K_t = torch.tensor(K_val, dtype=DTYPE, device=DEVICE)
            pay = {
                name: _atm_call_price(I_v[name], I_V2[name], alpha_v, rho_v, rho_comp_v, K_val, K_t)
                for name in all_arms
            }
            iv = {
                name: float(
                    bachelier_iv_newton(
                        pay[name].mean().unsqueeze(0),
                        K_t.unsqueeze(0),
                        T_t.unsqueeze(0),
                        alpha_t.unsqueeze(0),
                    ).item()
                )
                * 1e4
                for name in all_arms
            }
            for name in ARMS_NONEXACT:
                delta_P = pay[name] - pay["exact"]  # paired difference per path
                se_paired = float(delta_P.std().item()) / math.sqrt(n_paths) / vega_approx * 1e4
                valid = bool(
                    np.isfinite(iv[name])
                    and np.isfinite(iv["exact"])
                    and IV_FLOOR_BPS < iv[name] < IV_CAP_BPS
                    and IV_FLOOR_BPS < iv["exact"] < IV_CAP_BPS
                )
                rows.append(
                    {
                        "H": H,
                        "T": T_exp,
                        "K": K_val,
                        "arm": name,
                        "IV_bps": iv[name],
                        "IV_exact_bps": iv["exact"],
                        "delta_bps": iv[name] - iv["exact"],
                        "SE_paired_bps": se_paired,
                        "valid": valid,
                    }
                )

    print(f" {time.time() - t0:.1f}s")
    return pd.DataFrame(rows), pd.DataFrame(ev_rows)


def _min_time(fn, n_warmup=2, n_repeat=5):
    # min over repeats, the standard robust microbenchmark estimator: it picks the least-contended
    # sample, filtering transient OS load / CPU throttling that a median still admits.
    for _ in range(n_warmup):
        fn()
    ts = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(min(ts))


def _time_build_components(H, params_df, rng_seed=42):
    """Decompose the hybrid driver-build wall-clock to detect an accidental second N·n² matmul.

    Run on the calibration grid (DT_CALIB). The build is one matmul (dz @ K_tailᵀ) + the
    kernel build + the near draw (randn, ΔZ·B, ε@L_condᵀ, banded shift-adds). A second dense matmul
    (the eps@L_residᵀ failure mode) would add ≈ t_matmul, leaving an unexplained matmul-sized
    residual in t_build − (t_matmul + t_kernel + t_near). The check is κ-robust (t_near scales with
    κ) and scale-robust (residual measured against the actual matmul), unlike a fixed build/κ0 ratio.

    Note: the near-noise randn is wall-clock-comparable to the matmul (both touch ≈N·n elements;
    randn's per-element cost ≈ BLAS matmul's per-N·n² cost here), so the build ratio is
    ~1.6–2.6× κ=0 (≈2× the near-noise at κ=2) with no second matmul. This is
    amortized: fbm/var_vec are parameter-independent and built once per H, then cached.
    """
    # float32 near-noise would cut randn ≈4×; keep it out of the Sobol bridge (grows its dimension)
    df_H = params_df[np.abs(params_df["H"] - H) < 1e-6]
    n_steps = int(math.ceil(float(df_H["Expiry"].max()) / DT_CALIB))
    g_dz = torch.Generator(device=DEVICE).manual_seed(rng_seed)
    dZ = torch.randn(N_PATHS, n_steps, generator=g_dz, device=DEVICE, dtype=DTYPE) * math.sqrt(
        DT_CALIB
    )

    # κ-independent components: kernel build, and the single matmul on a dense (K_tail) operand.
    steps = torch.arange(n_steps, device=DEVICE)
    lag_m1 = steps[:, None] - steps[None, :]
    K = build_fractional_kernel(H, n_steps, DT_CALIB, DEVICE, DTYPE)
    K_tail = K.clone()
    K_tail[(lag_m1 >= 0) & (lag_m1 < 1)] = 0.0
    t_kernel = _min_time(lambda: build_fractional_kernel(H, n_steps, DT_CALIB, DEVICE, DTYPE))
    t_matmul = _min_time(lambda: dZ @ K_tail.T)
    t_build0 = _min_time(
        lambda: build_hybrid_rough_driver(dZ, H, n_steps, DT_CALIB, 0, DEVICE, DTYPE)
    )

    def _near_only(kappa):
        # Mirrors build_hybrid_rough_driver's near block exactly (no matmul on K_tail here).
        gamma = torch.exp(torch.lgamma(torch.tensor(H + 0.5, device=DEVICE, dtype=DTYPE)))
        inv_g = 1.0 / gamma
        cell = build_hybrid_cell_covariance(H, DT_CALIB, kappa, device=DEVICE, dtype=DTYPE)
        B = cell["B"]
        evals, evecs = torch.linalg.eigh(cell["Sigma_cond"])
        L_cond = evecs @ torch.diag(torch.sqrt(torch.clamp(evals, min=0.0)))
        eps = torch.randn(N_PATHS, n_steps, kappa, device=DEVICE, dtype=DTYPE)
        W_near = dZ.unsqueeze(-1) * B + eps @ L_cond.T
        near_total = torch.zeros(N_PATHS, n_steps, device=DEVICE, dtype=DTYPE)
        for k in range(kappa):
            near_total[:, k:] = near_total[:, k:] + W_near[:, : n_steps - k, k]
        return inv_g * near_total

    out = {
        "n_steps": n_steps,
        "t_kernel": t_kernel,
        "t_matmul": t_matmul,
        "t_build0": t_build0,
        "per_kappa": {},
    }
    for kappa in HYBRID_KAPPAS:
        t_near = _min_time(lambda kp=kappa: _near_only(kp))
        t_build = _min_time(
            lambda kp=kappa: build_hybrid_rough_driver(dZ, H, n_steps, DT_CALIB, kp, DEVICE, DTYPE)
        )
        residual = t_build - (t_matmul + t_kernel + t_near)
        out["per_kappa"][kappa] = {
            "t_near": t_near,
            "t_build": t_build,
            "residual": residual,
            "ratio": t_build / t_build0,
        }
    return out


def _arm_stats(df_bias, atm_only=False):
    """Per-arm {mean|Δ|, max|Δ|, mean SE} over valid cells, plus the overall mean paired SE.

    atm_only=True restricts to the near-ATM band |K| ≤ ATM_BAND_K (where IV inversion is
    reliable), which the success-criteria asserts use; reporting uses the full smile.
    """
    valid = df_bias[df_bias["valid"]]
    if atm_only:
        valid = valid[valid["K"].abs() <= ATM_BAND_K + 1e-9]
    stats = {}
    for arm in ARMS_NONEXACT:
        sub = valid[valid["arm"] == arm]
        ad = sub["delta_bps"].abs()
        stats[arm] = {
            "mean_abs": float(ad.mean()),
            "max_abs": float(ad.max()),
            "mean_se": float(sub["SE_paired_bps"].mean()),
            "n_valid": int(len(sub)),
        }
    mean_se = float(valid["SE_paired_bps"].mean())
    return stats, mean_se


def _print_table(df_bias, df_ev=None, stage1_rmse=None):
    """Per-arm, per-expiry summary over valid cells, the κ-chain (bps), and the E[V] summary.

    Full per-(T,K) cells (with the `valid` flag for the failed-inversion wings) are in the CSV;
    here we print compact per-expiry mean|Δ|/max|Δ| to keep three arms readable.
    """
    valid = df_bias[df_bias["valid"]]
    n_total = len(df_bias[df_bias["arm"] == ARMS_NONEXACT[0]])
    n_drop = n_total - len(valid[valid["arm"] == ARMS_NONEXACT[0]])
    for arm in ARMS_NONEXACT:
        sub = valid[valid["arm"] == arm]
        if sub.empty:
            print(f"\n  arm = {arm}: no valid cells")
            continue
        print(f"\n  arm = {arm}   ({'T':>6} : mean|Δ|  max|Δ|  meanSE  [bps])")
        for T_exp, g in sub.groupby("T"):
            ad = g["delta_bps"].abs()
            print(
                f"    {T_exp:6.2f} : {ad.mean():8.4f} {ad.max():8.4f} "
                f"{g['SE_paired_bps'].mean():8.4f}"
            )
        ad = sub["delta_bps"].abs()
        print(
            f"    overall : mean|Δ|={ad.mean():.4f}  max|Δ|={ad.max():.4f}  "
            f"meanSE={sub['SE_paired_bps'].mean():.4f}"
        )

    stats, mean_se = _arm_stats(df_bias)
    print(
        f"\n  Δ-chain full smile (mean|Δ| vs exact):  "
        f"κ0={stats['euler']['mean_abs']:.3f} → "
        f"κ1={stats['hybrid_k1']['mean_abs']:.3f} → "
        f"κ2={stats['hybrid_k2']['mean_abs']:.3f} bps   (paired SE ≈ {mean_se:.3f} bps)"
    )
    atm, atm_se = _arm_stats(df_bias, atm_only=True)
    print(
        f"  Δ-chain ATM |K|≤{ATM_BAND_K} (mean|Δ|):  "
        f"κ0={atm['euler']['mean_abs']:.3f} → "
        f"κ1={atm['hybrid_k1']['mean_abs']:.3f} → "
        f"κ2={atm['hybrid_k2']['mean_abs']:.3f} bps   (paired SE ≈ {atm_se:.3f} bps)"
    )
    print(
        f"  Valid cells: {stats['euler']['n_valid']}/{n_total} "
        f"({n_drop} dropped — failed IV inversion at deep-OTM × short-T wings)"
    )
    if stage1_rmse is not None:
        flag = "NEGLIGIBLE" if stats["hybrid_k2"]["max_abs"] < mean_se else "RESIDUAL"
        print(f"  Stage-1 RMSE at this H ≈ {stage1_rmse:.2f} bps  →  hybrid κ=2 bias is {flag}")
    if df_ev is not None:
        print(
            "  E[V] worst |E[V]−1| (z-score):  "
            + "  ".join(
                f"{r.arm}={r.max_abs_EV_dev:.1e}({r.worst_z:.1f}σ)" for _, r in df_ev.iterrows()
            )
        )


def _assert_ev(df_ev, H):
    worst = df_ev["worst_z"].max()
    assert worst < EV_Z_TOL, (
        f"H={H}: E[V] martingality failed — worst z={worst:.2f} > {EV_Z_TOL} "
        f"(arm {df_ev.loc[df_ev['worst_z'].idxmax(), 'arm']})"
    )


def _run_full_assert_arm(H, params_df, K_vals, rng_seed=42):
    """Full production-style battery for one H (run on DT_CALIB). Returns (ATM Euler mean|Δ|, ATM
    stats dict, bias df); the df and stats let the additive dt-sweep verify its 1/50 cell reproduces
    this anchor bit-identically.

    Both H=0.10 (production) and H=0.05 (κ1≪κ0 anchor) go through here: the κ1≪κ0
    necessity claim is sharpest at small H, so 0.05 must carry the same asserts as the production
    point, not a lighter set.
    """
    print(f"\n{'=' * 70}")
    print(f"{('H = %.2f  FULL-ASSERT RUN' % H):^70}")
    print(f"{'=' * 70}")

    rmse = _stage1_rmse(params_df, H)
    df, ev = run_bias_diagnostic(H, params_df, K_vals, rng_seed=rng_seed)
    _print_table(df, ev, stage1_rmse=rmse)
    df.to_csv(os.path.join(RESULTS_DIR, f"kernel_bias_H{H:.2f}.csv"), index=False)

    atm, atm_se = _arm_stats(df, atm_only=True)
    # Euler bias material; κ=1 ≪ κ=0; κ=2 within a few paired SE of exact; soft-monotone.
    # Asserted on the near-ATM band (reliable inversion). The multipliers are fixed thresholds;
    # a trip means the threshold needs retuning, not loosening.
    assert atm["euler"]["mean_abs"] > 10.0 * atm_se, (
        f"H={H}: ATM Euler bias {atm['euler']['mean_abs']:.3f} not material vs SE {atm_se:.3f}"
    )
    assert atm["hybrid_k1"]["mean_abs"] < 0.5 * atm["euler"]["mean_abs"], (
        f"H={H}: ATM hybrid κ=1 mean|Δ|={atm['hybrid_k1']['mean_abs']:.3f} not ≪ "
        f"Euler {atm['euler']['mean_abs']:.3f}"
    )
    assert atm["hybrid_k2"]["mean_abs"] < KAPPA2_SE_MULT * atm["hybrid_k2"]["mean_se"], (
        f"H={H}: ATM hybrid κ=2 mean|Δ|={atm['hybrid_k2']['mean_abs']:.3f} not within "
        f"{KAPPA2_SE_MULT}×SE ({atm['hybrid_k2']['mean_se']:.3f}) of exact"
    )
    assert atm["hybrid_k1"]["mean_abs"] <= atm["euler"]["mean_abs"], (
        f"H={H}: κ-chain not monotone (κ0≥κ1)"
    )
    assert atm["hybrid_k2"]["mean_abs"] <= atm["hybrid_k1"]["mean_abs"] + 2.0 * atm_se, (
        f"H={H}: κ-chain not monotone (κ1≥κ2 within SE)"
    )
    _assert_ev(ev, H)
    return atm["euler"]["mean_abs"], atm, df


def _decision_n(n_steps):
    """Largest power-of-two N that keeps the per-pass working set within the memory budget.

    Working set ≈ DECISION_TENSOR_MULT live (N×n) float64 tensors. Clamped to
    [DECISION_N_FLOOR, DECISION_N_CEIL]; short expiries (small n) thus get a far larger N → a
    tighter paired SE exactly in the production-relevant range. On the coarser dt=1/24 grid more
    expiries hit the ceiling (smaller n → larger budget-N → clamp); peak RAM stays ≤ budget.
    """
    budget_bytes = DECISION_MEM_BUDGET_GB * (1024**3)
    n_raw = budget_bytes / (DECISION_TENSOR_MULT * n_steps * 8)
    n_pow2 = 1 << int(math.floor(math.log2(max(n_raw, 1.0))))
    return int(min(max(n_pow2, DECISION_N_FLOOR), DECISION_N_CEIL))


def _print_kappa_readoff(per_expiry, rmse, H, grid_label):
    """Emit the absolute-bps κ-ladder per decision expiry and the production-κ read-off.

    per_expiry : list of (T, n_steps, n_paths, stats_atm, se_atm) where stats_atm is
    _arm_stats(atm_only=True)[0] for that single-expiry run. Production κ = smallest κ whose ATM
    max|Δ| ≤ DECISION_NEGLIGIBLE_BPS at every decision expiry; the absolute floor alone is the
    materiality gate. ≲SE (mean|Δ| ≤ pairedSE) is printed as a diagnostic but does not gate: SE is
    path-count-dependent (→0 as N grows), so a tight SE at large N would reject an immaterial residual
    regardless of materiality. The se_limited guard reports when the SE has shrunk below the κ=1
    residual, which is why SE is unfit to gate; it does not contradict the verdict.
    """
    print(f"\n{'=' * 70}")
    print(
        f"{('PRODUCTION-κ DECISION READ-OFF  (H=%.2f, dt=%s, ATM |K|≤%.3f)' % (H, grid_label, ATM_BAND_K)):^70}"
    )
    print(f"{'=' * 70}")
    print("  Driver: production build_hybrid_rough_driver, plain-PRNG CRN ΔZ, antithetic=False")
    print(
        f"  Stage-1 calibration RMSE ≈ {rmse:.2f} bps; negligible-bps floor = {DECISION_NEGLIGIBLE_BPS:.2f} bps"
    )
    print(
        f"\n  {'T(Y)':>5} {'n':>5} {'N':>9}  κ  {'mean|Δ|':>9} {'max|Δ|':>9} {'pairedSE':>9} "
        f"{'|Δ|/SE':>7}  ≲SE  ≲{DECISION_NEGLIGIBLE_BPS:g}bps"
    )

    qualifies = {0: True, 1: True, 2: True}  # κ qualifies via the absolute floor, at every expiry
    se_limited = False
    for T, n_steps, n_paths, stats, se in per_expiry:
        for kappa in (0, 1, 2):
            st = stats[KAPPA_ARM[kappa]]
            within_se = st["mean_abs"] <= se  # diagnostic only (printed); does not gate
            within_abs = st["max_abs"] <= DECISION_NEGLIGIBLE_BPS
            qualifies[kappa] = qualifies[kappa] and within_abs
            ratio = st["mean_abs"] / se if se > 0 else float("inf")
            print(
                f"  {T:5.1f} {n_steps:5d} {n_paths:9d}  {kappa}  {st['mean_abs']:9.4f} "
                f"{st['max_abs']:9.4f} {se:9.4f} {ratio:7.2f}  "
                f"{'yes' if within_se else ' no':>3}  {'yes' if within_abs else ' no':>3}"
            )
        # SE-limited guard: if κ=1's residual is itself below the SE, the SE cannot resolve κ=2.
        if stats["hybrid_k1"]["mean_abs"] <= se:
            se_limited = True
        print()

    prod_kappa = next((k for k in (0, 1, 2) if qualifies[k]), None)
    if prod_kappa is None:
        print(
            "  → no κ∈{0,1,2} clears both tests at every expiry — κ=2 still shows resolvable bias."
        )
    else:
        print(
            f"  → smallest κ with max|Δ| ≤ {DECISION_NEGLIGIBLE_BPS:g} bps at every expiry:  "
            f"PRODUCTION κ = {prod_kappa}"
        )
    if se_limited:
        print(
            "  [guard] at ≥1 expiry the κ=1 residual already sits within paired SE — i.e. the SE has "
            "shrunk below the residual at this N, which is precisely why SE does not gate the verdict; "
            "materiality is judged by the absolute floor, not SE."
        )


def run_kappa_decision(params_df, K_vals, rmse, H, dt):
    """Additive high-N pass at H over the production-relevant ATM subset, on the given grid dt.

    Called once per grid: dt=1/50 (matches calibration) and dt=1/24 (the production pricing grid
    that gates Bermudan κ). Bias rises on the coarser 1/24 grid (the near-lag truncation is a larger
    fraction of total variance), so this pass answers whether κ=1 still clears the negligible-bps
    floor at 1/24 or whether the production grid needs κ=2.
    """
    grid_label = _grid_label(dt)
    print(f"\n{'=' * 70}")
    print(f"{('κ-DECISION PASS — per-expiry budget-sized N, dt=%s (additive)' % grid_label):^70}")
    print(f"{'=' * 70}")
    per_expiry = []
    for T in DECISION_EXPIRIES:
        n_steps = int(math.ceil(T / dt))
        N = _decision_n(n_steps)
        ws_gb = DECISION_TENSOR_MULT * N * n_steps * 8 / (1024**3)
        print(
            f"\n  ── decision expiry T={T:.1f}Y: n_steps={n_steps}, N={N} (≈{ws_gb:.1f} GB peak) ──"
        )
        df_T, _ = run_bias_diagnostic(
            H, params_df, K_vals, rng_seed=42, n_paths=N, expiry_subset=[T], dt=dt
        )
        df_T.to_csv(
            os.path.join(RESULTS_DIR, f"kernel_kappa_decision_dt{round(1.0 / dt)}_T{T:.0f}.csv"),
            index=False,
        )
        stats, se = _arm_stats(df_T, atm_only=True)
        per_expiry.append((T, n_steps, N, stats, se))
    _print_kappa_readoff(per_expiry, rmse, H, grid_label)


def run_dt_sweep(params_df, K_vals, H, anchor_df, anchor_atm):
    """Additive κ-vs-dt sweep at H over the full expiry grid at fixed N=N_PATHS (the anchor's N/seed).

    Same seed and N as the H=0.10 full-assert anchor, so the dt=DT_CALIB cell reproduces it
    bit-identically, the regression guard for the dt path. dt=1/100 is an explicit fine reference
    (DT_SWEEP_FINE). Writes results/kernel_dt_sweep_H0.10.csv.
    """
    grids = (DT_PRICE, DT_CALIB, DT_SWEEP_FINE)
    cols = ["IV_bps", "IV_exact_bps", "delta_bps", "SE_paired_bps"]
    T_max = float(params_df[np.abs(params_df["H"] - H) < 1e-6]["Expiry"].max())

    print(f"\n{'=' * 70}")
    print(f"{('dt-SWEEP — full grid, N=%d, H=%.2f (additive)' % (N_PATHS, H)):^70}")
    print(f"{'=' * 70}")

    rows = []
    for dt in grids:
        n_steps = int(math.ceil(T_max / dt))
        ws_gb = DECISION_TENSOR_MULT * N_PATHS * n_steps * 8 / (1024**3)
        print(
            f"\n  ── dt={_grid_label(dt)}: n_steps={n_steps}, N={N_PATHS} (≈{ws_gb:.2f} GB peak) ──"
        )
        if abs(dt - DT_SWEEP_FINE) < 1e-12:  # confirm C_exact quadrature accuracy at the fine grid
            _self_check_quadrature(np.arange(1, n_steps + 1, dtype=np.float64) * dt, H)
        df, _ = run_bias_diagnostic(H, params_df, K_vals, rng_seed=42, n_paths=N_PATHS, dt=dt)
        atm, _ = _arm_stats(df, atm_only=True)

        if abs(dt - DT_CALIB) < 1e-12 and anchor_df is not None:
            same = np.allclose(
                df[cols].to_numpy(), anchor_df[cols].to_numpy(), rtol=0.0, atol=1e-10
            )
            tag = "BIT-IDENTICAL" if same else "*** MISMATCH ***"
            print(
                f"    [consistency] 1/50 vs H=0.10 anchor  κ0={atm['euler']['mean_abs']:.4f}/"
                f"{anchor_atm['euler']['mean_abs']:.4f}  κ1={atm['hybrid_k1']['mean_abs']:.4f}/"
                f"{anchor_atm['hybrid_k1']['mean_abs']:.4f}  κ2={atm['hybrid_k2']['mean_abs']:.4f}/"
                f"{anchor_atm['hybrid_k2']['mean_abs']:.4f}  → {tag}"
            )
            if not same:
                print(
                    "\n  *** dt-SWEEP REGRESSION GUARD FAILED: the 1/50 cell does not reproduce the anchor ***"
                )
                sys.exit(1)

        for kappa in (0, 1, 2):
            st = atm[KAPPA_ARM[kappa]]
            print(
                f"    κ={kappa}: ATM mean|Δ|={st['mean_abs']:.4f}  pairedSE={st['mean_se']:.4f} bps"
            )
            rows.append(
                {
                    "dt": dt,
                    "n_steps": n_steps,
                    "kappa": kappa,
                    "atm_mean_abs_bps": st["mean_abs"],
                    "atm_paired_se_bps": st["mean_se"],
                }
            )

    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "kernel_dt_sweep_H0.10.csv"), index=False)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(STAGE1_CSV):
        raise FileNotFoundError(f"Stage-1 CSV not found: {STAGE1_CSV}")
    params_df = pd.read_csv(STAGE1_CSV)

    vol_surface = load_swaption_vol_surface(
        os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV),
        config.STAGE1_TARGET_TENOR,
    )
    K_vals = vol_surface.columns.values.astype(float)

    print("\n" + "=" * 70)
    print(f"{'KERNEL DISCRETISATION BIAS DIAGNOSTIC — Euler vs hybrid(κ) vs exact':^70}")
    print("=" * 70)
    print(
        f"  DT_CALIB={DT_CALIB}, DT_PRICE={DT_PRICE:.5f}, N_PATHS={N_PATHS}, "
        f"COV_THRESHOLD={COV_THRESHOLD}, κ={list(HYBRID_KAPPAS)}"
    )
    print(f"  GJ nodes={N_QUAD_GJ}, GL nodes={N_QUAD_GL}")
    print(f"  Strikes: {K_vals}")

    # quadrature self-check at the most-singular full-assert H first
    # H=0.05 (α=−0.45, smallest near-diagonal r−1) is the conservative covering case: it bounds the
    # milder H=0.10 production arm and the 1/24 decision pass a fortiori. H=0.10 also checked (cheap).
    for H_chk in (0.05, H_PRODUCTION):
        T_max_chk = float(params_df[np.abs(params_df["H"] - H_chk) < 1e-6]["Expiry"].max())
        n_steps_chk = int(math.ceil(T_max_chk / DT_CALIB))
        t_grid_chk = np.arange(1, n_steps_chk + 1, dtype=np.float64) * DT_CALIB
        _self_check_quadrature(t_grid_chk, H_chk)

    euler_bias_by_H = {}  # for the cross-H shrink assertion

    print(f"\n{'─' * 70}")
    print("  H=0.5 sanity gate  (α=0 → all arms coincide; Δ ≈ 0 within paired SE)")
    print(f"{'─' * 70}")

    df_gate, ev_gate = run_bias_diagnostic(H_SANITY, params_df, K_vals, rng_seed=0)
    _print_table(df_gate, ev_gate)
    df_gate.to_csv(os.path.join(RESULTS_DIR, f"kernel_bias_H{H_SANITY:.2f}.csv"), index=False)

    stats_gate, se_gate = _arm_stats(df_gate)
    for arm in ARMS_NONEXACT:
        if stats_gate[arm]["max_abs"] > max(5.0 * se_gate, ABS_TOL_BPS):
            print(
                f"\n  *** H=0.5 GATE FAILED ({arm}) ***\n"
                f"      max|Δ|={stats_gate[arm]['max_abs']:.4f} bps > "
                f"max(5×SE={5 * se_gate:.4f}, {ABS_TOL_BPS}) bps — a compensator/covariance entry is wrong."
            )
            sys.exit(1)
    _assert_ev(ev_gate, H_SANITY)
    euler_bias_by_H[H_SANITY] = _arm_stats(df_gate, atm_only=True)[0]["euler"]["mean_abs"]
    print("  Gate PASSED (all arms ≈ exact, E[V] ≈ 1).")

    anchor_atm_010 = anchor_df_010 = None
    for H in H_FULL_ASSERT:
        eb, atm_H, df_H = _run_full_assert_arm(H, params_df, K_vals, rng_seed=42)
        euler_bias_by_H[H] = eb
        if abs(H - H_PRODUCTION) < 1e-9:
            anchor_atm_010, anchor_df_010 = atm_H, df_H

    # The 1/24 read-off gates Bermudan pricing; the 1/50 read-off matches calibration.
    rmse_prod = _stage1_rmse(params_df, H_PRODUCTION)
    for dt in (DT_CALIB, DT_PRICE):
        run_kappa_decision(params_df, K_vals, rmse_prod, H_PRODUCTION, dt)

    run_dt_sweep(params_df, K_vals, H_PRODUCTION, anchor_df_010, anchor_atm_010)

    # optional H values: bias must shrink toward H=0.5
    for H_opt in H_COMPARISON:
        if params_df[np.abs(params_df["H"] - H_opt) < 1e-6].empty:
            print(f"\n  No parameters for H={H_opt:.2f}, skipping.")
            continue
        print(f"\n{'─' * 70}")
        print(f"  H = {H_opt:.2f}  (BLP-consistency — bias shrinks toward H=0.5)")
        print(f"{'─' * 70}")
        rmse_opt = _stage1_rmse(params_df, H_opt)
        df_opt, ev_opt = run_bias_diagnostic(H_opt, params_df, K_vals, rng_seed=42)
        _print_table(df_opt, ev_opt, stage1_rmse=rmse_opt)
        df_opt.to_csv(os.path.join(RESULTS_DIR, f"kernel_bias_H{H_opt:.2f}.csv"), index=False)

        atm_opt, atm_se_opt = _arm_stats(df_opt, atm_only=True)
        assert (
            atm_opt["hybrid_k2"]["mean_abs"] < KAPPA2_SE_MULT * atm_opt["hybrid_k2"]["mean_se"]
        ), f"H={H_opt}: ATM hybrid κ=2 mean|Δ| not within {KAPPA2_SE_MULT}×SE of exact"
        _assert_ev(ev_opt, H_opt)
        euler_bias_by_H[H_opt] = atm_opt["euler"]["mean_abs"]

    # cross-H shrink: Euler bias monotone in H toward 0 at H=0.5
    ordered = sorted(euler_bias_by_H)  # ascending H
    biases = [euler_bias_by_H[h] for h in ordered]
    print(
        "\n  Cross-H Euler bias (mean|Δ|): "
        + "  ".join(f"H={h:.2f}:{b:.3f}" for h, b in zip(ordered, biases))
    )
    assert all(biases[i] > biases[i + 1] for i in range(len(biases) - 1)), (
        f"Euler bias not monotone-decreasing in H: {dict(zip(ordered, biases))}"
    )

    # Wall-clock anti-second-matmul guard. Perf-only and run after the bias checks: min-of-repeats
    # timing, and a regression prints a WARNING instead of aborting the bias run.
    tinfo = _time_build_components(H_PRODUCTION, params_df, rng_seed=42)
    print(
        f"\n  [timing] driver build @ n={tinfo['n_steps']}, N={N_PATHS}:  "
        f"t_kernel={tinfo['t_kernel'] * 1e3:.1f}ms  t_matmul={tinfo['t_matmul'] * 1e3:.1f}ms  "
        f"κ=0 build={tinfo['t_build0'] * 1e3:.1f}ms"
    )
    perf_ok = True
    for kappa in HYBRID_KAPPAS:
        pk = tinfo["per_kappa"][kappa]
        print(
            f"    κ={kappa}: build={pk['t_build'] * 1e3:6.1f}ms ({pk['ratio']:.2f}× κ=0)  "
            f"t_near={pk['t_near'] * 1e3:6.1f}ms  "
            f"residual={pk['residual'] * 1e3:+6.1f}ms (< {MATMUL_RESIDUAL_FRAC}×t_matmul="
            f"{MATMUL_RESIDUAL_FRAC * tinfo['t_matmul'] * 1e3:.1f}ms)"
        )
        if pk["residual"] >= MATMUL_RESIDUAL_FRAC * tinfo["t_matmul"]:
            perf_ok = False
            print(
                f"    WARNING κ={kappa}: residual {pk['residual'] * 1e3:.1f}ms ≥ "
                f"{MATMUL_RESIDUAL_FRAC}×t_matmul — possible second N·n² matmul. A real one adds "
                f"≈ +1.0×t_matmul (~{tinfo['t_matmul'] * 1e3:.0f}ms) CONSISTENTLY, not as a transient "
                f"spike; min-of-repeats already filters load — re-run on a quiet host before treating as a regression."
            )
    if perf_ok:
        print(
            "    → build fully accounted for by one matmul + kernel + near draw (no second matmul)"
        )

    print("\n" + "=" * 70)
    print(f"{'ALL HYBRID-ARM BIAS CHECKS PASSED':^70}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
