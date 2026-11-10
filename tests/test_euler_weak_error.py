"""
Euler (left-point) weak error of the rough-vol scheme, resolved on the 1D conditional engine.

Why the 1D engine. At β=0 with frozen drift the FMM's time discretisation enters only through the
two left-point sums I_V = Σ V(t_{i−1})ΔZ and I_{V²} = Σ V(t_{i−1})²Δt, which is exactly what
mc_rough_sabr_pricer forms. Each swaption is priced through its SMM-mapped 1D process;
the proxy rests on the mapping fidelity (SMM against FMM in tests/oos.py). The FMM-only drift-coupling term is measured separately (test_drift_freezing_bias).

Construction
------------
Grids are independent (own seeds); within a grid every arm shares the same ΔZ (PRNG, float64).
- P  production: build_hybrid_rough_driver(κ=1) and its construction compensator.
- E  exact: Ẑ = ΔZ·Kᵀ + ε·L_residᵀ with K the cell-average (κ=0) kernel, so E[Ẑ|ΔZ] is exact and
     C_resid = C_exact − Δt·K·Kᵀ is the Schur complement (PSD; min eigenvalue reported). C_exact is
     the quadrature of test_kernel_discretization_bias; compensator ½ν²·diag(C_exact). Only the
     left-point approximation of I_V, I_{V²} remains, so E isolates the Euler term.
- M  P with second-moment-matching tail weights ((1/h)∫k²)^½ (Gassiat 2023 Eq 3.3, after HJM17),
     same ΔZ and the same near-cell draw; compensator recomputed from M's own weights.
P draws the residual of the near cell only and drops the tail residual; E draws all of it. That
dropped tail variance is the n^{−2H} deficit Gassiat (2023, §3) shows for the MSE-optimal
cell-average weights, which is why P and E differ by a kernel term with a slow rate.

Cells (one 30Y driver prices all, ATM): four SMM-mapped swaptions (5Yx5Y, 21Yx10Y, 26Yx5Y,
30Yx1Y) at their mapped ρ_SMM, the same four at ρ=0 (the FSW Thm 1.1(ii) rate-1 gate), and a
control: the Stage-1 5Y marginal (α, ρ), which reproduces the kernel-bias harness's 5Y ATM cell.
The CV coefficient comes from independent pilot batches (an in-batch OLS β is biased O(1/batch)).

Read-off
--------
- Unit-mean and E[I_{V²}] law checks on every arm/grid (asserted, |z| < 3) before any output.
- Control: P−E on the Stage-1 5Y cell at 1/24, 1/50 against the kernel harness (decision-pass 5Y
  ATM cell, signed; kernel_dt_sweep mean|Δ|), within 2× the combined SE.
- ρ=0 gate: joint E fit with a common p. fail ⇔ 95% CI excludes 1; inconclusive ⇔ no cell resolved
  at 3σ or half-width > 0.5.
- Per ρ_SMM cell, E-only fit IV(Δt) = IV₀ + c·Δt^p (free p, and p ∈ {0.8, 1.0}); Euler = E(Δt) − IV₀;
  kernel = P(Δt) − E(Δt) (paired); total = P(Δt) − IV₀ = kernel + Euler; the production gap
  P(1/24) − P(1/50), which is what reaches the pricers (calibration at 1/50 absorbs the 1/50 bias
  into α to first order only: Stage 1 fits the 1Y-tenor marginals, not these mapped cells).
- Cross-check: P(1/24) − P(1/96) on 26Yx5Y against the coupled FMM harness two-grid number.
CIs are percentile bootstraps over batches (resampled jointly across arms within a grid).

Usage:   python tests/test_euler_weak_error.py [--tier-b] [--analyse-only] [--batches N --out DIR]
Outputs: results/euler_weak_error_H0.10.csv, results/euler_weak_error_H0.10.json,
         per-grid checkpoints in results/euler_weak_error_ckpt/ (resume: finished grids are skipped).
"""

import argparse
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares

from test_kernel_discretization_bias import _build_C_exact  # reuse, don't re-derive

from jolto import config, val_config
from jolto.api.assembly import build_calibrated_model, load_valuation_inputs
from jolto.calibration.pricers import (
    INV_SQRT_2PI,
    _compute_mapped_smm_parameters,
    bachelier_iv_newton,
    mc_rough_sabr_pricer,
)
from jolto.core.analytics import build_fractional_kernel, build_hybrid_rough_driver

# CONFIG CONTRACT: the scheme under test is the production one; fail before a 2 h run if not.
assert config.HYBRID_KAPPA == 1, "production κ changed — this harness measures κ=1"
assert abs(val_config.PRICE_DT - 1 / 24) < 1e-15 and abs(config.CALI_DT - 1 / 50) < 1e-15

DEVICE, DTYPE = "cpu", torch.float64
H = val_config.VAL_H_TARGET
SEED = 42

# grid denominator → arms. M only on the two production grids.
TIER_A = {24: ("P", "E", "M"), 48: ("E",), 50: ("P", "E", "M"), 96: ("P", "E"), 192: ("E",)}
TIER_B = {384: ("E",)}
BATCH = {d: 8192 for d in TIER_A} | {384: 4096}  # memory: ~15 live (B×n) float64 tensors
N_BATCHES = {d: 1221 for d in TIER_A} | {384: 977}  # ≈10⁷ paths (Tier B ≈4·10⁶)
N_PILOT = 8  # pilot batches for the CV coefficient

INSTRUMENTS = [(5.0, 5.0), (21.0, 10.0), (26.0, 5.0), (30.0, 1.0)]
T_MAX = 30.0
SPOT_T = (1.0, 5.0, 10.0, 21.0, 26.0, 30.0)  # unit-mean spot checks
Z_MAX = 3.0
RESOLVE_SIGMA = 3.0
N_BOOT = 2000
M_GATE_MULT = 3.0

RESULTS = config.RESULTS_DIR


def _kernel_1_24():
    """κ=1 ATM mean|Δ| at 1/24 from the dt sweep (results/kernel_dt_sweep_H0.10.csv), the scale
    of the M-arm gate; read from the CSV so it follows the current calibration."""
    path = os.path.join(RESULTS, "kernel_dt_sweep_H0.10.csv")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing: run the kernel dt sweep first (the M-arm gate needs it)")
    sw = pd.read_csv(path)
    row = sw[(np.round(1 / sw.dt) == 24) & (sw.kappa == 1)]
    if row.empty:
        raise SystemExit(f"{path} has no dt = 1/24, kappa = 1 row")
    return float(row.atm_mean_abs_bps.iloc[0])


def _label(d):
    return f"1/{d}"


def _seed(d, b, stream):
    # disjoint streams per (grid, batch): dz=0, near=2e6, eps=4e6; pilot batches at b=1e6+k
    return SEED * 10**10 + d * 10**7 + stream * 2 * 10**6 + b


def _cells(model, Sigma):
    """(name, T, α, ρ, kind) for the 9 cells; ν is the global model.nu."""
    ex = [e for e, _ in INSTRUMENTS]
    te = [t for _, t in INSTRUMENTS]
    a, r = _compute_mapped_smm_parameters(model, Sigma.to(DTYPE), ex, te)
    s1 = pd.read_csv(os.path.join(RESULTS, config.STAGE1_PARAMS_CSV))
    s1 = s1[(s1.H - H).abs() < 1e-6]
    row5 = s1[(s1.Expiry - 5.0).abs() < 1e-9].iloc[0]
    assert abs(float(row5.Nu) - float(model.nu)) < 1e-12, (
        "Stage-1 ν ≠ model.nu (ν is one global scalar)"
    )
    cells = []
    for (e, t), ai, ri in zip(INSTRUMENTS, a.tolist(), r.tolist()):
        cells.append((f"{int(e)}Yx{int(t)}Y", e, ai, ri, "rho_smm"))
    for (e, t), ai in zip(INSTRUMENTS, a.tolist()):
        cells.append((f"{int(e)}Yx{int(t)}Y_rho0", e, ai, 0.0, "rho0"))
    cells.append(("stage1_5Y", 5.0, float(row5.Alpha), float(row5.Rho), "control"))
    return cells


def _moment_matched_tail(n, dt):
    """Toeplitz tail kernel with weights ((1/Δt)∫_{(k−1)Δt}^{kΔt} k(r)² dr)^½, lags k ≥ 2."""
    g = math.gamma(H + 0.5)
    k = torch.arange(1, n + 1, dtype=DTYPE)
    w = torch.sqrt(dt ** (2 * H - 1) * (k ** (2 * H) - (k - 1) ** (2 * H)) / (2 * H)) / g
    w[0] = 0.0  # lag 1 is the near cell, drawn exactly as in P
    i = torch.arange(n)
    lag = (i[:, None] - i[None, :]).clamp(min=0)
    return torch.where(i[:, None] >= i[None, :], w[lag], torch.zeros((), dtype=DTYPE))


def _grid_setup(d, arms):
    dt = 1.0 / d
    n = int(round(T_MAX * d))
    for T in [T_MAX] + list(SPOT_T):
        assert math.ceil(T / dt) == int(round(T * d)), f"T={T} not on grid 1/{d}"
    t_grid = np.arange(1, n + 1, dtype=np.float64) * dt
    K = build_fractional_kernel(H, n, dt, DEVICE, DTYPE)
    K_tail = K.clone()
    K_tail.fill_diagonal_(0.0)
    s = {"dt": dt, "n": n, "K": K}
    t0 = time.time()
    C_exact = _build_C_exact(t_grid, H)
    s["var_exact"] = torch.tensor(np.diag(C_exact).copy(), dtype=DTYPE)
    if "E" in arms:
        Kn = K.numpy()
        C_resid = C_exact - dt * (Kn @ Kn.T)
        C_resid = 0.5 * (C_resid + C_resid.T)
        min_eig = float(np.linalg.eigvalsh(C_resid).min())
        s["min_eig_resid"] = min_eig
        # PSD by construction; a negative min_eig is quadrature round-off, nugget as the kernel test
        if min_eig < 0:
            C_resid += max(-min_eig * 1.01, 1e-14) * np.eye(n)
        s["L_resid"] = torch.tensor(np.linalg.cholesky(C_resid), dtype=DTYPE)
    if "M" in arms:
        Kc = _moment_matched_tail(n, dt)
        s["D_M"] = Kc - K_tail
        s["dvar_M"] = dt * ((Kc**2).sum(1) - (K_tail**2).sum(1))
    s["setup_s"] = time.time() - t0
    return s


def _drivers(s, d, b, arms, batch):
    dt, n = s["dt"], s["n"]
    g = torch.Generator().manual_seed(_seed(d, b, 0))
    dZ = torch.randn(batch, n, generator=g, dtype=DTYPE) * math.sqrt(dt)
    out = {}
    if "P" in arms or "M" in arms:
        g_near = torch.Generator().manual_seed(_seed(d, b, 1))
        fbm, var, _ = build_hybrid_rough_driver(
            dZ, H, n, dt, 1, DEVICE, DTYPE, near_generator=g_near
        )
        if "P" in arms:
            out["P"] = (fbm, var)
        if "M" in arms:
            out["M"] = (fbm + dZ @ s["D_M"].T, var + s["dvar_M"])
    if "E" in arms:
        g_eps = torch.Generator().manual_seed(_seed(d, b, 2))
        eps = torch.randn(batch, n, generator=g_eps, dtype=DTYPE)
        out["E"] = (dZ @ s["K"].T + eps @ s["L_resid"].T, s["var_exact"])
    return dZ, out


def _price(s, dZ, Z, var, cells, nu, beta=None):
    """Batch-mean call prices (n_cells,) through the library engine; or the batch β if beta is None."""
    T = torch.tensor([c[1] for c in cells], dtype=DTYPE)
    a = torch.tensor([c[2] for c in cells], dtype=DTYPE)
    r = torch.tensor([c[3] for c in cells], dtype=DTYPE)
    # hybrid_kappa=1 only selects the var_vec compensator branch; every arm passes its own variance
    cd = {"fbm": Z, "kernel": s["K"], "dz_vol": dZ, "var_vec": var, "hybrid_kappa": 1}
    kw = dict(dt=s["dt"], device=DEVICE, cached_drivers=cd, dtype=DTYPE, hybrid_kappa=1)
    K0 = torch.zeros(len(cells), dtype=DTYPE)
    if beta is None:
        _, bt = mc_rough_sabr_pricer(K0, T, a, r, nu, H, return_beta=True, **kw)
        return bt
    return mc_rough_sabr_pricer(K0, T, a, r, nu, H, precomputed_beta=beta, **kw)


def _sanity_accumulate(acc, s, Z, var, nu):
    """Σ, Σ² of V at spot times and of the left-point I_{V²}(T) at the instrument expiries."""
    d = int(round(1 / s["dt"]))
    spot = [int(round(t * d)) - 1 for t in SPOT_T]  # fbm column c ↔ t_{c+1}
    V = torch.exp(nu * Z[:, spot] - 0.5 * nu**2 * var[spot])
    cs = torch.cumsum(torch.exp(2 * nu * Z - nu**2 * var), dim=1)
    # engine: unit_vols = [1, V(t_1..t_{n_T−1})] ⇒ I_{V²}(T) = Δt·(1 + Σ_{c ≤ n_T−2} V_c²)
    iv2 = torch.stack(
        [s["dt"] * (1.0 + cs[:, int(round(e * d)) - 2]) for e, _ in INSTRUMENTS], dim=1
    )
    for key, x in (("V", V), ("IV2", iv2)):
        acc[key + "_s"] = acc.get(key + "_s", 0) + x.sum(0)
        acc[key + "_ss"] = acc.get(key + "_ss", 0) + (x * x).sum(0)
    acc["n"] = acc.get("n", 0) + Z.shape[0]


def run_grid(d, arms, n_batches, batch, cells, nu, ckpt_path):
    s = _grid_setup(d, arms)
    print(
        f"\n── grid {_label(d)}  n={s['n']}  arms={arms}  batches={n_batches}×{batch}  "
        f"setup {s['setup_s']:.1f}s"
        + (f"  min_eig(C_resid)={s['min_eig_resid']:.3e}" if "min_eig_resid" in s else ""),
        flush=True,
    )
    betas = {}
    for k in range(N_PILOT):
        dZ, drv = _drivers(s, d, 10**6 + k, arms, batch)
        for arm, (Z, var) in drv.items():
            betas.setdefault(arm, []).append(_price(s, dZ, Z, var, cells, nu))
        del dZ, drv
    betas = {a: torch.stack(v).mean(0) for a, v in betas.items()}

    means = {a: np.zeros((n_batches, len(cells))) for a in arms}
    acc = {a: {} for a in arms}
    var_arm = {}
    t0 = time.time()
    for b in range(n_batches):
        dZ, drv = _drivers(s, d, b, arms, batch)
        for arm, (Z, var) in drv.items():
            means[arm][b] = _price(s, dZ, Z, var, cells, nu, betas[arm]).numpy()
            _sanity_accumulate(acc[arm], s, Z, var, nu)
            var_arm[arm] = var.numpy()
        del dZ, drv
        if (b + 1) % max(1, n_batches // 10) == 0:
            el = time.time() - t0
            print(
                f"   {b + 1}/{n_batches}  {el:.0f}s  eta {el / (b + 1) * (n_batches - b - 1):.0f}s",
                flush=True,
            )

    # M's variance from its own weights must equal the exact one (Gassiat Assumption 1, Eq 3.4)
    m_mis = None
    if "M" in arms:
        m_mis = float(np.abs(var_arm["M"] / s["var_exact"].numpy() - 1).max())
    payload = {
        "d": d,
        "dt": s["dt"],
        "n": s["n"],
        "arms": list(arms),
        "batch": batch,
        "n_batches": n_batches,
        "betas": {a: v.numpy() for a, v in betas.items()},
        "means": means,
        "acc": {
            a: {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in acc[a].items()}
            for a in arms
        },
        "var_arm": var_arm,
        "min_eig_resid": s.get("min_eig_resid"),
        "M_rel_mismatch": m_mis,
        "run_s": time.time() - t0,
        "cells": cells,
    }
    torch.save(payload, ckpt_path)
    print(f"   done {payload['run_s']:.0f}s → {ckpt_path}", flush=True)


def _sanity(grids, nu):
    rows = []
    for d, g in grids.items():
        for arm in g["arms"]:
            a = g["acc"][arm]
            n = a["n"]
            var = np.asarray(g["var_arm"][arm])
            for key, expect in (
                ("V", np.ones(len(SPOT_T))),
                (
                    "IV2",
                    np.array(
                        [
                            g["dt"] * (1.0 + np.exp(nu**2 * var[: int(round(e * d)) - 1]).sum())
                            for e, _ in INSTRUMENTS
                        ]
                    ),
                ),
            ):
                m = a[key + "_s"] / n
                se = np.sqrt(np.maximum(a[key + "_ss"] / n - m**2, 0) / n)
                z = (m - expect) / se
                rows.append(
                    {
                        "grid": _label(d),
                        "arm": arm,
                        "check": key,
                        "worst_abs_z": float(np.abs(z).max()),
                        "worst_rel_dev": float(np.abs(m / expect - 1).max()),
                    }
                )
    return pd.DataFrame(rows)


def _iv(price, T):
    return price / (np.sqrt(T) * INV_SQRT_2PI) * 1e4  # exact at the money (Bachelier)


def _boot_ivs(grids, cells, rng):
    """Point IVs and bootstrap replicates: {(d, arm): (iv (cells,), reps (N_BOOT, cells))}."""
    Ts = np.array([c[1] for c in cells])
    out = {}
    for d, g in grids.items():
        idx = rng.integers(0, g["n_batches"], size=(N_BOOT, g["n_batches"]))
        for arm in g["arms"]:
            m = g["means"][arm]
            out[(d, arm)] = (_iv(m.mean(0), Ts), _iv(m[idx].mean(1), Ts))
    return out


def _fit_free(dts, y, se):
    p0 = 0.8
    c0 = (y[0] - y[-1]) / (dts[0] ** p0 - dts[-1] ** p0)
    x0 = [y[-1] - c0 * dts[-1] ** p0, c0, p0]
    res = least_squares(
        lambda x: (x[0] + x[1] * dts ** x[2] - y) / se,
        x0,
        bounds=([-np.inf, -np.inf, 0.05], [np.inf, np.inf, 3.0]),
    )
    iv0, c, p = res.x
    return {"iv0": iv0, "c": c, "p": p, "chi2_dof": float(2 * res.cost / max(len(y) - 3, 1))}


def _fit_fixed(dts, y, se, p):
    X = np.stack([np.ones_like(dts), dts**p], 1) / se[:, None]
    coef, *_ = np.linalg.lstsq(X, y / se, rcond=None)
    chi2 = float(((X @ coef - y / se) ** 2).sum())
    return {"iv0": coef[0], "c": coef[1], "p": p, "chi2_dof": chi2 / max(len(y) - 2, 1)}


def _fit_joint(dts, Y, SE):
    """Common p, per-cell (IV₀, c). Y, SE: (cells, grids)."""
    m = Y.shape[0]
    p0 = 0.8
    c0 = (Y[:, 0] - Y[:, -1]) / (dts[0] ** p0 - dts[-1] ** p0)
    x0 = np.concatenate([[p0], Y[:, -1] - c0 * dts[-1] ** p0, c0])

    def r(x):
        p, iv0, c = x[0], x[1 : m + 1], x[m + 1 :]
        return ((iv0[:, None] + c[:, None] * dts[None, :] ** p - Y) / SE).ravel()

    lo = np.full(2 * m + 1, -np.inf)
    hi = np.full(2 * m + 1, np.inf)
    lo[0], hi[0] = 0.05, 3.0
    res = least_squares(r, x0, bounds=(lo, hi))
    return {"p": float(res.x[0]), "chi2_dof": float(2 * res.cost / max(Y.size - 2 * m - 1, 1))}


def _ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def _stat(point, reps):
    se = float(np.std(reps))
    return {
        "value": float(point),
        "se": se,
        "ci95": _ci(reps),
        "resolved_3sigma": bool(abs(point) > RESOLVE_SIGMA * se),
    }


def analyse(grids, cells, nu, out_dir):
    san = _sanity(grids, nu)
    print("\nSanity (worst |z| per grid/arm/check):")
    print(san.to_string(index=False))
    bad = san[san.worst_abs_z >= Z_MAX]
    assert bad.empty, f"sanity check failed — no outputs written:\n{bad.to_string(index=False)}"

    rng = np.random.default_rng(SEED)
    bi = _boot_ivs(grids, cells, rng)
    names = [c[0] for c in cells]

    rows = []
    for (d, arm), (iv, reps) in sorted(bi.items()):
        g = grids[d]
        for j, c in enumerate(cells):
            rows.append(
                {
                    "cell": c[0],
                    "kind": c[4],
                    "T": c[1],
                    "alpha": c[2],
                    "rho": c[3],
                    "grid": _label(d),
                    "dt": g["dt"],
                    "arm": arm,
                    "n_paths": g["n_batches"] * g["batch"],
                    "price": float(g["means"][arm][:, j].mean()),
                    "iv_bps": float(iv[j]),
                    "se_bps": float(np.std(reps[:, j])),
                    "beta_cv": float(np.asarray(g["betas"][arm])[j]),
                }
            )
    df = pd.DataFrame(rows)
    # the ATM linear inversion must agree with the library Newton inversion
    chk = df.iloc[0]
    iv_newton = (
        float(
            bachelier_iv_newton(
                torch.tensor([chk.price], dtype=DTYPE),
                torch.zeros(1, dtype=DTYPE),
                torch.tensor([chk["T"]], dtype=DTYPE),
                torch.tensor([chk.alpha], dtype=DTYPE),
            ).item()
        )
        * 1e4
    )
    assert abs(iv_newton - chk.iv_bps) < 1e-6, (iv_newton, chk.iv_bps)

    e_grids = sorted([d for d in grids if "E" in grids[d]["arms"]])
    dts = np.array([1.0 / d for d in e_grids])

    def E(j):
        y = np.array([bi[(d, "E")][0][j] for d in e_grids])
        R = np.stack([bi[(d, "E")][1][:, j] for d in e_grids], 1)
        return y, R, R.std(0)

    summary = {
        "H": H,
        "nu": nu,
        "kappa": 1,
        "grids": {
            _label(d): {
                "arms": grids[d]["arms"],
                "n_paths": grids[d]["n_batches"] * grids[d]["batch"],
                "min_eig_C_resid": grids[d]["min_eig_resid"],
                "M_rel_var_mismatch": grids[d]["M_rel_mismatch"],
                "run_s": grids[d]["run_s"],
            }
            for d in grids
        },
        "sanity_worst_abs_z": float(san.worst_abs_z.max()),
        "assumptions": [
            "1D SMM-mapped proxy for the FMM swaption.",
            "At beta=0 with frozen drift the FMM time discretisation enters only via I_V and I_V2; "
            "the drift's own left-point error is second order.",
            "Calibration at 1/50 absorbs the 1/50 bias into alpha to first order only: Stage 1 "
            "fits 1Y-tenor marginals, not these mapped cells.",
            "Euler term = E-arm extrapolated bias at 1/24; the production gap is "
            "P(1/24)-P(1/50); kernel (P-E) + Euler (E-IV0) is the additive time-discretisation "
            "budget and equals total = P - IV0.",
        ],
    }

    # control: reproduce the kernel harness's Stage-1 5Y ATM κ=1 cell
    jc = names.index("stage1_5Y")
    ctrl = {"pass": True}
    for d in (24, 50):
        kd = bi[(d, "P")][1][:, jc] - bi[(d, "E")][1][:, jc]
        val = bi[(d, "P")][0][jc] - bi[(d, "E")][0][jc]
        se = float(kd.std())
        dec = pd.read_csv(os.path.join(RESULTS, f"kernel_kappa_decision_dt{d}_T5.csv"))
        dec = dec[(dec.arm == "hybrid_k1") & (dec.K.abs() < 1e-12)].iloc[0]
        sw = pd.read_csv(os.path.join(RESULTS, "kernel_dt_sweep_H0.10.csv"))
        sw = sw[(np.round(1 / sw.dt) == d) & (sw.kappa == 1)]
        comb_dec = math.hypot(se, float(dec.SE_paired_bps))
        ok_dec = abs(val - float(dec.delta_bps)) <= 2 * comb_dec
        ok_sw = True
        sw_ref = None
        if len(sw):
            sw_ref = {
                "atm_mean_abs_bps": float(sw.atm_mean_abs_bps.iloc[0]),
                "atm_paired_se_bps": float(sw.atm_paired_se_bps.iloc[0]),
            }
            comb_sw = math.hypot(se, sw_ref["atm_paired_se_bps"])
            ok_sw = abs(val) <= sw_ref["atm_mean_abs_bps"] + 2 * comb_sw
        ctrl[_label(d)] = {
            "P_minus_E_bps": float(val),
            "se": se,
            "decision_pass_5Y_atm": {
                "delta_bps": float(dec.delta_bps),
                "se": float(dec.SE_paired_bps),
                "agree_2se": ok_dec,
            },
            "kernel_dt_sweep": sw_ref,
            "agree_2se_meanabs": ok_sw,
        }
        ctrl["pass"] &= bool(ok_dec and ok_sw)
    summary["kernel_control"] = ctrl

    # ρ = 0 gate: joint E fit, common p
    j0 = [j for j, c in enumerate(cells) if c[4] == "rho0"]
    Y = np.array([E(j)[0] for j in j0])
    R = np.array([E(j)[1] for j in j0])  # (cells, boot, grids)
    SE = R.std(1)
    diff = R[:, :, 0] - R[:, :, -1]
    resolved = [
        names[j]
        for j, dd, k in zip(j0, Y[:, 0] - Y[:, -1], range(len(j0)))
        if abs(dd) > RESOLVE_SIGMA * diff[k].std()
    ]
    pj = _fit_joint(dts, Y, SE)
    pb = np.array([_fit_joint(dts, R[:, b, :], SE)["p"] for b in range(N_BOOT)])
    ci = _ci(pb)
    if not resolved:
        status = "inconclusive"
    elif ci[0] > 1.0 or ci[1] < 1.0:
        status = "fail"
    elif (ci[1] - ci[0]) / 2 > 0.5:
        status = "inconclusive"
    else:
        status = "pass"
    summary["rho0_gate"] = {
        "status": status,
        "p_joint": pj["p"],
        "p_ci95": ci,
        "chi2_dof": pj["chi2_dof"],
        "resolved_cells": resolved,
        "theory": "FSW 2024 Thm 1.1(ii): rate 1 at rho=0",
    }
    print(f"\nρ=0 gate: {status}  p={pj['p']:.3f}  CI {ci}  resolved={resolved}")
    print(f"kernel control: {'PASS' if ctrl['pass'] else 'FAIL'}")

    cellsum = {}
    if status != "fail":
        jr = [j for j, c in enumerate(cells) if c[4] in ("rho_smm", "control")]
        for j in jr:
            y, Rj, sej = E(j)
            free = _fit_free(dts, y, sej)
            fb = [_fit_free(dts, Rj[b], sej) for b in range(N_BOOT)]
            iv0_b = np.array([f["iv0"] for f in fb])
            p_b = np.array([f["p"] for f in fb])
            cs = {
                "E_fit_free": {
                    **{k: float(v) for k, v in free.items()},
                    "p_ci95": _ci(p_b),
                    "iv0_se": float(iv0_b.std()),
                },
                "E_fit_fixed": {
                    str(p): {k: float(v) for k, v in _fit_fixed(dts, y, sej, p).items()}
                    for p in (0.8, 1.0)
                },
            }
            # fixed-p IV₀ replicates: the free fit is unidentified when c·Δt^p is only a few σ
            iv0_fix = {
                p: np.array([_fit_fixed(dts, Rj[b], sej, p)["iv0"] for b in range(N_BOOT)])
                for p in (0.8, 1.0)
            }
            for d in (24, 50):
                Pp, PR = bi[(d, "P")][0][j], bi[(d, "P")][1][:, j]
                Ep, ER = bi[(d, "E")][0][j], bi[(d, "E")][1][:, j]
                fx = {p: cs["E_fit_fixed"][str(p)]["iv0"] for p in (0.8, 1.0)}
                blk = {
                    "euler": _stat(Ep - free["iv0"], ER - iv0_b),
                    "kernel": _stat(Pp - Ep, PR - ER),
                    "total": _stat(Pp - free["iv0"], PR - iv0_b),
                    "euler_fixed_p": {
                        str(p): _stat(Ep - fx[p], ER - iv0_fix[p]) for p in (0.8, 1.0)
                    },
                    "total_fixed_p": {
                        str(p): _stat(Pp - fx[p], PR - iv0_fix[p]) for p in (0.8, 1.0)
                    },
                }
                if (d, "M") in bi:
                    MR = bi[(d, "M")][1][:, j]
                    blk["M_kernel"] = _stat(bi[(d, "M")][0][j] - Ep, MR - ER)
                cs[_label(d)] = blk
            gap_R = bi[(24, "P")][1][:, j] - bi[(50, "P")][1][:, j]
            cs["production_gap_24_50"] = _stat(bi[(24, "P")][0][j] - bi[(50, "P")][0][j], gap_R)
            g96 = bi[(24, "P")][1][:, j] - bi[(96, "P")][1][:, j]
            cs["production_gap_24_96"] = _stat(bi[(24, "P")][0][j] - bi[(96, "P")][0][j], g96)
            cellsum[names[j]] = cs

        # M is reported only when the kernel term is real and large
        k24 = [
            abs(cellsum[names[j]]["1/24"]["kernel"]["value"])
            for j in jr
            if cells[j][4] == "rho_smm"
        ]
        m_gate = bool(ctrl["pass"] and max(k24) > M_GATE_MULT * _kernel_1_24())
        summary["M_arm_reported"] = m_gate
        if not m_gate:
            for cs in cellsum.values():
                for d in ("1/24", "1/50"):
                    cs[d].pop("M_kernel", None)

        # FMM cross-check, 26Yx5Y, like with like (production scheme, 1/24 − 1/96)
        fmm = pd.read_csv(os.path.join(RESULTS, "euler_timestep_bias_coupled.csv"))
        fmm = fmm[fmm.instrument == "26Yx5Y"].set_index("grid")
        tot = pd.read_csv(os.path.join(RESULTS, "discretisation_bias_total.csv"))
        fmm_se = float(tot[tot.instrument == "26Yx5Y"].c3_se.iloc[0])
        fmm_v = float(fmm.loc["1/24", "iv_bps"] - fmm.loc["1/96", "iv_bps"])
        one_d = cellsum["26Yx5Y"]["production_gap_24_96"]
        z = (one_d["value"] - fmm_v) / math.hypot(one_d["se"], fmm_se)
        summary["fmm_crosscheck_26Yx5Y"] = {
            "fmm_two_grid_bps": fmm_v,
            "fmm_se": fmm_se,
            "one_d_P_24_minus_96_bps": one_d["value"],
            "one_d_se": one_d["se"],
            "z": float(z),
            "gross_conflict": bool(abs(z) > 3.0),
        }
    summary["cells"] = cellsum

    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f"euler_weak_error_H{H:.2f}.csv"), index=False)
    with open(os.path.join(out_dir, f"euler_weak_error_H{H:.2f}.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    _print_summary(summary)
    return summary


def _print_summary(s):
    print(f"\n{'cell':>10} {'grid':>5} {'kernel(P-E)':>16} {'euler(E-IV0)':>16} {'total':>16}")
    for name, cs in s["cells"].items():
        for d in ("1/24", "1/50"):
            b = cs[d]
            f = " ".join(
                f"{b[k]['value']:+8.3f}±{b[k]['se']:.3f}" for k in ("kernel", "euler", "total")
            )
            print(f"{name:>10} {d:>5} {f}")
        fr = cs["E_fit_free"]
        print(
            f"{'':>10} p={fr['p']:.3f} CI {np.round(fr['p_ci95'], 3).tolist()}  "
            f"gap24-50={cs['production_gap_24_50']['value']:+.3f}±"
            f"{cs['production_gap_24_50']['se']:.3f}"
        )
    if "fmm_crosscheck_26Yx5Y" in s:
        x = s["fmm_crosscheck_26Yx5Y"]
        print(
            f"\nFMM cross-check 26Yx5Y: 1D {x['one_d_P_24_minus_96_bps']:+.3f}±{x['one_d_se']:.3f} "
            f"vs FMM {x['fmm_two_grid_bps']:+.2f}±{x['fmm_se']:.2f}  z={x['z']:+.2f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-b", action="store_true")
    ap.add_argument("--analyse-only", action="store_true")
    ap.add_argument("--batches", type=int, default=None, help="override batches per grid (smoke)")
    ap.add_argument("--out", default=RESULTS)
    args = ap.parse_args()

    model, _, Sigma = build_calibrated_model(load_valuation_inputs(val_config), DEVICE)
    nu = float(model.nu)
    cells = _cells(model, Sigma)
    ckpt_dir = os.path.join(args.out, "euler_weak_error_ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)

    _kernel_1_24()  # fail before hours of simulation, not at the analysis
    plan = TIER_A | (TIER_B if args.tier_b else {})
    if not args.analyse_only:
        for d, arms in plan.items():
            path = os.path.join(ckpt_dir, f"grid_{d}.pt")
            nb = args.batches or N_BATCHES[d]
            if os.path.exists(path):
                ck = torch.load(path, weights_only=False)
                if ck["n_batches"] == nb and ck["arms"] == list(arms) and ck["cells"] == cells:
                    print(f"grid {_label(d)}: checkpoint present, skipped")
                    continue
                print(f"grid {_label(d)}: checkpoint is from other inputs or settings, recomputing")
            run_grid(d, arms, nb, BATCH[d], cells, nu, path)

    grids = {}
    for d in sorted(TIER_A | TIER_B):
        path = os.path.join(ckpt_dir, f"grid_{d}.pt")
        if os.path.exists(path):
            ck = torch.load(path, weights_only=False)
            # a recalibration changes the cells; never analyse a grid simulated on other inputs
            if ck["cells"] != cells:
                raise SystemExit(f"{path} was simulated on other cells; delete it and re-run")
            grids[d] = ck
    analyse(grids, cells, nu, args.out)


if __name__ == "__main__":
    main()
