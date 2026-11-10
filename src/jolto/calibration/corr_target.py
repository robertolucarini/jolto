"""Historical correlation target for the Stage-2 decay penalty.

Pipeline (historical_target): daily forward-rate panel (market_data.load_forward_history)
-> drop_glitches -> historical_correlation -> fit_family -> embed_partial.

Families are from Schoenmakers & Coffey (2000) and Brigo & Mercurio (2006, §6.9); estimation follows
Brigo & Mercurio §6.19 (one-year window ending on the calibration date) with absolute instead of log
changes, since the model is normal, and weekly instead of daily changes, since daily changes carry
pillar noise (jolto.calibration.regul.targets measures it). The target
is put on the partial correlation given the vol driver, C = ρρᵀ + S R S with S = diag(√(1−ρ²)),
because the Stage-1 column ρ bounds the attainable correlations.
"""

import copy

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def drop_glitches(panel, z_min=15.0, revert_max=0.35):
    """Drop dates on which one forward jumps by z_min robust s.d. and reverts the next day (a single
    bad pillar quote), so the change is taken across the bad date. Genuine curve moves do not revert.
    Returns (clean_panel, dropped_dates)."""
    d = panel.diff().iloc[1:]
    mad = 1.4826 * d.abs().median()
    z = (d / mad).abs()
    bad = []
    for t in range(len(d) - 1):
        j = z.iloc[t].idxmax()
        if z.iloc[t][j] >= z_min and abs(d.iloc[t][j] + d.iloc[t + 1][j]) <= revert_max * abs(
            d.iloc[t][j]
        ):
            bad.append(d.index[t])
    return panel.drop(index=bad), bad


def historical_correlation(panel, start, end, step=1):
    """Sample correlation of absolute changes over `step` observations (non-overlapping), on the
    panel rows in [start, end]."""
    window = panel.loc[pd.Timestamp(start) : pd.Timestamp(end)]
    changes = window.iloc[::step].diff().iloc[1:]
    return changes.corr().to_numpy(), len(changes)


def _exp2(p, M):
    beta, rho_inf = p
    i = np.arange(1, M + 1, dtype=float)
    return rho_inf + (1.0 - rho_inf) * np.exp(-beta * np.abs(i[:, None] - i[None, :]))


def _sc2(p, M):
    # Schoenmakers–Coffey "improved" two-parameter form, SC (2000) eq. 22
    eta, rho_inf = p
    i = np.arange(1, M + 1, dtype=float)[:, None]
    j = i.T
    q = (i**2 + j**2 + i * j - 3 * M * i - 3 * M * j + 3 * i + 3 * j + 2 * M**2 - M - 4) / (
        (M - 2) * (M - 3)
    )
    return np.exp(-np.abs(i - j) / (M - 1) * (-np.log(rho_inf) + eta * q))


def _reb3(p, M):
    # Rebonato three-parameter form, BM (2006) eq. 6.45; β = α(M−1) + γ keeps entries ≤ 1
    alpha, gamma, rho_inf = p
    i = np.arange(1, M + 1, dtype=float)[:, None]
    j = i.T
    beta = alpha * (M - 1) + gamma
    return rho_inf + (1.0 - rho_inf) * np.exp(
        -np.abs(i - j) * (beta - alpha * (np.maximum(i, j) - 1))
    )


# name -> (builder, start, lower, upper); the sc2 bound η < −ln ρ∞ is enforced in fit_family
FAMILIES = {
    "exp": (_exp2, [0.05, 0.3], [1e-4, 0.0], [5.0, 0.999]),
    "sc2": (_sc2, [0.5, 0.3], [0.0, 1e-3], [20.0, 0.999]),
    "reb3": (_reb3, [0.001, 0.05, 0.3], [0.0, 1e-4, 0.0], [0.5, 5.0, 0.999]),
}


def family_matrix(name, params, M):
    return FAMILIES[name][0](np.asarray(params, dtype=float), M)


def embed_partial(R, rho):
    """Unconditional forward block C = ρρᵀ + S R S, S = diag(√(1−ρ²)): valid next to column 0 = ρ."""
    s = np.sqrt(1.0 - rho**2)
    C = np.outer(rho, rho) + np.outer(s, s) * R
    np.fill_diagonal(C, 1.0)
    return C


def fit_family(name, rho_hat, rho=None, first=3):
    """Unweighted least squares of the family to the historical correlation on the strict lower
    triangle of forwards first..M (1-based). With rho (the Stage-1 column), fits the embedded
    ρρᵀ + S R S instead of R. Returns a dict with params, rmse (correlation units), min eigenvalues."""
    build, x0, lo, hi = FAMILIES[name]
    M = rho_hat.shape[0]
    idx = np.arange(first - 1, M)
    rows, cols = np.tril_indices(len(idx), -1)
    ii, jj = idx[rows], idx[cols]

    def model(p):
        R = build(p, M)
        return embed_partial(R, rho) if rho is not None else R

    def resid(p):
        r = model(p)[ii, jj] - rho_hat[ii, jj]
        if name == "sc2":  # outside η < −ln ρ∞ the form is not PSD
            r = np.append(r, 10.0 * max(0.0, p[0] + np.log(p[1])))
        return r

    best = None
    for scale in (0.5, 1.0, 2.0):
        start = np.clip(np.asarray(x0) * scale, np.asarray(lo) + 1e-9, np.asarray(hi) - 1e-9)
        sol = least_squares(resid, start, bounds=(lo, hi))
        if best is None or sol.cost < best.cost:
            best = sol
    R = build(best.x, M)
    C = model(best.x)
    err = C[ii, jj] - rho_hat[ii, jj]
    out = {
        "family": name,
        "params": best.x.tolist(),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "min_eig_R": float(np.linalg.eigvalsh(R)[0]),
        "embedded": rho is not None,
    }
    if rho is not None:  # the full (M+1)x(M+1) target with column 0 = ρ
        full = np.block([[np.ones((1, 1)), rho[None, :]], [rho[:, None], C]])
        out["min_eig_full"] = float(np.linalg.eigvalsh(full)[0])
    return out


# shortcut: process-lifetime memo, assumes the curve files do not change during a run; the cv_1se
# fold fits build ~70 calibrators on the same inputs, and each build re-read every curve file
_HIST_CACHE = {}


def historical_target(curve_dir, end, years, step, family, rho, grid_tau, max_maturity):
    """Embedded target forward block for Stage 2: family fitted to the correlation of `step`-observation
    changes of the forwards over `years` ending on `end`, on the Stage-1 column rho.
    Returns (C, info) with C the (M, M) forward block and info the fit record plus provenance.
    Memoised on every argument; callers get copies, so the cached entry cannot be mutated."""
    rho = np.asarray(rho)
    rho_key = (rho.dtype.str, rho.shape, rho.tobytes())
    key = (curve_dir, end, years, step, family, rho_key, grid_tau, max_maturity)
    if key not in _HIST_CACHE:
        _HIST_CACHE[key] = _historical_target(
            curve_dir, end, years, step, family, rho, grid_tau, max_maturity
        )
    C, info = _HIST_CACHE[key]
    return C.copy(), copy.deepcopy(info)


def _historical_target(curve_dir, end, years, step, family, rho, grid_tau, max_maturity):
    from jolto.core.market_data import load_forward_history

    start = (pd.Timestamp(end) - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    panel, dropped = drop_glitches(
        load_forward_history(curve_dir, grid_tau, max_maturity, end=pd.Timestamp(end))
    )
    rho_hat, n = historical_correlation(panel, start, end, step)
    if rho_hat.shape[0] != rho.size:
        raise ValueError(f"history has {rho_hat.shape[0]} forwards, Stage 1 has {rho.size}")
    fit = fit_family(family, rho_hat, rho=rho)
    C = embed_partial(family_matrix(family, fit["params"], rho.size), rho)
    info = {
        **fit,
        "window": [start, end],
        "step": step,
        "n_changes": n,
        "glitches_dropped": [d.strftime("%Y-%m-%d") for d in dropped],
    }
    return C, info
