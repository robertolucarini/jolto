"""Stage-2 λ selection (config.STAGE2_LAMBDA_RULE). Every candidate λ is an ordinary Stage-2
calibration, so the AMMO loop is untouched.

"monotone" reruns the --sweep selector (smallest λ with monotone rows). "cv_1se" is the
one-standard-error rule on K-fold cross-validation over held-out ATM pairs, ported from
tests/stage2_cv.py, which stays as the independent reference.

"cv_1se" stops ascending: the grid is walked upward, each λ is scored on every fold seed
and fold, and the rule returns the largest λ before the first one whose CV RMSE exceeds the running
minimum + that minimum's SE (the last grid λ if none does). The stopping λ is fitted and tabled; the
λ above it are not. This equals the full-grid one-SE pick (largest λ within one SE of the global
minimum) when the CV keeps rising above the pick; tests/stage2_cv.py scores the full grid as the check.
"""

import contextlib
import io
import time

import numpy as np
import pandas as pd
import torch

from jolto import config
from jolto.calibration.stage2 import CorrelationCalibrator
from jolto.calibration.pricers import mapped_smm_mc_pricer


@contextlib.contextmanager
def _quiet(**cfg):
    """Temporary config overrides with stdout silenced."""
    old = {k: getattr(config, k) for k in cfg}
    for k, v in cfg.items():
        setattr(config, k, v)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def _calibrate(atm, model, mc_store=None, **cfg):
    # log_file="" writes no log; selection is always AMMO_MC, as in the reference
    with _quiet(**cfg):
        cal = CorrelationCalibrator(atm, model)
        cal.mc_store = mc_store
        return cal.calibrate(method="AMMO_MC", log_file="")


def _make_folds(st, grid_T, k, seed):
    """Fold per quote, stratified by Σ row: round-robin from a random offset within each row."""
    rows = np.array(
        [
            int(np.argmin(np.abs(grid_T - (e + t))))
            for e, t in zip(st["global_exp"], st["global_ten"])
        ]
    )
    rng = np.random.default_rng(seed)
    fold = np.empty(len(rows), dtype=int)
    for r in np.unique(rows):
        idx = rng.permutation(np.flatnonzero(rows == r))
        fold[idx] = (rng.integers(k) + np.arange(len(idx))) % k
    # each row keeps ≥ 1 quote in every training set, so its angles stay in the optimisation
    for r in np.unique(rows):
        assert all(np.any((rows == r) & (fold != f)) for f in range(k)), r
    return fold


def _drop_pairs(atm, st, mask):
    out = atm.copy()
    for e, t in zip(st["global_exp"][mask], st["global_ten"][mask]):
        out.loc[
            out.index[np.argmin(np.abs(out.index.values - e))],
            out.columns[np.argmin(np.abs(out.columns.values - t))],
        ] = np.nan
    return out


def _price(cal, st, Sigma):
    """High-fidelity mapped-SMM normal vols (decimal) of the full target set on its drivers."""
    S = torch.as_tensor(Sigma, device=cal.device, dtype=cal.model.dtype)
    with torch.no_grad():
        v = mapped_smm_mc_pricer(
            cal.model,
            S,
            st["global_exp"],
            st["global_ten"],
            st["global_str"],
            dt=st["dt_val"],
            n_paths=st["n_paths_val"],
            cached_drivers=st["global_cached_drivers"],
        )
    return np.asarray(v, dtype=float)


def _cv_1se(atm, model):
    # fold fits rebuild the same seed-42 drivers; a fold that drops the longest expiry needs fewer
    # steps, hence the store is keyed by step count
    mc_store = {}
    cal0 = CorrelationCalibrator(atm, model)
    cal0.mc_store = mc_store
    with _quiet():
        st = cal0._build_correlation_target_grid()
    y = st["global_vol"]
    k = config.STAGE2_CV_FOLDS
    folds = {seed: _make_folds(st, cal0.grid_T, k, seed) for seed in config.STAGE2_CV_SEEDS}
    rows, cv, se = [], {}, {}
    lam_min, chosen = None, None
    # ascending stop: each λ is scored on all seeds and folds before the next is fitted
    for lam in sorted(config.STAGE2_CV_LAMBDA_GRID):
        err, test = [], []
        for seed, fold in folds.items():
            pred = np.full(len(y), np.nan)
            for f in range(k):
                t0 = time.time()
                held = fold == f
                res = _calibrate(_drop_pairs(atm, st, held), model, mc_store, LAMBDA_CURVATURE=lam)
                v = _price(cal0, st, res["Sigma_matrix"])
                pred[held] = v[held]
                test_rmse = float(np.sqrt(np.mean(((v - y)[held]) ** 2)) * 1e4)
                test.append(test_rmse)
                rows.append(
                    {
                        "seed": seed,
                        "lam": lam,
                        "fold": f,
                        "n_held": int(held.sum()),
                        "train_rmse": float(np.sqrt(np.mean(((v - y)[~held]) ** 2)) * 1e4),
                        "test_rmse": test_rmse,
                        "secs": round(time.time() - t0, 1),
                    }
                )
            err.append(pred - y)
        # pooled over folds and seeds; the SE of the RMSE is the delta method on the fold MSEs
        cv[lam] = float(np.sqrt(np.mean(np.concatenate(err) ** 2)) * 1e4)
        mse = np.asarray(test) ** 2
        se[lam] = float(np.std(mse, ddof=1) / np.sqrt(len(mse)) / (2 * cv[lam]))
        if lam_min is None or cv[lam] < cv[lam_min]:
            lam_min = lam
        if cv[lam] > cv[lam_min] + se[lam_min]:
            break
        chosen = lam

    # one row per seed, λ and fold, sorted by seed, then λ, then fold
    tab = pd.DataFrame(rows).sort_values(["seed", "lam", "fold"], kind="stable")
    tab = tab.reset_index(drop=True)
    tab["cv_rmse"] = tab["lam"].map(cv)
    tab["se"] = tab["lam"].map(se)
    tab["selected"] = tab["lam"] == chosen
    return float(chosen), tab


def _monotone(atm, model):
    res = _calibrate(atm, model, AUTO_LAMBDA=True)
    tab = pd.DataFrame(res["lambda_sweep_data"])
    return float(tab.loc[tab["advisory_selected"], "lambda"].iloc[0]), tab


def select_lambda(atm_matrix, model, rule):
    """(λ, table) under `rule`: "monotone" (one row per sweep λ, the --sweep table) or "cv_1se"
    (one row per fold seed, fitted λ and fold; ascending stop). Uses the current config for
    everything but λ."""
    if rule == "monotone":
        return _monotone(atm_matrix, model)
    if rule == "cv_1se":
        return _cv_1se(atm_matrix, model)
    raise ValueError(f"unknown Stage-2 λ rule {rule!r}")
