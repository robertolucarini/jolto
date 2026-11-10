"""K-fold cross-validation of the Stage-2 regularisation on held-out ATM pairs.

    python tests/stage2_cv.py [--form linear hist_exp] [--first 1 3] \
        [--lam 0.1 0.316 1] [--folds 5] [--seed 0] [--hist-end YYYY-MM-DD] [--tag NAME]

The Stage-2 quotes are the ATM swaptions whose expiry + tenor falls on the forward grid; row i of Σ
is fitted to the quotes ending at T_i. Folds are stratified by row (round-robin from a random offset),
so every row keeps at least one quote and its angles stay in the optimisation. Each fold recalibrates
Stage 2 without its quotes, then prices them with the high-fidelity pricer used in calibration. Every
quote is predicted exactly once; the score is the RMSE of the held-out predictions.
Production inputs and Stage 1; form, first column, λ grid, folds and window end default to the
production config. Scores the full grid, so it is the reference for the ascending stop of cv_1se.
Writes results/regul/cv/<tag>.csv (one row per variant, λ and fold).
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
from jolto.calibration.regul.common import PROD, build_problem, calibrate, price, setup_of

from jolto import config

OUT = os.path.join(config.RESULTS_DIR, "regul", "cv")


def make_folds(st, grid_T, k, seed):
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
    for r in np.unique(rows):  # each row keeps ≥ 1 quote in every training set
        assert all(np.any((rows == r) & (fold != f)) for f in range(k)), r
    return fold


def drop_pairs(atm, st, mask):
    out = atm.copy()
    for e, t in zip(st["global_exp"][mask], st["global_ten"][mask]):
        out.loc[
            out.index[np.argmin(np.abs(out.index.values - e))],
            out.columns[np.argmin(np.abs(out.columns.values - t))],
        ] = np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--form", nargs="+", default=[config.STAGE2_DECAY_FORM])
    ap.add_argument("--first", nargs="+", type=int, default=[config.STAGE2_PENALTY_FIRST_FWD])
    ap.add_argument("--lam", nargs="+", type=float, default=list(config.STAGE2_CV_LAMBDA_GRID))
    ap.add_argument("--folds", type=int, default=config.STAGE2_CV_FOLDS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hist-end", default=config.STAGE2_HIST_END)
    ap.add_argument("--tag", default="cv")
    a = ap.parse_args()

    model, atm = build_problem(PROD["disc"], PROD["vol"], PROD["stage1"])
    _, cal0 = calibrate(model, atm)
    st = setup_of(cal0)
    y = st["global_vol"]
    fold = make_folds(st, cal0.grid_T, a.folds, a.seed)
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for form in a.form:
        for first in a.first:
            for lam in a.lam:
                pred = np.full(len(y), np.nan)
                for f in range(a.folds):
                    t0 = time.time()
                    res, _ = calibrate(
                        model,
                        drop_pairs(atm, st, fold == f),
                        STAGE2_DECAY_FORM=form,
                        STAGE2_PENALTY_FIRST_FWD=first,
                        STAGE2_HIST_END=a.hist_end,
                        LAMBDA_CURVATURE=lam,
                    )
                    v = price(cal0, st, res["Sigma_matrix"], st["global_cached_drivers"])
                    held = fold == f
                    pred[held] = v[held]
                    rows.append(
                        {
                            "form": form,
                            "first_fwd": first,
                            "lam": lam,
                            "fold": f,
                            "n_held": int(held.sum()),
                            "train_rmse": float(np.sqrt(np.mean(((v - y)[~held]) ** 2)) * 1e4),
                            "test_rmse": float(np.sqrt(np.mean(((v - y)[held]) ** 2)) * 1e4),
                            "secs": round(time.time() - t0, 1),
                        }
                    )
                cv = float(np.sqrt(np.mean((pred - y) ** 2)) * 1e4)
                for r in rows[-a.folds :]:
                    r["cv_rmse"] = cv
                print(f"{form:9s} c{first} λ={lam:<6} CV RMSE {cv:.3f} bp", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{a.tag}.csv"), index=False)
    print(f"-> {os.path.join(OUT, a.tag + '.csv')}")


if __name__ == "__main__":
    main()
