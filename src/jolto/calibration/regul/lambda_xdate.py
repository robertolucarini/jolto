"""cv_1se λ pick on each cube of config.REGUL_DATES: dated cube + same-date curve + window ending on the date,
Stage 1 at config.STAGE2_H only, defaults otherwise, through the Stage-1 and Stage-2 pipelines.

    python -m jolto.calibration.regul.lambda_xdate

Writes results/regul/lambda_stability/xdate/<date>/ (Stage-1 and Stage-2 outputs, logs) and
results/regul/lambda_stability/pick_summary.csv. regul.lambda_sigma and regul.lambda_price read
both. Run from the repo root.
"""

import contextlib
import json
import os
import time

import numpy as np
import pandas as pd

from jolto import config
from jolto.calibration.pipeline import run_stage1, run_stage2
from jolto.calibration.regul.common import overrides

DATES = list(config.REGUL_DATES)
OUT = os.path.join(config.RESULTS_DIR, "regul", "lambda_stability")
XD = os.path.join(OUT, "xdate")


def pick_and_neighbours(lam):
    """{role: λ} for the pick and its neighbours on the cv_1se grid (the grid floor below 0.1)."""
    grid = sorted(config.STAGE2_CV_LAMBDA_GRID)
    i = int(np.argmin(np.abs(np.array(grid) - lam)))
    roles = {"pick": grid[i]}
    if i > 0:
        roles["lower"] = grid[i - 1]
    if i + 1 < len(grid):
        roles["upper"] = grid[i + 1]
    return roles


def picks():
    """{date: selected λ} from pick_summary.csv."""
    df = pd.read_csv(os.path.join(OUT, "pick_summary.csv"))
    return dict(zip(df["date"], df["lambda"].astype(float)))


def _run(stage, log):
    with open(log, "w") as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        stage()


def main():
    rows = []
    for date in DATES:
        s = date.replace("-", "")
        out = os.path.join(XD, date)
        os.makedirs(out, exist_ok=True)
        with overrides(
            VOL_SURFACE_CSV=f"history/vols/{config.VOL_PREFIX}{s}.csv",
            DISCOUNT_CSV=f"history/curves/{config.CURVE_PREFIX}{s}.csv",
            STAGE2_HIST_END=date,
            RESULTS_DIR=out,
            H_GRID=np.array([0.10]),
        ):
            t0 = time.time()
            _run(run_stage1, os.path.join(out, "stage1.log"))
            t1 = time.time()
            _run(run_stage2, os.path.join(out, "stage2.log"))
            t2 = time.time()
            fit = json.load(open(os.path.join(out, config.STAGE2_FIT_SUMMARY_JSON)))
            cv = pd.read_csv(os.path.join(out, config.STAGE2_LAMBDA_CV_CSV_TPL.format(h=0.10)))
        s1 = pd.read_csv(os.path.join(out, config.STAGE1_PARAMS_CSV))
        curve = cv.groupby("lam")[["cv_rmse", "se"]].first()
        row = {
            "date": date,
            "lambda": fit["lambda_selected"],
            "stage1_rmse": round(float(s1["RMSE_bps"].mean()), 3),
            "stage2_rmse": round(float(fit["final_rmse_bps"]), 3),
            "cv": "  ".join(f"{lam:g}:{r.cv_rmse:.3f}±{r.se:.3f}" for lam, r in curve.iterrows()),
            "s1_s": round(t1 - t0, 1),
            "s2_s": round(t2 - t1, 1),
        }
        rows.append(row)
        print(row, flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "pick_summary.csv"), index=False)
    print(pd.DataFrame(rows).drop(columns="cv").to_string(index=False))


if __name__ == "__main__":
    main()
