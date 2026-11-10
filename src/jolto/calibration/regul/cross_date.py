"""Stage 1 (at config.STAGE2_H) and Stage 2 variants on each cube of config.REGUL_DATES, each with its own history
window, to measure how much Σ moves between dates under each regularisation.

    python -m jolto.calibration.regul.cross_date [--lam 1 3.16 10 | --lam cv_1se]

--lam cv_1se selects λ per date and variant with the production rule (lambda_rule.select_lambda,
under that variant's target and window), then fits at it. Writes results/regul/<out-name>/<date>/
(Stage-1 output, Σ per variant as .npy) and results/regul/<out-name>/summary.csv, one row per date,
variant and λ arm (column lam_arm). Never touches production results.
"""

import argparse
import contextlib
import io
import os

import numpy as np
import pandas as pd

from jolto import config
from jolto.calibration.lambda_rule import select_lambda
from jolto.calibration.pipeline import run_stage1
from jolto.calibration.regul.common import build_problem, calibrate, ff_rms, overrides
from jolto.core.analytics import compute_sigma_structural_metrics

DATES = list(config.REGUL_DATES)
FORMS = ["linear", "hist_exp", "hist_sc2", "hist_reb3"]
OUT = os.path.join(config.RESULTS_DIR, "regul", "xdate")


def vol_file(date, out, drop):
    """The date's cube, minus the dropped expiries (written next to the run's outputs)."""
    src = os.path.join(
        config.DATA_DIR, "history", "vols", f"{config.VOL_PREFIX}{date.replace('-', '')}.csv"
    )
    if not drop:
        return os.path.abspath(src)  # run_stage1 joins a relative path onto DATA_DIR again
    dst = os.path.join(out, "vol_filtered.csv")
    cube = pd.read_csv(src, sep=";")
    cube[~cube["Expiry"].isin(drop)].to_csv(dst, sep=";", index=False)
    return os.path.abspath(dst)


def stage1(date, out, vol):
    s1 = os.path.join(out, config.STAGE1_PARAMS_CSV)
    if os.path.exists(s1):
        return s1
    with (
        overrides(
            VOL_SURFACE_CSV=vol,
            H_GRID=np.array([0.10]),
            RESULTS_DIR=out,
        ),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        run_stage1()
    return s1


def rank_deficit(S, tol=1e-8):
    idx = [i for i in range(1, S.shape[0]) if i != 2]
    rho = S[idx, 0]
    A = S[np.ix_(idx, idx)] - np.outer(rho, rho)
    d = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    return int(np.sum(np.linalg.eigvalsh(A / np.outer(d, d)) < tol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam", nargs="+", default=["1", "3.16", "10"], help="λ values, or cv_1se")
    ap.add_argument("--drop-expiries", nargs="*", default=[], help="expiry labels to drop, e.g. 1M")
    ap.add_argument("--out-name", default="xdate")
    a = ap.parse_args()
    global OUT
    OUT = os.path.join(config.RESULTS_DIR, "regul", a.out_name)
    rows = []
    for date in DATES:
        out = os.path.join(OUT, date)
        os.makedirs(out, exist_ok=True)
        vol = vol_file(date, out, a.drop_expiries)
        s1 = stage1(date, out, vol)
        s1_rmse = float(pd.read_csv(s1)["RMSE_bps"].mean())
        stamp = date.replace("-", "")
        model, atm = build_problem(
            os.path.join(config.DATA_DIR, "history", "curves", f"{config.CURVE_PREFIX}{stamp}.csv"),
            vol,
            s1,
            h=0.10,
        )
        for form in FORMS:
            for first in (1, 3):
                for arm in a.lam:
                    variant = dict(
                        STAGE2_DECAY_FORM=form, STAGE2_PENALTY_FIRST_FWD=first, STAGE2_HIST_END=date
                    )
                    if arm == "cv_1se":
                        with overrides(**variant), contextlib.redirect_stdout(io.StringIO()):
                            lam, _ = select_lambda(atm, model, "cv_1se")
                    else:
                        lam = float(arm)
                    res, cal = calibrate(model, atm, LAMBDA_CURVATURE=lam, **variant)
                    S = res["Sigma_matrix"]
                    name = f"{form}_c{first}_" + ("cv1se" if arm == "cv_1se" else f"lam{lam:g}")
                    np.save(os.path.join(out, f"sigma_{name}.npy"), S)
                    if cal._hist_target is not None:
                        T = np.eye(S.shape[0])
                        T[1:, 1:] = cal._hist_target.cpu().numpy()
                        np.save(os.path.join(out, f"target_{name}.npy"), T)
                    rows.append(
                        {
                            "date": date,
                            "form": form,
                            "first_fwd": first,
                            "lam_arm": arm,
                            "lam": lam,
                            "stage1_rmse": s1_rmse,
                            "rmse": res["final_rmse_bps"],
                            "max_err": res["final_max_err_bps"],
                            "mono": compute_sigma_structural_metrics(S, S.shape[0] - 1)[
                                "mono_violations"
                            ],
                            "rank_deficit": rank_deficit(S),
                            "target_params": (cal.hist_target_info or {}).get("params"),
                            "rho1": float(S[1, 0]),
                        }
                    )
                    print(
                        f"{date} {name:22s} RMSE {res['final_rmse_bps']:.3f} mono {rows[-1]['mono']} "
                        f"rankdef {rows[-1]['rank_deficit']}",
                        flush=True,
                    )
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "summary.csv"), index=False)

    # stability: RMS change of the forward block (forwards 3..31) between consecutive dates
    df = pd.DataFrame(rows)
    drift = []
    for (form, first, arm), _ in df.groupby(["form", "first_fwd", "lam_arm"]):
        name = f"{form}_c{first}_" + ("cv1se" if arm == "cv_1se" else f"lam{float(arm):g}")
        S = [np.load(os.path.join(OUT, d, f"sigma_{name}.npy")) for d in DATES]
        steps = [ff_rms(S[k + 1], S[k], 3) for k in range(len(S) - 1)]
        tfile = [os.path.join(OUT, d, f"target_{name}.npy") for d in DATES]
        tsteps = (
            [ff_rms(np.load(tfile[k + 1]), np.load(tfile[k]), 3) for k in range(len(S) - 1)]
            if all(os.path.exists(f) for f in tfile)
            else [0.0]
        )
        drift.append(
            {
                "form": form,
                "first_fwd": first,
                "lam_arm": arm,
                "drift_mean": float(np.mean(steps)),
                "drift_max": float(np.max(steps)),
                "target_drift_mean": float(np.mean(tsteps)),
            }
        )
    drift = pd.DataFrame(drift)
    df = df.merge(drift, on=["form", "first_fwd", "lam_arm"])
    df.to_csv(os.path.join(OUT, "summary.csv"), index=False)
    print(
        df.groupby(["form", "first_fwd", "lam_arm"])
        .agg(
            lam=("lam", "mean"),
            rmse=("rmse", "mean"),
            mono=("mono", "mean"),
            rankdef=("rank_deficit", "mean"),
            drift=("drift_mean", "first"),
            target_drift=("target_drift_mean", "first"),
        )
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
