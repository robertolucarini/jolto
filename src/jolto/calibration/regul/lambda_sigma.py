"""Effect on Σ of the neighbouring λ (one grid step either side of each date's cv_1se pick), per
date of config.REGUL_DATES, on the Stage-1 outputs of regul.lambda_xdate.

    python -m jolto.calibration.regul.lambda_sigma

Writes results/regul/lambda_stability/sigma_effect.csv, one row per date and role (pick, lower,
upper). check_vs_pick_run is max |Σ(pick) − Σ of the regul.lambda_xdate run| (0 up to rounding).
Run from the repo root.
"""

import os

import numpy as np
import pandas as pd
import torch

from jolto import config
from jolto.calibration.pricers import _compute_mapped_smm_parameters
from jolto.calibration.regul.common import build_problem, calibrate, ff_rms
from jolto.calibration.regul.lambda_xdate import DATES, OUT, XD, pick_and_neighbours, picks

# ATM swaption grid for the mapped normal vol (expiry, tenor), expiry + tenor <= 30
SWP = [(e, t) for e in (1, 2, 5, 10, 20) for t in (1, 2, 5, 10, 20) if e + t <= 30]


def to_np(S):
    return S.detach().cpu().numpy() if torch.is_tensor(S) else np.asarray(S)


def gap_mean(S, gap):
    n = S.shape[0] - 1  # forwards at Σ indices 1..n
    return float(np.mean([S[i, i + gap] for i in range(1, n - gap + 1)]))


def main():
    rows, prev = [], None
    for date, lam_pick in picks().items():
        assert date in DATES, date
        s = date.replace("-", "")
        model, atm = build_problem(
            os.path.join(config.DATA_DIR, "history", "curves", f"{config.CURVE_PREFIX}{s}.csv"),
            os.path.join(config.DATA_DIR, "history", "vols", f"{config.VOL_PREFIX}{s}.csv"),
            os.path.join(XD, date, config.STAGE1_PARAMS_CSV),
            h=0.10,
        )
        roles = pick_and_neighbours(lam_pick)
        sig, alpha, by_role = {}, {}, {}
        for role, lam in roles.items():
            res, _ = calibrate(model, atm, LAMBDA_CURVATURE=lam, STAGE2_HIST_END=date)
            S = to_np(res["Sigma_matrix"])
            sig[role] = S
            a, _ = _compute_mapped_smm_parameters(
                model,
                torch.tensor(S, dtype=model.dtype, device=model.device),
                [e for e, _ in SWP],
                [t for _, t in SWP],
            )
            alpha[role] = to_np(torch.stack(list(a)) if isinstance(a, (list, tuple)) else a) * 1e4
            by_role[role] = {
                "date": date,
                "role": role,
                "lam": lam,
                "rmse": res["final_rmse_bps"],
                "gap25_mean": gap_mean(S, 25),
            }
        prod = pd.read_csv(
            os.path.join(XD, date, config.STAGE2_CORR_CSV_TPL.format(h=0.10)), header=None
        )
        by_role["pick"].update(
            {
                "check_vs_pick_run": float(np.max(np.abs(prod.to_numpy() - sig["pick"]))),
                "quarter_ff_rms": ff_rms(sig["pick"], prev, 3) if prev is not None else np.nan,
            }
        )
        for role in roles.keys() - {"pick"}:
            d = sig[role] - sig["pick"]
            da = alpha[role] - alpha["pick"]
            by_role[role].update(
                {
                    "ff_rms_vs_pick": ff_rms(sig[role], sig["pick"], 3),
                    "ff_max_vs_pick": float(np.max(np.abs(d[1:, 1:]))),
                    "dalpha_smm_mean_bp": float(np.mean(np.abs(da))),
                    "dalpha_smm_max_bp": float(np.max(np.abs(da))),
                }
            )
        rows += [by_role[r] for r in roles]
        prev = sig["pick"]
        print(date, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "sigma_effect.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
