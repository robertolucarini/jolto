"""Bermudan and CMS-spread prices under Σ(λ) for each date's cv_1se pick and its two grid
neighbours, per date of config.REGUL_DATES, on one seed (common random numbers across λ). Stage-1 outputs from
regul.lambda_xdate.

    python -m jolto.calibration.regul.lambda_price

Writes results/regul/lambda_stability/price/<date>/<λ>/ (the Σ priced) and
results/regul/lambda_stability/price_effect.csv, one row per date and role. Run from the repo root.
"""

import contextlib
import dataclasses
import io
import os
import shutil
import time

import numpy as np
import pandas as pd

from jolto import config, val_config
from jolto.calibration.regul.common import build_problem, calibrate
from jolto.calibration.regul.lambda_xdate import OUT, XD, pick_and_neighbours, picks


def main():
    # the valuation layer sits above calibration; imported here so `regul` adds no import-time cycle
    from jolto.api import get_price
    from jolto.api.assembly import load_valuation_inputs
    from jolto.api.trade_specs import BermudanSwaptionSpec, CmsSpreadSpec

    base = load_valuation_inputs(val_config)
    berm_spec = BermudanSwaptionSpec.from_val_config(val_config)
    spread_spec = CmsSpreadSpec.from_val_config(val_config)
    berm_opts = dataclasses.replace(
        get_price.BermudanRunOpts.from_val_config(val_config), verbose=False
    )
    cms_opts = get_price.CmsRunOpts.from_val_config(val_config)

    rows = []
    for date, lam_pick in picks().items():
        s = date.replace("-", "")
        curve = f"history/curves/{config.CURVE_PREFIX}{s}.csv"
        s1 = os.path.join(XD, date, config.STAGE1_PARAMS_CSV)
        model, atm = build_problem(
            os.path.join(config.DATA_DIR, curve),
            os.path.join(config.DATA_DIR, "history", "vols", f"{config.VOL_PREFIX}{s}.csv"),
            s1,
            h=0.10,
        )
        for role, lam in pick_and_neighbours(lam_pick).items():
            d = os.path.join(OUT, "price", date, f"lam{lam:g}")
            os.makedirs(d, exist_ok=True)
            shutil.copy(s1, d)
            res, _ = calibrate(model, atm, LAMBDA_CURVATURE=lam, STAGE2_HIST_END=date)
            pd.DataFrame(np.asarray(res["Sigma_matrix"])).to_csv(
                os.path.join(d, base.sigma_tpl.format(h=base.H)), header=False, index=False
            )
            inputs = dataclasses.replace(base, results_dir=d, discount_name=curve)
            t0 = time.time()
            with contextlib.redirect_stdout(io.StringIO()):
                b = get_price.bermudan_swaptions(
                    berm_spec, inputs, with_greeks=False, run_opts=berm_opts
                )
                c = get_price.cms_spread_option(
                    spread_spec, inputs, with_greeks=False, run_opts=cms_opts
                )
            rows.append(
                {
                    "date": date,
                    "role": role,
                    "lam": lam,
                    "bermudan_bps": b.price_bps,
                    "cms_spread_bps": c.price_bps,
                    "secs": round(time.time() - t0, 1),
                }
            )
            print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    for col in ("bermudan_bps", "cms_spread_bps"):
        ref = df[df["role"] == "pick"].set_index("date")[col]
        df[col + "_rel_vs_pick_pct"] = 100 * (df[col] / df["date"].map(ref) - 1)
    df.to_csv(os.path.join(OUT, "price_effect.csv"), index=False)
    pd.set_option("display.width", 250)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
