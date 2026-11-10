"""Report: historical correlation targets across windows, change frequencies and cube dates.

    python -m jolto.calibration.regul.targets   (needs the dated curves under data/history)

Writes results/regul/targets.json, the window and frequency evidence behind the hist_* targets.
The 1Y and 2Y windows end on config.STAGE2_HIST_END, the production window end.
"""

import json
import os

import numpy as np
import pandas as pd

from jolto import config
from jolto.calibration.corr_target import (
    FAMILIES,
    drop_glitches,
    embed_partial,
    family_matrix,
    fit_family,
    historical_correlation,
)
from jolto.core.market_data import load_forward_history

HIST = os.path.join(config.DATA_DIR, "history")
OUT = os.path.join(config.RESULTS_DIR, "regul")
END = config.STAGE2_HIST_END
CUBE_DATES = list(config.REGUL_DATES)
SIGMA_PROD = os.path.join(config.RESULTS_DIR, config.STAGE2_CORR_CSV_TPL.format(h=config.STAGE2_H))


def spectrum(C):
    ev = np.sort(np.linalg.eigvalsh(C))[::-1]
    share = np.cumsum(ev) / ev.sum()
    return {
        "n90": int(np.searchsorted(share, 0.90) + 1),
        "n99": int(np.searchsorted(share, 0.99) + 1),
        "min_eig": float(ev[-1]),
    }


def describe(C, n):
    adj = np.array([C[i, i + 1] for i in range(C.shape[0] - 1)])
    return {
        "n_changes": n,
        "adj_corr_F3_F13": [round(float(x), 3) for x in adj[2:13]],
        "adj_corr_min_F14_F31": round(float(adj[13:].min()), 3),
        "corr_3_31": round(float(C[2, 30]), 3),
        "corr_10_20": round(float(C[9, 19]), 3),
        **spectrum(C),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    panel, glitches = drop_glitches(
        load_forward_history(
            os.path.join(HIST, "curves"), config.GRID_STEP_SIZE, config.MAX_MATURITY
        )
    )
    # Stage-1 column of the production Σ as the stand-in ρ for the embedded fits
    rho = np.loadtxt(SIGMA_PROD, delimiter=",")[0, 1:]
    sigma_prod = np.loadtxt(SIGMA_PROD, delimiter=",")[1:, 1:]

    start_1y, start_2y = (
        (pd.Timestamp(END) - pd.DateOffset(years=y)).strftime("%Y-%m-%d") for y in (1, 2)
    )
    variants = {
        "daily_1y": (panel, start_1y, END, 1),
        "weekly_1y": (panel, start_1y, END, 5),
        "daily_2y": (panel, start_2y, END, 1),
        "weekly_2y": (panel, start_2y, END, 5),
    }
    for d in CUBE_DATES[:-1]:
        start = (pd.Timestamp(d) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
        variants[f"daily_1y_to_{d}"] = (panel, start, d, 1)
        variants[f"weekly_1y_to_{d}"] = (panel, start, d, 5)

    M = rho.size
    ii, jj = np.tril_indices(M - 2, -1)
    ii, jj = ii + 2, jj + 2  # forwards 3..31
    lin = np.maximum(
        1.0 - config.STAGE2_DECAY_SLOPE * np.abs(np.subtract.outer(np.arange(M), np.arange(M))), 0.0
    )

    report = {"glitches_dropped": [d.strftime("%Y-%m-%d") for d in glitches], "variants": {}}
    for key, (panel, start, end, step) in variants.items():
        C, n = historical_correlation(panel, start, end, step)
        entry = {"window": [start, end], "step": step, "raw": describe(C, n), "fits": {}}
        entry["raw"]["rms_to_production_sigma"] = float(
            np.sqrt(np.mean((C - sigma_prod)[ii, jj] ** 2))
        )
        entry["raw"]["rms_to_linear_target"] = float(np.sqrt(np.mean((C - lin)[ii, jj] ** 2)))
        entry["raw"]["mean_by_separation"] = {
            d: round(float(np.mean([C[i, i - d] for i in range(2 + d, M)])), 3)
            for d in (1, 2, 3, 5, 10, 15, 20, 25, 28)
        }
        for name in FAMILIES:
            plain = fit_family(name, C)
            emb = fit_family(name, C, rho=rho)
            T = embed_partial(family_matrix(name, emb["params"], M), rho)
            emb["rms_to_production_sigma"] = float(np.sqrt(np.mean((T - sigma_prod)[ii, jj] ** 2)))
            entry["fits"][name] = {"plain": plain, "embedded": emb}
        report["variants"][key] = entry
        f = entry["fits"]
        print(
            f"{key:28s} n={n:3d} n90={entry['raw']['n90']:2d} "
            + " ".join(
                f"{k}:{np.round(v['embedded']['params'], 4)} rmse={v['embedded']['rmse']:.3f}"
                for k, v in f.items()
            )
        )

    with open(os.path.join(OUT, "targets.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"glitches dropped: {report['glitches_dropped']} -> {OUT}/targets.json")


if __name__ == "__main__":
    main()
