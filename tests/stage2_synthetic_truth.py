"""Benchmark: recover a known Σ from synthetic Stage-2 quotes.

    python tests/stage2_synthetic_truth.py --truth prod --noise 0 1 \
        [--form linear hist_sc2] [--first 1 3] [--lam 1 3.16 10] [--hist-end YYYY-MM-DD] [--tag NAME]

Form, first column, λ grid and window end default to the production config.

Production marginals and inputs fixed. For each truth Σ*:
    y = HF mapped-SMM vols of Σ* on Sobol seed 4242 (calibration uses seed 42: no inverse crime)
        + Gaussian noise of s bp (seed 7 + 1000·s).
Each (form, first column, λ) recalibrates Stage 2 on y and is scored on:
    rmse_syn (fit to y), rmse_clean (to the noiseless y), sigma_rms (Σ̂ vs Σ* on forwards 3..31),
    sigma_rms_all (forwards 1..31), mono violations, rank deficit of the partial-correlation block.
Truths:
    prod      the production Σ (production target at the production λ of the fit summary);
    hist_sc2  the Σ calibrated with the hist_sc2 target, forwards 1–2 excluded, at the same λ;
    hist_raw  the representable Σ closest to the raw weekly historical correlation (no prior).
Writes results/regul/synthetic/<tag>.csv.
"""

import argparse
import os
import time

import numpy as np
import pandas as pd
import torch
from jolto.calibration.regul.common import (
    PROD,
    build_problem,
    calibrate,
    ff_rms,
    mc_cache,
    price,
    prod_lambda,
    setup_of,
    sigma_of,
)
from scipy.optimize import least_squares

from jolto import config
from jolto.calibration.corr_target import drop_glitches, embed_partial, historical_correlation
from jolto.core.analytics import (
    build_rapisarda_correlation_matrix,
    compute_sigma_structural_metrics,
)
from jolto.core.market_data import load_forward_history

OUT = os.path.join(config.RESULTS_DIR, "regul", "synthetic")


def rank_deficit(S, tol=1e-8):
    idx = [i for i in range(1, S.shape[0]) if i != 2]
    rho = S[idx, 0]
    A = S[np.ix_(idx, idx)] - np.outer(rho, rho)
    d = np.sqrt(np.clip(np.diag(A), 1e-300, None))
    return int(np.sum(np.linalg.eigvalsh(A / np.outer(d, d)) < tol))


def project_to_angles(cal, st, C_ff):
    """Free angles whose Σ is closest to the forward block C_ff (forwards 3..31, least squares)."""
    o = torch.tensor(cal.omega, dtype=cal.model.dtype)
    r, c = st["r_idx"].cpu(), st["c_idx"].cpu()
    n = o.shape[0]
    ii, jj = np.tril_indices(n, -1)
    m = (ii >= 3) & (jj >= 3)
    ii, jj = ii[m], jj[m]
    tgt = torch.tensor(C_ff[ii - 1, jj - 1], dtype=cal.model.dtype)

    def resid(p):
        S = build_rapisarda_correlation_matrix(o.index_put((r, c), torch.tensor(p, dtype=o.dtype)))
        return (S[ii, jj] - tgt).numpy()

    def jac(p):
        f = lambda q: build_rapisarda_correlation_matrix(o.index_put((r, c), q))[ii, jj]  # noqa: E731
        return torch.autograd.functional.jacobian(f, torch.tensor(p, dtype=o.dtype)).numpy()

    sol = least_squares(resid, np.full(len(r), 0.8), jac=jac, bounds=(1e-4, np.pi - 1e-4))
    return o.index_put((r, c), torch.tensor(sol.x, dtype=o.dtype)).numpy()


def truth_omega(name, model, atm, hist_end, cal, st):
    if name == "prod":
        res, _ = calibrate(model, atm, LAMBDA_CURVATURE=prod_lambda())
        return res["omega_matrix"]
    if name == "hist_sc2":
        res, _ = calibrate(
            model,
            atm,
            STAGE2_DECAY_FORM="hist_sc2",
            STAGE2_PENALTY_FIRST_FWD=3,
            STAGE2_HIST_END=hist_end,
            LAMBDA_CURVATURE=prod_lambda(),
        )
        return res["omega_matrix"]
    if name == "hist_raw":
        panel, _ = drop_glitches(
            load_forward_history(
                config.STAGE2_HIST_CURVE_DIR,
                config.GRID_STEP_SIZE,
                config.MAX_MATURITY,
                end=hist_end,
            )
        )
        start = (pd.Timestamp(hist_end) - pd.DateOffset(years=config.STAGE2_HIST_YEARS)).strftime(
            "%Y-%m-%d"
        )
        C, _ = historical_correlation(panel, start, hist_end, config.STAGE2_HIST_STEP)
        rho = cal.model.rhos.detach().cpu().numpy()
        # the raw estimate is noisy; embed on ρ after a symmetric eigen-clip so it is a valid target
        w, V = np.linalg.eigh(C)
        C = (V * np.clip(w, 1e-6, None)) @ V.T
        d = np.sqrt(np.diag(C))
        C = C / np.outer(d, d)
        return project_to_angles(cal, st, embed_partial(C, rho))
    raise ValueError(name)


def synthetic_atm(atm, st, y):
    syn = atm.copy()
    for e, t, v in zip(st["global_exp"], st["global_ten"], y):
        r = syn.index[np.argmin(np.abs(syn.index.values - e))]
        c = syn.columns[np.argmin(np.abs(syn.columns.values - t))]
        syn.loc[r, c] = v
    return syn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", nargs="+", default=["prod"])
    ap.add_argument("--noise", nargs="+", type=float, default=[0.0])
    ap.add_argument("--form", nargs="+", default=[config.STAGE2_DECAY_FORM])
    ap.add_argument("--first", nargs="+", type=int, default=[config.STAGE2_PENALTY_FIRST_FWD])
    ap.add_argument("--lam", nargs="+", type=float, default=list(config.STAGE2_CV_LAMBDA_GRID))
    ap.add_argument("--hist-end", default=config.STAGE2_HIST_END)
    ap.add_argument("--tag", default="run")
    a = ap.parse_args()

    model, atm = build_problem(PROD["disc"], PROD["vol"], PROD["stage1"])
    _, cal0 = calibrate(model, atm)
    st = setup_of(cal0)
    n_steps = int(np.ceil(st["global_exp"].max() / st["dt_val"]))
    data_drivers = mc_cache(cal0, n_steps, seed=4242)

    rows = []
    os.makedirs(OUT, exist_ok=True)
    for truth in a.truth:
        om_true = truth_omega(truth, model, atm, a.hist_end, cal0, st)
        S_true = sigma_of(om_true)
        y_clean = price(cal0, st, S_true, data_drivers)
        for s in a.noise:
            eps = np.random.default_rng(7 + int(1000 * s)).standard_normal(len(y_clean))
            y = y_clean + s * 1e-4 * eps
            syn = synthetic_atm(atm, st, y)
            for form in a.form:
                for first in a.first:
                    for lam in a.lam:
                        t0 = time.time()
                        res, _ = calibrate(
                            model,
                            syn,
                            STAGE2_DECAY_FORM=form,
                            STAGE2_PENALTY_FIRST_FWD=first,
                            STAGE2_HIST_END=a.hist_end,
                            LAMBDA_CURVATURE=lam,
                        )
                        S = res["Sigma_matrix"]
                        v = price(cal0, st, S, st["global_cached_drivers"])
                        row = {
                            "truth": truth,
                            "noise_bp": s,
                            "form": form,
                            "first_fwd": first,
                            "lam": lam,
                            "rmse_syn": float(np.sqrt(np.mean((v - y) ** 2)) * 1e4),
                            "rmse_clean": float(np.sqrt(np.mean((v - y_clean) ** 2)) * 1e4),
                            "sigma_rms": ff_rms(S, S_true, 3),
                            "sigma_rms_all": ff_rms(S, S_true, 1),
                            "mono": compute_sigma_structural_metrics(S, S.shape[0] - 1)[
                                "mono_violations"
                            ],
                            "rank_deficit": rank_deficit(S),
                            "truth_rank_deficit": rank_deficit(S_true),
                            "secs": round(time.time() - t0, 1),
                        }
                        rows.append(row)
                        print(
                            f"{truth:8s} s={s:<4} {form:9s} c{first} λ={lam:<6} fit {row['rmse_syn']:.2f} "
                            f"clean {row['rmse_clean']:.2f} Σrms {row['sigma_rms']:.4f} "
                            f"mono {row['mono']} rankdef {row['rank_deficit']} ({row['secs']}s)",
                            flush=True,
                        )
                        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{a.tag}.csv"), index=False)
    print(f"-> {os.path.join(OUT, a.tag + '.csv')}")


if __name__ == "__main__":
    main()
