"""Diagnostic: how many angle directions the Stage-2 quotes identify.

    python tests/stage2_identifiability.py

At the production solution (production target, λ = lambda_selected of results/stage2_fit_summary.json)
and at the unregularised one (λ → 0), on the production inputs:
1. SVD of the high-fidelity Jacobian d(vol)/d(angles) (bp per radian), quotes × free angles.
2. Directions identified above the Monte Carlo noise: σ_k·δθ ≥ s_MC, with s_MC the per-quote RMS
   difference between two Sobol seeds, for angle moves δθ of 0.01 and 0.05 rad.
3. The share of the residual lying in the identified directions, and the RMSE a Gauss-Newton step
   restricted to them would leave (a floor in the identified directions is not model error).
4. Identification strength per Σ row (column norms of J grouped by row).
Writes results/regul/identifiability.json.
"""

import json
import os

import numpy as np
from jolto.calibration.regul.common import (
    PROD,
    build_problem,
    calibrate,
    free_angles,
    mc_cache,
    price,
    prod_lambda,
    setup_of,
)

from jolto import config

OUT = os.path.join(config.RESULTS_DIR, "regul", "identifiability.json")


def analyse(cal, res, s_mc):
    st = setup_of(cal)
    p = free_angles(res["omega_matrix"], st)
    v, J = cal._get_global_hf_vols_and_jac(
        p,
        (st["r_idx"], st["c_idx"]),
        st["global_exp"],
        st["global_ten"],
        st["global_str"],
        st["dt_val"],
        st["n_paths_val"],
        st["global_cached_drivers"],
        calculate_jac=True,
        smm_cache=st["global_smm_cache"],
    )
    J = J * 1e4  # bp per radian
    r = (np.asarray(v) - st["global_vol"]) * 1e4
    U, s, _ = np.linalg.svd(J, full_matrices=False)
    m = len(r)
    out = {
        "rmse_bps": float(np.sqrt(np.mean(r**2))),
        # angles near 0 or π have sin ω ≈ 0, so the rows below them lose their derivative
        "angles_near_boundary": {f"{t:g}": int(np.sum(np.sin(p) < t)) for t in (0.01, 0.05, 0.1)},
        "sv_bp_per_rad": {q: float(np.quantile(s, q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)},
        "rank_rel": {f"{t:g}": int(np.sum(s >= t * s[0])) for t in (1e-1, 1e-2, 1e-3, 1e-4)},
    }
    for dtheta in (0.01, 0.05):
        k = int(np.sum(s * dtheta >= s_mc))
        proj = U[:, :k].T @ r
        out[f"identified_dtheta_{dtheta}"] = {
            "n_directions": k,
            "residual_share_in_identified": float(proj @ proj / (r @ r)),
            "rmse_after_gn_step_bps": float(np.sqrt(max(r @ r - proj @ proj, 0.0) / m)),
        }
    rows = st["r_idx"].cpu().numpy()
    col_norm = np.linalg.norm(J, axis=0)
    out["row_strength_bp_per_rad"] = {
        int(i): round(float(np.sqrt(np.sum(col_norm[rows == i] ** 2))), 1) for i in np.unique(rows)
    }
    out["quotes_per_row"] = {
        int(i): int(np.sum(np.abs(st["global_exp"] + st["global_ten"] - cal.grid_T[i]) < 1e-4))
        for i in np.unique(rows)
    }
    return out, st


def main():
    lam_prod = prod_lambda()
    model, atm = build_problem(PROD["disc"], PROD["vol"], PROD["stage1"])
    report = {"production_lambda": lam_prod}
    for tag, lam in (("production", lam_prod), ("lambda_floor", 1e-6)):
        res, cal = calibrate(model, atm, LAMBDA_CURVATURE=lam)
        if "s_mc_bps" not in report:
            st = setup_of(cal)
            n_steps = int(np.ceil(st["global_exp"].max() / st["dt_val"]))
            v42 = price(cal, st, res["Sigma_matrix"], st["global_cached_drivers"])
            v2 = price(cal, st, res["Sigma_matrix"], mc_cache(cal, n_steps, seed=4242))
            report["s_mc_bps"] = float(np.sqrt(np.mean((v42 - v2) ** 2)) * 1e4 / np.sqrt(2))
        report[tag], _ = analyse(cal, res, report["s_mc_bps"])
        r = report[tag]
        print(
            f"{tag}: RMSE {r['rmse_bps']:.3f} bp | σ max {r['sv_bp_per_rad'][1.0]:.0f}, median "
            f"{r['sv_bp_per_rad'][0.5]:.1f} bp/rad | rank rel 1e-2 {r['rank_rel']['0.01']} | "
            + " | ".join(
                f"δθ={d}: {r[f'identified_dtheta_{d}']['n_directions']} dirs, residual share "
                f"{r[f'identified_dtheta_{d}']['residual_share_in_identified']:.2f}, GN-step RMSE "
                f"{r[f'identified_dtheta_{d}']['rmse_after_gn_step_bps']:.2f}"
                for d in (0.01, 0.05)
            )
        )
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(report, fh, indent=1)
    print(f"s_MC per quote {report['s_mc_bps']:.3f} bp -> {OUT}")


if __name__ == "__main__":
    main()
