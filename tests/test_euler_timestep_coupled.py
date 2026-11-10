"""
W-coupled Euler time-stepping bias (C3).

An uncoupled C3 ladder cannot resolve C3: the across-grid difference is non-CRN, so the
differenced option-payoff variance (present even on the driftless deflator-free gate) swamps C3.
Every grid level is therefore driven from the same fine Brownian path,
so the payoff/deflator randomness is common-mode and cancels in the level-difference (rough-vol
MLMC coupling; McCrickerd–Pakkanen).

Construction
------------
Nested ladder dt ∈ {1/24, 1/48, 1/96} (2× refinements), anchored on the production grid 1/24.
- One fine path dz_fine at 1/96, iid N(0, dt_fine) per (step, factor).
- Coarse increments by block-summing fine increments (Brownian increments add): dz_level[:,k,:] =
  dz_fine[:, r·k:r·k+r, :].sum(time-block), r = n_steps_fine / n_steps_level. Variance r·dt_fine = dt_level.
- Each level: inject dz_level via the simulate_forward_curve dz_override seam and run the production
  frozen κ=config.HYBRID_KAPPA scheme. The same W is convolved against each grid's kernel.

Partial coupling (κ=1): the base dz (bulk variance: the dz@Kᵀ vol convolution + the rate increments) is
coupled; the hybrid near-lag correction stays internal/uncoupled. The near-lag is a single near-diagonal
covariance residual → its standalone payoff-variance contribution is small, so the coupled-difference SE
should still collapse while the quantity measured is the production κ=1 scheme. (κ=0 would be invalid: it
measures the κ=0 kernel's dt-convergence, the bias κ=1 removes, not the forward time-stepping.)
Gate: coupled gate SE ≪ the uncoupled SE. If the residual near-lag variance keeps the SE too
large, the near-lag draw can be coupled as well (share its standard normal across grids, scaled per
grid by each grid's residual-covariance Cholesky).

Read-off
--------
Coupled differences Δ₁=price(1/24)−price(1/48), Δ₂=price(1/48)−price(1/96) are CRN-resolved (small SE).
Measured rate p=log₂(Δ₁/Δ₂) (no rate-1 assumption); C3(1/24)=Δ₁+Δ₂+Δ₂/(2ᵖ−1). Unphysical ratio →
conservative two-grid bound |price(1/24)−price(1/96)|, flagged. Then total = C1 + C2 + C3 (orthogonality
caveat: C3 absorbs only the κ=1 kernel's residual dt-refinement, not the κ=0 kernel bias, so
nothing is double-counted). The floor is absolute and the SE diagnostic, the result a reported
measurement; only seam-transparency is a hard assert.

Swap helpers come from bermudan_swaption.py and tests/_common.py. The engine is driven through
simulate_forward_curve's dz_override argument (default None ⇒ byte-identical). Price-only no_grad; N=2048
suffices because coupling collapses the difference SE; one full path held at a time.

Usage:  python tests/test_euler_timestep_coupled.py
Outputs: results/euler_timestep_bias_coupled.csv · results/discretisation_bias_total.csv (re-assembled)
"""

import math
import os
import time

import numpy as np
import pandas as pd
import torch

from jolto import config, val_config
from jolto.api.assembly import build_calibrated_model, load_valuation_inputs
from jolto.calibration.pricers import INV_SQRT_2PI, bachelier_iv_newton
from jolto.core.analytics import generate_sobol_bridge_increments
from jolto.valuation.bermudan_swaption import _swap_and_deflated_annuity
from _common import _grid_label, _stage1_rmse, _time0_swap

DEVICE = "cpu"
DTYPE = torch.float64
H = val_config.VAL_H_TARGET
KAPPA = config.HYBRID_KAPPA  # production scheme; coupled C3 is measured at this κ
SEED = val_config.PRICE_SEED
PROD_DT = val_config.PRICE_DT  # 1/24 anchor (coarsest level)
C3_FINE_DT = 1.0 / 96.0  # diagnostic-only fine reference, not a config grid
LADDER = (PROD_DT, 1.0 / 48.0, C3_FINE_DT)  # nested 2× refinements

N_PATHS = 2048  # coupling collapses the difference SE
DECISION_NEGLIGIBLE_BPS = 1.0  # absolute floor; SE diagnostic only
COUPLED_SE_MAX_BPS = 1.0  # gate verdict: coupled C3 SE below this ⇒ coupling worked


def _iv_bps(PV, A0, K, S0, T_a):
    bach = torch.tensor(PV / float(A0), dtype=DTYPE).reshape(1)
    K_rel = (K - S0).reshape(1)
    T_t = torch.tensor(T_a, dtype=DTYPE).reshape(1)
    init = torch.clamp(bach / (math.sqrt(T_a) * INV_SQRT_2PI), min=1e-6)
    return float(bachelier_iv_newton(bach, K_rel, T_t, init).item()) * 1e4


def _to_bps(dprice, A0, T_a):
    """Price (difference) → normal-vol bps via the ATM vega-approx (C2 convention)."""
    return dprice / (float(A0) * math.sqrt(T_a) * INV_SQRT_2PI) * 1e4


def _coarsen(dz_fine, r):
    if r == 1:
        return dz_fine
    N, nfs, nf = dz_fine.shape
    n_level = nfs // r
    return dz_fine[:, : n_level * r, :].reshape(N, n_level, r, nf).sum(dim=2)


def _level_payoff(model, dt, n_steps_fine, dz_fine, T_N, k_idx, K):
    """Per-path deflated payoff for one grid level, driven by the coupled (block-summed) fine W."""
    n_steps = int(round(T_N / dt))
    dz_level = _coarsen(dz_fine, n_steps_fine // n_steps)
    time_grid = torch.linspace(0.0, T_N, n_steps + 1, dtype=DTYPE, device=DEVICE)
    ex_step = int(torch.argmin(torch.abs(time_grid - model.T[k_idx])).item())
    # driver_type="prng" ⇒ antithetic=False (correct for the non-antithetic injected dz); the κ=1
    # near-noise stays internal (seed+7919, uncoupled). dz_override bypasses generation.
    F_paths = model.simulate_forward_curve(
        N_PATHS, time_grid, freeze_drift=True, seed=SEED, driver_type="prng", dz_override=dz_level
    )
    S, A_def, _ = _swap_and_deflated_annuity(F_paths[:, ex_step, k_idx:], model.tau[k_idx:])
    pay = torch.clamp(S - K, min=0.0) * A_def
    del F_paths, dz_level
    return pay.detach()


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    inputs = load_valuation_inputs(val_config)
    model, _, _ = build_calibrated_model(inputs, DEVICE)
    N, T_N, n_factors = model.N, float(model.T[-1]), model.n_factors
    n_steps_fine = int(round(T_N / C3_FINE_DT))

    rmse = _stage1_rmse(pd.read_csv(os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)), H)
    c2_df = pd.read_csv(os.path.join(config.RESULTS_DIR, "drift_freezing_bias.csv"))
    # C1: κ-scheme kernel residual at the pricing grid, from test_kernel_discretization_bias
    sw = pd.read_csv(os.path.join(config.RESULTS_DIR, f"kernel_dt_sweep_H{H:.2f}.csv"))
    c1_bps = float(
        sw[np.isclose(sw["dt"], PROD_DT) & (sw["kappa"] == KAPPA)]["atm_mean_abs_bps"].iloc[0]
    )

    print("\n" + "=" * 80)
    print(f"{'W-COUPLED EULER TIME-STEPPING BIAS (C3) — production κ=%d scheme' % KAPPA:^80}")
    print("=" * 80)
    print(
        f"  H={H}  κ={KAPPA}  β={float(model.beta_sabr)}  T_N={T_N:.0f}Y  N={N_PATHS}  seed={SEED}"
    )
    print(f"  Nested ladder dt={[_grid_label(d) for d in LADDER]} (n_steps_fine={n_steps_fine})")
    print(f"  Stage-1 RMSE≈{rmse:.2f} bps  floor={DECISION_NEGLIGIBLE_BPS:.2f} bps")

    # hard check: seam transparency (dz_override bypasses generation byte-identically)
    n_chk = int(round(T_N / PROD_DT))
    tg_chk = torch.linspace(0.0, T_N, n_chk + 1, dtype=DTYPE, device=DEVICE)
    dz_native = generate_sobol_bridge_increments(
        N_PATHS, n_chk, n_factors, PROD_DT, DEVICE, DTYPE, seed=SEED
    )
    f_native = model.simulate_forward_curve(N_PATHS, tg_chk, freeze_drift=True, seed=SEED)
    f_inject = model.simulate_forward_curve(
        N_PATHS, tg_chk, freeze_drift=True, seed=SEED, dz_override=dz_native
    )
    assert torch.equal(f_native, f_inject), (
        "SEAM CHECK FAILED: dz_override does not reproduce the native sobol path — injection seam broken."
    )
    del dz_native, f_native, f_inject
    print("\n  Seam check PASSED — dz_override reproduces the native sobol path bit-identically.")

    g = torch.Generator(device=DEVICE).manual_seed(SEED)
    dz_fine = torch.randn(
        N_PATHS, n_steps_fine, n_factors, generator=g, dtype=DTYPE, device=DEVICE
    ) * math.sqrt(C3_FINE_DT)

    instruments = [(N - 1, 1), (N - 5, 5)]  # gate (driftless, no deflator) + short-swap
    level_rows, total_rows = [], []
    for k_idx, nf in instruments:
        T_a = float(model.T[k_idx])
        S0, A0, P0_TN = _time0_swap(model, k_idx)
        K = S0
        inst = f"{int(T_a)}Yx{int(T_N - T_a)}Y"
        print(f"\n{'─' * 80}\n  {inst}  (n_forwards={nf}, expiry={T_a:.0f}Y)")

        pays = {}
        print(f"  {'dt':>6} {'n_steps':>8} {'price':>16} {'IV_bps':>10} {'SE_arm_bps':>11}")
        for dt in LADDER:
            t0 = time.time()
            pay = _level_payoff(model, dt, n_steps_fine, dz_fine, T_N, k_idx, K)
            pays[dt] = pay
            PV = float(P0_TN) * float(pay.mean())
            se_arm = _to_bps(float(P0_TN) * float(pay.std()) / math.sqrt(N_PATHS), A0, T_a)
            iv = _iv_bps(PV, A0, K, S0, T_a)
            print(
                f"  {_grid_label(dt):>6} {int(round(T_N / dt)):8d} {PV:16.10f} {iv:10.4f} "
                f"{se_arm:11.4f}   ({time.time() - t0:.0f}s)"
            )
            level_rows.append(
                {
                    "instrument": inst,
                    "n_forwards": nf,
                    "dt": dt,
                    "grid": _grid_label(dt),
                    "n_paths": N_PATHS,
                    "price": PV,
                    "iv_bps": iv,
                    "se_arm_bps": se_arm,
                }
            )

        # coupled differences (common-mode payoff variance cancels)
        p0tn = float(P0_TN)
        d1_path = p0tn * (pays[LADDER[0]] - pays[LADDER[1]])  # price(1/24) − price(1/48)
        d2_path = p0tn * (pays[LADDER[1]] - pays[LADDER[2]])  # price(1/48) − price(1/96)
        dtot_path = p0tn * (pays[LADDER[0]] - pays[LADDER[2]])  # full coupled gap (1/24 − 1/96)
        D1, D2 = float(d1_path.mean()), float(d2_path.mean())
        se_d1 = _to_bps(float(d1_path.std()) / math.sqrt(N_PATHS), A0, T_a)
        se_d2 = _to_bps(float(d2_path.std()) / math.sqrt(N_PATHS), A0, T_a)
        se_c3 = _to_bps(float(dtot_path.std()) / math.sqrt(N_PATHS), A0, T_a)  # resolved-C3 SE

        ratio = D1 / D2 if abs(D2) > 1e-300 else float("nan")
        if np.isfinite(ratio) and 1.0 < ratio < 8.0:  # converging, rate p = log2(ratio) ∈ (0,3)
            p = math.log2(ratio)
            c3_price = D1 + D2 + D2 / (2.0**p - 1.0)  # = price(1/24) − p₀ (geometric tail)
            mode = f"richardson(p={p:.2f})"
        else:
            p = float("nan")
            c3_price = D1 + D2  # = price(1/24) − price(1/96), conservative two-grid bound
            mode = "2grid-bound"
        c3_bps = _to_bps(c3_price, A0, T_a)

        print(
            f"  → coupled Δ1={_to_bps(D1, A0, T_a):+.4f}±{se_d1:.4f}  "
            f"Δ2={_to_bps(D2, A0, T_a):+.4f}±{se_d2:.4f} bps  ratio={ratio:.3f}  [{mode}]"
        )
        print(f"    C3(1/24) = {c3_bps:+.4f} ± {se_c3:.4f} bps   (coupled SE {se_c3:.3f})")
        if nf == 1:
            verdict = "COUPLING WORKED" if se_c3 < COUPLED_SE_MAX_BPS else "SE STILL LARGE"
            print(
                f"    [gate verdict] {verdict}: gate coupled SE {se_c3:.3f} bps "
                f"{'<' if se_c3 < COUPLED_SE_MAX_BPS else '>='} {COUPLED_SE_MAX_BPS} bps"
            )
            if se_c3 >= COUPLED_SE_MAX_BPS:
                print(
                    "    → ESCALATE: also couple the near-lag draw (shared standard normal across "
                    "grids, per-grid residual-Cholesky scaling) before trusting C3."
                )

        c2_row = c2_df[(c2_df["grid"] == _grid_label(PROD_DT)) & (c2_df["n_forwards"] == nf)].iloc[
            0
        ]
        c2 = float(c2_row["delta_iv_bps"])
        total = abs(c1_bps) + abs(c2) + abs(c3_bps)
        total_rows.append(
            {
                "instrument": inst,
                "n_forwards": nf,
                "c1_bps": c1_bps,
                "c2_bps": c2,
                "c2_se": float(c2_row["se_paired_bps"]),
                "c3_bps": c3_bps,
                "c3_se": se_c3,
                "c3_mode": mode,
                "total_bps": total,
                "total_pct_rmse": 100.0 * total / rmse,
            }
        )
        del pays, d1_path, d2_path, dtot_path

    pd.DataFrame(level_rows).to_csv(
        os.path.join(config.RESULTS_DIR, "euler_timestep_bias_coupled.csv"), index=False
    )
    tdf = pd.DataFrame(total_rows)
    tdf.to_csv(os.path.join(config.RESULTS_DIR, "discretisation_bias_total.csv"), index=False)

    print(f"\n{'=' * 80}")
    print(f"{'TOTAL PRODUCTION DISCRETISATION BIAS  (1/24, |C1|+|C2|+|C3_coupled|)':^80}")
    print(f"{'=' * 80}")
    print(
        f"  {'instrument':>10} {'C1':>6} {'C2':>10} {'C3(±SE)':>20} {'total':>8} {'%RMSE':>7}  mode"
    )
    for r in total_rows:
        print(
            f"  {r['instrument']:>10} {r['c1_bps']:6.2f} {r['c2_bps']:+10.4f} "
            f"{r['c3_bps']:+9.4f}±{r['c3_se']:7.4f} {r['total_bps']:8.4f} "
            f"{r['total_pct_rmse']:6.1f}%  {r['c3_mode']}"
        )
    worst = tdf.loc[tdf["total_bps"].idxmax()]
    print(
        f"\n  max total = {worst['total_bps']:.4f} bps @ {worst['instrument']} "
        f"= {worst['total_pct_rmse']:.1f}% of the {rmse:.2f}-bps RMSE;  floor "
        f"= {DECISION_NEGLIGIBLE_BPS:.2f} bps"
    )
    print(
        "  CAVEAT: C3 measured at κ=1 absorbs only the κ=1 kernel's residual dt-refinement "
        "(not the κ=0 kernel bias) — leading-order sum, kernel not double-counted."
    )
    print()


if __name__ == "__main__":
    main()
