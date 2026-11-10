"""
Drift-freezing bias diagnostic: component C2 of the discretisation bias.

Measures the bias of the production frozen-coefficient FMM scheme (freeze_drift=True, F₀ substituted
into the strict-upper-triangle drift) against the state-dependent Euler reference (freeze_drift=False,
live F_t in the drift).

Method (paired CRN, mirrors the kernel-discretisation test)
-----------------------------------------------------------
Price the same instrument with freeze_drift=True vs False at the same dt and seed → identical
internal draws dz_all. The arms then differ only in the drift computation, so Δ = price(frozen) −
price(state-dependent) is the drift-freezing bias: the shared Euler time-stepping error (C3) cancels
in the paired difference. Because the arms share draws, the paired-Δ SE sits far below the per-arm SE,
so moderate N resolves a sub-bps Δ.

Pure-drift note (β=0 / shift=0)
-------------------------------
At config.BETA_SABR=0 / SHIFT_SABR=0 the vol-scale is `alphas` in both branches (model.py), so the
diffusion term is byte-identical across arms and the entire Δ comes from the drift's F-dependence
(frozen 1/(1+τF₀) vs live 1/(1+τF_t)). At the production point this is a pure drift
test. With β>0 it would additionally capture vol-scale freezing.

Instruments (co-terminal, European, ATM)
----------------------------------------
- Single-forward sanity gate: swap = terminal accrual period [T_{N-1}, T_N]. The terminal forward has
  an empty last Lambda_upper row → zero drift in both arms by construction, so Δ=0. This is a
  harness-integrity check (CRN alignment, diffusion byte-identity, pipeline determinism), not a physics
  claim about small drift; it is the only hard pass/fail assert.
- Multi-forward sweep (n_forwards 5→28, co-terminal to T_N): carries the bias-grows-with-drift physics.
- Truncated pairs (TRUNCATED_PAIRS, e.g. 5Y×5Y): priced on a model whose grid ends at expiry+tenor
  (tests/oos.py truncation), so a non-co-terminal swap is co-terminal on its own grid.
  Its max|Δ_IV| is a reported measurement vs the 1.0-bps floor and the Stage-1 RMSE. Exceeding the
  floor is a finding (the production frozen-drift scheme carries material bias, which bears on the
  frozen-drift invariant), not a test failure to suppress.

Execution / cost
----------------
Price-only under no_grad (no AAD). The freeze_drift=False step-loop is the runtime long pole: the
long-expiry 1/50 cell loops ~1550 steps with a per-step inter-forward einsum; expect a few minutes per
state-dependent arm.
Memory: one full 30Y path held at a time (~3 GB at N=8192, dt=1/50; step-loop transient ~2×), modest vs
the Bermudan. Sobol dim (n_steps·32) exceeds the 21201 cap → seeded-antithetic PRNG fallback;
CRN/antithetic still hold, which is all the paired test needs.

Usage
-----
    python tests/test_drift_freezing_bias.py

Outputs
-------
    results/drift_freezing_bias.csv   : one row per (instrument, grid)
    Console: per-grid Δ table + production-grid materiality read-off.
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
from jolto.core.market_data import load_discount_curve
from jolto.valuation.bermudan_swaption import _swap_and_deflated_annuity
from oos import S1_CSV, SIGMA_CSV, DISC_CSV, _csv, _stage1_functions, truncated_model_for
from _common import _grid_label, _stage1_rmse, _time0_swap

# operating point sourced from config/val_config, no literals
DEVICE = "cpu"
DTYPE = torch.float64
H = val_config.VAL_H_TARGET  # model built at this H by value
KAPPA = config.HYBRID_KAPPA  # production rough-driver near-lag truncation
SEED = val_config.PRICE_SEED  # CRN seed shared by both arms
GRIDS = (config.CALI_DT, val_config.PRICE_DT)  # 1/50 (calibration) and 1/24 (production pricing)
PROD_DT = val_config.PRICE_DT  # the grid whose read-off gates Bermudan pricing

N_PATHS = (
    8192  # CRN paired SE ≪ per-arm SE → moderate N resolves sub-bps Δ; raise if near the floor
)

DECISION_NEGLIGIBLE_BPS = 1.0  # absolute materiality floor; paired SE is diagnostic only
GATE_REL_TOL = (
    1e-10  # harness-integrity gate: |Δ|/price below this ⇒ CRN/diffusion/determinism intact
)
NEAR_FLOOR_SE_MULT = 3.0  # |Δ−floor| within this many paired SE ⇒ recommend a higher-N re-run
# Non-co-terminal instruments (expiry, tenor): priced on a model truncated at expiry+tenor, as in
# tests/oos.py, so the target swap is co-terminal on its own grid. 5Y×5Y matches the Euler test.
TRUNCATED_PAIRS = ((5.0, 5.0),)


def _price_arm(model, time_grid, freeze_drift, ex_step, k_idx, K, P0_TN):
    """Per-path deflated payoff and discounted PV for one drift scheme. Holds one full path; frees it."""
    F_paths = model.simulate_forward_curve(N_PATHS, time_grid, freeze_drift=freeze_drift, seed=SEED)
    S, A_deflated, _ = _swap_and_deflated_annuity(F_paths[:, ex_step, k_idx:], model.tau[k_idx:])
    pay = torch.clamp(S - K, min=0.0) * A_deflated
    del F_paths
    PV = P0_TN * pay.mean()
    return pay.detach(), PV.detach()


def _invert_iv_bps(PV, A0, S0, K, T_a):
    """Normal-vol (bps) of a per-annuity forward Bachelier price; forward shifted to 0 so the relative
    strike is K−S₀ (=0 at ATM), matching bachelier_iv_newton's convention."""
    bach_price = (PV / A0).reshape(1)
    K_rel = (K - S0).reshape(1)
    T_t = torch.as_tensor(T_a, dtype=DTYPE, device=DEVICE).reshape(1)
    init = torch.clamp(bach_price / (math.sqrt(T_a) * INV_SQRT_2PI), min=1e-6)
    sigma = bachelier_iv_newton(bach_price, K_rel, T_t, init)
    return float(sigma.item()) * 1e4


def _measure_cell(model, time_grid, dt, label, k_idx, nf, rmse):
    """Paired frozen-vs-state Δ for the co-terminal ATM swap T_{k_idx} → T_N of `model`."""
    T_N = float(model.T[-1])
    T_a = float(model.T[k_idx])
    ex_step = int(torch.argmin(torch.abs(time_grid - T_a)).item())
    S0, A0, P0_TN = _time0_swap(model, k_idx)
    K = S0  # ATM

    t0 = time.time()
    pay_fz, PV_fz = _price_arm(model, time_grid, True, ex_step, k_idx, K, P0_TN)
    pay_st, PV_st = _price_arm(model, time_grid, False, ex_step, k_idx, K, P0_TN)

    delta_path = P0_TN * (pay_fz - pay_st)
    se_price = float(delta_path.std().item()) / math.sqrt(N_PATHS)
    se_iv_bps = se_price / (float(A0) * math.sqrt(T_a) * INV_SQRT_2PI) * 1e4

    iv_fz = _invert_iv_bps(PV_fz, A0, S0, K, T_a)
    iv_st = _invert_iv_bps(PV_st, A0, S0, K, T_a)
    d_iv = iv_fz - iv_st
    d_price = float(PV_fz - PV_st)

    inst = f"{int(T_a)}Yx{int(T_N - T_a)}Y"
    print(
        f"  {inst:>10} {T_a:7.0f} {T_N - T_a:6.0f} {nf:5d} "
        f"{iv_fz:10.4f} {iv_st:10.4f} {d_iv:9.4f} {se_iv_bps:9.4f}  "
        f"{100.0 * abs(d_iv) / rmse:6.1f}%   ({time.time() - t0:.0f}s)"
    )
    return {
        "grid": label,
        "dt": dt,
        "instrument": inst,
        "expiry_T": T_a,
        "swap_len": T_N - T_a,
        "n_forwards": nf,
        "T_N": T_N,
        "n_paths": N_PATHS,
        "S0": float(S0),
        "strike": float(K),
        "price_frozen": float(PV_fz),
        "price_state": float(PV_st),
        "delta_price": d_price,
        "iv_frozen_bps": iv_fz,
        "iv_state_bps": iv_st,
        "delta_iv_bps": d_iv,
        "se_paired_bps": se_iv_bps,
    }


def main():
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    torch.manual_seed(SEED)

    inputs = load_valuation_inputs(val_config)
    model, _, _ = build_calibrated_model(inputs, DEVICE)  # built once; dt enters only via time_grid
    disc_func = load_discount_curve(str(DISC_CSV))
    sigma_np = _csv(SIGMA_CSV, header=None).to_numpy().astype(np.float64)
    s1_funcs = _stage1_functions(S1_CSV, H)
    N = model.N
    T_N = float(model.T[-1])

    rmse = _stage1_rmse(pd.read_csv(os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)), H)

    # Instruments by forward count: gate (1 forward = terminal period) + multi-forward sweep.
    n_forwards_grid = [1, 5, 10, 15, 20, 25, 28]
    instruments = [(N - nf, nf) for nf in n_forwards_grid]  # (k_idx, n_forwards)

    print("\n" + "=" * 74)
    print(f"{'DRIFT-FREEZING BIAS (C2) — frozen vs state-dependent FMM drift':^74}")
    print("=" * 74)
    print(
        f"  H={H}  κ={KAPPA}  β={float(model.beta_sabr)}  N_forwards_total={N}  T_N={T_N:.0f}Y  "
        f"N_paths={N_PATHS}  seed={SEED}"
    )
    print(
        f"  Stage-1 RMSE @H={H} ≈ {rmse:.2f} bps   negligible floor = {DECISION_NEGLIGIBLE_BPS:.2f} bps"
    )
    print(f"  Grids: {[_grid_label(dt) for dt in GRIDS]}   (β=0 ⇒ pure drift test)")

    rows = []
    with torch.no_grad():
        for dt in GRIDS:
            label = _grid_label(dt)
            n_steps = int(round(T_N / dt))
            time_grid = torch.linspace(0.0, T_N, n_steps + 1, dtype=DTYPE, device=DEVICE)
            print(f"\n{'─' * 74}")
            print(f"  grid {label}  (dt={dt:.5f}, n_steps={n_steps})")
            print(
                f"  {'inst':>10} {'expiry':>7} {'swlen':>6} {'nfwd':>5} "
                f"{'IV_frozen':>10} {'IV_state':>10} {'ΔIV_bps':>9} {'pairedSE':>9}  {'%RMSE':>7}"
            )
            for k_idx, nf in instruments:
                rows.append(_measure_cell(model, time_grid, dt, label, k_idx, nf, rmse))
            for pair in TRUNCATED_PAIRS:
                m_tr, _, _ = truncated_model_for(
                    pair, disc_func, sigma_np, H, s1_funcs, DEVICE, DTYPE
                )
                T_tr = float(m_tr.T[-1])
                tg_tr = torch.linspace(
                    0.0, T_tr, int(round(T_tr / dt)) + 1, dtype=DTYPE, device=DEVICE
                )
                nf_tr = int(round(pair[1] / float(m_tr.tau[0])))
                rows.append(_measure_cell(m_tr, tg_tr, dt, label, m_tr.N - nf_tr, nf_tr, rmse))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(config.RESULTS_DIR, "drift_freezing_bias.csv"), index=False)

    # harness-integrity gate (hard): single-forward Δ must be machine-zero on every grid
    gate = df[df["n_forwards"] == 1]
    for _, r in gate.iterrows():
        rel = abs(r["delta_price"]) / max(abs(r["price_frozen"]), 1e-300)
        assert rel < GATE_REL_TOL, (
            f"HARNESS GATE FAILED (grid {r['grid']}): single-forward Δ_price rel={rel:.2e} "
            f"> {GATE_REL_TOL:.0e} — the terminal forward is driftless in both arms by construction, "
            f"so a nonzero Δ means broken CRN alignment / diffusion byte-identity / determinism."
        )
    print(
        "\n  Harness gate PASSED — single-forward Δ machine-zero on all grids (CRN/diffusion intact)."
    )

    # production-grid materiality read-off (reported measurement, not a pass/fail assert)
    prod = df[(df["dt"] == PROD_DT) & (df["n_forwards"] > 1)]
    worst = prod.loc[prod["delta_iv_bps"].abs().idxmax()]
    max_abs = abs(worst["delta_iv_bps"])
    print(f"\n{'=' * 74}")
    print(f"{('PRODUCTION-GRID READ-OFF  (dt=%s)' % _grid_label(PROD_DT)):^74}")
    print(f"{'=' * 74}")
    print(
        f"  max|Δ_IV| = {max_abs:.4f} bps @ {worst['instrument']} (n_fwd={int(worst['n_forwards'])})  "
        f"= {100.0 * max_abs / rmse:.1f}% of the {rmse:.2f}-bps RMSE;  floor = {DECISION_NEGLIGIBLE_BPS:.2f} bps"
    )
    if max_abs <= DECISION_NEGLIGIBLE_BPS:
        print("  → drift-freezing bias is IMMATERIAL at the production grid (≤ floor).")
    else:
        print(
            "  → drift-freezing bias EXCEEDS the floor — MATERIAL finding: the production frozen-drift "
            "scheme carries resolvable bias. Escalate (bears on the frozen-drift invariant)."
        )
    near = prod[
        (DECISION_NEGLIGIBLE_BPS - prod["delta_iv_bps"].abs()).abs()
        <= NEAR_FLOOR_SE_MULT * prod["se_paired_bps"]
    ]
    if not near.empty:
        cells = ", ".join(f"{r['instrument']}@{r['grid']}" for _, r in near.iterrows())
        print(
            f"  [confidence] {cells} sit within {NEAR_FLOOR_SE_MULT:g}× paired SE of the floor — "
            f"re-run these at higher N before stating the verdict (confidence in the call, not an SE gate)."
        )
    print()


if __name__ == "__main__":
    main()
