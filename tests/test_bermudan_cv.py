"""Control-variate validation for the Bermudan pricer (valuation._bermudan_gaussian_cv).

Acceptance gate (binding): CV price and greek sums match the no-CV path within GATE_SE standard
errors across seeds; the script asserts it. Also reports the variance-reduction factor per quantity
and trials anchor dates. Research check, not pytest. Run: python tests/test_bermudan_cv.py
"""

import time
import numpy as np
import torch

from jolto import val_config
from jolto.api.assembly import load_valuation_inputs, build_calibrated_model
from jolto.api.trade_specs import BermudanSwaptionSpec
from jolto.valuation.bermudan_swaption import bermudan_pricer, oos_bermudan_pricer

GATE_SE = 3.0  # |CV − no-CV| must stay below this many standard errors of the difference


def _price_and_greeks(model, trade, time_grid, seed, use_cv, anchor):
    for p in model.differentiable_parameters().values():
        p.requires_grad_(True)
        if p.grad is not None:
            p.grad = None
    price = oos_bermudan_pricer(
        model,
        trade,
        n_paths_main=N_MAIN,
        n_paths_pilot=N_PILOT,
        time_grid=time_grid,
        seed=seed,
        verbose=False,
        use_control_variate=use_cv,
        cv_anchor_step=anchor,
    )
    price.backward()
    g = model.read_greeks(tuple(val_config.VAL_GREEKS))
    gsum = {k: float(np.sum(np.asarray(v))) for k, v in g.items()}
    return float(price.item()) * 1e4, gsum


if __name__ == "__main__":
    N_MAIN = 4096
    N_PILOT = 2048
    N_SEEDS = 12
    device = "cpu"
    torch.manual_seed(0)

    inputs = load_valuation_inputs(val_config)
    model, grid_T, _ = build_calibrated_model(inputs, device)
    trade = BermudanSwaptionSpec.from_val_config(val_config).to_trade(inputs.grid_step)
    max_T = float(grid_T[-1])
    n_steps = int(np.ceil(max_T / val_config.PRICE_DT))
    time_grid = torch.linspace(0.0, max_T, n_steps + 1, device=device, dtype=torch.float64)

    ex_dates = torch.tensor(trade["Ex_Dates"], dtype=torch.float64)
    ex_steps = [int(torch.argmin(torch.abs(time_grid - d)).item()) for d in ex_dates]
    anchors = {"first": ex_steps[0], "last": ex_steps[-1]}

    # byte-identical no-CV regression (single seed)
    p_a = bermudan_pricer(model, trade, n_paths=N_MAIN, time_grid=time_grid, seed=7)
    p_b = bermudan_pricer(
        model, trade, n_paths=N_MAIN, time_grid=time_grid, seed=7, use_control_variate=False
    )
    assert float(p_a) == float(p_b), f"no-CV path changed: {float(p_a)} vs {float(p_b)}"
    print(f"[regression] no-CV path byte-identical: {float(p_a) * 1e4:.6f} bps  OK\n")

    seeds = [1000 + 7 * i for i in range(N_SEEDS)]
    gkeys = list(val_config.VAL_GREEKS)

    base_p, base_g = [], {k: [] for k in gkeys}
    t0 = time.time()
    for s in seeds:
        p, g = _price_and_greeks(model, trade, time_grid, s, use_cv=False, anchor=None)
        base_p.append(p)
        for k in gkeys:
            base_g[k].append(g[k])
    print(f"[no-CV] {N_SEEDS} seeds in {time.time() - t0:.1f}s")
    base_p = np.array(base_p)

    # use_control_variate=True is the (1+n_ex) vector CV (per-exercise-date deflator stack);
    # the anchor only picks the Gaussian control's date (the deflator vector is anchor-independent),
    # so the two arms below differ only in the Gaussian anchor.
    failures = []
    for name, anchor in anchors.items():
        cv_p, cv_g = [], {k: [] for k in gkeys}
        t0 = time.time()
        for s in seeds:
            p, g = _price_and_greeks(model, trade, time_grid, s, use_cv=True, anchor=anchor)
            cv_p.append(p)
            for k in gkeys:
                cv_g[k].append(g[k])
        cv_p = np.array(cv_p)
        dt = time.time() - t0

        dprice = cv_p.mean() - base_p.mean()
        se = np.sqrt(base_p.var(ddof=1) + cv_p.var(ddof=1)) / np.sqrt(N_SEEDS)
        vr_price = base_p.std(ddof=1) / cv_p.std(ddof=1)
        if abs(dprice) > GATE_SE * se:
            failures.append(f"{name} price {dprice / se:+.2f}σ")
        print(f"\n===== vector-CV, Gaussian anchor={name} (step {anchor}) — {dt:.1f}s =====")
        print(
            f"  price:  noCV {base_p.mean():8.4f}  CV {cv_p.mean():8.4f}  "
            f"Δ {dprice:+.4f} bps ({dprice / se:+.2f}σ)  VR×std {vr_price:.2f}"
        )
        for k in gkeys:
            b, c = np.array(base_g[k]), np.array(cv_g[k])
            dg = c.mean() - b.mean()
            seg = np.sqrt(b.var(ddof=1) + c.var(ddof=1)) / np.sqrt(N_SEEDS)
            vr = b.std(ddof=1) / max(c.std(ddof=1), 1e-30)
            print(f"  {k:>8}: Δ {dg:+.4e} ({dg / max(seg, 1e-30):+.2f}σ)  VR×std {vr:.2f}")
            if abs(dg) > GATE_SE * seg:
                failures.append(f"{name} {k} {dg / max(seg, 1e-30):+.2f}σ")

    assert not failures, f"CV differs from no-CV beyond {GATE_SE:g} SE: {failures}"
    print(f"\nGATE PASSED: CV matches no-CV within {GATE_SE:g} SE on price and greeks")
