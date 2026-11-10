import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from jolto.core.utils import log_progress
from jolto.calibration.pricers import torch_bachelier, mapped_smm_ode_pricer
from jolto import config


def _cms_run_defaults(n_paths, dt, seed):
    """Fill unset CMS run options from config: (n_paths, dt, seed)."""
    dt = dt if dt is not None else config.CMS_DT
    n_paths = n_paths if n_paths is not None else config.CMS_PATHS_MAIN
    seed = seed if seed is not None else config.CMS_SEED
    return n_paths, dt, seed


def _truncated_grid(model, fixing_date, dt):
    """Uniform grid on [0, T_a]; dt must divide T_a, else linspace would silently rescale it."""
    n = float(fixing_date) / dt
    n_steps = int(round(n))
    if abs(n - n_steps) > 1e-9:
        raise ValueError(f"CMS dt={dt} does not divide fixing_date {fixing_date} (T_a/dt={n})")
    time_grid = torch.linspace(
        0.0, float(fixing_date), n_steps + 1, device=model.device, dtype=model.dtype
    )
    return time_grid, n_steps


def _fixing_forwards(model, fixing_date, n_paths, dt, seed, freeze_drift, use_checkpoint):
    """Simulate to T_a on a grid truncated at the fixing; return F_a = F(T_a), (n_paths, N).

    Nothing past the fixing is read and the kernel cost grows ~O(n_steps²). F_i(T_a) depends only
    on the path up to T_a, so truncation leaves every bond built from F_a unchanged. With no
    model.T[-1] ceiling left, the fixing is guarded against the curve.
    """
    assert fixing_date <= model.T[-1].item() + 1e-9, (
        f"fixing_date {fixing_date} is beyond the curve maturity {model.T[-1].item()}"
    )
    time_grid, a_idx = _truncated_grid(model, fixing_date, dt)  # fixing is the terminal step

    # only the fixing-step forwards are read
    return model.simulate_forward_curve(
        n_paths,
        time_grid,
        freeze_drift=freeze_drift,
        use_checkpoint=use_checkpoint,
        seed=seed,
        steps=[a_idx],
    )[:, 0]


def _terminal_deflator(model, F_a, a):
    """Pathwise terminal deflator 1/P(T_a,T_N) over the slice [a:] of the fixing-step forwards.

    The clamp mirrors bermudan_pricer's terminal-bond guard against a vanishing P(T_a,T_N).
    """
    P_Ta_TN = torch.cumprod(1.0 / (1.0 + model.tau[a:] * F_a[:, a:]), dim=1)[:, -1]
    return 1.0 / torch.clamp(P_Ta_TN, min=1e-12)


def _initial_bonds(model):
    """Initial discount curve P(0,T_{k+1}) from model.F0, on-tape."""
    return torch.cumprod(1.0 / (1.0 + model.tau * model.F0), dim=0)


def _initial_bond_at(model, P0_T, a):
    """P(0,T_a) from the initial curve; 1 at a = 0."""
    return P0_T[a - 1] if a >= 1 else torch.ones((), device=model.device, dtype=model.dtype)


def cms_caplet_floorlet_price(
    model,
    fixing_date,
    swap_end,
    strike,
    year_fraction,
    is_cap=True,
    n_paths=None,
    dt=None,
    seed=None,
    freeze_drift=True,
    use_checkpoint=False,
    return_diagnostics=False,
):
    """
    Monte-Carlo CMS caplet/floorlet on the par swap rate S_{a,b}(T_a): δ·(S−K)^+ or δ·(K−S)^+,
    settling at the fixing date T_a (a later payment would need a pathwise P(T_a,T_p) factor).

    The model simulates under the terminal measure P(·,T_N), so the coupon is deflated pathwise by
    1/P(T_a,T_N) and discounted with P(0,T_N), as in bermudan_pricer; a deterministic P(0,T_p)
    discount would give wrong-signed convexity. Bonds are rebuilt inline and stay on-tape, so
    price.backward() gives PV01/Vega in model.F0.grad / model.alphas.grad.
    """
    n_paths, dt, seed = _cms_run_defaults(n_paths, dt, seed)

    # β=0 Bachelier is the supported path: vol_scale = alphas, so the model's
    # clamp(F+shift, min=0) (β≠0 branch) never enters the graph and the only non-smooth
    # point is the payoff kink (measure-zero under MC; fine for pathwise AAD).
    assert model.beta_sabr == 0.0, "CMS pricer supports the β=0 (Bachelier) default only"

    # Tenor indices (annual grid), distinct from the time-step index a_idx of the fixing. Resolve
    # them exactly or fail: an off-by-one in `a` shifts the swap window a year and still prices
    # plausibly.
    a = round(fixing_date / config.GRID_STEP_SIZE)
    b = round(swap_end / config.GRID_STEP_SIZE)
    assert abs(model.T[a].item() - fixing_date) < 1e-9, "fixing_date is not a tenor node"
    assert abs(model.T[b].item() - swap_end) < 1e-9, "swap_end is not a tenor node"
    assert b > a, "swap window [T_a, T_b] must be non-empty"

    log_progress("Action", "Starting CMS caplet/floorlet MC pricing...", 0)
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)

    # F_a = F(T_a), (n_paths, N), on the grid truncated at T_a
    F_a = _fixing_forwards(model, fixing_date, n_paths, dt, seed, freeze_drift, use_checkpoint)

    # Path-wise bonds at the fixing: P(T_a,T_k) = Π_{i=a..k-1} 1/(1+τ_i F_i(T_a)).
    # The swap rate uses [a:b], the deflator below runs [a:] to T_N: two slices of the same
    # fixing-step forwards, do not unify them.
    F_fix = F_a[:, a:b]  # (n_paths, b-a)
    tau_win = model.tau[a:b]
    P_partial = torch.cumprod(1.0 / (1.0 + tau_win * F_fix), dim=1)  # P(T_a,T_{a+1..b})

    P_Ta_Tb = P_partial[:, -1]  # P(T_a, T_b)
    annuity = torch.sum(tau_win * P_partial, dim=1)  # Σ_{j=a+1..b} τ_{j-1} P(T_a,T_j)
    S = (1.0 - P_Ta_Tb) / annuity  # S_{a,b}(T_a)

    payoff = torch.clamp(S - strike, min=0.0) if is_cap else torch.clamp(strike - S, min=0.0)

    deflator = _terminal_deflator(model, F_a, a)  # 1/P(T_a,T_N) over [a:]

    # One curve, built once and reused for the discount and the diagnostics S_fwd / P(0,T_a).
    P0_T = _initial_bonds(model)  # P(0,T_{k+1})

    price = P0_T[-1] * year_fraction * torch.mean(payoff * deflator)

    if not return_diagnostics:
        return price

    # Forward swap rate from the same P0_T (the P(0,T_a) factors cancel to this form).
    P0_at = _initial_bond_at(model, P0_T, a)
    S_fwd = ((P0_at - P0_T[b - 1]) / torch.sum(tau_win * P0_T[a:b])).item()

    # Deflator-weighted (T_a-forward) CMS rate: the convexity-adjusted rate that exceeds
    # S_fwd. Self-normalised IS estimator: w = deflator / mean(deflator), S_cms = mean(w·S).
    with torch.no_grad():
        w = deflator / torch.mean(deflator)
        S_cms_t = torch.mean(w * S)
        S_cms = S_cms_t.item()
        S_cms_stderr = (torch.std(w * (S - S_cms_t)) / (n_paths**0.5)).item()

    diagnostics = {
        "S_fwd": S_fwd,
        "S_cms": S_cms,  # convexity-adjusted rate E^{T_a}[S]
        "S_cms_stderr": S_cms_stderr,
        "P0_Ta": P0_at.item(),  # P(0,T_a), discount for the deep-ITM / parity identities
        "n_paths": n_paths,
    }
    return price, diagnostics


def self_validate_cms(
    model, fixing_date=5.0, swap_end=10.0, year_fraction=1.0, seed=None, use_checkpoint=False
):
    """
    Benchmark-free checks under the terminal-deflator measure: (a) positive convexity
    S_cms > S_fwd; (b) finite, signed Greeks; (c) deep-OTM ≈ 0, deep-ITM (K=0) ==
    δ·P(0,T_a)·S_cms, and cap/floor parity.

    Returns a dict of the printed quantities.
    """
    model.F0.requires_grad_(True)
    model.alphas.requires_grad_(True)

    # every call prices the same trade and run; only strike / is_cap / diagnostics vary
    caplet = partial(
        cms_caplet_floorlet_price,
        model,
        fixing_date,
        swap_end,
        year_fraction=year_fraction,
        seed=seed,
        use_checkpoint=use_checkpoint,
    )

    # K=0 diagnostics call: strike-independent S_fwd / S_cms / P(0,T_a), and this price IS
    # the deep-ITM (K=0) cap used in check (c).
    price_itm_t, diag = caplet(strike=0.0, is_cap=True, return_diagnostics=True)
    S_fwd, S_cms, S_cms_stderr, P0_Ta = (
        diag["S_fwd"],
        diag["S_cms"],
        diag["S_cms_stderr"],
        diag["P0_Ta"],
    )

    print(
        f"  (a) convexity   : S_fwd={S_fwd * 1e4:.2f}bps  S_cms={S_cms * 1e4:.2f}bps  "
        f"adj={(S_cms - S_fwd) * 1e4:+.2f}bps  (stderr={S_cms_stderr * 1e4:.2f}bps)"
    )
    assert S_cms > S_fwd, "convexity wrong-signed: S_cms should exceed forward swap rate"

    # (b) Greeks: ATM cap struck at S_fwd, backward pass.
    model.F0.grad = None
    model.alphas.grad = None
    price_atm = caplet(strike=S_fwd, is_cap=True)
    price_atm.backward()
    pv01 = model.F0.grad * 1e-4
    vega = model.alphas.grad * 1e-4
    assert torch.isfinite(pv01).all() and torch.isfinite(vega).all(), "non-finite Greeks"
    print(f"  (b) ATM price   : {price_atm.item() * 1e4:.4f}bps")
    print(f"      PV01        : {pv01.detach().cpu().numpy()}")
    print(f"      Vega        : {vega.detach().cpu().numpy()}")

    # (c) limits + parity (detached; no grad needed). Identities discount with P(0,T_a) and
    # the deflator-weighted S_cms, not P(0,T_p) / raw mean(S).
    with torch.no_grad():
        otm = caplet(strike=10.0 * S_fwd, is_cap=True).item()
        floor_atm = caplet(strike=S_fwd, is_cap=False).item()

    itm = price_itm_t.item()
    intrinsic = year_fraction * P0_Ta * S_cms  # K=0 cap == δ·P(0,T_a)·S_cms
    # parity holds against the adjusted rate S_cms by construction; do not rewrite to S_fwd
    parity_resid = (price_atm.item() - floor_atm) - year_fraction * P0_Ta * (S_cms - S_fwd)

    print(f"  (c) deep-OTM    : {otm * 1e4:.4f}bps  (expect ≈ 0)")
    print(
        f"      deep-ITM    : {itm * 1e4:.4f}bps  vs δ·P(0,T_a)·S_cms={intrinsic * 1e4:.4f}bps  "
        f"(rel {abs(itm - intrinsic) / max(intrinsic, 1e-12):.2e})"
    )
    print(f"      cap−floor parity resid: {parity_resid * 1e4:+.2e}bps")

    return {
        "S_fwd": S_fwd,
        "S_cms": S_cms,
        "S_cms_stderr": S_cms_stderr,
        "price_atm": price_atm.item(),
        "otm": otm,
        "itm": itm,
        "intrinsic": intrinsic,
        "parity_resid": parity_resid,
    }


def cms_spread_option_price(
    model,
    fixing_date,
    long_tenor,
    short_tenor,
    strike,
    year_fraction,
    is_cap=True,
    n_paths=None,
    dt=None,
    seed=None,
    freeze_drift=True,
    use_checkpoint=False,
    return_diagnostics=False,
):
    """
    Monte-Carlo CMS spread option on two co-fixing par swap rates at T_a:

        S¹ = S_{a,a+long}(T_a)  (long leg),   S² = S_{a,a+short}(T_a)  (short leg)
        payoff = δ·(S¹ − S² − K)^+  (is_cap)  /  δ·(K − (S¹ − S²))^+  (floor)

    Brigo–Mercurio (2006) §13.16.2, Eqs (13.30)–(13.31), valued as P(0,T_a)·E^a[·]; the pricer
    uses the caplet's terminal-deflator form P(0,T_N)·E^{T_N}[·/P(T_a,T_N)], equal by change of
    numéraire. Both legs read the same fixing-step forwards, so their correlation runs through Σ.
    Brigo's closed form (13.34) needs lognormal rates and a historical correlation, so there is no
    benchmark; validation is self_validate_cms_spread. Grid truncation, inline bonds and
    AAD work as in cms_caplet_floorlet_price.
    """
    n_paths, dt, seed = _cms_run_defaults(n_paths, dt, seed)

    # β=0 only, for the same reason as the caplet
    assert model.beta_sabr == 0.0, "CMS spread pricer supports the β=0 (Bachelier) default only"
    assert long_tenor > short_tenor > 0.0, "need long_tenor > short_tenor > 0"

    # Tenor indices a (fixing), b_short / b_long (window ends), distinct from the time-step index
    # a_idx below. Resolve them exactly or fail: an off-by-one shifts a window a year unnoticed.
    a = round(fixing_date / config.GRID_STEP_SIZE)
    b_short = a + round(short_tenor / config.GRID_STEP_SIZE)
    b_long = a + round(long_tenor / config.GRID_STEP_SIZE)
    assert abs(model.T[a].item() - fixing_date) < 1e-9, "fixing_date is not a tenor node"
    assert b_long > b_short > a, "need b_long > b_short > a (non-empty windows)"
    assert b_long <= model.N, "long window / deflator runs past the curve (b_long > N)"

    log_progress("Action", "Starting CMS spread-option MC pricing...", 0)
    log_progress("Simulation", f"Generating {n_paths} FMM paths...", 1)

    # F_a = F(T_a), (n_paths, N), on the grid truncated at T_a as in the caplet
    F_a = _fixing_forwards(model, fixing_date, n_paths, dt, seed, freeze_drift, use_checkpoint)

    # one cumprod over the longest window [a:b_long]; the short leg is its (b_short−a) prefix.
    # Keep the deflator slice below separate.
    F_fix = F_a[:, a:b_long]  # (n_paths, b_long-a)
    tau_win = model.tau[a:b_long]
    P_partial = torch.cumprod(1.0 / (1.0 + tau_win * F_fix), dim=1)  # P(T_a,T_{a+1..b_long})

    m = b_short - a  # short-leg prefix length
    S1 = (1.0 - P_partial[:, -1]) / torch.sum(tau_win * P_partial, dim=1)
    # short leg on the prefix: bit-identical to a standalone [a:b_short] cumprod
    S2 = (1.0 - P_partial[:, m - 1]) / torch.sum(tau_win[:m] * P_partial[:, :m], dim=1)

    spread = S1 - S2
    payoff = (
        torch.clamp(spread - strike, min=0.0) if is_cap else torch.clamp(strike - spread, min=0.0)
    )

    # terminal deflator 1/P(T_a,T_N) on its own slice [a:]; same as the caplet, so it shares
    # the caplet's heavy-tailed deflator
    deflator = _terminal_deflator(model, F_a, a)

    # One initial curve, reused for the discount and the diagnostics S_fwd / P(0,T_a).
    P0_T = _initial_bonds(model)  # P(0,T_{k+1})

    price = P0_T[-1] * year_fraction * torch.mean(payoff * deflator)

    if not return_diagnostics:
        return price

    # Forward swap rates from the same P0_T (the P(0,T_a) factors cancel to this form).
    P0_at = _initial_bond_at(model, P0_T, a)
    S1_fwd = ((P0_at - P0_T[b_long - 1]) / torch.sum(tau_win * P0_T[a:b_long])).item()
    S2_fwd = ((P0_at - P0_T[b_short - 1]) / torch.sum(tau_win[:m] * P0_T[a:b_short])).item()

    # Deflator-weighted (T_a-forward) leg rates and covariance: self-normalised IS estimator
    # w = deflator/mean(deflator), so X_cms = mean(w·X) = E^{T_a}[X]. The Σ-exhibit moments
    # (Var, Cov, corr) are reported under the same T_a-forward measure.
    with torch.no_grad():
        w = deflator / torch.mean(deflator)
        S1_cms_t = torch.mean(w * S1)
        S2_cms_t = torch.mean(w * S2)
        S1_cms, S2_cms = S1_cms_t.item(), S2_cms_t.item()
        d1, d2 = S1 - S1_cms_t, S2 - S2_cms_t
        var_S1 = torch.mean(w * d1 * d1).item()
        var_S2 = torch.mean(w * d2 * d2).item()
        cov_S1S2 = torch.mean(w * d1 * d2).item()
        corr = cov_S1S2 / max((var_S1 * var_S2) ** 0.5, 1e-300)
        # A short leg's convexity can be ≈0 within noise, so validation gates its sign (≥ 0 in
        # theory) against the per-leg stderr.
        S1_cms_stderr = (torch.std(w * d1) / (n_paths**0.5)).item()
        S2_cms_stderr = (torch.std(w * d2) / (n_paths**0.5)).item()
        spread_dev = w * ((S1 - S2) - (S1_cms_t - S2_cms_t))
        spread_cms_stderr = (torch.std(spread_dev) / (n_paths**0.5)).item()

    diagnostics = {
        "S1_fwd": S1_fwd,
        "S2_fwd": S2_fwd,
        "S1_cms": S1_cms,  # convexity-adjusted long rate E^{T_a}[S¹]
        "S2_cms": S2_cms,  # convexity-adjusted short rate E^{T_a}[S²]
        "spread_fwd": S1_fwd - S2_fwd,
        "spread_cms": S1_cms - S2_cms,
        "var_S1": var_S1,
        "var_S2": var_S2,
        "cov_S1S2": cov_S1S2,
        "corr": corr,
        "P0_Ta": P0_at.item(),  # P(0,T_a), discount for the deep-ITM / parity identities
        "S1_cms_stderr": S1_cms_stderr,
        "S2_cms_stderr": S2_cms_stderr,
        "spread_cms_stderr": spread_cms_stderr,
        "n_paths": n_paths,
    }
    return price, diagnostics


def _inject_sigma(model, Sigma):
    """
    Bake a correlation matrix Σ into model.loadings / model.Lambda_upper (Cholesky → loadings,
    rate-block → strict-upper Lambda). Returns the prior (loadings, Lambda_upper) clones so the
    caller can restore them. api.assembly.build_calibrated_model injects the Stage-2 Σ with it;
    the Σ-exhibit reprice in self_validate_cms_spread swaps Σ and restores it.
    """
    prev = (model.loadings.detach().clone(), model.Lambda_upper.detach().clone())
    jitter = 1e-8 * torch.eye(Sigma.shape[0], device=model.device, dtype=model.dtype)
    L = torch.linalg.cholesky(Sigma + jitter)
    model.loadings.data.copy_(L)
    corr_rates = L[1:, :] @ L[1:, :].T
    model.Lambda_upper.data.copy_(torch.triu(corr_rates, diagonal=1))
    return prev


def self_validate_cms_spread(
    model,
    Sigma_matrix,
    fixing_date=5.0,
    long_tenor=10.0,
    short_tenor=2.0,
    year_fraction=1.0,
    seed=None,
    use_checkpoint=False,
):
    """
    Benchmark-free validation of the CMS spread option under the terminal-deflator measure:

    (a) per-leg convexity S¹_cms > S¹_fwd, S²_cms > S²_fwd, and a cross-anchor of S¹_cms against
        the standalone CMS caplet (same seed/fixing/long tenor) to machine precision.
    (b) finite AAD Greeks with risk concentrated in [a:b_long] and a long-positive tail.
    (c) deep-OTM → 0; deep-ITM → δ·P(0,T_a)·(S¹_cms − S²_cms − K); numéraire identity
        P(0,T_N)·mean(deflator) ≈ P(0,T_a); spread put–call parity.
    (d) Σ exhibit: covariance decomposition, implied corr, and a decorrelated-Σ reprice.

    The (a) cross-anchor is the precise check; the (c) deflator-mean arms are MC-bounded, since
    1/P(T_a,T_N) is heavy-tailed.
    Sigma_matrix must be the Σ baked into model.loadings (the (d) reprice needs it).
    Returns a dict of the printed quantities.
    """
    a = round(fixing_date / config.GRID_STEP_SIZE)
    b_long = a + round(long_tenor / config.GRID_STEP_SIZE)
    b_short = a + round(short_tenor / config.GRID_STEP_SIZE)
    kappa = config.HYBRID_KAPPA
    n_paths, dt, seed = _cms_run_defaults(None, None, seed)
    print(
        f"  config         : H={model.H.item():.4f}  HYBRID_KAPPA={kappa}  "
        f"paths={n_paths}  dt=1/{round(1 / dt)}  seed={seed}"
    )

    model.F0.requires_grad_(True)
    model.alphas.requires_grad_(True)

    # every spread call prices the same trade and run; only strike / is_cap / diagnostics vary
    spread_option = partial(
        cms_spread_option_price,
        model,
        fixing_date,
        long_tenor,
        short_tenor,
        year_fraction=year_fraction,
        seed=seed,
        use_checkpoint=use_checkpoint,
    )

    # K=0 diagnostics call: strike-independent leg rates / moments / P(0,T_a).
    _, diag = spread_option(strike=0.0, is_cap=True, return_diagnostics=True)
    S1_fwd, S2_fwd = diag["S1_fwd"], diag["S2_fwd"]
    S1_cms, S2_cms = diag["S1_cms"], diag["S2_cms"]
    spread_fwd, spread_cms = diag["spread_fwd"], diag["spread_cms"]
    var_S1, var_S2, cov_S1S2, corr = (
        diag["var_S1"],
        diag["var_S2"],
        diag["cov_S1S2"],
        diag["corr"],
    )
    P0_Ta = diag["P0_Ta"]
    S1_se, S2_se = diag["S1_cms_stderr"], diag["S2_cms_stderr"]
    sigma_spread = max(var_S1 + var_S2 - 2.0 * cov_S1S2, 0.0) ** 0.5

    # (a) per-leg convexity + cross-anchor to the standalone caplet
    # CMS convexity is ≥ 0 per leg. A short leg's can be ≈ 0 within MC noise, so its sign is
    # gated at a 3·stderr band; strict positivity would fail on noise.
    adj1, adj2 = S1_cms - S1_fwd, S2_cms - S2_fwd
    print(
        f"  (a) convexity   : S¹_fwd={S1_fwd * 1e4:.2f}bps  S¹_cms={S1_cms * 1e4:.2f}bps  "
        f"adj={adj1 * 1e4:+.2f}bps  (stderr={S1_se * 1e4:.2f}bps)"
    )
    print(
        f"                    S²_fwd={S2_fwd * 1e4:.2f}bps  S²_cms={S2_cms * 1e4:.2f}bps  "
        f"adj={adj2 * 1e4:+.2f}bps  (stderr={S2_se * 1e4:.2f}bps)"
    )
    print(
        f"      spread      : fwd={spread_fwd * 1e4:.2f}bps  cms={spread_cms * 1e4:.2f}bps  "
        f"(stderr={diag['spread_cms_stderr'] * 1e4:.2f}bps)"
    )
    assert adj1 > 3.0 * S1_se, "long-leg convexity not significantly positive (expect clear +adj)"
    assert adj2 > -3.0 * S2_se, "short-leg convexity significantly negative (should be ≥ 0)"

    # cross-anchor: the long leg of the two-window construction must reproduce the standalone
    # single-window caplet's S_cms to machine precision (same seed/paths/deflator/construction).
    _, cap_diag = cms_caplet_floorlet_price(
        model,
        fixing_date,
        swap_end=fixing_date + long_tenor,
        strike=0.0,
        year_fraction=year_fraction,
        is_cap=True,
        seed=seed,
        use_checkpoint=use_checkpoint,
        return_diagnostics=True,
    )
    anchor_err = abs(S1_cms - cap_diag["S_cms"])
    print(f"      cross-anchor: |S¹_cms − caplet S_cms| = {anchor_err:.2e}  (expect ≲ 1e-12)")
    assert anchor_err < 1e-12, (
        "spread pricer's long leg disagrees with the standalone CMS caplet rate"
    )

    # (b) Greeks: ATM spread cap struck at spread_fwd
    model.F0.grad = None
    model.alphas.grad = None
    price_atm = spread_option(strike=spread_fwd, is_cap=True)
    price_atm.backward()
    pv01 = model.F0.grad * 1e-4
    vega = model.alphas.grad * 1e-4
    assert torch.isfinite(pv01).all() and torch.isfinite(vega).all(), "non-finite Greeks"

    pv01_np = pv01.detach().cpu().numpy()
    in_window = float(np.abs(pv01_np[a:b_long]).sum())
    total = float(np.abs(pv01_np).sum())
    concentration = in_window / max(total, 1e-300)
    long_tail = float(pv01_np[b_short:b_long].sum())  # pure long-leg risk → positive tilt
    print(f"  (b) ATM price   : {price_atm.item() * 1e4:.4f}bps")
    print(f"      PV01        : {pv01_np}")
    print(f"      Vega        : {vega.detach().cpu().numpy()}")
    print(
        f"      structure   : |PV01| in [a:b_long]/total = {concentration:.3f}  "
        f"long-tail [b_short:b_long] PV01 = {long_tail * 1e4:+.4f}bps"
    )
    # P(0,T_N) discount and drift coupling leak small sub-a sensitivities, so "zero before T_a"
    # is approximate: assert concentration in the window and the long-positive tilt instead.
    assert concentration > 0.9, "risk not concentrated in the swap window [a:b_long]"
    assert long_tail > 0.0, "long-only tail [b_short:b_long] PV01 should be positive"

    # (c) limits + parity (detached); the spread forward can be flat or negative, so the strikes
    # are offsets in σ_spread rather than multiples of spread_fwd
    K_otm = spread_fwd + 10.0 * sigma_spread  # deep-OTM regardless of sign(spread_fwd)
    K_itm = spread_fwd - 10.0 * sigma_spread - abs(S1_fwd)  # very negative → payoff always linear
    with torch.no_grad():
        otm = spread_option(strike=K_otm, is_cap=True).item()
        itm = spread_option(strike=K_itm, is_cap=True).item()
        put_atm = spread_option(strike=spread_fwd, is_cap=False).item()
        # Numéraire identity P(0,T_N)·mean(deflator) ≈ P(0,T_a): rebuild the deflator/discount.
        nume_lhs, nume_rhs = _numeraire_identity(model, fixing_date, seed, use_checkpoint)

    intrinsic = year_fraction * P0_Ta * (spread_cms - K_itm)  # deep-ITM → δ·P(0,T_a)·(spread_cms−K)
    # spread put–call parity: call − put == δ·P(0,T_a)·(spread_cms − K), residual second-order
    parity_resid = (price_atm.item() - put_atm) - year_fraction * P0_Ta * (spread_cms - spread_fwd)

    # deep-ITM rel and numéraire rel test one identity, itm/intrinsic = P(0,T_N)·mean(deflator)/
    # P(0,T_a), so both carry the MC error of a heavy-tailed mean (the 1/P(T_a,T_N) tail)
    # that varies run to run; the 1e-2 bound leaves room for it.
    nume_rel = abs(nume_lhs - nume_rhs) / max(abs(nume_rhs), 1e-12)
    itm_rel = abs(itm - intrinsic) / max(abs(intrinsic), 1e-12)
    print(f"  (c) deep-OTM    : {otm * 1e4:.4f}bps  (K={K_otm * 1e4:.1f}bps, expect ≈ 0)")
    print(
        f"      deep-ITM    : {itm * 1e4:.4f}bps  vs δ·P(0,T_a)·(spread_cms−K)="
        f"{intrinsic * 1e4:.4f}bps  (rel {itm_rel:.2e})"
    )
    print(
        f"      numéraire   : P(0,T_N)·E[deflator]={nume_lhs:.6f}  P(0,T_a)={nume_rhs:.6f}  "
        f"(rel {nume_rel:.2e}; heavy-tailed)"
    )
    print(f"      call−put parity resid: {parity_resid * 1e4:+.2e}bps")
    # deep-OTM leaves a small MC residual (a few deflator-weighted >10σ paths), hence a bps bound
    assert abs(otm) < 0.5 / 1e4, "deep-OTM spread option should be ≈ 0"
    assert itm_rel < 1e-2, "deep-ITM ≠ intrinsic beyond the deflator-mean MC band"
    assert nume_rel < 1e-2, "numéraire identity off beyond the deflator-mean MC band"
    assert abs(parity_resid) < 0.5 / 1e4, "parity residual not second-order"

    # (d) Σ exhibit: covariance decomposition + decorrelated-Σ reprice
    var_spread = var_S1 + var_S2 - 2.0 * cov_S1S2
    print(
        f"  (d) Σ exhibit   : Var[S¹−S²]={var_spread:.3e} = Var[S¹]{var_S1:.3e} + "
        f"Var[S²]{var_S2:.3e} − 2·Cov{cov_S1S2:.3e}"
    )
    print(f"      corr(S¹,S²) : {corr:+.4f}")
    assert -1.0 - 1e-9 <= corr <= 1.0 + 1e-9, "implied corr out of [-1,1]"

    # PSD-safe decorrelation: Σ_dec = ζ·Σ + (1−ζ)·I with ζ=0 (identity), PSD by construction.
    # (Zeroing only the rate-block off-diagonals would leave [[1,rᵀ],[r,I]] non-PSD for ‖r‖²>1.)
    base_price = spread_option(strike=spread_fwd, is_cap=True).item()
    Sigma_dec = torch.eye(Sigma_matrix.shape[0], device=model.device, dtype=model.dtype)
    prev = _inject_sigma(model, Sigma_dec)
    try:
        decorr_price = spread_option(strike=spread_fwd, is_cap=True).item()
    finally:
        model.loadings.data.copy_(prev[0])
        model.Lambda_upper.data.copy_(prev[1])

    print(
        f"      reprice (ATM, K=spread_fwd={spread_fwd * 1e4:.2f}bps): base Σ={base_price * 1e4:.4f}bps  "
        f"decorrelated(ζ=0)={decorr_price * 1e4:.4f}bps  Δ={(decorr_price - base_price) * 1e4:+.4f}bps"
    )
    print(
        "      (qualitative Σ exhibit — decorrelation cuts −2·Cov, raising Var[spread]; "
        f"price {'RISES' if decorr_price > base_price else 'does NOT rise'})"
    )

    return {
        "S1_fwd": S1_fwd,
        "S2_fwd": S2_fwd,
        "S1_cms": S1_cms,
        "S2_cms": S2_cms,
        "spread_fwd": spread_fwd,
        "spread_cms": spread_cms,
        "var_S1": var_S1,
        "var_S2": var_S2,
        "cov_S1S2": cov_S1S2,
        "corr": corr,
        "P0_Ta": P0_Ta,
        "cross_anchor_err": anchor_err,
        "price_atm": price_atm.item(),
        "otm": otm,
        "itm": itm,
        "intrinsic": intrinsic,
        "parity_resid": parity_resid,
        "var_spread": var_spread,
        "base_price": base_price,
        "decorr_price": decorr_price,
    }


def _numeraire_identity(model, fixing_date, seed, use_checkpoint):
    """
    Recompute P(0,T_N)·E^{T_N}[1/P(T_a,T_N)] and the deterministic P(0,T_a) for the
    numéraire-consistency check in self_validate_cms_spread (c). Mirrors the deflator/discount
    construction in cms_spread_option_price exactly (same grid truncation, same clamp).
    """
    a = round(fixing_date / config.GRID_STEP_SIZE)
    time_grid, n_steps = _truncated_grid(model, fixing_date, config.CMS_DT)
    F_paths = model.simulate_forward_curve(
        config.CMS_PATHS_MAIN,
        time_grid,
        freeze_drift=True,
        use_checkpoint=use_checkpoint,
        seed=seed,
    )
    deflator = _terminal_deflator(model, F_paths[:, n_steps], a)
    P0_T = _initial_bonds(model)
    lhs = (P0_T[-1] * torch.mean(deflator)).item()
    rhs = _initial_bond_at(model, P0_T, a).item()
    return lhs, rhs


def cms_convexity_benchmark(
    model,
    Sigma_matrix,
    fixing_date,
    swap_end,
    n_strikes=None,
    vol_stds=None,
):
    """
    Closed-form CMS convexity by static replication (Hagan 2003, "Convexity Conundrums";
    Andersen–Piterbarg Vol. III), with no path simulation. It replicates the T_a-forward CMS rate
    of the MC's deflate-to-T_a convention, E^{T_a}[S] = (A(0)/P(0,T_a))·E^A[S·α(S)] with
    α(S) = 1/A(T_a); a linear TSR α(S) ≈ P(0,T_a)/A(0) + a·(S−S_fwd) gives
    CMS_adj = (A(0)/P(0,T_a))·a·Var^A(S).

    Two arms. Flat (independent): Var = σ_market²·T_a from the market ATM quote, no MC or surrogate.
    Smile (not independent): the ODE-surrogate smile (mapped_smm_ode_pricer) rescaled to the market
    ATM and integrated by Carr–Madan; the raw surrogate ATM is reported too, since its level can differ
    from the market ATM. factor_tsr and factor_exact (exact-curve annuity slope) both act on the flat
    Var, separating annuity convention from vol level. Sigma_matrix must be the Σ baked into the model.
    """
    n_strikes = n_strikes if n_strikes is not None else config.CMS_BENCH_N_STRIKES
    vol_stds = vol_stds if vol_stds is not None else config.CMS_BENCH_VOL_STDS
    device, dtype = model.device, model.dtype

    assert model.beta_sabr == 0.0, "CMS benchmark supports the β=0 (Bachelier) default only"

    a = round(fixing_date / config.GRID_STEP_SIZE)
    b = round(swap_end / config.GRID_STEP_SIZE)
    assert abs(model.T[a].item() - fixing_date) < 1e-9, "fixing_date is not a tenor node"
    assert abs(model.T[b].item() - swap_end) < 1e-9, "swap_end is not a tenor node"
    assert a >= 1 and b > a, "need T_a >= first tenor and a non-empty swap window"

    tenor = float(swap_end - fixing_date)
    T_a = float(fixing_date)
    tau_a = float(config.GRID_STEP_SIZE)
    n_pay = b - a

    # initial-curve quantities (shared input, not MC output): same P0_T as the MC pricer
    P0_T = _initial_bonds(model)  # P(0,T_{k+1})
    P0_Ta = P0_T[a - 1]
    A0 = torch.sum(model.tau[a:b] * P0_T[a:b])
    S_fwd = ((P0_Ta - P0_T[b - 1]) / A0).item()
    A0_f, P0_Ta_f = A0.item(), P0_Ta.item()
    prefactor = A0_f / P0_Ta_f  # A(0)/P(0,T_a)

    # (i) linear-TSR slope: a = d/dS [1/A_bm(S)] at S_fwd, flat-yield bond-math annuity.
    S_t = torch.tensor(S_fwd, device=device, dtype=dtype, requires_grad=True)
    i_idx = torch.arange(1, n_pay + 1, device=device, dtype=dtype)
    A_bm = torch.sum(tau_a / (1.0 + tau_a * S_t) ** i_idx)
    (a_tsr,) = torch.autograd.grad(1.0 / A_bm, S_t)
    factor_tsr = prefactor * a_tsr.item()

    # (ii) exact-curve annuity slope: parallel-shift the window forwards, differentiate the
    # actual discounted α=P(0,T_a)/A(0) wrt the par rate (dα/dδ / dS/dδ). Real bonds, not flat
    # yield, isolate the TSR-convention share of any benchmark/MC gap.
    delta = torch.zeros((), device=device, dtype=dtype, requires_grad=True)
    mask = torch.zeros_like(model.F0.detach())
    mask[a:b] = 1.0
    F_sh = model.F0.detach() + delta * mask
    P0_sh = torch.cumprod(1.0 / (1.0 + model.tau * F_sh), dim=0)
    A_sh = torch.sum(model.tau[a:b] * P0_sh[a:b])
    S_sh = (P0_sh[a - 1] - P0_sh[b - 1]) / A_sh
    alpha_sh = P0_sh[a - 1] / A_sh
    (dalpha,) = torch.autograd.grad(alpha_sh, delta, retain_graph=True)
    (dS,) = torch.autograd.grad(S_sh, delta)
    factor_exact = prefactor * (dalpha / dS).item()

    # arm 1: market ATM normal vol (an input quote, not MC output), column "0.0" at Expiry == T_a,
    # else linear in expiry; this keeps the flat arm independent of the MC and the surrogate
    mkt_path = os.path.join(config.RESULTS_DIR, config.STAGE1_SURFACE_MARKET_CSV)
    surf = pd.read_csv(mkt_path, index_col="Expiry")
    exp_grid = surf.index.to_numpy(dtype=float)
    atm_vols = surf["0.0"].to_numpy(dtype=float)
    hit = np.abs(exp_grid - T_a) < 1e-6
    if hit.any():
        sigma_atm_market = float(atm_vols[hit][0])
    else:
        assert exp_grid.min() - 1e-9 <= T_a <= exp_grid.max() + 1e-9, (
            f"T_a {T_a} outside market-surface expiry range [{exp_grid.min()}, {exp_grid.max()}]"
        )
        order = exp_grid.argsort()
        sigma_atm_market = float(np.interp(T_a, exp_grid[order], atm_vols[order]))

    var_flat = sigma_atm_market**2 * T_a
    adj_flat = factor_tsr * var_flat  # arm 1: independent (market ATM × analytic TSR factor)
    adj_flat_exact = factor_exact * var_flat  # attribution: exact-annuity factor on flat Var

    # arm 2: surrogate smile (ODE, model-derived shape) rescaled to the market ATM level; keeps the
    # skew/curvature, replaces the surrogate level, not independent of the model.
    # Carr–Madan strip ±vol_stds·σ_market·√T_a; wings bound by surrogate tail accuracy.
    R = vol_stds * sigma_atm_market * (T_a**0.5)
    offsets = torch.linspace(-R, R, n_strikes, device=device, dtype=dtype)
    sigma_surr = (
        mapped_smm_ode_pricer(model, Sigma_matrix, [T_a], [tenor], offsets.tolist())
        .detach()
        .reshape(-1)
        .to(dtype)
    )
    assert sigma_surr.numel() == n_strikes, "surrogate smile shape mismatch"
    atm_idx = int(torch.argmin(torch.abs(offsets)))
    surrogate_atm = sigma_surr[atm_idx].item()
    sigma_rescaled = sigma_surr * (sigma_atm_market / surrogate_atm)

    K = S_fwd + offsets
    T_t = torch.tensor(T_a, device=device, dtype=dtype)
    F_t = torch.tensor(S_fwd, device=device, dtype=dtype)
    b_pay = torch_bachelier(F_t, K, T_t, sigma_rescaled)
    b_rec = b_pay - (S_fwd - K)  # normal put-call parity: receiver = payer − (F − K)
    integrand = torch.where(offsets < 0, b_rec, b_pay)
    var_smile = (2.0 * torch.trapezoid(integrand, K)).item()

    adj_smile = factor_tsr * var_smile  # arm 2: TSR slope × rescaled-smile Var
    skew_delta = adj_smile - adj_flat  # the smile's skew/curvature contribution

    return {
        "S_fwd": S_fwd,
        "S_cms_benchmark": S_fwd + adj_smile,  # full estimate (market level + surrogate skew)
        "adj_flat": adj_flat,  # arm 1, independent headline
        "adj_smile": adj_smile,  # arm 2, skew refinement (not independent)
        "skew_delta": skew_delta,
        "adj_flat_exact": adj_flat_exact,  # annuity-convention attribution on flat Var
        "sigma_atm_market": sigma_atm_market,
        "surrogate_atm": surrogate_atm,
        "var_flat": var_flat,
        "var_smile": var_smile,
        "factor_tsr": factor_tsr,
        "factor_exact": factor_exact,
    }


def compare_cms_mc_vs_benchmark(
    model,
    Sigma_matrix,
    fixing_date=5.0,
    swap_end=10.0,
    year_fraction=1.0,
    seed=None,
    use_checkpoint=False,
):
    """
    Run the MC convexity diagnostic and the replication benchmark on the same model/Σ/curve/trade.
    The verdict compares the flat arm (market ATM × analytic factor) with the MC in three tiers,
    since linear TSR vs the MC's exact pathwise annuity bounds the gap, not only MC noise:
      - same sign and order of magnitude      -> the cross-check passes
      - |diff| within max(8 bps, 5·stderr)    -> consistent within the convention difference
      - sign flip or order-of-magnitude miss  -> FAIL, investigate
    The smile arm is reported as the skew refinement and is not independent.
    """
    # MC diagnostic (deflate-to-T_a => payment at T_a; strike is irrelevant to S_cms).
    _, diag = cms_caplet_floorlet_price(
        model,
        fixing_date,
        swap_end,
        strike=0.0,
        year_fraction=year_fraction,
        is_cap=True,
        seed=seed,
        use_checkpoint=use_checkpoint,
        return_diagnostics=True,
    )
    S_fwd, S_cms, stderr = diag["S_fwd"], diag["S_cms"], diag["S_cms_stderr"]
    adj_mc = S_cms - S_fwd

    bench = cms_convexity_benchmark(model, Sigma_matrix, fixing_date, swap_end)
    adj_flat = bench["adj_flat"]  # independent arm carries the verdict
    adj_smile = bench["adj_smile"]
    diff = adj_flat - adj_mc

    adj_mc_bps, adj_flat_bps, stderr_bps = adj_mc * 1e4, adj_flat * 1e4, stderr * 1e4

    same_sign = (adj_mc > 0) == (adj_flat > 0)
    lo, hi = min(abs(adj_mc_bps), abs(adj_flat_bps)), max(abs(adj_mc_bps), abs(adj_flat_bps))
    same_order = lo > 1.0 and hi < 10.0 * max(lo, 1e-12)
    within_conv = abs(diff * 1e4) <= max(8.0, 5.0 * stderr_bps)

    if not same_sign or not same_order:
        verdict = "FAIL — flat arm sign flip / order-of-magnitude miss; investigate"
    elif within_conv:
        verdict = "PASS (tight) — flat arm consistent with MC within convention difference"
    else:
        verdict = "PASS (cross-check) — flat arm same sign & order as MC; gap is TSR convention"

    print("  CMS convexity: MC vs closed-form replication")
    print(f"    S_fwd              : {S_fwd * 1e4:.2f}bps")
    print(
        f"    ATM normal vol     : market={bench['sigma_atm_market'] * 1e4:.2f}bps  "
        f"surrogate(raw)={bench['surrogate_atm'] * 1e4:.2f}bps"
    )
    print(f"    adj MC             : {adj_mc_bps:+.2f}bps  (stderr {stderr_bps:.2f}bps)")
    print(f"    adj FLAT (indep)   : {adj_flat_bps:+.2f}bps  [market ATM × analytic factor]")
    print(
        f"    adj SMILE (skew)   : {adj_smile * 1e4:+.2f}bps  [surrogate shape, NOT independent]"
        f"  skew Δ={bench['skew_delta'] * 1e4:+.2f}bps"
    )
    print(
        f"    annuity factor     : TSR-slope={adj_flat_bps:+.2f}bps  "
        f"exact-annuity={bench['adj_flat_exact'] * 1e4:+.2f}bps  (on flat Var)"
    )
    print(f"    diff (FLAT−MC)     : {diff * 1e4:+.2f}bps")
    print(f"    verdict            : {verdict}")

    return {
        "S_fwd": S_fwd,
        "adj_mc": adj_mc,
        "adj_flat": adj_flat,
        "adj_smile": adj_smile,
        "diff": diff,
        "stderr": stderr,
        "verdict": verdict,
        **bench,
    }
