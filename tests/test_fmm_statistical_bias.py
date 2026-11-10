"""
Engine-B statistical-bias & MC-convergence diagnostic.

Engine B = the Rough SABR FMM valuation engine (jolto.valuation.bermudan_swaption,
jolto.core.model.RoughSABRFMM). The instrument is a single-exercise **co-terminal European**
swaption (Ex_Dates=[T_a]). It exercises the engine (kernel convolution, correlation projection
through L, frozen drift, path integration, terminal-measure deflator) and nothing else: with
n_ex=1 the LSM backward loop is empty, so there is no exercise boundary. The European is observed
at T_a, so the path is simulated only to T_a (truncated grid), much cheaper than the 30Y
Bermudan while keeping the heavy right tail of the terminal deflator.

Strike = model ATM: K = S₀, the time-0 co-terminal forward swap rate computed per model
(S₀=(1−P_TN)/A from model.F0/tau), the max-vega point where the cleanest bias measurement
lives. Computed, never configured.

Convergence reference = a **PRNG oracle**: the same European priced with driver_type="prng" at
identical dt (driver-only swap, no approximation between arms). Welch t-test (Satterthwaite df).
Single estimator `is_brute` (QMC): the split_sample/pilot arm is degenerate for a European
(no boundary ⇒ identical to is_brute) and is dropped; the `estimator` column is kept constant
for schema parity with the Engine-A CSV. The Gaussian-CV diag is retained only to report the
variance-reduction factor and the coefficient of variation of the deflated cashflow.

Two distinct, never-conflated metrics:
  • bias_bps        = PRNG-oracle gap = FMM engine statistical bias, the headline.
  • xengine_gap_bps = SMM-mapping fidelity: FMM-European − Engine-A-European (the 1-D process
    Adachi-mapped at T_a via Σ, priced through mc_rough_sabr_pricer). Carries the mapping
    approximation, not engine bias. Reported with its combined SE. Production + sweep rows only.

Two modes (config.CONVERGENCE; param source is the only switch):
  (a) production: calibrated model (H=VAL_H_TARGET, production Σ via build_calibrated_model);
  (b) sweep:      coarse (ρ,ν) grid on a parametric injected-Σ model (not calibrated).

Engine B is float64 throughout; the cross-engine Engine-A call uses dt=PRICE_DT + float64 so the
gap isolates mapping fidelity, not a grid/precision difference.

Usage:  python tests/test_fmm_statistical_bias.py [--full]
Outputs: results/mc_convergence_engineB.csv
"""

import math
import os

import numpy as np
import pandas as pd
import scipy.stats as stats
import torch

from jolto import config, val_config
from jolto.api.assembly import build_calibrated_model, load_valuation_inputs
from jolto.calibration.pricers import (
    INV_SQRT_2PI,
    apply_sigma_to_smm,
    bachelier_iv_newton,
    mc_rough_sabr_pricer,
    precompute_smm_weights,
)
from jolto.core.model import RoughSABRFMM
from jolto.valuation.cms import _inject_sigma
from jolto.valuation.bermudan_swaption import (
    bermudan_pricer,
    oos_bermudan_pricer,
    _terminal_deflator_mean,
)
from _common import _cli_profile, _slope, _welch

DEVICE = "cpu"
DT = val_config.PRICE_DT  # production FMM pricing grid (Engine B); also the cross-engine A grid
H = val_config.VAL_H_TARGET
SEED = val_config.PRICE_SEED
N_PROD = val_config.PRICE_PATHS_MAIN  # production path count → the bias verdict point
CFG = config.CONVERGENCE
T_A = CFG["euro_expiry_B"]  # European fixing; swap runs co-terminal T_a → T_N
ESTIMATORS = ("is_brute",)  # split_sample dropped (degenerate at n_ex=1); column kept for parity
XENGINE_COLS = {"xengine_iv_bps": np.nan, "xengine_gap_bps": np.nan, "xengine_se_bps": np.nan}
CV_DEFLATOR_COLS = {
    "se_cv_deflator": np.nan,
    "vr_deflator": np.nan,
    "bias_cv_bps": np.nan,
    "p_value_cv": np.nan,
    "in_oracle_CI_cv": np.nan,
}


def _euro_grid(T_a):
    """Time grid truncated to T_a: the European is observed there, so the path need not reach T_N."""
    n = int(math.ceil(T_a / DT))
    return torch.linspace(0.0, T_a, n + 1, device=DEVICE, dtype=torch.float64)


def _atm_strike_annuity(model, T_a):
    """K=S₀ (co-terminal forward swap rate) + time-0 annuity A₀, from model.F0/tau (the
    _bermudan_gaussian_cv expression; k0 = first forward at/after T_a). Computed, not configured."""
    k0 = int(torch.sum(model.T[:-1] < T_a - 1e-6).item())
    taus = model.tau[k0:]
    P = torch.cumprod(1.0 / (1.0 + taus * model.F0[k0:].detach()), dim=0)
    S0 = float((1.0 - P[-1]) / torch.sum(taus * P))
    # cumprod from k0 gives forward factors P(T_a,T_j), so Σ τ P = A(0)/P(0,T_a). bermudan_pricer
    # returns a time-zero value, so the Bachelier inversion needs A(0) = Σ τ_j P(0,T_j): rescale
    # by P(0,T_a). S0 is a ratio in which the factor cancels and is left untouched.
    P0_Ta = torch.prod(1.0 / (1.0 + model.tau[:k0] * model.F0[:k0].detach()))
    A0 = float(torch.sum(taus * P) * P0_Ta)
    return S0, A0


def _t_ci(samples, level):
    n = len(samples)
    m = float(np.mean(samples))
    se = float(np.std(samples, ddof=1)) / math.sqrt(n)
    tcrit = stats.t.ppf(0.5 + level / 2.0, df=n - 1)
    return m, se, m - tcrit * se, m + tcrit * se, n - 1


def _oracle(model, trade, tg, total, batch, cell=0):
    """Batched co-terminal-European PRNG ground truth (driver_type='prng'). bps. t-CI.
    `cell` offsets the base stream and the near-noise seeds so each cell is an independent sample."""
    nb = total // batch
    torch.manual_seed(
        SEED + cell
    )  # reproducible base-increment stream; near-noise varies via seed=b
    px = []
    with torch.no_grad():
        for b in range(nb):
            p = bermudan_pricer(
                model,
                trade,
                batch,
                tg,
                freeze_drift=True,
                seed=b + 100_000 * cell,
                driver_type="prng",
            )
            px.append(float(p) * 1e4)
    return _t_ci(np.asarray(px), CFG["ci_level"])


def _estimator_means(model, trade, tg, N, m_seeds, cell=0):
    """m_seeds independent QMC scrambles of the is_brute European. bps. Seeds SEED+1000·cell+s."""
    px = []
    with torch.no_grad():
        for s in range(m_seeds):
            px.append(
                float(
                    bermudan_pricer(
                        model, trade, N, tg, freeze_drift=True, seed=SEED + 1000 * cell + s
                    )
                )
                * 1e4
            )
    return np.asarray(px)


def _estimator_means_cv(model, trade, tg, N, m_seeds, cell=0):
    """m_seeds OOS CV-corrected co-terminal-European prices. bps.

    Calls oos_bermudan_pricer(use_control_variate=True), the production
    estimator (pilot→main OOS beta, PRICE_PATHS_PILOT pilot). Same Sobol seed as
    _estimator_means, so CV and raw arms share the same main-run draws."""
    px = []
    with torch.enable_grad():  # _bermudan_gaussian_cv calls autograd.grad internally
        for s in range(m_seeds):
            p = oos_bermudan_pricer(
                model,
                trade,
                n_paths_main=N,
                n_paths_pilot=val_config.PRICE_PATHS_PILOT,
                time_grid=tg,
                freeze_drift=True,
                seed=SEED + 1000 * cell + s,
                use_control_variate=True,
            )
            px.append(float(p) * 1e4)
    return np.asarray(px)


def _per_path_diag(model, trade, tg, N, m_def):
    """Single-rep SE diagnostics (6 values) in bps.

    Gaussian-only (se_cv, vr_factor): scalar in-sample beta on pay_cv.
    Deflator CV (se_cv_deflator, vr_deflator): production OOS pilot beta applied to main-run
    per-path tensors. Pilot uses seed=SEED+999 and PRICE_PATHS_PILOT paths, matching the
    oos_bermudan_pricer OOS structure used in _estimator_means_cv."""
    P0_TN = float(torch.cumprod(1.0 / (1.0 + model.tau * model.F0), dim=0)[-1])
    with torch.enable_grad():  # _bermudan_gaussian_cv calls autograd.grad internally
        # pilot: OOS beta via the same structure as oos_bermudan_pricer (seed+999, PILOT paths).
        # β is the (1+n_ex)-vector; cond is the design-collinearity diagnostic (dropped here).
        _, beta_pilot, _cond = bermudan_pricer(
            model,
            trade,
            val_config.PRICE_PATHS_PILOT,
            tg,
            freeze_drift=True,
            seed=SEED + 999,
            use_control_variate=True,
            return_weights=True,
            return_beta=True,
        )
        # main: per-path tensors at seed=SEED. D_stack/m_stack/cv_flag_stack are per-date stacks
        # (n_ex columns); on the single-exercise European n_ex=1, so each is a single column.
        cf, pay_cv, true_cv_mean, D_stack, m_stack, cv_flag_stack = bermudan_pricer(
            model,
            trade,
            N,
            tg,
            freeze_drift=True,
            seed=SEED,
            use_control_variate=True,
            return_cv_tensors=True,
        )
    # cross-check: the pricer's on-tape terminal mean equals the harness-computed m_def
    assert abs(float(m_stack[-1]) - float(m_def)) < 1e-6 * max(abs(float(m_def)), 1.0), (
        f"m_stack[-1]={float(m_stack[-1])!r} vs m_def={float(m_def)!r}"
    )
    cf = cf.double()
    se_raw = P0_TN * float(cf.std()) / math.sqrt(N) * 1e4

    # Gaussian-only scalar beta
    beta_g = ((cf - cf.mean()) * (pay_cv.double() - pay_cv.double().mean())).mean() / torch.clamp(
        pay_cv.double().var(), min=1e-24
    )
    vr_gauss = cf - beta_g * (pay_cv.double() - true_cv_mean)
    se_cv = P0_TN * float(vr_gauss.std()) / math.sqrt(N) * 1e4
    deflator_cv = float(cf.std() / torch.clamp(cf.mean(), min=1e-12))

    # Vector deflator CV: production OOS β-vector applied to main-run per-path tensors. The
    # deflator-control guard zeroes only each βᵢ deflator term on its own flagged paths (identical
    # per-component mask to the pricer's production CV branch → consistent vr_deflator). On the
    # European (n_ex=1) this reduces to the 2-control expression exactly.
    def_terms = beta_pilot[1:] * (D_stack.double() - m_stack.double())  # (N, n_ex)
    def_terms = torch.where(cv_flag_stack, torch.zeros_like(def_terms), def_terms)
    vr_def = cf - beta_pilot[0] * (pay_cv.double() - true_cv_mean) - def_terms.sum(dim=1)
    se_cv_deflator = P0_TN * float(vr_def.std()) / math.sqrt(N) * 1e4
    vr_deflator = (se_raw / max(se_cv_deflator, 1e-30)) ** 2

    return (
        se_raw,
        se_cv,
        (se_raw / max(se_cv, 1e-30)) ** 2,
        deflator_cv,
        se_cv_deflator,
        vr_deflator,
    )


def _xengine_gap(model, T_a, A0, fmm_pv_bps, fmm_se_bps, N):
    """Cross-engine SMM-mapping fidelity: FMM-European − Engine-A-European, in normal-vol bps.

    Maps through precompute_smm_weights / apply_sigma_to_smm, prices with mc_rough_sabr_pricer and
    inverts with bachelier_iv_newton; Σ reconstructed from the model's Cholesky loadings (rate-rate block +
    vol-rate column 0). Both legs at K=S₀ (relative offset 0 → ATM vega = √T·1/√2π), dt=PRICE_DT,
    float64. Combined SE = hypot of the two per-path vega-scaled SEs."""
    T_N = float(model.T[-1])
    Sigma = model.loadings @ model.loadings.T
    cache = precompute_smm_weights(model, [T_a], [T_N - T_a])
    a_smm, r_smm = apply_sigma_to_smm(Sigma, cache)

    K_rel = torch.tensor([0.0], dtype=torch.float64, device=DEVICE)
    T_t = torch.tensor([T_a], dtype=torch.float64, device=DEVICE)
    with torch.no_grad():
        price_A, std_A = mc_rough_sabr_pricer(
            K_rel,
            T_t,
            a_smm,
            r_smm,
            model.nu,
            model.H,
            n_paths=N,
            dt=DT,
            device=DEVICE,
            return_std=True,
            dtype=torch.float64,
        )
    vega = math.sqrt(T_a) * INV_SQRT_2PI  # ATM (K_rel=0)
    iv_A = float(bachelier_iv_newton(price_A, K_rel, T_t, initial_guess_vol=a_smm)) * 1e4
    se_A = float(std_A) / math.sqrt(N) / vega * 1e4

    pv_per_annuity = torch.tensor([fmm_pv_bps * 1e-4 / A0], dtype=torch.float64, device=DEVICE)
    iv_fmm = float(bachelier_iv_newton(pv_per_annuity, K_rel, T_t, initial_guess_vol=a_smm)) * 1e4
    se_fmm = (fmm_se_bps * 1e-4 / A0) / vega * 1e4
    return {
        "xengine_iv_bps": iv_A,
        "xengine_gap_bps": iv_fmm - iv_A,
        "xengine_se_bps": math.hypot(se_fmm, se_A),
    }


def _eval(param_source, label, model, trade, tg, N, m_seeds, orc, diag, cv_est=None, cell=0):
    """CSV rows for the is_brute European at one N against a precomputed oracle (xengine NaN here;
    stamped by the caller on production/sweep rows). cv_est: pre-computed _estimator_means_cv array
    (N_PROD only); None → CV bias columns left NaN. se_cv_deflator/vr_deflator filled from diag
    on every row regardless."""
    orc_m, orc_se, orc_lo, orc_hi, orc_df = orc
    se_raw, se_cv, vr, dcv, se_cv_deflator, vr_deflator = diag
    rows = []
    for kind in ESTIMATORS:
        est = _estimator_means(model, trade, tg, N, m_seeds, cell)
        em, ese = float(est.mean()), float(est.std(ddof=1)) / math.sqrt(m_seeds)
        bias = em - orc_m
        t, df, p = _welch(bias, ese, m_seeds - 1, orc_se, orc_df)

        cv_cols = {**CV_DEFLATOR_COLS, "se_cv_deflator": se_cv_deflator, "vr_deflator": vr_deflator}
        if cv_est is not None:
            cv_em = float(cv_est.mean())
            cv_ese = float(cv_est.std(ddof=1)) / math.sqrt(m_seeds)
            cv_bias = cv_em - orc_m
            _, _, cv_p = _welch(cv_bias, cv_ese, m_seeds - 1, orc_se, orc_df)
            cv_cols["bias_cv_bps"] = cv_bias
            cv_cols["p_value_cv"] = cv_p
            cv_cols["in_oracle_CI_cv"] = bool(orc_lo <= cv_em <= orc_hi)

        rows.append(
            {
                "param_source": param_source,
                "instrument": label,
                "estimator": kind,
                "N": N,
                "price": em,
                "se_raw": se_raw,
                "se_cv": se_cv,
                "vr_factor": vr,
                "oracle_price": orc_m,
                "bias_bps": bias,
                "welch_t": t,
                "welch_df": df,
                "p_value": p,
                "in_oracle_CI": bool(orc_lo <= em <= orc_hi),
                # a right-skewed scramble distribution makes the Welch t anti-conservative low
                "seed_skew": float(stats.skew(est)),
                "deflator_cv": dcv,
                **cv_cols,
                **XENGINE_COLS,
            }
        )
    return rows


def _parametric_sigma(rho_v, beta_decay, tenors):
    """PSD (N+1)×(N+1) parametric correlation: exp-decay rate-rate block + flat vol-rate ρ
    (index 0 = vol driver). Eigen-clipped to PSD + re-normalised to unit diagonal. The model's
    omega=None fallback leaves the vol row at identity (ρ inert), so ρ
    must be injected here to enter the sim (verified to move the price)."""
    Ti = tenors[:-1]
    n = len(Ti)
    S = np.eye(n + 1)
    S[1:, 1:] = np.exp(-beta_decay * np.abs(Ti[:, None] - Ti[None, :]))
    S[0, 1:] = rho_v
    S[1:, 0] = rho_v
    w, V = np.linalg.eigh(S)
    S = V @ np.diag(np.clip(w, 1e-8, None)) @ V.T
    d = np.sqrt(np.diag(S))
    return torch.tensor(S / np.outer(d, d), dtype=torch.float64, device=DEVICE)


def _parametric_model(rho_v, nu_v):
    """Mode (b): parametric (not calibrated) model with scalar ν, ρ via an injected PSD Σ."""
    tenors = np.arange(0.0, config.MAX_MATURITY + config.GRID_STEP_SIZE, config.GRID_STEP_SIZE)
    F0 = np.full(len(tenors) - 1, val_config.PRICE_STRIKE)

    def flat(v):  # constant term-structure callable expected by RoughSABRFMM
        return lambda t: np.full_like(np.asarray(t, float), v)

    model = RoughSABRFMM(
        tenors,
        F0,
        flat(CFG["sweep_alpha"]),
        flat(rho_v),
        nu_v,
        H,
        beta_decay=CFG["beta_decay_B"],
        beta_sabr=config.BETA_SABR,
        shift=config.SHIFT_SABR,
        device=DEVICE,
    )
    _inject_sigma(model, _parametric_sigma(rho_v, CFG["beta_decay_B"], tenors))
    return model


def main():
    prof_key, prof = _cli_profile(CFG)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    print("\n" + "=" * 84)
    print(f"{'ENGINE B — STATISTICAL BIAS & MC CONVERGENCE (co-terminal European)':^84}")
    print("=" * 84)
    print(
        f"  profile={prof_key}  H={H}  dt={DT:.5f}  seed={SEED}  N_prod={N_PROD}  "
        f"T_a={T_A:.0f}Y  M_seeds={prof['m_seeds_B']}"
    )

    rows = []

    # Mode (a): production point, calibrated model, ATM co-terminal European
    inputs = load_valuation_inputs(val_config)
    model, _, _ = build_calibrated_model(inputs, DEVICE)
    T_N = float(model.T[-1])
    S0, A0 = _atm_strike_annuity(model, T_A)
    trade = {"Strike": S0, "Ex_Dates": [T_A]}
    tg = _euro_grid(T_A)
    label = f"{T_A:.0f}Yx{T_N - T_A:.0f}Y@ATM"
    # analytic deflator mean m_def = A(0)/P(0,T_N) for the co-terminal swap fixing at T_A;
    # k0 = first forward at/after T_A (same convention as _atm_strike_annuity).
    k0 = int(torch.sum(model.T[:-1] < T_A - 1e-6).item())
    m_def = _terminal_deflator_mean(model, k0)
    print(
        f"\n[mode a] European {label}  K=S₀={S0:.5f}  ({int(round(T_A / DT))} steps)  "
        f"oracle N={prof['oracle_n_B']:,} ({prof['oracle_n_B'] // prof['oracle_batch_B']} batches)"
    )
    orc = _oracle(model, trade, tg, prof["oracle_n_B"], prof["oracle_batch_B"])
    print(f"  oracle: {orc[0]:.3f} bps  SE {orc[1]:.3f} bps (df={orc[4]})")

    print(
        f"[mode a] SE-vs-N ladder + bias: N={prof['n_ladder_B']}  (N_prod={N_PROD} = verdict point)"
    )
    ladder_se_raw, ladder_Ns = [], []
    # Union with N_PROD so the production verdict is always evaluated at the true N_PROD in both
    # profiles, decoupled from the ladder's contents (the reduced ladder does not contain it).
    # The slope fit just gains a point.
    for N in sorted(set(prof["n_ladder_B"]) | {N_PROD}):
        diag = _per_path_diag(model, trade, tg, N, m_def)
        src = "production" if N == N_PROD else "ladder"
        cv_est = None
        if N == N_PROD:
            print(f"    [N={N_PROD}] running CV bias arm ({prof['m_seeds_B']} seeds) ...")
            cv_est = _estimator_means_cv(model, trade, tg, N, prof["m_seeds_B"])
        erows = _eval(src, label, model, trade, tg, N, prof["m_seeds_B"], orc, diag, cv_est=cv_est)
        if N == N_PROD:
            xe = _xengine_gap(model, T_A, A0, erows[0]["price"], erows[0]["se_raw"], N)
            for r in erows:
                r.update(xe)
        rows += erows
        ladder_se_raw.append(diag[0])
        ladder_Ns.append(N)
        print(
            f"    N={N:7d}  se_raw={diag[0]:.3f}  se_cv={diag[1]:.3f}  VR={diag[2]:.2f}  "
            f"se_cv_def={diag[4]:.3f}  VR_def={diag[5]:.2f}  deflator_CV={diag[3]:.2f}"
        )
    sl = _slope(np.array(ladder_Ns), np.array(ladder_se_raw))
    print(f"  SE-vs-N log-log slope = {sl:+.3f}  (expect −0.5)")

    # Mode (a), continued: production-point bias at additional co-terminal expiries
    _extra = [e for e in CFG["euro_expiries_B"] if e != T_A]
    if _extra:
        print(f"\n[mode a] production-point bias at additional expiries: {_extra}")
    for cell_x, T_a_x in enumerate(_extra, start=1):
        S0_x, A0_x = _atm_strike_annuity(model, T_a_x)
        trade_x = {"Strike": S0_x, "Ex_Dates": [T_a_x]}
        tg_x = _euro_grid(T_a_x)
        label_x = f"{T_a_x:.0f}Yx{T_N - T_a_x:.0f}Y@ATM"
        k0_x = int(torch.sum(model.T[:-1] < T_a_x - 1e-6).item())
        m_def_x = _terminal_deflator_mean(model, k0_x)
        orc_x = _oracle(model, trade_x, tg_x, prof["oracle_n_B"], prof["oracle_batch_B"], cell_x)
        print(f"  oracle {label_x}: {orc_x[0]:.3f} bps  SE {orc_x[1]:.3f} bps (df={orc_x[4]})")
        diag_x = _per_path_diag(model, trade_x, tg_x, N_PROD, m_def_x)
        print(f"    [N={N_PROD}] running CV bias arm ({prof['m_seeds_B']} seeds) ...")
        cv_est_x = _estimator_means_cv(model, trade_x, tg_x, N_PROD, prof["m_seeds_B"], cell_x)
        erows_x = _eval(
            "production",
            label_x,
            model,
            trade_x,
            tg_x,
            N_PROD,
            prof["m_seeds_B"],
            orc_x,
            diag_x,
            cv_est=cv_est_x,
            cell=cell_x,
        )
        xe_x = _xengine_gap(model, T_a_x, A0_x, erows_x[0]["price"], erows_x[0]["se_raw"], N_PROD)
        for r in erows_x:
            r.update(xe_x)
        rows += erows_x
        print(
            f"    {label_x}: bias={erows_x[0]['bias_bps']:+.3f} bps  p={erows_x[0]['p_value']:.3f}  "
            f"skew={erows_x[0]['seed_skew']:.2f}  "
            f"in-CI={erows_x[0]['in_oracle_CI']}  VR_def={diag_x[5]:.2f}  "
            f"xengine_gap={xe_x['xengine_gap_bps']:+.3f}±{xe_x['xengine_se_bps']:.3f}"
        )

    # Mode (b): coarse (ρ,ν) sweep on parametric Σ, reduced oracle N
    print(f"\n[mode b] parametric-Σ sweep  oracle N={prof['oracle_n_B_sweep']:,}")
    in_ci = []
    sweep_cell = 10
    for rho_v in CFG["sweep_rho_B"]:
        for nu_v in CFG["sweep_nu_B"]:
            sweep_cell += 1
            pm = _parametric_model(rho_v, nu_v)
            pS0, pA0 = _atm_strike_annuity(pm, T_A)
            ptrade = {"Strike": pS0, "Ex_Dates": [T_A]}
            ptg = _euro_grid(T_A)
            tag = f"rho{rho_v:+.2f}_nu{nu_v:.2f}"
            pm_k0 = int(torch.sum(pm.T[:-1] < T_A - 1e-6).item())
            pm_m_def = _terminal_deflator_mean(pm, pm_k0)
            o = _oracle(
                pm, ptrade, ptg, prof["oracle_n_B_sweep"], prof["oracle_batch_B"], sweep_cell
            )
            diag = _per_path_diag(pm, ptrade, ptg, N_PROD, pm_m_def)
            r = _eval(
                f"sweep[{tag}]",
                tag,
                pm,
                ptrade,
                ptg,
                N_PROD,
                prof["m_seeds_B"],
                o,
                diag,
                cell=sweep_cell,
            )
            xe = _xengine_gap(pm, T_A, pA0, r[0]["price"], r[0]["se_raw"], N_PROD)
            for row in r:
                row.update(xe)
            rows += r
            in_ci += [x["in_oracle_CI"] for x in r]
            print(
                f"    {tag}: oracle {o[0]:.2f}±{o[1]:.2f}  bias={r[0]['bias_bps']:+.3f} bps  "
                f"in-CI {sum(x['in_oracle_CI'] for x in r)}/{len(r)}  "
                f"xengine_gap={xe['xengine_gap_bps']:+.3f}±{xe['xengine_se_bps']:.3f}  "
                f"deflator_CV={diag[3]:.2f}  VR_def={diag[5]:.2f}"
            )
    print(
        f"  [sweep summary] {sum(in_ci)}/{len(in_ci)} cells within the oracle {CFG['ci_level']:.0%} CI"
    )

    df = pd.DataFrame(rows)
    out = os.path.join(config.RESULTS_DIR, "mc_convergence_engineB.csv")
    df.to_csv(out, index=False)

    prod_rows = df[df.param_source == "production"]
    print(f"\n{'=' * 84}\n  PRODUCTION-POINT UNBIASEDNESS (mode a, N={N_PROD}, K=S₀ per expiry)")
    for _, prod in prod_rows.iterrows():
        verdict = "INDISTINGUISHABLE from 0" if float(prod.p_value) > 0.05 else "SIGNIFICANT"
        cv_verdict = "INDISTINGUISHABLE from 0" if float(prod.p_value_cv) > 0.05 else "SIGNIFICANT"
        print(f"\n  [{prod.instrument}]")
        print(
            f"    FMM engine bias  : {float(prod.bias_bps):+.3f} bps  p={float(prod.p_value):.3f}  "
            f"seed skew={float(prod.seed_skew):.2f}  "
            f"in-CI={bool(prod.in_oracle_CI)}  → {verdict}"
        )
        print(
            f"    CV-deflator arm  : {float(prod.bias_cv_bps):+.3f} bps  p={float(prod.p_value_cv):.3f}  "
            f"in-CI={bool(prod.in_oracle_CI_cv)}  VR={float(prod.vr_deflator):.2f}x  → {cv_verdict}"
        )
        print(
            f"    SMM-map fidelity : xengine_gap={float(prod.xengine_gap_bps):+.3f} ± "
            f"{float(prod.xengine_se_bps):.3f} bps  (vs Engine-A SMM IV {float(prod.xengine_iv_bps):.2f} bps)"
        )
        print(
            f"    deflator CV ≈ {float(prod.deflator_cv):.2f} (terminal-deflator tail)  "
            f"Gaussian VR≈{float(prod.vr_factor):.2f}"
        )
    print(f"\n  SE-vs-N slope (5Y×26Y) {sl:+.3f} (−0.5 ref).  wrote {out}\n")


if __name__ == "__main__":
    main()
