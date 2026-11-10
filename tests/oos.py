"""
Out-of-sample repricing exhibit.

Reprices European swaptions the calibration never saw, with two engines, and reports the
error distribution with a block bootstrap confidence interval.

Columns
-------
SMM : mapped one-dimensional conditional MC engine at the mapped parameters.
FMM : full joint valuation engine under the terminal measure, physically settled
      payoff deflated by P(T_a,T_N) and scaled by P(0,T_N), inverted to a Bachelier normal
      vol at A(0) and S(0).

The FMM engine prices co-terminal swaptions only: the underlying runs from the exercise
date to the end of the model's forward grid. Each (expiry, tenor) pair therefore gets its
own model truncated so the grid ends at expiry+tenor, with Sigma restricted to the leading
principal block. Truncation changes the terminal measure and hence the frozen drift, since
mu_j(0) sums over k = j+1..N and N differs: under an exact model the numeraire change is
exact and prices agree, under frozen drift they do not agree exactly. Disclosed choice.

Exhibit definition lives in the module constants below, is exposed as keyword arguments of
run_exhibit, and is stamped into results/oos_exhibit.json, so the values travel with the
artefact they produced.
"""

import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from jolto import config, val_config
from jolto.core.model import RoughSABRFMM
from jolto.calibration.pricers import (
    INV_SQRT_2PI,
    _compute_mapped_smm_parameters,
    bachelier_iv_newton,
    oos_mc_rough_sabr_pricer,
)
from jolto.core.market_data import (
    load_discount_curve,
    load_atm_matrix,
    load_swaption_vol_surface,
    bootstrap_forward_rates,
    parse_tenor,
    parse_strike,
)
from jolto.calibration import CorrelationCalibrator
from jolto.valuation.bermudan_swaption import bermudan_pricer
from jolto.valuation.cms import _inject_sigma

OOS_TENORS = (2.0, 5.0, 7.0, 10.0)
OOS_EXPIRIES = (1.0, 2.0, 3.0, 5.0, 10.0)
OOS_STRIKE_OFFSETS = (-0.01, -0.005, 0.005, 0.01)  # ATM is the in-sample control
ATM_CONTROL_STRIKE = 0.0  # second in-sample control, never an out-of-sample statistic
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260805
BOOTSTRAP_CI = 0.95
CONTROL_EXPIRIES = (1.0, 5.0, 10.0)
CONTROL_TOL_SE = 3.0
SE_REPLICATES = 5
SE_BASE_SEED = int(val_config.PRICE_SEED)
SE_SEED_STRIDE = 10007
SMM_SE_SCRAMBLES = 30  # SMM leg: SE of the reported run from its spread across scrambles

FREEZE_DRIFT = val_config.PRICE_FREEZE_DRIFT  # FMM leg drift treatment; --no-freeze-drift overrides

H_TARGET = val_config.VAL_H_TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / config.DATA_DIR
RES_DIR = PROJECT_ROOT / config.RESULTS_DIR
DISC_CSV = DATA_DIR / config.DISCOUNT_CSV
VOL_CSV = DATA_DIR / config.VOL_SURFACE_CSV
S1_CSV = RES_DIR / config.STAGE1_PARAMS_CSV
S1_MODEL_CSV = RES_DIR / config.STAGE1_SURFACE_MODEL_CSV_TPL.format(method=config.CALI_MODE)
SIGMA_CSV = RES_DIR / config.STAGE2_CORR_CSV_TPL.format(h=config.STAGE2_H)
OUT_JSON = RES_DIR / config.OOS_EXHIBIT_JSON

GRID_STEP_SIZE = config.GRID_STEP_SIZE
MAX_MATURITY = config.MAX_MATURITY
N_PATHS_MC = int(config.CALI_PATHS)
DT_VAL = float(config.CALI_DT)


class ExhibitError(RuntimeError):
    """Hard stop: a precondition failed. Never widened, never worked around."""


def _stamp(path):
    p = Path(path)
    if not p.is_file():
        raise ExhibitError(f"input not found: {p}")
    raw = p.read_bytes()
    return {
        "path": str(p.resolve()),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        "bytes": len(raw),
    }


def _csv(path, **kw):
    return pd.read_csv(path, float_precision="round_trip", **kw)


EXPECTED_CUBE_DELIM = ";"


def load_vol_cube(csv_path):
    # Sniff rather than hardcode, but a wrong sniff would silently mangle the cube.
    with open(csv_path, newline="") as fh:
        delim = csv.Sniffer().sniff(fh.readline()).delimiter
    if delim != EXPECTED_CUBE_DELIM:
        raise ExhibitError(
            f"sniffed delimiter {delim!r} for {csv_path}, expected {EXPECTED_CUBE_DELIM!r}"
        )
    df = pd.read_csv(csv_path, sep=delim, float_precision="round_trip")
    df.columns = [c.strip().upper() for c in df.columns]
    expiry_col = next(c for c in df.columns if "EXPIRY" in c)
    tenor_col = next(c for c in df.columns if "UNDERLYING" in c or "TENOR" in c)
    strike_col = next(c for c in df.columns if "STRIKE" in c)
    vol_col = next(c for c in df.columns if ("VALUE" in c or "VOL" in c) and "ID" not in c)
    df["Expiry"] = df[expiry_col].apply(parse_tenor)
    df["Tenor"] = df[tenor_col].apply(parse_tenor)
    df["Strike"] = df[strike_col].apply(parse_strike)
    df["Vol"] = pd.to_numeric(df[vol_col], errors="coerce")
    return df.dropna(subset=["Expiry", "Tenor", "Strike", "Vol"]).reset_index(drop=True), delim


def _key(e, t, k):
    return (round(float(e), 6), round(float(t), 6), round(float(k), 8))


def derive_exclusion_set(grid_T):
    """Replays both calibration selections rather than assuming either."""
    # Stage 1: jolto calibrate-sabr -> load_swaption_vol_surface(vol_csv, STAGE1_TARGET_TENOR),
    # which keeps |tenor - target| < 0.05 and every strike at every expiry of that column.
    s1_surface = load_swaption_vol_surface(str(VOL_CSV), config.STAGE1_TARGET_TENOR)
    stage1 = {
        _key(exp, config.STAGE1_TARGET_TENOR, k)
        for exp in s1_surface.index
        for k in s1_surface.columns
        if not pd.isna(s1_surface.loc[exp, k])
    }

    # Stage 2: CorrelationCalibrator.__init__ ingests load_atm_matrix (ATM rows only), and
    # _build_correlation_target_grid keeps |expiry + tenor - grid_T[i+1]| < 1e-4 for
    # i in range(2, N) -- stage2.py.
    atm = load_atm_matrix(str(VOL_CSV))
    exp_a, ten_a, vol_a = [], [], []
    for exp in atm.index:
        for ten in atm.columns:
            v = atm.loc[exp, ten]
            if not np.isnan(v):
                exp_a.append(float(exp))
                ten_a.append(float(ten))
                vol_a.append(float(v))
    exp_a = np.array(exp_a)
    ten_a = np.array(ten_a)
    N = len(grid_T) - 1
    stage2 = set()
    for i in range(2, N):
        target_maturity = grid_T[i + 1]
        mask = np.abs((exp_a + ten_a) - target_maturity) < 1e-4
        for e, t in zip(exp_a[mask], ten_a[mask]):
            stage2.add(_key(e, t, 0.0))

    return stage1, stage2


def _stage1_functions(s1_csv, h_target):
    df = _csv(s1_csv)
    df_h = df[np.abs(df["H"] - h_target) < 1e-3].sort_values("Expiry")
    if df_h.empty:
        raise ExhibitError(f"no Stage-1 rows at H={h_target} in {s1_csv}")
    alpha_f = PchipInterpolator(df_h["Expiry"].values, df_h["Alpha"].values, extrapolate=True)
    rho_f = PchipInterpolator(df_h["Expiry"].values, df_h["Rho"].values, extrapolate=True)
    nu_val = float(df_h["Nu"].iloc[0])  # one global vol-of-vol
    return alpha_f, rho_f, nu_val


def build_model(sigma_np, grid_T, F0_rates, h_target, s1_funcs, device, dtype):
    """Model on the given grid with Sigma restricted to its leading principal block."""
    alpha_f, rho_f, nu_val = s1_funcs
    N = len(F0_rates)
    if sigma_np.shape[0] < N + 1:
        raise ExhibitError(f"Sigma is {sigma_np.shape[0]}x{sigma_np.shape[0]}, too small for N={N}")
    sigma_blk = sigma_np[: N + 1, : N + 1]

    # No omega_matrix: loadings and Lambda_upper are overwritten from Sigma below.
    model = RoughSABRFMM(
        tenors=grid_T,
        F0=F0_rates,
        alpha_f=alpha_f,
        rho_f=rho_f,
        nu=nu_val,
        H=h_target,
        beta_sabr=config.BETA_SABR,
        shift=config.SHIFT_SABR,
        device=device,
    )
    Sigma_t = torch.tensor(sigma_blk, dtype=dtype, device=device)
    _inject_sigma(model, Sigma_t)

    diag_dev = float(np.max(np.abs(np.diag(sigma_blk) - 1.0)))
    min_eig = float(np.linalg.eigvalsh(sigma_blk).min())
    return model, Sigma_t, {"N": N, "block": N + 1, "min_eig": min_eig, "max_diag_dev": diag_dev}


def truncated_model_for(pair, disc_func, sigma_np, h_target, s1_funcs, device, dtype):
    """Grid ending exactly at expiry+tenor, so the co-terminal swap is the target swap."""
    expiry, tenor = pair
    end = expiry + tenor
    grid_T, F0 = bootstrap_forward_rates(
        disc_func, grid_tau=GRID_STEP_SIZE, max_maturity=end - GRID_STEP_SIZE
    )
    if abs(float(grid_T[-1]) - end) > 1e-9:
        raise ExhibitError(f"truncated grid ends at {grid_T[-1]}, expected {end} for pair {pair}")
    return build_model(sigma_np, grid_T, F0, h_target, s1_funcs, device, dtype)


def _p0_curve(model):
    p = torch.cumprod(1.0 / (1.0 + model.tau * model.F0), dim=0)
    return torch.cat([torch.ones(1, dtype=p.dtype, device=p.device), p])


def annuity_and_swap0(model, expiry):
    """A(0) and the forward par swap rate for the co-terminal swap starting at expiry."""
    tau = float(model.tau[0])
    P0 = _p0_curve(model)
    a = int(round(expiry / tau))
    A0 = tau * torch.sum(P0[a + 1 :])
    S0 = (P0[a] - P0[-1]) / A0
    return float(A0), float(S0)


@torch.no_grad()
def smm_vols_and_se(
    model,
    Sigma_t,
    expiries,
    tenors,
    strikes,
    n_paths,
    dt_val,
    n_paths_pilot=4096,
    n_scrambles=SMM_SE_SCRAMBLES,
):
    device, dtype = model.device, model.dtype
    alpha_smm, rho_smm = _compute_mapped_smm_parameters(model, Sigma_t, expiries, tenors)
    T_t = torch.tensor(expiries, device=device, dtype=dtype)
    K_t = torch.tensor(strikes, device=device, dtype=dtype)

    def price(seed):
        return oos_mc_rough_sabr_pricer(
            K_t,
            T_t,
            alpha_smm,
            rho_smm,
            model.nu,
            model.H,
            n_paths_main=n_paths,
            n_paths_pilot=n_paths_pilot,
            dt=dt_val,
            device=device,
            dtype=dtype,
            seed=seed,
        )

    # The exhibit reports the SE_BASE_SEED run. Its SE is the sd of that estimator across
    # independent scrambles, not sd/sqrt(N) within the run: the Sobol-antithetic paths of one run
    # are not independent. Not divided by sqrt(n_scrambles): one run is reported, not a mean.
    prices = price(SE_BASE_SEED)
    reps = torch.stack(
        [prices] + [price(SE_BASE_SEED + SE_SEED_STRIDE * r) for r in range(1, n_scrambles)]
    )
    std_p = reps.std(dim=0)
    iv = bachelier_iv_newton(prices, K_t, T_t, initial_guess_vol=alpha_smm)
    std_iv = torch.clamp(iv * torch.sqrt(T_t), min=1e-12)
    vega = torch.clamp(
        torch.sqrt(T_t) * INV_SQRT_2PI * torch.exp(-0.5 * (K_t / std_iv) ** 2), min=1e-12
    )
    se_vol = std_p / vega
    return iv.cpu().numpy().astype(np.float64), se_vol.cpu().numpy().astype(np.float64)


def fmm_vol(model, expiry, k_offset, seed, n_paths, dt, market_guess, freeze_drift=True):
    A0, S0 = annuity_and_swap0(model, expiry)
    n_steps = max(int(math.ceil(expiry / dt)), 1)
    time_grid = torch.linspace(
        0.0, float(expiry), n_steps + 1, device=model.device, dtype=model.dtype
    )
    with torch.no_grad():
        pv = bermudan_pricer(
            model,
            {"Strike": S0 + k_offset, "Ex_Dates": [float(expiry)]},
            n_paths=n_paths,
            time_grid=time_grid,
            seed=seed,
            freeze_drift=freeze_drift,
        )
    c = float(pv) / A0
    iv = bachelier_iv_newton(
        torch.tensor([c], dtype=model.dtype, device=model.device),
        torch.tensor([float(k_offset)], dtype=model.dtype, device=model.device),
        torch.tensor([float(expiry)], dtype=model.dtype, device=model.device),
        initial_guess_vol=torch.tensor([market_guess], dtype=model.dtype, device=model.device),
    )
    return float(iv.item()), float(pv), A0, S0


def block_bootstrap(errors_by_block, n_rep, seed, ci_level):
    """Resample whole smiles with replacement; errors within a smile are correlated."""
    rng = np.random.default_rng(seed)
    blocks = [np.asarray(b, dtype=float) for b in errors_by_block]
    nb = len(blocks)
    rmse, mae, bias = np.empty(n_rep), np.empty(n_rep), np.empty(n_rep)
    for r in range(n_rep):
        pick = rng.integers(0, nb, size=nb)
        e = np.concatenate([blocks[i] for i in pick])
        rmse[r] = math.sqrt(np.mean(e**2))
        mae[r] = np.mean(np.abs(e))
        bias[r] = np.mean(e)
    lo, hi = (1.0 - ci_level) / 2.0 * 100.0, (1.0 + ci_level) / 2.0 * 100.0
    return {
        stat: {"lo": float(np.percentile(v, lo)), "hi": float(np.percentile(v, hi))}
        for stat, v in (("rmse", rmse), ("mae", mae), ("bias", bias))
    }


def bucket_stats(errs):
    e = np.asarray(errs, dtype=float)
    return {
        "count": int(e.size),
        "bias_bps": float(np.mean(e)),
        "mae_bps": float(np.mean(np.abs(e))),
        "rmse_bps": float(np.sqrt(np.mean(e**2))),
    }


def _rule(t):
    print("\n" + "=" * 86 + f"\n{t}\n" + "=" * 86)


def run_exhibit(
    tenors=OOS_TENORS,
    expiries=OOS_EXPIRIES,
    strike_offsets=OOS_STRIKE_OFFSETS,
    atm_control_strike=ATM_CONTROL_STRIKE,
    bootstrap_n=BOOTSTRAP_N,
    bootstrap_seed=BOOTSTRAP_SEED,
    bootstrap_ci=BOOTSTRAP_CI,
    control_expiries=CONTROL_EXPIRIES,
    control_tol_se=CONTROL_TOL_SE,
    se_replicates=SE_REPLICATES,
    se_base_seed=SE_BASE_SEED,
    h_target=H_TARGET,
    n_paths_smm=N_PATHS_MC,
    dt_smm=DT_VAL,
    n_paths_fmm=int(val_config.PRICE_PATHS_MAIN),
    dt_fmm=float(val_config.PRICE_DT),
    runtime_budget_s=40 * 60,
    stop_after_projection=True,
    freeze_drift=FREEZE_DRIFT,
    out_json=OUT_JSON,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64
    t_start = time.time()

    inputs = {
        k: _stamp(p)
        for k, p in (
            ("discount_curve", DISC_CSV),
            ("vol_cube", VOL_CSV),
            ("stage1_params", S1_CSV),
            ("stage1_surface_model", S1_MODEL_CSV),
            ("sigma_reg", SIGMA_CSV),
        )
    }

    _rule("INPUTS")
    for k, v in inputs.items():
        print(f"  {k:22s} {v['sha256'][:16]}  {v['mtime']}  {v['bytes']:>9d} B")
    print(f"  H target {h_target} (val_config.VAL_H_TARGET) | device {device}")

    disc_func = load_discount_curve(str(DISC_CSV))
    grid_T, F0 = bootstrap_forward_rates(
        disc_func, grid_tau=GRID_STEP_SIZE, max_maturity=MAX_MATURITY
    )
    sigma_np = _csv(SIGMA_CSV, header=None).to_numpy().astype(np.float64)
    s1_funcs = _stage1_functions(S1_CSV, h_target)
    cube, cube_delim = load_vol_cube(VOL_CSV)
    full_model, Sigma_full, full_diag = build_model(
        sigma_np, grid_T, F0, h_target, s1_funcs, device, dtype
    )

    _rule("EXCLUSION SET — derived from the calibration code")
    stage1, stage2 = derive_exclusion_set(grid_T)
    # cross-check against the target grid the Stage-2 calibrator itself builds
    setup = CorrelationCalibrator(
        load_atm_matrix(str(VOL_CSV)), full_model
    )._build_correlation_target_grid()
    n_cal = 0 if setup is None else len(setup["global_exp"])
    if len(stage2) != n_cal:
        raise ExhibitError(
            f"Stage-2 reconstruction produced {len(stage2)} ATM targets, the calibrator {n_cal}. "
            f"The derivation does not reproduce the calibration; stopping."
        )
    overlap = stage1 & stage2
    excl = stage1 | stage2
    print(f"  cube delimiter sniffed                                              : {cube_delim!r}")
    print(
        f"  Stage 1 (load_swaption_vol_surface, |tenor-{config.STAGE1_TARGET_TENOR}|<0.05, "
        f"all strikes) : {len(stage1)}"
    )
    print(
        f"  Stage 2 (_build_correlation_target_grid, ATM co-terminals)          : "
        f"{len(stage2)}   [matches the calibrator]"
    )
    print(f"  overlap (computed by intersection)                                  : {len(overlap)}")
    print(f"  union (exclusion set)                                               : {len(excl)}")
    print(f"  cube population                                                     : {len(cube)}")
    print(
        f"  residual out-of-sample population                                   : "
        f"{len(cube) - len(excl)}"
    )

    # ATM census: why the exhibit is away-from-the-money
    atm_rows = cube[np.abs(cube["Strike"]) < 1e-12]
    atm_oos = [
        (float(r.Expiry), float(r.Tenor))
        for r in atm_rows.itertuples()
        if _key(r.Expiry, r.Tenor, 0.0) not in stage2
    ]
    first_node = float(grid_T[3])  # loop is range(2, N) -> first target grid_T[3]
    frac = [(e, t) for e, t in atm_oos if abs(e - round(e)) > 1e-9]
    beyond = [
        (e, t) for e, t in atm_oos if abs(e - round(e)) <= 1e-9 and e + t > float(grid_T[-1]) + 1e-9
    ]
    below = [(e, t) for e, t in atm_oos if abs(e - round(e)) <= 1e-9 and e + t < first_node - 1e-9]
    census = {
        "atm_cells_in_cube": int(len(atm_rows)),
        "atm_in_sample": int(len(stage2)),
        "atm_out_of_sample": int(len(atm_oos)),
        "oos_atm_fractional_expiry": len(frac),
        "oos_atm_sum_beyond_grid_end": len(beyond),
        "oos_atm_sum_below_first_node": len(below),
        "oos_atm_sum_below_first_node_cells": sorted(below),
        "categories_exhaust_oos_atm": len(frac) + len(beyond) + len(below) == len(atm_oos),
        "grid_step_size": float(GRID_STEP_SIZE),
        "grid_end": float(grid_T[-1]),
        "first_target_node": first_node,
    }
    _rule("ATM CENSUS")
    for k, v in census.items():
        print(f"  {k:52s} {v}")

    _rule("INSTRUMENT GRID")
    cube_idx = {_key(r.Expiry, r.Tenor, r.Strike): float(r.Vol) for r in cube.itertuples()}
    quotes, atm_control, missing, inside = [], [], [], []
    for ten in tenors:
        for exp in expiries:
            for k in strike_offsets:
                kk = _key(exp, ten, k)
                if kk not in cube_idx:
                    missing.append(kk)
                    continue
                if kk in excl:
                    inside.append(kk)
                quotes.append(
                    {
                        "expiry": float(exp),
                        "tenor": float(ten),
                        "strike_offset": float(k),
                        "market_vol": cube_idx[kk],
                    }
                )
            ak = _key(exp, ten, atm_control_strike)
            if ak not in cube_idx:
                missing.append(ak)
            else:
                atm_control.append(
                    {
                        "expiry": float(exp),
                        "tenor": float(ten),
                        "strike_offset": float(atm_control_strike),
                        "market_vol": cube_idx[ak],
                        "in_sample": ak in excl,
                    }
                )
    if missing:
        raise ExhibitError(
            f"{len(missing)} selected cells absent from the cube: {missing}. "
            f"Not dropping them and not substituting neighbours; stopping."
        )
    if inside:
        raise ExhibitError(
            f"{len(inside)} out-of-sample cells fall INSIDE the exclusion set: {inside}"
        )
    pairs = sorted({(q["expiry"], q["tenor"]) for q in quotes})
    print(f"  {len(quotes)} out-of-sample quotes, all present, all outside the exclusion set")
    print(
        f"  {len(atm_control)} ATM control quotes "
        f"({sum(a['in_sample'] for a in atm_control)} confirmed in-sample) — "
        f"reported separately, never in an out-of-sample statistic"
    )
    print(f"  {len(pairs)} (expiry, tenor) blocks: {pairs}")

    _rule("STEP 1 — HARNESS PROOF: truncation is the identity at N' = N")
    end_full = float(grid_T[-1])
    proof_exp = 10.0
    proof_pair = (proof_exp, end_full - proof_exp)
    m_trunc, _, d_proof = truncated_model_for(
        proof_pair, disc_func, sigma_np, h_target, s1_funcs, device, dtype
    )
    v_trunc, pv_trunc, A_t, S_t = fmm_vol(
        m_trunc, proof_exp, 0.0, se_base_seed, n_paths_fmm, dt_fmm, 0.005, freeze_drift
    )
    v_full, pv_full, A_f, S_f = fmm_vol(
        full_model, proof_exp, 0.0, se_base_seed, n_paths_fmm, dt_fmm, 0.005, freeze_drift
    )
    print(f"  co-terminal {proof_pair}, ATM, seed {se_base_seed}, {n_paths_fmm} paths")
    print(f"    truncation path  PV = {pv_trunc!r}")
    print(f"    untruncated path PV = {pv_full!r}")
    print(f"    difference          = {abs(pv_trunc - pv_full):.6e}")
    print(f"    N' {m_trunc.N} vs N {full_model.N} | A(0) {A_t!r} vs {A_f!r}")
    if abs(pv_trunc - pv_full) > 0.0:
        raise ExhibitError(
            f"truncation harness is not the identity at N'=N: PV differs by "
            f"{abs(pv_trunc - pv_full):.3e}. Stopping before pricing anything else."
        )
    print("  PASS — bit-identical")

    _rule("STEP 2 — TRUNCATED SIGMA BLOCKS: PSD and unit diagonal")
    t_build0 = time.time()
    models, blocks_diag = {}, {}
    print(
        f"  {'pair':>12} | {'end':>4} | {'N':>3} | {'block':>6} | {'min eig':>13} | "
        f"{'max|diag-1|':>12}"
    )
    print("  " + "-" * 68)
    for pair in pairs:
        m, S, d = truncated_model_for(pair, disc_func, sigma_np, h_target, s1_funcs, device, dtype)
        models[pair] = (m, S)
        blocks_diag[f"{pair[0]:g}x{pair[1]:g}"] = d
        print(
            f"  {str(pair):>12} | {pair[0] + pair[1]:>4.0f} | {d['N']:>3} | {d['block']:>6} | "
            f"{d['min_eig']:>+13.3e} | {d['max_diag_dev']:>12.3e}"
        )
        if d["min_eig"] < -1e-10 or d["max_diag_dev"] > 1e-9:
            raise ExhibitError(f"truncated Sigma block for {pair} is not PSD/unit-diagonal")
    t_build = time.time() - t_build0
    print(f"  PASS — all {len(pairs)} blocks PSD with unit diagonal | build {t_build:.2f}s")

    _rule("STEP 3 — CONTROL GATE: in-sample SMM against the Stage-1 surface")
    s1_surf = _csv(S1_MODEL_CSV)
    s1_h = s1_surf[np.isclose(s1_surf["H"], h_target)]
    if s1_h.empty:
        raise ExhibitError(f"no H={h_target} slice in {S1_MODEL_CSV}")
    c_exp, c_ten, c_str, c_ref = [], [], [], []
    for e in control_expiries:
        row = s1_h[np.isclose(s1_h["Expiry"], e)]
        if row.empty:
            raise ExhibitError(f"control expiry {e} absent from the Stage-1 surface")
        for k in (0.0, -0.005, 0.005):
            col = next(
                (c for c in s1_h.columns if c not in ("H", "Expiry") and abs(float(c) - k) < 1e-9),
                None,
            )
            if col is None:
                raise ExhibitError(f"control strike {k} absent from the Stage-1 surface")
            c_exp.append(float(e))
            c_ten.append(float(config.STAGE1_TARGET_TENOR))
            c_str.append(float(k))
            c_ref.append(float(row[col].iloc[0]))
    c_iv, c_se = smm_vols_and_se(
        full_model,
        Sigma_full,
        np.array(c_exp),
        np.array(c_ten),
        np.array(c_str),
        n_paths_smm,
        dt_smm,
    )
    print(
        f"  {'expiry':>7} {'strike':>8} | {'stage1':>9} {'harness':>9} {'diff':>8} "
        f"{'se':>7} {'z':>8}"
    )
    print("  " + "-" * 60)
    gate_fail, gate_rows = [], []
    for i in range(len(c_exp)):
        d_bps = (c_iv[i] - c_ref[i]) * 1e4
        se = c_se[i] * 1e4
        z = d_bps / se if se > 0 else float("inf")
        gate_rows.append(
            {
                "expiry": c_exp[i],
                "strike_offset": c_str[i],
                "stage1_bps": c_ref[i] * 1e4,
                "harness_bps": c_iv[i] * 1e4,
                "diff_bps": d_bps,
                "se_bps": se,
                "z": z,
            }
        )
        print(
            f"  {c_exp[i]:>7.2f} {c_str[i]:>8.4f} | {c_ref[i] * 1e4:>9.2f} {c_iv[i] * 1e4:>9.2f} "
            f"{d_bps:>+8.2f} {se:>7.2f} {z:>+8.2f}"
            f"{'' if abs(z) <= control_tol_se else '  <-- FAIL'}"
        )
        if abs(z) > control_tol_se:
            gate_fail.append((c_exp[i], c_str[i], z))
    if gate_fail:
        raise ExhibitError(
            f"control gate FAILED at {len(gate_fail)} of {len(c_exp)} cells "
            f"(|z| > {control_tol_se}): {gate_fail}. The harness does not reproduce the "
            f"Stage-1 residual, so out-of-sample numbers would be meaningless. Stopping; "
            f"tolerance not widened."
        )
    print(f"  PASS — all {len(c_exp)} cells within {control_tol_se} MC sigma")

    _rule("STEP 4 — RUNTIME PROJECTION")
    seeds = [se_base_seed + SE_SEED_STRIDE * r for r in range(se_replicates)]
    per_pair = len(strike_offsets) + 1  # +1 ATM control quote per pair
    probe_t = {}
    for pair in pairs:
        m, _ = models[pair]
        q0 = next(q for q in quotes if (q["expiry"], q["tenor"]) == pair)
        t0 = time.time()
        fmm_vol(
            m,
            pair[0],
            q0["strike_offset"],
            seeds[0],
            n_paths_fmm,
            dt_fmm,
            q0["market_vol"],
            freeze_drift,
        )
        probe_t[pair] = time.time() - t0
    per_pass = sum(probe_t.values())
    total_calls = (len(quotes) + len(atm_control)) * se_replicates
    projected = per_pass * per_pair * se_replicates
    print(f"  probe: one FMM call per pair, {len(pairs)} calls, {per_pass:.1f}s")
    for p in pairs:
        print(f"    {str(p):>12} {probe_t[p]:>7.2f}s")
    print(f"\n  model build time (cached, one set)   : {t_build:.2f}s")
    print(f"  projected pricing time              : {projected / 60:.1f} min")
    print(f"    = {per_pass:.1f}s/pass x {per_pair} quotes/pair x {se_replicates} seeds")
    print(
        f"    {total_calls} FMM calls ({len(quotes)} OOS + {len(atm_control)} ATM control, "
        f"x{se_replicates} seeds)"
    )
    print(f"  budget                              : {runtime_budget_s / 60:.0f} min")
    print(
        f"  NOTE bermudan_pricer re-simulates per strike, so each smile pays {per_pair} "
        f"full simulations rather than one."
    )
    if projected > runtime_budget_s:
        raise ExhibitError(
            f"projected pricing time {projected / 60:.1f} min exceeds the "
            f"{runtime_budget_s / 60:.0f} min budget (build {t_build:.1f}s). Stopping before the "
            f"run; subset the exhibit or raise the budget."
        )

    if stop_after_projection:
        _rule("HALTED AFTER STEP 4 AS INSTRUCTED — no exhibit pricing performed")
        print(f"  steps 1-3 PASS. Projection above. Total elapsed {time.time() - t_start:.1f}s")
        return {
            "halted_after": "projection",
            "census": census,
            "blocks_diag": blocks_diag,
            "control_gate": gate_rows,
            "projection_s": projected,
            "build_s": t_build,
            "per_pair_probe_s": {f"{p[0]:g}x{p[1]:g}": v for p, v in probe_t.items()},
        }

    _rule("PRICING")
    t_price0 = time.time()
    all_q = quotes + atm_control
    q_arr = {
        k: np.array([q[k] for q in all_q])
        for k in ("expiry", "tenor", "strike_offset", "market_vol")
    }
    smm_iv, smm_se = smm_vols_and_se(
        full_model,
        Sigma_full,
        q_arr["expiry"],
        q_arr["tenor"],
        q_arr["strike_offset"],
        n_paths_smm,
        dt_smm,
    )
    fmm_reps = np.empty((se_replicates, len(all_q)))
    for r, sd in enumerate(seeds):
        for i, q in enumerate(all_q):
            m, _ = models[(q["expiry"], q["tenor"])]
            fmm_reps[r, i] = fmm_vol(
                m,
                q["expiry"],
                q["strike_offset"],
                sd,
                n_paths_fmm,
                dt_fmm,
                q["market_vol"],
                freeze_drift,
            )[0]
        print(f"  seed {sd} done ({time.time() - t_price0:.0f}s elapsed)")
    fmm_iv = fmm_reps.mean(axis=0)
    fmm_se = fmm_reps.std(axis=0, ddof=1) / math.sqrt(se_replicates)
    t_price = time.time() - t_price0

    for i, q in enumerate(all_q):
        q["smm_vol"] = float(smm_iv[i])
        q["fmm_vol"] = float(fmm_iv[i])
        q["smm_error_bps"] = float((smm_iv[i] - q["market_vol"]) * 1e4)
        q["fmm_error_bps"] = float((fmm_iv[i] - q["market_vol"]) * 1e4)
        q["smm_se_bps"] = float(smm_se[i] * 1e4)
        q["fmm_se_bps"] = float(fmm_se[i] * 1e4)
        q["smm_minus_fmm_bps"] = float((smm_iv[i] - fmm_iv[i]) * 1e4)

    return _finish(
        quotes, atm_control, pairs, inputs, census, blocks_diag, gate_rows, cube_delim, locals()
    )


def _finish(quotes, atm_control, pairs, inputs, census, blocks_diag, gate_rows, cube_delim, L):
    se_replicates, seeds = L["se_replicates"], L["seeds"]

    _rule("PER-QUOTE — out of sample")
    print(
        f"  {'exp':>5} {'ten':>5} {'k(bp)':>7} | {'mkt':>7} {'SMM':>7} {'FMM':>7} | "
        f"{'e_SMM':>7} {'e_FMM':>7} | {'se_S':>5} {'se_F':>5} | {'S-F':>7}"
    )
    print("  " + "-" * 92)
    for q in quotes:
        print(
            f"  {q['expiry']:>5.0f} {q['tenor']:>5.0f} {q['strike_offset'] * 1e4:>7.0f} | "
            f"{q['market_vol'] * 1e4:>7.2f} {q['smm_vol'] * 1e4:>7.2f} {q['fmm_vol'] * 1e4:>7.2f} | "
            f"{q['smm_error_bps']:>+7.2f} {q['fmm_error_bps']:>+7.2f} | "
            f"{q['smm_se_bps']:>5.2f} {q['fmm_se_bps']:>5.2f} | {q['smm_minus_fmm_bps']:>+7.2f}"
        )

    _rule("SECOND IN-SAMPLE CONTROL — ATM cells (Stage-2 targets), NOT out-of-sample")
    print(f"  {'exp':>5} {'ten':>5} | {'mkt':>7} {'SMM':>7} {'FMM':>7} | {'e_SMM':>7} {'e_FMM':>7}")
    print("  " + "-" * 60)
    for q in atm_control:
        print(
            f"  {q['expiry']:>5.0f} {q['tenor']:>5.0f} | {q['market_vol'] * 1e4:>7.2f} "
            f"{q['smm_vol'] * 1e4:>7.2f} {q['fmm_vol'] * 1e4:>7.2f} | "
            f"{q['smm_error_bps']:>+7.2f} {q['fmm_error_bps']:>+7.2f}"
        )
    atm_stats = {
        c: bucket_stats([q[f"{c}_error_bps"] for q in atm_control]) for c in ("smm", "fmm")
    }
    for c in ("smm", "fmm"):
        s = atm_stats[c]
        print(
            f"  {c.upper()} in-sample ATM: bias {s['bias_bps']:+.2f}  MAE {s['mae_bps']:.2f}  "
            f"RMSE {s['rmse_bps']:.2f}  n {s['count']}"
        )

    buckets = {"by_tenor": {}, "by_moneyness": {}}
    for col in ("smm", "fmm"):
        for ten in sorted({q["tenor"] for q in quotes}):
            e = [q[f"{col}_error_bps"] for q in quotes if q["tenor"] == ten]
            buckets["by_tenor"].setdefault(f"{ten:g}Y", {})[col] = bucket_stats(e)
        for k in sorted({q["strike_offset"] for q in quotes}):
            e = [q[f"{col}_error_bps"] for q in quotes if q["strike_offset"] == k]
            buckets["by_moneyness"].setdefault(f"{k * 1e4:+.0f}bp", {})[col] = bucket_stats(e)

    for name, grp in buckets.items():
        _rule(f"BUCKETED — {name.replace('by_', 'by ')} (out of sample)")
        print(
            f"  {'bucket':>10} | {'n':>3} | {'SMM bias':>9} {'MAE':>7} {'RMSE':>7} | "
            f"{'FMM bias':>9} {'MAE':>7} {'RMSE':>7}"
        )
        print("  " + "-" * 72)
        for b, cols in grp.items():
            s, f = cols["smm"], cols["fmm"]
            print(
                f"  {b:>10} | {s['count']:>3} | {s['bias_bps']:>+9.2f} {s['mae_bps']:>7.2f} "
                f"{s['rmse_bps']:>7.2f} | {f['bias_bps']:>+9.2f} {f['mae_bps']:>7.2f} "
                f"{f['rmse_bps']:>7.2f}"
            )

    overall = {c: bucket_stats([q[f"{c}_error_bps"] for q in quotes]) for c in ("smm", "fmm")}
    blocks = {
        c: [[q[f"{c}_error_bps"] for q in quotes if (q["expiry"], q["tenor"]) == p] for p in pairs]
        for c in ("smm", "fmm")
    }
    boot = {
        c: block_bootstrap(blocks[c], L["bootstrap_n"], L["bootstrap_seed"], L["bootstrap_ci"])
        for c in ("smm", "fmm")
    }

    _rule("BLOCK BOOTSTRAP")
    print(
        f"  {len(pairs)} smile blocks, {L['bootstrap_n']} replicates, seed "
        f"{L['bootstrap_seed']}, {L['bootstrap_ci']:.0%} interval (2.5th / 97.5th percentile)"
    )
    for c in ("smm", "fmm"):
        print(f"\n  {c.upper()}")
        for stat in ("rmse", "mae", "bias"):
            print(
                f"    {stat:>5} {overall[c][stat + '_bps']:>+8.2f}  "
                f"[{boot[c][stat]['lo']:>+7.2f}, {boot[c][stat]['hi']:>+7.2f}]"
            )

    agg_se = {
        c: float(np.sqrt(np.mean(np.array([q[f"{c}_se_bps"] for q in quotes]) ** 2)))
        for c in ("smm", "fmm")
    }
    _rule("AGGREGATE MC STANDARD ERROR — context only, NOT an input to the interval")
    print(
        f"  SMM {agg_se['smm']:.3f} bps   (sd of one run across {SMM_SE_SCRAMBLES} scrambles, "
        f"vega-converted)"
    )
    print(f"  FMM {agg_se['fmm']:.3f} bps   ({se_replicates}-seed replicate estimate)")

    d = np.array([q["smm_minus_fmm_bps"] for q in quotes])
    _rule("SMM MINUS FMM")
    print(
        f"  mean {d.mean():+.2f} bps | sd {d.std(ddof=1):.2f} | min {d.min():+.2f} | "
        f"max {d.max():+.2f} | n {d.size}"
    )

    ledger = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inputs": inputs,
        "cube_delimiter": cube_delim,
        "exhibit_definition": {
            "tenors": list(L["tenors"]),
            "expiries": list(L["expiries"]),
            "strike_offsets": list(L["strike_offsets"]),
            "atm_control_strike": L["atm_control_strike"],
            "bootstrap_replicates": L["bootstrap_n"],
            "bootstrap_seed": L["bootstrap_seed"],
            "bootstrap_ci_level": L["bootstrap_ci"],
            "control_expiries": list(L["control_expiries"]),
            "control_tol_se": L["control_tol_se"],
            "se_replicates": se_replicates,
            "se_base_seed": L["se_base_seed"],
            "se_seeds": seeds,
            "se_seed_stride": SE_SEED_STRIDE,
            "smm_se_scrambles": SMM_SE_SCRAMBLES,
        },
        "engine_config": {
            "h_target": L["h_target"],
            "h_target_source": "val_config.VAL_H_TARGET",
            "smm_n_paths": L["n_paths_smm"],
            "smm_dt": L["dt_smm"],
            "fmm_n_paths": L["n_paths_fmm"],
            "fmm_dt": L["dt_fmm"],
            "device": L["device"],
            "fmm_freeze_drift": L["freeze_drift"],
            "fmm_truncation": "per (expiry,tenor): grid ends at expiry+tenor, Sigma leading "
            "principal block; changes the terminal measure and hence the "
            "frozen drift (mu_j(0) sums to N) — disclosed modelling choice",
        },
        "exclusion_set": {
            "stage1_count": len(L["stage1"]),
            "stage2_count": len(L["stage2"]),
            "overlap_count": len(L["overlap"]),
            "union_count": len(L["excl"]),
            "cube_population": len(L["cube"]),
            "residual_population": len(L["cube"]) - len(L["excl"]),
            "stage1_source": "jolto calibrate-sabr -> load_swaption_vol_surface(vol_csv, "
            "config.STAGE1_TARGET_TENOR)",
            "stage2_source": "CorrelationCalibrator._build_correlation_target_grid "
            "against bootstrapped grid_T",
        },
        "atm_census": census,
        "sigma_blocks": blocks_diag,
        "control_gate_stage1": gate_rows,
        "control_atm_in_sample": {"quotes": atm_control, "stats": atm_stats},
        "quotes": quotes,
        "buckets": buckets,
        "overall": overall,
        "bootstrap_ci": boot,
        "aggregate_mc_se_bps": agg_se,
        "smm_minus_fmm_bps": {
            "mean": float(d.mean()),
            "sd": float(d.std(ddof=1)),
            "min": float(d.min()),
            "max": float(d.max()),
        },
        "timing_s": {
            "model_build": L["t_build"],
            "pricing": L["t_price"],
            "total": time.time() - L["t_start"],
        },
    }
    out_json = Path(L["out_json"])
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n")
    _rule("OUTPUT")
    print(f"  {out_json}")
    print(
        f"  build {L['t_build']:.1f}s | pricing {L['t_price']:.1f}s | "
        f"total {ledger['timing_s']['total']:.1f}s   (wall-clock)"
    )
    return ledger


def main():
    import argparse

    p = argparse.ArgumentParser(description="Out-of-sample repricing exhibit.")
    p.add_argument(
        "--run",
        action="store_true",
        help="Price the exhibit. Default halts after the runtime projection.",
    )
    p.add_argument(
        "--no-freeze-drift",
        dest="freeze_drift",
        action="store_false",
        default=FREEZE_DRIFT,
        help="Run the FMM leg on the state-dependent Euler drift.",
    )
    p.add_argument(
        "--out",
        default=str(OUT_JSON),
        help="Destination JSON (default: the canonical exhibit artefact).",
    )
    args = p.parse_args()
    try:
        run_exhibit(
            stop_after_projection=not args.run, freeze_drift=args.freeze_drift, out_json=args.out
        )
    except ExhibitError as exc:
        print("\n" + "!" * 86)
        print("STOPPED")
        print("!" * 86)
        print(exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
