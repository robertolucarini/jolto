"""Stage 1 and Stage 2 calibration pipelines.

Each reads the inputs named in jolto.config and writes its outputs to config.RESULTS_DIR.
jolto calibrate-sabr and jolto calibrate-correlation wrap them for the command line; the regularisation studies
(jolto.calibration.regul) call them under temporary config overrides.
"""

import hashlib
import json
import os
import re
import time

import numpy as np
import pandas as pd
import torch
from scipy.interpolate import PchipInterpolator

from jolto import config
from jolto.calibration.stage1 import RoughSABRCalibrator
from jolto.calibration.stage2 import CorrelationCalibrator
from jolto.calibration.lambda_rule import select_lambda
from jolto.core.market_data import (
    bootstrap_forward_rates,
    load_atm_matrix,
    load_discount_curve,
    load_swaption_vol_surface,
)
from jolto.core.model import RoughSABRFMM
from jolto.core.utils import print_header


def run_stage1():
    """Stage 1 on every H of config.H_GRID: read VOL_SURFACE_CSV, write the parameter table and the
    market and model surfaces to config.RESULTS_DIR."""
    vol_csv = os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV)
    print_header("STAGE 1 CALIBRATION — 1D ROUGH SABR MARGINALS")

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    out_params = os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)
    out_market = os.path.join(config.RESULTS_DIR, config.STAGE1_SURFACE_MARKET_CSV)
    out_model = os.path.join(
        config.RESULTS_DIR,
        config.STAGE1_SURFACE_MODEL_CSV_TPL.format(method=config.CALI_MODE),
    )

    print(f"  Vol surface   : {vol_csv}")
    print(f"  Target tenor  : {config.STAGE1_TARGET_TENOR}Y")
    print(f"  Mode          : {config.CALI_MODE}")
    print(f"  H grid        : {np.asarray(config.H_GRID)}")
    print(f"  Params out    : {out_params}")
    print(f"  Market surf   : {out_market}")
    print(f"  Model surf    : {out_model}")

    vol_surface = load_swaption_vol_surface(vol_csv, config.STAGE1_TARGET_TENOR)
    print(f"  Loaded surface: {vol_surface.shape[0]} expiries × {vol_surface.shape[1]} strikes")

    t0 = time.time()
    calibrator = RoughSABRCalibrator(vol_surface)
    result = calibrator.calibrate(method=config.CALI_MODE, H_grid=np.asarray(config.H_GRID))
    elapsed = time.time() - t0

    # stage1_parameters.csv: every H, every expiry
    param_rows = []
    for rec in result["profile_likelihood"]:
        for j, exp in enumerate(calibrator.expiries):
            param_rows.append(
                {
                    "Method": config.CALI_MODE,
                    "H": rec["H"],
                    "Expiry": float(exp),
                    "Alpha": float(rec["alphas"][j]),
                    "Rho": float(rec["rhos"][j]),
                    "Nu": float(rec["nu"]),
                    "RMSE_bps": float(rec["rmse_bps"]),
                }
            )
    df_params = pd.DataFrame(
        param_rows, columns=["Method", "H", "Expiry", "Alpha", "Rho", "Nu", "RMSE_bps"]
    )
    df_params.to_csv(out_params, index=False)

    vol_surface.to_csv(out_market, index=True)

    # model surface: rows (H, Expiry), NaN where the model was not evaluated
    strikes = vol_surface.columns.tolist()
    surf_rows = []
    for rec in result["profile_likelihood"]:
        tk_to_v = {
            (float(T), float(K)): float(v)
            for T, K, v in zip(calibrator.T_flat, calibrator.K_flat, rec["v_surface_flat"])
        }
        for exp in vol_surface.index:
            row = {"H": rec["H"], "Expiry": float(exp)}
            for K in strikes:
                row[K] = tk_to_v.get((float(exp), float(K)), np.nan)
            surf_rows.append(row)
    df_model = pd.DataFrame(surf_rows, columns=["H", "Expiry"] + strikes)
    df_model.to_csv(out_model, index=False)

    print_header("STAGE 1 COMPLETE")
    print(f"  Best H        : {result['H']:.4f}")
    print(f"  Best RMSE     : {result['rmse_bps']:.4f} bps")
    print(f"  Global ν      : {result['best_params']['nu']:.4f}")
    print(f"  Mean ρ        : {np.mean(result['best_params']['rhos']):.4f}")
    print(f"  Elapsed       : {elapsed:.2f}s")
    print(f"  Params        : {out_params}  ({len(df_params)} rows)")
    print(f"  Market surf   : {out_market}")
    print(f"  Model surf    : {out_model}  ({len(df_model)} rows)")


def build_model_from_stage1(s1_df, grid_T, F0_rates, h_target, device, beta_sabr=None, shift=None):
    """RoughSABRFMM from one H slice of the Stage-1 parameter table, α(T) and ρ(T) by PCHIP across
    the calibrated expiries; beta_sabr and shift default to config.BETA_SABR and config.SHIFT_SABR."""
    df_h = s1_df[np.abs(s1_df["H"] - h_target) < 1e-3].sort_values("Expiry")
    if df_h.empty:
        raise ValueError(f"No Stage 1 rows found for H={h_target}")

    alpha_f = PchipInterpolator(df_h["Expiry"].values, df_h["Alpha"].values, extrapolate=True)
    rho_f = PchipInterpolator(df_h["Expiry"].values, df_h["Rho"].values, extrapolate=True)
    if df_h["Nu"].nunique() != 1:
        raise ValueError(f"Stage-1 Nu is not a single value at H={h_target}: {df_h['Nu'].unique()}")
    nu_val = float(df_h["Nu"].iloc[0])  # one global vol-of-vol

    return RoughSABRFMM(
        tenors=grid_T,
        F0=F0_rates,
        alpha_f=alpha_f,
        rho_f=rho_f,
        nu=nu_val,
        H=h_target,
        beta_sabr=config.BETA_SABR if beta_sabr is None else beta_sabr,
        shift=config.SHIFT_SABR if shift is None else shift,
        device=device,
    )


def check_hist_end():
    """A hist_* prior must end on the quote date; the cube's date comes from its filename."""
    if not config.STAGE2_DECAY_FORM.startswith("hist_"):
        return
    m = re.search(r"_(\d{8})\.csv$", config.VOL_SURFACE_CSV)
    if m is None:
        raise ValueError(
            f"VOL_SURFACE_CSV '{config.VOL_SURFACE_CSV}' carries no date, so STAGE2_HIST_END cannot "
            f"be checked. Name the cube <name>_YYYYMMDD.csv, or set STAGE2_DECAY_FORM to 'exp' or 'linear'."
        )
    cube_date = pd.Timestamp(m.group(1))
    if cube_date != pd.Timestamp(config.STAGE2_HIST_END):
        raise ValueError(
            f"STAGE2_HIST_END {config.STAGE2_HIST_END} differs from the vol cube date "
            f"{cube_date.date()} ({config.VOL_SURFACE_CSV}): a later end is look-ahead, an earlier one stale."
        )


def _file_stamp(path):
    """Self-stamp: sha256 is the identity, mtime is informational only."""
    with open(path, "rb") as f:
        raw = f.read()
    return {
        "path": os.path.abspath(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
        "bytes": len(raw),
    }


def _apply_sweep_override():
    """Entry-point mode override: sets only the sweep switch, no numeric value."""
    config.AUTO_LAMBDA = True
    grid = np.logspace(
        np.log10(config.LAMBDA_GRID_LO),
        np.log10(config.LAMBDA_GRID_HI),
        config.LAMBDA_GRID_N,
    )
    floor = config.STAGE2_LAMBDA_FLOOR
    print("!" * 80)
    print("  --sweep: λ sweep for this process only (config.AUTO_LAMBDA set True in memory)")
    print("  config.py is NOT modified; the override lives only in this process.")
    print("  Mode      : λ sweep. The production Σ is not written in this mode.")
    print(f"  λ floor   : {floor:.3e}   (unregularised row, written to the _unreg CSV)")
    print(f"  λ grid    : {len(grid)} points, {grid[0]:.4e} … {grid[-1]:.4e}")
    print(f"  Total rows: {len(grid) + 1}  (floor + grid)")
    print("!" * 80)


def run_stage2(sweep=False):
    """Stage 2 at config.STAGE2_H: build the model from the Stage-1 table in config.RESULTS_DIR,
    select λ by config.STAGE2_LAMBDA_RULE and fit Σ, writing Σ, the fit summary and the λ table there.
    sweep=True runs the λ sweep instead and writes the sweep table and the unregularised Σ."""
    check_hist_end()

    print_header("STAGE 2 CALIBRATION — SPATIAL CORRELATION")

    if sweep:
        _apply_sweep_override()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    disc_csv = os.path.join(config.DATA_DIR, config.DISCOUNT_CSV)
    vol_csv = os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV)
    s1_csv = os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    if not os.path.exists(s1_csv):
        raise FileNotFoundError(f"Stage 1 output not found at {s1_csv}; run Stage 1 first.")

    print(f"  Device         : {device.upper()}")
    print(f"  Discount curve : {disc_csv}")
    print(f"  Vol surface    : {vol_csv}")
    print(f"  Stage 1 params : {s1_csv}")
    print(f"  Mode           : {config.CORR_CALI_MODE}")

    s1_df = pd.read_csv(s1_csv)
    h_target = config.STAGE2_H  # Stage 1 writes every H_GRID slice; Stage 2 runs at config.STAGE2_H
    print(f"  Calibrated H   : {h_target:.4f}")

    disc_func = load_discount_curve(disc_csv)
    grid_T, F0_rates = bootstrap_forward_rates(
        disc_func,
        grid_tau=config.GRID_STEP_SIZE,
        max_maturity=config.MAX_MATURITY,
    )
    print(
        f"  Forward curve  : {len(F0_rates)} rates, τ={config.GRID_STEP_SIZE}Y, max={config.MAX_MATURITY}Y"
    )

    model = build_model_from_stage1(s1_df, grid_T, F0_rates, h_target, device)
    print(f"  Model          : RoughSABRFMM (N={model.N}, H={model.H.item():.4f})")

    atm_matrix = load_atm_matrix(vol_csv)
    print(f"  ATM matrix     : {atm_matrix.shape[0]} expiries × {atm_matrix.shape[1]} tenors")

    # the rule sets λ for this process only; --sweep is diagnostic and keeps the configured λ
    lambda_csv = None
    rule = config.STAGE2_LAMBDA_RULE
    if rule != "fixed" and not sweep:
        t_sel = time.time()
        lam, table = select_lambda(atm_matrix, model, rule)
        tpl = (
            config.STAGE2_LAMBDA_CV_CSV_TPL
            if rule == "cv_1se"
            else config.STAGE2_LAMBDA_SWEEP_CSV_TPL
        )
        lambda_csv = os.path.join(config.RESULTS_DIR, tpl.format(h=h_target))
        table.to_csv(lambda_csv, index=False)
        config.LAMBDA_CURVATURE = lam
        print(
            f"  λ rule         : {rule} -> λ = {lam:.6g} ({time.time() - t_sel:.1f}s, {lambda_csv})"
        )

    t0 = time.time()
    calibrator = CorrelationCalibrator(atm_matrix, model)
    result = calibrator.calibrate(
        method=config.CORR_CALI_MODE,
        log_file=os.path.join(config.RESULTS_DIR, config.STAGE2_LOG_TXT),
        stage1_csv=s1_csv,
    )
    elapsed = time.time() - t0

    if result.get("sweep_only"):
        sweep_name = config.STAGE2_LAMBDA_SWEEP_CSV_TPL.format(h=h_target)
        sweep_csv = os.path.join(config.RESULTS_DIR, sweep_name)
        pd.DataFrame(result["lambda_sweep_data"]).to_csv(sweep_csv, index=False)

        # only the sweep produces the λ→0 row, so only the sweep writes the unregularised Σ;
        # a production run must not refresh it
        unreg_csv = None
        Sigma_floor = result.get("Sigma_floor")
        if Sigma_floor is not None:
            unreg_csv = os.path.join(
                config.RESULTS_DIR,
                config.STAGE2_CORR_UNREG_CSV_TPL.format(h=h_target),
            )
            pd.DataFrame(Sigma_floor).to_csv(unreg_csv, header=False, index=False)

        print_header("STAGE 2 λ-SWEEP COMPLETE (production Σ not written)")
        print(f"  λ-sweep grid   : {sweep_csv}")
        if unreg_csv is not None:
            print(f"  Unregularised Σ: {unreg_csv}")
            print(f"  λ (floor row)  : {result['lambda_floor']:.6e}")
            print(f"  Σ_unreg shape  : {np.asarray(Sigma_floor).shape}")
        else:
            print("  Unregularised Σ: NOT WRITTEN (no floor row in this result)")
        print(f"  Elapsed        : {elapsed:.2f}s")
        print('  Next: to fit at a λ from the table, set config.STAGE2_LAMBDA_RULE = "fixed" and')
        print(
            "        config.LAMBDA_CURVATURE to that λ, then re-run jolto calibrate-correlation without --sweep."
        )
        return

    Sigma = result["Sigma_matrix"]

    # no header/index, as the tests/oos.py loader expects. PURE_MC is a parity run: the suffix
    # keeps it from overwriting the canonical AMMO Σ.
    _suffix = "_puremc" if config.CORR_CALI_MODE == "PURE_MC" else ""
    root, ext = os.path.splitext(config.STAGE2_CORR_CSV_TPL.format(h=h_target))
    out_name = root + _suffix + ext
    out_csv = os.path.join(config.RESULTS_DIR, out_name)
    pd.DataFrame(Sigma).to_csv(out_csv, header=False, index=False)

    # the suffix gate keeps a PURE_MC parity run from overwriting the canonical fit summary
    fit_json = None
    _triple = {
        k: result.get(k) for k in ("final_rmse_bps", "final_max_err_bps", "final_mean_bias_bps")
    }
    if _suffix:
        print(f"  Fit summary    : NOT WRITTEN ({config.CORR_CALI_MODE} is a comparison run)")
    elif any(v is None for v in _triple.values()):
        # A null in a self-stamping artefact is worse than an absent artefact.
        print("  Fit summary    : NOT WRITTEN (calibrate() returned no RMSE, max error or bias)")
    else:
        fit_json = os.path.join(config.RESULTS_DIR, config.STAGE2_FIT_SUMMARY_JSON)
        with open(fit_json, "w") as f:
            json.dump(
                {
                    "schema_version": 1,
                    "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    **_triple,
                    "lambda": float(config.LAMBDA_CURVATURE),
                    "method": config.CORR_CALI_MODE,
                    "H": float(h_target),
                    "n_paths": int(config.CALI_PATHS),
                    "dt": float(config.CALI_DT),
                    "hybrid_kappa": int(config.HYBRID_KAPPA),
                    # only off the default rule, so default summaries stay byte-identical
                    **(
                        {
                            "lambda_rule": rule,
                            "lambda_selected": float(config.LAMBDA_CURVATURE),
                            "lambda_table": lambda_csv,
                        }
                        if lambda_csv
                        else {}
                    ),
                    # the fitted hist_* target, at full precision (the log line rounds it)
                    **(
                        {"decay_target": calibrator.hist_target_info}
                        if calibrator.hist_target_info is not None
                        else {}
                    ),
                    "inputs": {
                        "stage1_params": _file_stamp(s1_csv),
                        "sigma_reg": _file_stamp(out_csv),
                    },
                },
                f,
                sort_keys=True,
                indent=2,
            )

    sweep_data = result.get("lambda_sweep_data")
    sweep_csv = None
    if sweep_data is not None:
        sweep_name = config.STAGE2_LAMBDA_SWEEP_CSV_TPL.format(h=h_target)
        sweep_csv = os.path.join(config.RESULTS_DIR, sweep_name)
        pd.DataFrame(sweep_data).to_csv(sweep_csv, index=False)

    eigvals = np.linalg.eigvalsh(Sigma)
    print_header("STAGE 2 COMPLETE")
    print(f"  Σ shape        : {Sigma.shape}")
    print(f"  Unit diagonal  : {np.allclose(np.diag(Sigma), 1.0)}")
    print(f"  PSD            : {bool(np.all(eigvals >= -1e-8))}  (min eigval {eigvals.min():.2e})")
    print(f"  Min off-diag   : {Sigma[~np.eye(Sigma.shape[0], dtype=bool)].min():.4f}")
    print(f"  Elapsed        : {elapsed:.2f}s")
    print(f"  Written        : {out_csv}")
    if sweep_csv is not None:
        print(f"  λ-sweep        : {sweep_csv}")
    if fit_json is not None:
        print(f"  Fit summary    : {fit_json}")
