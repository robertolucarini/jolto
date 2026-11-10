"""
jolto calibrate-correlation: Stage 2 calibration entry point (spatial correlation).

Reconstructs the FMM model from the Stage 1 marginals (results/stage1_parameters.csv),
calibrates the Rapisarda spatial correlation matrix Σ to the ATM co-terminal
swaption surface, then writes Σ to results/stage2_correlation_H_<H>.csv.

Parameters and filenames come from jolto/config.py.

Prerequisite:
    jolto calibrate-sabr must have been run first (produces stage1_parameters.csv).

Usage:
    jolto calibrate-correlation            # production: select λ by STAGE2_LAMBDA_RULE, emit the canonical Σ
    jolto calibrate-correlation --sweep    # λ-sweep: emit the sweep grid + the λ→0 unregularised Σ

Output (production):
    results/stage2_correlation_H_<H>.csv         (the (N+1)x(N+1) Σ matrix, no header/index)
    (schema consumed by jolto price and tests/oos.py; PURE_MC adds a _puremc suffix)
    results/stage2_fit_summary.json              (RMSE, max error, mean bias, input hashes; not for PURE_MC)
    results/stage2_lambda_cv_H_<H>.csv           (STAGE2_LAMBDA_RULE="cv_1se": per seed/λ/fold CV)
    results/stage2_lambda_sweep_H_<H>.csv        (STAGE2_LAMBDA_RULE="monotone": the sweep table)

Output (--sweep):
    results/stage2_lambda_sweep_H_<H>.csv        (per-λ diagnostics, incl. the λ→0 floor row)
    results/stage2_correlation_H_<H>_unreg.csv   (the λ→0 floor Σ, unregularised)
    No canonical Σ is written in sweep mode.
"""

import argparse
import os
import sys

from jolto import config
from jolto.calibration.pipeline import build_model_from_stage1, check_hist_end, run_stage2
from jolto.core.utils import check_config, require_files

_build_model_from_stage1 = build_model_from_stage1  # kept for scripts that import it from here


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CONTRACT
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = [
    "DATA_DIR",  # str: directory holding market input CSVs
    "RESULTS_DIR",  # str: output directory (also where stage1_parameters.csv lives)
    "DISCOUNT_CSV",  # str: discount curve filename (within DATA_DIR)
    "VOL_SURFACE_CSV",  # str: swaption vol surface filename (within DATA_DIR), ATM matrix source
    "MIN_EXPIRY_YEARS",  # float: smallest expiry kept by the vol loader (inclusive)
    "GRID_STEP_SIZE",  # float: tenor grid spacing (years)
    "MAX_MATURITY",  # float: forward curve runs to this maturity
    "BETA_SABR",  # float: SABR beta
    "SHIFT_SABR",  # float: displaced-diffusion shift
    "CORR_CALI_MODE",  # str: Stage 2 mode, "AMMO_MC" | "PURE_MC"
    "STAGE2_DECAY_BETA",  # float: β of the "exp" decay target
    "STAGE2_DECAY_RHO_INF",  # float: long-end floor ρ∞ of the decay target (both forms)
    "STAGE2_CORR_UNREG_CSV_TPL",  # str: unregularised Σ filename template ({h:.3f})
    "STAGE2_H",  # float: Hurst slice Stage 2 calibrates at
    "STAGE2_DECAY_FORM",  # str: decay-target form, "linear" | "exp" | "hist_exp" (default) | "hist_sc2" | "hist_reb3"
    "STAGE2_HIST_CURVE_DIR",  # str: dated curve files for the hist_* targets
    "STAGE2_HIST_END",  # str: hist_* window end, must equal the vol cube's file date (_check_hist_end)
    "STAGE2_HIST_YEARS",  # int: hist_* window length (years)
    "STAGE2_HIST_STEP",  # int: hist_* observations per change (5 = weekly)
    "STAGE2_PENALTY_FIRST_FWD",  # int: first forward column in R_h, R_v (3 = default)
    "STAGE2_DECAY_SLOPE",  # float: slope s of the "linear" decay target
    "LAMBDA_CURVATURE",  # float: λ of the "fixed" rule
    "LAMBDA_GRID_LO",  # float: --sweep λ grid lower bound
    "LAMBDA_GRID_HI",  # float: --sweep λ grid upper bound
    "LAMBDA_GRID_N",  # int: --sweep λ grid size
    "STAGE2_LAMBDA_FLOOR",  # float: λ→0 floor row of the --sweep
    "CALI_PATHS",  # int: MC path count (stamped into the fit summary)
    "CALI_DT",  # float: MC time step (stamped into the fit summary)
    "HYBRID_KAPPA",  # int: hybrid-scheme near lags (stamped into the fit summary)
    "STAGE1_PARAMS_CSV",  # str: Stage 1 parameter table filename (within RESULTS_DIR)
    "STAGE2_LAMBDA_SWEEP_CSV_TPL",  # str: λ-sweep filename template ({h:.3f})
    "STAGE2_FIT_SUMMARY_JSON",  # str: fit summary filename (within RESULTS_DIR)
    "STAGE2_LAMBDA_RULE",  # str: production λ rule, "fixed" | "monotone" | "cv_1se"
    "STAGE2_CV_LAMBDA_GRID",  # tuple[float]: cv_1se candidate λ
    "STAGE2_CV_FOLDS",  # int: cv_1se folds
    "STAGE2_CV_SEEDS",  # tuple[int]: cv_1se fold-assignment seeds
    "STAGE2_LAMBDA_CV_CSV_TPL",  # str: cv_1se table filename template ({h:.3f})
    "STAGE2_LOG_TXT",  # str: Stage-2 run log filename (within RESULTS_DIR)
]


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="jolto calibrate-correlation", description="Stage 2 correlation calibration."
    )
    p.add_argument(
        "--sweep",
        action="store_true",
        help="Run the λ sweep instead of the production fit: writes the sweep table and the "
        "unregularised Σ (λ→0), and no production Σ. config.py is not modified.",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    check_config(config, REQUIRED, "jolto calibrate-correlation")
    require_files(
        [
            os.path.join(config.DATA_DIR, config.DISCOUNT_CSV),
            os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV),
        ]
    )
    require_files(
        [os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV)],
        hint="Run `jolto calibrate-sabr` first.",
    )
    try:
        check_hist_end()
    except ValueError as e:
        sys.exit(f"ERROR: {e}")
    if config.STAGE2_DECAY_FORM.startswith("hist_"):
        require_files([config.STAGE2_HIST_CURVE_DIR])
    run_stage2(sweep=args.sweep)


if __name__ == "__main__":
    main()
