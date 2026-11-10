"""
jolto calibrate-sabr: Stage 1 calibration entry point.

Calibrates the per-expiry 1D rough SABR marginals (α, ρ, ν, H) to a market
swaption vol surface for every H in config.H_GRID. Parameters and filenames
come from jolto/config.py.

Usage:
    jolto calibrate-sabr

Output (in results/):
    stage1_parameters.csv  with columns:
        Method, H, Expiry, Alpha, Rho, Nu, RMSE_bps
    (consumed by jolto calibrate-correlation, jolto price and tests/oos.py)
    stage1_surface_MARKET.csv     market normal vol surface (decimal)
    stage1_surface_<METHOD>.csv   model normal vol surface, every H
    stage1_cali_<timestamp>.txt   run log

Required config.py entries (see "CONFIG CONTRACT" below). If any are missing,
the script reports which one and exits.
"""

import argparse
import datetime
import os

from jolto import config
from jolto.calibration.pipeline import run_stage1
from jolto.core.utils import DualLogger, check_config, require_files


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CONTRACT
# These names must exist in jolto/config.py. Listed here so a missing entry
# produces a clear message instead of an AttributeError deep in the run.
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = [
    "DATA_DIR",  # str: directory holding market input CSVs
    "VOL_SURFACE_CSV",  # str: swaption vol surface filename (within DATA_DIR)
    "MIN_EXPIRY_YEARS",  # float: smallest expiry kept by the vol loader (inclusive)
    "STAGE1_TARGET_TENOR",  # float: underlying tenor (years) to calibrate Stage 1 against
    "RESULTS_DIR",  # str: output directory (created if absent)
    "CALI_MODE",  # str: "AMMO_MC" | "AMMO_ODE" | "PURE_MC"
    "H_GRID",  # np.ndarray: Hurst candidates for the profile search
    "STAGE1_PARAMS_CSV",  # str: stage1 parameter table filename
    "STAGE1_SURFACE_MARKET_CSV",  # str: market vol surface filename
    "STAGE1_SURFACE_MODEL_CSV_TPL",  # str: model vol surface filename template ({method})
]


def main(argv=None):
    argparse.ArgumentParser(
        prog="jolto calibrate-sabr",
        description="Stage 1: calibrate the rough SABR marginals on every H of config.H_GRID. "
        "Settings come from jolto/config.py; outputs go to config.RESULTS_DIR.",
    ).parse_args(argv)
    check_config(config, REQUIRED, "jolto calibrate-sabr")
    require_files([os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV)])
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    with DualLogger(os.path.join(config.RESULTS_DIR, f"stage1_cali_{ts}.txt")):
        run_stage1()


if __name__ == "__main__":
    main()
