"""Helpers shared by the Stage-2 regularisation diagnostics: build the Stage-2 problem
from files, calibrate in-process under temporary config overrides, and price a Σ through the same
mapped-SMM high-fidelity pricer on an independent Sobol seed."""

import contextlib
import io
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

from jolto import config
from jolto.calibration.pricers import _hybrid_driver
from jolto.calibration.stage2 import CorrelationCalibrator
from jolto.calibration.pipeline import build_model_from_stage1
from jolto.calibration.pricers import mapped_smm_mc_pricer
from jolto.core.analytics import (
    build_rapisarda_correlation_matrix,
    generate_sobol_bridge_increments,
)
from jolto.core.market_data import (
    bootstrap_forward_rates,
    load_atm_matrix,
    load_discount_curve,
)

PROD = {
    "disc": os.path.join(config.DATA_DIR, config.DISCOUNT_CSV),
    "vol": os.path.join(config.DATA_DIR, config.VOL_SURFACE_CSV),
    "stage1": os.path.join(config.RESULTS_DIR, config.STAGE1_PARAMS_CSV),
}


def prod_lambda():
    """The λ of the stored production fit: lambda_selected of the Stage-2 fit summary in results/
    (cv_1se), or its lambda under the "fixed" rule. config.LAMBDA_CURVATURE is only the λ of
    the "fixed" rule."""
    path = os.path.join(config.RESULTS_DIR, config.STAGE2_FIT_SUMMARY_JSON)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"no Stage-2 fit summary at {path}: run jolto calibrate-correlation first"
        )
    with open(path) as fh:
        fit = json.load(fh)
    return float(fit.get("lambda_selected", fit["lambda"]))


def build_problem(disc_csv, vol_csv, stage1_csv, h=None):
    """(model, atm_matrix) exactly as pipeline.run_stage2 builds them."""
    h = config.STAGE2_H if h is None else h
    grid_T, F0 = bootstrap_forward_rates(
        load_discount_curve(disc_csv), config.GRID_STEP_SIZE, config.MAX_MATURITY
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model_from_stage1(pd.read_csv(stage1_csv), grid_T, F0, h, device)
    return model, load_atm_matrix(vol_csv)


@contextlib.contextmanager
def overrides(**kv):
    old = {k: getattr(config, k) for k in kv}
    for k, v in kv.items():
        setattr(config, k, v)
    try:
        yield
    finally:
        for k, v in old.items():
            setattr(config, k, v)


def calibrate(model, atm, quiet=True, **cfg):
    """Stage-2 AMMO calibration under config overrides. Returns (result, calibrator)."""
    with overrides(**cfg):
        cal = CorrelationCalibrator(atm, model)
        sink = io.StringIO() if quiet else sys.stdout
        with contextlib.redirect_stdout(sink):
            res = cal.calibrate(method="AMMO_MC", log_file="")
    return res, cal


def setup_of(cal, **cfg):
    """The calibrator's target grid and MC cache (as calibrate builds them), quietly."""
    with overrides(**cfg), contextlib.redirect_stdout(io.StringIO()):
        return cal._build_correlation_target_grid()


def free_angles(omega, setup):
    return omega[setup["r_idx"].cpu().numpy(), setup["c_idx"].cpu().numpy()]


def mc_cache(cal, n_steps, seed):
    """cal._generate_mc_cache on another Sobol seed (for synthetic quotes without the inverse crime)."""
    dz = generate_sobol_bridge_increments(
        config.CALI_PATHS, n_steps, 1, config.CALI_DT, cal.device, cal.model.dtype, seed=seed
    )[..., 0]
    with torch.no_grad():
        fbm, var_vec, kernel = _hybrid_driver(
            dz,
            cal.model.H,
            n_steps,
            config.CALI_DT,
            config.HYBRID_KAPPA,
            cal.device,
            cal.model.dtype,
        )
    return {
        "fbm": fbm.to(torch.float32),
        "kernel": kernel.to(torch.float32),
        "dz_vol": dz.to(torch.float32),
        "var_vec": var_vec.to(torch.float32) if var_vec is not None else None,
        "hybrid_kappa": config.HYBRID_KAPPA,
    }


def price(cal, setup, Sigma, drivers):
    """High-fidelity mapped-SMM normal vols (decimal) of the Stage-2 targets under Σ."""
    S = torch.as_tensor(Sigma, device=cal.device, dtype=cal.model.dtype)
    with torch.no_grad():
        v = mapped_smm_mc_pricer(
            cal.model,
            S,
            setup["global_exp"],
            setup["global_ten"],
            setup["global_str"],
            dt=setup["dt_val"],
            n_paths=setup["n_paths_val"],
            cached_drivers=drivers,
        )
    return np.asarray(v, dtype=float)


def sigma_of(omega):
    return build_rapisarda_correlation_matrix(torch.as_tensor(omega)).cpu().numpy()


def ff_rms(A, B, first=3):
    """RMS difference over the strict lower forward-forward block, forwards first..N (Σ indices)."""
    n = A.shape[0]
    ii, jj = np.tril_indices(n, -1)
    m = (ii >= first) & (jj >= first)
    return float(np.sqrt(np.mean((A - B)[ii[m], jj[m]] ** 2)))
