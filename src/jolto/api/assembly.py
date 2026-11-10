"""Shared calibrated-model assembly.

`build_calibrated_model` holds the mechanics (discount curve → forward-rate bootstrap →
Stage-1 marginal reconstruction → Stage-2 Σ injection) in one place. The Stage-1 H slice is selected by value from `inputs.H`.
A missing Stage-1 CSV or Σ file, an absent H (the error lists the available ones), and a Σ
that is not (N+1)×(N+1) all raise.

`load_valuation_inputs(val_config)` is the only place config-derived values enter; the
returned `ValuationInputs` is self-contained so `build_calibrated_model` stays config-free.
"""

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from jolto.calibration.pipeline import build_model_from_stage1
from jolto.core.market_data import bootstrap_forward_rates, load_discount_curve
from jolto.valuation.cms import _inject_sigma


@dataclass(frozen=True)
class ValuationInputs:
    """Self-contained inputs for building a calibrated model (no config dependency)."""

    data_dir: str
    discount_name: str
    results_dir: str
    stage1_name: str
    sigma_tpl: str
    H: float
    grid_step: float
    max_maturity: float
    beta_sabr: float
    shift_sabr: float


def load_valuation_inputs(val_config) -> ValuationInputs:
    """Capture every needed scalar from val_config into a config-free ValuationInputs."""
    return ValuationInputs(
        data_dir=val_config.DATA_DIR,
        discount_name=val_config.DISCOUNT_CSV,
        results_dir=val_config.RESULTS_DIR,
        stage1_name=val_config.STAGE1_PARAMS_CSV,
        sigma_tpl=val_config.STAGE2_CORR_CSV_TPL,
        H=float(val_config.VAL_H_TARGET),
        grid_step=val_config.GRID_STEP_SIZE,
        max_maturity=val_config.MAX_MATURITY,
        beta_sabr=val_config.BETA_SABR,
        shift_sabr=val_config.SHIFT_SABR,
    )


def _load_sigma(inputs: ValuationInputs, device, dtype):
    """Locate and load the Stage 2 Σ matrix for the pricing H (raises on absence)."""
    name = inputs.sigma_tpl.format(h=inputs.H)
    path = os.path.join(inputs.results_dir, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Stage 2 correlation matrix not found at {path}. Run jolto calibrate-correlation first."
        )
    sigma_np = pd.read_csv(path, header=None).values.astype(np.float64)
    return torch.tensor(sigma_np, device=device, dtype=dtype)


def build_calibrated_model(inputs: ValuationInputs, device):
    """Single calibrated-model build path; returns (model, grid_T, Sigma_t)."""

    disc_csv = os.path.join(inputs.data_dir, inputs.discount_name)
    s1_csv = os.path.join(inputs.results_dir, inputs.stage1_name)

    if not os.path.exists(s1_csv):
        raise FileNotFoundError(
            f"Stage 1 output not found at {s1_csv}. Run jolto calibrate-sabr first."
        )

    s1_df = pd.read_csv(s1_csv)
    h_target = float(inputs.H)
    available_H = sorted(s1_df["H"].unique())
    if not any(abs(h - h_target) < 1e-3 for h in available_H):
        raise ValueError(
            f"H={h_target} not found in {s1_csv}. "
            f"Available H slices: {[round(h, 3) for h in available_H]}"
        )

    disc_func = load_discount_curve(disc_csv)
    grid_T, F0_rates = bootstrap_forward_rates(
        disc_func,
        grid_tau=inputs.grid_step,
        max_maturity=inputs.max_maturity,
    )

    model = build_model_from_stage1(
        s1_df, grid_T, F0_rates, h_target, device, inputs.beta_sabr, inputs.shift_sabr
    )

    Sigma_t = _load_sigma(inputs, device, model.dtype)

    if Sigma_t.shape[0] != model.N + 1:
        raise ValueError(
            f"Σ is {Sigma_t.shape[0]}×{Sigma_t.shape[0]} but model has N+1={model.N + 1}."
        )

    _inject_sigma(model, Sigma_t)  # the returned prior-state clones are not needed here

    return model, grid_T, Sigma_t
