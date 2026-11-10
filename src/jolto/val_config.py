"""Consolidated valuation config: owns valuation-only literals, re-exports shared ones.

Single source of truth per value: each lives in exactly one file. Valuation-only literals
(Bermudan / CMS trade economics + run knobs + the pricing-H knob) are defined here. Shared
market/grid/SABR constants and the pricer-internal CMS fallbacks (CMS_DT/CMS_PATHS_MAIN/CMS_SEED)
stay defined in config.py and are re-exported here, because library internals (cms.py) read them
directly.

Dependency is one-way: val_config -> config (config must never import val_config).

VAL_H_TARGET is the single canonical pricing-H knob.
Latent H coupling: VAL_H_TARGET must match the H at which Σ was calibrated; assembly's
Σ-file-existence check enforces this implicitly, so keep VAL_H_TARGET and config.STAGE2_H
in step.
"""

from jolto.config import (
    # shared market / grid / SABR (defined in config; read by library internals)
    DATA_DIR,
    RESULTS_DIR,
    DISCOUNT_CSV,
    GRID_STEP_SIZE,
    MAX_MATURITY,
    BETA_SABR,
    SHIFT_SABR,
    # valuation input locations (where assembly finds the calibration outputs)
    STAGE1_PARAMS_CSV,
    STAGE2_CORR_CSV_TPL,
    # CMS pricer-internal fallbacks (cms.py reads these directly)
    CMS_DT,
    CMS_PATHS_MAIN,
    CMS_FREEZE_DRIFT,
    CMS_SEED,
)

# Valuation-only literals (defined here)

# Bermudan
PRICE_STRIKE = 0.03  # fixed rate (decimal)
PRICE_FIRST_EX = 2.0  # first exercise date (years)
PRICE_LAST_EX = 29.0  # last exercise date (years)
PRICE_PATHS_MAIN = int(32768)  # main-run path count
PRICE_PATHS_PILOT = 4096  # pilot-run path count (OOS boundary estimation)
PRICE_DT = 1.0 / 24.0  # FMM simulation time step (years)
PRICE_SEED = 42  # RNG seed
PRICE_FREEZE_DRIFT = True  # True = frozen-coefficient drift; False = state-dependent Euler
PRICE_USE_CONTROL_VARIATE = (
    False  # vector deflator CV; off: its mean is exact only in continuous time
)

# CMS caplet/floorlet
CMS_FIXING = 5.0  # T_a: 5Y fixing
CMS_SWAP_END = 15.0  # T_b: 10Y underlying swap → "5Y into 10Y"
CMS_STRIKE = 0.025  # K: 2.5%
CMS_YEAR_FRACTION = 1.0  # δ: annual accrual
CMS_IS_CAP = True  # caplet (S−K)^+

# CMS spread option
CMS_SPREAD_FIXING = 5.0  # T_a fixing (years)
CMS_SPREAD_LONG_TENOR = 10.0  # long-leg underlying tenor  → S¹ = S_{a,a+10}(T_a)
CMS_SPREAD_SHORT_TENOR = 2.0  # short-leg underlying tenor → S² = S_{a,a+2}(T_a)
CMS_SPREAD_STRIKE = 0.0  # K on the spread (decimal); 0.0 = pure spread
CMS_SPREAD_YEAR_FRACTION = 1.0  # δ: annual accrual
CMS_SPREAD_IS_CAP = True  # call on the spread (S¹−S²−K)^+; False = put

# Val-layer knobs
VAL_H_TARGET = 0.10  # single canonical pricing-H knob; selects Stage-1 slice + Σ file
VAL_GREEKS = ("PV01", "Vega")  # default greek set the façade/CLI request

# Curve simulation (jolto simulate-curves)
SIM_PATHS = 4096  # paths kept on disk; the .npz grows as paths × dates × (2N + 1) float64
SIM_DATES = None  # observation dates in years, on the tenor grid; None = every T_0..T_{N-1}
SIM_DT = PRICE_DT  # time step of the simulation (years)
SIM_SEED = PRICE_SEED
SIM_NPZ_TPL = "simulated_curves_H_{h:.3f}.npz"
SIM_SUMMARY_CSV_TPL = "simulated_curves_summary_H_{h:.3f}.csv"

__all__ = (
    # shared market / grid / SABR
    "DATA_DIR",
    "RESULTS_DIR",
    "DISCOUNT_CSV",
    "GRID_STEP_SIZE",
    "MAX_MATURITY",
    "BETA_SABR",
    "SHIFT_SABR",
    # valuation input locations
    "STAGE1_PARAMS_CSV",
    "STAGE2_CORR_CSV_TPL",
    # CMS pricer-internal fallbacks
    "CMS_DT",
    "CMS_PATHS_MAIN",
    "CMS_FREEZE_DRIFT",
    "CMS_SEED",
    # Bermudan
    "PRICE_STRIKE",
    "PRICE_FIRST_EX",
    "PRICE_LAST_EX",
    "PRICE_PATHS_MAIN",
    "PRICE_PATHS_PILOT",
    "PRICE_DT",
    "PRICE_SEED",
    "PRICE_FREEZE_DRIFT",
    "PRICE_USE_CONTROL_VARIATE",
    # CMS
    "CMS_FIXING",
    "CMS_SWAP_END",
    "CMS_STRIKE",
    "CMS_YEAR_FRACTION",
    "CMS_IS_CAP",
    # CMS-spread
    "CMS_SPREAD_FIXING",
    "CMS_SPREAD_LONG_TENOR",
    "CMS_SPREAD_SHORT_TENOR",
    "CMS_SPREAD_STRIKE",
    "CMS_SPREAD_YEAR_FRACTION",
    "CMS_SPREAD_IS_CAP",
    # val-layer knobs
    "VAL_H_TARGET",
    "VAL_GREEKS",
    # curve simulation
    "SIM_PATHS",
    "SIM_DATES",
    "SIM_DT",
    "SIM_SEED",
    "SIM_NPZ_TPL",
    "SIM_SUMMARY_CSV_TPL",
)
