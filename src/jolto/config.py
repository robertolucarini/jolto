import os

import numpy as np

# Entry points
DATA_DIR = "data"  # directory holding market input CSVs
RESULTS_DIR = "results"  # output directory (created if absent)
# Dated input files are named <prefix>YYYYMMDD.csv. Set the two file names, STAGE2_HIST_END and
# REGUL_DATES to your data, here or in an untracked src/jolto/config_local.py (read at the end).
CURVE_PREFIX = "disc_"  # dated discount-curve files in STAGE2_HIST_CURVE_DIR
VOL_PREFIX = "vol_"  # dated swaption-surface files under DATA_DIR/history/vols
VOL_SURFACE_CSV = "history/vols/vol_YYYYMMDD.csv"  # swaption vol surface (within DATA_DIR)
DISCOUNT_CSV = "history/curves/disc_YYYYMMDD.csv"  # discount curve (within DATA_DIR), same date
MIN_EXPIRY_YEARS = 0.25  # smallest expiry kept by the vol loaders, inclusive
STAGE1_TARGET_TENOR = 1.0  # underlying tenor (years) for Stage 1 calibration
MAX_MATURITY = 30.0  # forward curve runs to this maturity (years)
CORR_CALI_MODE = "AMMO_MC"  # Stage 2 correlation mode: AMMO_MC | PURE_MC

STAGE2_H = 0.1

# Stage 1 (SABR marginals)
BETA_SABR = 0.0
SHIFT_SABR = 0.0
H_GRID = np.array([0.025, 0.05, 0.075, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50])
# CALI_MODE options: AMMO_ODE, PURE_MC, AMMO_MC
CALI_MODE = "AMMO_MC"
CALI_PATHS = 32768
# Calibration MC time step (years) for Stage 1 and Stage 2. Distinct from CMS_DT (product grid).
CALI_DT = 1 / 50

# Bennedsen–Lunde–Pakkanen (2017) hybrid rough-vol driver: number of exact near lags.
# 0 = Volterra-Euler scheme; 1 = production hybrid; 2 = robustness check.
HYBRID_KAPPA = 1

GRID_STEP_SIZE = 1.0

USE_TORCH_COMPILE = False  # off by default; TorchInductor is unreliable on Apple-Silicon CPU


# AMMO trust region: Alexandrov (2001) Algorithm 2 constants
TR_R1 = 0.1  # poor-prediction threshold (shrink radius)
TR_R2 = 0.75  # good-prediction threshold (expand radius)
TR_C1 = 0.5  # shrink factor
TR_C2 = 2.0  # expansion factor
TR_DELTA_0 = 0.1  # initial radius
TR_DELTA_MAX = 2.0  # maximum radius (Delta*)
AMMO_MAX_ITER = 20  # Stage-1 AMMO outer-iteration cap


# Stage 2 (correlation) regularisation
USE_TIKHONOV = True
AUTO_LAMBDA = (
    False  # sweep switch, set True only by --sweep; production λ comes from STAGE2_LAMBDA_RULE
)
LAMBDA_CURVATURE = 3.16  # λ of the "fixed" rule
LAMBDA_GRID_LO = 0.02  # λ-sweep lower bound
LAMBDA_GRID_HI = 50.0  # λ-sweep upper bound
LAMBDA_GRID_N = 18  # 18 points over 3.4 decades ≈ 5.3 per decade
STAGE_2_FIRST_ORDER = True  # first-order β-correction (eq::grad_beta); False = zero-order
STAGE2_DECAY_BETA = 0.5  # 'exp' form only: tgt(d) = ρ∞ + (1−ρ∞)·exp(−β·d)
STAGE2_DECAY_RHO_INF = 0.0  # long-end floor ρ∞
STAGE2_DECAY_FORM = (
    "hist_exp"  # decay-target form: "linear" | "exp" | "hist_exp" | "hist_sc2" | "hist_reb3"
)
STAGE2_DECAY_SLOPE = 0.02  # 'linear' form only: tgt(d) = max(1 − s·d, ρ∞)
# hist_* forms: family fitted to the forward-rate history, embedded on the Stage-1 column (corr_target)
STAGE2_HIST_CURVE_DIR = os.path.join(
    DATA_DIR, "history", "curves"
)  # dated <CURVE_PREFIX>YYYYMMDD.csv files
STAGE2_HIST_END = "YYYY-MM-DD"  # window end = the vol cube's file date (jolto calibrate-correlation checks it); later is look-ahead
STAGE2_HIST_YEARS = 1  # window length (Brigo–Mercurio §6.19)
STAGE2_HIST_STEP = 5  # observations per change: 5 = weekly
# regularisation studies (jolto.calibration.regul): dated cubes and curves under data/history
REGUL_DATES = ()  # "YYYY-MM-DD" dates, each with a dated surface and curve
STAGE2_PENALTY_FIRST_FWD = (
    3  # first forward column in R_h, R_v: 3 drops forwards 1–2, which Stage 1 fixes
)
STAGE2_USE_DECAY_TARGET = True  # False = smoothness-only fallback, drops the decay term
STAGE2_LAMBDA_FLOOR = 1e-6  # λ→0 floor row prepended to the --sweep grid (the unregularised Σ)
# λ rule for the production fit: "fixed" = LAMBDA_CURVATURE | "monotone" = smallest-monotone sweep
# pick | "cv_1se" = one-SE cross-validation on held-out ATM pairs, ascending stop
STAGE2_LAMBDA_RULE = "cv_1se"
STAGE2_CV_LAMBDA_GRID = (
    1e-6,
    0.1,
    0.316,
    1.0,
    3.16,
    10.0,
    31.6,
)  # cv_1se candidate λ, walked upward
STAGE2_CV_FOLDS = 5  # cv_1se folds, stratified by Σ row
STAGE2_CV_SEEDS = (
    0,
    1,
)  # cv_1se fold-assignment seeds (not MC seeds)

# Stage 2 inner L-BFGS-B budget
# a large budget fully minimises the surrogate each outer iteration, saving HF MC anchors
STAGE2_LBFGSB_MAXITER = 200
STAGE2_LBFGSB_MAXFUN = 500
STAGE2_AMMO_MAX_ITER = 15  # Stage 2 AMMO outer-iteration cap; the loop stops early on convergence
STAGE2_MERIT_RTOL = 1e-4  # Stage 2 AMMO stop: relative merit reduction per iteration below this
CALI_PATHS_JAC = 2048  # Stage 2 HF Jacobian path count (separate smaller driver cache)

# Stage 1 output filenames (all relative to RESULTS_DIR)
STAGE1_PARAMS_CSV = "stage1_parameters.csv"
STAGE1_SURFACE_MARKET_CSV = "stage1_surface_MARKET.csv"
STAGE1_SURFACE_MODEL_CSV_TPL = "stage1_surface_{method}.csv"

# Stage 2 output filenames (all relative to RESULTS_DIR)
STAGE2_CORR_CSV_TPL = "stage2_correlation_H_{h:.3f}.csv"
STAGE2_CORR_UNREG_CSV_TPL = "stage2_correlation_H_{h:.3f}_unreg.csv"
STAGE2_LAMBDA_SWEEP_CSV_TPL = "stage2_lambda_sweep_H_{h:.3f}.csv"
STAGE2_LAMBDA_CV_CSV_TPL = (
    "stage2_lambda_cv_H_{h:.3f}.csv"  # cv_1se table, one row per seed, λ, fold
)
STAGE2_LOG_TXT = "stage2_global_correlation_log.txt"

# Stage-2 RMSE/max/bias with run settings and input stamps
STAGE2_FIT_SUMMARY_JSON = "stage2_fit_summary.json"
OOS_EXHIBIT_JSON = "oos_exhibit.json"  # out-of-sample exhibit written by tests/oos.py

# CMS engine knobs: cms.py reads them as defaults, get_price.py and jolto price via val_config.
# CMS/CMS-spread trade economics and the pricing-H knob (VAL_H_TARGET) live in val_config.py.
CMS_DT = 1 / 50
CMS_PATHS_MAIN = 32768
CMS_SEED = 56
# Drift treatment for the FMM valuation engine. True = frozen-coefficient drift (the
# default scheme); False = state-dependent Euler, which
# recomputes the 1/(1+tau*F) factor from the simulated curve at every step.
CMS_FREEZE_DRIFT = True
CMS_BENCH_VOL_STDS = 8.0
CMS_BENCH_N_STRIKES = 401

# Valuation control variate
# Deflator CV (valuation.bermudan_pricer): the 2-control β-vector is estimated on winsorised
# pilot tensors because the raw pilot β is tail-noisy. Winsorising only the pilot β-estimation (never the applied price)
# leaves the estimator unbiased: β is detached and the means are analytic.
# Two-sided fraction clipped off each pilot tensor before the OLS fit.
CV_PILOT_WINSOR_Q = 0.01

# Deflator-control guard: zero the β1·(D−m_def) CV term on paths where any co-terminal
# forward factor (1+τF) falls to/below this margin: the sign-flip region where the deflated
# annuity D blows up (clamp(P_TN,1e-12)). Legitimate rates put 1+τF near 1.0; a flip needs
# F ≤ −1/τ. Control-only; never touches cf or the raw price.
CV_DEFLATOR_GUARD_EPS = 1e-2

# Engine A/B convergence and bias harness settings.
# Both engines (A: 1-D conditional-MC; B: FMM Bermudan) and both modes (a: calibrated
# production point; b: representative (ρ,ν) sweep) read this single block; param source is
# the only switch, so the harnesses carry no literals. The two profiles share one schema:
# "full" = full-size validation (the Engine-B PRNG oracle is the runtime long pole);
# "reduced" = end-to-end validation. Oracle dt = CALI_DT (A) / PRICE_DT (B); N-ladder tops are
# CALI_PATHS (A) / PRICE_PATHS_MAIN (B). df_oracle = oracle_n/oracle_batch − 1.
CONVERGENCE = {
    "full": {
        "oracle_n_A": 10_000_000,
        "oracle_batch_A": 500_000,
        "oracle_n_B": 10_000_000,
        "oracle_batch_B": 10_000,
        "oracle_n_B_sweep": 10_000_000,
        "m_seeds": 30,
        "m_seeds_B": 200,  # Engine B only: more scrambles for right-skewed means
        "n_ladder_A": (2048, 4096, 8192, 16384, 32768),
        "n_ladder_B": (
            4096,
            8192,
            16384,
            32768,
            65536,
        ),
    },
    "reduced": {
        "oracle_n_A": 1_000_000,
        "oracle_batch_A": 100_000,
        "oracle_n_B": 100_000,
        "oracle_batch_B": 5_000,
        "oracle_n_B_sweep": 50_000,
        "m_seeds": 8,
        "m_seeds_B": 8,
        "n_ladder_A": (2048, 4096, 8192, 16384),
        "n_ladder_B": (2048, 4096, 8192, 16384),
    },
    # profile-independent (shared by full + reduced)
    "ci_level": 0.99,  # oracle CI + in_oracle_CI test level
    "expiries_A": (1.0, 5.0, 10.0),
    "strike_offsets_A": (-0.01, -0.005, 0.0, 0.005, 0.01),  # rel. strikes (decimal); 0.0 = ATM
    "atm_expiry_A": 5.0,  # expiry of the SE-vs-N ladder instrument (ATM)
    "sweep_rho_A": (-0.5, -0.25, 0.0),  # mode-(b) Engine-A ρ grid (fuller; oracle cheap)
    "sweep_nu_A": (0.3, 0.5),  # mode-(b) Engine-A ν grid
    "sweep_rho_B": (-0.25, 0.0),  # mode-(b) Engine-B ρ grid (coarse, 2 pts)
    "sweep_nu_B": (0.3, 0.5),  # mode-(b) Engine-B ν grid
    "sweep_alpha": 0.015,  # flat base normal vol (decimal) for the (ρ,ν) sweep
    "beta_decay_B": 0.05,  # Engine-B mode-(b) parametric exp-decay Σ rate (PSD; not calibrated)
    "euro_expiry_B": 5.0,  # Engine-B co-terminal European fixing T_a (years); swap runs T_a → T_N
    "euro_expiries_B": (1.0, 5.0, 10.0),  # Engine-B Mode-a production-point expiry set; T_N fixed
}

# Machine-local overrides (data file names, dates), untracked: src/jolto/config_local.py
try:
    from jolto.config_local import *  # noqa: F403
except ImportError:
    pass
