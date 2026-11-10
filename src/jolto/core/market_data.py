"""Market-data I/O: tenor/strike parsing, discount-curve and swaption-vol loaders, forward bootstrap.

Conventions: times are year fractions read from the tenor labels (Y = 1, M = 1/12, W = 1/52,
D = 1/365.25), with no calendar or day-count adjustment; vols are normal (Bachelier) and decimal;
strikes are absolute offsets from the forward, decimal; the forward grid is annual
(config.GRID_STEP_SIZE).
"""

import os

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from jolto import config


def parse_strike(strike_str):
    """Strike string ('ATM', '-50BP') to an absolute rate offset (50 bp = 0.0050).

    Raises ValueError on anything else: a silent 0.0 would merge the quote into the ATM column.
    """
    s = str(strike_str).upper().strip()
    if s == "ATM":
        return 0.0
    if s.endswith("BP"):
        try:
            return float(s[:-2]) / 10000.0
        except ValueError:
            pass
    raise ValueError(f"unparsable strike {strike_str!r}; expected 'ATM' or e.g. '-50bp'")


def parse_tenor(tenor_str):
    """Converts a tenor string ('1Y', '6M', '1W') to float year fraction; np.nan if unrecognised."""
    s = str(tenor_str).upper().strip()
    if "/" in s:
        s = s.split("/")[-1]

    try:
        if s.endswith("Y"):
            return float(s[:-1])
        if s.endswith("M"):
            return float(s[:-1]) / 12.0
        if s.endswith("W"):
            return float(s[:-1]) / 52.0
        if s.endswith("D"):
            return float(s[:-1]) / 365.25
    except ValueError:
        pass

    return np.nan  # NaN so duplicate T=0 rows can be dropped by callers


def load_discount_curve(csv_path):
    """Loads discount factors from a CSV and builds a continuous PCHIP interpolator."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Curve file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip().upper() for c in df.columns]

    time_col = next((c for c in df.columns if "TICKER" in c or "TENOR" in c or c == "T"), None)
    value_col = next((c for c in df.columns if "DF" in c or "VALUE" in c or "RATE" in c), None)

    if not time_col or not value_col:
        raise ValueError("Missing time or value columns in the curve CSV.")

    df["T"] = df[time_col].apply(parse_tenor)
    df["DF"] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=["T", "DF"]).sort_values("T")
    # drop duplicate tenors, keep last (usually cleanest quote)
    df = df.drop_duplicates(subset=["T"], keep="last")

    if df["T"].min() > 0.001:
        t0_row = pd.DataFrame({"T": [0.0], "DF": [1.0]})
        df = pd.concat([t0_row, df[["T", "DF"]]], ignore_index=True)

    # PCHIP on log DF preserves monotonicity: a decreasing DF curve gives positive forwards
    log_df_interpolator = PchipInterpolator(
        df["T"].values, np.log(df["DF"].values), extrapolate=True
    )
    return lambda t: np.exp(log_df_interpolator(t))


def bootstrap_forward_rates(discount_func, grid_tau, max_maturity):
    """Extracts discrete forward rates: F = (1/τ)(P(T₁)/P(T₂) − 1)."""
    time_grid = np.arange(0, max_maturity + grid_tau + 0.001, grid_tau)
    dfs = discount_func(time_grid)
    forward_rates = (dfs[:-1] / dfs[1:] - 1) / grid_tau

    return time_grid, forward_rates


def load_forward_history(curve_dir, grid_tau, max_maturity, start=None, end=None):
    """Daily forward panel from dated curve files `<config.CURVE_PREFIX>YYYYMMDD.csv` in
    curve_dir (layout in
    data/README.md), each built exactly as the model builds F₀: load_discount_curve, then
    bootstrap_forward_rates. Returns a DataFrame indexed by date, column j = forward on (T_{j-1}, T_j]."""
    if not os.path.isdir(curve_dir):
        raise FileNotFoundError(
            f"History curve directory not found: {curve_dir} (config.STAGE2_HIST_CURVE_DIR). The hist_* "
            f"decay targets need dated {config.CURVE_PREFIX}YYYYMMDD.csv files there; data/README.md gives the "
            "layout."
        )
    files = sorted(
        f for f in os.listdir(curve_dir) if f.startswith(config.CURVE_PREFIX) and f.endswith(".csv")
    )
    if not files:
        raise FileNotFoundError(f"No {config.CURVE_PREFIX}YYYYMMDD.csv files in {curve_dir}")
    rows = {}
    for f in files:
        date = pd.Timestamp(f[len(config.CURVE_PREFIX) : -len(".csv")])
        if (start is not None and date < pd.Timestamp(start)) or (
            end is not None and date > pd.Timestamp(end)
        ):
            continue
        _, fwd = bootstrap_forward_rates(
            load_discount_curve(os.path.join(curve_dir, f)), grid_tau, max_maturity
        )
        rows[date] = fwd
    panel = pd.DataFrame.from_dict(rows, orient="index").sort_index()
    panel.columns = range(1, panel.shape[1] + 1)
    return panel


def load_swaption_vol_surface(csv_path, target_underlying_tenor):
    """Loads normal swaption vol surface for a given underlying tenor, expiries from
    config.MIN_EXPIRY_YEARS up; returns Expiry×Strike pivot."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Vol file not found: {csv_path}")

    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip().upper() for c in df.columns]

    expiry_col = next((c for c in df.columns if "EXPIRY" in c), None)
    tenor_col = next((c for c in df.columns if "UNDERLYING" in c or "TENOR" in c), None)
    strike_col = next((c for c in df.columns if "STRIKE" in c), None)
    vol_col = next((c for c in df.columns if ("VALUE" in c or "VOL" in c) and "ID" not in c), None)

    if not all([expiry_col, tenor_col, strike_col, vol_col]):
        raise ValueError("Missing required columns in volatility CSV.")

    df["Expiry"] = df[expiry_col].apply(parse_tenor)
    df["Underlying_Tenor"] = df[tenor_col].apply(parse_tenor)
    df["Strike_Offset"] = df[strike_col].apply(parse_strike)
    df["Vol"] = pd.to_numeric(df[vol_col], errors="coerce")

    df = df.dropna(subset=["Expiry", "Underlying_Tenor", "Strike_Offset", "Vol"])
    df = df[
        df["Expiry"] >= config.MIN_EXPIRY_YEARS - 1e-9
    ]  # inclusive; tolerance for tenor rounding
    # 0.05 tolerance avoids floating-point mismatch on tenor comparisons
    target_tenor_df = df[np.abs(df["Underlying_Tenor"] - target_underlying_tenor) < 0.05].copy()

    if target_tenor_df.empty:
        raise ValueError(
            f"No volatility data found for target underlying tenor: {target_underlying_tenor}Y"
        )

    vol_matrix = target_tenor_df.pivot_table(
        values="Vol", index="Expiry", columns="Strike_Offset", aggfunc="first"
    )
    return vol_matrix.sort_index().sort_index(axis=1)


def load_atm_matrix(csv_path):
    """Loads the ATM volatility matrix from a CSV, expiries from config.MIN_EXPIRY_YEARS up; returns
    Expiry×Tenor pivot."""
    df = pd.read_csv(csv_path, sep=None, engine="python")
    df.columns = [str(c).strip().upper() for c in df.columns]

    expiry_col = next(c for c in df.columns if "EXPIRY" in c)
    tenor_col = next(c for c in df.columns if "UNDERLYING" in c or "TENOR" in c)
    strike_col = next(c for c in df.columns if "STRIKE" in c)
    vol_col = next(c for c in df.columns if "VOL" in c or "VALUE" in c)

    df = df[df[strike_col].astype(str).str.strip().str.upper() == "ATM"].copy()
    df[expiry_col] = df[expiry_col].apply(parse_tenor)
    df[tenor_col] = df[tenor_col].apply(parse_tenor)
    df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce")
    df = df.dropna(subset=[expiry_col, tenor_col, vol_col])
    df = df[
        df[expiry_col] >= config.MIN_EXPIRY_YEARS - 1e-9
    ]  # inclusive; tolerance for tenor rounding

    return df.pivot_table(values=vol_col, index=expiry_col, columns=tenor_col, aggfunc="first")
