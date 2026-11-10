"""Simulated term structures: the forward curve and the discount curve along Monte Carlo paths.

`curves_at_dates` works on any RoughSABRFMM. `simulate_curves` builds the calibrated model from the
Stage-1 and Stage-2 outputs, as the pricers do, and calls it. Config-free like get_price: the
caller passes `SimulationRunOpts.from_val_config(val_config)`.
"""

import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from jolto.api.assembly import build_calibrated_model


@dataclass(frozen=True)
class SimulationRunOpts:
    n_paths: int
    dt: float
    seed: int
    dates: tuple | None = None  # None = every tenor date T_0..T_{N-1}
    freeze_drift: bool = True

    @classmethod
    def from_val_config(cls, val_config) -> "SimulationRunOpts":
        dates = val_config.SIM_DATES
        return cls(
            n_paths=val_config.SIM_PATHS,
            dt=val_config.SIM_DT,
            seed=val_config.SIM_SEED,
            dates=None if dates is None else tuple(float(d) for d in dates),
            freeze_drift=val_config.PRICE_FREEZE_DRIFT,
        )


@dataclass(frozen=True)
class SimulationResult:
    """Simulated curves at the observation dates, decimal, float64.

    dates    : (D,) observation dates in years, all tenor dates.
    tenors   : (N+1,) the tenor grid T_0..T_N in years.
    forwards : (paths, D, N); forward j accrues over (T_j, T_{j+1}] and fixes at T_j. NaN where it
               has fixed before the date (T_j < date), since the engine keeps evolving it.
    bonds    : (paths, D, N+1), P(date, T_k) from the simulated forwards; 1 at T_k = date, NaN for
               T_k < date.
    meta     : H, N, n_paths, dt, seed, freeze_drift, device and the simulation time in seconds.
    """

    dates: np.ndarray
    tenors: np.ndarray
    forwards: np.ndarray
    bonds: np.ndarray
    meta: dict


def curves_at_dates(model, dates, n_paths, dt, seed, freeze_drift=True):
    """(forwards, bonds) of `model` at `dates` (tenor dates before T_N), as in SimulationResult."""
    T = model.T.detach().cpu().numpy()
    e_idx = []
    for d in dates:
        e = int(np.argmin(np.abs(T - d)))
        if abs(T[e] - d) > 1e-9 or e >= model.N:
            raise ValueError(f"simulation date {d} is not a tenor date before T_N = {T[-1]}")
        e_idx.append(e)

    # uniform grid from 0 to the last date, as the pricers build it; every tenor date is a node
    grid_end = max(float(max(dates)), dt)
    n_steps = int(np.ceil(grid_end / dt - 1e-9))
    time_grid = torch.linspace(0.0, grid_end, n_steps + 1, device=model.device, dtype=model.dtype)
    steps = [round(d / grid_end * n_steps) for d in dates]
    for d, s in zip(dates, steps):
        if abs(time_grid[s].item() - d) > 1e-9:
            raise ValueError(
                f"simulation date {d} is not on the time grid of step {grid_end / n_steps}"
            )

    with torch.no_grad():
        F = model.simulate_forward_curve(
            n_paths, time_grid, freeze_drift=freeze_drift, seed=seed, steps=steps
        )
    F = F.cpu().numpy()
    tau = model.tau.cpu().numpy()

    forwards = F.copy()
    bonds = np.full((n_paths, len(dates), model.N + 1), np.nan)
    for i, e in enumerate(e_idx):
        forwards[:, i, :e] = np.nan
        bonds[:, i, e] = 1.0
        # P(T_e, T_k) = Π_{j=e}^{k-1} 1/(1 + τ_j F_j(T_e)), the pricers' pathwise bonds
        bonds[:, i, e + 1 :] = np.cumprod(1.0 / (1.0 + tau[e:] * F[:, i, e:]), axis=1)
    return forwards, bonds


def simulate_curves(inputs, *, run_opts, device=None) -> SimulationResult:
    """Forward and discount curves of the calibrated model at run_opts.dates.

    inputs: assembly.ValuationInputs; run_opts: SimulationRunOpts.from_val_config(val_config).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, grid_T, _ = build_calibrated_model(inputs, device)
    dates = run_opts.dates if run_opts.dates is not None else tuple(float(t) for t in grid_T[:-1])
    t0 = time.time()
    forwards, bonds = curves_at_dates(
        model, dates, run_opts.n_paths, run_opts.dt, run_opts.seed, run_opts.freeze_drift
    )
    meta = {
        "H": inputs.H,
        "N": model.N,
        "n_paths": run_opts.n_paths,
        "dt": run_opts.dt,
        "seed": run_opts.seed,
        "freeze_drift": run_opts.freeze_drift,
        "device": device,
        "seconds": time.time() - t0,
    }
    return SimulationResult(
        np.asarray(dates, dtype=float), np.asarray(grid_T, dtype=float), forwards, bonds, meta
    )


def summary_table(result: SimulationResult) -> pd.DataFrame:
    """Mean, sd and the 5/50/95% quantiles across paths, one row per date, curve and tenor.

    tenor is the accrual start T_j for a forward and the maturity T_k for a bond; NaN entries
    (fixed forwards, past maturities) are left out.
    """
    rows = []
    for curve, arr, tenors in (
        ("forward", result.forwards, result.tenors[:-1]),
        ("bond", result.bonds, result.tenors),
    ):
        live = ~np.isnan(arr[0])  # (D, K); NaN is path-independent
        with warnings.catch_warnings():  # all-NaN columns are dropped below
            warnings.simplefilter("ignore", RuntimeWarning)
            q05, q50, q95 = np.nanquantile(arr, [0.05, 0.5, 0.95], axis=0)
            mean, sd = np.nanmean(arr, axis=0), np.nanstd(arr, axis=0, ddof=1)
        for i, date in enumerate(result.dates):
            for k in np.flatnonzero(live[i]):
                rows.append(
                    {
                        "date": date,
                        "curve": curve,
                        "tenor": tenors[k],
                        "mean": mean[i, k],
                        "sd": sd[i, k],
                        "q05": q05[i, k],
                        "q50": q50[i, k],
                        "q95": q95[i, k],
                    }
                )
    return pd.DataFrame(rows)
