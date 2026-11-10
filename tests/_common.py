"""Helpers shared by the validation scripts in tests/.

The scripts run as `python tests/<name>.py`, which puts tests/ on sys.path, so they import these
with `from _common import ...`. Callers outside tests/ insert tests/ on sys.path before importing a
test module.
"""

import argparse
import math

import numpy as np
import scipy.stats as stats
import torch


def _grid_label(dt):
    return f"1/{round(1.0 / dt)}"


def _stage1_rmse(df, H):
    """Mean Stage-1 RMSE (bps) of the rows of the Stage-1 parameter frame at H."""
    return float(df[np.abs(df["H"] - H) < 1e-6]["RMSE_bps"].mean())


def _time0_swap(model, k_idx):
    """Time-0 forward swap rate S₀ (ATM strike), annuity A₀, and terminal bond P(0,T_N) for the
    co-terminal swap T_{k_idx} → T_N, from the deterministic F₀ curve."""
    tau = model.tau.detach()
    F0 = model.F0.detach()
    P0 = torch.cumprod(1.0 / (1.0 + tau * F0), dim=0)  # P(0, T_{j+1})
    P_Ta = P0[k_idx - 1] if k_idx > 0 else torch.tensor(1.0, dtype=torch.float64, device="cpu")
    P_TN = P0[-1]
    A0 = torch.sum(tau[k_idx:] * P0[k_idx:])
    S0 = (P_Ta - P_TN) / A0
    return S0, A0, P_TN


def _welch(bias, se_est, df_est, se_oracle, df_oracle):
    """Welch two-sample t on a mean difference; Satterthwaite df; two-tailed p."""
    var = se_est**2 + se_oracle**2
    t = abs(bias) / math.sqrt(max(var, 1e-300))
    df = var**2 / (se_est**4 / max(df_est, 1) + se_oracle**4 / max(df_oracle, 1))
    return t, df, 2.0 * stats.t.sf(t, df=df)


def _slope(Ns, ses):
    """log-log slope of SE vs N (expect ≈ −0.5)."""
    return float(np.polyfit(np.log(Ns), np.log(ses), 1)[0])


def _cli_profile(cfg):
    """`--full` picks the paper-grade profile of config.CONVERGENCE, else the reduced one.
    Returns (profile key, profile dict)."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--full", action="store_true", help="paper-grade config (else reduced validation)"
    )
    prof_key = "full" if ap.parse_args().full else "reduced"
    return prof_key, cfg[prof_key]
