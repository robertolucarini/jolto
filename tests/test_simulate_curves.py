"""Simulated curves on a small model: no market data or calibration outputs needed.

Checks the layout of curves_at_dates (fixed forwards and past maturities masked, P(t,t) = 1), that
date 0 returns the initial curve exactly, the pathwise bonds against the forwards, and the summary.
"""

import numpy as np

from jolto.api.simulate import SimulationResult, curves_at_dates, summary_table
from jolto.core.model import RoughSABRFMM


def _model():
    tenors = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    return RoughSABRFMM(
        tenors=tenors,
        F0=np.array([0.02, 0.025, 0.03, 0.03]),
        alpha_f=lambda t: 0.01 * np.ones_like(t),
        rho_f=lambda t: -0.3 * np.ones_like(t),
        nu=0.5,
        H=0.1,
        device="cpu",
    )


def test_curves_at_dates():
    model = _model()
    dates = (0.0, 1.0, 3.0)
    fwd, bonds = curves_at_dates(model, dates, n_paths=256, dt=1 / 24, seed=42)
    assert fwd.shape == (256, 3, 4) and bonds.shape == (256, 3, 5)

    F0 = model.F0.detach().numpy()
    tau = model.tau.numpy()
    assert np.array_equal(
        fwd[:, 0, :], np.broadcast_to(F0, (256, 4))
    )  # date 0 is the initial curve
    assert np.allclose(bonds[0, 0, 1:], np.cumprod(1 / (1 + tau * F0)))

    for i, e in enumerate((0, 1, 3)):
        assert np.isnan(fwd[:, i, :e]).all() and not np.isnan(fwd[:, i, e:]).any()
        assert np.isnan(bonds[:, i, :e]).all() and (bonds[:, i, e] == 1.0).all()
        rebuilt = np.cumprod(1 / (1 + tau[e:] * fwd[:, i, e:]), axis=1)
        assert np.allclose(bonds[:, i, e + 1 :], rebuilt)
    assert fwd[:, 1, 1:].std(axis=0).min() > 0  # forwards move after time 0

    try:
        curves_at_dates(model, (1.5,), n_paths=8, dt=1 / 24, seed=42)
        raise AssertionError("an off-grid date must raise")
    except ValueError:
        pass


def test_summary_table():
    model = _model()
    dates = (0.0, 2.0)
    fwd, bonds = curves_at_dates(model, dates, n_paths=128, dt=1 / 24, seed=7)
    res = SimulationResult(np.array(dates), model.T.numpy(), fwd, bonds, {})
    tab = summary_table(res)
    # forwards: 4 live at date 0, 2 at date 2; bonds: 5 and 3
    assert len(tab) == 4 + 2 + 5 + 3
    assert not tab[["mean", "sd", "q05", "q50", "q95"]].isna().any().any()
    assert (tab["q05"] <= tab["q50"]).all() and (tab["q50"] <= tab["q95"]).all()


if __name__ == "__main__":
    test_curves_at_dates()
    test_summary_table()
    print("ALL CHECKS PASSED")
