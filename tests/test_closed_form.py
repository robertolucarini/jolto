"""The closed-form surrogate (sec::closed_form) against Hagan's normal SABR at H = 1/2.

At H = 1/2 the Fukasawa-Gatheral formula with the level of Hagan et al. (2002) must reduce to their
normal (beta = 0) SABR formula at the model's vol-of-vol nu, which checks the factor 2 in the scaled
moneyness y (their kernel drives the forward variance alpha^2). The numpy (Stage 1) and torch
(Stage-1 subproblem, Stage 2) copies must agree for any H.
"""

import numpy as np
import torch

from jolto.calibration import RoughSABRCalibrator
from jolto.calibration.pricers import _core_fukasawa_ode

K = np.array([-0.02, -0.01, -0.0025, 0.0025, 0.01, 0.02])  # strike offset K - F
T, ALPHA, RHO, NU = 2.0, 0.008, -0.3, 0.6


def _hagan_normal(k, T, alpha, rho, nu):
    z = nu / alpha * (-k)  # Hagan's zeta = (nu / alpha) (F - K)
    x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
    return alpha * z / x * (1.0 + (2.0 - 3.0 * rho**2) / 24.0 * nu**2 * T)


def test_closed_form_reduces_to_hagan_at_half():
    got = RoughSABRCalibrator.rough_sabr_vol_ode(None, K, T, ALPHA, RHO, NU, 0.5)
    np.testing.assert_allclose(got, _hagan_normal(K, T, ALPHA, RHO, NU), rtol=1e-6)


def test_numpy_and_torch_copies_agree():
    for H in (0.1, 0.5):
        a = RoughSABRCalibrator.rough_sabr_vol_ode(None, K, T, ALPHA, RHO, NU, H)
        t = lambda v: torch.tensor(v, dtype=torch.float64)  # noqa: E731
        b = _core_fukasawa_ode(t(K), t(T), t(ALPHA), t(RHO), t(NU), t(H)).numpy()
        np.testing.assert_allclose(a, b, rtol=1e-10)
