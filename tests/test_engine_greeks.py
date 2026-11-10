"""Standalone greek-surface check; needs no calibration artefacts.

Asserts read_greeks() reproduces the entry scripts' `grad * 1e-4` ladders byte-for-byte,
that RoughSABRFMM exposes the member surface the pricers read, that differentiable_parameters
hands back the live leaves (not detached copies), and that read_greeks raises on a missing
grad and on an unknown name.
"""

import numpy as np
import torch

from jolto.core.model import RoughSABRFMM


def _build_small_model():
    # Tiny tenor grid; constant marginal callables are enough to exercise the greek surface.
    tenors = np.array([1.0, 2.0, 3.0, 4.0])
    F0 = np.array([0.03, 0.03, 0.03])
    return RoughSABRFMM(
        tenors=tenors,
        F0=F0,
        alpha_f=lambda t: 0.01 * np.ones_like(t),
        rho_f=lambda t: -0.3 * np.ones_like(t),
        nu=0.5,
        H=0.1,
        device="cpu",
    )


def test_engine_greeks():
    torch.manual_seed(0)
    model = _build_small_model()

    # Known grads, matching parameter shapes/dtype.
    model.F0.grad = torch.randn_like(model.F0)
    model.alphas.grad = torch.randn_like(model.alphas)

    g = model.read_greeks()
    assert np.array_equal(g["PV01"], model.F0.grad.detach().cpu().numpy() * 1e-4)
    assert np.array_equal(g["Vega"], model.alphas.grad.detach().cpu().numpy() * 1e-4)
    print("OK: read_greeks() reproduces grad * 1e-4 byte-for-byte (PV01, Vega)")

    # Duck-typed surface the pricers (bermudan_swaption, cms) read from the model.
    expected = (
        "T",
        "tau",
        "N",
        "device",
        "dtype",
        "simulate_forward_curve",
        "get_terminal_bond",
        "differentiable_parameters",
        "read_greeks",
    )
    missing = [a for a in expected if not hasattr(model, a)]
    assert not missing, f"missing engine members: {missing}"
    print("OK: RoughSABRFMM exposes every pricer-facing member (hasattr surface)")

    params = model.differentiable_parameters()
    assert set(params) == {"F0", "alphas"}
    assert params["F0"] is model.F0 and params["alphas"] is model.alphas
    print("OK: differentiable_parameters() returns the live leaves (not detached)")

    model.alphas.grad = None
    raised = False
    try:
        model.read_greeks(("Vega",))
    except RuntimeError:
        raised = True
    assert raised
    print("OK: read_greeks raises RuntimeError when a requested grad is None")

    raised = False
    try:
        model.read_greeks(("Theta",))
    except KeyError:
        raised = True
    assert raised
    print("OK: read_greeks raises KeyError on an unknown name")

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    test_engine_greeks()
