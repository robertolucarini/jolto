"""Valuation façade: one library function per product, plus the validate/benchmark harnesses.

Wraps assembly, trade specs and the engines. No CSV writes or argparse (both live in
jolto price); console output comes only from the engines' progress logs,
`BermudanRunOpts.verbose`, and the validation harnesses.

Config-free, like assembly and trade_specs: this module never imports val_config. The caller
injects run options as `<Product>RunOpts.from_val_config(val_config)`; `run_opts=None` raises.

AAD invariant: the parameter tensors from build_calibrated_model are the autograd leaves;
nothing detaches or rebuilds them between assembly and backward(), and read_greeks reads those
leaves' .grad. Reproducible from `seed` end to end, with no global-RNG pin here.
"""

import time
from dataclasses import dataclass

import numpy as np
import torch

from jolto.api.assembly import build_calibrated_model
from jolto.valuation.cms import (
    cms_caplet_floorlet_price,
    cms_spread_option_price,
    compare_cms_mc_vs_benchmark,
    self_validate_cms,
    self_validate_cms_spread,
)
from jolto.valuation.bermudan_swaption import oos_bermudan_pricer

_DEFAULT_GREEKS = ("PV01", "Vega")


@dataclass(frozen=True)
class PriceResult:
    """Output of a pricing call. All values are per unit notional.

    price_bps : price × 1e4, in bps of notional.
    greeks    : {"PV01": ∂P/∂F_j(0) · 1e-4, "Vega": ∂P/∂α_j · 1e-4} per forward j, the price change
                (decimal) for a 1 bp move in the initial forward rate F_j(0) or in its normal-vol
                level α_j; ×1e4 gives bps of notional, as jolto price writes them. None without
                Greeks.
    tenors    : start T_{j-1} of each forward j, in years, aligned with the Greek ladders.
    timings   : seconds, keys "price" and "aad".
    meta      : {"H", "N", "device"} of the model that priced.
    """

    price_bps: float
    greeks: dict[str, np.ndarray] | None
    tenors: np.ndarray
    timings: dict[str, float]
    meta: dict


@dataclass(frozen=True)
class BermudanRunOpts:
    paths_main: int
    paths_pilot: int
    dt: float
    seed: int
    freeze_drift: bool = True
    verbose: bool = True
    use_checkpoint: bool = False
    use_control_variate: bool = False
    greeks: tuple = _DEFAULT_GREEKS

    @classmethod
    def from_val_config(cls, val_config) -> "BermudanRunOpts":
        return cls(
            paths_main=val_config.PRICE_PATHS_MAIN,
            paths_pilot=val_config.PRICE_PATHS_PILOT,
            dt=val_config.PRICE_DT,
            seed=val_config.PRICE_SEED,
            freeze_drift=val_config.PRICE_FREEZE_DRIFT,
            use_control_variate=val_config.PRICE_USE_CONTROL_VARIATE,
            greeks=tuple(val_config.VAL_GREEKS),
        )


@dataclass(frozen=True)
class CmsRunOpts:
    n_paths: int
    dt: float
    seed: int
    freeze_drift: bool = True
    use_checkpoint: bool = False
    greeks: tuple = _DEFAULT_GREEKS

    @classmethod
    def from_val_config(cls, val_config) -> "CmsRunOpts":
        return cls(
            n_paths=val_config.CMS_PATHS_MAIN,
            dt=val_config.CMS_DT,
            seed=val_config.CMS_SEED,
            freeze_drift=val_config.CMS_FREEZE_DRIFT,
            greeks=tuple(val_config.VAL_GREEKS),
        )


def _resolve_device(device):
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def _require_run_opts(run_opts, product, cls_name):
    if run_opts is None:
        raise ValueError(
            f"{product}: run_opts is required — pass {cls_name}.from_val_config(val_config) "
            "(this module is config-free and will not reach into config)."
        )
    return run_opts


def _finish(model, grid_T, inputs, device, price, t_price, with_greeks, greek_names):
    """Shared tail: optional backward + read_greeks, then assemble PriceResult."""
    greeks = None
    t_aad = 0.0
    if with_greeks:
        t0 = time.time()
        price.backward()
        t_aad = time.time() - t0
        greeks = model.read_greeks(greek_names)
    return PriceResult(
        price_bps=price.item() * 1e4,
        greeks=greeks,
        tenors=np.asarray(grid_T[:-1]),  # grid_T is the numpy bootstrap grid
        timings={"price": t_price, "aad": t_aad},
        meta={"H": inputs.H, "N": model.N, "device": device},
    )


def _enable_greeks(model):
    # requires_grad on the live engine leaves, no detach/rebuild (AAD invariant)
    for tensor in model.differentiable_parameters().values():
        tensor.requires_grad_(True)


def bermudan_swaptions(trade_spec, inputs, *, with_greeks=True, run_opts=None, device=None):
    """Payer Bermudan swaption by least-squares Monte Carlo, exercise policy fitted on a pilot run.

    trade_spec: BermudanSwaptionSpec; inputs: assembly.ValuationInputs; run_opts:
    BermudanRunOpts.from_val_config(val_config). Returns a PriceResult.
    """
    ro = _require_run_opts(run_opts, "bermudan_swaptions", "BermudanRunOpts")
    device = _resolve_device(device)
    model, grid_T, _ = build_calibrated_model(inputs, device)
    if with_greeks:
        _enable_greeks(model)

    trade = trade_spec.to_trade(inputs.grid_step)
    max_T = float(grid_T[-1])
    n_steps = int(np.ceil(max_T / ro.dt))
    time_grid = torch.linspace(0.0, max_T, n_steps + 1, device=device, dtype=torch.float64)

    t0 = time.time()
    price = oos_bermudan_pricer(
        model,
        trade,
        n_paths_main=ro.paths_main,
        n_paths_pilot=ro.paths_pilot,
        time_grid=time_grid,
        freeze_drift=ro.freeze_drift,
        seed=ro.seed,
        verbose=ro.verbose,
        use_checkpoint=ro.use_checkpoint,
        use_control_variate=ro.use_control_variate,
    )
    t_price = time.time() - t0
    return _finish(model, grid_T, inputs, device, price, t_price, with_greeks, ro.greeks)


def cms_caplet(trade_spec, inputs, *, with_greeks=True, run_opts=None, device=None):
    """CMS caplet or floorlet settling at its fixing date, under the terminal measure.

    trade_spec: CmsCapletSpec; inputs: assembly.ValuationInputs; run_opts:
    CmsRunOpts.from_val_config(val_config). Returns a PriceResult.
    """
    ro = _require_run_opts(run_opts, "cms_caplet", "CmsRunOpts")
    device = _resolve_device(device)
    model, grid_T, _ = build_calibrated_model(inputs, device)
    if with_greeks:
        _enable_greeks(model)

    t0 = time.time()
    price = cms_caplet_floorlet_price(
        model,
        **trade_spec.to_kwargs(),
        n_paths=ro.n_paths,
        dt=ro.dt,
        seed=ro.seed,
        freeze_drift=ro.freeze_drift,
        use_checkpoint=ro.use_checkpoint,
    )
    t_price = time.time() - t0
    return _finish(model, grid_T, inputs, device, price, t_price, with_greeks, ro.greeks)


def cms_spread_option(trade_spec, inputs, *, with_greeks=True, run_opts=None, device=None):
    """Call or put on the spread of two co-fixing CMS rates, under the terminal measure.

    trade_spec: CmsSpreadSpec; inputs: assembly.ValuationInputs; run_opts:
    CmsRunOpts.from_val_config(val_config). Returns a PriceResult.
    """
    ro = _require_run_opts(run_opts, "cms_spread_option", "CmsRunOpts")
    device = _resolve_device(device)
    model, grid_T, _ = build_calibrated_model(inputs, device)
    if with_greeks:
        _enable_greeks(model)

    t0 = time.time()
    price = cms_spread_option_price(
        model,
        **trade_spec.to_kwargs(),
        n_paths=ro.n_paths,
        dt=ro.dt,
        seed=ro.seed,
        freeze_drift=ro.freeze_drift,
        use_checkpoint=ro.use_checkpoint,
    )
    t_price = time.time() - t0
    return _finish(model, grid_T, inputs, device, price, t_price, with_greeks, ro.greeks)


# The diagnostic harnesses return their own dict, not PriceResult.
# requires_grad: on for validate and spread-validate, off for benchmark.


def validate_cms(trade_spec, inputs, *, run_opts=None, device=None):
    """Self-consistency checks of the CMS caplet pricer; prints them and returns their dict."""
    ro = _require_run_opts(run_opts, "validate_cms", "CmsRunOpts")
    device = _resolve_device(device)
    model, _, _ = build_calibrated_model(inputs, device)
    _enable_greeks(model)
    return self_validate_cms(
        model,
        fixing_date=trade_spec.fixing_date,
        swap_end=trade_spec.swap_end,
        year_fraction=trade_spec.year_fraction,
        seed=ro.seed,
    )


def benchmark_cms(trade_spec, inputs, *, run_opts=None, device=None):
    """CMS caplet Monte Carlo against static replication; prints it and returns its dict."""
    ro = _require_run_opts(run_opts, "benchmark_cms", "CmsRunOpts")
    device = _resolve_device(device)
    model, _, Sigma_t = build_calibrated_model(inputs, device)
    # no requires_grad: the harness runs its own backward passes; pass the Σ baked into the
    # model loadings
    return compare_cms_mc_vs_benchmark(
        model,
        Sigma_t,
        fixing_date=trade_spec.fixing_date,
        swap_end=trade_spec.swap_end,
        year_fraction=trade_spec.year_fraction,
        seed=ro.seed,
    )


def validate_cms_spread(trade_spec, inputs, *, run_opts=None, device=None):
    """Self-consistency checks and Σ exhibit of the CMS spread pricer; returns their dict."""
    ro = _require_run_opts(run_opts, "validate_cms_spread", "CmsRunOpts")
    device = _resolve_device(device)
    model, _, Sigma_t = build_calibrated_model(inputs, device)
    _enable_greeks(model)
    return self_validate_cms_spread(
        model,
        Sigma_t,
        fixing_date=trade_spec.fixing_date,
        long_tenor=trade_spec.long_tenor,
        short_tenor=trade_spec.short_tenor,
        year_fraction=trade_spec.year_fraction,
        seed=ro.seed,
    )
