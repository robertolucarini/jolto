"""jolto price: valuation entry point (Bermudan / CMS caplet / CMS spread).

Owns argparse and the CSV writes; the library prints only progress logs and harness output.
Builds inputs, spec and run options from val_config, calls the get_price façade, then prints
the banner and Greek ladder and writes the per-product Greeks CSV.

    jolto price --product cms-caplet              # price + Greeks CSV
    jolto price --product cms-caplet --validate   # self-consistency checks
    jolto price --product cms-caplet --benchmark  # MC vs closed-form
    jolto price --product bermudan --price-only   # price only, no Greeks/CSV
"""

import argparse
import os

import pandas as pd
import torch

from jolto import val_config
from jolto.api import get_price
from jolto.api.assembly import load_valuation_inputs
from jolto.api.trade_specs import BermudanSwaptionSpec, CmsCapletSpec, CmsSpreadSpec
from jolto.core.utils import check_config, print_greek_ladder, print_header, require_files

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CONTRACT: required val_config entries per product (validated up front)
# ─────────────────────────────────────────────────────────────────────────────
_SHARED = [
    "DATA_DIR",
    "RESULTS_DIR",
    "DISCOUNT_CSV",
    "GRID_STEP_SIZE",
    "MAX_MATURITY",
    "BETA_SABR",
    "SHIFT_SABR",
    "STAGE1_PARAMS_CSV",
    "STAGE2_CORR_CSV_TPL",
    "VAL_H_TARGET",
    "VAL_GREEKS",
]
_REQUIRED = {
    "bermudan": _SHARED
    + [
        "PRICE_STRIKE",
        "PRICE_FIRST_EX",
        "PRICE_LAST_EX",
        "PRICE_PATHS_MAIN",
        "PRICE_PATHS_PILOT",
        "PRICE_DT",
        "PRICE_SEED",
        "PRICE_FREEZE_DRIFT",
        "PRICE_USE_CONTROL_VARIATE",
    ],
    "cms-caplet": _SHARED
    + [
        "CMS_FIXING",
        "CMS_SWAP_END",
        "CMS_STRIKE",
        "CMS_YEAR_FRACTION",
        "CMS_IS_CAP",
        "CMS_PATHS_MAIN",
        "CMS_DT",
        "CMS_SEED",
        "CMS_FREEZE_DRIFT",
    ],
    "cms-spread": _SHARED
    + [
        "CMS_SPREAD_FIXING",
        "CMS_SPREAD_LONG_TENOR",
        "CMS_SPREAD_SHORT_TENOR",
        "CMS_SPREAD_STRIKE",
        "CMS_SPREAD_YEAR_FRACTION",
        "CMS_SPREAD_IS_CAP",
        "CMS_PATHS_MAIN",
        "CMS_DT",
        "CMS_SEED",
        "CMS_FREEZE_DRIFT",
    ],
}

# Per-product dispatch: spec / run-opts builders, pricing fn, siblings, CSV name, labels.
_DISPATCH = {
    "bermudan": {
        "spec": BermudanSwaptionSpec,
        "run_opts": get_price.BermudanRunOpts,
        "price_fn": get_price.bermudan_swaptions,
        "validate": None,
        "benchmark": None,
        "csv": "pricing_greeks.csv",
        "price_label": "Bermudan price :",
        "banner": "PRICING — BERMUDAN SWAPTION + AAD GREEKS",
    },
    "cms-caplet": {
        "spec": CmsCapletSpec,
        "run_opts": get_price.CmsRunOpts,
        "price_fn": get_price.cms_caplet,
        "validate": get_price.validate_cms,
        "benchmark": get_price.benchmark_cms,
        "csv": "cms_greeks.csv",
        "price_label": "CMS price      :",
        "banner": "PRICING — CMS CAPLET/FLOORLET + AAD GREEKS",
    },
    "cms-spread": {
        "spec": CmsSpreadSpec,
        "run_opts": get_price.CmsRunOpts,
        "price_fn": get_price.cms_spread_option,
        "validate": get_price.validate_cms_spread,
        "benchmark": None,
        "csv": "cms_spread_greeks.csv",
        "price_label": "CMS spread price:",
        "banner": "PRICING — CMS SPREAD OPTION + AAD GREEKS",
    },
}

_VALIDATE_BANNER = {
    "cms-caplet": "CMS PRICER — SELF-VALIDATION (benchmark-free)",
    "cms-spread": "CMS SPREAD PRICER — SELF-VALIDATION (benchmark-free + Σ exhibit)",
}
_BENCHMARK_BANNER = {"cms-caplet": "CMS PRICER — MC vs CLOSED-FORM BENCHMARK"}


def _parse_args(argv):
    p = argparse.ArgumentParser(
        prog="jolto price",
        description="Price a product with the calibrated model and write its AAD Greeks. Trade "
        "and run settings come from jolto/val_config.py; Stage 1 and Stage 2 outputs must exist.",
    )
    p.add_argument(
        "--product",
        required=True,
        choices=["bermudan", "cms-caplet", "cms-spread"],
        help="the product to price",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate",
        action="store_true",
        help="run the self-consistency checks (cms-caplet, cms-spread)",
    )
    mode.add_argument(
        "--benchmark",
        action="store_true",
        help="compare Monte Carlo with the closed-form replication (cms-caplet)",
    )
    p.add_argument("--price-only", action="store_true", help="price only, no Greeks or CSV")
    args = p.parse_args(argv)

    # Support matrix: validate (caplet+spread), benchmark (caplet only); bermudan neither.
    if args.validate and _DISPATCH[args.product]["validate"] is None:
        p.error(f"--validate is not supported for product '{args.product}'")
    if args.benchmark and _DISPATCH[args.product]["benchmark"] is None:
        p.error(f"--benchmark is not supported for product '{args.product}'")
    if args.price_only and (args.validate or args.benchmark):
        p.error("--price-only cannot be combined with --validate/--benchmark")
    return args


def _run_diagnostic(product, mode, inputs, spec, run_opts, device):
    d = _DISPATCH[product]
    if mode == "validate":
        print_header(_VALIDATE_BANNER[product])
        d["validate"](spec, inputs, run_opts=run_opts, device=device)
    else:
        print_header(_BENCHMARK_BANNER[product])
        d["benchmark"](spec, inputs, run_opts=run_opts, device=device)


def _trade_banner(product, spec, inputs, run_opts):
    if product == "bermudan":
        ex = spec.to_trade(inputs.grid_step)["Ex_Dates"]
        print(
            f"  Trade          : Bermudan payer, K={spec.strike * 1e4:.0f}bps, "
            f"{len(ex)} exercise dates [{ex[0]:.0f}Y–{ex[-1]:.0f}Y]"
        )
        print(f"  Paths          : main={run_opts.paths_main}, pilot={run_opts.paths_pilot}")
    elif product == "cms-caplet":
        kind = "caplet" if spec.is_cap else "floorlet"
        print(
            f"  Trade          : CMS {kind}, T_a={spec.fixing_date:.0f}Y, "
            f"T_b={spec.swap_end:.0f}Y, K={spec.strike * 1e4:.0f}bps"
        )
        print(f"  Paths          : main={run_opts.n_paths} (single estimator)")
    else:
        kind = "call" if spec.is_cap else "put"
        print(
            f"  Trade          : CMS spread {kind}, T_a={spec.fixing_date:.0f}Y, "
            f"S¹={spec.long_tenor:.0f}Y − S²={spec.short_tenor:.0f}Y, "
            f"K={spec.strike * 1e4:.0f}bps"
        )
        print(f"  Paths          : main={run_opts.n_paths} (single estimator)")
    print(f"  Time step      : dt=1/{round(1 / run_opts.dt)}")


def main(argv=None):
    args = _parse_args(argv)
    product = args.product
    check_config(val_config, _REQUIRED[product], "jolto price")
    d = _DISPATCH[product]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = load_valuation_inputs(val_config)
    require_files([os.path.join(inputs.data_dir, inputs.discount_name)])
    require_files(
        [
            os.path.join(inputs.results_dir, inputs.stage1_name),
            os.path.join(inputs.results_dir, inputs.sigma_tpl.format(h=inputs.H)),
        ],
        hint="Run jolto calibrate-sabr and then jolto calibrate-correlation first.",
    )
    spec = d["spec"].from_val_config(val_config)
    run_opts = d["run_opts"].from_val_config(val_config)

    if args.validate or args.benchmark:
        _run_diagnostic(
            product, "validate" if args.validate else "benchmark", inputs, spec, run_opts, device
        )
        return

    with_greeks = not args.price_only
    sigma_path = os.path.join(inputs.results_dir, inputs.sigma_tpl.format(h=inputs.H))

    print_header(d["banner"])
    print(f"  Device         : {device.upper()}")
    print(f"  Discount curve : {os.path.join(inputs.data_dir, inputs.discount_name)}")
    print(f"  Stage 1 params : {os.path.join(inputs.results_dir, inputs.stage1_name)}")
    print(f"  H target       : {inputs.H:.4f}")
    print(f"  Correlation Σ  : {sigma_path}")
    _trade_banner(product, spec, inputs, run_opts)

    res = d["price_fn"](spec, inputs, with_greeks=with_greeks, run_opts=run_opts, device=device)

    print_header("PRICING COMPLETE")
    print(f"  {d['price_label']} {res.price_bps:.4f} bps")
    print(f"  Model          : N={res.meta['N']}, H={res.meta['H']:.4f}")
    print(f"  Pricing time   : {res.timings['price']:.2f}s")

    if not with_greeks:
        return

    pv01, vega = res.greeks["PV01"], res.greeks["Vega"]
    print(f"  AAD Greeks time: {res.timings['aad']:.3f}s")
    print(f"  PV01 (sum)     : {pv01.sum() * 1e4:.4f} bps")
    print(f"  Vega (sum)     : {vega.sum() * 1e4:.4f} bps")

    print_greek_ladder(pv01 * 1e4, vega * 1e4, res.tenors)

    os.makedirs(inputs.results_dir, exist_ok=True)
    out_csv = os.path.join(inputs.results_dir, d["csv"])
    pd.DataFrame({"Tenor": res.tenors, "PV01_bps": pv01 * 1e4, "Vega_bps": vega * 1e4}).to_csv(
        out_csv, index=False
    )
    print(f"  Written        : {out_csv}")


if __name__ == "__main__":
    main()
