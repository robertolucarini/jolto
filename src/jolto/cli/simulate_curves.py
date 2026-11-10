"""jolto simulate-curves: simulate the forward and discount curves of the calibrated model.

Rebuilds the model from the Stage 1 and Stage 2 outputs, as jolto price does, and simulates the
forward curve under the terminal measure without pricing anything. Settings are the SIM_* entries
of jolto/val_config.py.

    jolto simulate-curves

Output (in results/):
    simulated_curves_H_<H>.npz          dates, tenors, forwards (paths, dates, N) and bonds
                                        (paths, dates, N+1), decimal; NaN marks fixed forwards and
                                        past maturities
    simulated_curves_summary_H_<H>.csv  mean, sd and 5/50/95% quantiles per date, curve and tenor
"""

import argparse
import os

import numpy as np

from jolto import val_config
from jolto.api.assembly import load_valuation_inputs
from jolto.api.simulate import SimulationRunOpts, simulate_curves, summary_table
from jolto.core.utils import check_config, print_header, require_files

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG CONTRACT: required val_config entries (validated up front)
# ─────────────────────────────────────────────────────────────────────────────
REQUIRED = [
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
    "PRICE_FREEZE_DRIFT",
    "SIM_PATHS",
    "SIM_DATES",
    "SIM_DT",
    "SIM_SEED",
    "SIM_NPZ_TPL",
    "SIM_SUMMARY_CSV_TPL",
]


def main(argv=None):
    argparse.ArgumentParser(
        prog="jolto simulate-curves",
        description="Simulate the forward and discount curves of the calibrated model at the "
        "SIM_DATES of jolto/val_config.py, without pricing; Stage 1 and Stage 2 outputs must exist.",
    ).parse_args(argv)
    check_config(val_config, REQUIRED, "jolto simulate-curves")
    inputs = load_valuation_inputs(val_config)
    require_files([os.path.join(inputs.data_dir, inputs.discount_name)])
    require_files(
        [
            os.path.join(inputs.results_dir, inputs.stage1_name),
            os.path.join(inputs.results_dir, inputs.sigma_tpl.format(h=inputs.H)),
        ],
        hint="Run `jolto calibrate-sabr` and then `jolto calibrate-correlation` first.",
    )
    run_opts = SimulationRunOpts.from_val_config(val_config)

    print_header("CURVE SIMULATION")
    print(f"  H target       : {inputs.H:.4f}")
    print(
        f"  Paths          : {run_opts.n_paths}, dt=1/{round(1 / run_opts.dt)}, seed={run_opts.seed}"
    )

    res = simulate_curves(inputs, run_opts=run_opts)

    os.makedirs(inputs.results_dir, exist_ok=True)
    npz = os.path.join(inputs.results_dir, val_config.SIM_NPZ_TPL.format(h=inputs.H))
    csv = os.path.join(inputs.results_dir, val_config.SIM_SUMMARY_CSV_TPL.format(h=inputs.H))
    np.savez_compressed(
        npz, dates=res.dates, tenors=res.tenors, forwards=res.forwards, bonds=res.bonds
    )
    summary_table(res).to_csv(csv, index=False)

    print_header("SIMULATION COMPLETE")
    print(f"  Model          : N={res.meta['N']}, H={res.meta['H']:.4f}")
    print(f"  Dates          : {len(res.dates)} ({res.dates[0]:g}Y to {res.dates[-1]:g}Y)")
    print(f"  Simulation time: {res.meta['seconds']:.2f}s")
    print(f"  Paths          : {npz}")
    print(f"  Summary        : {csv}")


if __name__ == "__main__":
    main()
