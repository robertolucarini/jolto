# jolto

<p align="center">
  <img src=".github/jolto.png" alt="jolto logo" width="220">
</p>

A Python library for interest-rate derivatives under rough volatility. It calibrates, simulates and
prices the Rough SABR Forward Market Model of Adachi et al. (2025), in PyTorch, on a standard CPU.

- **Calibration in two stages.** First the rough SABR marginals of each expiry (α, ρ, a global ν, and
  H on a grid), then the correlation matrix Σ of the forward rates, regularised towards a historical
  target. Both stages run an AMMO trust-region loop, a closed-form surrogate corrected against
  conditional Monte Carlo.
- **Valuation** of Bermudan swaptions (least-squares Monte Carlo), CMS caplets and floorlets, and CMS
  spread options, with Greeks by adjoint algorithmic differentiation.
- **Simulation** of the forward and discount curves of the calibrated model.

## Install

Python 3.11 or 3.12.

```bash
git clone https://github.com/robertolucarini/jolto.git
cd jolto
pip install -e ".[dev]"
```

On Linux without a GPU, install the CPU build of PyTorch first:
`pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Quick start

No market data ships with the repository. Put your discount curve and swaption volatility surface in
`data/` (layout in `data/README.md`), then run from the repository folder:

```bash
jolto calibrate-sabr                # Stage 1  -> results/stage1_parameters.csv
jolto calibrate-correlation         # Stage 2  -> results/stage2_correlation_H_<H>.csv
jolto price --product bermudan      # or cms-caplet, cms-spread  -> results/*_greeks.csv
jolto simulate-curves               # forward and discount curves -> results/simulated_curves_H_<H>.npz
```

`jolto <subcommand> --help` lists the options. Settings live in `src/jolto/config.py` (calibration)
and `src/jolto/val_config.py` (trades and pricing runs).

## Conventions

Rates, normal (Bachelier) volatilities and strike offsets are decimal (`0.005` = 50 bp). Times are year
fractions on an annual forward grid. Prices are per unit notional and reported in bps of notional; the
Greek ladders give the price change for a 1 bp move in each initial forward rate (PV01) or in its
volatility level (Vega). The model is simulated under the terminal measure with β = 0 (Bachelier), the
only specification the library supports.

## Tests

`pytest` runs the smoke tests, which need no market data. The other scripts in `tests/` are the Monte
Carlo validation studies (bias against pseudo-random oracles, discretisation bias, out-of-sample
repricing). They read the calibration outputs, take minutes to hours, and run as
`python tests/<name>.py`.

## Citation and license

Cite the software with `CITATION.cff`. MIT license, see `LICENSE`.

## Use of AI tools

I used Anthropic's Claude models to help write, refactor and review the code, to build the test suite
and to draft the documentation. The literature research, the model and its mathematics, the choice of
experiments and the interpretation of the results are mine. I reviewed every AI suggestion before it
entered the repository, and I take full responsibility for the code and the documentation.
