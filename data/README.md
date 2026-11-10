# Market data inputs

The `data/` directory holds the market-data files the calibration pipeline reads: an OIS
**discount curve** and a **swaption volatility surface** of one date, and the dated history of discount
curves behind the Stage-2 correlation target. The first two are referenced by name from
`src/jolto/config.py` (`DISCOUNT_CSV`, `VOL_SURFACE_CSV`) and resolved relative to `DATA_DIR`; the history
directory is `STAGE2_HIST_CURVE_DIR`.

No market data is distributed with this repository. Supply your own in the format documented here.
The layout is dated, one file per date:

```
data/history/curves/disc_YYYYMMDD.csv   discount curve of each date (the history, and DISCOUNT_CSV)
data/history/vols/vol_YYYYMMDD.csv      swaption surface of each date (VOL_SURFACE_CSV)
```

The history needs one year of curves ending on the surface's date. The prefixes are
`config.CURVE_PREFIX` and `config.VOL_PREFIX`.

Set `VOL_SURFACE_CSV`, `DISCOUNT_CSV` and `STAGE2_HIST_END` to your date, in `src/jolto/config.py` or in
an untracked `src/jolto/config_local.py`, which `config.py` reads last. With a historical Stage-2 target
(`STAGE2_DECAY_FORM = "hist_*"`, the default), the surface filename must end in `_YYYYMMDD.csv` and that
date must equal `STAGE2_HIST_END`; `jolto calibrate-correlation` stops otherwise, since a later window end would let
the prior see curves from after the quotes.

Both files are semicolon-delimited (`;`). The loaders auto-detect the delimiter and identify columns
by keyword, so comma-delimited files and minor header variations are tolerated; the canonical layouts
below are recommended.

## Discount curve

**File:** `DISCOUNT_CSV` (e.g. `history/curves/disc_YYYYMMDD.csv`). Consumed at Stage 2 and in
pricing; the dated files in `STAGE2_HIST_CURVE_DIR` share this layout.

A single OIS discount factor per market tenor. Loaded into a monotone (PCHIP) interpolator, from which
the forward curve and zero-coupon bond prices are bootstrapped.

| Column   | Meaning                                                                                                                                            |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `TICKER` | Market tenor as a string (see **Conventions** below).                                                                                              |
| `VALUE`  | Discount factor `P(0, T)` at that tenor — a number in `(0, 1]`, strictly decreasing with maturity. This is a **discount factor, not a zero rate**. |

Example:

```
TICKER;VALUE
1D;0.99990
1M;0.99800
1Y;0.97000
10Y;0.75000
30Y;0.40000
```

The tenor set may be any reasonable spanning grid; the interpolator handles irregular spacing.

## Swaption volatility surface

**File:** `VOL_SURFACE_CSV` (e.g. `history/vols/vol_YYYYMMDD.csv`). Stage 1 reads the per-expiry smile for
the target underlying tenor; Stage 2 reads the ATM slice across tenors.

One row per (expiry, underlying tenor, strike) cell, in long/tidy form.

| Column            | Meaning                                                                                                                                                                               |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Expiry`          | Option expiry as a tenor string.                                                                                                                                                      |
| `UnderlyingTenor` | Tenor of the underlying swap, as a tenor string.                                                                                                                                      |
| `Strike`          | `ATM`, or an offset from the ATM forward such as `-50bp` or `100bp`. Parsed as `K − F` in absolute rate terms (`-50bp → −0.0050`, `ATM → 0`); negative offsets lie below the forward. |
| `Volatility`      | **Normal (Bachelier) implied volatility** in absolute decimal terms — e.g. `0.0050` is 50 bps (0.50%). Consistent with `BETA_SABR = 0`.                                        |

Example:

```
Expiry;UnderlyingTenor;Strike;Volatility
3M;1Y;ATM;0.005000
3M;1Y;-50bp;0.005500
3M;1Y;50bp;0.005400
1Y;5Y;ATM;0.006000
10Y;10Y;ATM;0.007000
```

The surface need not be a complete rectangular grid — only the rows present are used, and any cell that
fails to parse or is blank is dropped. A given expiry may carry a different tenor set from its
neighbours. Expiries shorter than `MIN_EXPIRY_YEARS` (default 0.25) are dropped on load.

## Conventions

**Tenor strings.** An integer followed by a unit — `D` (days), `W` (weeks), `M` (months), `Y` (years).
Examples: `3D`, `2W`, `18M`, `1Y`, `30Y`. Days convert as `n / 365.25` years, weeks as `n / 52`,
months as `n / 12`.

**Strike offsets.** `ATM` maps to `0`; `±NNNbp` maps to `±NNN × 10⁻⁴` in absolute rate terms, interpreted
as `K − F`. Any other strike label raises an error.

**Units.** Discount factors are dimensionless in `(0, 1]`. Volatilities are normal/Bachelier in absolute
decimals — not percent, not basis points; multiply by `10⁴` to read a vol in basis points.

**Delimiter and headers.** Canonical files use `;`. The loaders auto-detect the separator and match
columns by keyword, so exact header capitalisation and delimiter are not critical, but matching the
canonical layout avoids ambiguity.
