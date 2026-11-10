"""Per-product trade specs: product economics only, no numerics.

Each spec holds the trade terms read from config and translates them to pricer arguments
(`to_trade` for the Bermudan, `to_kwargs` for the CMS products). Run options (n_paths, dt, seed,
freeze_drift) are not trade economics: the façade passes them as separate keyword arguments.

`from_val_config` mirrors assembly.load_valuation_inputs: the single place config-derived
values enter, keeping the specs themselves config-free.
"""

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class BermudanSwaptionSpec:
    strike: float
    first_ex: float
    last_ex: float

    @classmethod
    def from_val_config(cls, val_config) -> "BermudanSwaptionSpec":
        return cls(
            strike=val_config.PRICE_STRIKE,
            first_ex=val_config.PRICE_FIRST_EX,
            last_ex=val_config.PRICE_LAST_EX,
        )

    def to_trade(self, grid_step) -> dict:
        # grid_step is a market/grid value (passed in), not trade economics (not stored).
        ex_dates = np.arange(self.first_ex, self.last_ex + 1e-6, grid_step)
        return {"Strike": self.strike, "Ex_Dates": ex_dates.tolist()}


@dataclass(frozen=True)
class CmsCapletSpec:
    fixing_date: float
    swap_end: float
    strike: float
    year_fraction: float
    is_cap: bool

    @classmethod
    def from_val_config(cls, val_config) -> "CmsCapletSpec":
        return cls(
            fixing_date=val_config.CMS_FIXING,
            swap_end=val_config.CMS_SWAP_END,
            strike=val_config.CMS_STRIKE,
            year_fraction=val_config.CMS_YEAR_FRACTION,
            is_cap=val_config.CMS_IS_CAP,
        )

    def to_kwargs(self) -> dict:
        return asdict(self)  # fields, in declaration order


@dataclass(frozen=True)
class CmsSpreadSpec:
    fixing_date: float
    long_tenor: float
    short_tenor: float
    strike: float
    year_fraction: float
    is_cap: bool

    @classmethod
    def from_val_config(cls, val_config) -> "CmsSpreadSpec":
        return cls(
            fixing_date=val_config.CMS_SPREAD_FIXING,
            long_tenor=val_config.CMS_SPREAD_LONG_TENOR,
            short_tenor=val_config.CMS_SPREAD_SHORT_TENOR,
            strike=val_config.CMS_SPREAD_STRIKE,
            year_fraction=val_config.CMS_SPREAD_YEAR_FRACTION,
            is_cap=val_config.CMS_SPREAD_IS_CAP,
        )

    def to_kwargs(self) -> dict:
        return asdict(self)  # fields, in declaration order
