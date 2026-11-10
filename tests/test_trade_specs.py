"""Trade specs reproduce the literal pricing arguments built from val_config (structural).

No model or pricing needed: build each spec from val_config, run its translator, and assert
the result equals the literal pricing arguments built from config. Also locks the spec-vs-run-opt boundary (no run-opt keys leak into to_kwargs).
"""

import numpy as np

from jolto import val_config
from jolto.api.trade_specs import BermudanSwaptionSpec, CmsCapletSpec, CmsSpreadSpec

_RUN_OPT_KEYS = {"freeze_drift", "n_paths", "dt", "seed", "use_checkpoint"}


def test_cms_caplet_to_kwargs():
    spec = CmsCapletSpec.from_val_config(val_config)
    # Literal cms_caplet_floorlet_price kwargs (economics only).
    expected = {
        "fixing_date": val_config.CMS_FIXING,
        "swap_end": val_config.CMS_SWAP_END,
        "strike": val_config.CMS_STRIKE,
        "year_fraction": val_config.CMS_YEAR_FRACTION,
        "is_cap": val_config.CMS_IS_CAP,
    }
    assert spec.to_kwargs() == expected
    assert _RUN_OPT_KEYS.isdisjoint(spec.to_kwargs()), "run-opt key leaked into caplet to_kwargs"
    print("OK: CmsCapletSpec.to_kwargs() == CMS caplet economics kwargs (no run-opts)")


def test_cms_spread_to_kwargs():
    spec = CmsSpreadSpec.from_val_config(val_config)
    expected = {
        "fixing_date": val_config.CMS_SPREAD_FIXING,
        "long_tenor": val_config.CMS_SPREAD_LONG_TENOR,
        "short_tenor": val_config.CMS_SPREAD_SHORT_TENOR,
        "strike": val_config.CMS_SPREAD_STRIKE,
        "year_fraction": val_config.CMS_SPREAD_YEAR_FRACTION,
        "is_cap": val_config.CMS_SPREAD_IS_CAP,
    }
    assert spec.to_kwargs() == expected
    assert _RUN_OPT_KEYS.isdisjoint(spec.to_kwargs()), "run-opt key leaked into spread to_kwargs"
    print("OK: CmsSpreadSpec.to_kwargs() == CMS spread economics kwargs (no run-opts)")


def test_bermudan_to_trade():
    spec = BermudanSwaptionSpec.from_val_config(val_config)
    # Literal Bermudan trade dict.
    ex_dates = np.arange(
        val_config.PRICE_FIRST_EX, val_config.PRICE_LAST_EX + 1e-6, val_config.GRID_STEP_SIZE
    )
    expected = {"Strike": val_config.PRICE_STRIKE, "Ex_Dates": ex_dates.tolist()}
    got = spec.to_trade(val_config.GRID_STEP_SIZE)
    assert got == expected
    assert got["Strike"] == val_config.PRICE_STRIKE
    assert got["Ex_Dates"] == ex_dates.tolist()
    print(f"OK: BermudanSwaptionSpec.to_trade() reproduces script trade ({len(ex_dates)} ex dates)")


if __name__ == "__main__":
    test_cms_caplet_to_kwargs()
    test_cms_spread_to_kwargs()
    test_bermudan_to_trade()
    print("\nALL TRADE-SPEC CHECKS PASSED")
