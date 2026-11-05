"""Definitions for feature groups beyond the original eight columns.

The original eight (QScore and raw analyst-rating counts) are copied from the
panel. Everything else is created here and switched on from ``config.default``:

- ``analyst``: Coverage, BuyRatio, SellRatio, RatingScore
- ``macro``: Yahoo daily levels such as vix, tnx, dxy
- ``engineered``: mom_21d, vol_21d, log_mcap, par_mkt

The event table that consumes these columns lives in ``src.events``.
"""

from src.features.analyst import attach_analyst_features
from src.features.engineered import attach_engineered_features
from src.features.macro import attach_macro

__all__ = [
    "attach_analyst_features",
    "attach_engineered_features",
    "attach_macro",
]
