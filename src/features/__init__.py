"""Definitions for feature groups beyond the original eight columns.

The original eight (QScore and raw analyst-rating counts) are copied from the
panel. Everything else is created here and switched on from ``config.config``:

- ``analyst``: Coverage, BuyRatio, SellRatio, RatingScore
- ``macro``: Yahoo daily levels (cache only; ``script/fetch_macro.py``)
- ``ibes`` / ``compustat``: WRDS features (cache only; ``script/fetch_wrds_*.py``)
- ``engineered``: multi-window mom/vol, rel_vol/amihud/range/gap, mom_exmkt_*, log_mcap, par_mkt

Comment out a name in the matching config group to disable it. The event table
that consumes these columns lives in ``src.events``.
"""

from __future__ import annotations

from src.features.analyst import attach_analyst_features
from src.features.compustat import attach_compustat
from src.features.engineered import attach_engineered_features
from src.features.ibes import attach_ibes
from src.features.macro import attach_macro


def enabled_features() -> list[str]:
    """Ordered catalog: ORIGINAL → ANALYST → MACRO → IBES → COMPUSTAT → ENGINEERED."""
    from config.config import (
        ANALYST_FEATURES,
        COMPUSTAT_FEATURES,
        ENGINEERED_FEATURES,
        IBES_FEATURES,
        MACRO_FEATURES,
        ORIGINAL_FEATURES,
    )

    ordered: list[str] = []
    for name in (
        *ORIGINAL_FEATURES,
        *ANALYST_FEATURES,
        *MACRO_FEATURES,
        *IBES_FEATURES,
        *COMPUSTAT_FEATURES,
        *ENGINEERED_FEATURES,
    ):
        if name not in ordered:
            ordered.append(name)
    return ordered


def registered_features() -> set[str]:
    """Enabled feature names (same membership as ``enabled_features()``)."""
    return set(enabled_features())


__all__ = [
    "attach_analyst_features",
    "attach_compustat",
    "attach_engineered_features",
    "attach_ibes",
    "attach_macro",
    "enabled_features",
    "registered_features",
]
