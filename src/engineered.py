"""Panel features built from existing columns (price / market cap).

Formulas (per ``TickerId``, sorted by ``Date``; no lookahead):

    mom_21d   = LastPrice.pct_change(21)
    vol_21d   = rolling std of 1d returns over 21 days (ddof=1, min_periods=21)
    log_mcap  = log(MarketCap) if MarketCap > 0 else NaN

Empty ``ENGINEERED_FEATURES`` disables attach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.default import ENGINEERED_FEATURES

PRICE_COLUMN = "LastPrice"
MCAP_COLUMN = "MarketCap"
DEFAULT_WINDOW = 21

FEAT_MOM_21D = "mom_21d"
FEAT_VOL_21D = "vol_21d"
FEAT_LOG_MCAP = "log_mcap"

ALL_ENGINEERED = (FEAT_MOM_21D, FEAT_VOL_21D, FEAT_LOG_MCAP)


def _mom_21d(ordered: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """W-day price momentum on a TickerId/Date-sorted frame."""
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    return price.groupby(ordered["TickerId"], sort=False).pct_change(window)


def _vol_21d(ordered: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """W-day rolling std of daily returns on a TickerId/Date-sorted frame."""
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    ret_1d = price.groupby(ordered["TickerId"], sort=False).pct_change()
    return (
        ret_1d.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )


def _log_mcap(ordered: pd.DataFrame) -> pd.Series:
    """Log market cap; non-positive / missing -> NaN."""
    mcap = pd.to_numeric(ordered[MCAP_COLUMN], errors="coerce").to_numpy(dtype=float)
    values = np.where(np.isfinite(mcap) & (mcap > 0), np.log(mcap), np.nan)
    return pd.Series(values, index=ordered.index, name=FEAT_LOG_MCAP)


_BUILDERS = {
    FEAT_MOM_21D: lambda ordered, window: _mom_21d(ordered, window),
    FEAT_VOL_21D: lambda ordered, window: _vol_21d(ordered, window),
    FEAT_LOG_MCAP: lambda ordered, window: _log_mcap(ordered),
}


def compute_engineered_features(
    frame: pd.DataFrame,
    *,
    engineered_features: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Return a copy with requested engineered columns attached."""
    wanted = list(ENGINEERED_FEATURES if engineered_features is None else engineered_features)
    if not wanted:
        return frame.copy()

    unknown = [name for name in wanted if name not in ALL_ENGINEERED]
    if unknown:
        raise ValueError(
            f"Unknown ENGINEERED_FEATURES {unknown}; allowed: {list(ALL_ENGINEERED)}."
        )
    if window < 2:
        raise ValueError(f"engineered window must be >= 2, got {window}.")

    need_price = FEAT_MOM_21D in wanted or FEAT_VOL_21D in wanted
    if need_price and PRICE_COLUMN not in frame.columns:
        raise KeyError(
            f"Panel missing {PRICE_COLUMN!r} required for mom/vol engineered features."
        )
    if FEAT_LOG_MCAP in wanted and MCAP_COLUMN not in frame.columns:
        raise KeyError(f"Panel missing {MCAP_COLUMN!r} required for log_mcap.")
    if "TickerId" not in frame.columns or "Date" not in frame.columns:
        raise KeyError("Panel missing TickerId / Date required for engineered features.")

    out = frame.copy()
    drop_cols = [c for c in wanted if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    ordered = out.sort_values(["TickerId", "Date"], kind="mergesort")
    for name in wanted:
        series = _BUILDERS[name](ordered, window)
        out.loc[ordered.index, name] = series.to_numpy(dtype=float)

    return out


def attach_engineered_features(
    frame: pd.DataFrame,
    *,
    engineered_features: list[str] | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Attach engineered features onto a daily / aligned panel."""
    return compute_engineered_features(
        frame,
        engineered_features=engineered_features,
        window=window,
    )
