"""Panel features built from existing columns.

Formulas (no lookahead):

    mom_21d   = LastPrice.pct_change(21)                         per TickerId
    vol_21d   = rolling std of 1d returns over 21 days (ddof=1)  per TickerId
    log_mcap  = log(MarketCap) if MarketCap > 0 else NaN
    par_mkt   = stock return from close at anchor-5 to close at anchor-2,
                minus the benchmark return over that same window.
                Written on the DecisionDate row only. Same window as
                reversal-comparison.

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
FEAT_PAR_MKT = "par_mkt"

# Close at anchor-5 -> close at anchor-2. Ends on DecisionDate (anchor-2).
PAR_START_OFFSET = -4
PAR_END_OFFSET = -2

ALL_ENGINEERED = (FEAT_MOM_21D, FEAT_VOL_21D, FEAT_LOG_MCAP, FEAT_PAR_MKT)


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


def _par_mkt(ordered: pd.DataFrame, window: int = DEFAULT_WINDOW) -> pd.Series:
    """Market-adjusted pre-announcement return on each decision-day row."""
    del window
    from src.benchmark import BENCHMARK_COLUMN
    from src.events import (
        DECISION_OFFSET,
        EVENT_COLUMN,
        MARKET_TIME_COLUMN,
        _event_anchors,
        _window_return,
    )

    required = (PRICE_COLUMN, BENCHMARK_COLUMN, EVENT_COLUMN, MARKET_TIME_COLUMN)
    missing = [column for column in required if column not in ordered.columns]
    if missing:
        raise KeyError(
            f"Panel missing {missing} required for par_mkt. "
            "Attach the benchmark before engineered features."
        )

    values = pd.Series(np.nan, index=ordered.index, dtype=float)
    for _, group in ordered.groupby("TickerId", sort=False):
        positional = group.reset_index(drop=True)
        prices = positional[PRICE_COLUMN].to_numpy(dtype=float)
        benchmark = positional[BENCHMARK_COLUMN].to_numpy(dtype=float)
        for _announcement, anchor in _event_anchors(positional):
            if anchor is None:
                continue
            decision_at = anchor + DECISION_OFFSET
            if decision_at < 0 or decision_at >= len(positional):
                continue
            par = _window_return(prices, anchor, PAR_START_OFFSET, PAR_END_OFFSET)
            par_bench = _window_return(benchmark, anchor, PAR_START_OFFSET, PAR_END_OFFSET)
            if np.isfinite(par) and np.isfinite(par_bench):
                values.loc[group.index[decision_at]] = par - par_bench
    return values


_BUILDERS = {
    FEAT_MOM_21D: lambda ordered, window: _mom_21d(ordered, window),
    FEAT_VOL_21D: lambda ordered, window: _vol_21d(ordered, window),
    FEAT_LOG_MCAP: lambda ordered, window: _log_mcap(ordered),
    FEAT_PAR_MKT: lambda ordered, window: _par_mkt(ordered, window),
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
