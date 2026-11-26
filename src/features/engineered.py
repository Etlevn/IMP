"""Panel features built from existing columns.

Formulas (no lookahead), windows in trading days:

    mom_{W}d       = LastPrice.pct_change(W)                         per TickerId
    vol_{W}d       = rolling std of 1d returns over W days (ddof=1)  per TickerId
    rel_vol_21d    = Volume / mean(Volume of prior 21 days)          exclude today
    amihud_21d     = mean(|ret_1d| / (LastPrice*Volume)) over 21d
    hl_range_21d   = mean((High-Low)/LastPrice) over 21d
    gap_1d         = (Open - prev Close) / prev Close
    mom_exmkt_{W}d = stock W-day return - benchmark W-day return
    log_mcap       = log(MarketCap) if MarketCap > 0 else NaN
    par_mkt        = stock return from close at anchor-5 to close at anchor-2,
                     minus the benchmark return over that same window.
                     Written on the DecisionDate row only.

Empty ``ENGINEERED_FEATURES`` disables attach.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import ENGINEERED_FEATURES
from src.prepare import _row_bar

PRICE_COLUMN = "LastPrice"
MCAP_COLUMN = "MarketCap"
DEFAULT_WINDOW = 21

FEAT_MOM_5D = "mom_5d"
FEAT_MOM_21D = "mom_21d"
FEAT_MOM_63D = "mom_63d"
FEAT_VOL_5D = "vol_5d"
FEAT_VOL_21D = "vol_21d"
FEAT_VOL_63D = "vol_63d"
FEAT_REL_VOL_21D = "rel_vol_21d"
FEAT_AMIHUD_21D = "amihud_21d"
FEAT_HL_RANGE_21D = "hl_range_21d"
FEAT_GAP_1D = "gap_1d"
FEAT_MOM_EXMKT_5D = "mom_exmkt_5d"
FEAT_MOM_EXMKT_21D = "mom_exmkt_21d"
FEAT_MOM_EXMKT_63D = "mom_exmkt_63d"
FEAT_LOG_MCAP = "log_mcap"
FEAT_PAR_MKT = "par_mkt"

# Close at anchor-5 -> close at anchor-2. Ends on DecisionDate (anchor-2).
PAR_START_OFFSET = -4
PAR_END_OFFSET = -2

ALL_ENGINEERED = (
    FEAT_MOM_5D,
    FEAT_MOM_21D,
    FEAT_MOM_63D,
    FEAT_VOL_5D,
    FEAT_VOL_21D,
    FEAT_VOL_63D,
    FEAT_REL_VOL_21D,
    FEAT_AMIHUD_21D,
    FEAT_HL_RANGE_21D,
    FEAT_GAP_1D,
    FEAT_MOM_EXMKT_5D,
    FEAT_MOM_EXMKT_21D,
    FEAT_MOM_EXMKT_63D,
    FEAT_LOG_MCAP,
    FEAT_PAR_MKT,
)

_MOM_VOL_FEATURES = frozenset(
    {
        FEAT_MOM_5D,
        FEAT_MOM_21D,
        FEAT_MOM_63D,
        FEAT_VOL_5D,
        FEAT_VOL_21D,
        FEAT_VOL_63D,
        FEAT_MOM_EXMKT_5D,
        FEAT_MOM_EXMKT_21D,
        FEAT_MOM_EXMKT_63D,
    }
)


def _mom_wd(ordered: pd.DataFrame, window: int) -> pd.Series:
    """W-day price momentum on a TickerId/Date-sorted frame."""
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    return price.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )


def _vol_wd(ordered: pd.DataFrame, window: int) -> pd.Series:
    """W-day rolling std of daily returns on a TickerId/Date-sorted frame."""
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    ret_1d = price.groupby(ordered["TickerId"], sort=False).pct_change(
        fill_method=None
    )
    return (
        ret_1d.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )


def _gap_1d(ordered: pd.DataFrame, window: int) -> pd.Series:
    del window
    open_ = pd.to_numeric(ordered["Open"], errors="coerce")
    close = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    prev_close = close.groupby(ordered["TickerId"], sort=False).shift(1)
    return (open_ - prev_close) / prev_close


def _rel_vol_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    vol = pd.to_numeric(ordered["Volume"], errors="coerce")
    prior_mean = (
        vol.groupby(ordered["TickerId"], sort=False)
        .shift(1)
        .groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return vol / prior_mean.replace(0, np.nan)


def _hl_range_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    high = pd.to_numeric(ordered["High"], errors="coerce")
    low = pd.to_numeric(ordered["Low"], errors="coerce")
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    daily = (high - low) / price.replace(0, np.nan)
    return (
        daily.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _amihud_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    volume = pd.to_numeric(ordered["Volume"], errors="coerce")
    ret_1d = price.groupby(ordered["TickerId"], sort=False).pct_change(
        fill_method=None
    )
    dollar = price * volume
    daily = ret_1d.abs() / dollar.replace(0, np.nan)
    daily = daily.where(dollar > 0)
    return (
        daily.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _mom_exmkt(ordered: pd.DataFrame, window: int) -> pd.Series:
    from src.benchmark import BENCHMARK_COLUMN

    if BENCHMARK_COLUMN not in ordered.columns:
        raise KeyError(
            f"Panel missing {BENCHMARK_COLUMN!r} required for mom_exmkt_*."
        )
    stock = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    bench = pd.to_numeric(ordered[BENCHMARK_COLUMN], errors="coerce")
    stock_mom = stock.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )
    bench_mom = bench.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )
    return stock_mom - bench_mom


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
    FEAT_MOM_5D: lambda ordered, window: _mom_wd(ordered, 5),
    FEAT_MOM_21D: lambda ordered, window: _mom_wd(ordered, 21),
    FEAT_MOM_63D: lambda ordered, window: _mom_wd(ordered, 63),
    FEAT_VOL_5D: lambda ordered, window: _vol_wd(ordered, 5),
    FEAT_VOL_21D: lambda ordered, window: _vol_wd(ordered, 21),
    FEAT_VOL_63D: lambda ordered, window: _vol_wd(ordered, 63),
    FEAT_REL_VOL_21D: lambda ordered, window: _rel_vol_21d(ordered, 21),
    FEAT_AMIHUD_21D: lambda ordered, window: _amihud_21d(ordered, 21),
    FEAT_HL_RANGE_21D: lambda ordered, window: _hl_range_21d(ordered, 21),
    FEAT_GAP_1D: lambda ordered, window: _gap_1d(ordered, window),
    FEAT_MOM_EXMKT_5D: lambda ordered, window: _mom_exmkt(ordered, 5),
    FEAT_MOM_EXMKT_21D: lambda ordered, window: _mom_exmkt(ordered, 21),
    FEAT_MOM_EXMKT_63D: lambda ordered, window: _mom_exmkt(ordered, 63),
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

    need_price = bool(_MOM_VOL_FEATURES.intersection(wanted)) or FEAT_AMIHUD_21D in wanted
    if need_price and PRICE_COLUMN not in frame.columns:
        raise KeyError(
            f"Panel missing {PRICE_COLUMN!r} required for mom/vol engineered features."
        )
    if FEAT_LOG_MCAP in wanted and MCAP_COLUMN not in frame.columns:
        raise KeyError(f"Panel missing {MCAP_COLUMN!r} required for log_mcap.")
    if FEAT_GAP_1D in wanted and "Open" not in frame.columns:
        raise KeyError("Panel missing 'Open' required for gap_1d.")
    if FEAT_HL_RANGE_21D in wanted and (
        "High" not in frame.columns or "Low" not in frame.columns
    ):
        raise KeyError("Panel missing High/Low required for hl_range_21d.")
    if (
        FEAT_REL_VOL_21D in wanted or FEAT_AMIHUD_21D in wanted
    ) and "Volume" not in frame.columns:
        raise KeyError("Panel missing 'Volume' required for rel_vol/amihud.")
    if "TickerId" not in frame.columns or "Date" not in frame.columns:
        raise KeyError("Panel missing TickerId / Date required for engineered features.")

    out = frame.copy()
    drop_cols = [c for c in wanted if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    ordered = out.sort_values(["TickerId", "Date"], kind="mergesort")
    with _row_bar(len(wanted), desc="engineered") as bar:
        for name in wanted:
            series = _BUILDERS[name](ordered, window)
            out.loc[ordered.index, name] = series.to_numpy(dtype=float)
            bar.update(1)

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
