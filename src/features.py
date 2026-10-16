"""Event-level feature and target construction from the merged daily panel."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.benchmark import BENCHMARK_COLUMN, attach_benchmark
from src.prepare import _row_bar
from src.sessions import align_trading_sessions

# Project-fixed target windows (not per-factor).
ANN_START_OFFSET = -1
ANN_END_OFFSET = 1
POST_START_OFFSET = 1
POST_END_OFFSET = 5
DECISION_OFFSET = -2

PRICE_COLUMN = "LastPrice"
EVENT_COLUMN = "EventDate"
MARKET_TIME_COLUMN = "MarketTimeId"
BEFORE_OPEN_ID = 1
AFTER_CLOSE_ID = 2

FEATURE_COLUMNS = [
    "QScoreTrendLong",
    "QScoreSpeedLong",
    "QScoreLong",
    "analystRatingsStrongBuy",
    "analystRatingsBuy",
    "analystRatingsHold",
    "analystRatingsSell",
    "analystRatingsStrongSell",
]

DROP_FROM_PANEL = [
    "symbol",
    "date",
    "ShortName_before",
    "tickerId",
]


def _window_return(prices: np.ndarray, position: int, start_offset: int, end_offset: int) -> float:
    start = position + start_offset - 1
    end = position + end_offset
    if start < 0 or end >= len(prices):
        return np.nan
    start_price = prices[start]
    end_price = prices[end]
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price == 0:
        return np.nan
    return float(end_price / start_price - 1.0)


def _window_vol(prices: np.ndarray, position: int, start_offset: int, end_offset: int) -> float:
    rets: list[float] = []
    for offset in range(start_offset, end_offset + 1):
        index = position + offset
        if index < 1 or index >= len(prices):
            return np.nan
        prev_price = prices[index - 1]
        price = prices[index]
        if not np.isfinite(prev_price) or not np.isfinite(price) or prev_price == 0:
            return np.nan
        rets.append(float(price / prev_price - 1.0))
    if not rets:
        return np.nan
    return float(np.nanstd(rets, ddof=1)) if len(rets) > 1 else float(np.nan)


def _one_step_return(prices: np.ndarray, position: int) -> float:
    nxt = position + 1
    if position < 0 or nxt >= len(prices):
        return np.nan
    start_price = prices[position]
    end_price = prices[nxt]
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price == 0:
        return np.nan
    return float(end_price / start_price - 1.0)


def _event_anchors(frame: pd.DataFrame) -> list[tuple[int, int | None]]:
    market_time = pd.to_numeric(frame[MARKET_TIME_COLUMN], errors="coerce").to_numpy(dtype=float)
    announcements = frame.index[frame[EVENT_COLUMN].notna()].to_numpy()
    last = len(frame) - 1
    anchors: list[tuple[int, int | None]] = []
    for announcement in announcements:
        announcement = int(announcement)
        code = market_time[announcement]
        if not np.isfinite(code):
            anchors.append((announcement, None))
            continue
        code = int(code)
        if code == BEFORE_OPEN_ID:
            anchors.append((announcement, announcement))
        elif code == AFTER_CLOSE_ID:
            nxt = announcement + 1
            anchors.append((announcement, nxt if nxt <= last else None))
        else:
            raise ValueError(f"MarketTimeId must be 1 or 2, got {code}.")
    return anchors


def _extract_earnings(df_all: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ("TickerId", "EventDate", "MarketTimeId", "Id") if c in df_all.columns]
    events = df_all.dropna(subset=["EventDate"])[cols].copy()
    events["EventDate"] = pd.to_datetime(events["EventDate"])
    return events.drop_duplicates(subset=["TickerId", "EventDate"], keep="last")


def _price_panel(df_all: pd.DataFrame) -> pd.DataFrame:
    panel = df_all.copy()
    panel["Date"] = pd.to_datetime(panel["Date"])
    drop_cols = [c for c in DROP_FROM_PANEL if c in panel.columns]
    panel = panel.drop(columns=drop_cols)
    return panel.drop_duplicates(subset=["TickerId", "Date"], keep="last")


def build_event_table(
    df_all: pd.DataFrame,
    market_ticker: str = "SPY",
    data_dir: Path | None = None,
) -> pd.DataFrame:
    """Build training event table ``df``: one row per announcement."""
    earnings = _extract_earnings(df_all)
    panel = _price_panel(df_all)
    aligned = align_trading_sessions(panel, earnings)
    aligned = attach_benchmark(aligned, ticker=market_ticker, data_dir=data_dir)

    rows: list[dict] = []
    groups = list(aligned.groupby("TickerId", sort=False))
    n_events = int(aligned[EVENT_COLUMN].notna().sum())
    with _row_bar(max(n_events, 1)) as bar:
        for ticker, frame in groups:
            frame = frame.sort_values("Date").reset_index(drop=True)
            prices = frame[PRICE_COLUMN].to_numpy(dtype=float)
            benchmark = frame[BENCHMARK_COLUMN].to_numpy(dtype=float)
            sessions = pd.to_datetime(frame["Date"]).to_numpy()
            for announcement, anchor in _event_anchors(frame):
                if anchor is None:
                    bar.update(1)
                    continue
                decision_at = anchor + DECISION_OFFSET
                if decision_at < 0 or decision_at >= len(frame):
                    bar.update(1)
                    continue

                ret_ann = _window_return(prices, anchor, ANN_START_OFFSET, ANN_END_OFFSET)
                ret_ann_mkt_b = _window_return(benchmark, anchor, ANN_START_OFFSET, ANN_END_OFFSET)
                ret_post = _window_return(prices, anchor, POST_START_OFFSET, POST_END_OFFSET)
                ret_post_mkt_b = _window_return(benchmark, anchor, POST_START_OFFSET, POST_END_OFFSET)
                ret_next = _one_step_return(prices, announcement)
                ret_next_mkt_b = _one_step_return(benchmark, announcement)

                ret_ann_mkt = (
                    ret_ann - ret_ann_mkt_b
                    if np.isfinite(ret_ann) and np.isfinite(ret_ann_mkt_b)
                    else np.nan
                )
                ret_post_mkt = (
                    ret_post - ret_post_mkt_b
                    if np.isfinite(ret_post) and np.isfinite(ret_post_mkt_b)
                    else np.nan
                )
                ret_next_mkt = (
                    ret_next - ret_next_mkt_b
                    if np.isfinite(ret_next) and np.isfinite(ret_next_mkt_b)
                    else np.nan
                )

                decision = frame.iloc[decision_at]
                release = frame.iloc[announcement]
                row = {
                    "TickerId": ticker,
                    "Symbol": decision.get("Symbol", release.get("Symbol")),
                    "ShortName": decision.get("ShortName", release.get("ShortName")),
                    "EventDate": pd.Timestamp(release[EVENT_COLUMN]).normalize(),
                    "release_session": pd.Timestamp(sessions[announcement]).normalize(),
                    "anchor_session": pd.Timestamp(sessions[anchor]).normalize(),
                    "decision_session": pd.Timestamp(sessions[decision_at]).normalize(),
                    "MarketTimeId": release.get(MARKET_TIME_COLUMN),
                    "ret_ann": ret_ann,
                    "ret_ann_mkt": ret_ann_mkt,
                    "ret_next": ret_next,
                    "ret_next_mkt": ret_next_mkt,
                    "ret_post": ret_post,
                    "ret_post_mkt": ret_post_mkt,
                    "dir_ann": (
                        float(np.sign(ret_ann_mkt)) if np.isfinite(ret_ann_mkt) else np.nan
                    ),
                    "vol_ann": _window_vol(prices, anchor, ANN_START_OFFSET, ANN_END_OFFSET),
                    "vol_post": _window_vol(prices, anchor, POST_START_OFFSET, POST_END_OFFSET),
                }
                for column in FEATURE_COLUMNS:
                    row[column] = decision[column] if column in frame.columns else np.nan
                if "Id" in frame.columns and pd.notna(release.get("Id")):
                    row["Id"] = release["Id"]
                rows.append(row)
                bar.update(1)

    df = pd.DataFrame(rows)
    if len(df):
        for column in (
            "EventDate",
            "release_session",
            "anchor_session",
            "decision_session",
        ):
            df[column] = pd.to_datetime(df[column]).dt.normalize()
        df = df.sort_values(["TickerId", "EventDate"]).reset_index(drop=True)
    return df
