"""Attach a cached Yahoo adjusted-close benchmark to a daily panel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.prepare import DATA_DIR, _row_bar

BENCHMARK_COLUMN = "BenchmarkAdjClose"


def benchmark_paths(folder: Path, ticker: str) -> tuple[Path, Path]:
    stem = f"benchmark_{ticker.strip().lower()}"
    return folder / f"{stem}.parquet", folder / f"{stem}.json"


def _download_benchmark(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No prices returned for {ticker}.")
    close = raw["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    dates = pd.to_datetime(close.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    frame = pd.DataFrame({
        "Date": dates.normalize(),
        "AdjClose": pd.to_numeric(close, errors="coerce").to_numpy(),
    })
    frame = frame.dropna(subset=["AdjClose"]).drop_duplicates("Date", keep="last")
    return frame.sort_values("Date").reset_index(drop=True)


def load_benchmark(folder: Path, ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path, meta_path = benchmark_paths(folder, ticker)
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cached = pd.read_parquet(path)
        cached["Date"] = pd.to_datetime(cached["Date"])
        covers = (
            meta.get("ticker") == ticker
            and cached["Date"].min() <= start
            and cached["Date"].max() >= end
        )
        if covers:
            with _row_bar(len(cached)) as bar:
                bar.update(len(cached))
            return cached
    frame = _download_benchmark(ticker, start, end)
    if frame["Date"].min() > start or frame["Date"].max() < end:
        raise RuntimeError(
            f"{ticker} prices cover {frame['Date'].min():%Y-%m-%d} to "
            f"{frame['Date'].max():%Y-%m-%d}, short of {start:%Y-%m-%d} to {end:%Y-%m-%d}."
        )
    frame.to_parquet(path, index=False)
    meta = {
        "source": "yahoo",
        "ticker": ticker,
        "field": "Adj Close",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": frame["Date"].min().strftime("%Y-%m-%d"),
        "end": frame["Date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(frame)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    with _row_bar(len(frame)) as bar:
        bar.update(len(frame))
    return frame


def attach_benchmark(
    frame: pd.DataFrame,
    ticker: str = "SPY",
    data_dir: Path | None = None,
    column: str = BENCHMARK_COLUMN,
) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    prices = load_benchmark(folder, ticker, dates.min(), dates.max())
    lookup = prices.set_index(pd.to_datetime(prices["Date"]).dt.normalize())["AdjClose"]
    out = frame.copy()
    out[column] = dates.map(lookup).to_numpy()
    return out
