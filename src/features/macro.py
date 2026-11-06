"""Attach cached Yahoo macro series (VIX / TNX / DXY, …) to a daily panel."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.default import MACRO_FEATURES
from src.prepare import DATA_DIR, _row_bar


def macro_paths(folder: Path) -> tuple[Path, Path]:
    return folder / "macro_daily.parquet", folder / "macro_daily.json"


def _download_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    import yfinance as yf

    raw = yf.download(
        ticker,
        start=start.strftime("%Y-%m-%d"),
        end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    if raw is None or len(raw) == 0:
        raise RuntimeError(f"No Yahoo prices returned for macro ticker {ticker!r}.")
    close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    dates = pd.to_datetime(close.index)
    if getattr(dates, "tz", None) is not None:
        dates = dates.tz_localize(None)
    series = pd.Series(
        pd.to_numeric(close, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(dates).normalize(),
        name=ticker,
    )
    return series.dropna()


def _download_macro_series(
    series: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not series:
        raise ValueError("MACRO_FEATURES is empty.")
    parts: dict[str, pd.Series] = {}
    for column, ticker in series.items():
        parts[column] = _download_one(ticker, start, end).rename(column)
    frame = pd.concat(parts.values(), axis=1, join="outer").sort_index()
    frame.index.name = "Date"
    out = frame.reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def load_macro(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    macro_features: dict[str, str] | None = None,
) -> pd.DataFrame:
    mapping = dict(MACRO_FEATURES if macro_features is None else macro_features)
    if not mapping:
        return pd.DataFrame(columns=["Date"])

    path, meta_path = macro_paths(folder)
    columns = list(mapping.keys())
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        cached = pd.read_parquet(path)
        cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
        meta_series = meta.get("series") or {}
        covers = (
            meta_series == mapping
            and set(columns).issubset(cached.columns)
            and cached["Date"].min() <= start
            and cached["Date"].max() >= end
        )
        if covers:
            with _row_bar(len(cached)) as bar:
                bar.update(len(cached))
            return cached.loc[:, ["Date", *columns]].sort_values("Date").reset_index(drop=True)

    frame = _download_macro_series(mapping, start, end)
    if frame["Date"].min() > start or frame["Date"].max() < end:
        raise RuntimeError(
            f"Macro series cover {frame['Date'].min():%Y-%m-%d} to "
            f"{frame['Date'].max():%Y-%m-%d}, short of {start:%Y-%m-%d} to {end:%Y-%m-%d}."
        )
    folder.mkdir(parents=True, exist_ok=True)
    out = frame.loc[:, ["Date", *columns]].copy()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    out.to_parquet(path, index=False)
    meta = {
        "source": "yahoo",
        "series": mapping,
        "field": "Close",
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": out["Date"].min().strftime("%Y-%m-%d"),
        "end": out["Date"].max().strftime("%Y-%m-%d"),
        "rows": int(len(out)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    with _row_bar(len(out)) as bar:
        bar.update(len(out))
    return out


def attach_macro(
    frame: pd.DataFrame,
    data_dir: Path | None = None,
    macro_features: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Left-merge Yahoo macro columns onto ``frame`` by ``Date``."""
    mapping = dict(MACRO_FEATURES if macro_features is None else macro_features)
    if not mapping:
        return frame.copy()

    folder = DATA_DIR if data_dir is None else data_dir
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    macro = load_macro(folder, dates.min(), dates.max(), macro_features=mapping)
    macro = macro.copy()
    macro["Date"] = pd.to_datetime(macro["Date"]).dt.normalize()

    out = frame.copy()
    out["Date"] = dates
    drop_cols = [c for c in mapping if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)
    return out.merge(macro, on="Date", how="left")
