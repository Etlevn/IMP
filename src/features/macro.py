"""Yahoo macro levels: fetch script writes cache; attach only reads cache."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config.config import MACRO_FEATURES
from src.prepare import DATA_DIR, _row_bar


def macro_paths(folder: Path) -> tuple[Path, Path]:
    return folder / "macro_daily.parquet", folder / "macro_daily.json"


def download_one(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Download one Yahoo Close series (used by ``script/fetch_macro.py``)."""
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


def download_macro_series(
    series: dict[str, str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    if not series:
        raise ValueError("MACRO_FEATURES is empty.")
    parts: dict[str, pd.Series] = {}
    for column, ticker in series.items():
        parts[column] = download_one(ticker, start, end).rename(column)
    frame = pd.concat(parts.values(), axis=1, join="outer").sort_index()
    frame.index.name = "Date"
    out = frame.reset_index()
    out["Date"] = pd.to_datetime(out["Date"]).dt.normalize()
    return out.drop_duplicates("Date", keep="last").sort_values("Date").reset_index(drop=True)


def write_macro_cache(
    folder: Path,
    frame: pd.DataFrame,
    mapping: dict[str, str],
) -> tuple[Path, Path]:
    """Persist macro parquet + json (called from fetch script only)."""
    if frame is None or len(frame) == 0:
        raise RuntimeError("Refusing to write empty macro cache.")
    columns = list(mapping.keys())
    path, meta_path = macro_paths(folder)
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
    return path, meta_path


def load_macro(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    macro_features: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Read macro cache only. Missing / short / mismatched → hard error.

    Download with: ``python script/fetch_macro.py``
    """
    mapping = dict(MACRO_FEATURES if macro_features is None else macro_features)
    if not mapping:
        return pd.DataFrame(columns=["Date"])

    path, meta_path = macro_paths(folder)
    columns = list(mapping.keys())
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing macro cache under {folder} "
            f"({path.name} / {meta_path.name}). "
            "Run: python script/fetch_macro.py"
        )

    meta = json.loads(meta_path.read_text())
    cached = pd.read_parquet(path)
    if len(cached) == 0:
        raise RuntimeError(f"Macro cache is empty: {path}")
    cached["Date"] = pd.to_datetime(cached["Date"]).dt.normalize()
    meta_series = meta.get("series") or {}
    covers = (
        meta_series == mapping
        and set(columns).issubset(cached.columns)
        and cached["Date"].min() <= pd.Timestamp(start).normalize()
        and cached["Date"].max() >= pd.Timestamp(end).normalize()
    )
    if not covers:
        raise RuntimeError(
            "Macro cache does not cover the requested series/date range "
            f"(need {pd.Timestamp(start):%Y-%m-%d}..{pd.Timestamp(end):%Y-%m-%d}, "
            f"mapping={mapping}). Re-run: python script/fetch_macro.py"
        )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("sha256") and meta["sha256"] != digest:
        raise RuntimeError(
            "Macro cache sha256 mismatch. Re-run: python script/fetch_macro.py"
        )

    with _row_bar(len(cached)) as bar:
        bar.update(len(cached))
    return cached.loc[:, ["Date", *columns]].sort_values("Date").reset_index(drop=True)


def attach_macro(
    frame: pd.DataFrame,
    data_dir: Path | None = None,
    macro_features: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Left-merge cached Yahoo macro columns onto ``frame`` by ``Date``."""
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
