"""Download Yahoo macro Close series into data/macro_daily.parquet (+ .json).

Pipeline attach only reads this cache; it does not call yfinance.

Usage (from repo root):
    python script/fetch_macro.py

Date window defaults to the configured stock panel's min/max Date.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import DATA_DIR, MACRO_FEATURES
from src.features.macro import (
    download_macro_series,
    load_macro,
    write_macro_cache,
)
from src.prepare import load_stock


def main() -> None:
    mapping = dict(MACRO_FEATURES)
    if not mapping:
        raise SystemExit("MACRO_FEATURES is empty in config/config.py; nothing to fetch.")

    stock = load_stock(DATA_DIR)
    start = pd.to_datetime(stock["Date"]).min()
    end = pd.to_datetime(stock["Date"]).max()
    print(f"fetching macro {start:%Y-%m-%d} .. {end:%Y-%m-%d}  n={len(mapping)}")
    frame = download_macro_series(mapping, start, end)
    if frame["Date"].min() > start or frame["Date"].max() < end:
        raise RuntimeError(
            f"Yahoo macro cover {frame['Date'].min():%Y-%m-%d} to "
            f"{frame['Date'].max():%Y-%m-%d}, short of {start:%Y-%m-%d} to {end:%Y-%m-%d}."
        )
    path, meta_path = write_macro_cache(DATA_DIR, frame, mapping)
    # Sanity: loader must accept the cache we just wrote.
    load_macro(DATA_DIR, start, end, macro_features=mapping)
    print(f"[WROTE] {path}")
    print(f"[WROTE] {meta_path}")


if __name__ == "__main__":
    main()
