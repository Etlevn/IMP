"""Download IBES features from WRDS into data/ibes_features.parquet (+ .json).

Attach paths only read the cache (no WRDS in main / feat_filter).

Usage (from repo root):
    python script/export_universe.py   # need permno
    python script/fetch_wrds_ibes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import DATA_DIR, IBES_FEATURES
from src.features.ibes import fetch_ibes_cache
from src.features.wrds_cache import cache_paths
from src.prepare import load_stock

WRDS_USERNAME: str | None = None


def main() -> None:
    if not IBES_FEATURES:
        raise SystemExit("IBES_FEATURES is empty in config/config.py; nothing to fetch.")
    stock = load_stock(DATA_DIR)
    start = pd.to_datetime(stock["Date"]).min()
    end = pd.to_datetime(stock["Date"]).max()
    print(f"fetching IBES {start:%Y-%m-%d} .. {end:%Y-%m-%d}")
    fetch_ibes_cache(
        DATA_DIR,
        start,
        end,
        ibes_features=list(IBES_FEATURES),
        username=WRDS_USERNAME,
    )
    path, meta_path = cache_paths(DATA_DIR, "ibes")
    print(f"[WROTE] {path}")
    print(f"[WROTE] {meta_path}")


if __name__ == "__main__":
    main()
