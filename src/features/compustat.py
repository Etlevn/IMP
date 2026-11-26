"""Compustat-derived features (WRDS): cache → download → as-of attach (macro pattern A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import DATA_DIR, COMPUSTAT_FEATURES
from src.features.wrds_cache import (
    ASOF_COLUMN,
    TICKER_COLUMN,
    asof_attach,
    load_cached_or_raise,
    load_universe,
    open_wrds,
    universe_hash,
    write_feature_cache,
)

ALL_COMPUSTAT = (
    "bm",
    "roe",
    "roa",
    "gross_margin",
    "asset_growth",
    "revenue_growth",
    "leverage",
    "current_ratio",
    "accruals",
    "log_assets",
    "ep",
    "fcf_yield",
)

QUERY_FINGERPRINT = "comp.fundq|v1"


def _connect_username() -> str | None:
    try:
        from script.export_universe import WRDS_USERNAME
    except Exception:  # noqa: BLE001
        return None
    return WRDS_USERNAME


def _download_compustat_frame(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    username: str | None = None,
) -> pd.DataFrame:
    universe = load_universe(folder)
    if "gvkey" not in universe.columns:
        raise ValueError(
            "stock_universe.csv needs gvkey. Re-run export_universe.py with USE_WRDS=True."
        )
    link = universe.dropna(subset=["gvkey"]).copy()
    link["gvkey"] = link["gvkey"].astype(str).str.zfill(6)
    link["TickerId"] = pd.to_numeric(link["TickerId"], errors="coerce")
    link = link.dropna(subset=["TickerId"])
    gvkeys = sorted(set(link["gvkey"].tolist()))
    if not gvkeys:
        raise RuntimeError("No gvkey values in stock_universe.csv for Compustat download.")

    q_start = (pd.Timestamp(start) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    q_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    gv_sql = ",".join(f"'{g}'" for g in gvkeys)

    db = open_wrds(username if username is not None else _connect_username())
    try:
        fundq = db.raw_sql(
            f"""
            SELECT gvkey, datadate, rdq, fyr, fqtr,
                   atq, ceqq, seqq, ltq, actq, lctq,
                   saleq, cogsq, niq, oancfy, capxy, cshoq, prccq,
                   rectq, invtq, apq
            FROM comp.fundq
            WHERE gvkey IN ({gv_sql})
              AND indfmt = 'INDL'
              AND datafmt = 'STD'
              AND popsrc = 'D'
              AND consol = 'C'
              AND datadate >= DATE '{q_start}'
              AND datadate < DATE '{q_end}'
            """
        )
    finally:
        db.close()

    if fundq is None or len(fundq) == 0:
        raise RuntimeError("Compustat fundq returned no rows for the universe.")

    df = fundq.copy()
    df["gvkey"] = df["gvkey"].astype(str).str.zfill(6)
    df["datadate"] = pd.to_datetime(df["datadate"]).dt.normalize()
    df["rdq"] = pd.to_datetime(df["rdq"], errors="coerce").dt.normalize()
    # Availability: report date when present, else conservative datadate + 45 days.
    df[ASOF_COLUMN] = df["rdq"].where(
        df["rdq"].notna(), df["datadate"] + pd.Timedelta(days=45)
    )

    num_cols = [
        "atq",
        "ceqq",
        "seqq",
        "ltq",
        "actq",
        "lctq",
        "saleq",
        "cogsq",
        "niq",
        "oancfy",
        "capxy",
        "cshoq",
        "prccq",
        "rectq",
        "invtq",
        "apq",
    ]
    for col in num_cols:
        if col in df.columns:
            # Plain float64 so numpy masks never see pandas NA.
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    equity = df["ceqq"].where(df["ceqq"].notna(), df["seqq"]).astype("float64")
    mcap = (df["cshoq"] * df["prccq"]).astype("float64")
    atq = df["atq"].to_numpy(dtype=float, copy=False)
    df["bm"] = equity / mcap.replace(0, np.nan)
    df["roe"] = df["niq"] / equity.replace(0, np.nan)
    df["roa"] = df["niq"] / df["atq"].replace(0, np.nan)
    df["gross_margin"] = (df["saleq"] - df["cogsq"]) / df["saleq"].replace(0, np.nan)
    df["leverage"] = df["ltq"] / df["atq"].replace(0, np.nan)
    df["current_ratio"] = df["actq"] / df["lctq"].replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        df["log_assets"] = np.where(np.isfinite(atq) & (atq > 0), np.log(atq), np.nan)
    df["ep"] = df["niq"] / mcap.replace(0, np.nan)
    # Quarterly FCF approx: operating CF YTD change is messy; use oancfy - capxy when both present.
    df["fcf_yield"] = (df["oancfy"] - df["capxy"]) / mcap.replace(0, np.nan)

    df = df.sort_values(["gvkey", ASOF_COLUMN])
    g = df.groupby("gvkey", sort=False)
    # Prior-year level of 0 → NaN (avoid ±inf growth).
    prev_atq = g["atq"].shift(4).replace(0, np.nan)
    prev_sale = g["saleq"].shift(4).replace(0, np.nan)
    asset_g = df["atq"] / prev_atq - 1.0
    rev_g = df["saleq"] / prev_sale - 1.0
    df["asset_growth"] = asset_g.where(np.isfinite(asset_g), np.nan)
    df["revenue_growth"] = rev_g.where(np.isfinite(rev_g), np.nan)
    # Simple accruals: (ΔCA - ΔCash) - (ΔCL - ΔDebt) - Dep; without cash/debt detail use
    # (niq - oancfy) / atq as a coarse accrual proxy when oancfy present.
    df["accruals"] = (df["niq"] - df["oancfy"]) / df["atq"].replace(0, np.nan)

    df = df.merge(
        link.loc[:, ["TickerId", "gvkey"]].drop_duplicates(),
        on="gvkey",
        how="inner",
    )
    if df.empty:
        raise RuntimeError("Compustat rows could not be linked to TickerId via gvkey.")

    df = df.sort_values([TICKER_COLUMN, ASOF_COLUMN]).drop_duplicates(
        subset=[TICKER_COLUMN, ASOF_COLUMN], keep="last"
    )
    for col in ALL_COMPUSTAT:
        if col not in df.columns:
            df[col] = np.nan
    cols = [TICKER_COLUMN, ASOF_COLUMN, *ALL_COMPUSTAT]
    return df.loc[:, cols].reset_index(drop=True)


def fetch_compustat_cache(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    compustat_features: list[str] | None = None,
    username: str | None = None,
) -> pd.DataFrame:
    """Download from WRDS and write cache (``script/fetch_wrds_compustat.py``)."""
    wanted = list(COMPUSTAT_FEATURES if compustat_features is None else compustat_features)
    if not wanted:
        raise ValueError("COMPUSTAT_FEATURES is empty; nothing to fetch.")
    unknown = [n for n in wanted if n not in ALL_COMPUSTAT]
    if unknown:
        raise ValueError(
            f"Unknown COMPUSTAT_FEATURES {unknown}; allowed: {list(ALL_COMPUSTAT)}"
        )

    universe = load_universe(folder)
    uhash = universe_hash(universe.get("gvkey", universe["TickerId"]).tolist())
    frame = _download_compustat_frame(folder, start, end, username=username)
    write_feature_cache(
        folder,
        "compustat",
        frame,
        features=wanted,
        universe_hash_value=uhash,
        query_fingerprint=QUERY_FINGERPRINT,
    )
    return load_cached_or_raise(
        folder, "compustat", features=wanted, start=start, end=end
    )


def load_compustat(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    compustat_features: list[str] | None = None,
) -> pd.DataFrame:
    """Read Compustat cache only. Missing → error pointing at fetch script."""
    wanted = list(COMPUSTAT_FEATURES if compustat_features is None else compustat_features)
    if not wanted:
        return pd.DataFrame(columns=[TICKER_COLUMN, ASOF_COLUMN])
    unknown = [n for n in wanted if n not in ALL_COMPUSTAT]
    if unknown:
        raise ValueError(
            f"Unknown COMPUSTAT_FEATURES {unknown}; allowed: {list(ALL_COMPUSTAT)}"
        )
    return load_cached_or_raise(
        folder, "compustat", features=wanted, start=start, end=end
    )


def attach_compustat(
    frame: pd.DataFrame,
    data_dir: Path | None = None,
    compustat_features: list[str] | None = None,
) -> pd.DataFrame:
    wanted = list(COMPUSTAT_FEATURES if compustat_features is None else compustat_features)
    if not wanted:
        return frame.copy()
    unknown = [n for n in wanted if n not in ALL_COMPUSTAT]
    if unknown:
        raise ValueError(
            f"Unknown COMPUSTAT_FEATURES {unknown}; allowed: {list(ALL_COMPUSTAT)}"
        )

    folder = DATA_DIR if data_dir is None else data_dir
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    cache = load_compustat(
        folder, dates.min(), dates.max(), compustat_features=wanted
    )
    return asof_attach(
        frame, cache, features=wanted, date_column="Date", desc="compustat"
    )
