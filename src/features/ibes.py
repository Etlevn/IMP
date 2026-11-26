"""IBES-derived features (WRDS): cache → download → as-of attach (macro pattern A)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.config import DATA_DIR, IBES_FEATURES
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
from src.prepare import _row_bar

ALL_IBES = (
    "eps_consensus",
    "eps_dispersion",
    "eps_coverage",
    "eps_revision_1m",
    "eps_revision_3m",
    "sue_lag1",
    "sue_lag2",
    "beat_rate_4q",
    "rev_surprise_lag1",
    "ltg",
    "eps_consensus_chg_std",
    "days_since_est_update",
)

QUERY_FINGERPRINT = "ibes.statsum_epsus|qtr_fpi6+ltg|v2"


def _connect_username() -> str | None:
    try:
        from script.export_universe import WRDS_USERNAME
    except Exception:  # noqa: BLE001
        return None
    return WRDS_USERNAME


def _download_ibes_frame(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    username: str | None = None,
) -> pd.DataFrame:
    universe = load_universe(folder)
    if "Symbol" not in universe.columns:
        raise ValueError("stock_universe.csv needs Symbol for IBES ticker matching.")
    link = universe.copy()
    link["TickerId"] = pd.to_numeric(link["TickerId"], errors="coerce")
    link = link.dropna(subset=["TickerId", "Symbol"])
    link["ticker_key"] = link["Symbol"].astype(str).str.upper().str.strip()
    link = link.drop_duplicates(subset=["ticker_key"], keep="first")
    symbols = sorted(set(link["ticker_key"].tolist()))
    if not symbols:
        raise RuntimeError("No Symbol values in stock_universe.csv for IBES download.")

    # Pad history for lag SUE / revisions.
    q_start = (pd.Timestamp(start) - pd.DateOffset(years=3)).strftime("%Y-%m-%d")
    q_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    sym_sql = ", ".join(f"'{s}'" for s in symbols)

    db = open_wrds(username if username is not None else _connect_username())
    try:
        # Match IBES oftic/ticker to universe Symbol (no crsp.stocknames.ofitic —
        # that column does not exist; CRSP uses ticker / namedt / nameenddt).
        stats = db.raw_sql(
            f"""
            SELECT ticker, oftic, measure, fiscalp, fpi, statpers,
                   numest, medest, meanest, stdev, actual, anndats_act
            FROM ibes.statsum_epsus
            WHERE measure = 'EPS'
              AND fiscalp = 'QTR'
              AND fpi = '6'
              AND statpers >= DATE '{q_start}'
              AND statpers < DATE '{q_end}'
              AND (
                    UPPER(TRIM(oftic)) IN ({sym_sql})
                 OR UPPER(TRIM(ticker)) IN ({sym_sql})
              )
            """
        )
        try:
            ltg = db.raw_sql(
                f"""
                SELECT ticker, oftic, statpers, medest AS ltg_med, meanest AS ltg_mean
                FROM ibes.statsum_epsus
                WHERE measure = 'EPS'
                  AND fiscalp = 'LTG'
                  AND statpers >= DATE '{q_start}'
                  AND statpers < DATE '{q_end}'
                  AND (
                        UPPER(TRIM(oftic)) IN ({sym_sql})
                     OR UPPER(TRIM(ticker)) IN ({sym_sql})
                  )
                """
            )
        except Exception:  # noqa: BLE001
            ltg = pd.DataFrame()
    finally:
        db.close()

    if stats is None or len(stats) == 0:
        raise RuntimeError("IBES statsum_epsus returned no EPS QTR rows for the universe.")

    stats = stats.copy()
    stats["statpers"] = pd.to_datetime(stats["statpers"]).dt.normalize()
    for col in ("medest", "meanest", "stdev", "numest", "actual"):
        if col in stats.columns:
            stats[col] = pd.to_numeric(stats[col], errors="coerce")

    # Prefer official ticker (oftic), else IBES ticker; join to universe Symbol.
    oftic = stats["oftic"] if "oftic" in stats.columns else pd.Series(pd.NA, index=stats.index)
    ticker = stats["ticker"] if "ticker" in stats.columns else pd.Series(pd.NA, index=stats.index)
    stats["ticker_key"] = oftic.fillna(ticker).astype(str).str.upper().str.strip()

    mapped = stats.merge(
        link.loc[:, ["TickerId", "ticker_key"]],
        on="ticker_key",
        how="inner",
    )
    if mapped.empty:
        raise RuntimeError(
            "IBES rows could not be linked to TickerId via Symbol↔oftic/ticker."
        )

    mapped = mapped.sort_values(["TickerId", "statpers"]).drop_duplicates(
        subset=["TickerId", "statpers"], keep="last"
    )

    # Consensus level features on each STATPERS.
    mapped["eps_consensus"] = mapped["medest"].where(
        mapped["medest"].notna(), mapped["meanest"]
    )
    mapped["eps_dispersion"] = mapped["stdev"]
    mapped["eps_coverage"] = mapped["numest"]
    # Filled at attach time as (Date - AsOfDate).days; placeholder in cache.
    mapped["days_since_est_update"] = np.nan

    # Revisions vs prior STATPERS for same ticker.
    g = mapped.groupby("TickerId", sort=False)
    mapped["eps_revision_1m"] = mapped["eps_consensus"] - g["eps_consensus"].shift(1)
    mapped["eps_revision_3m"] = mapped["eps_consensus"] - g["eps_consensus"].shift(3)
    mapped["eps_consensus_chg_std"] = g["eps_consensus"].transform(
        lambda s: s.diff().rolling(6, min_periods=3).std()
    )

    # SUE at announcement: (actual - consensus) / stdev; lag onto later STATPERS.
    surprise = (mapped["actual"] - mapped["eps_consensus"]) / mapped["stdev"].replace(
        0, np.nan
    )
    mapped["_sue"] = surprise
    mapped["sue_lag1"] = g["_sue"].shift(1)
    mapped["sue_lag2"] = g["_sue"].shift(2)
    beat = (mapped["actual"] > mapped["eps_consensus"]).astype(float)
    mapped["beat_rate_4q"] = (
        beat.groupby(mapped["TickerId"], sort=False)
        .transform(lambda s: s.rolling(4, min_periods=1).mean())
    )
    # Revenue surprise not in EPS summary; leave NaN unless extended later.
    mapped["rev_surprise_lag1"] = np.nan

    mapped["ltg"] = np.nan
    if ltg is not None and len(ltg):
        ltg = ltg.copy()
        ltg["statpers"] = pd.to_datetime(ltg["statpers"]).dt.normalize()
        ltg_oftic = ltg["oftic"] if "oftic" in ltg.columns else pd.Series(pd.NA, index=ltg.index)
        ltg_ticker = ltg["ticker"] if "ticker" in ltg.columns else pd.Series(pd.NA, index=ltg.index)
        ltg["ticker_key"] = ltg_oftic.fillna(ltg_ticker).astype(str).str.upper().str.strip()
        ltg_val = ltg["ltg_med"] if "ltg_med" in ltg.columns else ltg.get("ltg_mean")
        ltg["ltg"] = pd.to_numeric(ltg_val, errors="coerce")
        ltg_m = ltg.merge(
            link.loc[:, ["TickerId", "ticker_key"]],
            on="ticker_key",
            how="inner",
        )
        ltg_m = ltg_m.sort_values(["TickerId", "statpers"]).drop_duplicates(
            ["TickerId", "statpers"], keep="last"
        )
        mapped = mapped.drop(columns=["ltg"], errors="ignore")
        mapped = mapped.merge(
            ltg_m.loc[:, ["TickerId", "statpers", "ltg"]],
            on=["TickerId", "statpers"],
            how="left",
        )

    out = mapped.rename(columns={"statpers": ASOF_COLUMN})
    cols = [TICKER_COLUMN, ASOF_COLUMN, *ALL_IBES]
    for col in ALL_IBES:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, cols].reset_index(drop=True)


def fetch_ibes_cache(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    ibes_features: list[str] | None = None,
    username: str | None = None,
) -> pd.DataFrame:
    """Download from WRDS and write cache (``script/fetch_wrds_ibes.py`` only)."""
    wanted = list(IBES_FEATURES if ibes_features is None else ibes_features)
    if not wanted:
        raise ValueError("IBES_FEATURES is empty; nothing to fetch.")
    unknown = [n for n in wanted if n not in ALL_IBES]
    if unknown:
        raise ValueError(f"Unknown IBES_FEATURES {unknown}; allowed: {list(ALL_IBES)}")

    universe = load_universe(folder)
    uhash = universe_hash(universe.get("permno", universe["TickerId"]).tolist())
    frame = _download_ibes_frame(folder, start, end, username=username)
    write_feature_cache(
        folder,
        "ibes",
        frame,
        features=wanted,
        universe_hash_value=uhash,
        query_fingerprint=QUERY_FINGERPRINT,
    )
    return load_cached_or_raise(
        folder, "ibes", features=wanted, start=start, end=end
    )


def load_ibes(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    ibes_features: list[str] | None = None,
) -> pd.DataFrame:
    """Read IBES cache only. Missing → error pointing at fetch script."""
    wanted = list(IBES_FEATURES if ibes_features is None else ibes_features)
    if not wanted:
        return pd.DataFrame(columns=[TICKER_COLUMN, ASOF_COLUMN])
    unknown = [n for n in wanted if n not in ALL_IBES]
    if unknown:
        raise ValueError(f"Unknown IBES_FEATURES {unknown}; allowed: {list(ALL_IBES)}")
    return load_cached_or_raise(
        folder, "ibes", features=wanted, start=start, end=end
    )


def attach_ibes(
    frame: pd.DataFrame,
    data_dir: Path | None = None,
    ibes_features: list[str] | None = None,
) -> pd.DataFrame:
    wanted = list(IBES_FEATURES if ibes_features is None else ibes_features)
    if not wanted:
        return frame.copy()
    unknown = [n for n in wanted if n not in ALL_IBES]
    if unknown:
        raise ValueError(f"Unknown IBES_FEATURES {unknown}; allowed: {list(ALL_IBES)}")

    folder = DATA_DIR if data_dir is None else data_dir
    dates = pd.to_datetime(frame["Date"]).dt.normalize()
    cache = load_ibes(folder, dates.min(), dates.max(), ibes_features=wanted)
    with _row_bar(len(cache)) as bar:
        bar.update(len(cache))
    return asof_attach(
        frame,
        cache,
        features=wanted,
        date_column="Date",
        age_feature="days_since_est_update",
    )
