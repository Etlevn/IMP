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
    cache_paths,
    load_cached_or_raise,
    load_universe,
    open_wrds,
    universe_hash,
    write_feature_cache,
)

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

# Quarterly surprise features: as-of on available_on (not STATPERS).
EVENT_FEATURES = ("sue_lag1", "sue_lag2", "beat_rate_4q")
STEM_IBES = "ibes"
ROW_KIND_COLUMN = "row_kind"
ROW_KIND_STATPERS = "statpers"
ROW_KIND_EVENT = "event"

QUERY_FINGERPRINT = "ibes.statsum_epsus|qtr_fpi6+ltg|v9_unified_rows"


def split_ibes_features(wanted: list[str]) -> tuple[list[str], list[str]]:
    """Split into STATPERS snapshot features vs event (SUE/beat) features."""
    snap = [f for f in wanted if f not in EVENT_FEATURES]
    event = [f for f in wanted if f in EVENT_FEATURES]
    return snap, event


def _connect_username() -> str | None:
    try:
        from script.export_universe import WRDS_USERNAME
    except Exception:  # noqa: BLE001
        return None
    return WRDS_USERNAME


def actual_pit_series(
    actual: pd.Series,
    anndats_act: pd.Series,
    statpers: pd.Series,
) -> pd.Series:
    """Point-in-time actual: only if announcement date is known and ≤ STATPERS.

    WRDS often backfills ``actual`` onto pre-announcement STATPERS rows; without
    this mask, SUE / beat would use non-public EPS at ``asof_date = statpers``.
    """
    act = pd.to_numeric(actual, errors="coerce")
    ann = pd.to_datetime(anndats_act, errors="coerce").dt.normalize()
    sp = pd.to_datetime(statpers, errors="coerce").dt.normalize()
    return act.where(ann.notna() & (ann <= sp))


def build_quarter_earnings_events(mapped: pd.DataFrame) -> pd.DataFrame:
    """One row per ``(TickerId, fpedats)`` earnings event with SUE / beat.

    Consensus / stdev taken from the last STATPERS with ``statpers < anndats_act``
    (strictly before announcement when no intraday timestamp is available).
    Panel availability is ``available_on = anndats_act + 1`` calendar day.
    Requires: ``TickerId``, ``statpers``, ``fpedats``, ``actual``, ``anndats_act``,
    ``eps_consensus``, ``stdev``.
    """
    need = [
        "TickerId",
        "statpers",
        "fpedats",
        "actual",
        "anndats_act",
        "eps_consensus",
        "stdev",
    ]
    missing = [c for c in need if c not in mapped.columns]
    if missing:
        raise ValueError(f"build_quarter_earnings_events missing columns: {missing}")

    rows: list[dict[str, object]] = []
    base = mapped.loc[:, need].copy()
    base["statpers"] = pd.to_datetime(base["statpers"]).dt.normalize()
    base["fpedats"] = pd.to_datetime(base["fpedats"], errors="coerce").dt.normalize()
    base["anndats_act"] = pd.to_datetime(
        base["anndats_act"], errors="coerce"
    ).dt.normalize()
    base["actual"] = pd.to_numeric(base["actual"], errors="coerce")
    base["eps_consensus"] = pd.to_numeric(base["eps_consensus"], errors="coerce")
    base["stdev"] = pd.to_numeric(base["stdev"], errors="coerce")
    base = base.dropna(subset=["TickerId", "fpedats"])

    for (ticker_id, fpedats), grp in base.groupby(["TickerId", "fpedats"], sort=False):
        ann = grp["anndats_act"].dropna()
        act = grp["actual"].dropna()
        if ann.empty or act.empty:
            continue
        anndats_act = ann.iloc[0]
        actual = float(act.iloc[0])
        pre = grp.loc[grp["statpers"] < anndats_act].sort_values("statpers")
        if pre.empty:
            continue
        last = pre.iloc[-1]
        consensus = last["eps_consensus"]
        stdev = last["stdev"]
        if pd.isna(consensus) or pd.isna(stdev) or float(stdev) == 0.0:
            sue = np.nan
        else:
            sue = (actual - float(consensus)) / float(stdev)
        if pd.isna(consensus):
            beat = np.nan
        else:
            beat = float(actual > float(consensus))
        rows.append(
            {
                "TickerId": ticker_id,
                "fpedats": fpedats,
                "anndats_act": anndats_act,
                # No intraday timestamp: features become usable the next calendar day.
                "available_on": pd.Timestamp(anndats_act) + pd.DateOffset(days=1),
                "_sue": sue,
                "_beat": beat,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "TickerId",
                "fpedats",
                "anndats_act",
                "available_on",
                "sue_lag1",
                "sue_lag2",
                "beat_rate_4q",
            ]
        )

    events = pd.DataFrame(rows)
    events = events.sort_values(["TickerId", "anndats_act"]).reset_index(drop=True)
    g = events.groupby("TickerId", sort=False)
    # lag1 = most recently announced SUE; lag2 = prior announced SUE.
    # (Do not shift before as-of: that made features one quarter too stale.)
    events["sue_lag1"] = events["_sue"]
    events["sue_lag2"] = g["_sue"].shift(1)
    events["beat_rate_4q"] = g["_beat"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )
    return events.loc[
        :,
        [
            "TickerId",
            "fpedats",
            "anndats_act",
            "available_on",
            "sue_lag1",
            "sue_lag2",
            "beat_rate_4q",
        ],
    ]


def attach_quarter_event_features(
    panel: pd.DataFrame, events: pd.DataFrame
) -> pd.DataFrame:
    """Backward as-of attach of event ``sue_lag*`` / ``beat_rate_4q`` onto panel.

    Uses ``available_on`` (= ``anndats_act + 1`` calendar day) so announcement-day
    STATPERS rows do not see the new SUE/beat. Requires panel ``TickerId``,
    ``statpers``. Mutates and returns ``panel``.
    """
    feat_cols = ["sue_lag1", "sue_lag2", "beat_rate_4q"]
    for col in feat_cols:
        panel[col] = np.nan
    if events is None or events.empty:
        return panel

    left = panel.loc[:, ["TickerId", "statpers"]].copy()
    left["_row"] = np.arange(len(panel))
    left["TickerId"] = pd.to_numeric(left["TickerId"], errors="coerce")
    left["statpers"] = pd.to_datetime(left["statpers"], errors="coerce").dt.normalize()
    left = left.dropna(subset=["TickerId", "statpers"])

    right = events.loc[:, ["TickerId", "available_on", *feat_cols]].copy()
    right["TickerId"] = pd.to_numeric(right["TickerId"], errors="coerce")
    right["available_on"] = pd.to_datetime(
        right["available_on"], errors="coerce"
    ).dt.normalize()
    right = right.dropna(subset=["TickerId", "available_on"])
    right = right.sort_values(["TickerId", "available_on"], kind="mergesort")

    pieces: list[pd.DataFrame] = []
    right_groups = {tid: g for tid, g in right.groupby("TickerId", sort=False)}
    for tid, left_g in left.groupby("TickerId", sort=False):
        left_g = left_g.sort_values("statpers", kind="mergesort")
        right_g = right_groups.get(tid)
        if right_g is None or right_g.empty:
            continue
        pieces.append(
            pd.merge_asof(
                left_g,
                right_g.drop(columns=["TickerId"]),
                left_on="statpers",
                right_on="available_on",
                direction="backward",
            )
        )
    if not pieces:
        return panel

    merged = pd.concat(pieces, ignore_index=True).sort_values("_row")
    idx = merged["_row"].to_numpy(dtype=int)
    for col in feat_cols:
        vals = np.full(len(panel), np.nan, dtype=float)
        vals[idx] = merged[col].to_numpy(dtype=float)
        panel[col] = vals
    return panel


def add_actual_derived_features(mapped: pd.DataFrame) -> pd.DataFrame:
    """Build quarter events from ``mapped`` snapshots and attach onto the same frame.

    Prefer building events from ``(TickerId, fpedats, statpers)`` snapshots, then
    calling ``attach_quarter_event_features`` on the panel grain.
    """
    events = build_quarter_earnings_events(mapped)
    return attach_quarter_event_features(mapped, events)


def _download_ibes_frame(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    username: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(statpers_panel, sue_event_panel)``.

    Snapshot panel uses ``asof_date = statpers`` (consensus / revisions).
    Event panel uses ``asof_date = available_on`` for direct daily as-of of SUE/beat.
    """
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
            SELECT ticker, oftic, measure, fiscalp, fpi, statpers, fpedats,
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
    stats["fpedats"] = pd.to_datetime(stats["fpedats"], errors="coerce").dt.normalize()
    stats["anndats_act"] = pd.to_datetime(
        stats["anndats_act"], errors="coerce"
    ).dt.normalize()
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

    # Keep one forecast snapshot per fiscal period × STATPERS for event SUE.
    snapshots = mapped.sort_values(
        ["TickerId", "fpedats", "statpers"]
    ).drop_duplicates(subset=["TickerId", "fpedats", "statpers"], keep="last")

    snapshots["eps_consensus"] = snapshots["medest"].where(
        snapshots["medest"].notna(), snapshots["meanest"]
    )
    snapshots["eps_dispersion"] = snapshots["stdev"]
    snapshots["eps_coverage"] = snapshots["numest"]

    events = build_quarter_earnings_events(snapshots)

    # Panel grain for cache / as-of attach: one row per TickerId × STATPERS.
    panel = snapshots.sort_values(["TickerId", "statpers"]).drop_duplicates(
        subset=["TickerId", "statpers"], keep="last"
    )
    # Filled at attach time as (Date - AsOfDate).days; placeholder in cache.
    panel["days_since_est_update"] = np.nan

    g = panel.groupby("TickerId", sort=False)
    panel["eps_revision_1m"] = panel["eps_consensus"] - g["eps_consensus"].shift(1)
    panel["eps_revision_3m"] = panel["eps_consensus"] - g["eps_consensus"].shift(3)
    panel["eps_consensus_chg_std"] = g["eps_consensus"].transform(
        lambda s: s.diff().rolling(6, min_periods=3).std()
    )

    # SUE/beat live on the event cache (available_on), not STATPERS rows.
    for col in EVENT_FEATURES:
        panel[col] = np.nan
    # Revenue surprise not in EPS summary; leave NaN unless extended later.
    panel["rev_surprise_lag1"] = np.nan

    panel["ltg"] = np.nan
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
        panel = panel.drop(columns=["ltg"], errors="ignore")
        panel = panel.merge(
            ltg_m.loc[:, ["TickerId", "statpers", "ltg"]],
            on=["TickerId", "statpers"],
            how="left",
        )

    snap_out = panel.rename(columns={"statpers": ASOF_COLUMN})
    snap_cols = [TICKER_COLUMN, ASOF_COLUMN, *ALL_IBES]
    for col in ALL_IBES:
        if col not in snap_out.columns:
            snap_out[col] = np.nan
    snap_out = snap_out.loc[:, snap_cols].reset_index(drop=True)

    if events.empty:
        event_out = pd.DataFrame(
            columns=[TICKER_COLUMN, ASOF_COLUMN, *EVENT_FEATURES]
        )
    else:
        event_out = events.rename(columns={"available_on": ASOF_COLUMN})
        for col in EVENT_FEATURES:
            if col not in event_out.columns:
                event_out[col] = np.nan
        event_out = event_out.loc[
            :, [TICKER_COLUMN, ASOF_COLUMN, *EVENT_FEATURES]
        ].reset_index(drop=True)

    return snap_out, event_out


def _align_ibes_cache_rows(
    snap_frame: pd.DataFrame, event_frame: pd.DataFrame, wanted: list[str]
) -> pd.DataFrame:
    """Stack STATPERS + event rows with ``row_kind`` for a unified IBES cache."""
    pieces: list[pd.DataFrame] = []
    if len(snap_frame):
        snap = snap_frame.copy()
        snap[ROW_KIND_COLUMN] = ROW_KIND_STATPERS
        pieces.append(snap)
    if len(event_frame):
        event = event_frame.copy()
        for col in wanted:
            if col not in event.columns:
                event[col] = np.nan
        event[ROW_KIND_COLUMN] = ROW_KIND_EVENT
        pieces.append(event)
    if not pieces:
        return pd.DataFrame(
            columns=[TICKER_COLUMN, ASOF_COLUMN, *wanted, ROW_KIND_COLUMN]
        )
    out = pd.concat(pieces, ignore_index=True, sort=False)
    for col in wanted:
        if col not in out.columns:
            out[col] = np.nan
    return out.loc[:, [TICKER_COLUMN, ASOF_COLUMN, *wanted, ROW_KIND_COLUMN]]


def _remove_legacy_ibes_sue_cache(folder: Path) -> None:
    for path in cache_paths(folder, "ibes_sue"):
        if path.exists():
            path.unlink()


def fetch_ibes_cache(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    ibes_features: list[str] | None = None,
    username: str | None = None,
) -> pd.DataFrame:
    """Download from WRDS and write unified ``ibes`` cache (statpers + event rows)."""
    wanted = list(IBES_FEATURES if ibes_features is None else ibes_features)
    if not wanted:
        raise ValueError("IBES_FEATURES is empty; nothing to fetch.")
    unknown = [n for n in wanted if n not in ALL_IBES]
    if unknown:
        raise ValueError(f"Unknown IBES_FEATURES {unknown}; allowed: {list(ALL_IBES)}")

    universe = load_universe(folder)
    uhash = universe_hash(universe.get("permno", universe["TickerId"]).tolist())
    snap_frame, event_frame = _download_ibes_frame(
        folder, start, end, username=username
    )
    unified = _align_ibes_cache_rows(snap_frame, event_frame, wanted)
    if unified.empty:
        raise RuntimeError("IBES download produced no cache rows.")

    write_feature_cache(
        folder,
        STEM_IBES,
        unified,
        features=wanted,
        universe_hash_value=uhash,
        query_fingerprint=QUERY_FINGERPRINT,
        extra_columns=[ROW_KIND_COLUMN],
        extra_meta={"row_kinds": [ROW_KIND_STATPERS, ROW_KIND_EVENT]},
    )
    _remove_legacy_ibes_sue_cache(folder)
    return load_cached_or_raise(
        folder,
        STEM_IBES,
        features=wanted,
        start=start,
        end=end,
        extra_columns=[ROW_KIND_COLUMN],
    )


def load_ibes(
    folder: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    ibes_features: list[str] | None = None,
) -> pd.DataFrame:
    """Read IBES STATPERS snapshot rows from the unified cache."""
    wanted = list(IBES_FEATURES if ibes_features is None else ibes_features)
    snap_feats, _event_feats = split_ibes_features(wanted)
    if not snap_feats:
        return pd.DataFrame(columns=[TICKER_COLUMN, ASOF_COLUMN])
    unknown = [n for n in snap_feats if n not in ALL_IBES]
    if unknown:
        raise ValueError(f"Unknown IBES_FEATURES {unknown}; allowed: {list(ALL_IBES)}")
    cache = load_cached_or_raise(
        folder,
        STEM_IBES,
        features=snap_feats,
        start=start,
        end=end,
        extra_columns=[ROW_KIND_COLUMN],
    )
    return cache.loc[cache[ROW_KIND_COLUMN] == ROW_KIND_STATPERS].drop(
        columns=[ROW_KIND_COLUMN]
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
    snap_feats, event_feats = split_ibes_features(wanted)
    out = frame.copy()

    cache = load_cached_or_raise(
        folder,
        STEM_IBES,
        features=wanted,
        start=dates.min(),
        end=dates.max(),
        extra_columns=[ROW_KIND_COLUMN],
    )

    if snap_feats:
        snap_cache = cache.loc[cache[ROW_KIND_COLUMN] == ROW_KIND_STATPERS].drop(
            columns=[ROW_KIND_COLUMN]
        )
        age = (
            "days_since_est_update"
            if "days_since_est_update" in snap_feats
            else None
        )
        out = asof_attach(
            out,
            snap_cache,
            features=snap_feats,
            date_column="Date",
            age_feature=age,
            desc="ibes",
        )

    if event_feats:
        event_cache = cache.loc[cache[ROW_KIND_COLUMN] == ROW_KIND_EVENT].drop(
            columns=[ROW_KIND_COLUMN]
        )
        if event_cache.empty:
            for col in event_feats:
                out[col] = np.nan
        else:
            out = asof_attach(
                out,
                event_cache,
                features=event_feats,
                date_column="Date",
                desc="ibes_sue",
            )
    return out
