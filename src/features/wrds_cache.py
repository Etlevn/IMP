"""WRDS feature cache: fetch scripts write; attach only reads.

No network in the training / feature-attach path.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.prepare import _row_bar

ASOF_COLUMN = "asof_date"
TICKER_COLUMN = "TickerId"


def cache_paths(folder: Path, stem: str) -> tuple[Path, Path]:
    return folder / f"{stem}_features.parquet", folder / f"{stem}_features.json"


def universe_hash(ids: Sequence[Any]) -> str:
    payload = ",".join(str(x) for x in sorted({str(i) for i in ids if pd.notna(i)}))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def write_feature_cache(
    folder: Path,
    stem: str,
    frame: pd.DataFrame,
    *,
    features: Sequence[str],
    universe_hash_value: str,
    query_fingerprint: str,
    extra_meta: dict[str, Any] | None = None,
    extra_columns: Sequence[str] = (),
) -> tuple[Path, Path]:
    if frame is None or len(frame) == 0:
        raise RuntimeError(
            f"Refusing to write empty {stem} feature cache. "
            "WRDS pull returned no rows."
        )
    extras = list(extra_columns)
    required = {TICKER_COLUMN, ASOF_COLUMN, *features, *extras}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{stem} cache frame missing columns: {missing}")

    path, meta_path = cache_paths(folder, stem)
    folder.mkdir(parents=True, exist_ok=True)
    out = frame.copy()
    out[ASOF_COLUMN] = pd.to_datetime(out[ASOF_COLUMN]).dt.normalize()
    out = out.sort_values([TICKER_COLUMN, ASOF_COLUMN]).reset_index(drop=True)
    cols = [TICKER_COLUMN, ASOF_COLUMN, *list(features), *extras]
    out.loc[:, cols].to_parquet(path, index=False)

    meta: dict[str, Any] = {
        "source": "wrds",
        "stem": stem,
        "features": list(features),
        "extra_columns": extras,
        "universe_hash": universe_hash_value,
        "query_fingerprint": query_fingerprint,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "start": out[ASOF_COLUMN].min().strftime("%Y-%m-%d"),
        "end": out[ASOF_COLUMN].max().strftime("%Y-%m-%d"),
        "rows": int(len(out)),
        "tickers": int(out[TICKER_COLUMN].nunique()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta_path.write_text(json.dumps(meta, indent=2))
    return path, meta_path


def load_cached_or_raise(
    folder: Path,
    stem: str,
    *,
    features: Sequence[str],
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
    extra_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Attach-path loader: cache only. Tell user which fetch script to run."""
    wanted = list(features)
    extras = list(extra_columns)
    if not wanted and not extras:
        return pd.DataFrame(columns=[TICKER_COLUMN, ASOF_COLUMN])

    path, meta_path = cache_paths(folder, stem)
    script = f"script/fetch_wrds_{stem}.py"
    if not path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {stem} feature cache under {folder} "
            f"({path.name} / {meta_path.name}). Run: python {script}"
        )

    meta = json.loads(meta_path.read_text())
    cached_features = list(meta.get("features") or [])
    missing_feats = [f for f in wanted if f not in cached_features]
    if missing_feats:
        raise ValueError(
            f"{stem} cache meta missing features {missing_feats}. "
            f"Re-run: python {script}"
        )

    frame = pd.read_parquet(path)
    if len(frame) == 0:
        raise RuntimeError(f"{stem} feature cache is empty: {path}. Re-run: python {script}")

    need_cols = [TICKER_COLUMN, ASOF_COLUMN, *wanted, *extras]
    missing_cols = [c for c in need_cols if c not in frame.columns]
    if missing_cols:
        raise ValueError(f"{stem} parquet missing columns: {missing_cols}")

    out = frame.loc[:, need_cols].copy()
    out[ASOF_COLUMN] = pd.to_datetime(out[ASOF_COLUMN]).dt.normalize()
    out[TICKER_COLUMN] = pd.to_numeric(out[TICKER_COLUMN], errors="coerce")

    if start is not None or end is not None:
        lo = out[ASOF_COLUMN].min()
        hi = out[ASOF_COLUMN].max()
        if start is not None and hi < pd.Timestamp(start).normalize():
            raise RuntimeError(
                f"{stem} cache ends {hi:%Y-%m-%d}, before panel start "
                f"{pd.Timestamp(start):%Y-%m-%d}. Re-run: python {script}"
            )
        if end is not None and lo > pd.Timestamp(end).normalize():
            raise RuntimeError(
                f"{stem} cache starts {lo:%Y-%m-%d}, after panel end "
                f"{pd.Timestamp(end):%Y-%m-%d}. Re-run: python {script}"
            )

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if meta.get("sha256") and meta["sha256"] != digest:
        raise RuntimeError(
            f"{stem} cache sha256 mismatch. Re-run: python {script}"
        )

    return out.sort_values([TICKER_COLUMN, ASOF_COLUMN]).reset_index(drop=True)


def asof_attach(
    panel: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    features: Sequence[str],
    date_column: str = "Date",
    age_feature: str | None = None,
    desc: str = "asof",
) -> pd.DataFrame:
    """Backward as-of join of cache features onto the panel.

    If ``age_feature`` is set (e.g. ``days_since_est_update``), that column is
    computed as ``date_column - AsOfDate`` in calendar days after the merge,
    not taken from the cache (avoids a constant stub).
    """
    wanted = list(features)
    if not wanted:
        return panel.copy()
    if TICKER_COLUMN not in panel.columns or date_column not in panel.columns:
        raise KeyError(
            f"Panel needs {TICKER_COLUMN!r} and {date_column!r} for WRDS as-of attach."
        )

    cache_feats = [c for c in wanted if c != age_feature]
    out = panel.copy()
    out[date_column] = pd.to_datetime(out[date_column]).dt.normalize()
    out[TICKER_COLUMN] = pd.to_numeric(out[TICKER_COLUMN], errors="coerce")
    drop_cols = [c for c in wanted if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    right = cache.copy()
    right[ASOF_COLUMN] = pd.to_datetime(right[ASOF_COLUMN]).dt.normalize()
    right[TICKER_COLUMN] = pd.to_numeric(right[TICKER_COLUMN], errors="coerce")
    right = right.dropna(subset=[TICKER_COLUMN, ASOF_COLUMN])
    right = right.sort_values([TICKER_COLUMN, ASOF_COLUMN])
    missing_cache = [c for c in cache_feats if c not in right.columns]
    if missing_cache:
        for col in missing_cache:
            right[col] = np.nan

    pieces: list[pd.DataFrame] = []
    groups = list(out.groupby(TICKER_COLUMN, sort=False))
    with _row_bar(max(len(groups), 1), desc=desc) as bar:
        for ticker, left_g in groups:
            left_g = left_g.sort_values(date_column)
            right_g = right.loc[right[TICKER_COLUMN] == ticker]
            if right_g.empty:
                merged = left_g.copy()
                for col in wanted:
                    merged[col] = np.nan
                pieces.append(merged)
                bar.update(1)
                continue
            merged = pd.merge_asof(
                left_g,
                right_g.loc[:, [ASOF_COLUMN, *cache_feats]],
                left_on=date_column,
                right_on=ASOF_COLUMN,
                direction="backward",
            )
            if age_feature is not None and age_feature in wanted:
                merged[age_feature] = (
                    merged[date_column] - merged[ASOF_COLUMN]
                ).dt.days.astype(float)
            if ASOF_COLUMN in merged.columns:
                merged = merged.drop(columns=[ASOF_COLUMN])
            pieces.append(merged)
            bar.update(1)

    if not pieces:
        return out
    return pd.concat(pieces, axis=0, ignore_index=True)


def open_wrds(username: str | None = None):
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "wrds is required for IBES / Compustat fetch scripts. "
            "Install with: pip install wrds"
        ) from exc
    if username:
        return wrds.Connection(wrds_username=username)
    return wrds.Connection()


def load_universe(folder: Path) -> pd.DataFrame:
    path = folder / "stock_universe.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run python script/export_universe.py first "
            "(with USE_WRDS=True for permno/gvkey)."
        )
    universe = pd.read_csv(path, dtype={"gvkey": "string"})
    if "TickerId" not in universe.columns:
        raise ValueError(f"{path} missing TickerId column.")
    return universe
