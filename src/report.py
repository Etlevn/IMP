"""Terminal descriptive summaries for the event training table."""

from __future__ import annotations

import pandas as pd

from src.features import FEATURE_COLUMNS

TARGET_COLUMNS = [
    "ret_ann",
    "ret_ann_mkt",
    "ret_next",
    "ret_next_mkt",
    "ret_post",
    "ret_post_mkt",
    "dir_ann",
    "vol_ann",
    "vol_post",
]
PRIMARY_TARGET = "ret_ann_mkt"
DATE_COLUMNS = (
    "EventDate",
    "release_session",
    "anchor_session",
    "decision_session",
)


def _configure_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def _section(title: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def _day(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%d")


def _for_print(df: pd.DataFrame) -> pd.DataFrame:
    """Copy for terminal display: dates as YYYY-MM-DD, cleaner integer ids."""
    out = df.copy()
    for column in DATE_COLUMNS:
        if column in out.columns:
            out[column] = pd.to_datetime(out[column]).dt.strftime("%Y-%m-%d")
    if "MarketTimeId" in out.columns:
        out["MarketTimeId"] = pd.to_numeric(out["MarketTimeId"], errors="coerce").astype("Int64")
    if "Id" in out.columns:
        out["Id"] = pd.to_numeric(out["Id"], errors="coerce").astype("Int64")
    if "dir_ann" in out.columns:
        out["dir_ann"] = pd.to_numeric(out["dir_ann"], errors="coerce").astype("Int64")
    return out


def _coverage(df: pd.DataFrame) -> None:
    print(f"rows={len(df)}  cols={df.shape[1]}  tickers={df['TickerId'].nunique()}")
    print(f"EventDate        {_day(df['EventDate'].min())} .. {_day(df['EventDate'].max())}")
    print(
        f"decision_session {_day(df['decision_session'].min())} .. "
        f"{_day(df['decision_session'].max())}"
    )
    if "MarketTimeId" in df.columns:
        counts = (
            pd.to_numeric(df["MarketTimeId"], errors="coerce")
            .astype("Int64")
            .value_counts(dropna=False)
            .sort_index()
        )
        print(counts.to_string())


def _as_day(value):
    """Format timestamps as YYYY-MM-DD; leave other values unchanged."""
    if value is pd.NA or value is None or (isinstance(value, float) and pd.isna(value)):
        return value
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError):
        return value
    if pd.isna(ts):
        return value
    return ts.strftime("%Y-%m-%d")


def _column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compact column stats: count/unique/min/max/mean/std/dtype + missingness."""
    desc = df.describe(include="all").T
    summary = pd.DataFrame(index=df.columns)
    summary["count"] = desc["count"] if "count" in desc.columns else df.count()
    summary["unique"] = df.nunique(dropna=True)
    for column in ("min", "max", "mean", "std"):
        summary[column] = desc[column] if column in desc.columns else pd.NA
    date_cols = [c for c in DATE_COLUMNS if c in summary.index]
    if date_cols:
        for column in ("min", "max", "mean"):
            summary.loc[date_cols, column] = summary.loc[date_cols, column].map(_as_day)
        summary.loc[date_cols, "std"] = pd.NA
    summary["dtype"] = df.dtypes.astype(str)
    summary["missing_n"] = df.isna().sum()
    summary["missing_rate"] = df.isna().mean().map(lambda x: f"{x:.2%}")
    return summary


def _corr_with_target(df: pd.DataFrame, target: str) -> None:
    cols = [c for c in FEATURE_COLUMNS + TARGET_COLUMNS if c in df.columns and c != target]
    if target not in df.columns:
        print(f"{target} not in df")
        return
    pair = df[[target] + cols].apply(pd.to_numeric, errors="coerce")
    corr = (
        pair.corr(method="pearson")[target]
        .drop(labels=[target])
        .sort_index(key=lambda idx: idx.str.lower())
    )
    print(f"pearson vs {target}")
    print(corr.round(4).to_string())


def print_event_report(df: pd.DataFrame, primary_target: str = PRIMARY_TARGET) -> None:
    """Print coverage, head, compact describe, correlations, and target summary."""
    _configure_display()
    view = _for_print(df)

    _section("df coverage")
    _coverage(df)

    _section("df head")
    print(view.head())

    _section("df describe")
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    print(_column_summary(df[feature_cols]))

    _section(f"df correlations ({primary_target})")
    _corr_with_target(df, primary_target)

    _section("df target summary")
    present = sorted(
        [c for c in TARGET_COLUMNS if c in df.columns],
        key=str.lower,
    )
    print(_column_summary(df[present]))
