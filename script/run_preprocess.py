"""Build the event table, run train-only preprocessing, and print a summary.

This script reads ONLY the [CONFIG] block below. Later main_lin / main_tr /
main_nn will read the matching file under config/ instead.

Usage (from repo root):
    python script/run_preprocess.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.features import FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS
from src.features import build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import PreprocessResult, run_preprocess
from src.report import _column_summary

# ============================== [CONFIG] ==============================

# Prediction target on event table df.
# Allowed: "ret_ann" | "ret_ann_mkt" | "ret_next" | "ret_next_mkt"
#          "ret_post" | "ret_post_mkt" | "dir_ann" | "vol_ann" | "vol_post"
TARGET = "ret_ann_mkt"

# Feature list. None uses src.features.FEATURE_COLUMNS.
# Allowed: None | non-empty subset of DEFAULT_FEATURE_COLUMNS
FEATURE_COLUMNS = None

# Chronological split column.
# Allowed: "EventDate" | "decision_session" | "anchor_session" | "release_session"
SPLIT_COLUMN = "EventDate"

# Inclusive train end as "YYYY-MM-DD".
# train: <= TRAIN_END
TRAIN_END = "2022-12-31"

# Inclusive validation end as "YYYY-MM-DD", or None for Train/Test only.
# val: TRAIN_END < d <= VAL_END; test: > VAL_END (or > TRAIN_END if None)
# Require VAL_END > TRAIN_END when set.
VAL_END = "2023-12-31"

# Feature missing values (statistics fit on train only).
# Allowed: "median" | "mean" | "drop" | None
#   median/mean: fill with train statistic
#   drop: drop rows with any feature NaN
#   None: keep NaNs (for tree models)
IMPUTE_STRATEGY = "median"

# Z-score features with train mean/std. Allowed: True | False
STANDARDIZE = True

# ======================================================================


def _configure_display() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)
    pd.set_option("display.expand_frame_repr", False)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")


def _section(title: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)


def _resolved_features() -> list[str]:
    if FEATURE_COLUMNS is None:
        return list(DEFAULT_FEATURE_COLUMNS)
    return list(FEATURE_COLUMNS)


def _print_config(features: list[str]) -> None:
    _section("preprocess config")
    print(f"TARGET            {TARGET}")
    print(f"FEATURE_COLUMNS   ({len(features)})")
    for name in features:
        print(f"                  - {name}")
    print(f"SPLIT_COLUMN      {SPLIT_COLUMN}")
    print(f"TRAIN_END         {TRAIN_END}")
    print(f"VAL_END           {VAL_END}")
    print(f"IMPUTE_STRATEGY   {IMPUTE_STRATEGY}")
    print(f"STANDARDIZE       {STANDARDIZE}")


def _target_summary_table(result: PreprocessResult) -> pd.DataFrame:
    rows: list[dict] = []
    for name, fold in result.folds.items():
        y = fold.y
        rows.append(
            {
                "fold": name,
                "count": int(y.notna().sum()),
                "unique": int(y.nunique(dropna=True)),
                "min": float(y.min()) if len(y) else float("nan"),
                "max": float(y.max()) if len(y) else float("nan"),
                "mean": float(y.mean()) if len(y) else float("nan"),
                "std": float(y.std(ddof=0)) if len(y) else float("nan"),
                "missing_n": int(y.isna().sum()),
                "missing_rate": f"{y.isna().mean():.2%}" if len(fold.frame) else "n/a",
            }
        )
    return pd.DataFrame(rows).set_index("fold")


def _print_target_summary(result: PreprocessResult) -> None:
    _section("target summary")
    print(
        f"target={TARGET}  events_in={result.n_before_target_drop}  "
        f"dropped_target={result.n_dropped_target}"
    )
    print(_target_summary_table(result))


def _print_feature_summary(result: PreprocessResult) -> None:
    _section("feature summary")
    print(
        f"impute={result.fitted.impute_strategy}  "
        f"standardize={result.fitted.standardize}  "
        f"n_features={len(result.fitted.feature_columns)}"
    )
    for name, fold in result.folds.items():
        print()
        print(f"[{name}] rows={len(fold.X)}")
        print(_column_summary(fold.X))


def main() -> None:
    _configure_display()
    features = _resolved_features()
    _print_config(features)

    stock_data = load_stock()
    earnings = load_earnings()
    historical_sentiment_scores = load_sentiment()
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    df = build_event_table(df_all)

    result = run_preprocess(
        df,
        target=TARGET,
        feature_columns=FEATURE_COLUMNS,
        split_column=SPLIT_COLUMN,
        train_end=TRAIN_END,
        val_end=VAL_END,
        impute_strategy=IMPUTE_STRATEGY,
        standardize=STANDARDIZE,
    )

    _section("train X head")
    print(result.folds["train"].X.head())

    _print_target_summary(result)
    _print_feature_summary(result)


if __name__ == "__main__":
    main()
