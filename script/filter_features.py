"""Step-1 filter screen for ``config.default.FEATURE_COLUMNS``.

Builds the event table, keeps the train fold only, and prints per-feature
missingness, dispersion, target IC (Pearson / Spearman), and max pairwise
correlation with other candidates. Prints a one-line header then the table.

Usage (from repo root):
    python script/filter_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from config.default import (
    BENCHMARK_SOURCE,
    DATA_DIR,
    FEATURE_COLUMNS,
    MARKET_TICKER,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.features import build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import chronological_split, drop_missing_target


def _pairwise_ic(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Pearson and Spearman IC on pairwise-complete rows."""
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.dropna()
    n = int(len(frame))
    if n < 3:
        return float("nan"), float("nan"), n
    pearson = float(frame["x"].corr(frame["y"], method="pearson"))
    spearman = float(frame["x"].corr(frame["y"], method="spearman"))
    return pearson, spearman, n


def filter_table(train: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    y = pd.to_numeric(train[target], errors="coerce")
    rows: list[dict[str, object]] = []
    numeric = {
        name: pd.to_numeric(train[name], errors="coerce") for name in features
    }

    corr = pd.DataFrame({k: v for k, v in numeric.items()}).corr(method="pearson")

    for name in features:
        x = numeric[name]
        missing_n = int(x.isna().sum())
        n = int(x.notna().sum())
        pearson, spearman, n_ic = _pairwise_ic(x, y)
        others = [c for c in features if c != name]
        if others and name in corr.columns:
            max_abs_other = float(corr.loc[name, others].abs().max())
            max_abs_other_name = str(corr.loc[name, others].abs().idxmax())
        else:
            max_abs_other = float("nan")
            max_abs_other_name = ""

        rows.append(
            {
                "feature": name,
                "n": n,
                "missing_n": missing_n,
                "missing_rate": missing_n / len(train) if len(train) else float("nan"),
                "mean": float(x.mean()) if n else float("nan"),
                "std": float(x.std(ddof=0)) if n else float("nan"),
                "pearson_ic": pearson,
                "spearman_ic": spearman,
                "abs_pearson_ic": abs(pearson) if np.isfinite(pearson) else float("nan"),
                "n_ic": n_ic,
                "max_abs_corr_other": max_abs_other,
                "max_corr_with": max_abs_other_name,
            }
        )

    out = pd.DataFrame(rows)
    out["_rank"] = out["spearman_ic"].abs()
    out = out.sort_values(
        ["abs_pearson_ic", "_rank", "feature"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns="_rank")
    return out.reset_index(drop=True)


def main() -> None:
    features = list(FEATURE_COLUMNS)
    if not features:
        raise ValueError("FEATURE_COLUMNS is empty in config.default.")

    stock_data = load_stock(DATA_DIR)
    earnings = load_earnings(DATA_DIR)
    sentiment = load_sentiment(DATA_DIR)
    df_all = merge_panel(stock_data, earnings, sentiment)
    df = build_event_table(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_DIR,
        feature_columns=features,
        benchmark_source=BENCHMARK_SOURCE,
    )
    df, n_dropped = drop_missing_target(df, TARGET)
    train = chronological_split(df, SPLIT_COLUMN, TRAIN_END, VAL_END)["train"]
    table = filter_table(train, features, TARGET)

    print(f"FILTER | target={TARGET}  split={SPLIT_COLUMN}<={TRAIN_END}  "
          f"n_train={len(train)}  features={len(features)}  dropped_target={n_dropped}\n")
    display = table.copy()
    display["missing_rate"] = display["missing_rate"].map(lambda v: f"{100 * v:.2f}%")
    for col in ("mean", "std", "pearson_ic", "spearman_ic", "abs_pearson_ic", "max_abs_corr_other"):
        display[col] = display[col].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    print(
        display.loc[
            :,
            [
                "feature",
                "n",
                "missing_rate",
                "std",
                "pearson_ic",
                "spearman_ic",
                "abs_pearson_ic",
                "max_abs_corr_other",
                "max_corr_with",
            ],
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
