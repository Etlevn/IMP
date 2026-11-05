"""Train-fold screen for the feature list in this file.

Builds the event table, keeps the train fold only, and writes per-feature
missingness, dispersion, target IC (Pearson / Spearman), and max pairwise
correlation with other candidates to ``output/filter_feat.csv``.

Only ``FEATURES`` below is screened. Target, split, and data paths still come
from ``config/default.py``. A name other than the original eight must also be
listed in ``ANALYST_FEATURES``, ``MACRO_FEATURES``, or ``ENGINEERED_FEATURES``
so the panel actually builds that column.

The terminal screen is a label only. It uses the train fold, and
``ic_noise_2se = 2 / sqrt(n_train)``. Mean is not used. Priority is drop,
then redundant, then keep or weak. A feature is drop when its missing rate
exceeds ``MISSING_DROP`` or its standard deviation is non-finite or zero; that
line is printed only when non-empty, with the reason in parentheses. It is
redundant when its strongest absolute correlation with another candidate is at
least ``CORR_REDUNDANT`` and its ``|pearson_ic|`` is strictly lower than that
partner's; the line is ``loser→partner``. Equal absolute Pearson is not
redundant, and the comparison follows ``max_corr_with`` only. Otherwise it is
keep when Pearson and Spearman have the same sign and at least one absolute
IC is at least the noise line; everything else is weak, including opposite
signs and a zero IC. Each group stays in descending ``|pearson_ic|`` order.

Usage (from repo root):
    python filter_feat.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.default import (
    BENCHMARK_SOURCE,
    DATA_DIR,
    MARKET_TICKER,
    OUTPUT_DIR,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.events import build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import chronological_split, drop_missing_target

# ======================================== [CONFIG] ========================================
# Candidates to screen. Comment out a line to drop it from this run.
FEATURES: tuple[str, ...] = (
    "QScoreTrendLong",
    "QScoreSpeedLong",
    "QScoreLong",
    "analystRatingsStrongBuy",
    "analystRatingsBuy",
    "analystRatingsHold",
    "analystRatingsSell",
    "analystRatingsStrongSell",
    "Coverage",
    "BuyRatio",
    "SellRatio",
    "RatingScore",
    "mom_21d",
    "vol_21d",
    "log_mcap",
    "vix",
    "tnx",
    "dxy",
    "par_mkt",
)

# Missing rate above this is labeled drop. Allowed: float in [0, 1].
MISSING_DROP = 0.20

# Absolute correlation at or above this, and a weaker |Pearson IC|, is redundant.Allowed: float in [0, 1].
CORR_REDUNDANT = 0.60
# ========================================================================================

_ORIGINAL_COLUMNS = (
    "QScoreTrendLong",
    "QScoreSpeedLong",
    "QScoreLong",
    "analystRatingsStrongBuy",
    "analystRatingsBuy",
    "analystRatingsHold",
    "analystRatingsSell",
    "analystRatingsStrongSell",
)


def _available_columns() -> set[str]:
    from config.default import ANALYST_FEATURES, ENGINEERED_FEATURES, MACRO_FEATURES

    return (
        set(_ORIGINAL_COLUMNS)
        | set(ANALYST_FEATURES)
        | set(MACRO_FEATURES)
        | set(ENGINEERED_FEATURES)
    )


def _features() -> list[str]:
    features = list(FEATURES)
    if not features:
        raise ValueError("FEATURES is empty.")
    duplicates = [name for name in features if features.count(name) > 1]
    if duplicates:
        raise ValueError(f"FEATURES repeats {sorted(set(duplicates))}.")
    unknown = [name for name in features if name not in _available_columns()]
    if unknown:
        raise ValueError(
            f"Unknown FEATURES {unknown}. Original eight columns are always "
            "available. Other names must be listed in ANALYST_FEATURES, "
            "MACRO_FEATURES, or ENGINEERED_FEATURES."
        )
    return features


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


def _screen(table: pd.DataFrame, ic_noise: float) -> dict[str, list[str]]:
    """Label each feature keep, weak, redundant, or drop."""
    by_name = table.set_index("feature")
    grouped: dict[str, list[str]] = {
        "keep": [],
        "weak": [],
        "redundant": [],
        "drop": [],
    }
    for row in table.itertuples(index=False):
        name = str(row.feature)
        missing = float(row.missing_rate)
        std = float(row.std)
        pearson = float(row.pearson_ic)
        spearman = float(row.spearman_ic)
        abs_pearson = float(row.abs_pearson_ic)
        max_corr = float(row.max_abs_corr_other)
        partner = str(row.max_corr_with)

        if np.isfinite(missing) and missing > MISSING_DROP:
            grouped["drop"].append(f"{name} (missing {missing:.1%})")
            continue
        if not np.isfinite(std) or std <= 0:
            grouped["drop"].append(f"{name} (std {std:.4g})")
            continue

        partner_abs = float("nan")
        if partner in by_name.index:
            partner_abs = float(by_name.loc[partner, "abs_pearson_ic"])
        loses_pair = (
            np.isfinite(max_corr)
            and max_corr >= CORR_REDUNDANT
            and np.isfinite(abs_pearson)
            and np.isfinite(partner_abs)
            and abs_pearson < partner_abs
        )
        if loses_pair:
            grouped["redundant"].append(f"{name}→{partner}")
            continue

        abs_spearman = abs(spearman) if np.isfinite(spearman) else float("nan")
        same_sign = (
            np.isfinite(pearson)
            and np.isfinite(spearman)
            and pearson * spearman > 0
        )
        above_noise = (
            (np.isfinite(abs_pearson) and abs_pearson >= ic_noise)
            or (np.isfinite(abs_spearman) and abs_spearman >= ic_noise)
        )
        if same_sign and above_noise:
            grouped["keep"].append(name)
        else:
            grouped["weak"].append(name)
    return grouped


def _print_screen(grouped: dict[str, list[str]]) -> None:
    labels = ("keep", "weak", "redundant")
    if grouped["drop"]:
        labels = ("drop", *labels)
    width = max(len(label) for label in labels)
    for label in labels:
        names = ", ".join(grouped[label]) if grouped[label] else "—"
        print(f"{label:<{width}}  {names}")


def _print_table(table: pd.DataFrame) -> None:
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


def main() -> None:
    features = _features()

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
    n_train = len(train)
    ic_noise = 2 / np.sqrt(n_train) if n_train else float("nan")
    print(
        f"FILTER | target={TARGET}  split={SPLIT_COLUMN}<={TRAIN_END}  "
        f"n_train={n_train}  features={len(features)}  dropped_target={n_dropped}  "
        f"ic_noise_2se={ic_noise:.4f}"
    )
    print()
    _print_table(table)
    print()
    _print_screen(_screen(table, ic_noise))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_DIR / "filter_feat.csv", index=False)


if __name__ == "__main__":
    main()
