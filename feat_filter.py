"""Train-fold screen for ``src.features.enabled_features()``.

Builds the event table, keeps the train fold only, and writes per-feature
missingness, dispersion, target IC (Pearson / Spearman), max pairwise
correlation with other candidates, and a ``suggest`` label
(keep / weak / noise / redundant / drop) to ``output/feat_filter.csv``,
plus settings-only ``output/config_filter.json``.

Feature names come from the four config groups
(``ORIGINAL_FEATURES`` / ``ANALYST_FEATURES`` / ``MACRO_FEATURES`` /
``ENGINEERED_FEATURES``); comment out a name in its group to disable it.

``suggest`` uses the train fold and
``ic_noise = IC_NOISE_MULT / sqrt(n_train)``. Mean is not used. Priority is
drop, then redundant, then IC tiers.
Drop when missing rate exceeds ``MISSING_DROP`` or std is non-finite / zero.
Redundant when max absolute correlation is at least ``CORR_REDUNDANT`` and
``|pearson_ic|`` is strictly lower than the partner's. Otherwise, with Pearson
and Spearman the same sign: ``keep`` if both absolute ICs clear the noise
line, ``weak`` if exactly one does, ``noise`` if neither does. Opposite signs
(or non-finite ICs that block same-sign) are ``noise``. Rows stay in
descending ``|pearson_ic|`` order.

Usage (from repo root):
    python feat_filter.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import (
    BENCHMARK_SOURCE,
    DATA_DIR,
    EARNINGS_FILE,
    MARKET_TICKER,
    OUTPUT_DIR,
    SENTIMENT_FILE,
    SPLIT_COLUMN,
    STOCK_FILE,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.compact_json import write_compact_json
from src.events import build_event_table
from src.features import enabled_features
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import chronological_split, drop_missing_target

# ======================================== [CONFIG] ========================================

# Missing rate above this is labeled drop. Allowed: float in [0, 1].
MISSING_DROP = 0.20

# Absolute correlation at or above this, and a weaker |Pearson IC|, is redundant. Allowed: float in [0, 1].
CORR_REDUNDANT = 0.80

# IC noise floor multiplier: ic_noise = IC_NOISE_MULT / sqrt(n_train); (= MULT × SE, where SE ≈ 1/sqrt(n_train)). Allowed: float > 0.
IC_NOISE_MULT = 1.96

# ==========================================================================================

SUGGEST_LABELS = ("keep", "weak", "noise", "redundant", "drop")

# Terminal print and CSV share this column order.
TABLE_COLUMNS = (
    "feature",
    "suggest",
    "n",
    "missing_rate",
    "std",
    "pearson_ic",
    "spearman_ic",
    "max_abs_corr_other",
    "max_corr_with",
)


def _features() -> list[str]:
    features = enabled_features()
    if not features:
        raise ValueError(
            "enabled_features() is empty. Populate ORIGINAL_FEATURES / "
            "ANALYST_FEATURES / MACRO_FEATURES / ENGINEERED_FEATURES in "
            "config/config.py."
        )
    return features


def _pairwise_ic(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Pearson and Spearman IC on pairwise-complete rows."""
    frame = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")})
    frame = frame.dropna()
    n = int(len(frame))
    if n < 3:
        return float("nan"), float("nan"), n
    # Constant series → undefined correlation (avoids ConstantInputWarning).
    if float(frame["x"].std(ddof=0)) == 0.0 or float(frame["y"].std(ddof=0)) == 0.0:
        return float("nan"), float("nan"), n
    pearson = float(frame["x"].corr(frame["y"], method="pearson"))
    spearman = float(frame["x"].corr(frame["y"], method="spearman"))
    return pearson, spearman, n


def filter_table(train: pd.DataFrame, features: list[str], target: str) -> pd.DataFrame:
    y = pd.to_numeric(train[target], errors="coerce")
    rows: list[dict[str, object]] = []
    numeric = {}
    for name in features:
        series = pd.to_numeric(train[name], errors="coerce")
        # ±inf breaks mean/std/corr (RuntimeWarning); treat as missing.
        numeric[name] = series.where(np.isfinite(series), np.nan)

    # Skip all-NA / zero-variance columns so corr does not emit divide warnings.
    corr_cols = [
        name
        for name, series in numeric.items()
        if int(series.notna().sum()) >= 2 and float(series.std(ddof=0) or 0.0) > 0.0
    ]
    corr = (
        pd.DataFrame({k: numeric[k] for k in corr_cols}).corr(method="pearson")
        if corr_cols
        else pd.DataFrame()
    )

    for name in features:
        x = numeric[name]
        missing_n = int(x.isna().sum())
        n = int(x.notna().sum())
        pearson, spearman, n_ic = _pairwise_ic(x, y)
        others = [c for c in corr_cols if c != name]
        if others and name in corr.index:
            row_abs = corr.loc[name, others].abs()
            if row_abs.notna().any():
                max_abs_other = float(row_abs.max())
                max_abs_other_name = str(row_abs.idxmax())
            else:
                max_abs_other = float("nan")
                max_abs_other_name = ""
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
                "n_ic": n_ic,
                "max_abs_corr_other": max_abs_other,
                "max_corr_with": max_abs_other_name,
            }
        )

    out = pd.DataFrame(rows)
    out["_abs_pearson"] = out["pearson_ic"].abs()
    out["_abs_spearman"] = out["spearman_ic"].abs()
    out = out.sort_values(
        ["_abs_pearson", "_abs_spearman", "feature"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns=["_abs_pearson", "_abs_spearman"])
    return out.reset_index(drop=True)


def suggest_labels(table: pd.DataFrame, ic_noise: float) -> pd.Series:
    """Per-row keep / weak / noise / redundant / drop suggestion (aligned to ``table``)."""
    by_name = table.set_index("feature")
    labels: list[str] = []
    for row in table.itertuples(index=False):
        missing = float(row.missing_rate)
        std = float(row.std)
        pearson = float(row.pearson_ic)
        spearman = float(row.spearman_ic)
        abs_pearson = abs(pearson) if np.isfinite(pearson) else float("nan")
        max_corr = float(row.max_abs_corr_other)
        partner = str(row.max_corr_with)

        if np.isfinite(missing) and missing > MISSING_DROP:
            labels.append("drop")
            continue
        if not np.isfinite(std) or std <= 0:
            labels.append("drop")
            continue

        partner_abs = float("nan")
        if partner in by_name.index:
            partner_pearson = float(by_name.loc[partner, "pearson_ic"])
            partner_abs = (
                abs(partner_pearson) if np.isfinite(partner_pearson) else float("nan")
            )
        loses_pair = (
            np.isfinite(max_corr)
            and max_corr >= CORR_REDUNDANT
            and np.isfinite(abs_pearson)
            and np.isfinite(partner_abs)
            and abs_pearson < partner_abs
        )
        if loses_pair:
            labels.append("redundant")
            continue

        abs_spearman = abs(spearman) if np.isfinite(spearman) else float("nan")
        same_sign = (
            np.isfinite(pearson)
            and np.isfinite(spearman)
            and pearson * spearman > 0
        )
        pearson_hit = np.isfinite(abs_pearson) and abs_pearson >= ic_noise
        spearman_hit = np.isfinite(abs_spearman) and abs_spearman >= ic_noise
        if not same_sign:
            labels.append("noise")
        elif pearson_hit and spearman_hit:
            labels.append("keep")
        elif pearson_hit or spearman_hit:
            labels.append("weak")
        else:
            labels.append("noise")
    return pd.Series(labels, index=table.index, name="suggest")


def _print_table(table: pd.DataFrame) -> None:
    display = table.copy()
    display["missing_rate"] = display["missing_rate"].map(lambda v: f"{100 * v:.2f}%")
    for col in ("mean", "std", "pearson_ic", "spearman_ic", "max_abs_corr_other"):
        display[col] = display[col].map(lambda v: f"{v:.4f}" if np.isfinite(v) else "")
    print(display.loc[:, list(TABLE_COLUMNS)].to_string(index=False))


def _snapshot(
    *,
    features: list[str],
    n_train: int,
    n_dropped_target: int,
    ic_noise: float,
) -> dict:
    """Settings only: thresholds + shared data knobs (no per-feature rows)."""
    return {
        "missing_drop": float(MISSING_DROP),
        "corr_redundant": float(CORR_REDUNDANT),
        "ic_noise_mult": float(IC_NOISE_MULT),
        "ic_noise": float(ic_noise) if np.isfinite(ic_noise) else None,
        "n_train": int(n_train),
        "n_dropped_target": int(n_dropped_target),
        "suggest_labels": list(SUGGEST_LABELS),
        "table_columns": list(TABLE_COLUMNS),
        "feature_columns": list(features),
        "shared": {
            "TARGET": TARGET,
            "BENCHMARK_SOURCE": BENCHMARK_SOURCE,
            "MARKET_TICKER": MARKET_TICKER,
            "DATA_FILES": [STOCK_FILE, EARNINGS_FILE, SENTIMENT_FILE],
            "SPLIT_COLUMN": SPLIT_COLUMN,
            "TRAIN_END": TRAIN_END,
            "VAL_END": VAL_END,
        },
    }


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
    ic_noise = IC_NOISE_MULT / np.sqrt(n_train) if n_train else float("nan")
    table["suggest"] = suggest_labels(table, ic_noise)
    print(
        f"[FILTER] target={TARGET}  split={SPLIT_COLUMN}<={TRAIN_END}  "
        f"n_train={n_train}  features={len(features)}  dropped_target={n_dropped}  "
        f"ic_noise={ic_noise:.4f} (×{IC_NOISE_MULT:g}/√n)"
    )
    print()
    _print_table(table)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "feat_filter.csv"
    config_path = OUTPUT_DIR / "config_filter.json"
    table.loc[:, list(TABLE_COLUMNS)].to_csv(out_path, index=False, float_format="%.4f")
    write_compact_json(
        config_path,
        _snapshot(
            features=features,
            n_train=n_train,
            n_dropped_target=n_dropped,
            ic_noise=ic_noise,
        ),
    )
    counts = table["suggest"].value_counts()
    summary = "  ".join(f"{lab}={int(counts.get(lab, 0))}" for lab in SUGGEST_LABELS)
    print()
    print(f"[SUMMARY] {summary}")
    print(f"[WROTE] {out_path}")
    print(f"[WROTE] {config_path}")


if __name__ == "__main__":
    main()
