"""Compare named feature sets on a fixed set of models.

Edit ``FEATURE_SETS`` and ``MODELS`` below, then run from the repo root:

    python main_feat.py

Target, split, impute, standardization, and model hyperparameters are read
the same way as ``main.py`` (``config/default.py`` and ``config/config_*.py``).
Writes ``output/metrics_feat.csv`` and ``output/run_config_feat.json``.
Does not overwrite ``metrics.csv`` or ``run_config.json``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.pipeline import run_family
from src.train import FamilyName, FamilySpec, get_family_spec

# ======================================== [CONFIG] ========================================

# metrics_feat.csv row order. Models stay in MODELS order and form separate groups. Within a model, higher values come first.
SORT_BY: tuple[str, ...] = ("oos_r2", "quintile_ls", "rank_ic")

# Fixed models. Comment out a family to skip it.
MODELS: dict[str, tuple[str, ...]] = {
    "lin": ("ols",),  # "ols" | "ridge" | "lasso"
    # "tr": ("random_forest",),  # decision_tree | random_forest | hist_gradient_boosting
    # "nn": ("mlp_deep",),  # mlp_shallow | mlp | mlp_deep
}

# Each name is one experiment. The same MODELS run on every set.
FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "orginal": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "QScoreLong",
        "analystRatingsStrongBuy",
        "analystRatingsBuy",
        "analystRatingsHold",
        "analystRatingsSell",
        "analystRatingsStrongSell",
    ),
    "base":(
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
    ),
    "base+eng": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
        "mom_21d",
        "vol_21d",
        "log_mcap",
    ),
    "base+macro": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
        "vix",
        "tnx",
        "dxy",
    ),
    "base+par": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
        "par_mkt",
    ),
    "par": ("par_mkt",),
    "base+eng+par": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
        "mom_21d",
        "vol_21d",
        "log_mcap",
        "par_mkt",
    ),
    "base+eng+par+par*vix": (
        "QScoreTrendLong",
        "QScoreSpeedLong",
        "Coverage",
        "RatingScore",
        "mom_21d",
        "vol_21d",
        "log_mcap",
        "par_mkt",
        "par*vix",
    ),
}

# ========================================================================================

_FAMILY_ORDER: tuple[FamilyName, ...] = ("lin", "tr", "nn")
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


def _validate_feature_sets() -> None:
    if not FEATURE_SETS:
        raise ValueError("FEATURE_SETS must name at least one feature combination.")
    available = _available_columns()
    for name, columns in FEATURE_SETS.items():
        if not str(name).strip():
            raise ValueError("FEATURE_SETS names must be non-empty.")
        if not columns:
            raise ValueError(f"Feature set {name!r} has no columns.")
        duplicates = [column for column in columns if columns.count(column) > 1]
        if duplicates:
            raise ValueError(f"Feature set {name!r} repeats {sorted(set(duplicates))}.")
        unknown = [column for column in columns if column not in available]
        if unknown:
            raise ValueError(
                f"Feature set {name!r} has unknown columns {unknown}. "
                "Original eight columns are always available. Other names must "
                "be listed in ANALYST_FEATURES, MACRO_FEATURES, or ENGINEERED_FEATURES."
            )


def _selected_specs() -> list[FamilySpec]:
    if not MODELS:
        raise ValueError("MODELS must include at least one of lin / tr / nn.")
    unknown = [name for name in MODELS if name not in _FAMILY_ORDER]
    if unknown:
        raise ValueError(f"Unknown model family {unknown!r}; expected lin / tr / nn.")

    specs: list[FamilySpec] = []
    for family in _FAMILY_ORDER:
        if family not in MODELS:
            continue
        selected = tuple(MODELS[family])
        if not selected:
            raise ValueError(f"MODELS[{family!r}] must list at least one sub-model.")
        spec = get_family_spec(family)
        allowed = set(spec.model_types)
        rejected = [name for name in selected if name not in allowed]
        if rejected:
            raise ValueError(
                f"Unknown {family} sub-model(s) {rejected}; allowed: {tuple(spec.model_types)}."
            )
        lines = tuple(
            (key, selected if key == spec.model_types_key else value)
            for key, value in spec.config_extra_lines
        )
        specs.append(replace(spec, model_types=selected, config_extra_lines=lines))
    return specs


def _model_order() -> list[str]:
    order: list[str] = []
    for family in _FAMILY_ORDER:
        if family not in MODELS:
            continue
        for name in MODELS[family]:
            if name not in order:
                order.append(name)
    return order


def _sort_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if not SORT_BY:
        raise ValueError("SORT_BY must list at least one metric.")
    missing = [column for column in SORT_BY if column not in metrics.columns]
    if missing:
        raise ValueError(f"SORT_BY columns missing from metrics: {missing}.")
    rank = {name: index for index, name in enumerate(_model_order())}
    out = metrics.copy()
    out["_model_order"] = out["model"].map(lambda name: rank.get(str(name), len(rank)))
    out = out.sort_values(
        ["_model_order", *SORT_BY],
        ascending=[True, *([False] * len(SORT_BY))],
        na_position="last",
        kind="mergesort",
    )
    return out.drop(columns="_model_order").reset_index(drop=True)


def _union_columns() -> list[str]:
    seen: list[str] = []
    for columns in FEATURE_SETS.values():
        for column in columns:
            if column not in seen:
                seen.append(column)
    return seen


def _load_events(columns: Sequence[str]) -> pd.DataFrame:
    from config.default import BENCHMARK_SOURCE, DATA_ROOT, MARKET_TICKER
    from src.events import build_aligned_panel, build_event_table
    from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel

    stock = load_stock(DATA_ROOT)
    earnings = load_earnings(DATA_ROOT)
    sentiment = load_sentiment(DATA_ROOT)
    df_all = merge_panel(stock, earnings, sentiment)
    df_align = build_aligned_panel(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_ROOT,
        benchmark_source=BENCHMARK_SOURCE,
    )
    return build_event_table(
        df_all,
        market_ticker=MARKET_TICKER,
        data_dir=DATA_ROOT,
        df_align=df_align,
        feature_columns=list(columns),
        benchmark_source=BENCHMARK_SOURCE,
    )


_SHARED_KEYS = (
    "TARGET",
    "BENCHMARK_SOURCE",
    "MARKET_TICKER",
    "DATA_ROOT",
    "SPLIT_COLUMN",
    "TRAIN_END",
    "VAL_END",
    "IMPUTE_STRATEGY",
    "RANDOM_STATE",
    "train_y_mean",
    "n_before_target_drop",
    "n_dropped_target",
)
_MODEL_TYPE_KEYS = {"LINEAR_MODEL_TYPES", "TREE_MODEL_TYPES", "NN_MODEL_TYPES"}


def _snapshot(family_configs: dict[str, dict]) -> dict:
    """One copy of the shared run settings, plus each feature set's columns."""
    first = next(iter(family_configs.values()))
    shared = {key: first[key] for key in _SHARED_KEYS}
    shared["folds"] = first["folds"]
    drop = set(_SHARED_KEYS) | _MODEL_TYPE_KEYS | {"family", "FEATURE_COLUMNS", "folds"}
    return {
        "feature_sets": {name: list(columns) for name, columns in FEATURE_SETS.items()},
        "models": {name: list(MODELS[name]) for name in family_configs},
        "sort_by": list(SORT_BY),
        "shared": shared,
        "by_family": {
            name: {key: value for key, value in cfg.items() if key not in drop}
            for name, cfg in family_configs.items()
        },
    }


def main() -> None:
    _validate_feature_sets()
    specs = _selected_specs()
    events = _load_events(_union_columns())
    output_dir = Path(specs[0].config.OUTPUT_DIR)

    metric_blocks: list[pd.DataFrame] = []
    family_configs: dict[str, dict] = {}
    for set_name, columns in FEATURE_SETS.items():
        for spec in specs:
            metrics, _predictions, run_config = run_family(
                spec,
                feature_columns=columns,
                events=events,
            )
            metrics.insert(0, "feature_set", set_name)
            metric_blocks.append(metrics)
            family_configs.setdefault(str(spec.name), run_config)

    metrics_all = _sort_metrics(pd.concat(metric_blocks, axis=0, ignore_index=True))
    snapshot = _snapshot(family_configs)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics_feat.csv"
    config_path = output_dir / "run_config_feat.json"
    metrics_all.to_csv(metrics_path, index=False)
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(snapshot, handle, indent=2, default=str)


if __name__ == "__main__":
    main()
