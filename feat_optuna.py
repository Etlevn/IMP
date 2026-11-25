"""Per-model Optuna / exhaustive search for feature subsets.

Prerequisite: run ``python feat_filter.py`` so ``output/feat_filter.csv`` exists.

Feature universe comes from ``src.features.enabled_features()`` (four config
groups). Pool order: ``FEATURE_WHITELIST`` first, then feat_filter ``suggest``
labels enabled below; ``FEATURE_BLACKLIST`` is excluded. Edit ``MODELS`` /
``INCLUDE_SUGGEST`` / ``N_TRIALS``, then:

    python feat_optuna.py

Selection metric is a validation fold score chosen by ``OBJECTIVE``.
Artifacts: ``output/feat_optuna_trial.csv`` (all trials),
``output/feat_optuna_best.csv`` (per-model best: n_val + OBJECTIVE val metric + test),
``output/config_optuna.json``. Does not modify ``main.py`` outputs.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config.config import (
    EARNINGS_FILE,
    FEATURE_BLACKLIST,
    FEATURE_WHITELIST,
    OUTPUT_DIR,
    RANDOM_STATE,
    SENTIMENT_FILE,
    STOCK_FILE,
)
from src.compact_json import write_compact_json
from src.features import enabled_features
from src.metrics import SUMMARY_METRIC_COLUMNS
from src.optuna_search import (
    OBJECTIVE_CHOICES,
    build_search_pool,
    mark_best_per_model,
    objective_column,
    planned_trial_count,
    run_model_search,
)
from src.prepare import sticky_stack
from src.train import FamilyName, FamilySpec, get_family_spec


# ======================================== [CONFIG] ========================================

# Fixed models. Comment out a family or sub-model to skip it.
MODELS: dict[str, tuple[str, ...]] = {
    "lin": ("ols", "ridge", "lasso"),
    "tr": ("decision_tree", "random_forest", "hist_gradient_boosting"),
    "nn": ("mlp_shallow", "mlp", "mlp_deep"),
}

# Which feat_filter ``suggest`` labels enter the pool after FEATURE_WHITELIST.
INCLUDE_SUGGEST: list[str] = ["keep", "weak","noise"]    # ["keep", "weak", "noise", "redundant", "drop"]

# Validation metric to maximize. One of: "oos_r2" | "rank_ic" | "quintile_ls".
OBJECTIVE: str = "oos_r2"

# Optuna trials per model when the searchable pool is larger than EXHAUSTIVE_MAX.
N_TRIALS = 120

# If len(searchable) <= this, enumerate all subsets instead of Optuna (2^n-1 subsets).
EXHAUSTIVE_MAX = 6

FILTER_PATH = OUTPUT_DIR / "feat_filter.csv"

# ========================================================================================

_FAMILY_ORDER: tuple[FamilyName, ...] = ("lin", "tr", "nn")


def _best_columns(objective: str) -> tuple[str, ...]:
    """Best-table columns: ids, n_val + OBJECTIVE val metric, then test SUMMARY."""
    return (
        "family",
        "model",
        "features",
        "n_features",
        "n_val",
        objective_column(objective),
        "n_test",
        *SUMMARY_METRIC_COLUMNS,
    )


def _selected_specs() -> list[tuple[FamilySpec, str]]:
    if not MODELS:
        raise ValueError("MODELS must include at least one of lin / tr / nn.")
    unknown = [name for name in MODELS if name not in _FAMILY_ORDER]
    if unknown:
        raise ValueError(f"Unknown model family {unknown!r}; expected lin / tr / nn.")

    pairs: list[tuple[FamilySpec, str]] = []
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
        narrowed = replace(spec, model_types=selected, config_extra_lines=lines)
        for model_type in selected:
            pairs.append((narrowed, model_type))
    return pairs


def _load_events(columns: list[str]) -> pd.DataFrame:
    from config.config import BENCHMARK_SOURCE, DATA_ROOT, MARKET_TICKER
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


def _snapshot(*, shared: dict) -> dict:
    """Entrypoint / data settings only — no trial or best results."""
    return {
        "objective": OBJECTIVE,
        "objective_column": objective_column(OBJECTIVE),
        "n_trials": N_TRIALS,
        "exhaustive_max": EXHAUSTIVE_MAX,
        "include_suggest": list(INCLUDE_SUGGEST),
        "feature_whitelist": list(FEATURE_WHITELIST),
        "feature_blacklist": list(FEATURE_BLACKLIST),
        "filter_path": str(FILTER_PATH),
        "feature_columns": list(enabled_features()),
        "models": {name: list(MODELS[name]) for name in MODELS},
        "shared": shared,
    }


def _validate_feature_config(catalog: list[str]) -> None:
    if OBJECTIVE not in OBJECTIVE_CHOICES:
        raise ValueError(
            f"OBJECTIVE must be one of {OBJECTIVE_CHOICES}; got {OBJECTIVE!r}."
        )
    if not catalog:
        raise ValueError(
            "enabled_features() is empty. Populate ORIGINAL_FEATURES / "
            "ANALYST_FEATURES / MACRO_FEATURES / ENGINEERED_FEATURES in "
            "config/config.py."
        )
    enabled = set(catalog)
    bad_whitelist = [name for name in FEATURE_WHITELIST if name not in enabled]
    if bad_whitelist:
        raise ValueError(
            f"FEATURE_WHITELIST must be a subset of enabled_features(); "
            f"missing {bad_whitelist}."
        )
    bad_blacklist = [name for name in FEATURE_BLACKLIST if name not in enabled]
    if bad_blacklist:
        raise ValueError(
            f"FEATURE_BLACKLIST must be a subset of enabled_features(); "
            f"missing {bad_blacklist}."
        )
    overlap = sorted(set(FEATURE_WHITELIST) & set(FEATURE_BLACKLIST))
    if overlap:
        raise ValueError(
            f"FEATURE_WHITELIST and FEATURE_BLACKLIST overlap: {overlap}."
        )


def main() -> None:
    if not FILTER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {FILTER_PATH}. Run `python feat_filter.py` first."
        )
    catalog = enabled_features()
    _validate_feature_config(catalog)

    table = pd.read_csv(FILTER_PATH)
    pool = build_search_pool(
        table,
        whitelist=FEATURE_WHITELIST,
        blacklist=FEATURE_BLACKLIST,
        include_suggest=INCLUDE_SUGGEST,
        catalog=catalog,
    )
    pairs = _selected_specs()
    events = _load_events(list(catalog))

    preprocess_cache: dict = {}
    blocks: list[pd.DataFrame] = []
    first_spec = pairs[0][0]
    shared = {
        "TARGET": first_spec.config.TARGET,
        "BENCHMARK_SOURCE": first_spec.config.BENCHMARK_SOURCE,
        "MARKET_TICKER": first_spec.config.MARKET_TICKER,
        "DATA_FILES": [STOCK_FILE, EARNINGS_FILE, SENTIMENT_FILE],
        "SPLIT_COLUMN": first_spec.config.SPLIT_COLUMN,
        "TRAIN_END": first_spec.config.TRAIN_END,
        "VAL_END": first_spec.config.VAL_END,
        "IMPUTE_STRATEGY": first_spec.config.IMPUTE_STRATEGY,
        "RANDOM_STATE": int(first_spec.config.RANDOM_STATE),
    }

    trial_total = planned_trial_count(
        pool, n_trials=N_TRIALS, exhaustive_max=EXHAUSTIVE_MAX
    )
    with sticky_stack((len(pairs), max(trial_total, 1))) as (model_bar, trial_bar):
        for spec, model_type in pairs:
            if model_bar is not None:
                model_bar.set_description_str(f"{model_type} ", refresh=True)
            if trial_bar is not None:
                trial_bar.reset(total=max(trial_total, 1))
                trial_bar.set_description_str("trial ", refresh=True)
            block = run_model_search(
                spec,
                model_type=model_type,
                pool=pool,
                events=events,
                n_trials=N_TRIALS,
                exhaustive_max=EXHAUSTIVE_MAX,
                random_state=RANDOM_STATE,
                objective=OBJECTIVE,
                trial_bar=trial_bar,
                preprocess_cache=preprocess_cache,
            )
            blocks.append(block)
            if model_bar is not None:
                model_bar.update(1)

    metrics = mark_best_per_model(
        pd.concat(blocks, axis=0, ignore_index=True),
        objective=OBJECTIVE,
    )
    best_cols = [col for col in _best_columns(OBJECTIVE) if col in metrics.columns]
    missing = [
        col for col in ("n_val", objective_column(OBJECTIVE)) if col not in metrics.columns
    ]
    if missing:
        raise ValueError(f"metrics missing columns for best table: {missing}.")
    sort_col = OBJECTIVE  # test column matching the val OBJECTIVE
    if sort_col not in metrics.columns:
        raise ValueError(f"metrics missing test column {sort_col!r} for OBJECTIVE.")
    best = (
        metrics.loc[metrics["best"] == 1, best_cols]
        .sort_values(sort_col, ascending=False, na_position="last", kind="mergesort")
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    trial_path = Path(OUTPUT_DIR) / "feat_optuna_trial.csv"
    best_path = Path(OUTPUT_DIR) / "feat_optuna_best.csv"
    config_path = Path(OUTPUT_DIR) / "config_optuna.json"
    metrics.to_csv(trial_path, index=False, float_format="%.4f")
    best.to_csv(best_path, index=False, float_format="%.4f")
    write_compact_json(config_path, _snapshot(shared=shared))

    display = best.copy()
    val_col = objective_column(OBJECTIVE)
    for column in (*SUMMARY_METRIC_COLUMNS, val_col):
        if column not in display.columns:
            continue
        display[column] = display[column].map(
            lambda value: f"{float(value):.4f}" if pd.notna(value) else ""
        )
    print()
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
        "display.max_colwidth",
        None,
    ):
        print(display.to_string(index=False))
    print()
    print(f"[WROTE] {trial_path}")
    print(f"[WROTE] {best_path}")
    print(f"[WROTE] {config_path}")


if __name__ == "__main__":
    main()
