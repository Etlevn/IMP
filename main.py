"""Train IMP models with per-model feature combinations.

Edit ``MODELS`` in ``config_main.py``, and knobs below, then:

    python main.py

Each model maps to one or more feature lists (empty = skip).
``VERBOSE_PRINT=False`` prints only the final metrics table.
Family hyperparameters stay in ``config/config_{lin,tr,nn}.py``.
Shared defaults: ``config/config.py``.
Artifacts: ``output/main_metrics.csv``, optional
``output/main_predictions.parquet``, ``output/config_main.json``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from config.config import EARNINGS_FILE, SENTIMENT_FILE, STOCK_FILE
from config_main import MODELS
from src.compact_json import write_compact_json
from src.features import registered_features
from src.metrics import print_summary_table
from src.pipeline import print_banner, run_family, save_run_artifacts
from src.prepare import sticky_model_bar
from src.train import FamilyName, FamilySpec, get_family_spec

# ======================================== [CONFIG] ========================================

# True: print METRICS / QUINTILES per model, then SUMMARY banner.
# False: print only the final metrics table.
VERBOSE_PRINT = False

# True: write output/main_predictions.parquet alongside main_metrics.csv.
SAVE_PREDICTIONS = False

# Model feature menus: edit ``MODELS`` in ``config_main.py`` (empty = skip).

# ========================================================================================

_FAMILY_ORDER: tuple[FamilyName, ...] = ("lin", "tr", "nn")
def _feature_key(columns: list[str]) -> str:
    """Single-quoted CSV cell, pasteable as a Python list: 'A','B','C'."""
    return ",".join(f"'{name}'" for name in columns)


def _normalize_combos(
    family: str,
    model_type: str,
    value: object,
) -> list[list[str]]:
    """Accept a flat feature list or a list of feature lists."""
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError(
            f"MODELS[{family!r}][{model_type!r}] must be a feature list "
            f"or a list of feature lists, got {type(value).__name__}."
        )
    if not value:
        return []
    if all(isinstance(item, str) for item in value):
        return [list(value)]
    combos: list[list[str]] = []
    for index, item in enumerate(value):
        if not isinstance(item, (list, tuple)):
            raise ValueError(
                f"MODELS[{family!r}][{model_type!r}][{index}] must be a "
                f"feature list, got {type(item).__name__}."
            )
        if not item:
            raise ValueError(
                f"MODELS[{family!r}][{model_type!r}][{index}] is empty; "
                "remove it or list features."
            )
        if not all(isinstance(name, str) for name in item):
            raise ValueError(
                f"MODELS[{family!r}][{model_type!r}][{index}] must contain "
                "only feature name strings."
            )
        combos.append(list(item))
    return combos


def _validate_models() -> None:
    if not MODELS:
        raise ValueError("MODELS must include at least one of lin / tr / nn.")
    unknown_families = [name for name in MODELS if name not in _FAMILY_ORDER]
    if unknown_families:
        raise ValueError(
            f"Unknown model family {unknown_families!r}; expected lin / tr / nn."
        )
    available = registered_features()
    for family in _FAMILY_ORDER:
        if family not in MODELS:
            continue
        model_map = MODELS[family]
        if not model_map:
            raise ValueError(f"MODELS[{family!r}] must list at least one sub-model.")
        spec = get_family_spec(family)
        allowed = set(spec.model_types)
        for model_type, raw in model_map.items():
            if model_type not in allowed:
                raise ValueError(
                    f"Unknown {family} sub-model {model_type!r}; "
                    f"allowed: {tuple(spec.model_types)}."
                )
            for combo_index, columns in enumerate(
                _normalize_combos(family, model_type, raw)
            ):
                label = f"MODELS[{family!r}][{model_type!r}][{combo_index}]"
                duplicates = [name for name in columns if columns.count(name) > 1]
                if duplicates:
                    raise ValueError(
                        f"{label} repeats {sorted(set(duplicates))}."
                    )
                unknown = [name for name in columns if name not in available]
                if unknown:
                    raise ValueError(
                        f"{label} has unknown features {unknown}. "
                        "Enable panel-native names in ORIGINAL_FEATURES, "
                        "or derived names in ANALYST_FEATURES / MACRO_FEATURES / "
                        "ENGINEERED_FEATURES (see enabled_features())."
                    )


def _selected_pairs() -> list[tuple[FamilySpec, str, list[str]]]:
    pairs: list[tuple[FamilySpec, str, list[str]]] = []
    for family in _FAMILY_ORDER:
        if family not in MODELS:
            continue
        base = get_family_spec(family)
        for model_type, raw in MODELS[family].items():
            combos = _normalize_combos(family, model_type, raw)
            if not combos:
                continue
            lines = tuple(
                (key, (model_type,) if key == base.model_types_key else value)
                for key, value in base.config_extra_lines
            )
            narrowed = replace(
                base,
                model_types=(model_type,),
                config_extra_lines=lines,
            )
            for columns in combos:
                pairs.append((narrowed, model_type, list(columns)))
    return pairs


def _feature_union(pairs: list[tuple[FamilySpec, str, list[str]]]) -> list[str]:
    ordered: list[str] = []
    for _spec, _model, columns in pairs:
        for name in columns:
            if name not in ordered:
                ordered.append(name)
    return ordered


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


def _shared_snapshot(pairs: list[tuple[FamilySpec, str, list[str]]]) -> dict:
    first = pairs[0][0].config
    standardize = {
        str(spec.name): bool(spec.config.STANDARDIZE) for spec, _, _ in pairs
    }
    shared = {
        "TARGET": first.TARGET,
        "BENCHMARK_SOURCE": first.BENCHMARK_SOURCE,
        "DATA_FILES": [STOCK_FILE, EARNINGS_FILE, SENTIMENT_FILE],
        "SPLIT_COLUMN": first.SPLIT_COLUMN,
        "TRAIN_END": first.TRAIN_END,
        "VAL_END": first.VAL_END,
        "IMPUTE_STRATEGY": first.IMPUTE_STRATEGY,
        "RANDOM_STATE": int(first.RANDOM_STATE),
    }
    if len(standardize) == 1:
        shared["STANDARDIZE"] = next(iter(standardize.values()))
    else:
        shared["STANDARDIZE"] = standardize
    return shared


def _used_hyperparams(spec: FamilySpec, model_type: str) -> dict:
    """Family hyperparams that this sub-model actually consumes."""
    extra = dict(spec.config_extra)
    family = str(spec.name)
    if family == "lin":
        if model_type in {"ridge", "lasso"}:
            return {
                key: extra[key]
                for key in ("ALPHA", "ALPHA_GRID")
                if key in extra
            }
        return {}
    if family == "tr":
        keys = ["MAX_DEPTH", "MAX_DEPTH_GRID", "MIN_SAMPLES_LEAF"]
        if model_type == "decision_tree":
            pass
        elif model_type == "random_forest":
            keys.append("N_ESTIMATORS")
        else:  # hist_gradient_boosting
            keys.extend(
                ("N_ESTIMATORS", "LEARNING_RATE", "EARLY_STOPPING_PATIENCE")
            )
        return {key: extra[key] for key in keys if key in extra}
    if family == "nn":
        out = {
            key: extra[key]
            for key in (
                "ALPHA",
                "ALPHA_GRID",
                "LR",
                "BATCH_SIZE",
                "MAX_ITER",
                "EARLY_STOPPING_PATIENCE",
            )
            if key in extra
        }
        hidden = extra.get("HIDDEN_SIZES")
        if isinstance(hidden, dict) and model_type in hidden:
            out["HIDDEN_SIZES"] = hidden[model_type]
        return out
    return {}


def _config_snapshot(
    pairs: list[tuple[FamilySpec, str, list[str]]],
) -> dict:
    """Settings only: shared once; each run keeps model/family/features/used hypers."""
    runs = []
    for spec, model_type, columns in pairs:
        run: dict = {
            "model": model_type,
            "family": str(spec.name),
            "features": list(columns),
        }
        run.update(_used_hyperparams(spec, model_type))
        runs.append(run)
    return {
        "shared": _shared_snapshot(pairs),
        "runs": runs,
    }


def main() -> None:
    _validate_models()
    pairs = _selected_pairs()
    if not pairs:
        raise ValueError(
            "MODELS produced no runnable (family, model) pairs "
            "(every listed model has an empty feature list?)."
        )

    events = _load_events(_feature_union(pairs))
    output_dir = Path(pairs[0][0].config.OUTPUT_DIR)

    metrics_blocks: list[pd.DataFrame] = []
    prediction_blocks: list[pd.DataFrame] = []

    with sticky_model_bar(len(pairs)) as progress:
        for spec, model_type, columns in pairs:
            metrics, predictions, _config = run_family(
                spec,
                feature_columns=columns,
                events=events,
                progress=progress,
                print_config=False,
                print_summary=False,
                quiet=not VERBOSE_PRINT,
            )
            feature_key = _feature_key(columns)
            metrics = metrics.copy()
            metrics["features"] = feature_key
            metrics_blocks.append(metrics)
            if SAVE_PREDICTIONS:
                predictions = predictions.copy()
                predictions["features"] = feature_key
                prediction_blocks.append(predictions)

    metrics_all = pd.concat(metrics_blocks, axis=0, ignore_index=True)
    if VERBOSE_PRINT:
        print_banner("SUMMARY")
    print_summary_table(metrics_all)
    metrics_path = output_dir / "main_metrics.csv"
    predictions_path = output_dir / "main_predictions.parquet"
    config_path = output_dir / "config_main.json"
    save_run_artifacts(
        output_dir=output_dir,
        metrics=metrics_all,
        predictions=(
            pd.concat(prediction_blocks, axis=0, ignore_index=True)
            if SAVE_PREDICTIONS
            else None
        ),
        save_predictions=SAVE_PREDICTIONS,
    )
    write_compact_json(config_path, _config_snapshot(pairs))
    print()
    print(f"[WROTE] {metrics_path}")
    if SAVE_PREDICTIONS:
        print(f"[WROTE] {predictions_path}")
    print(f"[WROTE] {config_path}")


if __name__ == "__main__":
    main()
