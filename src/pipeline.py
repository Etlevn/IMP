"""End-to-end experiment pipeline: load → preprocess → evaluate → artifacts.

Family-specific tune / fit adapters live in ``src/train.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.events import build_event_table
from src.metrics import (
    infer_task,
    metrics_table,
    permutation_importance_regression,
    print_model_fit_report,
    order_summary_table,
    print_summary_table,
    quintile_table,
    summary_row_from_fold_metrics,
)
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel, sticky_model_bar
from src.preprocess import PreprocessResult, run_preprocess
from src.train import (
    FamilyName,
    FamilySpec,
    FitBundle,
    PredictFn,
    get_family_spec,
)

PREDICTION_META_COLS = (
    "TickerId",
    "EventDate",
    "DecisionDate",
    "AnchorDate",
    "ReleaseDate",
)
BANNER = "=" * 120


def print_banner(title: str) -> None:
    print()
    print(BANNER)
    print()
    print(title.upper())
    print()
    print(BANNER)


def fold_date_range(result: PreprocessResult, split_column: str) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for name, fold in result.folds.items():
        dates = pd.to_datetime(fold.frame[split_column])
        summary[name] = {
            "rows": int(len(fold.frame)),
            "date_min": dates.min().strftime("%Y-%m-%d"),
            "date_max": dates.max().strftime("%Y-%m-%d"),
            "y_mean": float(fold.y.mean()),
            "y_std": float(fold.y.std(ddof=0)),
            "feature_nan": int(fold.X.isna().sum().sum()),
        }
    return summary


def build_predictions(
    result: PreprocessResult,
    *,
    model_type: str,
    predict: PredictFn,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for name, fold in result.folds.items():
        meta_cols = [c for c in PREDICTION_META_COLS if c in fold.frame.columns]
        block = fold.frame.loc[fold.X.index, meta_cols].copy()
        block["model"] = model_type
        block["fold"] = name
        block["y_true"] = fold.y.to_numpy(dtype=float)
        block["y_pred"] = np.asarray(predict(fold.X), dtype=float)
        block["residual"] = block["y_true"] - block["y_pred"]
        blocks.append(block)
    return pd.concat(blocks, axis=0).reset_index(drop=True)


def save_run_artifacts(
    *,
    output_dir: Path,
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    run_config: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    stale = output_dir / "quintiles.csv"
    if stale.exists():
        stale.unlink()
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, default=str)


def print_family_config(
    spec: FamilySpec,
    feature_columns: Sequence[str] | None = None,
) -> None:
    columns = list(
        spec.config.FEATURE_COLUMNS if feature_columns is None else feature_columns
    )
    print("-" * 60)
    print("CONFIG")
    print("-" * 60)
    print(f"MODEL_LABEL       {spec.config.MODEL_LABEL}")
    for key, value in spec.config_extra_lines:
        print(f"{key:<18}{value}")
    print(f"TARGET            {spec.config.TARGET}")
    print(f"BENCHMARK_SOURCE  {spec.config.BENCHMARK_SOURCE}")
    print(f"MARKET_TICKER     {spec.config.MARKET_TICKER}")
    print(f"DATA_ROOT         {spec.config.DATA_ROOT}")
    print(f"FEATURE_COLUMNS   ({len(columns)})")
    for name in columns:
        print(f"                  - {name}")
    print(f"SPLIT_COLUMN      {spec.config.SPLIT_COLUMN}")
    print(f"TRAIN_END         {spec.config.TRAIN_END}")
    print(f"VAL_END           {spec.config.VAL_END}")
    print(f"IMPUTE_STRATEGY   {spec.config.IMPUTE_STRATEGY}")
    print(f"STANDARDIZE       {spec.config.STANDARDIZE}")
    print(f"RANDOM_STATE      {spec.config.RANDOM_STATE}")
    print(f"FEATURE_IMPORTANCE {spec.config.FEATURE_IMPORTANCE}")
    print(f"FEATURE_IMPORTANCE_REPEATS {spec.config.FEATURE_IMPORTANCE_REPEATS}")


def _json_number(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not np.isfinite(number):
        return None
    return number


def _selected_val_scores(
    bundle: FitBundle,
    fold_metrics: pd.DataFrame,
) -> tuple[float | None, float | None]:
    """Validation scores for the chosen setting, else the fitted val fold."""
    tuning = bundle.tuning
    column = {"ALPHA": "alpha", "MAX_DEPTH": "max_depth"}.get(bundle.tuning_label)
    if (
        tuning is not None
        and not tuning.empty
        and bundle.selected is not None
        and column in tuning.columns
    ):
        values = pd.to_numeric(tuning[column], errors="coerce")
        matched = tuning.loc[np.isclose(values, float(bundle.selected), equal_nan=False)]
        if not matched.empty:
            row = matched.iloc[0]
            return _json_number(row["val_oos_r2"]), _json_number(row["val_rmse"])
    if "val" in fold_metrics.index:
        row = fold_metrics.loc["val"]
        return _json_number(row["oos_r2"]), _json_number(row["rmse"])
    return None, None


def model_fit_record(
    model_type: str,
    bundle: FitBundle,
    fold_metrics: pd.DataFrame,
    random_state: int,
) -> dict[str, object]:
    """Per-model fit facts stored in ``run_config.json`` and not in metrics.csv."""
    val_oos_r2, val_rmse = _selected_val_scores(bundle, fold_metrics)
    selected_alpha = None
    selected_max_depth = None
    if bundle.selected is not None and bundle.tuning is not None and not bundle.tuning.empty:
        if bundle.tuning_label == "ALPHA":
            selected_alpha = _json_number(bundle.selected)
        elif bundle.tuning_label == "MAX_DEPTH":
            selected_max_depth = int(bundle.selected)
    record: dict[str, object] = {
        "model": model_type,
        "hidden_sizes": (
            None if bundle.hidden_sizes is None else [int(width) for width in bundle.hidden_sizes]
        ),
        "selected_alpha": selected_alpha,
        "selected_max_depth": selected_max_depth,
        "best_iteration": (
            None if bundle.best_iteration is None else int(bundle.best_iteration)
        ),
        "random_state": int(random_state),
        "val_oos_r2": val_oos_r2,
        "val_rmse": val_rmse,
    }
    return record


def run_one_model(
    spec: FamilySpec,
    *,
    model_type: str,
    result: PreprocessResult,
    train_y_mean: float,
    task: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame, dict[str, object]]:
    """Fit one model via the family adapter, then shared metrics / report."""
    bundle = spec.fit_one(model_type, result)
    cfg = spec.config

    fold_predictions = {
        name: (fold.y, bundle.predict(fold.X)) for name, fold in result.folds.items()
    }
    fold_metrics = metrics_table(
        fold_predictions, task=task, train_y_mean=train_y_mean
    )

    importance = None
    if cfg.FEATURE_IMPORTANCE:
        test = result.folds["test"]
        importance = permutation_importance_regression(
            bundle.predict,
            test.X,
            test.y,
            n_repeats=cfg.FEATURE_IMPORTANCE_REPEATS,
            random_state=cfg.RANDOM_STATE,
            train_y_mean=train_y_mean,
        )

    predictions = build_predictions(
        result, model_type=model_type, predict=bundle.predict
    )
    quintiles = quintile_table(predictions)
    print_model_fit_report(
        model_type,
        fold_metrics=fold_metrics,
        quintiles=quintiles,
        tuning=bundle.tuning,
        selected=bundle.selected,
        tuning_label=bundle.tuning_label,
        best_iteration=bundle.best_iteration,
        max_iter=bundle.max_iter,
        importance=importance,
    )
    summary = summary_row_from_fold_metrics(
        model_type,
        fold_metrics,
        predictions=predictions,
    )
    record = model_fit_record(
        model_type,
        bundle,
        fold_metrics,
        random_state=int(cfg.RANDOM_STATE),
    )
    return summary, predictions, record


def _show_model(progress: tqdm | None, model_type: str, group: str | None) -> None:
    """Label the pinned bar with the model, and the feature set when given."""
    if progress is None:
        return
    label = model_type if group is None else f"{group} · {model_type}"
    progress.set_description_str(f"{label} ", refresh=True)


def run_family(
    spec: FamilySpec,
    *,
    feature_columns: Sequence[str] | None = None,
    events: pd.DataFrame | None = None,
    progress: tqdm | None = None,
    progress_group: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Full batch run for one model family (config → data → fit).

    ``feature_columns`` overrides ``config.FEATURE_COLUMNS`` for this call.
    ``events`` skips the panel load when the event table is already built.
    Returns ``(metrics, predictions, run_config)`` with a leading ``family``
    column on the frames. Does not write artifacts; callers merge and save.
    """
    cfg = spec.config
    family = str(spec.name)
    columns = list(cfg.FEATURE_COLUMNS if feature_columns is None else feature_columns)
    if not columns:
        raise ValueError("feature_columns must be non-empty.")
    model_types = tuple(spec.model_types)
    if not model_types:
        raise ValueError(f"{spec.model_types_key} must be non-empty.")
    _show_model(progress, model_types[0], progress_group)
    print_family_config(spec, columns)

    task = infer_task(cfg.TARGET)
    if task != "regression":
        raise NotImplementedError(
            "Classification metrics are not implemented yet; "
            f"use a regression TARGET (got TARGET={cfg.TARGET!r})."
        )

    if events is None:
        stock_data = load_stock(cfg.DATA_ROOT)
        earnings = load_earnings(cfg.DATA_ROOT)
        historical_sentiment_scores = load_sentiment(cfg.DATA_ROOT)
        df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
        df = build_event_table(
            df_all,
            market_ticker=cfg.MARKET_TICKER,
            data_dir=cfg.DATA_ROOT,
            feature_columns=columns,
            benchmark_source=cfg.BENCHMARK_SOURCE,
        )
    else:
        missing = [name for name in columns if name not in events.columns]
        if missing:
            raise KeyError(
                "Feature columns missing from the event table: "
                f"{missing}. Attach them before build_event_table."
            )
        df = events

    result = run_preprocess(
        df,
        target=cfg.TARGET,
        feature_columns=columns,
        split_column=cfg.SPLIT_COLUMN,
        train_end=cfg.TRAIN_END,
        val_end=cfg.VAL_END,
        impute_strategy=cfg.IMPUTE_STRATEGY,
        standardize=cfg.STANDARDIZE,
    )
    if "val" not in result.folds or "test" not in result.folds:
        raise RuntimeError(
            "Expected train/val/test folds; check TRAIN_END / VAL_END in config."
        )

    train_y_mean = float(np.asarray(result.folds["train"].y, dtype=float).mean())
    summary_rows: list[dict[str, float | int | str]] = []
    prediction_blocks: list[pd.DataFrame] = []
    fits: list[dict[str, object]] = []

    for model_type in model_types:
        _show_model(progress, model_type, progress_group)
        summary, predictions, record = run_one_model(
            spec,
            model_type=model_type,
            result=result,
            train_y_mean=train_y_mean,
            task=task,
        )
        summary_rows.append(summary)
        prediction_blocks.append(predictions)
        fits.append(record)
        if progress is not None:
            progress.update(1)

    metrics = order_summary_table(pd.DataFrame(summary_rows))
    metrics.insert(0, "family", family)
    predictions_all = pd.concat(prediction_blocks, axis=0).reset_index(drop=True)
    predictions_all.insert(0, "family", family)
    print_banner("SUMMARY")
    print_summary_table(metrics)

    run_config = {
        "family": family,
        "MODEL_LABEL": cfg.MODEL_LABEL,
        spec.model_types_key: list(model_types),
        **spec.run_config_extra,
        "TARGET": cfg.TARGET,
        "BENCHMARK_SOURCE": cfg.BENCHMARK_SOURCE,
        "MARKET_TICKER": cfg.MARKET_TICKER,
        "DATA_ROOT": str(cfg.DATA_ROOT),
        "FEATURE_COLUMNS": columns,
        "SPLIT_COLUMN": cfg.SPLIT_COLUMN,
        "TRAIN_END": cfg.TRAIN_END,
        "VAL_END": cfg.VAL_END,
        "IMPUTE_STRATEGY": cfg.IMPUTE_STRATEGY,
        "STANDARDIZE": cfg.STANDARDIZE,
        "RANDOM_STATE": cfg.RANDOM_STATE,
        "train_y_mean": train_y_mean,
        "n_before_target_drop": result.n_before_target_drop,
        "n_dropped_target": result.n_dropped_target,
        "folds": fold_date_range(result, cfg.SPLIT_COLUMN),
        "fits": fits,
    }
    return metrics, predictions_all, run_config


def run_family_by_name(family: FamilyName | Sequence[FamilyName]) -> None:
    """Run one or more families sequentially (``main.py``).

    Each family keeps its own preprocess (``STANDARDIZE``) and SUMMARY print.
    Artifacts are merged and written once under ``output/``:
    ``metrics.csv``, ``predictions.parquet``, ``run_config.json``.
    """
    if isinstance(family, str):
        families: tuple[FamilyName, ...] = (family,)
    else:
        families = tuple(family)
    if not families:
        raise ValueError("families must be non-empty; expected lin / tr / nn.")

    unknown = [name for name in families if name not in ("lin", "tr", "nn")]
    if unknown:
        raise ValueError(
            f"Unknown family name(s) {unknown!r}; expected lin / tr / nn."
        )

    metrics_blocks: list[pd.DataFrame] = []
    prediction_blocks: list[pd.DataFrame] = []
    family_configs: dict[str, dict] = {}
    specs = [get_family_spec(name) for name in families]
    output_dir = Path(specs[0].config.OUTPUT_DIR)

    with sticky_model_bar(sum(len(spec.model_types) for spec in specs)) as progress:
        for spec in specs:
            metrics, predictions, run_config = run_family(spec, progress=progress)
            metrics_blocks.append(metrics)
            prediction_blocks.append(predictions)
            family_configs[str(spec.name)] = run_config

    metrics_all = pd.concat(metrics_blocks, axis=0, ignore_index=True)
    predictions_all = pd.concat(prediction_blocks, axis=0, ignore_index=True)
    run_config_all = {
        "families": list(families),
        "by_family": family_configs,
    }
    save_run_artifacts(
        output_dir=output_dir,
        metrics=metrics_all,
        predictions=predictions_all,
        run_config=run_config_all,
    )
