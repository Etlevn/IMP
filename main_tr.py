"""Train tree models on the IMP event table.

Reads settings from ``config/config_tr.py`` only. Runs every entry in
``TREE_MODEL_TYPES`` under the same preprocess split, prints each model
behind a banner, and writes one summary row per model to ``metrics.csv``.

Usage (from repo root):
    python main_tr.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config.config_tr as C
from model.model_tr import (
    EARLY_STOP_TYPES,
    create_tree_model,
    fit_tree_model,
    predict_tree,
    select_max_depth_on_val,
)
from src.features import build_event_table
from src.metrics import (
    infer_task,
    metrics_table,
    permutation_importance_regression,
    print_feature_importance,
    print_metrics_table,
    print_quintile_table,
    print_summary_table,
    quintile_table,
)
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.preprocess import PreprocessResult, run_preprocess

PREDICTION_META_COLS = (
    "TickerId",
    "EventDate",
    "decision_session",
    "anchor_session",
    "release_session",
)
BANNER = "=" * 120


def _print_config() -> None:
    print("-" * 60)
    print("CONFIG")
    print("-" * 60)
    print(f"MODEL_LABEL       {C.MODEL_LABEL}")
    print(f"TREE_MODEL_TYPES  {tuple(C.TREE_MODEL_TYPES)}")
    print(f"N_ESTIMATORS      {C.N_ESTIMATORS}")
    print(f"MAX_DEPTH         {C.MAX_DEPTH}")
    print(f"MAX_DEPTH_GRID    {tuple(C.MAX_DEPTH_GRID)}")
    print(f"LEARNING_RATE     {C.LEARNING_RATE}")
    print(f"MIN_SAMPLES_LEAF  {C.MIN_SAMPLES_LEAF}")
    print(f"EARLY_STOPPING_PATIENCE {C.EARLY_STOPPING_PATIENCE}")
    print(f"TARGET            {C.TARGET}")
    print(f"FEATURE_COLUMNS   ({len(C.FEATURE_COLUMNS)})")
    for name in C.FEATURE_COLUMNS:
        print(f"                  - {name}")
    print(f"SPLIT_COLUMN      {C.SPLIT_COLUMN}")
    print(f"TRAIN_END         {C.TRAIN_END}")
    print(f"VAL_END           {C.VAL_END}")
    print(f"IMPUTE_STRATEGY   {C.IMPUTE_STRATEGY}")
    print(f"STANDARDIZE       {C.STANDARDIZE}")
    print(f"RANDOM_STATE      {C.RANDOM_STATE}")
    print(f"FEATURE_IMPORTANCE {C.FEATURE_IMPORTANCE}")
    print(f"FEATURE_IMPORTANCE_REPEATS {C.FEATURE_IMPORTANCE_REPEATS}")
    print(f"OUTPUT_DIR        {C.OUTPUT_DIR}")


def _print_banner(title: str) -> None:
    print()
    print(BANNER)
    print()
    print(title.upper())
    print()
    print(BANNER)


def _fold_date_range(result: PreprocessResult, split_column: str) -> dict[str, dict]:
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


def _build_predictions(
    result: PreprocessResult,
    model,
    *,
    model_type: str,
) -> pd.DataFrame:
    blocks: list[pd.DataFrame] = []
    for name, fold in result.folds.items():
        meta_cols = [c for c in PREDICTION_META_COLS if c in fold.frame.columns]
        block = fold.frame.loc[fold.X.index, meta_cols].copy()
        block["model"] = model_type
        block["fold"] = name
        block["y_true"] = fold.y.to_numpy(dtype=float)
        block["y_pred"] = predict_tree(model, fold.X)
        block["residual"] = block["y_true"] - block["y_pred"]
        blocks.append(block)
    return pd.concat(blocks, axis=0).reset_index(drop=True)


def _print_tuning_table(title: str, tuning: pd.DataFrame, selected_line: str) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(tuning.to_string(index=False))
    print(selected_line)


def _model_summary_row(
    *,
    model_type: str,
    selected_max_depth: int,
    best_iteration: int | None,
    fold_metrics: pd.DataFrame,
) -> dict[str, float | int | str]:
    """One comparison row: test-fold metrics only (raw values for CSV)."""
    test = fold_metrics.loc["test"]
    row: dict[str, float | int | str] = {
        "model": model_type,
        "max_depth": int(selected_max_depth),
        "n_estimators": int(C.N_ESTIMATORS),
        "n_test": int(test["n"]),
        "r2": float(test["r2"]),
        "oos_r2": float(test["oos_r2"]),
        "rmse": float(test["rmse"]),
        "mae": float(test["mae"]),
        "rmse_train_mean": float(test["rmse_train_mean"]),
        "mae_train_mean": float(test["mae_train_mean"]),
        "rmse_zero": float(test["rmse_zero"]),
        "mae_zero": float(test["mae_zero"]),
        "ic": float(test["ic"]),
        "rank_ic": float(test["rank_ic"]),
        "hit_rate": float(test["hit_rate"]),
    }
    if best_iteration is not None:
        row["best_iteration"] = int(best_iteration)
    return row


def _print_summary(table: pd.DataFrame) -> None:
    _print_banner("SUMMARY")
    print_summary_table(table)


def _save_run_artifacts(
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


def _run_one_model(
    *,
    model_type: str,
    result: PreprocessResult,
    train_y_mean: float,
    task: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    """Fit one model, print its sections, return summary row + predictions."""
    _print_banner(model_type)

    train = result.folds["train"]
    val = result.folds["val"]
    selected_depth, tuning = select_max_depth_on_val(
        model_type,
        train.X,
        train.y,
        val.X,
        val.y,
        depth_grid=tuple(C.MAX_DEPTH_GRID),
        default_max_depth=int(C.MAX_DEPTH),
        n_estimators=C.N_ESTIMATORS,
        learning_rate=C.LEARNING_RATE,
        min_samples_leaf=C.MIN_SAMPLES_LEAF,
        early_stopping_patience=C.EARLY_STOPPING_PATIENCE,
        random_state=C.RANDOM_STATE,
    )
    if not tuning.empty:
        _print_tuning_table(
            "MAX_DEPTH",
            tuning,
            f"SELECTED_MAX_DEPTH {selected_depth}",
        )

    model = create_tree_model(
        model_type,
        n_estimators=C.N_ESTIMATORS,
        max_depth=selected_depth,
        learning_rate=C.LEARNING_RATE,
        min_samples_leaf=C.MIN_SAMPLES_LEAF,
        random_state=C.RANDOM_STATE,
    )
    use_val_stop = model_type in EARLY_STOP_TYPES
    model, fit_info = fit_tree_model(
        model,
        train.X,
        train.y,
        X_val=val.X if use_val_stop else None,
        y_val=val.y if use_val_stop else None,
        early_stopping_patience=C.EARLY_STOPPING_PATIENCE,
    )
    best_iteration = (
        int(fit_info["best_iteration"]) if "best_iteration" in fit_info else None
    )
    if best_iteration is not None:
        print()
        print("-" * 60)
        print("EARLY_STOP")
        print("-" * 60)
        print(f"BEST_ITERATION    {best_iteration}")
        print(f"MAX_ITER          {C.N_ESTIMATORS}")

    fold_predictions = {
        name: (fold.y, predict_tree(model, fold.X))
        for name, fold in result.folds.items()
    }
    fold_metrics = metrics_table(
        fold_predictions, task=task, train_y_mean=train_y_mean
    )
    print()
    print_metrics_table(fold_metrics)

    if C.FEATURE_IMPORTANCE:
        test = result.folds["test"]
        importance = permutation_importance_regression(
            lambda X: predict_tree(model, X),
            test.X,
            test.y,
            n_repeats=C.FEATURE_IMPORTANCE_REPEATS,
            random_state=C.RANDOM_STATE,
            train_y_mean=train_y_mean,
        )
        print()
        print_feature_importance(importance)

    predictions = _build_predictions(result, model, model_type=model_type)
    quintiles = quintile_table(predictions)
    print_quintile_table(quintiles)

    summary = _model_summary_row(
        model_type=model_type,
        selected_max_depth=selected_depth,
        best_iteration=best_iteration,
        fold_metrics=fold_metrics,
    )
    return summary, predictions


def main() -> None:
    _print_config()

    task = infer_task(C.TARGET)
    if task != "regression":
        raise NotImplementedError(
            "Classification metrics are not implemented yet; "
            f"use a regression TARGET (got TARGET={C.TARGET!r})."
        )

    model_types = tuple(C.TREE_MODEL_TYPES)
    if not model_types:
        raise ValueError("TREE_MODEL_TYPES must be non-empty.")

    stock_data = load_stock()
    earnings = load_earnings()
    historical_sentiment_scores = load_sentiment()
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    df = build_event_table(df_all)

    result = run_preprocess(
        df,
        target=C.TARGET,
        feature_columns=C.FEATURE_COLUMNS,
        split_column=C.SPLIT_COLUMN,
        train_end=C.TRAIN_END,
        val_end=C.VAL_END,
        impute_strategy=C.IMPUTE_STRATEGY,
        standardize=C.STANDARDIZE,
    )
    if "val" not in result.folds or "test" not in result.folds:
        raise RuntimeError(
            "Expected train/val/test folds; check TRAIN_END / VAL_END in config."
        )

    train_y_mean = float(np.asarray(result.folds["train"].y, dtype=float).mean())
    summary_rows: list[dict[str, float | int | str]] = []
    prediction_blocks: list[pd.DataFrame] = []

    for model_type in model_types:
        summary, predictions = _run_one_model(
            model_type=model_type,
            result=result,
            train_y_mean=train_y_mean,
            task=task,
        )
        summary_rows.append(summary)
        prediction_blocks.append(predictions)

    metrics = pd.DataFrame(summary_rows)
    predictions_all = pd.concat(prediction_blocks, axis=0).reset_index(drop=True)
    _print_summary(metrics)

    run_config = {
        "MODEL_LABEL": C.MODEL_LABEL,
        "TREE_MODEL_TYPES": list(model_types),
        "N_ESTIMATORS": C.N_ESTIMATORS,
        "MAX_DEPTH": C.MAX_DEPTH,
        "MAX_DEPTH_GRID": list(C.MAX_DEPTH_GRID),
        "LEARNING_RATE": C.LEARNING_RATE,
        "MIN_SAMPLES_LEAF": C.MIN_SAMPLES_LEAF,
        "EARLY_STOPPING_PATIENCE": C.EARLY_STOPPING_PATIENCE,
        "TARGET": C.TARGET,
        "FEATURE_COLUMNS": list(C.FEATURE_COLUMNS),
        "SPLIT_COLUMN": C.SPLIT_COLUMN,
        "TRAIN_END": C.TRAIN_END,
        "VAL_END": C.VAL_END,
        "IMPUTE_STRATEGY": C.IMPUTE_STRATEGY,
        "STANDARDIZE": C.STANDARDIZE,
        "RANDOM_STATE": C.RANDOM_STATE,
        "train_y_mean": train_y_mean,
        "n_before_target_drop": result.n_before_target_drop,
        "n_dropped_target": result.n_dropped_target,
        "folds": _fold_date_range(result, C.SPLIT_COLUMN),
    }
    _save_run_artifacts(
        output_dir=Path(C.OUTPUT_DIR),
        metrics=metrics,
        predictions=predictions_all,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
