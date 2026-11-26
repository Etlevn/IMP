"""Tree estimators for the IMP event table."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.metrics import (
    evaluate_regression,
    is_tuning_improvement,
    normalize_tuning_objective,
    pick_tuning_row,
    regression_tuning_metrics,
    tuning_metric_value,
)

# Avoid OpenMP runtime clashes when torch and GBDT share one process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

ALLOWED_TREE_TYPES = frozenset(
    {
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
        "lightgbm",
        "xgboost",
        "catboost",
    }
)
# Tree models select max_depth on the chronological val fold.
DEPTH_TUNABLE_TYPES = ALLOWED_TREE_TYPES
EARLY_STOP_TYPES = frozenset(
    {"hist_gradient_boosting", "lightgbm", "xgboost", "catboost"}
)


def _require_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "lightgbm is required for model_type='lightgbm'. "
            "Install with: pip install lightgbm"
        ) from exc
    return lgb


def _require_xgboost():
    try:
        import xgboost as xgb
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "xgboost is required for model_type='xgboost'. "
            "Install with: pip install xgboost"
        ) from exc
    return xgb


def _require_catboost():
    try:
        from catboost import CatBoostRegressor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "catboost is required for model_type='catboost'. "
            "Install with: pip install catboost"
        ) from exc
    return CatBoostRegressor


def create_tree_model(
    model_type: str,
    *,
    n_estimators: int = 200,
    max_depth: int = 4,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> Any:
    """Build an unfitted tree / ensemble regressor.

    HistGradientBoosting and external GBDT models do **not** use random
    validation splits for early stopping; call ``fit_tree_model`` with the
    chronological val fold.
    """
    if model_type not in ALLOWED_TREE_TYPES:
        raise ValueError(
            f"Unknown TREE_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_TREE_TYPES)}"
        )
    if model_type == "decision_tree":
        return DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
    if model_type == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=-1,
        )
    if model_type == "hist_gradient_boosting":
        return HistGradientBoostingRegressor(
            max_iter=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            early_stopping=False,
            warm_start=False,
            random_state=random_state,
        )
    if model_type == "lightgbm":
        lgb = _require_lightgbm()
        return lgb.LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_samples=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
            verbosity=-1,
        )
    if model_type == "xgboost":
        xgb = _require_xgboost()
        return xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            min_child_weight=min_samples_leaf,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
            objective="reg:squarederror",
        )
    CatBoostRegressor = _require_catboost()
    return CatBoostRegressor(
        iterations=n_estimators,
        depth=max_depth,
        learning_rate=learning_rate,
        min_data_in_leaf=min_samples_leaf,
        random_seed=random_state,
        thread_count=1,
        verbose=False,
        loss_function="RMSE",
        allow_writing_files=False,
    )


def _booster_max_rounds(model: Any) -> int:
    """Configured boosting rounds for LGBM / XGB / CatBoost templates."""
    name = type(model).__name__.lower()
    if "catboost" in name:
        params = model.get_params()
        return int(params.get("iterations") or getattr(model, "iterations", 0) or 1)
    return int(getattr(model, "n_estimators", 1) or 1)


def _clone_booster_with_rounds(model: Any, n_rounds: int) -> Any:
    """Clone an external booster and set its tree/iteration budget."""
    from sklearn.base import clone

    probe = clone(model)
    name = type(model).__name__.lower()
    if "catboost" in name:
        probe.set_params(iterations=int(n_rounds))
    else:
        probe.set_params(n_estimators=int(n_rounds))
    return probe


def _booster_predict_at_iteration(
    model: Any, X: np.ndarray, n_iter: int
) -> np.ndarray:
    """Predict using only the first ``n_iter`` trees of a fitted booster."""
    name = type(model).__name__.lower()
    n_iter = max(1, int(n_iter))
    if "lgbm" in name:
        return np.asarray(
            model.predict(X, num_iteration=n_iter), dtype=float
        )
    if "xgb" in name:
        return np.asarray(
            model.predict(X, iteration_range=(0, n_iter)), dtype=float
        )
    if "catboost" in name:
        return np.asarray(
            model.predict(X, ntree_end=n_iter), dtype=float
        )
    raise TypeError(
        f"Staged prediction not supported for {type(model).__name__}."
    )


def _fit_hgb_on_chronological_val(
    *,
    max_depth: int,
    n_estimators: int,
    learning_rate: float,
    min_samples_leaf: int,
    random_state: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    train_y_mean: float,
    early_stopping_patience: int,
    objective: str = "oos_r2",
) -> tuple[Any, int]:
    """Grow boosters with warm_start; stop on chronological val OBJECTIVE."""
    objective = normalize_tuning_objective(objective)
    probe = HistGradientBoostingRegressor(
        max_iter=1,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        warm_start=True,
        random_state=random_state,
    )
    best_iter = 1
    best_metric = float("-inf")
    best_rmse = float("inf")
    stale = 0
    for n_iter in range(1, int(n_estimators) + 1):
        probe.set_params(max_iter=n_iter)
        probe.fit(X_train, y_train)
        metrics = regression_tuning_metrics(
            y_train,
            probe.predict(X_train),
            y_val,
            probe.predict(X_val),
            train_y_mean=train_y_mean,
        )
        if is_tuning_improvement(
            metrics,
            best_metric=best_metric,
            best_rmse=best_rmse,
            objective=objective,
        ):
            best_iter = n_iter
            best_metric = tuning_metric_value(metrics, objective)
            best_rmse = float(metrics["val_rmse"])
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                break

    final = HistGradientBoostingRegressor(
        max_iter=best_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        early_stopping=False,
        warm_start=False,
        random_state=random_state,
    )
    final.fit(X_train, y_train)
    return final, best_iter


def _fit_external_booster_on_val(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    early_stopping_patience: int,
    train_y_mean: float,
    objective: str = "oos_r2",
) -> tuple[Any, int]:
    """Fit LGBM / XGB / CatBoost once; pick best iter via staged val OBJECTIVE.

    Trains the full ``max_rounds`` booster once, scores prefixes with
    library staged-predict APIs, then refits at ``best_iter`` when needed
    (at most two full fits total — never O(N^2) restarts).
    """
    objective = normalize_tuning_objective(objective)
    max_rounds = max(1, _booster_max_rounds(model))
    full = _clone_booster_with_rounds(model, max_rounds)
    full.fit(X_train, y_train)

    best_iter = 1
    best_metric = float("-inf")
    best_rmse = float("inf")
    stale = 0
    for n_iter in range(1, max_rounds + 1):
        metrics = regression_tuning_metrics(
            y_train,
            _booster_predict_at_iteration(full, X_train, n_iter),
            y_val,
            _booster_predict_at_iteration(full, X_val, n_iter),
            train_y_mean=train_y_mean,
        )
        if is_tuning_improvement(
            metrics,
            best_metric=best_metric,
            best_rmse=best_rmse,
            objective=objective,
        ):
            best_iter = n_iter
            best_metric = tuning_metric_value(metrics, objective)
            best_rmse = float(metrics["val_rmse"])
            stale = 0
        else:
            stale += 1
            if stale >= int(early_stopping_patience):
                break

    if int(best_iter) == int(max_rounds):
        return full, int(best_iter)
    best_model = _clone_booster_with_rounds(model, best_iter)
    best_model.fit(X_train, y_train)
    return best_model, int(best_iter)


def fit_tree_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    early_stopping_patience: int = 10,
    objective: str = "oos_r2",
) -> tuple[Any, dict[str, float | int]]:
    """Fit on train; boosted trees early-stop on chronological val OBJECTIVE."""
    objective = normalize_tuning_objective(objective)
    X_tr = X_train.to_numpy(dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    info: dict[str, float | int] = {}

    if isinstance(model, HistGradientBoostingRegressor):
        if X_val is None or y_val is None:
            raise ValueError(
                "hist_gradient_boosting requires chronological X_val / y_val "
                "for early stopping."
            )
        X_va = X_val.to_numpy(dtype=float)
        y_va = np.asarray(y_val, dtype=float)
        train_mean = float(y_tr.mean())
        fitted, best_iter = _fit_hgb_on_chronological_val(
            max_depth=int(model.max_depth),
            n_estimators=int(model.max_iter),
            learning_rate=float(model.learning_rate),
            min_samples_leaf=int(model.min_samples_leaf),
            random_state=int(model.random_state),
            X_train=X_tr,
            y_train=y_tr,
            X_val=X_va,
            y_val=y_va,
            train_y_mean=train_mean,
            early_stopping_patience=int(early_stopping_patience),
            objective=objective,
        )
        info["best_iteration"] = int(best_iter)
        return fitted, info

    model_name = type(model).__name__.lower()
    is_external = any(
        token in model_name for token in ("lgbm", "xgb", "catboost")
    )
    if is_external:
        if X_val is None or y_val is None:
            raise ValueError(
                f"{type(model).__name__} requires chronological X_val / y_val "
                "for early stopping."
            )
        X_va = X_val.to_numpy(dtype=float)
        y_va = np.asarray(y_val, dtype=float)
        fitted, best_iter = _fit_external_booster_on_val(
            model,
            X_tr,
            y_tr,
            X_va,
            y_va,
            early_stopping_patience=int(early_stopping_patience),
            train_y_mean=float(y_tr.mean()),
            objective=objective,
        )
        info["best_iteration"] = int(best_iter)
        return fitted, info

    model.fit(X_tr, y_tr)
    return model, info


def predict_tree(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Continuous predictions."""
    return np.asarray(model.predict(X.to_numpy(dtype=float)), dtype=float)


def select_max_depth_on_val(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    depth_grid: tuple[int, ...] | list[int],
    default_max_depth: int,
    n_estimators: int = 200,
    learning_rate: float = 0.05,
    min_samples_leaf: int = 20,
    early_stopping_patience: int = 10,
    random_state: int = 42,
    objective: str = "oos_r2",
) -> tuple[int, pd.DataFrame]:
    """Pick max_depth on chronological val by maximizing ``objective``.

    ``objective`` is one of ``oos_r2`` | ``rank_ic`` | ``quintile_ls``.
    Returns ``(best_depth, tuning_table)``. Empty grid -> ``default_max_depth``.
    """
    objective = normalize_tuning_objective(objective)
    if model_type not in DEPTH_TUNABLE_TYPES or not depth_grid:
        return int(default_max_depth), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float | int]] = []
    for depth in depth_grid:
        model = create_tree_model(
            model_type,
            n_estimators=n_estimators,
            max_depth=int(depth),
            learning_rate=learning_rate,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
        )
        fitted, _ = fit_tree_model(
            model,
            X_train,
            y_train,
            X_val=X_val if model_type in EARLY_STOP_TYPES else None,
            y_val=y_val if model_type in EARLY_STOP_TYPES else None,
            early_stopping_patience=early_stopping_patience,
            objective=objective,
        )
        metrics = regression_tuning_metrics(
            y_train,
            predict_tree(fitted, X_train),
            y_val,
            predict_tree(fitted, X_val),
            train_y_mean=train_mean,
        )
        rows.append({"max_depth": int(depth), **metrics})
    table = pick_tuning_row(pd.DataFrame(rows), objective)
    best_depth = int(table.loc[0, "max_depth"])
    return best_depth, table
