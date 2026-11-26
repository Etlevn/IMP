"""Linear estimators for the IMP event table."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    QuantileRegressor,
    Ridge,
)

from src.metrics import (
    normalize_tuning_objective,
    pick_tuning_row,
    regression_tuning_metrics,
)

ALLOWED_LINEAR_TYPES = frozenset(
    {"ols", "ridge", "lasso", "logistic", "elasticnet", "huber", "quantile"}
)
CLASSIFICATION_TYPES = frozenset({"logistic"})
ALPHA_TUNABLE_TYPES = frozenset(
    {"ridge", "lasso", "elasticnet", "huber", "quantile"}
)


def create_linear_model(
    model_type: str,
    *,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    random_state: int = 42,
) -> Any:
    """Build an unfitted sklearn linear estimator."""
    if model_type not in ALLOWED_LINEAR_TYPES:
        raise ValueError(
            f"Unknown LINEAR_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_LINEAR_TYPES)}"
        )
    if model_type == "ols":
        return LinearRegression()
    if model_type == "ridge":
        return Ridge(alpha=alpha, random_state=random_state)
    if model_type == "lasso":
        return Lasso(alpha=alpha, random_state=random_state, max_iter=10_000)
    if model_type == "elasticnet":
        return ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            random_state=random_state,
            max_iter=10_000,
        )
    if model_type == "huber":
        return HuberRegressor(alpha=alpha, max_iter=500)
    if model_type == "quantile":
        return QuantileRegressor(
            quantile=0.5,
            alpha=alpha,
            solver="highs",
        )
    return LogisticRegression(random_state=random_state, max_iter=2_000)


def fit_linear_model(model: Any, X: pd.DataFrame, y: pd.Series) -> Any:
    """Fit ``model`` on training features / target."""
    model.fit(X.to_numpy(dtype=float), np.asarray(y))
    return model


def predict_linear(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Point predictions (class labels for logistic, continuous otherwise)."""
    return model.predict(X.to_numpy(dtype=float))


def select_alpha_on_val(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    alpha_grid: tuple[float, ...] | list[float],
    default_alpha: float,
    l1_ratio: float = 0.5,
    random_state: int = 42,
    objective: str = "oos_r2",
) -> tuple[float, pd.DataFrame]:
    """Pick alpha on validation by maximizing ``objective`` (then min RMSE).

    ``objective`` is one of ``oos_r2`` | ``rank_ic`` | ``quintile_ls``.
    Returns ``(best_alpha, tuning_table)``. For non-tunable models, returns
    ``default_alpha`` and an empty table.
    """
    objective = normalize_tuning_objective(objective)
    if model_type not in ALPHA_TUNABLE_TYPES or not alpha_grid:
        return float(default_alpha), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float]] = []
    for alpha in alpha_grid:
        model = create_linear_model(
            model_type,
            alpha=float(alpha),
            l1_ratio=float(l1_ratio),
            random_state=random_state,
        )
        fit_linear_model(model, X_train, y_train)
        y_hat_train = predict_linear(model, X_train)
        y_hat_val = predict_linear(model, X_val)
        metrics = regression_tuning_metrics(
            y_train,
            y_hat_train,
            y_val,
            y_hat_val,
            train_y_mean=train_mean,
        )
        # All-zero L1 solutions yield constant preds and meaningless IC; deprioritize.
        finite = np.asarray(y_hat_val, dtype=float)
        finite = finite[np.isfinite(finite)]
        degenerate = float(
            finite.size <= 1 or np.unique(finite).size <= 1
        )
        rows.append(
            {
                "alpha": float(alpha),
                **metrics,
                "degenerate": degenerate,
            }
        )
    table = pick_tuning_row(
        pd.DataFrame(rows), objective, prefer_non_degenerate=True
    )
    best_alpha = float(table.loc[0, "alpha"])
    return best_alpha, table
