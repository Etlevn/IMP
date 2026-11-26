"""Neural estimators for the IMP event table (sklearn MLP + PyTorch nets)."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from model.torch_nn import (
    TorchBundle,
    create_torch_model,
    fit_torch_regressor,
    predict_torch,
)
from src.metrics import (
    evaluate_regression,
    is_tuning_improvement,
    normalize_tuning_objective,
    pick_tuning_row,
    regression_tuning_metrics,
    tuning_metric_value,
)

SKLEARN_MLP_TYPES = frozenset({"mlp_shallow", "mlp", "mlp_deep"})
TORCH_NN_TYPES = frozenset({"mlp_dropout", "wide_deep", "ft_transformer"})
ALLOWED_NN_TYPES = SKLEARN_MLP_TYPES | TORCH_NN_TYPES
ALPHA_TUNABLE_TYPES = ALLOWED_NN_TYPES

DEFAULT_HIDDEN_SIZES: dict[str, tuple[int, ...]] = {
    "mlp_shallow": (64,),
    "mlp": (64, 32),
    "mlp_deep": (128, 64, 32),
    "mlp_dropout": (64, 32),
    "wide_deep": (64, 32),
}


def create_nn_model(
    model_type: str,
    *,
    n_features: int | None = None,
    hidden_sizes: dict[str, tuple[int, ...]] | None = None,
    alpha: float = 1e-4,
    learning_rate_init: float = 1e-3,
    batch_size: int = 64,
    max_iter: int = 200,
    dropout: float = 0.1,
    ft_d_model: int = 32,
    ft_layers: int = 2,
    ft_heads: int = 4,
    device: str = "cpu",
    random_state: int = 42,
) -> Any:
    """Build an unfitted neural estimator (sklearn MLP or TorchBundle).

    Does **not** use random ``validation_fraction`` early stopping;
    call ``fit_nn_model`` with the chronological val fold.
    """
    if model_type not in ALLOWED_NN_TYPES:
        raise ValueError(
            f"Unknown NN_MODEL_TYPE {model_type!r}; "
            f"allowed: {sorted(ALLOWED_NN_TYPES)}"
        )
    sizes_map = DEFAULT_HIDDEN_SIZES if hidden_sizes is None else hidden_sizes

    if model_type in TORCH_NN_TYPES:
        if n_features is None or int(n_features) < 1:
            raise ValueError(
                f"{model_type!r} requires n_features >= 1 at create time."
            )
        hidden = tuple(int(x) for x in sizes_map.get(model_type, (64, 32)))
        return create_torch_model(
            model_type,
            n_features=int(n_features),
            hidden_sizes=hidden,
            dropout=float(dropout),
            ft_d_model=int(ft_d_model),
            ft_layers=int(ft_layers),
            ft_heads=int(ft_heads),
            device=str(device),
        )

    if model_type not in sizes_map:
        raise KeyError(f"HIDDEN_SIZES missing entry for {model_type!r}")
    hidden = tuple(int(x) for x in sizes_map[model_type])
    if not hidden or any(h < 1 for h in hidden):
        raise ValueError(f"Invalid hidden sizes for {model_type!r}: {hidden}")

    return MLPRegressor(
        hidden_layer_sizes=hidden,
        activation="relu",
        solver="adam",
        alpha=alpha,
        batch_size=batch_size,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=False,
        warm_start=False,
        random_state=random_state,
    )


def _clone_mlp_template(model: MLPRegressor) -> MLPRegressor:
    return MLPRegressor(
        hidden_layer_sizes=model.hidden_layer_sizes,
        activation=model.activation,
        solver=model.solver,
        alpha=model.alpha,
        batch_size=model.batch_size,
        learning_rate_init=model.learning_rate_init,
        max_iter=1,
        early_stopping=False,
        warm_start=True,
        random_state=model.random_state,
    )


def fit_nn_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    early_stopping_patience: int = 8,
    weight_decay: float | None = None,
    learning_rate_init: float = 1e-3,
    batch_size: int = 64,
    max_iter: int = 200,
    random_state: int = 42,
    objective: str = "oos_r2",
) -> tuple[Any, dict[str, float | int]]:
    """Fit with chronological validation early stopping on ``objective``."""
    objective = normalize_tuning_objective(objective)
    if isinstance(model, TorchBundle):
        fitted = fit_torch_regressor(
            model,
            X_train,
            y_train,
            X_val,
            y_val,
            weight_decay=float(0.0 if weight_decay is None else weight_decay),
            learning_rate=float(learning_rate_init),
            batch_size=int(batch_size),
            max_epochs=int(max_iter),
            early_stopping_patience=int(early_stopping_patience),
            random_state=int(random_state),
            objective=objective,
        )
        return fitted, {"best_iteration": int(fitted.best_iteration)}

    X_tr = X_train.to_numpy(dtype=float)
    y_tr = np.asarray(y_train, dtype=float)
    X_va = X_val.to_numpy(dtype=float)
    y_va = np.asarray(y_val, dtype=float)
    train_mean = float(y_tr.mean())
    max_epochs = int(model.max_iter)

    probe = _clone_mlp_template(model)
    best_epoch = 1
    best_metric = float("-inf")
    best_rmse = float("inf")
    best_model: MLPRegressor | None = None
    stale = 0

    for epoch in range(1, max_epochs + 1):
        probe.partial_fit(X_tr, y_tr)
        metrics = regression_tuning_metrics(
            y_tr,
            probe.predict(X_tr),
            y_va,
            probe.predict(X_va),
            train_y_mean=train_mean,
        )
        if is_tuning_improvement(
            metrics,
            best_metric=best_metric,
            best_rmse=best_rmse,
            objective=objective,
        ):
            best_epoch = epoch
            best_metric = tuning_metric_value(metrics, objective)
            best_rmse = float(metrics["val_rmse"])
            best_model = deepcopy(probe)
            stale = 0
        else:
            stale += 1
            if stale >= early_stopping_patience:
                break

    if best_model is None:
        raise RuntimeError("MLP early stopping failed to record any weights.")

    best_model.n_iter_ = best_epoch
    best_model.set_params(max_iter=max_epochs, warm_start=False)
    return best_model, {"best_iteration": int(best_epoch)}


def predict_nn(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Continuous predictions."""
    if isinstance(model, TorchBundle):
        return predict_torch(model, X)
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
    hidden_sizes: dict[str, tuple[int, ...]] | None = None,
    learning_rate_init: float = 1e-3,
    batch_size: int = 64,
    max_iter: int = 200,
    early_stopping_patience: int = 8,
    dropout: float = 0.1,
    ft_d_model: int = 32,
    ft_layers: int = 2,
    ft_heads: int = 4,
    device: str = "cpu",
    random_state: int = 42,
    objective: str = "oos_r2",
) -> tuple[float, pd.DataFrame]:
    """Pick L2 / weight-decay ``alpha`` on val by maximizing ``objective``."""
    objective = normalize_tuning_objective(objective)
    if model_type not in ALPHA_TUNABLE_TYPES or not alpha_grid:
        return float(default_alpha), pd.DataFrame()

    train_mean = float(np.asarray(y_train, dtype=float).mean())
    rows: list[dict[str, float]] = []
    n_features = int(X_train.shape[1])
    for alpha in alpha_grid:
        model = create_nn_model(
            model_type,
            n_features=n_features,
            hidden_sizes=hidden_sizes,
            alpha=float(alpha),
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            dropout=dropout,
            ft_d_model=ft_d_model,
            ft_layers=ft_layers,
            ft_heads=ft_heads,
            device=device,
            random_state=random_state,
        )
        fitted, _ = fit_nn_model(
            model,
            X_train,
            y_train,
            X_val=X_val,
            y_val=y_val,
            early_stopping_patience=early_stopping_patience,
            weight_decay=float(alpha) if model_type in TORCH_NN_TYPES else None,
            learning_rate_init=learning_rate_init,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=random_state,
            objective=objective,
        )
        metrics = regression_tuning_metrics(
            y_train,
            predict_nn(fitted, X_train),
            y_val,
            predict_nn(fitted, X_val),
            train_y_mean=train_mean,
        )
        rows.append({"alpha": float(alpha), **metrics})
    table = pick_tuning_row(pd.DataFrame(rows), objective)
    best_alpha = float(table.loc[0, "alpha"])
    return best_alpha, table
