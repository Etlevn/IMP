"""PyTorch tabular regressors for IMP (dropout MLP / wide-deep / FT-Transformer)."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.metrics import evaluate_regression

# Harmonize with GBDT OpenMP runtimes in the same process.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "torch is required for mlp_dropout / wide_deep / ft_transformer. "
            "Install with: pip install torch"
        ) from exc
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    return torch, nn


@dataclass
class TorchBundle:
    """Fitted torch regressor + metadata needed for prediction."""

    module: Any
    device: str
    n_features: int
    best_iteration: int
    model_type: str


def _build_modules(
    model_type: str,
    n_features: int,
    *,
    hidden_sizes: tuple[int, ...],
    dropout: float,
    ft_d_model: int,
    ft_layers: int,
    ft_heads: int,
):
    torch, nn = _require_torch()

    class _DropoutMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            layers: list[nn.Module] = []
            prev = n_features
            for width in hidden_sizes:
                layers.append(nn.Linear(prev, width))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                prev = width
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):  # noqa: ANN001
            return self.net(x).squeeze(-1)

    class _WideDeep(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.wide = nn.Linear(n_features, 1)
            deep_layers: list[nn.Module] = []
            prev = n_features
            for width in hidden_sizes:
                deep_layers.append(nn.Linear(prev, width))
                deep_layers.append(nn.ReLU())
                deep_layers.append(nn.Dropout(dropout))
                prev = width
            deep_layers.append(nn.Linear(prev, 1))
            self.deep = nn.Sequential(*deep_layers)

        def forward(self, x):  # noqa: ANN001
            return self.wide(x).squeeze(-1) + self.deep(x).squeeze(-1)

    class _FTTransformer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token = nn.Linear(1, ft_d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=ft_d_model,
                nhead=ft_heads,
                dim_feedforward=ft_d_model * 2,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=ft_layers
            )
            self.head = nn.Linear(ft_d_model, 1)

        def forward(self, x):  # noqa: ANN001
            # (batch, n_features) -> (batch, n_features, 1) -> tokens
            tokens = self.token(x.unsqueeze(-1))
            encoded = self.encoder(tokens)
            pooled = encoded.mean(dim=1)
            return self.head(pooled).squeeze(-1)

    if model_type == "mlp_dropout":
        return _DropoutMLP()
    if model_type == "wide_deep":
        return _WideDeep()
    if model_type == "ft_transformer":
        if ft_d_model % ft_heads != 0:
            raise ValueError(
                f"FT_D_MODEL ({ft_d_model}) must be divisible by FT_HEADS ({ft_heads})."
            )
        return _FTTransformer()
    raise ValueError(f"Unknown torch model_type {model_type!r}")


def create_torch_model(
    model_type: str,
    *,
    n_features: int,
    hidden_sizes: tuple[int, ...] = (64, 32),
    dropout: float = 0.1,
    ft_d_model: int = 32,
    ft_layers: int = 2,
    ft_heads: int = 4,
    device: str = "cpu",
) -> TorchBundle:
    """Build an unfitted torch regressor wrapped in ``TorchBundle``."""
    torch, _ = _require_torch()
    module = _build_modules(
        model_type,
        n_features,
        hidden_sizes=hidden_sizes,
        dropout=dropout,
        ft_d_model=ft_d_model,
        ft_layers=ft_layers,
        ft_heads=ft_heads,
    )
    module = module.to(device)
    return TorchBundle(
        module=module,
        device=device,
        n_features=n_features,
        best_iteration=0,
        model_type=model_type,
    )


def fit_torch_regressor(
    bundle: TorchBundle,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    *,
    weight_decay: float = 1e-4,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 200,
    early_stopping_patience: int = 8,
    random_state: int = 42,
) -> TorchBundle:
    """Train with Adam; early-stop on chronological val ``oos_r2``."""
    torch, _ = _require_torch()
    torch.manual_seed(int(random_state))
    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:  # noqa: BLE001
            pass

    device = torch.device(bundle.device)
    module = bundle.module.to(device)
    X_tr = torch.as_tensor(X_train.to_numpy(dtype=np.float32), device=device)
    y_tr = torch.as_tensor(
        np.asarray(y_train, dtype=np.float32).reshape(-1), device=device
    )
    X_va = X_val.to_numpy(dtype=np.float32)
    y_va = np.asarray(y_val, dtype=float)
    train_mean = float(np.asarray(y_train, dtype=float).mean())

    optimizer = torch.optim.Adam(
        module.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    loss_fn = torch.nn.MSELoss()
    n = int(X_tr.shape[0])
    bs = max(1, min(int(batch_size), n))

    best_oos = float("-inf")
    best_rmse = float("inf")
    best_epoch = 1
    best_state = deepcopy(module.state_dict())
    stale = 0

    for epoch in range(1, int(max_epochs) + 1):
        module.train()
        perm = torch.randperm(n, device=device)
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            pred = module(X_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        module.eval()
        with torch.no_grad():
            val_pred = (
                module(torch.as_tensor(X_va, device=device)).detach().cpu().numpy()
            )
        metrics = evaluate_regression(y_va, val_pred, train_y_mean=train_mean)
        oos = float(metrics["oos_r2"])
        rmse = float(metrics["rmse"])
        improved = oos > best_oos + 1e-12 or (
            np.isclose(oos, best_oos) and rmse < best_rmse
        )
        if improved:
            best_oos = oos
            best_rmse = rmse
            best_epoch = epoch
            best_state = deepcopy(module.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(early_stopping_patience):
                break

    module.load_state_dict(best_state)
    module.eval()
    return TorchBundle(
        module=module,
        device=bundle.device,
        n_features=bundle.n_features,
        best_iteration=int(best_epoch),
        model_type=bundle.model_type,
    )


def predict_torch(bundle: TorchBundle, X: pd.DataFrame) -> np.ndarray:
    """Continuous predictions from a fitted ``TorchBundle``."""
    torch, _ = _require_torch()
    device = torch.device(bundle.device)
    module = bundle.module.to(device)
    module.eval()
    with torch.no_grad():
        arr = torch.as_tensor(X.to_numpy(dtype=np.float32), device=device)
        out = module(arr).detach().cpu().numpy()
    return np.asarray(out, dtype=float)
