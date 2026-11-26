"""Neural model settings for the IMP event table.

Shared paths / data / split / target defaults: ``config/config.py``.
"""

from __future__ import annotations

from config.config import (
    BENCHMARK_SOURCE,
    DATA_ROOT,
    IMPUTE_STRATEGY,
    MARKET_TICKER,
    OUTPUT_DIR,
    RANDOM_STATE,
    SPLIT_COLUMN,
    TARGET,
    TRAIN_END,
    VAL_END,
)
from src.features import enabled_features

# Snapshot of enabled catalog at import (ORIGINAL → ANALYST → MACRO → ENGINEERED).
FEATURE_COLUMNS = enabled_features()

# Label for config / logs. Any non-empty string.
MODEL_LABEL = "nn"

# Allowed: mlp_shallow | mlp | mlp_deep | mlp_dropout | wide_deep | ft_transformer
NN_MODEL_TYPES = (
    "mlp_shallow",
    "mlp",
    "mlp_deep",
    "mlp_dropout",
    "wide_deep",
    "ft_transformer",
)

# Hidden widths keyed by model type (sklearn MLP + torch MLP/wide_deep).
HIDDEN_SIZES = {
    "mlp_shallow": (64,),
    "mlp": (64, 32),
    "mlp_deep": (128, 64, 32),
    "mlp_dropout": (64, 32),
    "wide_deep": (64, 32),
}

# Default L2 / Adam weight-decay when ALPHA_GRID is empty.
# Allowed: float >= 0
ALPHA = 1e-4

# Candidate L2 / weight-decay values tuned on chronological validation.
# Allowed: non-empty sequence of floats >= 0, or () to use ALPHA as-is.
ALPHA_GRID = (1e-5, 1e-4, 1e-3, 1e-2)

# Adam learning rate. Allowed: float > 0
LR = 1e-3

# Mini-batch size. Allowed: int >= 1
BATCH_SIZE = 64

# Maximum training iterations (epochs). Allowed: int >= 1
MAX_ITER = 200

# Stop if chronological val OOS R² does not improve for this many epochs.
# Allowed: int >= 1
EARLY_STOPPING_PATIENCE = 8

# Dropout rate for torch nets. Allowed: float in [0, 1)
DROPOUT = 0.1

# Lightweight FT-Transformer dims (feature-token attention).
FT_D_MODEL = 32
FT_LAYERS = 2
FT_HEADS = 4

# Torch device. Allowed: "cpu" | "cuda" | "mps" (as supported by torch)
DEVICE = "cpu"

# Z-score features with train-set mean/std. Allowed: True | False
STANDARDIZE = True
