"""Tree model settings for the IMP event table.

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
MODEL_LABEL = "tr"

# Allowed: decision_tree | random_forest | hist_gradient_boosting |
#          lightgbm | xgboost | catboost
TREE_MODEL_TYPES = (
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
    "lightgbm",
    "xgboost",
    "catboost",
)

# Number of trees / boosting stages (cap for early stopping).
# Allowed: int >= 1. Ignored for decision_tree.
N_ESTIMATORS = 200

# Default max depth when MAX_DEPTH_GRID is empty.
# Allowed: int >= 1
MAX_DEPTH = 4

# Candidate depths tuned on chronological validation (all tree types).
# Allowed: non-empty sequence of ints >= 1, or () to use MAX_DEPTH as-is.
MAX_DEPTH_GRID = (2, 3, 4, 6, 8)

# Boosting step size. Allowed: float in (0, 1]. Ignored for non-boosted trees.
LEARNING_RATE = 0.05

# Minimum samples in a leaf. Allowed: int >= 1
MIN_SAMPLES_LEAF = 20

# Boosted trees: stop if chronological val metric does not improve.
# Allowed: int >= 1
EARLY_STOPPING_PATIENCE = 10

# Z-score features. Allowed: True | False (usually False for trees)
STANDARDIZE = False
