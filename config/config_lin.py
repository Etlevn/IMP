"""Linear model settings for the IMP event table.

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
MODEL_LABEL = "lin"

# Allowed: ols | ridge | lasso | elasticnet | huber | quantile
# ("logistic" reserved until classification metrics exist.)
LINEAR_MODEL_TYPES = (
    "ols",
    "ridge",
    "lasso",
    "elasticnet",
    "huber",
    "quantile",
)

# Default L2/L1 strength when ALPHA_GRID is empty / unused.
# Allowed: float > 0. Ignored for ols.
ALPHA = 1.0

# Candidate alphas tuned on validation only (ridge/lasso/elasticnet/huber/quantile).
# Allowed: non-empty sequence of floats > 0, or () to use ALPHA as-is.
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# ElasticNet mixing parameter (fixed this round; not grid-searched).
# Allowed: float in (0, 1].
L1_RATIO = 0.5

# Z-score features with train-set mean/std. Allowed: True | False
STANDARDIZE = True
