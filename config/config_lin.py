"""Linear model settings for the IMP event table."""

from __future__ import annotations

from pathlib import Path

from src.features import FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS

# Paths (do not change unless the repo layout changes).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"  # parquet inputs
OUTPUT_DIR = PROJECT_ROOT / "output" / "lin"  # run outputs

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "lin_batch"

# Models to run under the same preprocess config (one section each).
# Allowed entries: "ols" | "ridge" | "lasso"
# ("logistic" reserved until classification metrics exist.)
LINEAR_MODEL_TYPES = ("ols", "ridge", "lasso")

# Default L2/L1 strength when ALPHA_GRID is empty / unused.
# Allowed: float > 0. Ignored for ols.
ALPHA = 1.0

# Candidate alphas tuned on validation only (ridge/lasso).
# Allowed: non-empty sequence of floats > 0, or () to use ALPHA as-is.
ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)

# Prediction target column on event table ``df``.
# Allowed: "ret_ann" | "ret_ann_mkt" | "ret_next" | "ret_next_mkt"
#          "ret_post" | "ret_post_mkt" | "dir_ann" | "vol_ann" | "vol_post"
TARGET = "ret_ann_mkt"

# Input feature columns. Allowed: any non-empty subset of DEFAULT_FEATURE_COLUMNS
# (QScore* and analystRatings*).
FEATURE_COLUMNS = list(DEFAULT_FEATURE_COLUMNS)

# Date column used for the chronological split.
# Allowed: "EventDate" | "decision_session" | "anchor_session" | "release_session"
SPLIT_COLUMN = "EventDate"

# Inclusive calendar cuts as "YYYY-MM-DD".
# train: <= TRAIN_END; val: TRAIN_END < d <= VAL_END; test: > VAL_END
# Require TRAIN_END < VAL_END.
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

# Missing-value handling on features (fit on train only).
# Allowed: "median" | "mean" | "drop"
IMPUTE_STRATEGY = "median"

# Z-score features with train-set mean/std. Allowed: True | False
STANDARDIZE = True

# RNG seed for any stochastic step. Allowed: int >= 0
RANDOM_STATE = 42

# In-model feature importance via column permutation on the test fold.
# Allowed: True | False
FEATURE_IMPORTANCE = True

# Number of shuffle repeats per feature. Allowed: int >= 1
FEATURE_IMPORTANCE_REPEATS = 10
