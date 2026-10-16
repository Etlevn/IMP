"""Tree model settings for the IMP event table."""

from __future__ import annotations

from pathlib import Path

from src.features import FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS

# Paths (do not change unless the repo layout changes).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"  # parquet inputs
OUTPUT_DIR = PROJECT_ROOT / "output" / "tr"  # run outputs

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "tr_batch"

# Models to run under the same preprocess config (one section each).
# Allowed: "decision_tree" | "random_forest" | "hist_gradient_boosting"
TREE_MODEL_TYPES = (
    "decision_tree",
    "random_forest",
    "hist_gradient_boosting",
)

# Number of trees / boosting stages (cap for HistGBDT early stopping).
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

# HistGBDT: stop if chronological val OOS R² does not improve.
# Allowed: int >= 1
EARLY_STOPPING_PATIENCE = 10

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
# Median so decision_tree / random_forest (no native NaN) share one preprocess.
IMPUTE_STRATEGY = "median"

# Z-score features. Allowed: True | False (usually False for trees)
STANDARDIZE = False

# RNG seed for any stochastic step. Allowed: int >= 0
RANDOM_STATE = 42

# In-model feature importance via column permutation on the test fold.
# Allowed: True | False
FEATURE_IMPORTANCE = True

# Number of shuffle repeats per feature. Allowed: int >= 1
FEATURE_IMPORTANCE_REPEATS = 10
