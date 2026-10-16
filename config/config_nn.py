"""Neural model settings for the IMP event table."""

from __future__ import annotations

from pathlib import Path

from src.features import FEATURE_COLUMNS as DEFAULT_FEATURE_COLUMNS

# Paths (do not change unless the repo layout changes).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"  # parquet inputs
OUTPUT_DIR = PROJECT_ROOT / "output" / "nn"  # run outputs

# Batch label for run_config / logs. Any non-empty string.
MODEL_LABEL = "nn_batch"

# Models to run under the same preprocess config (one section each).
# Allowed: "mlp_shallow" | "mlp" | "mlp_deep"
NN_MODEL_TYPES = ("mlp_shallow", "mlp", "mlp_deep")

# Hidden widths keyed by model type (sklearn MLPRegressor).
HIDDEN_SIZES = {
    "mlp_shallow": (64,),
    "mlp": (64, 32),
    "mlp_deep": (128, 64, 32),
}

# Default L2 penalty when ALPHA_GRID is empty.
# Allowed: float >= 0
ALPHA = 1e-4

# Candidate L2 penalties tuned on chronological validation (all MLP types).
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
