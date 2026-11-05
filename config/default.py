"""Global defaults shared by lin / tr / nn (and scripts).

Model-family overrides stay in ``config/config_*.py``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- paths ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_ROOT = DATA_DIR  # alias used by model configs
# Shared run artifacts (metrics / predictions / run_config); no per-family subdirs.
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- source parquet filenames under DATA_DIR ---
STOCK_FILE = "stock-data_us.parquet"
EARNINGS_FILE = "earnings_us.parquet"
SENTIMENT_FILE = "historical_sentiment_scores_us.parquet"
# Backward asof window for monthly analyst snapshots (calendar days).
SENTIMENT_TOLERANCE_DAYS = 31

# --- benchmark (market-adjusted returns) ---
# Source: "yahoo" | "equal" | "cap"
# - yahoo: external ETF via yfinance (uses MARKET_TICKER)
# - equal: equal-weighted index from the ~200-name panel (LastPrice)
# - cap: same-day MarketCap-weighted index from the panel
BENCHMARK_SOURCE = "equal"
# Yahoo ticker when BENCHMARK_SOURCE == "yahoo"; ignored for equal / cap.
# Common: SPY (default), QQQ, IWM, DIA, VTI.
MARKET_TICKER = "SPY"

# --- feature groups (formulas in src/features/) ---
# Each group is a switch: an empty list or dict skips that attach.
# A name enters a model only after it is also listed in FEATURE_COLUMNS.

# Allowed: Coverage | BuyRatio | SellRatio | RatingScore.
ANALYST_FEATURES = [
    "Coverage",
    "BuyRatio",
    "SellRatio",
    "RatingScore",
]

# Maps feature column name -> yfinance ticker. Cached under data/macro_daily.parquet (+ .json).
MACRO_FEATURES = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
}

# Allowed: mom_21d | vol_21d | log_mcap | par_mkt.
# par_mkt is the reversal-comparison window: close at anchor−5 through close at
# anchor−2, stock minus benchmark, stored on the decision-day row.
ENGINEERED_FEATURES = [
    "mom_21d",
    "vol_21d",
    "log_mcap",
    "par_mkt",
]

# --- experiment ---
# Allowed: ret_ann | ret_ann_mkt | ret_next | ret_next_mkt |
#          ret_post | ret_post_mkt | dir_ann | vol_ann | vol_post
TARGET = "ret_ann_mkt"

# Active feature list for main / config_* / scripts.
# Original 8: QScoreTrendLong, QScoreSpeedLong, QScoreLong, analystRatingsStrongBuy, analystRatingsBuy, analystRatingsHold, analystRatingsSell, analystRatingsStrongSell
FEATURE_COLUMNS = [
    "QScoreTrendLong",
    "QScoreSpeedLong",
    # "QScoreLong",
    # "analystRatingsStrongBuy",
    # "analystRatingsBuy",
    # "analystRatingsHold",
    # "analystRatingsSell",
    # "analystRatingsStrongSell",
    "Coverage",
    # "BuyRatio",
    # "SellRatio",
    "RatingScore",
    # "vix",
    # "tnx",
    # "dxy",
    "mom_21d",
    "vol_21d",
    "log_mcap",
    "par_mkt",
]

# Allowed: EventDate | DecisionDate | AnchorDate | ReleaseDate
SPLIT_COLUMN = "DecisionDate"

# Inclusive calendar cuts (YYYY-MM-DD).
# train: <= TRAIN_END; val: TRAIN_END < d <= VAL_END; test: > VAL_END
TRAIN_END = "2022-12-31"
VAL_END = "2023-12-31"

# Feature missing values (fit on train only): median | mean | drop
IMPUTE_STRATEGY = "median"

RANDOM_STATE = 42
FEATURE_IMPORTANCE = True
FEATURE_IMPORTANCE_REPEATS = 10
