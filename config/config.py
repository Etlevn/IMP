"""Global defaults shared by lin / tr / nn (and scripts).

Model-family overrides stay in ``config/config_*.py``.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- paths ---
DATA_DIR = PROJECT_ROOT / "data"
DATA_ROOT = DATA_DIR  # alias used by model configs
# Shared run artifacts (metrics / predictions / config); no per-family subdirs.
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- source parquet filenames under DATA_DIR ---
# Expanded panel: 2019-01-02 through 2025-12-31, about 800 names.
# Sentiment stays on the original ~200-symbol monthly file.
STOCK_FILE = "stock_data_us_expanded-2019-2026.parquet"  # "stock-data_us.parquet", "stock_data_us_expanded-2019-2026.parquet"
EARNINGS_FILE = "earnings_us_expanded-2019-2026.parquet"  # "earnings_us.parquet", "earnings_us_expanded-2019-2026.parquet"
SENTIMENT_FILE = "historical_sentiment_scores_us.parquet"
# Backward asof window for monthly analyst snapshots (calendar days).
SENTIMENT_TOLERANCE_DAYS = 31

# --- benchmark (market-adjusted returns) ---
# Source: "yahoo" | "equal" | "cap"
# - yahoo: external ETF via yfinance (requires MARKET_TICKER)
# - equal: equal-weighted index from the ~200-name panel (LastPrice)
# - cap: same-day MarketCap-weighted index from the panel
BENCHMARK_SOURCE = "equal"
# Yahoo ticker when BENCHMARK_SOURCE == "yahoo" (e.g. SPY, QQQ, IWM, DIA, VTI).
# May be "" / None when source is equal or cap (ignored). Required for yahoo.
MARKET_TICKER = ""

# --- features (register = enable) ---
# 1) Implement derived formulas under src/features/.
# 2) List them in the groups below. Comment out a name to disable it.
#    Empty ANALYST / MACRO / ENGINEERED = do not attach that whole group.
# 3) feat_filter / feat_optuna / main use src.features.enabled_features()
#    (ORIGINAL → ANALYST → MACRO → ENGINEERED). Panel-native vendor columns belong in ORIGINAL_FEATURES (not computed in src/features/).

ORIGINAL_FEATURES = [
    "QScoreTrendLong",
    "QScoreSpeedLong",
    "QScoreLong",
    "analystRatingsStrongBuy",
    "analystRatingsBuy",
    "analystRatingsHold",
    "analystRatingsSell",
    "analystRatingsStrongSell",
]

# Allowed: Coverage | BuyRatio | SellRatio | RatingScore (src/features/analyst.py).
ANALYST_FEATURES = [
    "Coverage",
    "BuyRatio",
    "SellRatio",
    "RatingScore",
]

# Maps feature column name -> yfinance ticker (src/features/macro.py).
# Cached under data/macro_daily.parquet (+ .json).
MACRO_FEATURES = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
}

# Allowed: mom_21d | vol_21d | log_mcap | par_mkt (src/features/engineered.py).
# par_mkt is the reversal-comparison window: close at anchor−5 through close at anchor−2, stock minus benchmark, stored on the decision-day row.
ENGINEERED_FEATURES = [
    "mom_21d",
    "vol_21d",
    "log_mcap",
    "par_mkt",
]

# --- experiment ---
# Allowed: ret_ann | ret_ann_mkt | ret_next | ret_next_mkt |
#          ret_post | ret_post_mkt | dir_ann | vol_ann | vol_post
TARGET = "ret_ann_mkt"  # "ret_ann"

# Forced into feat_optuna's search pool before feat_filter suggest labels.
# Must be a subset of enabled_features(). Empty = no whitelist.
FEATURE_WHITELIST = []

# Excluded from feat_optuna's search pool (even if suggested or whitelisted).
# Must be a subset of enabled_features(); must not overlap FEATURE_WHITELIST.
# Empty = no blacklist.
FEATURE_BLACKLIST = []

# Allowed: EventDate | DecisionDate | AnchorDate | ReleaseDate
SPLIT_COLUMN = "DecisionDate"

# Inclusive calendar cuts (YYYY-MM-DD).
# train: <= TRAIN_END; val: TRAIN_END < d <= VAL_END; test: > VAL_END
TRAIN_END = "2022-12-31"  
VAL_END = "2023-12-31"  

# Feature missing values (fit on train only): median | mean | drop
IMPUTE_STRATEGY = "median"

RANDOM_STATE = 42
