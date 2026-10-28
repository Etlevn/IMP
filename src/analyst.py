"""Derived analyst-rating features from recommendation-count columns.

Source columns are fixed to the sentiment panel field names. Formulas
(Coverage == 0 or non-finite -> NaN for ratio / score columns):

    Coverage    = SB + B + H + S + SS
    BuyRatio    = (SB + B) / Coverage
    SellRatio   = (S + SS) / Coverage
    RatingScore = (2*SB + B - S - 2*SS) / Coverage
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.default import ANALYST_DERIVED_FEATURES

# Fixed parquet / panel field names after sentiment merge.
COL_STRONG_BUY = "analystRatingsStrongBuy"
COL_BUY = "analystRatingsBuy"
COL_HOLD = "analystRatingsHold"
COL_SELL = "analystRatingsSell"
COL_STRONG_SELL = "analystRatingsStrongSell"
COUNT_COLUMNS = (COL_STRONG_BUY, COL_BUY, COL_HOLD, COL_SELL, COL_STRONG_SELL)

DERIVED_COVERAGE = "Coverage"
DERIVED_BUY_RATIO = "BuyRatio"
DERIVED_SELL_RATIO = "SellRatio"
DERIVED_RATING_SCORE = "RatingScore"

ALL_DERIVED = (
    DERIVED_COVERAGE,
    DERIVED_BUY_RATIO,
    DERIVED_SELL_RATIO,
    DERIVED_RATING_SCORE,
)


def _count_arrays(frame: pd.DataFrame) -> tuple[np.ndarray, ...]:
    missing = [c for c in COUNT_COLUMNS if c not in frame.columns]
    if missing:
        raise KeyError(
            f"Panel missing analyst count columns: {missing}. "
            "Merge sentiment before attach_analyst_features."
        )
    return tuple(
        pd.to_numeric(frame[c], errors="coerce").to_numpy(dtype=float) for c in COUNT_COLUMNS
    )


def compute_analyst_features(
    frame: pd.DataFrame,
    *,
    derived_features: list[str] | None = None,
) -> pd.DataFrame:
    """Return a frame with requested derived analyst columns added."""
    wanted = list(ANALYST_DERIVED_FEATURES if derived_features is None else derived_features)
    if not wanted:
        return frame.copy()

    unknown = [name for name in wanted if name not in ALL_DERIVED]
    if unknown:
        raise ValueError(
            f"Unknown ANALYST_DERIVED_FEATURES {unknown}; allowed: {list(ALL_DERIVED)}."
        )

    sb, b, h, s, ss = _count_arrays(frame)
    stacked = np.vstack([sb, b, h, s, ss])
    # All five missing => no snapshot. Otherwise treat missing buckets as 0.
    all_missing = ~np.isfinite(stacked).any(axis=0)
    sb0 = np.where(np.isfinite(sb), sb, 0.0)
    b0 = np.where(np.isfinite(b), b, 0.0)
    h0 = np.where(np.isfinite(h), h, 0.0)
    s0 = np.where(np.isfinite(s), s, 0.0)
    ss0 = np.where(np.isfinite(ss), ss, 0.0)
    coverage = sb0 + b0 + h0 + s0 + ss0
    coverage = np.where(all_missing, np.nan, coverage)
    valid = np.isfinite(coverage) & (coverage > 0)

    out = frame.copy()
    drop_cols = [c for c in wanted if c in out.columns]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    if DERIVED_COVERAGE in wanted:
        out[DERIVED_COVERAGE] = coverage

    if DERIVED_BUY_RATIO in wanted:
        buy_ratio = np.full(len(out), np.nan, dtype=float)
        buy_ratio[valid] = (sb0[valid] + b0[valid]) / coverage[valid]
        out[DERIVED_BUY_RATIO] = buy_ratio

    if DERIVED_SELL_RATIO in wanted:
        sell_ratio = np.full(len(out), np.nan, dtype=float)
        sell_ratio[valid] = (s0[valid] + ss0[valid]) / coverage[valid]
        out[DERIVED_SELL_RATIO] = sell_ratio

    if DERIVED_RATING_SCORE in wanted:
        score = np.full(len(out), np.nan, dtype=float)
        score[valid] = (
            2.0 * sb0[valid] + b0[valid] - s0[valid] - 2.0 * ss0[valid]
        ) / coverage[valid]
        out[DERIVED_RATING_SCORE] = score

    return out


def attach_analyst_features(
    frame: pd.DataFrame,
    *,
    derived_features: list[str] | None = None,
) -> pd.DataFrame:
    """Attach derived analyst features onto a daily / aligned panel."""
    return compute_analyst_features(frame, derived_features=derived_features)
