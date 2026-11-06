# Features

Columns that can enter a model. Source tables are in [`dataset.md`](dataset.md). Dates and labels are in [`fields.md`](fields.md).

A group is attached only when its switch in `config/default.py` lists the name (`ANALYST_FEATURES`, `MACRO_FEATURES`, `ENGINEERED_FEATURES`). The original eight are already on the panel. A column is copied onto the event row from the `DecisionDate` close, and a model sees it only when that name is also in `FEATURE_COLUMNS` or in a `main_feat.py` feature set.

Attach order on the aligned panel: benchmark, then macro, then derived analyst ratings, then engineered columns. `par*vix` needs `vix` on the panel, so `vix` stays in `MACRO_FEATURES` even when it is not a model input.

---

## Price file

Copied from `stock-data_us.parquet` on the decision day. Not recomputed.

| Feature | Definition | Note |
| --- | --- | --- |
| `QScoreTrendLong` | Long-horizon trend score on that row. | Vendor field. |
| `QScoreSpeedLong` | Long-horizon speed / momentum score on that row. | Vendor field. |
| `QScoreLong` | `QScoreTrendLong + QScoreSpeedLong` on observed rows. | Redundant when Trend and Speed are both in the model. |

## Analyst counts

Copied from the monthly sentiment file after a backward `merge_asof` (31-day tolerance). The decision-day value is the latest snapshot on or before that date inside the window. Missing when no snapshot qualifies.

| Feature | Definition | Note |
| --- | --- | --- |
| `analystRatingsStrongBuy` | Strong Buy count in the snapshot. | Raw count. |
| `analystRatingsBuy` | Buy count. | Raw count. |
| `analystRatingsHold` | Hold count. | Raw count. |
| `analystRatingsSell` | Sell count. | Raw count. |
| `analystRatingsStrongSell` | Strong Sell count. | Raw count. |

## Derived analyst ratings

Built in `src/features/analyst.py` from the five counts. If every count is missing, all four derived columns are missing. If at least one count is present, a missing bucket is treated as 0. Ratios and `RatingScore` are missing when coverage is 0.

Write \(SB, B, H, S, SS\) for the five counts.

| Feature | Definition | Note |
| --- | --- | --- |
| `Coverage` | \(SB + B + H + S + SS\) | Number of ratings in the snapshot. |
| `BuyRatio` | \((SB + B) / Coverage\) | Share of buy-side ratings. |
| `SellRatio` | \((S + SS) / Coverage\) | Share of sell-side ratings. Hold is in the denominator only. |
| `RatingScore` | \((2\,SB + B - S - 2\,SS) / Coverage\) | Average signed rating. Range is \([-2, 2]\). |

## Macro levels

Yahoo daily close, left-merged on `Date` in `src/features/macro.py`. Cached under `data/macro_daily.*`. These are levels, not returns. A stock row keeps a missing value when that calendar date has no macro print.

| Feature | Definition | Note |
| --- | --- | --- |
| `vix` | Close of `^VIX`. | CBOE volatility index level. |
| `tnx` | Close of `^TNX`. | Yahoo 10-year Treasury yield field. |
| `dxy` | Close of `DX-Y.NYB`. | U.S. dollar index level. |

## Engineered

Built in `src/features/engineered.py` from prices, market cap, the benchmark, and `vix`. Windows use trading days within each `TickerId`. No row uses a later close.

| Feature | Definition | Note |
| --- | --- | --- |
| `mom_21d` | `LastPrice` percent change over 21 trading days. | Same-name history only. |
| `vol_21d` | Sample standard deviation (`ddof=1`) of 21 daily simple returns. | Needs 21 returns; earlier rows are missing. |
| `log_mcap` | Natural log of `MarketCap`. | Missing when market cap is missing or not positive. |
| `par_mkt` | Stock return from the close at `AnchorDate−5` to the close at `AnchorDate−2`, minus the benchmark return over that same window. | Stored only on the decision row (`AnchorDate−2`). Same pre-announcement window as the reversal comparison. |
| `par*vix` | `par_mkt × vix` on that decision row. | Missing unless both factors are finite. The product uses the raw VIX level. |

`par_mkt` uses `BenchmarkAdjClose`. That series is the configured benchmark (`BENCHMARK_SOURCE`: `equal`, `cap`, or Yahoo `MARKET_TICKER`), not a fixed SPY series. The default in `config/default.py` is `equal`.
