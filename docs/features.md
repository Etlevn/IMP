# Features

Columns that can enter a model. Source tables are in [`dataset.md`](dataset.md). Dates and labels are in [`fields.md`](fields.md).

Two layers: implement under `src/features/`, then list the name in `ORIGINAL_FEATURES` / `ANALYST_FEATURES` / `MACRO_FEATURES` / `ENGINEERED_FEATURES` (comment out to disable). `src.features.enabled_features()` concatenates those groups in that order for `feat_filter` / `feat_optuna` / catalog defaults. `feat_filter.py` screens the catalog into a `suggest` column. `feat_optuna.py` builds its pool as `FEATURE_WHITELIST` first, then feat_filter labels enabled by `INCLUDE_SUGGEST`, excluding `FEATURE_BLACKLIST`. `main.py` trains from `config_main.py` `MODELS` (family → model → one or more feature lists; empty skips). Optuna selection: `GAP_MODE` `none` | `abs` | `pos` with `GAP_PENALTY` on val `OBJECTIVE` (`oos_r2` / `rank_ic` / `quintile_ls`); pools ≤ `EXHAUSTIVE_MAX` are fully enumerated.

Attach order on the aligned panel: benchmark, then macro, then derived analyst ratings, then engineered columns.

---

## Price file

Copied from `stock_data_us.parquet` on the decision day. Not recomputed.

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

Built in `src/features/engineered.py` from prices, market cap, and the benchmark. Windows use trading days within each `TickerId`. No row uses a later close.

| Feature | Definition | Note |
| --- | --- | --- |
| `mom_21d` | `LastPrice` percent change over 21 trading days. | Same-name history only. |
| `vol_21d` | Sample standard deviation (`ddof=1`) of 21 daily simple returns. | Needs 21 returns; earlier rows are missing. |
| `log_mcap` | Natural log of `MarketCap`. | Missing when market cap is missing or not positive. |
| `par_mkt` | Stock return from the close at `AnchorDate−5` to the close at `AnchorDate−2`, minus the benchmark return over that same window. | Stored only on the decision row (`AnchorDate−2`). Same pre-announcement window as the reversal comparison. |

`par_mkt` uses `BenchmarkAdjClose`. That series is the configured benchmark (`BENCHMARK_SOURCE`: `equal`, `cap`, or Yahoo `MARKET_TICKER`), not a fixed SPY series. The default in `config/config.py` is `equal`.
