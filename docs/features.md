# Features

Columns that can enter a model. Source tables are in [`dataset.md`](dataset.md). Dates and labels are in [`fields.md`](fields.md).

Register under `ORIGINAL_FEATURES` / `ANALYST_FEATURES` / `MACRO_FEATURES` / `IBES_FEATURES` / `COMPUSTAT_FEATURES` / `ENGINEERED_FEATURES` (comment out to disable). `src.features.enabled_features()` concatenates in that order. Yahoo/WRDS groups are **cache-only** at attach time — download first with `script/fetch_macro.py`, `script/fetch_wrds_ibes.py`, `script/fetch_wrds_compustat.py`. `feat_filter.py` screens the catalog into a `suggest` column. `feat_optuna.py` builds its pool as `FEATURE_WHITELIST` first, then feat_filter labels enabled by `INCLUDE_SUGGEST`, excluding `FEATURE_BLACKLIST`. `main.py` trains from `config_main.py` `MODELS` (family → model → one or more feature lists; empty skips). Expanded model menus: lin adds `elasticnet` / `huber` / `quantile`; tr adds `lightgbm` / `xgboost` / `catboost`; nn adds torch `mlp_dropout` / `wide_deep` / `ft_transformer` (see `docs/superpowers/specs/2026-11-25-nine-model-expansion-design.md`). Optuna selection: `GAP_MODE` `none` | `abs` | `pos` with `GAP_PENALTY` on val `OBJECTIVE` (`oos_r2` / `rank_ic` / `quintile_ls`); pools ≤ `EXHAUSTIVE_MAX` are fully enumerated.

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

Yahoo daily close, left-merged on `Date` in `src/features/macro.py`. Cached under `data/macro_daily.*` after `python script/fetch_macro.py`. Attach does not call yfinance. Levels, not returns; missing calendar days stay NaN.

## IBES (WRDS)

Fetch: `python script/export_universe.py` then `python script/fetch_wrds_ibes.py` → `data/ibes_features.*` (STATPERS consensus) and `data/ibes_sue_features.*` (quarterly SUE/beat). Consensus/revisions as-of on `statpers`. SUE/beat are built on `(TickerId, fpedats)` events (consensus from last `statpers < anndats_act`), become usable on `available_on = anndats_act + 1` calendar day, and are as-of joined **directly** onto the trading-day panel (not gated on the next STATPERS). `sue_lag1` / `sue_lag2` are the most recent and second-most-recent announced-quarter SUE. PIT regressions: `pytest tests/test_ibes_sue_pit.py`. See `IBES_FEATURES` in `config/config.py`.

## Compustat (WRDS)

Fetch: `python script/fetch_wrds_compustat.py` → `data/compustat_features.*`. As-of on report availability (`rdq` or `datadate+45`). Ratios such as `bm`, `roe`, `leverage`, growth, accruals (see `COMPUSTAT_FEATURES`).

| Feature | Definition | Note |
| --- | --- | --- |
| `vix` | Close of `^VIX`. | CBOE volatility index level. |
| `tnx` | Close of `^TNX`. | 10-year Treasury yield. |
| `dxy` | Close of `DX-Y.NYB`. | U.S. dollar index. |
| `spy` | Close of `SPY`. | S&P 500 ETF. |
| `qqq` | Close of `QQQ`. | Nasdaq-100 ETF. |
| `iwm` | Close of `IWM`. | Russell 2000 ETF. |
| `irx` | Close of `^IRX`. | 13-week T-bill yield. |
| `hyg` | Close of `HYG`. | High-yield corporate bond ETF. |
| `oil` | Close of `CL=F`. | WTI crude futures. |
| `gold` | Close of `GC=F`. | Gold futures. |
| `tyx` | Close of `^TYX`. | 30-year Treasury yield. |
| `lqd` | Close of `LQD`. | Investment-grade corporate bond ETF. |
| `copper` | Close of `HG=F`. | Copper futures. |
| `move` | Close of `^MOVE`. | ICE BofA MOVE rate-vol index. |
| `vnq` | Close of `VNQ`. | U.S. REIT ETF. |

## Engineered

Built in `src/features/engineered.py` from prices, volume, market cap, and the benchmark. Windows use trading days within each `TickerId`. No row uses a later close.

| Feature | Definition | Note |
| --- | --- | --- |
| `mom_5d` / `mom_21d` / `mom_63d` | `LastPrice` percent change over W trading days. | Same-name history only. |
| `vol_5d` / `vol_21d` / `vol_63d` | Sample std (`ddof=1`) of W daily simple returns. | Needs a full W-return window. |
| `rel_vol_21d` | `Volume` / mean volume of the **prior 21** trading days (**excludes today**). | Missing until 21 prior days exist. |
| `amihud_21d` | 21-day mean of `|ret_1d| / (LastPrice * Volume)`. | Non-positive dollar volume → missing that day. |
| `hl_range_21d` | 21-day mean of `(High - Low) / LastPrice`. | Uses session high/low. |
| `gap_1d` | `(Open - previous close) / previous close`. | First row per ticker missing. |
| `mom_exmkt_5d` / `mom_exmkt_21d` / `mom_exmkt_63d` | Stock W-day return minus same-window `BenchmarkAdjClose` return. | Requires benchmark attach first. |
| `log_mcap` | Natural log of `MarketCap`. | Missing when market cap is missing or not positive. |
| `par_mkt` | Stock return from the close at `AnchorDate−5` to the close at `AnchorDate−2`, minus the benchmark return over that same window. | Stored only on the decision row (`AnchorDate−2`). Same pre-announcement window as the reversal comparison. |

`par_mkt` and `mom_exmkt_*` use `BenchmarkAdjClose`. That series is the configured benchmark (`BENCHMARK_SOURCE`: `equal`, `cap`, or Yahoo `MARKET_TICKER`), not a fixed SPY series. The default in `config/config.py` is `equal`.
