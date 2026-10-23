# Fields

Columns built (or snapped) when constructing the announcement table `df` from `df_all`.  
Source Parquet columns are documented in `dataset.md`.

Pipeline: `df_all` → session align → anchors → **decision-day features** + **targets** → `df` (one row per earnings event).

---

## Session / calendar fields

Built from `EventDate`, the trading calendar, and `MarketTimeId`.

| Column             | Description                                                                                                                                          |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EventDate`        | Original earnings calendar date from the earnings table (carried onto `df`).                                                                         |
| `release_session`  | Trading session where the announcement is placed. If `EventDate` is a closed day, it is moved to the next session.                                   |
| `anchor_session`   | Return-window origin (`t = 0`). Before open (`MarketTimeId = 1`): same as `release_session`. After close (`MarketTimeId = 2`): next trading session. |
| `decision_session` | Feature as-of date: `anchor_session` minus 2 trading sessions (`DECISION_OFFSET = -2`). Predictors must not use information after this close.        |
| `MarketTimeId`     | Release timing flag from earnings: `1` = before open, `2` = after close.                                                                             |

---

## Identity helpers

| Column      | Description                                                                       |
| ----------- | --------------------------------------------------------------------------------- |
| `TickerId`  | Internal ticker id (from the panel).                                              |
| `Symbol`    | Ticker symbol split from `ShortName` (e.g. `ABNB`). Used for analyst asof merge.  |
| `Exchange`  | Exchange code split from `ShortName` (e.g. `XNGS`). Selects the trading calendar. |
| `ShortName` | Full `SYMBOL:EXCHANGE` string on the event row.                                   |
| `Id`        | Earnings-event id when present in the source table.                               |

---

## Intermediate market series

| Column              | Description                                                                                                                           |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `BenchmarkAdjClose` | SPY adjusted close (Yahoo), attached on the aligned daily panel and used only to build market-adjusted returns. Cached under `data/`. |

Stock return legs use `LastPrice`; market legs use `BenchmarkAdjClose`. Treat `*_mkt` columns as approximate excess returns across vendors/fields.

---

## Target / label columns

All windows are in **trading sessions relative to `anchor_session`**, unless noted.  
Window return: close at `start_offset − 1` through close at `end_offset`.

| Column         | Description                                                                                            |
| -------------- | ------------------------------------------------------------------------------------------------------ |
| `ret_ann`      | Stock return over the announcement window **[-1, +1]**.                                                |
| `ret_ann_mkt`  | `ret_ann` minus the same window on SPY. Default modeling target.                                       |
| `ret_post`     | Stock return over the post window **[+1, +5]**.                                                        |
| `ret_post_mkt` | `ret_post` minus the same window on SPY.                                                               |
| `ret_next`     | One-step forward stock return from the **announcement** (`release_session`) close to the next session. |
| `ret_next_mkt` | `ret_next` minus the same one-step move on SPY.                                                        |
| `dir_ann`      | Sign of `ret_ann_mkt` (`+1` / `0` / `-1`); reserved for classification.                                |
| `vol_ann`      | Sample standard deviation of daily stock returns inside the announcement window.                       |
| `vol_post`     | Sample standard deviation of daily stock returns inside the post window.                               |

Offsets are fixed in `src/features.py`:  
`ANN_START_OFFSET = -1`, `ANN_END_OFFSET = 1`, `POST_START_OFFSET = 1`, `POST_END_OFFSET = 5`.

---

## Decision-day feature columns (not newly engineered)

These are **copied** from the `decision_session` row of `df_all` (not recomputed by IMP). They are the model inputs in `FEATURE_COLUMNS`.

| Column                     | Source          | Description                                      |
| -------------------------- | --------------- | ------------------------------------------------ |
| `QScoreTrendLong`          | `stock-data_us` | Long-horizon trend technical score at decision.  |
| `QScoreSpeedLong`          | `stock-data_us` | Long-horizon speed / momentum score at decision. |
| `QScoreLong`               | `stock-data_us` | Combined long-horizon QScore at decision.        |
| `analystRatingsStrongBuy`  | sentiment asof  | Strong Buy count as of decision.                 |
| `analystRatingsBuy`        | sentiment asof  | Buy count as of decision.                        |
| `analystRatingsHold`       | sentiment asof  | Hold count as of decision.                       |
| `analystRatingsSell`       | sentiment asof  | Sell count as of decision.                       |
| `analystRatingsStrongSell` | sentiment asof  | Strong Sell count as of decision.                |

Analyst columns exist on the daily panel only after the backward `merge_asof` in `src/prepare.py` (31-day tolerance).

---

## Timing sketch

```
decision_session          release_session / EventDate
        |                            |
        |<-  (depends on MarketTimeId)
        |                            v
        *-----*-----*=====ann=====*-----*-----*-----*  post
                    ^             ^
              anchor-1      anchor / anchor+1
                    |             |
              [-1, +1] ann window; [+1, +5] post window from anchor
```

Features: values on `decision_session` only.  
Labels: returns / vols around `anchor_session` (and `ret_next` from the announcement session).
