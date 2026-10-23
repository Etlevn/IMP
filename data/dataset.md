# Dataset

Source Parquet tables under `data/`. They are merged into a daily panel (`df_all`), then collapsed into one row per earnings announcement (`df`).

| File                                     | Granularity              | Role                                |
| ---------------------------------------- | ------------------------ | ----------------------------------- |
| `stock-data_us.parquet`                  | Ticker × trading day     | Prices and technical scores         |
| `earnings_us.parquet`                    | One row per announcement | Event date and pre/post-market flag |
| `historical_sentiment_scores_us.parquet` | Ticker × month           | Analyst rating counts               |

---

## `stock-data_us.parquet`

Daily equity panel (prices, volume, market cap, and long-horizon QScore features).

| Column            | Description                                                    |
| ----------------- | -------------------------------------------------------------- |
| `TickerId`        | Internal numeric ticker identifier; join key with earnings.    |
| `ShortName`       | Symbol and exchange, e.g. `ABNB:XNGS`.                         |
| `InstrumentName`  | Company / instrument display name.                             |
| `Date`            | Trading calendar date for the row.                             |
| `LastPrice`       | Closing price used for return construction.                    |
| `Open`            | Session open price.                                            |
| `High`            | Session high price.                                            |
| `Low`             | Session low price.                                             |
| `Volume`          | Traded share volume for the session.                           |
| `CurrencyId`      | Currency code for prices (U.S. listings commonly `840` = USD). |
| `QScoreTrendLong` | Long-horizon trend technical score.                            |
| `QScoreSpeedLong` | Long-horizon speed / momentum technical score.                 |
| `QScoreLong`      | Combined long-horizon QScore.                                  |
| `MarketCap`       | Market capitalization on that date.                            |

---

## `earnings_us.parquet`

Earnings announcement events (calendar date and market timing).

| Column         | Description                                                                                          |
| -------------- | ---------------------------------------------------------------------------------------------------- |
| `Id`           | Unique earnings-event identifier.                                                                    |
| `TickerId`     | Internal ticker id; matches `stock-data_us.TickerId`.                                                |
| `EventDate`    | Calendar date of the earnings release.                                                               |
| `MarketTimeId` | Release timing vs the session: `1` = before open, `2` = after close. Sets the return anchor session. |

Joined to the price panel with an exact left join on  
`Date` + `TickerId` ↔ `EventDate` + `TickerId`.

---

## `historical_sentiment_scores_us.parquet`

Monthly analyst-rating snapshots (counts by recommendation bucket).

| Column                     | Description                                                                     |
| -------------------------- | ------------------------------------------------------------------------------- |
| `symbol`                   | Ticker symbol string, e.g. `ABNB` (join key after splitting stock `ShortName`). |
| `date`                     | Snapshot date (typically month-start).                                          |
| `analystRatingsStrongBuy`  | Number of Strong Buy ratings in the snapshot.                                   |
| `analystRatingsBuy`        | Number of Buy ratings.                                                          |
| `analystRatingsHold`       | Number of Hold ratings.                                                         |
| `analystRatingsSell`       | Number of Sell ratings.                                                         |
| `analystRatingsStrongSell` | Number of Strong Sell ratings.                                                  |
| `ShortName_before`         | Historical short name / alias used for matching.                                |
| `tickerId`                 | Internal ticker id (lowercase `t`; same entity as stock `TickerId`).            |

Attached to the daily panel with a backward `merge_asof` on symbol and date (latest snapshot on or before each price date, within a 31-day tolerance).

---

## How the three tables connect

```
stock-data_us
    ├─ left join earnings_us          (Date/TickerId = EventDate/TickerId)
    └─ asof join historical_sentiment (Symbol/Date ← symbol/date)
                ↓
             df_all   (daily panel)
                ↓
        build_event_table → df   (one row per announcement)
```
