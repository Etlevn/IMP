# Documentation vs. data inconsistencies

Found by running `script/run_data.py`, `script/run_feature.py` and `script/run_preprocess.py`, then checking `data/dataset.md` and `data/fields.md` against the data. Observed schema: `data/swagger.yaml`.

## Doc vs. data

| # | Doc | Claim | Observed |
| - | --- | ----- | -------- |
| 1 | both | Pipeline yields one row per earnings announcement | Raw earnings has 3,928 rows, `df_all` has 3,841 and `df` has 3,837. The 91 dropped events are not documented. Causes: ticker 16577 has no price data (3 events), 11 events fall on weekends, and the rest fall on days with no price row, because the join is exact. |
| 2 | fields.md | A closed `EventDate` is moved to the next session | `release_session == EventDate` on all 3,837 rows. Weekend events are lost at the exact join before they can be snapped. |
| 3 | fields.md | `Exchange` is in the identity helpers of `df` | `Exchange` exists only in `df_all`. `df` has 26 columns and no `Exchange`. `BenchmarkAdjClose` is also absent from `df`. |
| 4 | fields.md | `ret_ann` is the window `[-1, +1]` | It runs from the close at anchor-2 to the close at anchor+1, which is a 3-day return. `ret_post` `[+1, +5]` runs from the close at anchor to the close at anchor+5. The "close at start−1" rule is stated, but the names mislead. The two windows overlap on day +1. |
| 5 | fields.md | `dir_ann` is +1 / 0 / −1 | Only ±1 occurs (+1: 2,027, −1: 1,807). 0 never occurs. 3 values are null. |
| 6 | dataset.md | `tickerId` is "same entity as stock `TickerId`" | ANET:XNYS has two stock TickerIds (1418, 15733). Stock IDs 1418 and 1448 are absent from sentiment. Sentiment has `-1` and `16577`, which stock lacks. The sentiment join uses the symbol string, so it still works. |
| 7 | dataset.md | `ShortName_before` is a "historical short name / alias" | It equals `symbol` wherever non-null. 5 values are null. |
| 8 | dataset.md | `QScoreLong` is a "combined" long-horizon score | It is exactly `QScoreTrendLong + QScoreSpeedLong` on every row. It is redundant as a model feature (perfect collinearity). |
| 9 | dataset.md | `CurrencyId` is "commonly 840" for U.S. listings | `ARE:XTSE` is priced in CAD (124). Its 1,255 rows share price and market-cap columns with USD rows. |
| 10 | README / dataset.md | ~200 names, 2020–2024 | 200 TickerIds, 199 ShortNames and 198 symbols. Sentiment spans 2019-01 to 2026-02, but only 2020–2024 is used. |

## Data quality

- Raw `EventDate` is a string. `MarketTimeId` is a float.
- `QScore*` has 102 nulls in raw data and 1 null in `df`.
- Analyst columns are null on 335 of 3,837 events (8.7%).
- `df_all` carries duplicate columns: `symbol`/`Symbol`, `date`, `tickerId`.
- Stock rows are sorted newest-first.
- Validation and test features drift from train (for example `analystRatingsStrongSell` test mean +0.68, std 1.97).
