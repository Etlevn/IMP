# Expanded panel (2019–2025)

Price and earnings inputs now use the expanded files. The analyst-rating file is unchanged. Session alignment keeps only listings that map to a trading calendar.

## Data files

`config/default.py`:

| Setting | Current file | Previous file |
| --- | --- | --- |
| `STOCK_FILE` | `stock_data_us_expanded-2019-2026.parquet` | `stock-data_us.parquet` |
| `EARNINGS_FILE` | `earnings_us_expanded-2019-2026.parquet` | `earnings_us.parquet` |
| `SENTIMENT_FILE` | `historical_sentiment_scores_us.parquet` | unchanged |

Column names match the previous tables, so `script/run_data.py` still loads and merges them the same way. The merged daily panel has 1,376,761 rows. The filenames say 2026; prices actually run from 2019-01-02 through 2025-12-31, and earnings from 2019-01-03 through 2025-12-19. The stock count rises from about 200 names to 817.

The rating file still covers about 200 symbols. About 187 price symbols match at least one rating row. Analyst features are missing for the rest.

## Trading calendar

`src/sessions.py` drops a `TickerId` before alignment when its listing has no usable calendar. `XASE`, `XNCM`, `ARCX`, and `XCBO` use the NYSE session calendar, as the original US venues do. `XTSE` still uses the Toronto calendar.

Dropped listings are `XUS`, `XLON`, `OTCPK`, and names whose `ShortName` has no exchange (for example `AERCAP HOLDINGS`, `FNMA:`, `DIDIY:`, `TCEHY:`). On this panel that is 43 names. Earnings rows fall from 20,365 to 20,188. 774 names remain, including `LNG:XASE`, `GSAT:XASE`, and `RKLB:XNCM`.

Dropped names do not enter the equal-weight benchmark. Most `XUS` rows are funds, not corporate earnings.

## Release timing

The expanded earnings file has 249 null `MarketTimeId` values and one value equal to 4. Only 1 (before the open) and 2 (after the close) are usable.

`src/sessions.py` does not place any other code on the session grid, so a bad row cannot occupy that day and block a valid announcement. A closed-day announcement is still moved to the next session only when its code is 1 or 2. `src/events.py` builds the anchor from those two codes: 1 uses that session, and 2 uses the next session. If another code still reaches the anchor step, it is skipped and does not stop the run.

## Unchanged settings

`TRAIN_END` remains 2022-12-31 and `VAL_END` remains 2023-12-31, so the test fold covers 2024 and 2025. The equal-weight benchmark is rebuilt from the new panel the next time the event table is built. The macro cache currently covers about 2020–2024 and is re-downloaded for the new date span when `vix` or `par*vix` is used.

The new sample uses a different benchmark universe and a longer test window than the earlier run of about 200 names with a 2024-only test. Those `metrics_feat` tables are not directly comparable.
