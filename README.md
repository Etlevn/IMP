# IMP

**Identifying Market Positioning Ahead of Earnings.**

Predict post-earnings stock behavior from information available **before** the announcement. The goal is not to forecast fundamentals from the earnings release itself, but to use pre-event price action and short-term signals known by a fixed decision session.

This repository builds an event-study panel from daily price, earnings, and analyst-rating Parquet tables, constructs fixed announcement windows and decision-day features, then trains linear, tree, and neural baselines under a shared chronological split.

## Features

- Load and merge daily price, earnings, and analyst-rating panels
- Trading-session alignment (NYSE calendar) with BMO / AMC anchors
- Fixed announcement / post windows and market-adjusted returns (SPY)
- Train / validation / test split by calendar date; impute and scale **on train only**
- Multi-model batches for linear, tree, and neural families
- Shared metrics (RMSE / MAE with baselines, IC, quintiles, permutation importance)
- Artifacts written under `output/{lin,tr,nn}/`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.10+ is recommended.

## Data

Place these files in `data/`. **Source tables are not redistributed with this repository**; obtain them from your own data provider.

| File                                     | Role                                    |
| ---------------------------------------- | --------------------------------------- |
| `stock-data_us.parquet`                  | Daily prices and technical indicators   |
| `earnings_us.parquet`                    | Earnings event dates and `MarketTimeId` |
| `historical_sentiment_scores_us.parquet` | Monthly analyst rating snapshots        |

Optional cache written by the pipeline on first use:

| File                              | Role                                                    |
| --------------------------------- | ------------------------------------------------------- |
| `benchmark_spy.parquet` / `.json` | Yahoo SPY adjusted close (downloaded once, then reused) |

Typical coverage in the sample panel used during development is on the order of **~200 U.S. names, 2020–2024**. Stock returns use `LastPrice`; the file does not state whether that series is split/dividend adjusted.

## Quick start

Inspect the panel and event table (print only; SPY may be cached under `data/`):

```bash
python script/run_data.py          # load / merge preview of df_all
python script/run_feature.py       # build df and print the descriptive report
python script/run_preprocess.py    # chronological split + train-only impute / scale
```

Train model batches (each reads its own `config/config_*.py`):

```bash
python main_lin.py    # ols, ridge, lasso
python main_tr.py     # decision_tree, random_forest, hist_gradient_boosting
python main_nn.py     # mlp_shallow, mlp, mlp_deep
```

## Pipeline overview

```text
data/*.parquet
    -> src/prepare.py      load + merge           ->  df_all  (daily panel)
    -> src/sessions.py     trading calendar alignment
    -> src/benchmark.py    SPY Adj Close
    -> src/features.py     event construction     ->  df  (one row per announcement)
    -> src/preprocess.py   train / val / test + train-only impute / scale
    -> model/ + main_*.py  fit, metrics, artifacts under output/
```

| Name     | Meaning                                        |
| -------- | ---------------------------------------------- |
| `df_all` | Daily panel after earnings and analyst merges  |
| `df`     | Event table: one row per earnings announcement |

## Design choices

These choices differ from an exact-date panel merge and a single next-session return label. They are intentional.

### 1. Earnings merge (exact date)

Earnings attach to the price panel with an exact left join on `Date` + `TickerId` ↔ `EventDate` + `TickerId`.

### 2. Analyst ratings: asof carry

Analyst ratings are **monthly snapshots**, not daily updates. An exact join leaves most decision days empty.

IMP uses **`merge_asof` (backward)** by symbol:

- For each price date, take the latest rating with `rating_date ≤ price_date`.
- Tolerance: **31 calendar days** (`SENTIMENT_TOLERANCE_DAYS` in `src/prepare.py`).
- Remaining NaNs are kept until train-only feature preprocessing.

### 3. Trading-session alignment

Each ticker is reindexed onto exchange sessions (`exchange_calendars`; U.S. listings use the NYSE calendar). Announcements on closed days are placed on the **next** session and treated as before the open (`MarketTimeId = 1`). A return window with a missing endpoint is NaN.

### 4. Announcement timing and the anchor

`MarketTimeId` comes from the earnings table:

| `MarketTimeId` | Meaning         | Anchor session                   |
| -------------- | --------------- | -------------------------------- |
| 1              | Before the open | Same session as the announcement |
| 2              | After the close | **Next** price session           |

All fixed return windows are counted in **trading sessions from this anchor**.

### 5. Decision time

**Decision session = anchor − 2 trading sessions** (`DECISION_OFFSET = -2`).

Features are taken from the decision-session row only. Predictors must not use prices or ratings after that close.

### 6. Target returns (fixed global windows)

Window returns use closes at `start_offset − 1` through `end_offset` relative to the **anchor**:

| Column                 | Definition                                                |
| ---------------------- | --------------------------------------------------------- |
| `ret_ann`              | Announcement window **[-1, +1]** stock return             |
| `ret_ann_mkt`          | `ret_ann` minus the same window on SPY                    |
| `ret_post`             | Post window **[+1, +5]** stock return                     |
| `ret_post_mkt`         | Market-adjusted `ret_post`                                |
| `ret_next`             | One-step forward return from the announcement session     |
| `ret_next_mkt`         | Market-adjusted `ret_next`                                |
| `dir_ann`              | Sign of `ret_ann_mkt`                                     |
| `vol_ann` / `vol_post` | Sample std of daily returns inside the ann / post windows |

Default modeling target: **`ret_ann_mkt`**.

Literature motivation (not a full paper replication): So and Wang (2014) use market-adjusted **[-1, +1]** with predictors by day **-2**; Aboody et al. (2010) emphasize post window **[+1, +5]** for past winners.

### 7. Market adjustment (SPY)

Excess returns use Yahoo **SPY Adj Close** (`BenchmarkAdjClose`), cached under `data/`. Stock legs still use `LastPrice`. Treat market-adjusted columns as approximate excess returns across vendors/fields.

### 8. Decision-day features

Copied from the decision session:

- Technical: `QScoreTrendLong`, `QScoreSpeedLong`, `QScoreLong`
- Analyst (after asof): `analystRatingsStrongBuy` … `analystRatingsStrongSell`

## Preprocessing

`src/preprocess.py` applies, in order:

1. Drop rows with missing `TARGET` (never impute the label).
2. Chronological split on `SPLIT_COLUMN` via `TRAIN_END` / `VAL_END`:
   - train: `date ≤ TRAIN_END`
   - val: `TRAIN_END < date ≤ VAL_END`
   - test: `date > VAL_END`
3. Fit feature impute / standardize **on train only**, then transform other folds.

Defaults in the model configs: `TRAIN_END = 2022-12-31`, `VAL_END = 2023-12-31`.

Feature impute: `median` | `mean` | `drop` | `None` (keep NaNs where a model supports them). Linear and neural configs use median + standardize; tree configs use median without standardize so all batch members share one preprocess.

### Validation protocol (shared)

The chronological **val** fold is used for **model selection and early stopping**, not only for reporting:

| Family | Selected on chronological val                | Early stopping monitor                           |
| ------ | -------------------------------------------- | ------------------------------------------------ |
| lin    | `alpha` (ridge / lasso via `ALPHA_GRID`)     | n/a                                              |
| tr     | `max_depth` (all types via `MAX_DEPTH_GRID`) | HistGBDT: val OOS R² (`EARLY_STOPPING_PATIENCE`) |
| nn     | L2 `alpha` (all MLPs via `ALPHA_GRID`)       | all MLPs: val OOS R² (`EARLY_STOPPING_PATIENCE`) |

Selection criterion: maximize validation **OOS R²** (tie-break: lower RMSE).  
HistGBDT and MLP do **not** use sklearn’s random `validation_fraction` holdout from train.

## Models

Each `main_*.py` runs every type listed in its config under the **same** preprocess, prints a banner per model, then a compact **SUMMARY**.

| Entry point   | Config                 | Models                                                     |
| ------------- | ---------------------- | ---------------------------------------------------------- |
| `main_lin.py` | `config/config_lin.py` | `ols`, `ridge`, `lasso`                                    |
| `main_tr.py`  | `config/config_tr.py`  | `decision_tree`, `random_forest`, `hist_gradient_boosting` |
| `main_nn.py`  | `config/config_nn.py`  | `mlp_shallow`, `mlp`, `mlp_deep` (sklearn `MLPRegressor`)  |

Classification (`dir_ann`, logistic) is reserved; regression targets only for now.

Edit the corresponding `config/config_*.py` to change target, features, cut dates, or hyperparameters.

## Metrics and terminal output

Per model, the run prints:

1. **METRICS** — train / val / test rows with `r2 (oos)`, `rmse (mean/zero)`, `mae (mean/zero)`, IC, rank IC, hit rate
   - `(mean/zero)` = train-mean baseline / zero baseline
   - `oos` = out-of-sample R² vs the train-mean forecast
2. **FEATURE** — permutation importance on the test fold (optional; `FEATURE_IMPORTANCE`)
3. **QUINTILES** — mean realized target by predicted quintile on val / test (skipped if predictions are constant)
4. **SUMMARY** — one row per model (test only): `r2 (oos)`, `rmse (mean/zero)`, `mae (mean/zero)`, IC, rank IC, hit rate

## Outputs

Each successful `main_*.py` run writes under `output/{lin,tr,nn}/` (directory is gitignored):

| File                  | Contents                                    |
| --------------------- | ------------------------------------------- |
| `metrics.csv`         | One summary row per model (numeric columns) |
| `predictions.parquet` | Per-row predictions with `model` and `fold` |
| `run_config.json`     | Resolved settings and fold date ranges      |

## Project layout

```text
config/
  config_lin.py      linear batch settings
  config_tr.py       tree batch settings
  config_nn.py       neural batch settings
model/
  model_lin.py       ols / ridge / lasso (+ alpha grid on val)
  model_tr.py        decision tree / RF / hist GBDT
  model_nn.py        MLPRegressor variants
src/
  prepare.py         load Parquet; earnings merge; analyst asof
  sessions.py        exchange-session grid; closed-day placement
  benchmark.py       SPY cache and attach
  features.py        anchors, returns, decision-day features → df
  preprocess.py      chronological split; train-only impute / scale
  metrics.py         regression metrics, quintiles, importance, display
  report.py          descriptive printout for df
script/
  run_data.py
  run_feature.py
  run_preprocess.py
main_lin.py
main_tr.py
main_nn.py
data/                local Parquet inputs (+ SPY cache); not redistributed
output/              run artifacts (gitignored)
requirements.txt
```

## License and attribution

Data files remain subject to their original provider terms and are not included here. Cite So and Wang (2014) and Aboody et al. (2010) if you rely on the window design motivated by those papers.
