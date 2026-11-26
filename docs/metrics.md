# Metrics

Evaluation definitions used by lin / tr / nn (`src/metrics.py`).  
Data columns are in `dataset.md` / `fields.md`; this file covers **how predictions are scored**.

Regression only for now. Classification (`dir_ann`) is reserved.

Notation: \(y\) = true target, \(\hat{y}\) = model prediction, \(n\) = rows in the fold.

---

## Core scores

| Name       | Definition                                                              | Notes                                                 |
| ---------- | ----------------------------------------------------------------------- | ----------------------------------------------------- |
| `n`        | Number of evaluated rows                                                | Integer in displays                                   |
| `rmse`     | \(\sqrt{\mathrm{mean}((y-\hat{y})^2)}\)                                 | Lower is better                                       |
| `mae`      | \(\mathrm{mean}(\lvert y-\hat{y}\rvert)\)                               | Lower is better                                       |
| `r2`       | sklearn `r2_score` vs **this fold’s** mean of \(y\)                     | In-fold R²; not the primary OOS score                 |
| `oos_r2`   | \(1 - \mathrm{SSE}(\hat{y}) / \mathrm{SSE}(\bar{y}\_{\mathrm{train}})\) | Primary OOS fit vs train-mean baseline                |
| `ic`       | Pearson corr\((y,\hat{y})\)                                             | `n/a` if \(y\) or \(\hat{y}\) is constant             |
| `rank_ic`  | Spearman corr\((y,\hat{y})\)                                            | Same constant-prediction rule                         |
| `hit_rate` | \(\mathrm{mean}(\mathrm{sign}(y)=\mathrm{sign}(\hat{y}))\)              | Directional agreement; zeros in \(y\) count as misses |
| `sign_ls`  | \(\mathrm{mean}(\mathrm{sign}(\hat{y})\, y)\) on the test fold          | Long a positive forecast, short a negative one       |

\[
\mathrm{SSE}(a) = \sum_i (y_i - a_i)^2
\]

For `oos_r2`, the baseline \(a*i\) is the constant **training-fold** mean \(\bar{y}*{\mathrm{train}}\) (known at train time). Prefer **`oos_r2` over `r2`** when judging val/test.

---

## Baselines (shown next to RMSE / MAE)

| Name       | Prediction                               | Reported as                         |
| ---------- | ---------------------------------------- | ----------------------------------- |
| Train-mean | \(\hat{y}_i = \bar{y}_{\mathrm{train}}\) | `rmse_train_mean`, `mae_train_mean` |
| Zero       | \(\hat{y}\_i = 0\)                       | `rmse_zero`, `mae_zero`             |

Terminal display packs them as:

```text
rmse (mean/zero)  ->  model_rmse (rmse_train_mean / rmse_zero)
mae  (mean/zero)  ->  model_mae  (mae_train_mean / mae_zero)
r2   (oos)        ->  sklearn_r2 (oos_r2)
```

A useful model should beat both baselines on RMSE/MAE and show non-negative `oos_r2` when possible.

---

## Model selection

On the chronological **val** fold, tuneables (e.g. ridge `alpha`, tree `max_depth`, MLP L2) maximize **`oos_r2`**, tie-break **lower RMSE**.  
HistGBDT / MLP early stopping also monitor val `oos_r2`.

---

## QUINTILES

On val/test predictions, sort by \(\hat{y}\) into five equal-count buckets (`pd.qcut`):

| Column             | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| `q1` … `q5`        | Mean realized \(y\) in lowest … highest predicted bucket |
| `top_minus_bottom` | `q5 − q1` (spread of realized target; higher is better)  |
| `n`                | Rows in that fold                                        |

Skipped when predictions are constant or have too few unique values.

`quintile_ls` is a separate test-set long-short return. Cutoffs are the 20th and
80th percentiles of **training** `y_pred`. On the test fold, go long at or above
the high cutoff and short at or below the low cutoff. Events between the cutoffs
are left out. The value is the equal-weight mean of position times realized
target over the traded events, the same averaging as `sign_ls`. It is `NaN` when
the training forecast does not rank (for example lasso shrunk to a constant) or
no test event is traded.

`sign_ls` is the test-fold mean of \(\mathrm{sign}(\hat{y})\, y\). A positive
forecast is bought and a negative forecast is shorted; a zero forecast stays
flat and contributes 0. It is `NaN` when every test forecast is zero.

---

## Terminal blocks and artifacts

Per model run (typical order):

1. **ALPHA** / depth grid / **EARLY_STOP** (if applicable)
2. **METRICS** — train / val / test rows
3. **QUINTILES** — val / test
4. **SUMMARY** — printed once at the end of the run (all models merged; with `VERBOSE_PRINT=False` in `main.py`, only the table rows, no banner)

Written under `output/` (gitignored); multi-family runs merge into one file each.
SUMMARY and `main_metrics.csv` keep `family`, `model`, `n_test`, then
`oos_r2`, `rank_ic`, `quintile_ls`, `ic`, `r2`, `mae`, `rmse`, `hit_rate`, `sign_ls`.
The comparison order is `oos_r2`, then `rank_ic`, then `quintile_ls`.
Baseline errors, `top_minus_bottom`, and model parameters stay out of that table.

`feat_optuna.csv` keeps per-model `best=1` rows (train/val/score + test
SUMMARY). Optional `feat_optuna_trial.csv` (when `SAVE_TRIALS`) stores every
Optuna / exhaustive trial with train/val scores, `selection_score`, test
SUMMARY columns, and a `best` flag. Selection maximizes
`selection_score` from `feat_optuna.OBJECTIVE` (`oos_r2` | `rank_ic` |
`quintile_ls`) with `GAP_MODE`: `none` (pure val), `abs`
(`val - λ|train-val|`), or `pos` (`val - λ·max(0, val-train)`);
`GAP_PENALTY` is λ. Test never enters selection. Pools with width at most
`EXHAUSTIVE_MAX` are fully enumerated. Companion config
(`config_optuna.json`) stores search settings only, not trial or best rows.

| File                  | Contents                                                                 |
| --------------------- | ------------------------------------------------------------------------ |
| `main_metrics.csv`    | SUMMARY rows with `family` / metrics / trailing `features`; sorted by `SORT_BY` |
| `feat_filter.csv`     | Train-fold screen rows (`suggest` + IC / corr) |
| `feat_optuna.csv` | Per-model best: train/val/score + test SUMMARY |
| `feat_optuna_trial.csv` | All trials when `SAVE_TRIALS` (`train_*` / `val_*` / `selection_score` + test, `best`) |
| `main_predictions.parquet` | Per-event predictions (`SAVE_PREDICTIONS` in `main.py`) |
| `config_main.json` | `main.py` settings: `shared` + `runs[]` (`model` / `family` / `features` + used hypers; no fits) |
| `config_filter.json` | `feat_filter.py` settings only (thresholds / features / shared; no rows) |
| `config_optuna.json` | Search settings only (no trial / best results) |

Implementation: `src/metrics.py`.
