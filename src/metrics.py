"""Unified prediction metrics for IMP models (regression first).

Classification helpers are reserved for a later extension.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PredictFn = Callable[[pd.DataFrame], np.ndarray]

REGRESSION_TARGETS = frozenset(
    {
        "ret_ann",
        "ret_ann_mkt",
        "ret_next",
        "ret_next_mkt",
        "ret_post",
        "ret_post_mkt",
        "vol_ann",
        "vol_post",
    }
)
CLASSIFICATION_TARGETS = frozenset({"dir_ann"})


def infer_task(target: str) -> str:
    """Map TARGET column name to ``regression`` or ``classification``."""
    if target in REGRESSION_TARGETS:
        return "regression"
    if target in CLASSIFICATION_TARGETS:
        return "classification"
    raise ValueError(
        f"Unknown TARGET {target!r}; "
        f"regression={sorted(REGRESSION_TARGETS)} "
        f"classification={sorted(CLASSIFICATION_TARGETS)}"
    )


def _oos_r2(y_true: np.ndarray, y_pred: np.ndarray, baseline: np.ndarray) -> float:
    """1 - SSE(model) / SSE(baseline); baseline is known at train time."""
    sse_model = float(np.sum((y_true - y_pred) ** 2))
    sse_base = float(np.sum((y_true - baseline) ** 2))
    if sse_base <= 0.0:
        return float("nan")
    return 1.0 - sse_model / sse_base


def _is_constant(arr: np.ndarray) -> bool:
    """True when finite values are identical (nanstd==0 is unreliable for floats)."""
    finite = arr[np.isfinite(arr)]
    if finite.size <= 1:
        return True
    return bool(np.unique(finite).size <= 1)


def evaluate_regression(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    train_y_mean: float | None = None,
) -> dict[str, float]:
    """Core regression metrics shared by lin / tree / nn.

    When ``train_y_mean`` is provided, also report train-mean / zero baselines
    and true OOS R² against the train-mean forecast.
    """
    yt = np.asarray(y_true, dtype=float).reshape(-1)
    yp = np.asarray(y_pred, dtype=float).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true={yt.shape} y_pred={yp.shape}")
    empty = {
        "n": 0.0,
        "r2": float("nan"),
        "rmse": float("nan"),
        "mae": float("nan"),
        "ic": float("nan"),
        "rank_ic": float("nan"),
        "hit_rate": float("nan"),
        "rmse_train_mean": float("nan"),
        "mae_train_mean": float("nan"),
        "rmse_zero": float("nan"),
        "mae_zero": float("nan"),
        "oos_r2": float("nan"),
    }
    if len(yt) == 0:
        return empty

    mse = mean_squared_error(yt, yp)
    # Constant predictions make Pearson/Spearman undefined (pandas warns).
    if _is_constant(yp) or _is_constant(yt):
        ic = float("nan")
        rank_ic = float("nan")
    else:
        pair = pd.DataFrame({"y": yt, "p": yp})
        ic = float(pair["y"].corr(pair["p"], method="pearson"))
        rank_ic = float(pair["y"].corr(pair["p"], method="spearman"))
    hit = float(np.mean(np.sign(yt) == np.sign(yp)))

    out: dict[str, float] = {
        "n": float(len(yt)),
        "r2": float(r2_score(yt, yp)),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(yt, yp)),
        "ic": ic,
        "rank_ic": rank_ic,
        "hit_rate": hit,
        "rmse_train_mean": float("nan"),
        "mae_train_mean": float("nan"),
        "rmse_zero": float(np.sqrt(mean_squared_error(yt, np.zeros_like(yt)))),
        "mae_zero": float(mean_absolute_error(yt, np.zeros_like(yt))),
        "oos_r2": float("nan"),
    }
    if train_y_mean is not None:
        baseline = np.full_like(yt, float(train_y_mean))
        out["rmse_train_mean"] = float(np.sqrt(mean_squared_error(yt, baseline)))
        out["mae_train_mean"] = float(mean_absolute_error(yt, baseline))
        out["oos_r2"] = _oos_r2(yt, yp, baseline)
    return out


def metrics_table(
    fold_predictions: dict[str, tuple[np.ndarray | pd.Series, np.ndarray | pd.Series]],
    *,
    task: str = "regression",
    train_y_mean: float | None = None,
) -> pd.DataFrame:
    """Build one-row-per-fold metrics table.

    ``fold_predictions`` maps fold name -> (y_true, y_pred).
    """
    if task != "regression":
        raise NotImplementedError(
            f"task={task!r} is not implemented yet; only regression is supported."
        )
    rows: list[dict[str, float | str]] = []
    for name, (y_true, y_pred) in fold_predictions.items():
        row: dict[str, float | str] = {"fold": name}
        row.update(evaluate_regression(y_true, y_pred, train_y_mean=train_y_mean))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("fold")
    table["n"] = table["n"].astype(int)
    return table


def _fmt4(value: float) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{float(value):.4f}"


def _with_baselines(model: float, train_mean: float, zero: float) -> str:
    """Format ``model (train_mean/zero)`` for terminal comparison."""
    return f"{_fmt4(model)} ({_fmt4(train_mean)}/{_fmt4(zero)})"


def format_metrics_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Compact METRICS view: baselines sit next to the compared model errors.

    Display columns:
      rmse (mean/zero)  ->  model_rmse (rmse_train_mean/rmse_zero)
      mae  (mean/zero)  ->  model_mae  (mae_train_mean/mae_zero)
      r2   (oos)        ->  sklearn_r2 (oos_r2)
    """
    rows: list[dict[str, object]] = []
    for fold, row in table.iterrows():
        rows.append(
            {
                "fold": fold,
                "n": int(row["n"]),
                "r2 (oos)": f"{_fmt4(row['r2'])} ({_fmt4(row['oos_r2'])})",
                "rmse (mean/zero)": _with_baselines(
                    row["rmse"], row["rmse_train_mean"], row["rmse_zero"]
                ),
                "mae (mean/zero)": _with_baselines(
                    row["mae"], row["mae_train_mean"], row["mae_zero"]
                ),
                "ic": _fmt4(row["ic"]),
                "rank_ic": _fmt4(row["rank_ic"]),
                "hit_rate": _fmt4(row["hit_rate"]),
            }
        )
    return pd.DataFrame(rows).set_index("fold")


SUMMARY_ID_COLUMNS = ("feature_set", "family", "model", "n_test")
SUMMARY_METRIC_COLUMNS = (
    "oos_r2",
    "rank_ic",
    "quintile_ls",
    "ic",
    "r2",
    "mae",
    "rmse",
    "hit_rate",
    "sign_ls",
)


def order_summary_table(table: pd.DataFrame) -> pd.DataFrame:
    """Keep identity columns, then the shared metric columns, and drop the rest.

    Dropped fields include baseline errors, ``top_minus_bottom``, and model
    parameters such as ``alpha``, ``hidden``, and ``best_iteration``.
    """
    columns = [
        column
        for column in (*SUMMARY_ID_COLUMNS, *SUMMARY_METRIC_COLUMNS)
        if column in table.columns
    ]
    return table.loc[:, columns].copy()


def format_summary_for_display(table: pd.DataFrame) -> pd.DataFrame:
    """Compact SUMMARY using the same columns as the saved metrics table."""
    ordered = order_summary_table(table)
    rows: list[dict[str, object]] = []
    for _, row in ordered.iterrows():
        out: dict[str, object] = {}
        for column in ordered.columns:
            value = row[column]
            if column in SUMMARY_METRIC_COLUMNS:
                out[column] = _fmt4(value)
            elif column == "n_test":
                out[column] = int(value)
            else:
                out[column] = value
        rows.append(out)
    return pd.DataFrame(rows, columns=list(ordered.columns))


def print_metrics_table(table: pd.DataFrame) -> None:
    """Print METRICS with baselines grouped beside the compared quantities."""
    print("-" * 60)
    print("METRICS")
    print("-" * 60)
    display = format_metrics_for_display(table)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
    ):
        print(display.to_string())


def print_summary_table(table: pd.DataFrame) -> None:
    """Print compact SUMMARY for cross-model comparison."""
    display = format_summary_for_display(table)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
    ):
        print(display.to_string(index=False))


def summary_row_from_fold_metrics(
    model_type: str,
    fold_metrics: pd.DataFrame,
    *,
    predictions: pd.DataFrame | None = None,
) -> dict[str, float | int | str]:
    """One comparison row: test-fold metrics for CSV and SUMMARY."""
    test = fold_metrics.loc["test"]
    return {
        "model": model_type,
        "n_test": int(test["n"]),
        "oos_r2": float(test["oos_r2"]),
        "rank_ic": float(test["rank_ic"]),
        "quintile_ls": train_quintile_long_short(predictions)
        if predictions is not None
        else float("nan"),
        "ic": float(test["ic"]),
        "r2": float(test["r2"]),
        "mae": float(test["mae"]),
        "rmse": float(test["rmse"]),
        "hit_rate": float(test["hit_rate"]),
        "sign_ls": sign_long_short(predictions),
    }


def print_model_fit_report(
    model_type: str,
    *,
    fold_metrics: pd.DataFrame,
    quintiles: pd.DataFrame,
    tuning: pd.DataFrame | None = None,
    selected: float | int | None = None,
    tuning_label: str = "ALPHA",
    best_iteration: int | None = None,
    max_iter: int | None = None,
    importance: pd.DataFrame | None = None,
) -> None:
    """Print one model's tuning, early-stop, metrics, importance, and quintiles."""
    print()
    print("=" * 60)
    print(model_type.upper())
    print("=" * 60)

    if tuning is not None and selected is not None:
        print()
        print("-" * 60)
        print(tuning_label)
        print("-" * 60)
        if not tuning.empty:
            with pd.option_context(
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.expand_frame_repr",
                False,
                "display.float_format",
                lambda x: f"{x:.4f}",
            ):
                print(tuning.to_string(index=False))
        print(f"SELECTED_{tuning_label}    {selected}")

    if best_iteration is not None:
        print()
        print("-" * 60)
        print("EARLY_STOP")
        print("-" * 60)
        print(f"BEST_ITERATION    {best_iteration}")
        if max_iter is not None:
            print(f"MAX_ITER          {max_iter}")

    print()
    print_metrics_table(fold_metrics)

    if importance is not None:
        print()
        print_feature_importance(importance)

    print_quintile_table(quintiles)


def permutation_importance_regression(
    predict_fn: PredictFn,
    X: pd.DataFrame,
    y: pd.Series | np.ndarray,
    *,
    n_repeats: int = 10,
    random_state: int = 42,
    train_y_mean: float | None = None,
) -> pd.DataFrame:
    """Model-conditional feature importance via column permutation.

    Positive ``delta_rmse`` / ``delta_mae`` and positive ``delta_ic`` /
    ``delta_rank_ic`` / ``delta_hit_rate`` / ``delta_oos_r2`` mean the feature
    helps the fitted model on this fold (shuffling hurts performance).
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    y_true = np.asarray(y, dtype=float)
    baseline_pred = np.asarray(predict_fn(X), dtype=float)
    base = evaluate_regression(y_true, baseline_pred, train_y_mean=train_y_mean)
    rng = np.random.default_rng(random_state)

    rows: list[dict[str, float | str]] = []
    for column in X.columns:
        rmse_deltas: list[float] = []
        mae_deltas: list[float] = []
        ic_deltas: list[float] = []
        rank_ic_deltas: list[float] = []
        hit_deltas: list[float] = []
        oos_deltas: list[float] = []
        for _ in range(n_repeats):
            shuffled = X.copy()
            values = shuffled[column].to_numpy(copy=True)
            rng.shuffle(values)
            shuffled[column] = values
            perm = evaluate_regression(
                y_true, predict_fn(shuffled), train_y_mean=train_y_mean
            )
            rmse_deltas.append(perm["rmse"] - base["rmse"])
            mae_deltas.append(perm["mae"] - base["mae"])
            ic_deltas.append(base["ic"] - perm["ic"])
            rank_ic_deltas.append(base["rank_ic"] - perm["rank_ic"])
            hit_deltas.append(base["hit_rate"] - perm["hit_rate"])
            oos_deltas.append(base["oos_r2"] - perm["oos_r2"])
        rows.append(
            {
                "feature": column,
                "delta_rmse": float(np.mean(rmse_deltas)),
                "delta_mae": float(np.mean(mae_deltas)),
                "delta_ic": float(np.mean(ic_deltas)),
                "delta_rank_ic": float(np.mean(rank_ic_deltas)),
                "delta_hit_rate": float(np.mean(hit_deltas)),
                "delta_oos_r2": float(np.mean(oos_deltas)),
            }
        )

    table = pd.DataFrame(rows).set_index("feature")
    return table.sort_values("delta_ic", ascending=False)


def print_feature_importance(table: pd.DataFrame) -> None:
    """Print the FEATURE block to the terminal."""
    print("-" * 60)
    print("FEATURE")
    print("-" * 60)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(table.to_string())


def _print_frame(title: str, table: pd.DataFrame) -> None:
    print()
    print("-" * 60)
    print(title)
    print("-" * 60)
    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        None,
        "display.expand_frame_repr",
        False,
        "display.float_format",
        lambda x: f"{x:.4f}",
    ):
        print(table.to_string(index=False))


def quintile_table(
    predictions: pd.DataFrame,
    *,
    folds: tuple[str, ...] = ("val", "test"),
    n_buckets: int = 5,
) -> pd.DataFrame:
    """One row per fold: mean realized target in each predicted quintile.

    Columns ``q1``..``q{n}`` are mean ``y_true`` from lowest to highest ``y_pred``.
    ``top_minus_bottom`` = ``q{n} - q1``.
    """
    if n_buckets < 2:
        raise ValueError(f"n_buckets must be >= 2, got {n_buckets}")

    required = {"fold", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")

    q_cols = [f"q{i}" for i in range(1, n_buckets + 1)]
    rows: list[dict[str, float | int | str]] = []
    for fold in folds:
        part = predictions.loc[predictions["fold"] == fold].copy()
        if len(part) < n_buckets:
            continue
        # Constant / near-constant predictions cannot form quintiles.
        if float(part["y_pred"].nunique(dropna=True)) < n_buckets:
            continue
        try:
            part["bucket"] = pd.qcut(
                part["y_pred"],
                q=n_buckets,
                labels=False,
                duplicates="drop",
            )
        except ValueError:
            continue
        codes = part["bucket"].dropna().astype(int)
        if codes.empty:
            continue
        mapping = {old: new for new, old in enumerate(sorted(codes.unique()), start=1)}
        part["bucket"] = part["bucket"].map(mapping)
        means = part.dropna(subset=["bucket"]).groupby("bucket")["y_true"].mean()
        if means.empty:
            continue
        row: dict[str, float | int | str] = {
            "fold": fold,
            "n": int(len(part)),
        }
        for i in range(1, n_buckets + 1):
            row[f"q{i}"] = float(means[i]) if i in means.index else float("nan")
        q1 = float(means[1]) if 1 in means.index else float(means.iloc[0])
        qn = float(means[means.index.max()])
        row["top_minus_bottom"] = qn - q1
        rows.append(row)

    columns = ["fold", "n", *q_cols, "top_minus_bottom"]
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


def print_quintile_table(table: pd.DataFrame) -> None:
    """Print compact QUINTILES block (one row per fold)."""
    if table.empty:
        print()
        print("-" * 60)
        print("QUINTILES")
        print("-" * 60)
        print("(skipped: predictions are constant / too few unique values)")
        return
    _print_frame("QUINTILES", table)


def train_quintile_long_short(
    predictions: pd.DataFrame | None,
    *,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
    train_fold: str = "train",
    test_fold: str = "test",
) -> float:
    """Test long-short return from training-set prediction percentiles.

    Thresholds are the training fold's ``low_percentile`` and
    ``high_percentile`` of ``y_pred``. On the test fold, a forecast at or
    above the high threshold is long and one at or below the low threshold
    is short. Events between the cutoffs are not traded. The value is the
    equal-weight mean of position times realized target over the traded
    events, the same averaging as ``sign_ls``. A flat training forecast, or
    no traded test event, yields NaN.
    """
    if predictions is None or predictions.empty:
        return float("nan")
    required = {"fold", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")

    train = pd.to_numeric(
        predictions.loc[predictions["fold"] == train_fold, "y_pred"], errors="coerce"
    ).dropna()
    if train.nunique() < 2:
        return float("nan")
    lo, hi = np.percentile(train.to_numpy(dtype=float), [low_percentile, high_percentile])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return float("nan")

    test = predictions.loc[predictions["fold"] == test_fold, ["y_true", "y_pred"]].copy()
    test["y_true"] = pd.to_numeric(test["y_true"], errors="coerce")
    test["y_pred"] = pd.to_numeric(test["y_pred"], errors="coerce")
    test = test.dropna()
    traded = test.loc[(test["y_pred"] >= hi) | (test["y_pred"] <= lo)]
    if traded.empty:
        return float("nan")
    position = np.where(traded["y_pred"].to_numpy(dtype=float) >= hi, 1.0, -1.0)
    realized = traded["y_true"].to_numpy(dtype=float)
    return float(np.mean(position * realized))


def sign_long_short(
    predictions: pd.DataFrame | None,
    *,
    test_fold: str = "test",
) -> float:
    """Test-fold mean return from following the forecast sign.

    A positive forecast is long and a negative forecast is short. A zero
    forecast is flat. The value is the equal-weight mean of
    sign(y_pred) * y_true. NaN when the test fold has no finite pair, or
    every forecast is zero.
    """
    if predictions is None or predictions.empty:
        return float("nan")
    required = {"fold", "y_true", "y_pred"}
    missing = required - set(predictions.columns)
    if missing:
        raise KeyError(f"predictions missing columns: {sorted(missing)}")

    test = predictions.loc[predictions["fold"] == test_fold, ["y_true", "y_pred"]].copy()
    test["y_true"] = pd.to_numeric(test["y_true"], errors="coerce")
    test["y_pred"] = pd.to_numeric(test["y_pred"], errors="coerce")
    test = test.dropna()
    if test.empty:
        return float("nan")
    position = np.sign(test["y_pred"].to_numpy(dtype=float))
    if np.all(position == 0):
        return float("nan")
    realized = test["y_true"].to_numpy(dtype=float)
    return float(np.mean(position * realized))
