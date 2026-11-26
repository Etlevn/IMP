"""Feature-pool helpers and per-model Optuna / exhaustive search."""

from __future__ import annotations

import itertools
from collections import OrderedDict
from collections.abc import MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BoundedPreprocessCache(MutableMapping[Any, Any]):
    """LRU map that caps preprocess retention during Optuna / exhaustive search.

    Each cached ``PreprocessResult`` holds train/val/test frames; without a
    bound, unique feature subsets accumulate until the process is OOM-killed.
    """

    def __init__(self, maxsize: int) -> None:
        if int(maxsize) < 1:
            raise ValueError(f"maxsize must be >= 1; got {maxsize!r}.")
        self.maxsize = int(maxsize)
        self._data: OrderedDict[Any, Any] = OrderedDict()

    def __getitem__(self, key: Any) -> Any:
        value = self._data[key]
        self._data.move_to_end(key)
        return value

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        while len(self._data) > self.maxsize:
            self._data.popitem(last=False)

    def __delitem__(self, key: Any) -> None:
        del self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()

from feat_filter import SUGGEST_LABELS
from src.pipeline import score_feature_subset
from src.train import FamilySpec

OBJECTIVE_CHOICES: tuple[str, ...] = ("oos_r2", "rank_ic", "quintile_ls")
# none = pure val; abs = λ·|train−val|; pos = λ·max(0, val−train).
GAP_MODE_CHOICES: tuple[str, ...] = ("none", "abs", "pos")
SELECTION_SCORE_COLUMN = "selection_score"


def objective_column(objective: str) -> str:
    """Map a public OBJECTIVE name to the val column used for reporting."""
    name = str(objective).strip()
    if name not in OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unknown OBJECTIVE={objective!r}; expected one of {OBJECTIVE_CHOICES}."
        )
    return f"val_{name}"


def train_column(objective: str) -> str:
    """Map a public OBJECTIVE name to the train-fold column."""
    name = str(objective).strip()
    if name not in OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unknown OBJECTIVE={objective!r}; expected one of {OBJECTIVE_CHOICES}."
        )
    return f"train_{name}"


def normalize_gap_mode(gap_mode: str) -> str:
    name = str(gap_mode).strip().lower()
    if name not in GAP_MODE_CHOICES:
        raise ValueError(
            f"Unknown GAP_MODE={gap_mode!r}; expected one of {GAP_MODE_CHOICES}."
        )
    return name


def selection_score(
    *,
    val_metric: float,
    train_metric: float,
    gap_penalty: float,
    gap_mode: str = "abs",
) -> float:
    """Val metric minus an optional train/val gap penalty (test never enters).

    ``gap_mode``:
    - ``none``: pure ``val_metric`` (``gap_penalty`` ignored)
    - ``abs``: ``val - λ * |train - val|``
    - ``pos``: ``val - λ * max(0, val - train)``
    """
    if not np.isfinite(val_metric):
        return float("-inf")
    mode = normalize_gap_mode(gap_mode)
    if mode == "none":
        return float(val_metric)
    penalty = float(gap_penalty)
    if penalty < 0:
        raise ValueError(f"gap_penalty must be >= 0; got {gap_penalty!r}.")
    if penalty == 0 or not np.isfinite(train_metric):
        return float(val_metric)
    val = float(val_metric)
    train = float(train_metric)
    if mode == "abs":
        gap = abs(train - val)
    else:  # pos
        gap = max(0.0, val - train)
    return val - penalty * gap


def screen_from_suggest_column(table: pd.DataFrame) -> dict[str, list[str]]:
    """Read suggest groups from ``feat_filter``'s suggest column."""
    if table.empty:
        raise ValueError("filter table is empty.")
    if "suggest" not in table.columns:
        raise ValueError(
            "filter table missing 'suggest'. "
            "Re-run `python feat_filter.py` to write suggest labels."
        )
    if "feature" not in table.columns:
        raise ValueError("filter table missing 'feature' column.")

    grouped: dict[str, list[str]] = {label: [] for label in SUGGEST_LABELS}
    for row in table.itertuples(index=False):
        name = str(row.feature)
        label = str(row.suggest).strip().lower()
        if label not in grouped:
            raise ValueError(
                f"Unknown suggest={label!r} for feature {name!r}; "
                f"expected one of {SUGGEST_LABELS}."
            )
        grouped[label].append(name)
    return grouped


@dataclass(frozen=True)
class SearchPool:
    searchable: tuple[str, ...]
    whitelist: tuple[str, ...]
    blacklist: tuple[str, ...]
    from_suggest: tuple[str, ...]
    screen: dict[str, list[str]]


def build_search_pool(
    table: pd.DataFrame,
    *,
    whitelist: Sequence[str],
    blacklist: Sequence[str] = (),
    include_suggest: Sequence[str],
    catalog: Sequence[str],
) -> SearchPool:
    """Build search pool: whitelist first, then enabled feat_filter suggest labels.

    ``blacklist`` names are removed from the pool (must not overlap whitelist).
    """
    catalog_set = set(catalog)
    if not catalog_set:
        raise ValueError("enabled_features() catalog is empty.")

    labels = [str(label).strip().lower() for label in include_suggest]
    unknown = [label for label in labels if label not in SUGGEST_LABELS]
    if unknown:
        raise ValueError(
            f"Unknown INCLUDE_SUGGEST labels {unknown}; expected {SUGGEST_LABELS}."
        )
    enabled = set(labels)

    blocked: list[str] = []
    blocked_set: set[str] = set()
    for name in blacklist:
        if name not in catalog_set:
            raise ValueError(
                f"FEATURE_BLACKLIST name {name!r} is not in enabled_features()."
            )
        if name not in blocked_set:
            blocked.append(name)
            blocked_set.add(name)

    forced: list[str] = []
    for name in whitelist:
        if name not in catalog_set:
            raise ValueError(
                f"FEATURE_WHITELIST name {name!r} is not in enabled_features()."
            )
        if name in blocked_set:
            raise ValueError(
                f"FEATURE_WHITELIST name {name!r} is also in FEATURE_BLACKLIST."
            )
        if name not in forced:
            forced.append(name)

    screen = screen_from_suggest_column(table)
    from_suggest: list[str] = []
    for label in SUGGEST_LABELS:
        if label not in enabled:
            continue
        for name in screen[label]:
            if name not in catalog_set:
                continue
            if name in blocked_set or name in forced or name in from_suggest:
                continue
            from_suggest.append(name)

    searchable = forced + from_suggest
    if not searchable:
        raise ValueError(
            "Searchable feature pool is empty. Add FEATURE_WHITELIST entries "
            "and/or INCLUDE_SUGGEST labels (e.g. ('keep',)), or clear "
            "FEATURE_BLACKLIST."
        )

    return SearchPool(
        searchable=tuple(searchable),
        whitelist=tuple(forced),
        blacklist=tuple(blocked),
        from_suggest=tuple(from_suggest),
        screen=screen,
    )


def _feature_key(columns: Sequence[str]) -> str:
    # Single quotes avoid CSV ""-escaping; paste into a Python list: 'A','B','C'
    return ",".join(f"'{name}'" for name in columns)


def _trial_row(
    *,
    family: str,
    model: str,
    trial: int,
    columns: Sequence[str],
    summary: dict[str, float | int | str],
    objective: str,
    gap_penalty: float,
    gap_mode: str,
) -> dict[str, object]:
    train_oos = float(summary.get("train_oos_r2", float("nan")))
    train_rank = float(summary.get("train_rank_ic", float("nan")))
    train_q = float(summary.get("train_quintile_ls", float("nan")))
    val_oos = float(summary.get("val_oos_r2", float("nan")))
    val_rank = float(summary.get("val_rank_ic", float("nan")))
    val_q = float(summary.get("val_quintile_ls", float("nan")))
    train_metric = float(summary.get(train_column(objective), float("nan")))
    val_metric = float(summary.get(objective_column(objective), float("nan")))
    return {
        "family": family,
        "model": model,
        "trial": int(trial),
        "features": _feature_key(columns),
        "n_features": len(columns),
        "n_val": int(summary.get("n_val", 0)),
        "train_oos_r2": train_oos,
        "train_rank_ic": train_rank,
        "train_quintile_ls": train_q,
        "val_oos_r2": val_oos,
        "val_rank_ic": val_rank,
        "val_quintile_ls": val_q,
        SELECTION_SCORE_COLUMN: selection_score(
            val_metric=val_metric,
            train_metric=train_metric,
            gap_penalty=gap_penalty,
            gap_mode=gap_mode,
        ),
        "n_test": int(summary.get("n_test", 0)),
        "oos_r2": float(summary.get("oos_r2", float("nan"))),
        "rank_ic": float(summary.get("rank_ic", float("nan"))),
        "quintile_ls": float(summary.get("quintile_ls", float("nan"))),
        "ic": float(summary.get("ic", float("nan"))),
        "r2": float(summary.get("r2", float("nan"))),
        "mae": float(summary.get("mae", float("nan"))),
        "rmse": float(summary.get("rmse", float("nan"))),
        "hit_rate": float(summary.get("hit_rate", float("nan"))),
        "sign_ls": float(summary.get("sign_ls", float("nan"))),
        "best": 0,
    }


def planned_trial_count(
    pool: SearchPool,
    *,
    n_trials: int,
    exhaustive_max: int,
) -> int:
    """Number of subset evaluations for one model under current search mode."""
    width = len(pool.searchable)
    if width <= int(exhaustive_max):
        return (1 << width) - 1
    return int(n_trials)


def run_model_search(
    spec: FamilySpec,
    *,
    model_type: str,
    pool: SearchPool,
    events: pd.DataFrame,
    n_trials: int,
    exhaustive_max: int,
    random_state: int,
    objective: str = "oos_r2",
    gap_penalty: float = 1.0,
    gap_mode: str = "abs",
    trial_bar: Any | None = None,
    preprocess_cache: dict[tuple[frozenset[str], str], Any] | None = None,
) -> pd.DataFrame:
    """Search feature subsets for one model; return trial rows (best unmarked)."""
    import optuna

    mode = normalize_gap_mode(gap_mode)
    if float(gap_penalty) < 0:
        raise ValueError(f"gap_penalty must be >= 0; got {gap_penalty!r}.")
    # Validate objective names up front.
    objective_column(objective)
    train_column(objective)
    family = str(spec.name)
    searchable = list(pool.searchable)
    if not searchable:
        raise ValueError("Searchable feature pool is empty.")
    cache = {} if preprocess_cache is None else preprocess_cache
    rows: list[dict[str, object]] = []

    def _tick_trial() -> None:
        if trial_bar is not None:
            trial_bar.update(1)

    def evaluate(columns: list[str], trial_id: int) -> float:
        if not columns:
            return float("-inf")
        summary, _record = score_feature_subset(
            spec,
            model_type=model_type,
            feature_columns=columns,
            events=events,
            preprocess_cache=cache,
        )
        row = _trial_row(
            family=family,
            model=model_type,
            trial=trial_id,
            columns=columns,
            summary=summary,
            objective=objective,
            gap_penalty=gap_penalty,
            gap_mode=mode,
        )
        rows.append(row)
        value = float(row[SELECTION_SCORE_COLUMN])
        if not np.isfinite(value):
            return float("-inf")
        return value

    if len(searchable) <= exhaustive_max:
        trial_id = 0
        for width in range(1, len(searchable) + 1):
            for chosen in itertools.combinations(searchable, width):
                evaluate(list(chosen), trial_id)
                _tick_trial()
                trial_id += 1
        return pd.DataFrame(rows)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(random_state))
    study = optuna.create_study(direction="maximize", sampler=sampler)

    def optuna_objective(trial: optuna.Trial) -> float:
        try:
            selected = [
                name
                for name in searchable
                if trial.suggest_categorical(f"use__{name}", [False, True])
            ]
            if not selected:
                # Empty feature set is invalid; tell Optuna this trial failed.
                raise optuna.TrialPruned("need at least one feature")
            return evaluate(selected, trial.number)
        finally:
            _tick_trial()

    study.optimize(optuna_objective, n_trials=int(n_trials))
    return pd.DataFrame(rows)


def mark_best_per_model(
    metrics: pd.DataFrame,
    *,
    objective: str = "oos_r2",
) -> pd.DataFrame:
    """Flag the highest selection_score trial within each model; sort."""
    if metrics.empty:
        return metrics
    if SELECTION_SCORE_COLUMN not in metrics.columns:
        raise ValueError(
            f"metrics missing {SELECTION_SCORE_COLUMN!r}; "
            "re-run feat_optuna after upgrading src.optuna_search."
        )
    # Keep objective validation for callers that still pass OBJECTIVE.
    objective_column(objective)
    out = metrics.copy()
    out["best"] = 0
    score_col = SELECTION_SCORE_COLUMN
    for model, group in out.groupby("model", sort=False):
        idx = group[score_col].astype(float).idxmax()
        out.loc[idx, "best"] = 1
    out = out.sort_values(
        ["model", score_col],
        ascending=[True, False],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    return out
