"""Feature-pool helpers and per-model Optuna / exhaustive search."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from feat_filter import SUGGEST_LABELS
from src.pipeline import score_feature_subset
from src.train import FamilySpec

OBJECTIVE_CHOICES: tuple[str, ...] = ("oos_r2", "rank_ic", "quintile_ls")


def objective_column(objective: str) -> str:
    """Map a public OBJECTIVE name to the val column used for selection."""
    name = str(objective).strip()
    if name not in OBJECTIVE_CHOICES:
        raise ValueError(
            f"Unknown OBJECTIVE={objective!r}; expected one of {OBJECTIVE_CHOICES}."
        )
    return f"val_{name}"


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
) -> dict[str, object]:
    return {
        "family": family,
        "model": model,
        "trial": int(trial),
        "features": _feature_key(columns),
        "n_features": len(columns),
        "n_val": int(summary.get("n_val", 0)),
        "val_oos_r2": float(summary.get("val_oos_r2", float("nan"))),
        "val_rank_ic": float(summary.get("val_rank_ic", float("nan"))),
        "val_quintile_ls": float(summary.get("val_quintile_ls", float("nan"))),
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
    trial_bar: Any | None = None,
    preprocess_cache: dict[tuple[frozenset[str], str], Any] | None = None,
) -> pd.DataFrame:
    """Search feature subsets for one model; return trial rows (best unmarked)."""
    import optuna

    score_col = objective_column(objective)
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
        rows.append(
            _trial_row(
                family=family,
                model=model_type,
                trial=trial_id,
                columns=columns,
                summary=summary,
            )
        )
        value = float(summary.get(score_col, float("nan")))
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
    """Flag the highest validation-objective trial within each model; sort."""
    if metrics.empty:
        return metrics
    score_col = objective_column(objective)
    if score_col not in metrics.columns:
        raise ValueError(f"metrics missing objective column {score_col!r}.")
    out = metrics.copy()
    out["best"] = 0
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
