"""Batch sweep of feat_optuna over OBJECTIVE × GAP_MODE × INCLUDE_SUGGEST.

Runs the full factorial grid in-process via ``feat_optuna.run_feat_optuna``,
reusing one loaded events table. Each finished model appends one row to
``output/feat_optuna_batch.csv`` with leading ``objective,gap,suggest``
columns (same shape as ``tmp/feat_optuna_merged.csv``).

Progress UI matches ``feat_optuna`` (sticky model + trial bars) with an
extra top cell bar labeled ``{objective}|{gap}|{suggest}``.

Edit ``RESUME`` / grid tuples below, then from repo root:

    python batch/batch_optuna.py

Shared settings (MODELS, N_TRIALS, EXHAUSTIVE_MAX, GAP_PENALTY, SAVE_TRIALS,
PREPROCESS_CACHE_MAX, whitelist/blacklist, …) are read live from
``feat_optuna`` — do not redefine them here. Only the grid + RESUME below
are batch-local.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import OUTPUT_DIR  # noqa: E402
from src.compact_json import write_compact_json  # noqa: E402
from src.features import enabled_features  # noqa: E402
from src.optuna_search import (  # noqa: E402
    BoundedPreprocessCache,
    build_search_pool,
    planned_trial_count,
)
from src.prepare import sticky_stack  # noqa: E402

import feat_optuna as fo  # noqa: E402

# ======================================== [CONFIG] ========================================

# True: skip (objective, gap, suggest, family, model) already in the batch CSV.
# False: delete feat_optuna_batch.csv at start and rerun the full grid.
RESUME = False

# Search grid only — everything else comes from feat_optuna module globals.
OBJECTIVES: tuple[str, ...] = ("oos_r2", "rank_ic", "quintile_ls")
GAP_MODES: tuple[str, ...] = ("none", "abs", "pos")
SUGGEST_GRID: tuple[tuple[str, ...], ...] = (
    ("keep",),
    ("keep", "weak"),
    ("keep", "weak", "noise"),
)

BATCH_BEST_PATH = OUTPUT_DIR / "feat_optuna_batch.csv"
BATCH_CONFIG_PATH = OUTPUT_DIR / "config_optuna_batch.json"

# ========================================================================================


def iter_grid() -> list[tuple[str, str, tuple[str, ...]]]:
    """Full factorial cells: (objective, gap_mode, include_suggest)."""
    return list(itertools.product(OBJECTIVES, GAP_MODES, SUGGEST_GRID))


def _load_done_keys(path: Path) -> set[tuple[str, str, str, str, str]]:
    """Return completed (objective, gap, suggest, family, model) keys."""
    if not path.is_file():
        return set()
    frame = pd.read_csv(path)
    required = ("objective", "gap", "suggest", "family", "model")
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(
            f"{path} missing columns for RESUME: {missing}. "
            f"Set RESUME=False or fix/delete the file."
        )
    keys: set[tuple[str, str, str, str, str]] = set()
    for row in frame.itertuples(index=False):
        keys.add(
            (
                str(getattr(row, "objective")),
                str(getattr(row, "gap")),
                str(getattr(row, "suggest")),
                str(getattr(row, "family")),
                str(getattr(row, "model")),
            )
        )
    return keys


def _skip_models_for_cell(
    done: set[tuple[str, str, str, str, str]],
    *,
    objective: str,
    gap: str,
    suggest: str,
    pairs: list,
) -> set[tuple[str, str]]:
    skip: set[tuple[str, str]] = set()
    for spec, model_type in pairs:
        key = (objective, gap, suggest, spec.name, model_type)
        if key in done:
            skip.add((spec.name, model_type))
    return skip


def _max_trial_total(catalog: list[str]) -> int:
    """Progress-bar trial total from feat_optuna N_TRIALS / EXHAUSTIVE_MAX."""
    table = pd.read_csv(fo.FILTER_PATH)
    totals = [
        planned_trial_count(
            build_search_pool(
                table,
                whitelist=fo.FEATURE_WHITELIST,
                blacklist=fo.FEATURE_BLACKLIST,
                include_suggest=list(include_suggest),
                catalog=catalog,
            ),
            n_trials=fo.N_TRIALS,
            exhaustive_max=fo.EXHAUSTIVE_MAX,
        )
        for include_suggest in SUGGEST_GRID
    ]
    return max(1, max(totals) if totals else 1)


def main() -> None:
    if not fo.FILTER_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {fo.FILTER_PATH}. Run `python feat_filter.py` first."
        )

    grid = iter_grid()
    if len(grid) != 27:
        raise RuntimeError(f"Expected 27 grid cells; got {len(grid)}.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not RESUME and BATCH_BEST_PATH.exists():
        BATCH_BEST_PATH.unlink()

    catalog = enabled_features()
    pairs = fo._selected_specs()
    events = fo._load_events(list(catalog))
    if int(fo.PREPROCESS_CACHE_MAX) < 1:
        raise ValueError(
            f"PREPROCESS_CACHE_MAX must be >= 1; got {fo.PREPROCESS_CACHE_MAX!r}."
        )
    preprocess_cache = BoundedPreprocessCache(fo.PREPROCESS_CACHE_MAX)
    trial_total = _max_trial_total(catalog)

    write_compact_json(
        BATCH_CONFIG_PATH,
        {
            "resume": bool(RESUME),
            "objectives": list(OBJECTIVES),
            "gap_modes": list(GAP_MODES),
            "suggest_grid": [list(item) for item in SUGGEST_GRID],
            "n_grid_cells": len(grid),
            "batch_best_path": str(BATCH_BEST_PATH),
            "gap_penalty": float(fo.GAP_PENALTY),
            "n_trials": int(fo.N_TRIALS),
            "exhaustive_max": int(fo.EXHAUSTIVE_MAX),
            "save_trials": bool(fo.SAVE_TRIALS),
            "preprocess_cache_max": int(fo.PREPROCESS_CACHE_MAX),
            "models": {name: list(fo.MODELS[name]) for name in fo.MODELS},
            "feature_whitelist": list(fo.FEATURE_WHITELIST),
            "feature_blacklist": list(fo.FEATURE_BLACKLIST),
            "filter_path": str(fo.FILTER_PATH),
            "feature_columns": list(catalog),
            "shared": fo._shared_from_pairs(pairs),
        },
    )

    done = _load_done_keys(BATCH_BEST_PATH) if RESUME else set()
    n_models = len(pairs)
    n_cells_done = 0
    n_models_run = 0
    n_models_skipped = 0

    with sticky_stack((len(grid), n_models, trial_total)) as (
        cell_bar,
        model_bar,
        trial_bar,
    ):
        for objective, gap_mode, include_suggest in grid:
            suggest = fo.suggest_label(include_suggest)
            if cell_bar is not None:
                cell_bar.set_description_str(
                    f"{objective}|{gap_mode}|{suggest} ",
                    refresh=True,
                )
            skip = _skip_models_for_cell(
                done,
                objective=objective,
                gap=gap_mode,
                suggest=suggest,
                pairs=pairs,
            )
            n_skip = len(skip)
            n_run = n_models - n_skip
            if n_run == 0:
                n_cells_done += 1
                n_models_skipped += n_skip
                if cell_bar is not None:
                    cell_bar.update(1)
                continue

            fo.run_feat_optuna(
                objective=objective,
                gap_mode=gap_mode,
                include_suggest=include_suggest,
                gap_penalty=fo.GAP_PENALTY,
                events=events,
                catalog=catalog,
                pairs=pairs,
                preprocess_cache=preprocess_cache,
                best_path=BATCH_BEST_PATH,
                config_path=BATCH_CONFIG_PATH,
                row_prefix={
                    "objective": objective,
                    "gap": gap_mode,
                    "suggest": suggest,
                },
                skip_models=skip,
                append_best=True,
                write_config=False,
                clear_best_path=False,
                print_summary=False,
                model_bar=model_bar,
                trial_bar=trial_bar,
            )
            done = _load_done_keys(BATCH_BEST_PATH)
            n_cells_done += 1
            n_models_run += n_run
            n_models_skipped += n_skip
            if cell_bar is not None:
                cell_bar.update(1)

    print()
    print(
        f"[DONE] cells={n_cells_done}/{len(grid)} "
        f"models_run={n_models_run} models_skipped={n_models_skipped}"
    )
    print(f"[WROTE] {BATCH_BEST_PATH}")
    print(f"[WROTE] {BATCH_CONFIG_PATH}")


if __name__ == "__main__":
    main()
