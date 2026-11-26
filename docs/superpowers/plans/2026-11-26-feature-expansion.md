# Feature Expansion (MACRO + ENGINEERED) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Register 15 Yahoo macro Close levels and 15 no-lookahead engineered panel features so `enabled_features()` exposes a balanced expanded catalog for `feat_filter` / `feat_optuna` / `main`.

**Architecture:** Keep the existing attach order (benchmark → macro → analyst → engineered). Expand `MACRO_FEATURES` tickers (cache rebuild on mapping change). Parameterize `engineered.py` builders by window and add volume/range/ex-market helpers; register names in `config/config.py` and document in `docs/features.md`.

**Tech Stack:** Python 3.10–3.14, pandas, numpy, yfinance (existing), unittest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-11-26-feature-expansion-design.md`
- Exactly 15 `MACRO_FEATURES` and 15 `ENGINEERED_FEATURES`; Close levels only for macro
- No lookahead; `rel_vol_21d` denominator excludes today
- Do not auto-edit `config_main.py` model menus
- Do not add SUE/IBES/text/options or macro return derivatives
- `data/` stays gitignored; refresh local `macro_daily.*` after mapping change
- `requires-python = ">=3.10,<3.15"`

## File map

| File | Role |
|---|---|
| `config/config.py` | Register 15 macro tickers + 15 engineered names |
| `src/features/engineered.py` | Windowed mom/vol + new builders |
| `src/features/macro.py` | Unchanged API; new mapping triggers cache miss |
| `src/features/__init__.py` | Docstring catalog blurb only if needed |
| `docs/features.md` | Definitions for all new columns |
| `tests/test_engineered_features.py` | Unit tests on synthetic panels |
| `tests/test_feature_catalog.py` | Assert catalog sizes / membership |

---

### Task 1: Catalog size guards

**Files:**
- Create: `tests/test_feature_catalog.py`
- Modify: `config/config.py` (MACRO first in this task; ENGINEERED list updated in Task 5)

**Interfaces:**
- Consumes: `MACRO_FEATURES`, `ENGINEERED_FEATURES`, `enabled_features`
- Produces: failing tests that lock final counts (ENGINEERED assertion enabled after Task 5)

- [ ] **Step 1: Write the failing catalog test**

```python
"""Catalog membership for expanded MACRO / ENGINEERED registers."""

from __future__ import annotations

import unittest

from config.config import ENGINEERED_FEATURES, MACRO_FEATURES
from src.features import enabled_features


EXPECTED_MACRO = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "irx": "^IRX",
    "hyg": "HYG",
    "oil": "CL=F",
    "gold": "GC=F",
    "tyx": "^TYX",
    "lqd": "LQD",
    "copper": "HG=F",
    "move": "^MOVE",
    "vnq": "VNQ",
}

EXPECTED_ENGINEERED = [
    "mom_5d",
    "mom_21d",
    "mom_63d",
    "vol_5d",
    "vol_21d",
    "vol_63d",
    "rel_vol_21d",
    "amihud_21d",
    "hl_range_21d",
    "gap_1d",
    "mom_exmkt_5d",
    "mom_exmkt_21d",
    "mom_exmkt_63d",
    "log_mcap",
    "par_mkt",
]


class FeatureCatalogTests(unittest.TestCase):
    def test_macro_mapping(self) -> None:
        self.assertEqual(dict(MACRO_FEATURES), EXPECTED_MACRO)
        self.assertEqual(len(MACRO_FEATURES), 15)

    def test_engineered_list(self) -> None:
        self.assertEqual(list(ENGINEERED_FEATURES), EXPECTED_ENGINEERED)
        self.assertEqual(len(ENGINEERED_FEATURES), 15)

    def test_enabled_contains_both_groups(self) -> None:
        enabled = enabled_features()
        for name in EXPECTED_MACRO:
            self.assertIn(name, enabled)
        for name in EXPECTED_ENGINEERED:
            self.assertIn(name, enabled)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_feature_catalog -v`

Expected: FAIL on `test_macro_mapping` and/or `test_engineered_list` (current catalog smaller).

- [ ] **Step 3: Expand `MACRO_FEATURES` in `config/config.py`**

Replace the macro block with:

```python
# Maps feature column name -> yfinance ticker (src/features/macro.py).
# Cached under data/macro_daily.parquet (+ .json). Close levels only.
MACRO_FEATURES = {
    "vix": "^VIX",
    "tnx": "^TNX",
    "dxy": "DX-Y.NYB",
    "spy": "SPY",
    "qqq": "QQQ",
    "iwm": "IWM",
    "irx": "^IRX",
    "hyg": "HYG",
    "oil": "CL=F",
    "gold": "GC=F",
    "tyx": "^TYX",
    "lqd": "LQD",
    "copper": "HG=F",
    "move": "^MOVE",
    "vnq": "VNQ",
}
```

Leave `ENGINEERED_FEATURES` unchanged until Task 5 (engineered tests will still fail).

- [ ] **Step 4: Re-run catalog tests**

Run: `.venv/bin/python -m unittest tests.FAKESECRET_a1b2c3d4e5f6g7h8i9j0 -v`

Expected: PASS for macro; engineered tests still FAIL.

- [ ] **Step 5: Commit**

```bash
git add config/config.py tests/test_feature_catalog.py
git commit -m "$(cat <<'EOF'
Expand MACRO_FEATURES to fifteen Yahoo Close levels.

EOF
)"
```

---

### Task 2: Multi-window momentum and volatility

**Files:**
- Create: `tests/test_engineered_features.py`
- Modify: `src/features/engineered.py`

**Interfaces:**
- Consumes: `compute_engineered_features(frame, engineered_features=..., window=...)` existing API; extend so named features encode their window (`mom_5d` does not use the global `window` kwarg for that column)
- Produces: builders for `mom_{5,21,63}d` and `vol_{5,21,63}d`

- [ ] **Step 1: Write failing tests for mom/vol windows**

Append to `tests/test_engineered_features.py`:

```python
"""Engineered feature formulas on a synthetic single-ticker panel."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.engineered import compute_engineered_features


def _mono_panel(n: int = 70) -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=n)
    price = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {
            "TickerId": 1,
            "Date": dates,
            "LastPrice": price,
            "Open": price,
            "High": price + 1.0,
            "Low": price - 1.0,
            "Volume": np.full(n, 1000.0),
            "MarketCap": np.full(n, 1.0e9),
            "BenchmarkAdjClose": price * 0.5,
        }
    )


class MomVolTests(unittest.TestCase):
    def test_mom_windows(self) -> None:
        frame = _mono_panel()
        out = compute_engineered_features(
            frame, engineered_features=["mom_5d", "mom_21d", "mom_63d"]
        )
        # price t / price t-5 - 1 at last row: 70/65 - 1
        self.assertAlmostEqual(float(out["mom_5d"].iloc[-1]), 70 / 65 - 1)
        self.assertAlmostEqual(float(out["mom_21d"].iloc[-1]), 70 / 49 - 1)
        self.assertAlmostEqual(float(out["mom_63d"].iloc[-1]), 70 / 7 - 1)
        self.assertTrue(np.isnan(out["mom_5d"].iloc[4]))

    def test_vol_21d_matches_rolling_std(self) -> None:
        frame = _mono_panel()
        out = compute_engineered_features(frame, engineered_features=["vol_21d"])
        rets = frame["LastPrice"].pct_change()
        expected = rets.rolling(21, min_periods=21).std(ddof=1)
        self.assertAlmostEqual(float(out["vol_21d"].iloc[-1]), float(expected.iloc[-1]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m unittest tests.test_engineered_features.MomVolTests -v`

Expected: FAIL (`mom_5d` / `mom_63d` unknown or wrong).

- [ ] **Step 3: Implement windowed mom/vol in `engineered.py`**

Replace fixed `FEAT_MOM_21D` / `FEAT_VOL_21D` registration with windowed names. Keep helpers `_mom_21d` / `_vol_21d` generalized:

```python
FEAT_MOM_5D = "mom_5d"
FEAT_MOM_21D = "mom_21d"
FEAT_MOM_63D = "mom_63d"
FEAT_VOL_5D = "vol_5d"
FEAT_VOL_21D = "vol_21d"
FEAT_VOL_63D = "vol_63d"

def _mom_wd(ordered: pd.DataFrame, window: int) -> pd.Series:
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    return price.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )

def _vol_wd(ordered: pd.DataFrame, window: int) -> pd.Series:
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    ret_1d = price.groupby(ordered["TickerId"], sort=False).pct_change(
        fill_method=None
    )
    return (
        ret_1d.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .std(ddof=1)
        .reset_index(level=0, drop=True)
    )
```

Register in `_BUILDERS` with lambdas that ignore the unused global `window` kwarg and pass the feature's own window (5 / 21 / 63). Extend `ALL_ENGINEERED` with the six names (plus existing `log_mcap` / `par_mkt` for now).

Update `need_price` check to true if any `mom_*` / `vol_*` requested.

- [ ] **Step 4: Run mom/vol tests**

Run: `.venv/bin/python -m unittest tests.test_engineered_features.MomVolTests -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/features/engineered.py tests/test_engineered_features.py
git commit -m "$(cat <<'EOF'
Add 5d/21d/63d momentum and volatility engineered features.

EOF
)"
```

---

### Task 3: Relative volume, Amihud, range, gap

**Files:**
- Modify: `tests/test_engineered_features.py`
- Modify: `src/features/engineered.py`

**Interfaces:**
- Produces: `rel_vol_21d`, `amihud_21d`, `hl_range_21d`, `gap_1d`

- [ ] **Step 1: Write failing tests**

```python
class VolumeRangeTests(unittest.TestCase):
    def test_gap_1d(self) -> None:
        frame = _mono_panel(5)
        frame.loc[frame.index[-1], "Open"] = float(frame["LastPrice"].iloc[-2]) * 1.1
        out = compute_engineered_features(frame, engineered_features=["gap_1d"])
        self.assertAlmostEqual(float(out["gap_1d"].iloc[-1]), 0.1)
        self.assertTrue(np.isnan(out["gap_1d"].iloc[0]))

    def test_rel_vol_excludes_today(self) -> None:
        frame = _mono_panel(30)
        frame["Volume"] = 100.0
        frame.loc[frame.index[-1], "Volume"] = 500.0
        out = compute_engineered_features(frame, engineered_features=["rel_vol_21d"])
        # prior 21 days all 100 → mean 100; today 500 → 5.0
        self.assertAlmostEqual(float(out["rel_vol_21d"].iloc[-1]), 5.0)

    def test_hl_range_21d(self) -> None:
        frame = _mono_panel(30)
        # High-Low = 2, LastPrice at t is t+1 (1..n) → mean of 2/price over last 21
        out = compute_engineered_features(frame, engineered_features=["hl_range_21d"])
        prices = frame["LastPrice"].iloc[-21:]
        expected = float(np.mean(2.0 / prices.to_numpy()))
        self.assertAlmostEqual(float(out["hl_range_21d"].iloc[-1]), expected)

    def test_amihud_21d(self) -> None:
        frame = _mono_panel(30)
        out = compute_engineered_features(frame, engineered_features=["amihud_21d"])
        price = frame["LastPrice"]
        ret = price.pct_change().abs()
        dollar = price * frame["Volume"]
        daily = ret / dollar.replace(0, np.nan)
        expected = float(daily.iloc[-21:].mean())
        self.assertAlmostEqual(float(out["amihud_21d"].iloc[-1]), expected)
```

- [ ] **Step 2: Run tests — expect FAIL**

Run: `.venv/bin/python -m unittest tests.test_engineered_features.VolumeRangeTests -v`

- [ ] **Step 3: Implement builders**

```python
FEAT_REL_VOL_21D = "rel_vol_21d"
FEAT_AMIHUD_21D = "amihud_21d"
FEAT_HL_RANGE_21D = "hl_range_21d"
FEAT_GAP_1D = "gap_1d"

def _gap_1d(ordered: pd.DataFrame, window: int) -> pd.Series:
    del window
    open_ = pd.to_numeric(ordered["Open"], errors="coerce")
    close = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    prev_close = close.groupby(ordered["TickerId"], sort=False).shift(1)
    return (open_ - prev_close) / prev_close

def _rel_vol_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    vol = pd.to_numeric(ordered["Volume"], errors="coerce")
    # rolling mean of prior days: shift then rolling
    prior_mean = (
        vol.groupby(ordered["TickerId"], sort=False)
        .shift(1)
        .groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return vol / prior_mean.replace(0, np.nan)

def _hl_range_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    high = pd.to_numeric(ordered["High"], errors="coerce")
    low = pd.to_numeric(ordered["Low"], errors="coerce")
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    daily = (high - low) / price.replace(0, np.nan)
    return (
        daily.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )

def _amihud_21d(ordered: pd.DataFrame, window: int = 21) -> pd.Series:
    price = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    volume = pd.to_numeric(ordered["Volume"], errors="coerce")
    ret_1d = price.groupby(ordered["TickerId"], sort=False).pct_change(
        fill_method=None
    )
    dollar = price * volume
    daily = ret_1d.abs() / dollar.replace(0, np.nan)
    daily = daily.where(dollar > 0)
    return (
        daily.groupby(ordered["TickerId"], sort=False)
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )
```

Wire into `_BUILDERS` / `ALL_ENGINEERED`. Require `Open`/`High`/`Low`/`Volume` when those features are requested.

- [ ] **Step 4: Run VolumeRangeTests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/features/engineered.py tests/test_engineered_features.py
git commit -m "$(cat <<'EOF'
Add relative volume, Amihud, range, and gap engineered features.

EOF
)"
```

---

### Task 4: Excess-market momentum

**Files:**
- Modify: `tests/test_engineered_features.py`
- Modify: `src/features/engineered.py`

**Interfaces:**
- Consumes: `BenchmarkAdjClose` (same column as `par_mkt`)
- Produces: `mom_exmkt_5d`, `mom_exmkt_21d`, `mom_exmkt_63d`

- [ ] **Step 1: Write failing test**

```python
class ExMktMomTests(unittest.TestCase):
    def test_mom_exmkt_5d(self) -> None:
        frame = _mono_panel(20)
        out = compute_engineered_features(
            frame, engineered_features=["mom_exmkt_5d"]
        )
        stock = 20 / 15 - 1
        bench = (20 * 0.5) / (15 * 0.5) - 1
        self.assertAlmostEqual(float(out["mom_exmkt_5d"].iloc[-1]), stock - bench)
```

- [ ] **Step 2: Run — expect FAIL**

Run: `.venv/bin/python -m unittest tests.test_engineered_features.ExMktMomTests -v`

- [ ] **Step 3: Implement**

```python
FEAT_MOM_EXMKT_5D = "mom_exmkt_5d"
FEAT_MOM_EXMKT_21D = "mom_exmkt_21d"
FEAT_MOM_EXMKT_63D = "mom_exmkt_63d"

def _mom_exmkt(ordered: pd.DataFrame, window: int) -> pd.Series:
    from src.benchmark import BENCHMARK_COLUMN

    if BENCHMARK_COLUMN not in ordered.columns:
        raise KeyError(
            f"Panel missing {BENCHMARK_COLUMN!r} required for mom_exmkt_*."
        )
    stock = pd.to_numeric(ordered[PRICE_COLUMN], errors="coerce")
    bench = pd.to_numeric(ordered[BENCHMARK_COLUMN], errors="coerce")
    stock_mom = stock.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )
    bench_mom = bench.groupby(ordered["TickerId"], sort=False).pct_change(
        window, fill_method=None
    )
    return stock_mom - bench_mom
```

Register three builders with windows 5 / 21 / 63.

- [ ] **Step 4: Run ExMktMomTests — expect PASS**

- [ ] **Step 5: Commit**

```bash
git add src/features/engineered.py tests/test_engineered_features.py
git commit -m "$(cat <<'EOF'
Add excess-market momentum engineered features.

EOF
)"
```

---

### Task 5: Register ENGINEERED_FEATURES and pass catalog tests

**Files:**
- Modify: `config/config.py`
- Modify: `src/features/__init__.py` (module docstring only)
- Modify: `src/features/engineered.py` module docstring / `ALL_ENGINEERED` completeness

**Interfaces:**
- Produces: `ENGINEERED_FEATURES` exact list matching `EXPECTED_ENGINEERED`

- [ ] **Step 1: Set `ENGINEERED_FEATURES` in `config/config.py`**

```python
# Allowed engineered names (src/features/engineered.py).
# par_mkt: close at anchor−5 through close at anchor−2, stock minus benchmark, on decision row.
ENGINEERED_FEATURES = [
    "mom_5d",
    "mom_21d",
    "mom_63d",
    "vol_5d",
    "vol_21d",
    "vol_63d",
    "rel_vol_21d",
    "amihud_21d",
    "hl_range_21d",
    "gap_1d",
    "mom_exmkt_5d",
    "mom_exmkt_21d",
    "mom_exmkt_63d",
    "log_mcap",
    "par_mkt",
]
```

Ensure `ALL_ENGINEERED` in `engineered.py` is exactly this set (frozenset equality in a quick assert during unknown-name check is enough).

Update `src/features/__init__.py` docstring bullet for engineered to mention the new families (one short line).

- [ ] **Step 2: Run full catalog + engineered suites**

Run:

```bash
.venv/bin/python -m unittest tests.test_feature_catalog tests.test_engineered_features -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add config/config.py src/features/__init__.py src/features/engineered.py
git commit -m "$(cat <<'EOF'
Register fifteen ENGINEERED_FEATURES in the global catalog.

EOF
)"
```

---

### Task 6: Documentation

**Files:**
- Modify: `docs/features.md`
- Modify: `README.md` (one-line macro cache note if it still says only vix/tnx/dxy)

- [ ] **Step 1: Expand Macro / Engineered tables in `docs/features.md`**

Replace the Macro levels table with all 15 ticker rows (Close level note unchanged).

Replace the Engineered table with all 15 definitions from the spec (include `rel_vol` exclude-today note; `mom_exmkt_*` uses `BenchmarkAdjClose`).

- [ ] **Step 2: Fix README data table line**

If README lists only `vix / tnx / dxy`, change to “Yahoo macro Close levels from `MACRO_FEATURES` (15 series)”.

- [ ] **Step 3: Commit**

```bash
git add docs/features.md README.md
git commit -m "$(cat <<'EOF'
Document expanded macro and engineered feature catalogs.

EOF
)"
```

---

### Task 7: Local macro cache refresh smoke

**Files:**
- None committed under `data/` (gitignored)

**Interfaces:**
- Consumes: `load_macro` / `attach_macro` via existing prepare/events path

- [ ] **Step 1: Invalidate stale cache**

```bash
rm -f data/macro_daily.parquet data/macro_daily.json
```

- [ ] **Step 2: Smoke-download macros over a short window**

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import pandas as pd
from config.config import MACRO_FEATURES, DATA_DIR
from src.features.macro import load_macro

start = pd.Timestamp("2020-01-02")
end = pd.Timestamp("2020-03-31")
frame = load_macro(Path(DATA_DIR), start, end)
assert list(MACRO_FEATURES) == [c for c in frame.columns if c != "Date"]
assert len(MACRO_FEATURES) == 15
assert frame["Date"].min() <= start
assert frame["Date"].max() >= end
print("macro ok", frame.shape)
PY
```

Expected: prints `macro ok` with shape ~ (rows, 16). If `^MOVE` or a future gaps, fix by confirming ticker or raising a clear RuntimeError (do not silently drop columns).

- [ ] **Step 3: Optional feat_filter smoke (network/data heavy — skip if panel missing)**

```bash
.venv/bin/python feat_filter.py
```

Expected: `output/feat_filter.csv` includes the new feature names in the `feature` column.

- [ ] **Step 4: Commit only if code fixes were required for Yahoo gaps; otherwise no commit**

If a ticker must change, update `config/config.py` + `docs/features.md` + `tests/test_feature_catalog.py` together and commit:

```bash
git commit -m "$(cat <<'EOF'
Fix macro ticker coverage after cache smoke.

EOF
)"
```

---

## Spec coverage self-check

| Spec requirement | Task |
|---|---|
| 15 MACRO Close levels + tickers | 1, 7 |
| 15 ENGINEERED definitions | 2–5 |
| `rel_vol` excludes today | 3 |
| `mom_exmkt_*` vs benchmark | 4 |
| No macro returns / no config_main edits | Global + omitted tasks |
| `docs/features.md` | 6 |
| Cache rebuild / coverage | 7 |
| `feat_filter` acceptance | 7 |
| Preserve `mom_21d`/`vol_21d`/`log_mcap`/`par_mkt` | 2, 5 (+ existing builders) |
