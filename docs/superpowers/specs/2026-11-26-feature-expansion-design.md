# Feature expansion (MACRO + ENGINEERED) — Design

Date: 2026-11-26  
Branch context: `models&features`  
Status: approved for planning

## Goal

Expand the earnings-event feature catalog using (1) Yahoo/yfinance daily series in `MACRO_FEATURES` and (2) local panel constructions in `ENGINEERED_FEATURES`, keeping the existing attach pipeline and register-via-config pattern. Target: **15 MACRO + 15 ENGINEERED** columns, all Close-level macros and no-lookahead engineered signals suitable for decision-day prediction of announcement-window returns.

## Approach

**Extend existing groups in place (no new framework).** Update `config/config.py` mappings/lists; parameterize builders in `src/features/engineered.py`; reuse `src/features/macro.py` download/cache; document in `docs/features.md`. Do not split modules or introduce a declarative recipe engine in this round.

## Feature inventory

### MACRO_FEATURES (15 Close levels)

Cached under `data/macro_daily.parquet` (+ `.json`). Left-merged on `Date`.

| Column | Yahoo ticker |
| --- | --- |
| `vix` | `^VIX` |
| `tnx` | `^TNX` |
| `dxy` | `DX-Y.NYB` |
| `spy` | `SPY` |
| `qqq` | `QQQ` |
| `iwm` | `IWM` |
| `irx` | `^IRX` |
| `hyg` | `HYG` |
| `oil` | `CL=F` |
| `gold` | `GC=F` |
| `tyx` | `^TYX` |
| `lqd` | `LQD` |
| `copper` | `HG=F` |
| `move` | `^MOVE` |
| `vnq` | `VNQ` |

Representation: **Close level only** (same as current `vix` / `tnx` / `dxy`). No macro return/change derivatives in this round.

### ENGINEERED_FEATURES (15)

Built per `TickerId` on the aligned panel after benchmark attach. Windows are trading days. Existing names `mom_21d`, `vol_21d`, `log_mcap`, `par_mkt` stay.

| Column | Definition |
| --- | --- |
| `mom_5d` / `mom_21d` / `mom_63d` | `LastPrice` percent change over W trading days |
| `vol_5d` / `vol_21d` / `vol_63d` | Rolling std (`ddof=1`) of daily simple returns over W days (full window) |
| `rel_vol_21d` | `Volume /` mean volume of the **prior 21** trading days (**exclude today**) |
| `amihud_21d` | 21-day mean of `|ret_1d| / dollar_volume` where dollar volume ≈ `LastPrice * Volume` |
| `hl_range_21d` | 21-day mean of `(High - Low) / LastPrice` |
| `gap_1d` | `(Open - previous close) / previous close` |
| `mom_exmkt_5d` / `mom_exmkt_21d` / `mom_exmkt_63d` | Stock W-day simple return minus same-window return of `BenchmarkAdjClose` |
| `log_mcap` | `log(MarketCap)` when positive |
| `par_mkt` | Pre-announcement stock minus benchmark return, anchor−5→anchor−2, written on decision row |

## Implementation touchpoints

1. `config/config.py` — set the 15+15 registers; comments list allowed names.
2. `src/features/macro.py` — no API change required; mapping change invalidates cache equality check and triggers re-download.
3. `src/features/engineered.py` — parameterize momentum/vol by window; register new builders; extend `ALL_ENGINEERED`.
4. `src/features/__init__.py` / `src/events.py` — keep attach order: benchmark → macro → analyst → engineered.
5. `docs/features.md` — document every new column and ticker.

## No-leakage and missingness rules

- Rolling features use only dates ≤ current row within each `TickerId`, sorted by `Date`.
- `rel_vol_21d` denominator excludes the current day.
- `amihud_21d` / `hl_range_21d` use a trailing W-day window consistent with existing `vol_21d` (realized same-day OHLC/returns allowed).
- Non-positive `Volume` or undefined dollar volume → Amihud/rel_vol missing for that day contribution; insufficient history → feature NaN.
- Macro missing calendar days remain NaN; imputed later via `IMPUTE_STRATEGY` on the event table.
- Do not use post-announcement prices or target columns as features.

## Risks

- Larger catalog pushes Optuna away from exhaustive enumeration more often; rely on `feat_filter`, `FEATURE_WHITELIST` / `FEATURE_BLACKLIST`, and `EXHAUSTIVE_MAX`.
- Some Yahoo series (`^MOVE`, futures) may have gaps; fail loudly if coverage does not span the panel date range.
- Local `data/macro_daily.*` must be refreshed after mapping changes; `data/` stays gitignored.

## Non-goals

- SUE / IBES / fundamentals, text, options.
- Macro returns or multi-day macro changes as separate features.
- Splitting `engineered.py` into multiple modules or a recipe-table framework.
- Auto-updating `config_main.py` model feature menus.

## Acceptance criteria

1. `MACRO_FEATURES` and `ENGINEERED_FEATURES` each list exactly 15 enabled names; all appear in `enabled_features()`.
2. After invalidating the old macro cache, rebuild attaches all 15 macro columns over the sample date range.
3. New engineered columns have no lookahead; `mom_21d` / `vol_21d` / `log_mcap` / `par_mkt` match prior behavior within floating-point tolerance.
4. `docs/features.md` documents the expanded sets.
5. `feat_filter.py` runs successfully over the expanded catalog.

## Out of scope follow-ups

- Optional later: macro changes as engineered transforms; file split if `engineered.py` grows further; curated Optuna whitelists after `feat_filter` suggests.
