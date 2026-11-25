"""Export the ~200-name stock universe to ``data/stock_universe.csv``.

One row per ``TickerId`` from the configured stock panel: identifiers and
price date coverage for WRDS / external downloads, plus the CRSP ``permno``
and Compustat ``gvkey`` looked up on WRDS when ``USE_WRDS`` is true.

gvkey mapping: ``Symbol`` -> ``crsp.stocknames`` (ticker valid inside the
price window) -> ``permno`` -> ``crsp.ccmxpf_lnkhist`` (LU/LC, primary links)
-> ``gvkey``. Only U.S. exchanges are looked up; anything else (e.g. XTSE)
keeps a blank gvkey.

Usage (from repo root):
    python script/export_universe.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from config.config import DATA_DIR
from src.prepare import load_stock

# --- run config (edit here) ---
# When True, look up CRSP permno / Compustat gvkey on WRDS (prompts if needed).
USE_WRDS = True
# Optional WRDS username; None prompts / uses the local WRDS credential store.
WRDS_USERNAME: str | None = None

OUTPUT_NAME = "stock_universe.csv"

UNIVERSE_COLUMNS = [
    "TickerId",
    "Symbol",
    "Exchange",
    "InstrumentName",
    "price_start",
    "price_end",
]

WRDS_COLUMNS = ["permno", "gvkey"]

# MIC codes covered by CRSP; other listings are left unmatched.
US_EXCHANGES = {"XNYS", "XNAS", "XNGS", "XNMS", "XNCM", "XASE", "ARCX", "BATS"}

LINK_TYPES = ("LU", "LC")
LINK_PRIMARY = ("P", "C")
# "J" = secondary share class (e.g. FOX vs FOXA); used only when no primary link exists.
LINK_SECONDARY = ("J",)

# comp.security exchg codes: NYSE, NYSE American, Nasdaq.
COMPUSTAT_US_EXCHG = (11, 12, 14)

_TICKER_RE = re.compile(r"^[A-Z0-9]{1,10}$")


def _split_short_name(short_name: object) -> tuple[object, object]:
    text = "" if short_name is None or (isinstance(short_name, float) and pd.isna(short_name)) else str(short_name)
    if ":" in text:
        symbol, exchange = text.split(":", 1)
        return symbol, exchange
    return (text or pd.NA), pd.NA


def _mode_or_last(series: pd.Series) -> object:
    clean = series.dropna()
    if clean.empty:
        return pd.NA
    modes = clean.mode(dropna=True)
    if len(modes) > 0:
        return modes.iloc[0]
    return clean.iloc[-1]


def build_universe(stock: pd.DataFrame) -> pd.DataFrame:
    """Collapse the daily stock panel to one universe row per ``TickerId``."""
    required = ["TickerId", "Date", "ShortName", "InstrumentName"]
    missing = [c for c in required if c not in stock.columns]
    if missing:
        raise KeyError(f"stock panel missing columns: {missing}")

    frame = stock.loc[:, required].copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame = frame.dropna(subset=["TickerId", "Date"])
    frame = frame.sort_values(["TickerId", "Date"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for ticker_id, group in frame.groupby("TickerId", sort=True):
        short_name = _mode_or_last(group["ShortName"])
        symbol, exchange = _split_short_name(short_name)
        rows.append(
            {
                "TickerId": int(ticker_id) if pd.notna(ticker_id) else pd.NA,
                "Symbol": symbol,
                "Exchange": exchange,
                "InstrumentName": _mode_or_last(group["InstrumentName"]),
                "price_start": group["Date"].min().strftime("%Y-%m-%d"),
                "price_end": group["Date"].max().strftime("%Y-%m-%d"),
            }
        )

    universe = pd.DataFrame(rows)
    return universe.loc[:, UNIVERSE_COLUMNS].sort_values(["Symbol", "TickerId"]).reset_index(drop=True)


def _crsp_ticker(symbol: object) -> object:
    """CRSP drops share-class suffixes (``BRK.B`` -> ``BRK``)."""
    if symbol is None or pd.isna(symbol):
        return pd.NA
    ticker = re.split(r"[./ ]", str(symbol).strip().upper())[0]
    return ticker if _TICKER_RE.match(ticker) else pd.NA


def _overlap_days(start_a, end_a, start_b, end_b) -> pd.Series:
    start = pd.concat([start_a, start_b], axis=1).max(axis=1)
    end = pd.concat([end_a, end_b], axis=1).min(axis=1)
    return (end - start).dt.days


def _pick_best(matches: pd.DataFrame, key: str, value: str) -> pd.DataFrame:
    """Keep the ``value`` with the longest window overlap for each ``key``."""
    matches = matches.loc[matches["overlap"] >= 0]
    best = (
        matches.groupby([key, value], as_index=False)["overlap"].sum()
        .sort_values([key, "overlap"], ascending=[True, False])
        .drop_duplicates(key)
    )
    return best.loc[:, [key, value]]


def _query_links(db, where: str) -> pd.DataFrame:
    """CCM links (LU/LC) matching ``where``; open-ended links get a far end date."""
    links = db.raw_sql(
        f"""
        select gvkey, lpermno as permno, linkprim, linkdt, linkenddt
        from crsp.ccmxpf_lnkhist
        where {where}
          and linktype in {LINK_TYPES}
          and linkprim in {LINK_PRIMARY + LINK_SECONDARY}
        """,
        date_cols=["linkdt", "linkenddt"],
    )
    links["permno"] = links["permno"].astype("Int64")
    links["linkenddt"] = links["linkenddt"].fillna(pd.Timestamp.max.normalize())
    return links


def _pick_link(frame: pd.DataFrame, links: pd.DataFrame, on: str, value: str) -> pd.DataFrame:
    """Best CCM ``value`` per ``TickerId``: primary links beat secondary (J), then longest overlap."""
    hits = frame.loc[:, ["TickerId", on, "_start", "_end"]].merge(links, on=on, how="inner")
    hits["overlap"] = _overlap_days(hits["_start"], hits["_end"], hits["linkdt"], hits["linkenddt"])
    hits = hits.loc[hits["overlap"] >= 0]
    hits["_secondary"] = ~hits["linkprim"].isin(LINK_PRIMARY)
    hits = hits.loc[hits["_secondary"] == hits.groupby("TickerId")["_secondary"].transform("min")]
    return _pick_best(hits, "TickerId", value)


def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def attach_gvkey(universe: pd.DataFrame, db) -> pd.DataFrame:
    """Add ``permno`` and ``gvkey`` via CRSP stocknames + the CCM link table.

    Rows still without a gvkey (e.g. tickers renamed after the price window,
    like SQ -> XYZ) fall back to Compustat's current ticker in ``comp.security``.
    """
    frame = universe.copy()
    frame["_ticker"] = frame["Symbol"].map(_crsp_ticker)
    frame.loc[~frame["Exchange"].isin(US_EXCHANGES), "_ticker"] = pd.NA
    frame["_start"] = pd.to_datetime(frame["price_start"])
    frame["_end"] = pd.to_datetime(frame["price_end"])
    frame["permno"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["gvkey"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame = frame.set_index("TickerId", drop=False)

    # Tickers / gvkeys are validated or come from WRDS, so inlining them is safe.
    tickers = sorted(frame["_ticker"].dropna().unique())
    if not tickers:
        return frame.drop(columns=["_ticker", "_start", "_end"]).reset_index(drop=True)

    # 1) ticker active during the price window -> permno
    names = db.raw_sql(
        f"""
        select permno, ticker, namedt, nameenddt
        from crsp.stocknames
        where ticker in ({_sql_list(tickers)})
        """,
        date_cols=["namedt", "nameenddt"],
    )
    names["permno"] = names["permno"].astype("Int64")
    hits = frame.loc[:, ["TickerId", "_ticker", "_start", "_end"]].reset_index(drop=True).merge(
        names, left_on="_ticker", right_on="ticker", how="inner"
    )
    hits["overlap"] = _overlap_days(hits["_start"], hits["_end"], hits["namedt"], hits["nameenddt"])
    permnos = _pick_best(hits, "TickerId", "permno").set_index("TickerId")["permno"]
    frame.loc[permnos.index, "permno"] = permnos

    # 2) permno -> gvkey through CCM
    permno_ids = frame["permno"].dropna().unique()
    if len(permno_ids):
        links = _query_links(db, f"lpermno in ({', '.join(str(int(p)) for p in permno_ids)})")
        gvkeys = _pick_link(frame.reset_index(drop=True), links, "permno", "gvkey").set_index("TickerId")["gvkey"]
        frame.loc[gvkeys.index, "gvkey"] = gvkeys

    # 3) fallback: Compustat current ticker -> gvkey (and permno back through CCM)
    todo = frame.loc[frame["gvkey"].isna() & frame["_ticker"].notna()]
    if not todo.empty:
        sec = db.raw_sql(
            f"""
            select gvkey, tic, dldtei
            from comp.security
            where tic in ({_sql_list(sorted(todo["_ticker"].unique()))})
              and exchg in {COMPUSTAT_US_EXCHG}
            """,
            date_cols=["dldtei"],
        )
        hits = todo.loc[:, ["TickerId", "_ticker", "_start"]].reset_index(drop=True).merge(
            sec, left_on="_ticker", right_on="tic", how="inner"
        )
        hits = hits.loc[hits["dldtei"].isna() | (hits["dldtei"] >= hits["_start"])]
        hits = hits.drop_duplicates(["TickerId", "gvkey"]).drop_duplicates("TickerId", keep=False)
        fallback = hits.set_index("TickerId")["gvkey"]
        frame.loc[fallback.index, "gvkey"] = fallback

        need_permno = frame.loc[fallback.index].loc[lambda f: f["permno"].isna()]
        if not need_permno.empty:
            links = _query_links(db, f"gvkey in ({_sql_list(need_permno['gvkey'].unique())})")
            back = _pick_link(need_permno.reset_index(drop=True), links, "gvkey", "permno")
            back = back.set_index("TickerId")["permno"]
            frame.loc[back.index, "permno"] = back

    return frame.drop(columns=["_ticker", "_start", "_end"]).reset_index(drop=True)


def export_universe(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    db=None,
) -> Path:
    """Write the universe CSV; pass a ``wrds.Connection`` as ``db`` to add permno / gvkey."""
    folder = DATA_DIR if data_dir is None else data_dir
    out = folder / OUTPUT_NAME if output_path is None else output_path
    stock = load_stock(folder)
    universe = build_universe(stock)
    if db is not None:
        universe = attach_gvkey(universe, db).loc[:, UNIVERSE_COLUMNS + WRDS_COLUMNS]
    out.parent.mkdir(parents=True, exist_ok=True)
    universe.to_csv(out, index=False)
    return out


def main() -> None:
    db = None
    if USE_WRDS:
        import wrds

        db = (
            wrds.Connection(wrds_username=WRDS_USERNAME)
            if WRDS_USERNAME
            else wrds.Connection()
        )
    try:
        path = export_universe(db=db)
    finally:
        if db is not None:
            db.close()

    universe = pd.read_csv(path, dtype={"gvkey": "string"})
    print(f"wrote {path}  rows={len(universe)}  cols={len(universe.columns)}")
    print(
        f"symbols={universe['Symbol'].nunique(dropna=True)}  "
        f"exchanges={sorted(universe['Exchange'].dropna().unique().tolist())}"
    )
    if "gvkey" in universe.columns:
        missing = universe.loc[universe["gvkey"].isna(), ["TickerId", "Symbol", "Exchange", "InstrumentName"]]
        print(f"gvkey matched={len(universe) - len(missing)}/{len(universe)}")
        if not missing.empty:
            print("unmatched:")
            print(missing.to_string(index=False))
    print(universe.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
