"""Align the merged panel onto exchange trading sessions."""

from __future__ import annotations

import exchange_calendars as xcals
import pandas as pd

from src.prepare import _row_bar

# US cash venues that share the NYSE session calendar.
US_EXCHANGES = {
    "XNYS",
    "XNGS",
    "XNAS",
    "XNMS",
    "BATS",
    "XASE",
    "XNCM",
    "ARCX",
    "XCBO",
}
SUPPORTED_EXCHANGES = US_EXCHANGES | {"XTSE"}
IDENTITY_COLUMNS = (
    "TickerId",
    "ShortName",
    "InstrumentName",
    "Symbol",
    "Exchange",
    "CurrencyId",
)
_CALENDARS: dict[str, object] = {}


def _calendar_code(exchange: str) -> str:
    if exchange == "XTSE":
        return "XTSE"
    if exchange in US_EXCHANGES:
        return "XNYS"
    raise ValueError(f"No trading calendar for exchange {exchange!r}.")


def _calendar(code: str):
    if code not in _CALENDARS:
        _CALENDARS[code] = xcals.get_calendar(code)
    return _CALENDARS[code]


def _sessions(code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    raw = _calendar(code).sessions_in_range(start, end)
    sessions = pd.DatetimeIndex(raw)
    if sessions.tz is not None:
        sessions = sessions.tz_localize(None)
    return sessions.normalize()


def _row_exchange(panel: pd.DataFrame) -> pd.Series:
    """Exchange code from ``Exchange``, otherwise the part after ``:`` in ``ShortName``."""
    if "Exchange" in panel.columns:
        exchange = panel["Exchange"].astype("string").str.strip()
    else:
        exchange = pd.Series(pd.NA, index=panel.index, dtype="string")
    short = (
        panel["ShortName"].astype("string")
        if "ShortName" in panel.columns
        else pd.Series(pd.NA, index=panel.index, dtype="string")
    )
    parsed = short.str.split(":", n=1).str[-1].str.strip()
    has_colon = short.str.contains(":", na=False)
    blank = exchange.isna() | exchange.eq("")
    return exchange.mask(blank & has_colon, parsed)


def _keep_listed(panel: pd.DataFrame, earnings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop names whose listing has no session calendar.

    Kept venues are ``SUPPORTED_EXCHANGES``. Names with a blank or missing
    exchange, and venues such as ``XUS``, ``XLON``, and ``OTCPK``, are removed
    before alignment so they never enter the universe index.
    """
    exchange = _row_exchange(panel)
    listed = pd.DataFrame(
        {"TickerId": panel["TickerId"].to_numpy(), "exchange": exchange.to_numpy()}
    )

    def _first(values: pd.Series) -> str:
        kept = values.dropna().astype("string").str.strip()
        kept = kept[kept.ne("")]
        return str(kept.iloc[0]) if len(kept) else ""

    by_ticker = listed.groupby("TickerId", sort=False)["exchange"].agg(_first)
    keep = set(by_ticker.index[by_ticker.isin(SUPPORTED_EXCHANGES)])
    panel = panel.loc[panel["TickerId"].isin(keep)].copy()
    if "TickerId" in earnings.columns:
        earnings = earnings.loc[earnings["TickerId"].isin(keep)].copy()
    return panel, earnings


def _exchange(frame: pd.DataFrame) -> str:
    if "Exchange" in frame.columns and frame["Exchange"].notna().any():
        return str(frame["Exchange"].dropna().iloc[0])
    return str(frame["ShortName"].dropna().iloc[0]).split(":")[-1]


def _usable_market_time(value: object) -> bool:
    """True only for before-open (1) and after-close (2)."""
    code = pd.to_numeric(value, errors="coerce")
    if pd.isna(code):
        return False
    return int(code) in (1, 2)


def _place_earnings(grid: pd.DataFrame, earnings: pd.DataFrame) -> None:
    sessions = pd.DatetimeIndex(grid.index)
    for row in earnings.itertuples(index=False):
        if not _usable_market_time(row.MarketTimeId):
            continue
        day = pd.Timestamp(row.EventDate).normalize()
        if day < sessions[0] or day > sessions[-1]:
            continue
        if day in grid.index and pd.notna(grid.at[day, "EventDate"]):
            continue
        if day in grid.index:
            target = day
            time_id = row.MarketTimeId
        else:
            position = sessions.searchsorted(day, side="left")
            if position >= len(sessions):
                continue
            target = sessions[position]
            time_id = 1.0
        if pd.notna(grid.at[target, "EventDate"]):
            continue
        grid.at[target, "EventDate"] = pd.Timestamp(row.EventDate)
        grid.at[target, "MarketTimeId"] = time_id
        if "Id" in grid.columns and hasattr(row, "Id"):
            grid.at[target, "Id"] = row.Id


def align_trading_sessions(panel: pd.DataFrame, earnings: pd.DataFrame) -> pd.DataFrame:
    """Reindex each listed stock onto exchange sessions.

    Names without a supported exchange are dropped first. Closed-day
    announcements that remain are mapped forward onto the next session.
    """
    pieces: list[pd.DataFrame] = []
    earnings = earnings.copy()
    earnings["EventDate"] = pd.to_datetime(earnings["EventDate"])
    panel, earnings = _keep_listed(panel, earnings)
    if panel.empty:
        return panel.reindex(columns=panel.columns)
    groups = list(panel.groupby("TickerId", sort=False))
    with _row_bar(len(panel)) as bar:
        for _ticker, frame in groups:
            frame = frame.copy()
            frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
            frame = frame.drop_duplicates("Date", keep="last").set_index("Date").sort_index()
            code = _calendar_code(_exchange(frame))
            sessions = _sessions(code, frame.index.min(), frame.index.max())
            grid = frame.reindex(sessions)
            for column in IDENTITY_COLUMNS:
                if column in frame.columns and frame[column].notna().any():
                    grid[column] = frame[column].dropna().iloc[0]
            owned = earnings.loc[earnings["TickerId"] == _ticker]
            if len(owned):
                _place_earnings(grid, owned)
            grid.index.name = "Date"
            pieces.append(grid.reset_index())
            bar.update(len(frame))
    aligned = pd.concat(pieces, ignore_index=True)
    return aligned.reindex(columns=panel.columns)
