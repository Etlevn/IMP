"""Load and merge the earnings-movements parquet tables (no modeling prep)."""

from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from config.config import (
    DATA_DIR,
    EARNINGS_FILE,
    SENTIMENT_FILE,
    SENTIMENT_TOLERANCE_DAYS,
    STOCK_FILE,
)

ROOT = Path(__file__).resolve().parents[1]

_CHUNK = 50_000
USD_CURRENCY_ID = 840
# Transient row bars sit above this many pinned footer bars.
_ROW_POS = 0


def _row_bar(total: int, desc: str = "") -> tqdm:
    label = f"{desc} " if desc and not desc.endswith(" ") else desc
    return tqdm(
        total=total,
        leave=False,
        position=_ROW_POS,
        dynamic_ncols=True,
        desc=label,
        unit="",
        miniters=1,
        mininterval=0,
        bar_format="{desc}{bar}| {n_fmt}/{total_fmt}",
    )


@contextmanager
def sticky_stack(totals: Sequence[int]) -> Iterator[tuple[tqdm | None, ...]]:
    """Pin one or more footer bars (outer first / top, inner last / bottom).

    ``totals[0]`` is the top sticky (e.g. models); ``totals[-1]`` is the
    bottom sticky (e.g. trials). Prints go through ``tqdm.write``. Row bars
    opened while active use positions above the stack.
    """
    global _ROW_POS
    values = [int(total) for total in totals]
    if not values:
        yield ()
        return

    previous_row = _ROW_POS
    previous_print = builtins.print
    n = len(values)
    bars: list[tqdm | None] = []
    for index, total in enumerate(values):
        if total <= 0:
            bars.append(None)
            continue
        # tqdm position 0 is the top line; later indexes render below.
        bars.append(
            tqdm(
                total=total,
                leave=False,
                position=index,
                dynamic_ncols=True,
                desc="",
                miniters=1,
                mininterval=0,
                bar_format="{desc}{bar}| {n_fmt}/{total_fmt}",
            )
        )
    _ROW_POS = previous_row + n

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file not in (None, sys.stdout):
            previous_print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        tqdm.write(sep.join(str(arg) for arg in args), file=sys.stdout, end=end)
        if flush:
            sys.stdout.flush()

    builtins.print = _print
    try:
        yield tuple(bars)
    finally:
        builtins.print = previous_print
        _ROW_POS = previous_row
        # Close bottom-up so tqdm clears nested positions without leaving the
        # cursor mid-line (which eats the next print's first character).
        for bar in reversed(bars):
            if bar is not None:
                bar.close()
        sys.stdout.write("\n")
        sys.stdout.flush()


@contextmanager
def sticky_model_bar(total: int) -> Iterator[tqdm | None]:
    """Pin one model bar to the bottom row while training is running."""
    with sticky_stack((total,)) as bars:
        yield bars[0] if bars else None


def _read_parquet(path: Path, desc: str = "") -> pd.DataFrame:
    file = pq.ParquetFile(path)
    batches: list[pa.RecordBatch] = []
    with _row_bar(file.metadata.num_rows, desc=desc) as bar:
        for batch in file.iter_batches(batch_size=_CHUNK):
            batches.append(batch)
            bar.update(batch.num_rows)
    if not batches:
        return pd.DataFrame()
    return pa.Table.from_batches(batches).to_pandas()


def load_stock(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    stock = _read_parquet(folder / STOCK_FILE, desc="stock")
    return stock[stock["CurrencyId"] == USD_CURRENCY_ID].reset_index(drop=True)


def load_earnings(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    return _read_parquet(folder / EARNINGS_FILE, desc="earnings")


def load_sentiment(data_dir: Path | None = None) -> pd.DataFrame:
    folder = DATA_DIR if data_dir is None else data_dir
    sentiment = _read_parquet(folder / SENTIMENT_FILE, desc="sentiment")
    sentiment["date"] = pd.to_datetime(sentiment["date"])
    sentiment = sentiment.drop_duplicates(["symbol", "date"], keep="last")
    sentiment["symbol"] = sentiment["symbol"].astype("string")
    return sentiment.sort_values("date")


def _merge_sentiment(
    prices: pd.DataFrame,
    sentiment: pd.DataFrame,
    tolerance_days: int = SENTIMENT_TOLERANCE_DAYS,
) -> pd.DataFrame:
    """Attach the latest rating snapshot on or before each price date.

    Snapshots are monthly. A rating older than the tolerance is left missing.
    """
    frame = prices.copy()
    frame["_row"] = range(len(frame))
    frame["Date"] = pd.to_datetime(frame["Date"])
    frame["Symbol"] = frame["Symbol"].astype("string")

    ratings = sentiment.copy()
    ratings["date"] = pd.to_datetime(ratings["date"])
    ratings["symbol"] = ratings["symbol"].astype("string")
    ratings = ratings.drop_duplicates(["symbol", "date"], keep="last")
    ratings = ratings.sort_values("date")

    parts: list[pd.DataFrame] = []
    with _row_bar(len(frame), desc="ratings") as bar:
        for start in range(0, len(frame), _CHUNK):
            chunk = frame.iloc[start : start + _CHUNK]
            ordered = chunk.sort_values("Date")
            merged = pd.merge_asof(
                ordered,
                ratings,
                left_on="Date",
                right_on="date",
                left_by="Symbol",
                right_by="symbol",
                direction="backward",
                tolerance=pd.Timedelta(days=tolerance_days),
            )
            parts.append(merged)
            bar.update(len(chunk))
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("_row")
        .drop(columns="_row")
        .reset_index(drop=True)
    )


def merge_panel(
    stock_data: pd.DataFrame,
    earnings: pd.DataFrame,
    historical_sentiment_scores: pd.DataFrame,
) -> pd.DataFrame:
    """Merge earnings on Date/TickerId, then asof-carry monthly ratings."""
    earnings = earnings.copy()
    earnings["EventDate"] = pd.to_datetime(earnings["EventDate"])

    parts: list[pd.DataFrame] = []
    with _row_bar(len(stock_data), desc="earnings+") as bar:
        for start in range(0, len(stock_data), _CHUNK):
            chunk = stock_data.iloc[start : start + _CHUNK]
            parts.append(
                chunk.merge(
                    earnings,
                    how="left",
                    left_on=["Date", "TickerId"],
                    right_on=["EventDate", "TickerId"],
                )
            )
            bar.update(len(chunk))
    temp = pd.concat(parts, ignore_index=True)
    temp[["Symbol", "Exchange"]] = temp["ShortName"].str.split(":", expand=True)
    return _merge_sentiment(temp, historical_sentiment_scores)
