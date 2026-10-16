"""Build the event table and print its descriptive report (no files written)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features import build_event_table
from src.prepare import load_earnings, load_sentiment, load_stock, merge_panel
from src.report import print_event_report


def main() -> None:
    stock_data = load_stock()
    earnings = load_earnings()
    historical_sentiment_scores = load_sentiment()
    df_all = merge_panel(stock_data, earnings, historical_sentiment_scores)
    df = build_event_table(df_all)
    print_event_report(df)


if __name__ == "__main__":
    main()
