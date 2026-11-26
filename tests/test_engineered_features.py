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
            "Open": price.copy(),
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
        self.assertAlmostEqual(float(out["rel_vol_21d"].iloc[-1]), 5.0)

    def test_hl_range_21d(self) -> None:
        frame = _mono_panel(30)
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


class ExMktMomTests(unittest.TestCase):
    def test_mom_exmkt_5d(self) -> None:
        frame = _mono_panel(20)
        out = compute_engineered_features(
            frame, engineered_features=["mom_exmkt_5d"]
        )
        stock = 20 / 15 - 1
        bench = (20 * 0.5) / (15 * 0.5) - 1
        self.assertAlmostEqual(float(out["mom_exmkt_5d"].iloc[-1]), stock - bench)


if __name__ == "__main__":
    unittest.main()
