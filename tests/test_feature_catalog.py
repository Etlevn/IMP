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
