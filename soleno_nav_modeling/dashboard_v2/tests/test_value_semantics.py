"""Tests sémantique valeurs et journal pilote."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

V2 = Path(__file__).resolve().parents[1]
MVP = V2.parent / "streamlit_app"
for p in (str(V2), str(MVP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: E402


class TestValueSemantics(unittest.TestCase):
    def test_artificial_zero_mi(self):
        from services.value_semantics import is_artificial_zero

        self.assertTrue(is_artificial_zero("mi", 0))
        self.assertFalse(is_artificial_zero("mi", 0.5))

    def test_true_zero_carbon_black(self):
        from services.value_semantics import column_value_breakdown

        s = pd.Series([None, 0, 1.2, 0])
        b = column_value_breakdown(s, "carbon_black", 4)
        self.assertEqual(b["n_absent"], 1)
        self.assertEqual(b["n_true_zero"], 2)
        self.assertEqual(b["n_measured"], 3)

    def test_breakdown_target_no_zero(self):
        from services.value_semantics import column_value_breakdown

        s = pd.Series([None, 0, 10.0])
        b = column_value_breakdown(s, "ncls", 3)
        self.assertEqual(b["n_artificial_zero"], 1)
        self.assertEqual(b["n_measured"], 1)


class TestPilotDelete(unittest.TestCase):
    def test_delete_test_predictions(self):
        from v2_config import settings
        from services import pilot_journal as pj

        with tempfile.TemporaryDirectory() as tmp:
            settings.PILOT_JOURNAL_DIR = Path(tmp)
            settings.PILOT_JOURNAL_FILE = Path(tmp) / "journal.csv"
            pj.PILOT_JOURNAL_DIR = settings.PILOT_JOURNAL_DIR
            pj.PILOT_JOURNAL_FILE = settings.PILOT_JOURNAL_FILE
            pj.append_pilot_prediction({"lot_id": "lot_pilote", "target": "oit_min", "predicted_value": 1})
            pj.append_pilot_prediction({"lot_id": "REAL-42", "target": "oit_min", "predicted_value": 2})
            self.assertEqual(len(pj.load_pilot_journal()), 2)
            n = pj.delete_test_predictions()
            self.assertEqual(n, 1)
            self.assertEqual(len(pj.load_pilot_journal()), 1)


if __name__ == "__main__":
    unittest.main()
