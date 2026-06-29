"""Tests de chargement dashboard_v2."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

V2 = Path(__file__).resolve().parents[1]
MVP = V2.parent / "streamlit_app"
for p in (str(V2), str(MVP)):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: E402


class TestLoading(unittest.TestCase):
    def test_maturity_table_not_empty(self):
        from services.maturity import build_target_maturity_table

        df = build_target_maturity_table()
        self.assertFalse(df.empty)
        self.assertIn("level", df.columns)

    def test_pilot_targets_from_phase5(self):
        from services.maturity import pilot_targets

        pilots = pilot_targets()
        self.assertIn("oit_min", pilots)
        self.assertNotIn("ncls", pilots)

    def test_validation_summary(self):
        from services.validation_service import validation_summary_table

        df = validation_summary_table()
        if not df.empty:
            self.assertIn("Cible", df.columns)

    def test_nav_stats(self):
        from services.data_service import nav_dataset_stats

        stats = nav_dataset_stats()
        self.assertIn("n_rows", stats)

    def test_pilot_journal_roundtrip(self):
        import tempfile
        from v2_config import settings
        from services import pilot_journal as pj

        with tempfile.TemporaryDirectory() as tmp:
            settings.PILOT_JOURNAL_DIR = Path(tmp)
            settings.PILOT_JOURNAL_FILE = Path(tmp) / "test_journal.csv"
            pj.PILOT_JOURNAL_DIR = settings.PILOT_JOURNAL_DIR
            pj.PILOT_JOURNAL_FILE = settings.PILOT_JOURNAL_FILE
            pid = pj.append_pilot_prediction({
                "lot_id": "TEST-001",
                "target": "oit_min",
                "model_used": "direct_A",
                "model_version": "A",
                "inputs_json": {"mi": 0.5},
                "predicted_value": 25.0,
                "prediction_unit": "min",
            })
            df = pj.load_pilot_journal()
            self.assertEqual(len(df), 1)
            self.assertEqual(str(df.iloc[0]["prediction_id"]), pid)


if __name__ == "__main__":
    unittest.main()
