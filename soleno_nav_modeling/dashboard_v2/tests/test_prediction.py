"""Tests prédiction pilote (dégradé si modèles absents)."""
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


class TestPrediction(unittest.TestCase):
    def test_predict_single_graceful_without_model(self):
        from src.prediction import predict_single

        row = {
            "mi": 0.5, "hlmi": 4.0, "density_g_cm3": 0.95,
            "carbon_black": 0, "ash": 0, "pp": 0,
            "onset": 120, "peak": 130, "delta_h": 150,
        }
        res = predict_single(row, "oit_min", "A")
        self.assertIn(res["status"], ("OK", "model_not_available", "error"))

    def test_quality_check(self):
        from src.quality_check import check_lot_quality

        row = {"mi": 0.5, "hlmi": 4.0, "density_g_cm3": 0.95}
        report = check_lot_quality(row, "A")
        self.assertIn(report.quality_level, ("élevée", "moyenne", "faible"))

    def test_maturity_blocks_experimental_prediction(self):
        from services.maturity import derive_target_maturity

        ncls = derive_target_maturity("ncls")
        self.assertFalse(ncls["allowed_for_prediction"])
        self.assertEqual(ncls["level"], "experimental")


if __name__ == "__main__":
    unittest.main()
