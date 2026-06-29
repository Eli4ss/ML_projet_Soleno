#!/usr/bin/env python
"""Pipeline phase 2 — validation des cibles et re-modélisation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.config import TABLES
from src.step14_target_validation import run_step14
from src.step15_modeling_after_cleaning import run_step15
from src.step16_before_after import run_step16
from src.step17_explainability_after import run_step17
from src.step18_uncertainty_after import run_step18
from src.step19_registry import run_step19
from src.step20_final_report import run_step20


def main() -> None:
    fe_path = TABLES / "04_NAV_feature_engineered.csv"
    if not fe_path.exists():
        raise FileNotFoundError(f"Exécutez d'abord run_pipeline.py — fichier absent : {fe_path}")

    print("=== Phase 2 — Étape 14 : Validation des cibles ===")
    df = pd.read_csv(fe_path, low_memory=False)
    run_step14(df)

    print("=== Phase 2 — Étape 15 : ML/DL après correction ===")
    _, _, _, best_ml = run_step15()

    print("=== Phase 2 — Étape 16 : Comparaison avant/après ===")
    run_step16()

    print("=== Phase 2 — Étape 17 : Explicabilité ===")
    run_step17()

    print("=== Phase 2 — Étape 18 : Incertitude ===")
    run_step18()

    print("=== Phase 2 — Étape 19 : Registre ===")
    run_step19(best_ml)

    print("=== Phase 2 — Étape 20 : Rapport final ===")
    run_step20()

    print("Phase 2 terminée. Voir outputs/reports/20_improvement_final_report.md")


if __name__ == "__main__":
    main()
