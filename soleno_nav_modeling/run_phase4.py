#!/usr/bin/env python
"""Phase 4 — Modélisation en cascade / PSPP.

Entraîne et compare 3 approches :
  - Direct   : Matière → cible (baseline)
  - Cascade  : Matière → Thermique → Mécanique → Performance
  - PSPP     : idem Cascade avec stacking OOF pour évaluation propre

Prérequis : run_pipeline.py (génère 04_NAV_feature_engineered.csv)

Usage :
    python run_phase4.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import PHASE4_CASCADE, PHASE4_COMPARISON, TABLES
from src.step24_cascade_modeling import run_step24
from src.step25_cascade_comparison import run_step25


def main() -> None:
    fe_path = TABLES / "04_NAV_feature_engineered.csv"
    if not fe_path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {fe_path}\n"
            "Exécutez d'abord : python run_pipeline.py"
        )

    print("=" * 60)
    print("PHASE 4 - Cascade / PSPP Modeling")
    print(f"Dataset : {fe_path}")
    print("=" * 60)

    print("\n=== Step 24 : Cascade training (Direct / Cascade / PSPP) ===")
    results_df = run_step24()

    print("\n=== Step 25 : Approach comparison ===")
    comparison = run_step25()

    print("\n" + "=" * 60)
    print("Phase 4 complete.")
    print(f"  Models    -> {PHASE4_CASCADE / 'models'}")
    print(f"  Metrics   -> {PHASE4_CASCADE / 'metrics'}")
    print(f"  Comparison-> {PHASE4_COMPARISON}")
    print(f"  Report    -> outputs/reports/25_cascade_comparison_report.md")
    print("")
    print("Pour visualiser dans le dashboard :")
    print("  cd soleno_nav_modeling/streamlit_app && streamlit run app.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
