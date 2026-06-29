#!/usr/bin/env python
"""Phase 3 — Généralisation industrielle (GroupKFold, temporel, erreurs par groupe)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import PHASE2_VALIDATION, TABLES
from src.step21_robust_generalization import run_step21
from src.step22_error_analysis import run_step22
from src.step23_ab_robust_comparison import run_step23


def main() -> None:
    fe = TABLES / "04_NAV_feature_engineered.csv"
    if not fe.exists():
        raise FileNotFoundError("Exécutez run_pipeline.py d'abord.")
    if not (PHASE2_VALIDATION / "corrected_target_datasets").exists():
        raise FileNotFoundError("Exécutez run_phase2.py d'abord.")

    print("=== Étape 21 : Validation robuste ===")
    results = run_step21()
    print("=== Étape 22 : Analyse erreurs par groupe ===")
    run_step22()
    print("=== Étape 23 : Comparaison A vs B ===")
    run_step23(results)
    print("Phase 3 terminée. Voir outputs/21_robust_generalization/")


if __name__ == "__main__":
    main()
