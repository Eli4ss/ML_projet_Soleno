#!/usr/bin/env python
"""Évaluation scientifique unifiée — Phase 5.

Consolide la validation robuste (phase 3) en tables maîtres pour le dashboard.
Réaligne la phase 4 sur les datasets validés si demandé.

Prérequis :
    python run_pipeline.py
    python run_phase2.py
    python run_phase3.py

Usage :
    python run_scientific_evaluation.py
    python run_scientific_evaluation.py --with-cascade
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.config import PHASE2_VALIDATION, PHASE3_ROBUST, PHASE5_UNIFIED
from src.step24_cascade_modeling import run_step24
from src.step25_cascade_comparison import run_step25
from src.step26_unified_evaluation import run_step26


def main() -> None:
    parser = argparse.ArgumentParser(description="Évaluation scientifique unifiée Soleno NAV")
    parser.add_argument(
        "--with-cascade",
        action="store_true",
        help="Ré-exécuter aussi phase 4 (cascade) sur datasets validés",
    )
    args = parser.parse_args()

    if not (PHASE2_VALIDATION / "corrected_target_datasets").exists():
        raise FileNotFoundError("Exécutez run_phase2.py d'abord.")
    if not (PHASE3_ROBUST / "robust_validation_results.csv").exists():
        raise FileNotFoundError("Exécutez run_phase3.py d'abord.")

    print("=" * 60)
    print("PHASE 5 — Évaluation scientifique unifiée")
    print("=" * 60)

    if args.with_cascade:
        print("\n=== Step 24 : Cascade (datasets validés) ===")
        run_step24()
        print("\n=== Step 25 : Comparaison approches ===")
        run_step25()

    print("\n=== Step 26 : Tables maîtres dashboard ===")
    outputs = run_step26()

    print("\n" + "=" * 60)
    print("Phase 5 terminée.")
    print(f"  Livrables -> {PHASE5_UNIFIED}")
    if not outputs["recommended"].empty:
        print("\n=== Modèle recommandé par cible ===")
        cols = ["target", "model_version", "deployment_status", "r2_group_supplier"]
        print(outputs["recommended"][[c for c in cols if c in outputs["recommended"].columns]].to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
