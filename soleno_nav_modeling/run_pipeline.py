#!/usr/bin/env python
"""Exécute le pipeline complet Soleno NAV (étapes 1-12)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.step01_audit import run_step01
from src.step02_features import run_step02
from src.step03_cleaning import run_step03
from src.step04_engineering import run_step04
from src.step05_targets import run_step05
from src.step06_categorization import run_step06
from src.step07_baseline import run_step07
from src.step08_deep_learning import run_step08
from src.step09_evaluation import run_step09
from src.step10_explainability import run_step10
from src.step11_uncertainty import run_step11
from src.step12_selection import run_step12


def main() -> None:
    print("=== Étape 1 : Audit ===")
    df = run_step01()
    print("=== Étape 2 : Features ===")
    run_step02(df)
    print("=== Étape 3 : Nettoyage ===")
    df, _ = run_step03(df)
    print("=== Étape 4 : Feature engineering ===")
    df = run_step04(df)
    print("=== Étape 5 : Cibles ===")
    run_step05(df)
    print("=== Étape 6 : Catégorisation ===")
    _, df_cls, df_rank = run_step06(df)
    print("=== Étape 7 : Baseline ML ===")
    reg, cls, best = run_step07(df, df_cls)
    print("=== Étape 8 : Deep Learning ===")
    dl = run_step08(df, df_cls)
    print("=== Étape 9 : Évaluation ===")
    run_step09(reg, dl, best)
    print("=== Étape 10 : Explainability ===")
    run_step10(df)
    print("=== Étape 11 : Incertitude ===")
    run_step11(df)
    print("=== Étape 12 : Sélection ===")
    run_step12(df, best)
    print("Pipeline terminé. Lancez Streamlit : streamlit run app/app.py")


if __name__ == "__main__":
    main()
