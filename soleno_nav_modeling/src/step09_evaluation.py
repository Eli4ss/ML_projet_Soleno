"""Étape 9 — Évaluation standard et comparaisons."""
from __future__ import annotations

import pandas as pd

from .config import FIGURES, METRICS, REPORTS, TABLES
from .utils import ensure_dirs


def run_step09(reg_baseline: pd.DataFrame, reg_dl: pd.DataFrame, best_baseline: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(METRICS, REPORTS, TABLES, FIGURES / "09_prediction_plots")
    frames = []
    if not reg_baseline.empty:
        rb = reg_baseline.copy()
        rb["pipeline"] = "baseline_ml"
        frames.append(rb)
    if not reg_dl.empty:
        rd = reg_dl.copy()
        rd["pipeline"] = "deep_learning"
        frames.append(rd)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    combined.to_csv(METRICS / "09_standard_evaluation_results.csv", index=False)

    if not best_baseline.empty:
        best_baseline.to_excel(TABLES / "09_model_comparison_by_target.xlsx", index=False)

    try:
        import matplotlib.pyplot as plt
        if not combined.empty and "mae" in combined.columns:
            sub = combined.dropna(subset=["mae"]).groupby("target")["mae"].min()
            fig, ax = plt.subplots(figsize=(8, 5))
            sub.plot(kind="bar", ax=ax)
            ax.set_ylabel("MAE (min par cible)")
            ax.set_title("Meilleur MAE baseline/DL par cible")
            fig.tight_layout()
            fig.savefig(FIGURES / "09_prediction_plots" / "mae_by_target.png", dpi=100)
            plt.close(fig)
    except Exception:
        pass

    lines = [
        "# Rapport d'évaluation standard",
        "",
        "Comparaison modèles A vs B et ML vs DL sur les métriques hold-out.",
        "",
        f"Résultats agrégés : `{METRICS / '09_standard_evaluation_results.csv'}`",
    ]
    (REPORTS / "09_standard_evaluation_report.md").write_text("\n".join(lines), encoding="utf-8")
    return combined
