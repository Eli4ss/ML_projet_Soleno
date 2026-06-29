"""Étape 25 — Comparaison des approches : Direct vs Cascade vs PSPP.

Génère :
  - Un tableau synthèse par (target, approach) avec MAE, RMSE, R², MAPE, Spearman
  - Un rapport Markdown avec interprétation
  - Des CSV pour le dashboard Streamlit
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CASCADE_APPROACH_LABELS,
    CASCADE_BLOC_ORDER,
    CASCADE_BLOCS,
    CASCADE_FINAL_TARGETS,
    CASCADE_MECHANICAL_TARGETS,
    CASCADE_THERMAL_TARGETS,
    PHASE4_CASCADE,
    PHASE4_COMPARISON,
    REPORTS,
)
from .utils import df_to_markdown, ensure_dirs

APPROACH_LABELS = CASCADE_APPROACH_LABELS

EVAL_MODE_LABELS = {
    "production": "Production (direct)",
    "lab_assisted": "Lab-assisté (intermédiaires mesurés)",
    "production_oof": "Production OOF (sans fuite)",
}

BLOC_LABELS = {
    "thermal": "Bloc 1 — Thermique",
    "mechanical": "Bloc 2 — Mécanique",
    "final": "Bloc 3 — Performance finale",
    "direct": "Direct",
}


def _load_results() -> pd.DataFrame:
    path = PHASE4_CASCADE / "metrics" / "cascade_all_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Résultats introuvables : {path}\n"
            "Exécutez d'abord run_phase4.py (étape 24)."
        )
    return pd.read_csv(path)


def _best_per_approach(df: pd.DataFrame) -> pd.DataFrame:
    """Meilleur algorithme par (target, approach) selon R²."""
    if df.empty or "r2" not in df.columns:
        return pd.DataFrame()
    return (
        df.sort_values("r2", ascending=False)
        .groupby(["target", "approach"], as_index=False)
        .first()
        .sort_values(["target", "approach"])
    )


def _pivot_metric(best: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Tableau croisé target × approach pour une métrique."""
    if best.empty or metric not in best.columns:
        return pd.DataFrame()
    pivot = best.pivot_table(index="target", columns="approach", values=metric, aggfunc="first")
    pivot.columns.name = None
    return pivot.round(4)


def _winner_per_target(best: pd.DataFrame) -> pd.DataFrame:
    """Pour chaque target, quelle approche a le meilleur R² ?"""
    if best.empty or "r2" not in best.columns:
        return pd.DataFrame()
    idx = best.groupby("target")["r2"].idxmax()
    winner = best.loc[idx, ["target", "approach", "algorithm", "r2", "mae"]].copy()
    winner["approach_label"] = winner["approach"].map(APPROACH_LABELS)
    return winner.reset_index(drop=True)


def _cascade_gain(best: pd.DataFrame) -> pd.DataFrame:
    """Pour les cibles finales (ncls, ucls), gain de R² cascade vs direct."""
    rows = []
    for target in CASCADE_FINAL_TARGETS:
        sub = best[best["target"] == target]
        r2_direct = sub.loc[sub["approach"] == "direct", "r2"].values
        r2_cascade = sub.loc[sub["approach"] == "cascade", "r2"].values
        r2_pspp = sub.loc[sub["approach"] == "pspp", "r2"].values
        rows.append({
            "target": target,
            "r2_direct": float(r2_direct[0]) if len(r2_direct) else np.nan,
            "r2_cascade": float(r2_cascade[0]) if len(r2_cascade) else np.nan,
            "r2_pspp": float(r2_pspp[0]) if len(r2_pspp) else np.nan,
            "gain_cascade_vs_direct": (
                float(r2_cascade[0]) - float(r2_direct[0])
                if len(r2_cascade) and len(r2_direct) else np.nan
            ),
            "gain_pspp_vs_direct": (
                float(r2_pspp[0]) - float(r2_direct[0])
                if len(r2_pspp) and len(r2_direct) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def _build_markdown_report(
    best: pd.DataFrame,
    pivot_r2: pd.DataFrame,
    pivot_mae: pd.DataFrame,
    winner: pd.DataFrame,
    gain: pd.DataFrame,
) -> str:
    lines = [
        "# Rapport de comparaison des approches — Phase 4",
        "",
        "## 1. Résumé",
        "",
        "Trois approches de modélisation sont comparées sur le même dataset :",
        "",
        "| Approche | Logique |",
        "|---|---|",
        "| **Direct** | Matière → cible (baseline) |",
        "| **Cascade** | Blocs 1→2→3 avec intermédiaires **mesurés** (optimiste) |",
        "| **PSPP** | Chaîne avec OOF — **référence pour l'inférence sans labo** |",
        "",
        "## 2. R² par cible et approche",
        "",
    ]
    if not pivot_r2.empty:
        lines.append(df_to_markdown(pivot_r2))
    lines += [
        "",
        "## 3. MAE par cible et approche",
        "",
    ]
    if not pivot_mae.empty:
        lines.append(df_to_markdown(pivot_mae))
    lines += [
        "",
        "## 4. Approche gagnante par cible",
        "",
    ]
    if not winner.empty:
        lines.append(df_to_markdown(winner[["target", "approach_label", "algorithm", "r2", "mae"]], index=False))
    lines += [
        "",
        "## 5. Gain cascade vs direct sur les cibles finales",
        "",
        "Ce tableau montre si l'injection des propriétés intermédiaires améliore",
        "la prédiction de NCLS et UCLS.",
        "",
    ]
    if not gain.empty:
        lines.append(df_to_markdown(gain, index=False))

    lines += [
        "",
        "## 6. Interprétation",
        "",
        "- La **cascade lab-assistée** utilise les vraies valeurs OIT/traction en CV —",
        "  elle surestime souvent la performance en production sans mesures intermédiaires.",
        "- **PSPP** est la référence honnête pour comparer direct vs chaîne prédite.",
        "- Un gain cascade > direct en lab-assisté mais PSPP ≈ direct : ne pas déployer la cascade seule.",
        "- Si **Direct ≈ PSPP**, les features matière sont déjà suffisantes.",
        "",
        "## 7. Recommandation",
        "",
    ]

    if not gain.empty:
        avg_gain_lab = gain["gain_cascade_vs_direct"].mean()
        avg_gain_pspp = gain["gain_pspp_vs_direct"].mean()
        if avg_gain_pspp > 0.05:
            lines.append(
                "> **PSPP recommandé** pour NCLS/UCLS en mode production : "
                f"gain moyen R² PSPP vs direct = +{avg_gain_pspp:.3f}."
            )
        elif avg_gain_lab > 0.05 and avg_gain_pspp <= 0:
            lines.append(
                "> **Cascade utile seulement avec mesures labo intermédiaires** : "
                f"gain lab-assisté +{avg_gain_lab:.3f}, mais PSPP ne bat pas le direct."
            )
        elif avg_gain_lab > 0:
            lines.append(
                "> **Gain marginal** — conserver le modèle direct si les mesures "
                "intermédiaires ne sont pas disponibles en production."
            )
        else:
            lines.append(
                "> **Modèle direct suffisant** — la cascade n'apporte pas d'amélioration significative."
            )
    return "\n".join(lines)


def run_step25() -> dict[str, pd.DataFrame]:
    """Génère la comparaison des approches et les sorties pour le dashboard."""
    ensure_dirs(PHASE4_COMPARISON, REPORTS)

    df = _load_results()
    best = _best_per_approach(df)
    pivot_r2 = _pivot_metric(best, "r2")
    pivot_mae = _pivot_metric(best, "mae")
    winner = _winner_per_target(best)
    gain = _cascade_gain(best)

    # Sauvegardes CSV
    best.to_csv(PHASE4_COMPARISON / "best_per_approach.csv", index=False)
    pivot_r2.to_csv(PHASE4_COMPARISON / "pivot_r2.csv")
    pivot_mae.to_csv(PHASE4_COMPARISON / "pivot_mae.csv")
    winner.to_csv(PHASE4_COMPARISON / "winner_per_target.csv", index=False)
    gain.to_csv(PHASE4_COMPARISON / "cascade_gain.csv", index=False)
    df.to_csv(PHASE4_COMPARISON / "all_results_full.csv", index=False)

    # Rapport Markdown
    report = _build_markdown_report(best, pivot_r2, pivot_mae, winner, gain)
    report_path = REPORTS / "25_cascade_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"Comparison saved to: {PHASE4_COMPARISON}")
    print(f"Report: {report_path}")

    if not gain.empty:
        print("\n=== Cascade gain vs Direct (NCLS / UCLS) ===")
        print(gain[["target", "gain_cascade_vs_direct", "gain_pspp_vs_direct"]].to_string(index=False))

    return {
        "all_results": df,
        "best_per_approach": best,
        "pivot_r2": pivot_r2,
        "pivot_mae": pivot_mae,
        "winner": winner,
        "gain": gain,
    }
