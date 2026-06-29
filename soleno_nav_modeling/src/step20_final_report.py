"""Étape 20 — Rapport final d'amélioration."""
from __future__ import annotations

import pandas as pd

from .config import (
    PHASE2_COMPARISON,
    PHASE2_FINAL_REPORT,
    PHASE2_MODELING,
    PHASE2_REGISTRY,
    PHASE2_VALIDATION,
    REPORTS,
)


def run_step20() -> None:
    parts = ["# Rapport final d'amélioration — Phase 2 Soleno NAV", ""]

    val_report = PHASE2_VALIDATION / "target_validation_report.md"
    if val_report.exists():
        parts.append("## 1. Problème initial\n")
        parts.append(
            "Les premières modélisations ont révélé des distributions aberrantes "
            "(OIT négatif/extrême, IZOD >600k, cell_class traitée en continu, temp_c comme cible)."
        )

    dist = PHASE2_VALIDATION / "target_distribution_summary.csv"
    if dist.exists():
        d = pd.read_csv(dist)
        parts.append("\n## 2. Anomalies par cible\n")
        parts.append(d[["target_name", "available_count", "min", "max", "n_negative", "n_iqr_outliers"]].to_string(index=False))

    rules = PHASE2_VALIDATION / "target_validity_rules.csv"
    if rules.exists():
        parts.append("\n## 3. Règles de validation\n")
        parts.append(f"Voir `{rules}` — seuils à valider par Soleno.")

    decision = PHASE2_VALIDATION / "target_modeling_decision_table.csv"
    if decision.exists():
        parts.append("\n## 5. Cibles exploitables\n")
        parts.append(pd.read_csv(decision).to_string(index=False))

    comp = PHASE2_COMPARISON / "before_after_summary_by_target.csv"
    if comp.exists():
        parts.append("\n## 6. Comparaison avant/après\n")
        parts.append(pd.read_csv(comp).to_string(index=False))

    best = PHASE2_MODELING / "ml_best_models.csv"
    if best.exists():
        parts.append("\n## 7. Meilleures cibles (MAE minimal phase 2)\n")
        b = pd.read_csv(best)
        if "mae" in b.columns:
            parts.append(b.nsmallest(10, "mae")[["target", "model_version", "dataset_version", "algorithm", "mae", "r2"]].to_string(index=False))

    reg = PHASE2_REGISTRY / "updated_best_model_registry.csv"
    if reg.exists():
        parts.append("\n## 11. Registre mis à jour\n")
        parts.append(f"`{reg}`")

    parts.extend([
        "\n## 12. Prochaine phase (non exécutée)\n",
        "- GroupKFold fournisseur / grade / origine\n",
        "- Validation temporelle\n",
        "- Généralisation industrielle\n",
    ])

    text = "\n".join(parts)
    PHASE2_FINAL_REPORT.parent.mkdir(parents=True, exist_ok=True)
    PHASE2_FINAL_REPORT.write_text(text, encoding="utf-8")
    (REPORTS / "20_improvement_final_report.md").write_text(text, encoding="utf-8")
