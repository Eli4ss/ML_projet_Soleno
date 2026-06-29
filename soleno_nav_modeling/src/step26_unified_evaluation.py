"""Étape 26 — Évaluation scientifique unifiée (tables maîtres pour dashboard)."""
from __future__ import annotations

import pandas as pd

from .config import PHASE3_ROBUST, PHASE5_UNIFIED, REPORTS, TARGET_COLUMNS_MODELING
from .evaluation_standard import (
    build_target_deployment_table,
    metrics_long_from_robust,
    pick_recommended_model_per_target,
    TARGET_TIERS,
    VALIDATION_SCHEME_LABELS,
)
from .utils import df_to_markdown, ensure_dirs


def _load_robust_results() -> pd.DataFrame:
    path = PHASE3_ROBUST / "robust_validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Résultats phase 3 introuvables : {path}\n"
            "Exécutez run_phase3.py d'abord."
        )
    return pd.read_csv(path)


def _build_report(
    deployment: pd.DataFrame,
    recommended: pd.DataFrame,
    metrics_long: pd.DataFrame,
) -> str:
    lines = [
        "# Rapport — Évaluation scientifique unifiée (Phase 5)",
        "",
        "## Principe",
        "",
        "Ce rapport consolide la **validation robuste (phase 3)** comme référence décisionnelle.",
        "Le KFold aléatoire est un indicateur interne ; la généralisation fournisseur/grade/site prime.",
        "",
        "## Tiers cibles (polymères recyclés)",
        "",
        "| Cible | Tier | Lecture métier |",
        "|---|---|---|",
    ]
    for target in TARGET_COLUMNS_MODELING:
        if target == "cell_class":
            continue
        info = TARGET_TIERS.get(target, {})
        lines.append(
            f"| {target} | {info.get('tier', '?')} | {info.get('polymer_rationale', '')} |"
        )

    lines += [
        "",
        "## Statut de déploiement recommandé (modèle A vs B)",
        "",
    ]
    if not recommended.empty:
        cols = [
            "target", "model_version", "deployment_status", "deployment_label",
            "r2_random", "r2_group_supplier", "mae_degradation_ratio", "status_reason",
        ]
        show = recommended[[c for c in cols if c in recommended.columns]]
        lines.append(df_to_markdown(show, index=False))

    lines += [
        "",
        "## Schémas de validation",
        "",
    ]
    for k, v in VALIDATION_SCHEME_LABELS.items():
        lines.append(f"- `{k}` : {v}")

    if not metrics_long.empty:
        lines += [
            "",
            "## Synthèse R² (référence)",
            "",
        ]
        pivot = (
            metrics_long.pivot_table(
                index="target",
                columns="validation_scheme",
                values="r2_mean",
                aggfunc="first",
            )
            .round(3)
        )
        lines.append(df_to_markdown(pivot))

    lines += [
        "",
        "## Usage dashboard",
        "",
        f"- `target_deployment_status.csv` — statut par cible",
        f"- `recommended_model_per_target.csv` — modèle recommandé",
        f"- `master_metrics_long.csv` — métriques long format pour graphiques",
    ]
    return "\n".join(lines)


def run_step26() -> dict[str, pd.DataFrame]:
    """Génère les tables maîtres consommées par le dashboard."""
    ensure_dirs(PHASE5_UNIFIED, REPORTS)

    robust = _load_robust_results()
    deployment = build_target_deployment_table(robust)
    recommended = pick_recommended_model_per_target(deployment)
    metrics_long = metrics_long_from_robust(robust)

    deployment.to_csv(PHASE5_UNIFIED / "target_deployment_status.csv", index=False)
    recommended.to_csv(PHASE5_UNIFIED / "recommended_model_per_target.csv", index=False)
    metrics_long.to_csv(PHASE5_UNIFIED / "master_metrics_long.csv", index=False)
    robust.to_csv(PHASE5_UNIFIED / "robust_validation_snapshot.csv", index=False)

    report = _build_report(deployment, recommended, metrics_long)
    report_path = REPORTS / "26_unified_evaluation_report.md"
    report_path.write_text(report, encoding="utf-8")
    (PHASE5_UNIFIED / "unified_evaluation_report.md").write_text(report, encoding="utf-8")

    print(f"Deployment status: {PHASE5_UNIFIED / 'target_deployment_status.csv'}")
    print(f"Master metrics:    {PHASE5_UNIFIED / 'master_metrics_long.csv'}")
    print(f"Report:            {report_path}")

    return {
        "deployment": deployment,
        "recommended": recommended,
        "metrics_long": metrics_long,
        "robust": robust,
    }
