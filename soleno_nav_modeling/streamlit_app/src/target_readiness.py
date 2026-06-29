"""Statut Model Readiness par cible — alimenté par l'évaluation unifiée phase 5."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from config import (
    DEPLOYMENT_STATUS_LABELS,
    PATHS,
    ROBUST_VALIDATION_CSV,
    TARGET_DECISION_TABLE_CSV,
    UPDATED_REGISTRY_CSV,
)
from src.utils import safe_read_csv

_STATUS_TO_CONFIDENCE = {
    "deploy": "élevée",
    "caution": "moyenne",
    "lab_only": "faible",
    "blocked": "faible",
}

_STATUS_TO_LABEL = {
    "deploy": "Déployable (domaine connu)",
    "caution": "Prudent — généralisation limitée",
    "lab_only": "Fragile — labo requis",
    "blocked": "Bloqué — données insuffisantes",
}


@lru_cache(maxsize=1)
def load_target_readiness() -> pd.DataFrame:
    """Table de readiness consommée par le dashboard."""
    deployment = safe_read_csv(PATHS.get("deployment_status"))
    recommended = safe_read_csv(PATHS.get("recommended_model"))

    if deployment is not None and not deployment.empty and recommended is not None and not recommended.empty:
        return _from_unified_evaluation(recommended)

    return _from_legacy_sources()


def _from_unified_evaluation(recommended: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in recommended.iterrows():
        target = str(row.get("target", ""))
        status_key = str(row.get("deployment_status", "caution"))
        rows.append({
            "target": target,
            "status": _STATUS_TO_LABEL.get(status_key, row.get("deployment_label", "À évaluer")),
            "usage_recommended": _usage_for_target(target, row),
            "confidence_prior": _STATUS_TO_CONFIDENCE.get(status_key, "moyenne"),
            "deployment_status": status_key,
            "deployment_label": row.get("deployment_label", DEPLOYMENT_STATUS_LABELS.get(status_key, "")),
            "tier": row.get("tier", ""),
            "model_version": row.get("model_version", "A"),
            "r2_random": row.get("r2_random"),
            "r2_group_supplier": row.get("r2_group_supplier"),
            "mae_degradation_ratio": row.get("mae_degradation_ratio"),
            "notes": row.get("status_reason", ""),
        })
    return pd.DataFrame(rows)


def _usage_for_target(target: str, row: pd.Series) -> str:
    status = str(row.get("deployment_status", ""))
    if status == "deploy":
        return "Tri préliminaire / priorisation essais"
    if status == "caution":
        return "Screening avec vérification labo recommandée"
    if status == "lab_only":
        return "Ne pas utiliser seul — essai labo obligatoire"
    if status == "blocked":
        return "Non prioritaire — collecter plus de données"
    return str(row.get("polymer_rationale", "prudence"))[:120]


def _from_legacy_sources() -> pd.DataFrame:
    """Fallback si phase 5 non exécutée."""
    rows = []
    decision = pd.DataFrame()
    if TARGET_DECISION_TABLE_CSV.exists():
        decision = pd.read_csv(TARGET_DECISION_TABLE_CSV)

    robust = pd.DataFrame()
    if ROBUST_VALIDATION_CSV.exists():
        robust = pd.read_csv(ROBUST_VALIDATION_CSV)

    targets: set[str] = set()
    tcol = "target_name" if "target" not in decision.columns and "target_name" in decision.columns else "target"
    if not decision.empty and tcol in decision.columns:
        targets.update(decision[tcol].astype(str))
    if not robust.empty and "target" in robust.columns:
        targets.update(robust["target"].astype(str))

    default_status = {
        "oit_min": ("Prudent", "prédiction avec vérification", "moyenne"),
        "ncls": ("Fragile en généralisation", "screening uniquement", "faible"),
        "izod": ("Fragile en généralisation", "non déployable sans revalidation", "faible"),
        "traction": ("Moyen", "prédiction prudente", "moyenne"),
        "pct_elongation": ("Prudent", "tri préliminaire", "moyenne"),
        "flexion": ("Prudent", "classification / ranking", "moyenne"),
        "ucls": ("Bloqué", "données insuffisantes", "faible"),
    }

    if not targets:
        targets = set(default_status.keys())

    for t in sorted(targets):
        status, usage, prior = default_status.get(t, ("À évaluer", "prudence", "moyenne"))
        notes: list[str] = []

        if not robust.empty and "target" in robust.columns:
            rsub = robust[robust["target"].astype(str) == t]
            if not rsub.empty and "mae_mean" in rsub.columns:
                rand = rsub[rsub["validation_scheme"].astype(str).str.contains("random", case=False, na=False)]
                group = rsub[rsub["validation_scheme"].astype(str).str.startswith("group", na=False)]
                mae_k = pd.to_numeric(rand["mae_mean"], errors="coerce").min() if not rand.empty else np.nan
                mae_g = pd.to_numeric(group["mae_mean"], errors="coerce").max() if not group.empty else np.nan
                if pd.notna(mae_g) and pd.notna(mae_k) and mae_k > 0:
                    ratio = float(mae_g) / float(mae_k)
                    if ratio > 2.0:
                        status = "Fragile en généralisation"
                        prior = "faible"
                        notes.append(f"MAE groupé / aléatoire ≈ {ratio:.1f}×")
                    elif ratio > 1.3:
                        status = "Prudent"
                        prior = "moyenne"

        rows.append({
            "target": t,
            "status": status,
            "usage_recommended": usage,
            "confidence_prior": prior,
            "deployment_status": "caution",
            "notes": "; ".join(notes) if notes else "Exécutez run_scientific_evaluation.py pour statut officiel",
        })

    return pd.DataFrame(rows)


def readiness_for_target(target: str) -> dict:
    df = load_target_readiness()
    sub = df[df["target"].astype(str) == str(target)]
    if sub.empty:
        return {
            "confidence_prior": "moyenne",
            "status": "À évaluer",
            "usage_recommended": "prudence",
            "deployment_status": "caution",
        }
    return sub.iloc[0].to_dict()
