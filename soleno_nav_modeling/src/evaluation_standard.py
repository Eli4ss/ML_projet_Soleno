"""Protocole d'évaluation scientifique unifié — référence pour pipeline et dashboard."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import PHASE2_VALIDATION, TARGET_COLUMNS_MODELING

# Priorité des schémas de validation (1 = référence décisionnelle)
VALIDATION_SCHEME_PRIORITY: dict[str, int] = {
    "group_supplier": 1,
    "group_grade": 2,
    "group_origin": 3,
    "temporal": 4,
    "kfold_random": 5,
}

VALIDATION_SCHEME_LABELS: dict[str, str] = {
    "kfold_random": "KFold aléatoire (référence interne)",
    "group_supplier": "GroupKFold fournisseur (généralisation)",
    "group_grade": "GroupKFold grade",
    "group_origin": "GroupKFold site",
    "temporal": "Validation temporelle",
}

# Tiers métier polymères recyclés
TARGET_TIERS: dict[str, dict[str, str]] = {
    "oit_min": {
        "tier": "A",
        "label": "Exploitable avec prudence",
        "polymer_rationale": "Proxy oxydation/stabilisation — signal matière partiellement capturé.",
    },
    "flexion": {
        "tier": "A",
        "label": "Exploitable avec prudence",
        "polymer_rationale": "Rigidité — liée morphologie/charges, mais sensible au grade.",
    },
    "pct_elongation": {
        "tier": "A",
        "label": "Exploitable avec prudence",
        "polymer_rationale": "Ductilité — utile en tri préliminaire.",
    },
    "traction": {
        "tier": "B",
        "label": "Usage limité",
        "polymer_rationale": "Résistance — fort effet process/fournisseur non mesuré.",
    },
    "ncls": {
        "tier": "B",
        "label": "Usage limité",
        "polymer_rationale": "Durabilité — dépend fortement du contexte test et du grade.",
    },
    "izod": {
        "tier": "C",
        "label": "Non déployable sans revalidation",
        "polymer_rationale": "Choc — variabilité élevée, généralisation fournisseur faible.",
    },
    "ucls": {
        "tier": "C",
        "label": "Données insuffisantes",
        "polymer_rationale": "Effectif trop faible pour modélisation fiable.",
    },
    "cell_class": {
        "tier": "B",
        "label": "Classification uniquement",
        "polymer_rationale": "Code PE — pas de régression continue.",
    },
}

DEPLOYMENT_STATUS_LABELS: dict[str, str] = {
    "deploy": "Déployable (domaine connu)",
    "caution": "Prudence (généralisation limitée)",
    "lab_only": "Labo requis (signal faible)",
    "blocked": "Bloqué (données ou métriques insuffisantes)",
}

DEFAULT_DATASET_VERSION: dict[str, str] = {
    "oit_min": "log",
    "ncls": "log",
    "ucls": "log",
    "izod": "log",
    "traction": "log",
    "pct_elongation": "log",
    "flexion": "log",
    "cell_class": "valid",
}

MIN_SAMPLES_DEPLOY = 50
MIN_SAMPLES_TIER_C = 120
MAE_DEGRADATION_CAUTION = 1.35
MAE_DEGRADATION_LAB_ONLY = 2.0
R2_GROUP_DEPLOY = 0.15
R2_GROUP_CAUTION = 0.0
R2_RANDOM_LAB_ONLY = 0.25


def get_dataset_version(target: str) -> str:
    """Version dataset canonique pour une cible (phase 2 si disponible)."""
    decision_path = PHASE2_VALIDATION / "target_modeling_decision_table.csv"
    if decision_path.exists():
        decision = pd.read_csv(decision_path)
        tcol = "target_name" if "target_name" in decision.columns else "target"
        if tcol in decision.columns:
            sub = decision[decision[tcol].astype(str) == target]
            if not sub.empty:
                if sub.iloc[0].get("log_transform_recommended") is True:
                    return "log"
                return "valid"
    return DEFAULT_DATASET_VERSION.get(target, "valid")


def normalize_validation_scheme(scheme: str) -> str:
    s = str(scheme or "").strip()
    if s.startswith("group_"):
        return s
    if "random" in s:
        return "kfold_random"
    if s == "temporal":
        return "temporal"
    return s


def _metric_row(
    results: pd.DataFrame,
    target: str,
    model_version: str,
    scheme: str,
) -> dict[str, Any] | None:
    if results.empty:
        return None
    sub = results[
        (results["target"].astype(str) == target)
        & (results["model_version"].astype(str) == model_version)
        & (results["validation_scheme"].astype(str) == scheme)
    ]
    if sub.empty:
        return None
    row = sub.iloc[0]
    if pd.notna(row.get("error")) and str(row.get("error", "")).strip():
        return None
    return row.to_dict()


def compute_deployment_status(
    target: str,
    model_version: str,
    results: pd.DataFrame,
) -> dict[str, Any]:
    """Détermine le statut opérationnel à partir des métriques phase 3."""
    tier_info = TARGET_TIERS.get(target, {"tier": "B", "label": "À évaluer", "polymer_rationale": ""})
    tier = tier_info["tier"]

    rand = _metric_row(results, target, model_version, "kfold_random")
    group_sup = _metric_row(results, target, model_version, "group_supplier")

    n_samples = int(rand.get("n_samples", 0)) if rand else 0
    r2_random = float(rand.get("r2_mean", np.nan)) if rand else np.nan
    mae_random = float(rand.get("mae_mean", np.nan)) if rand else np.nan
    r2_group = float(group_sup.get("r2_mean", np.nan)) if group_sup else np.nan
    mae_group = float(group_sup.get("mae_mean", np.nan)) if group_sup else np.nan

    mae_ratio = np.nan
    if pd.notna(mae_random) and pd.notna(mae_group) and mae_random > 0:
        mae_ratio = float(mae_group / mae_random)

    status = "caution"
    reasons: list[str] = []

    if tier == "C" or n_samples < MIN_SAMPLES_DEPLOY:
        status = "blocked"
        if tier == "C":
            reasons.append(f"Tier {tier} — {tier_info['label']}")
        if n_samples < MIN_SAMPLES_DEPLOY:
            reasons.append(f"Effectif insuffisant ({n_samples} < {MIN_SAMPLES_DEPLOY})")
    elif pd.isna(r2_random) or r2_random < R2_RANDOM_LAB_ONLY:
        status = "lab_only"
        reasons.append(f"R² random faible ({r2_random:.3f})" if pd.notna(r2_random) else "Pas de CV random")
    elif pd.notna(r2_group) and r2_group < R2_GROUP_CAUTION:
        status = "lab_only"
        reasons.append(f"R² group fournisseur négatif ({r2_group:.3f})")
    elif pd.notna(mae_ratio) and mae_ratio >= MAE_DEGRADATION_LAB_ONLY:
        status = "lab_only"
        reasons.append(f"Dégradation MAE fournisseur ×{mae_ratio:.1f}")
    elif (
        pd.notna(r2_group)
        and r2_group >= R2_GROUP_DEPLOY
        and pd.notna(mae_ratio)
        and mae_ratio < MAE_DEGRADATION_CAUTION
        and tier == "A"
    ):
        status = "deploy"
        reasons.append("Généralisation fournisseur acceptable")
    elif pd.notna(mae_ratio) and mae_ratio >= MAE_DEGRADATION_CAUTION:
        status = "caution"
        reasons.append(f"Dégradation MAE fournisseur ×{mae_ratio:.1f}")
    else:
        status = "caution"
        reasons.append("Performance acceptable en interne, généralisation à confirmer")

    recommended_model = "A" if status in ("deploy", "caution") and model_version == "A" else model_version
    if status == "lab_only" and model_version == "B":
        recommended_model = "A"

    return {
        "target": target,
        "model_version": model_version,
        "tier": tier,
        "tier_label": tier_info["label"],
        "polymer_rationale": tier_info.get("polymer_rationale", ""),
        "dataset_version": get_dataset_version(target),
        "deployment_status": status,
        "deployment_label": DEPLOYMENT_STATUS_LABELS[status],
        "n_samples": n_samples,
        "r2_random": r2_random,
        "r2_group_supplier": r2_group,
        "mae_random": mae_random,
        "mae_group_supplier": mae_group,
        "mae_degradation_ratio": mae_ratio,
        "recommended_model": recommended_model,
        "status_reason": "; ".join(reasons),
    }


def build_target_deployment_table(results: pd.DataFrame) -> pd.DataFrame:
    """Table de déploiement par cible et version modèle."""
    rows: list[dict[str, Any]] = []
    targets = [t for t in TARGET_COLUMNS_MODELING if t != "cell_class"]
    for target in targets:
        for version in ("A", "B"):
            rows.append(compute_deployment_status(target, version, results))
    return pd.DataFrame(rows)


def pick_recommended_model_per_target(deployment: pd.DataFrame) -> pd.DataFrame:
    """Une ligne par cible : meilleur compromis déploiement (priorité statut puis R² group)."""
    if deployment.empty:
        return pd.DataFrame()

    status_rank = {"deploy": 0, "caution": 1, "lab_only": 2, "blocked": 3}
    df = deployment.copy()
    df["_status_rank"] = df["deployment_status"].map(status_rank).fillna(9)
    df["_r2g"] = pd.to_numeric(df["r2_group_supplier"], errors="coerce").fillna(-999)
    df = df.sort_values(["target", "_status_rank", "_r2g"], ascending=[True, True, False])
    best = df.groupby("target", as_index=False).first()
    return best.drop(columns=["_status_rank", "_r2g"], errors="ignore")


def metrics_long_from_robust(results: pd.DataFrame) -> pd.DataFrame:
    """Format long pour visualisations dashboard."""
    if results.empty:
        return pd.DataFrame()
    rows = []
    for _, row in results.iterrows():
        if pd.notna(row.get("error")) and str(row.get("error", "")).strip():
            continue
        scheme = normalize_validation_scheme(str(row.get("validation_scheme", "")))
        rows.append({
            "target": row.get("target"),
            "model_version": row.get("model_version"),
            "validation_scheme": scheme,
            "validation_label": VALIDATION_SCHEME_LABELS.get(scheme, scheme),
            "validation_priority": VALIDATION_SCHEME_PRIORITY.get(scheme, 99),
            "mae_mean": row.get("mae_mean"),
            "mae_std": row.get("mae_std"),
            "r2_mean": row.get("r2_mean"),
            "r2_std": row.get("r2_std"),
            "n_samples": row.get("n_samples"),
            "n_groups": row.get("n_groups"),
            "metric_tier": 1 if scheme.startswith("group") or scheme == "temporal" else 2,
        })
    return pd.DataFrame(rows)
