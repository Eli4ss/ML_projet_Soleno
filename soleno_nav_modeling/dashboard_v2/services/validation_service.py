"""Agrégation des métriques de validation — phase 5 prioritaire."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import PATHS
from services.maturity import explain_negative_r2


@st.cache_data
def load_master_metrics() -> pd.DataFrame:
    path = PATHS.get("master_metrics_long")
    if path and Path(path).exists():
        return pd.read_csv(path)
    path = PATHS.get("robust_validation")
    if path and Path(path).exists():
        legacy = pd.read_csv(path)
        if "validation_scheme" in legacy.columns:
            legacy = legacy.rename(columns={
                "mae_mean": "mae_mean",
                "r2_mean": "r2_mean",
            })
        return legacy
    return pd.DataFrame()


@st.cache_data
def load_deployment_full() -> pd.DataFrame:
    path = PATHS.get("deployment_status")
    if path and Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@st.cache_data
def load_ab_comparison() -> pd.DataFrame:
    path = PATHS.get("ab_robust")
    if path and Path(path).exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def validation_summary_table() -> pd.DataFrame:
    """Une ligne par cible (modèle recommandé)."""
    from config import PATHS as MVP_PATHS
    from services.maturity import build_target_maturity_table

    rec_path = MVP_PATHS.get("recommended_model")
    if not rec_path or not Path(rec_path).exists():
        return pd.DataFrame()

    rec = pd.read_csv(rec_path)
    mat = build_target_maturity_table().set_index("target")

    rows = []
    for _, r in rec.iterrows():
        target = str(r["target"])
        m = mat.loc[target].to_dict() if target in mat.index else {}
        rows.append({
            "Cible": m.get("label", target),
            "target": target,
            "Échantillons": r.get("n_samples"),
            "Modèle recommandé": r.get("recommended_model"),
            "Niveau maturité": m.get("level_label", ""),
            "R² aléatoire": r.get("r2_random"),
            "R² fournisseur": r.get("r2_group_supplier"),
            "MAE aléatoire": r.get("mae_random"),
            "MAE fournisseur": r.get("mae_group_supplier"),
            "Ratio dégradation MAE": r.get("mae_degradation_ratio"),
            "Statut": r.get("deployment_label"),
            "Commentaire": r.get("status_reason") or r.get("polymer_rationale"),
        })
    return pd.DataFrame(rows)


def metrics_for_target(target: str, model_version: str | None = None) -> pd.DataFrame:
    df = load_master_metrics()
    if df.empty:
        return df
    sub = df[df["target"].astype(str) == target]
    if model_version:
        sub = sub[sub["model_version"].astype(str) == model_version]
    if "validation_priority" in sub.columns:
        sub = sub.sort_values("validation_priority")
    return sub


def format_metric_cell(value, metric: str = "r2") -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    v = float(value)
    text = f"{v:.3f}"
    if metric == "r2" and v < 0:
        text += " ⚠"
    return text


def degradation_interpretation(ratio: float | None) -> str:
    if ratio is None or (isinstance(ratio, float) and pd.isna(ratio)):
        return "Non calculé"
    r = float(ratio)
    if r >= 2.0:
        return "Forte dégradation — généralisation fragile"
    if r >= 1.35:
        return "Dégradation modérée — prudence recommandée"
    return "Dégradation limitée sur ce protocole"
