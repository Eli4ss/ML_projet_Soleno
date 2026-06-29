"""Chargement des données et artefacts Soleno NAV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import DATA, NAV_ROOT, OUTPUTS, PATHS, TARGETS
from src.utils import safe_read_csv


@st.cache_data(show_spinner="Chargement du dataset NAV…")
def load_feature_engineered() -> pd.DataFrame | None:
    return safe_read_csv(PATHS["feature_engineered"])


@st.cache_data
def load_cleaned() -> pd.DataFrame | None:
    return safe_read_csv(PATHS["cleaned"])


@st.cache_data
def load_feature_decision() -> pd.DataFrame | None:
    df = safe_read_csv(PATHS["feature_decision"])
    if df is None:
        p = PATHS["feature_decision"].with_suffix(".xlsx")
        if p.exists():
            try:
                return pd.read_excel(p)
            except Exception:
                pass
    return df


@st.cache_data
def load_registry() -> pd.DataFrame:
    df = safe_read_csv(PATHS["registry_phase2"])
    if df is not None and not df.empty:
        return df
    df = safe_read_csv(PATHS["registry_phase1"])
    return df if df is not None else pd.DataFrame()


@st.cache_data
def load_unified_evaluation() -> dict[str, pd.DataFrame]:
    """Tables maîtres phase 5 — référence dashboard fiabilité."""
    keys = ["deployment_status", "recommended_model", "master_metrics_long", "robust_predictions"]
    out: dict[str, pd.DataFrame] = {}
    for k in keys:
        df = safe_read_csv(PATHS.get(k))
        if df is not None and not df.empty:
            out[k] = df
    return out


@st.cache_data
def load_metrics_bundle() -> dict[str, pd.DataFrame]:
    keys = [
        "baseline_reg", "baseline_cls", "dl_reg", "ml_after_cleaning",
        "before_after", "robust_validation", "ab_robust",
    ]
    out = {}
    for k in keys:
        df = safe_read_csv(PATHS[k])
        if df is not None:
            out[k] = df
    return out


@st.cache_data
def load_target_valid(target_name: str) -> pd.DataFrame | None:
    """Validated target dataset from Phase 2 Step 14.

    Returns only rows where *target_name* is non-null and physically valid
    (invalid rows removed by step14_target_validation). Same 70 columns as
    04_NAV_feature_engineered.csv.
    """
    path = (
        OUTPUTS
        / "14_target_validation"
        / "corrected_target_datasets"
        / f"target_valid_{target_name}.csv"
    )
    df = safe_read_csv(path)
    if df is None:
        return None
    if target_name in df.columns:
        df = df[df[target_name].notna()].copy()
    return df if not df.empty else None


@st.cache_data
def load_target_outlier_summary() -> pd.DataFrame:
    df = safe_read_csv(PATHS["target_outlier_summary"])
    return df if df is not None else pd.DataFrame()


@st.cache_data
def load_target_distribution_summary() -> pd.DataFrame:
    df = safe_read_csv(PATHS["target_distribution_summary"])
    return df if df is not None else pd.DataFrame()


@st.cache_data
def load_target_quality_flags_summary() -> pd.DataFrame:
    df = safe_read_csv(PATHS["target_quality_flags_summary"])
    return df if df is not None else pd.DataFrame()


@st.cache_data
def load_importance() -> pd.DataFrame:
    df = safe_read_csv(PATHS["importance_phase2"])
    if df is not None and not df.empty:
        return df
    df = safe_read_csv(PATHS["importance_phase1"])
    return df if df is not None else pd.DataFrame()


@st.cache_data
def discover_saved_models() -> list[dict]:
    """Liste les modèles .joblib dans outputs/models/."""
    found = []
    for folder_key in ("models_saved_phase2", "models_saved_phase1"):
        folder = PATHS[folder_key]
        if not folder.exists():
            continue
        for p in folder.glob("*.joblib"):
            name = p.stem
            parts = name.replace("best_", "").split("_")
            found.append({
                "path": str(p),
                "filename": p.name,
                "folder": folder_key,
            })
    reg = load_registry()
    if not reg.empty:
        for _, row in reg.iterrows():
            mp = row.get("model_path", "")
            if mp and Path(str(mp)).exists():
                found.append({
                    "path": str(mp),
                    "target": row.get("target", ""),
                    "version": row.get("model_A_or_B", row.get("model_version", "")),
                    "registry": True,
                })
    return found


def dataset_summary(df: pd.DataFrame | None) -> dict:
    if df is None or df.empty:
        return {"rows": 0, "cols": 0, "targets_labeled": {}}
    labeled = {}
    for t in TARGETS:
        if t in df.columns:
            labeled[t] = int(df[t].notna().sum())
    return {"rows": len(df), "cols": len(df.columns), "targets_labeled": labeled}
