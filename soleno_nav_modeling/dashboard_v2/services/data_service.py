"""Couche données pour dashboard_v2 — wrappe le MVP data_loader."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from config import FILTER_COLUMNS, PATHS, TARGETS
from src.data_loader import (
    dataset_summary,
    load_cleaned,
    load_feature_decision,
    load_feature_engineered,
    load_target_distribution_summary,
    load_target_outlier_summary,
    load_target_quality_flags_summary,
    load_target_valid,
    load_unified_evaluation,
)
from src.utils import file_status


@st.cache_data
def nav_dataset_stats() -> dict:
    df = load_feature_engineered()
    summary = dataset_summary(df)
    if df is None or df.empty:
        return summary

    stats = {
        **summary,
        "n_rows": summary.get("rows", len(df)),
        "n_suppliers": df["supplier_code"].nunique() if "supplier_code" in df.columns else 0,
        "n_grades": df["description_fr"].nunique() if "description_fr" in df.columns else 0,
        "year_min": None,
        "year_max": None,
    }
    if "reception_year" in df.columns:
        yrs = pd.to_numeric(df["reception_year"], errors="coerce").dropna()
        if not yrs.empty:
            stats["year_min"] = int(yrs.min())
            stats["year_max"] = int(yrs.max())
    return stats


def target_coverage_table() -> pd.DataFrame:
    from config import MVP_TARGETS

    rows = []
    fe = load_feature_engineered()
    n_total = len(fe) if fe is not None else 0
    dist_sum = load_target_distribution_summary()
    dist_idx = (
        dist_sum.set_index("target_name")
        if dist_sum is not None and not dist_sum.empty and "target_name" in dist_sum.columns
        else None
    )

    from services.maturity import derive_target_maturity
    from services.value_semantics import column_value_breakdown, verify_target_counts

    for t in MVP_TARGETS:
        absent = artificial = true_zero = measured = raw_non_null = None
        check = None
        if fe is not None and t in fe.columns:
            b = column_value_breakdown(fe[t], t, n_total)
            absent = b["n_absent"]
            artificial = b["n_artificial_zero"]
            true_zero = b["n_true_zero"]
            measured = b["n_measured"]
            raw_non_null = b["n_non_null_raw"]
            if dist_idx is not None and t in dist_idx.index:
                check = verify_target_counts(fe, t, dist_idx.loc[t])

        vdf = load_target_valid(t)
        n_valid = len(vdf) if vdf is not None else None
        m = derive_target_maturity(t)
        n_train = m.get("n_samples")

        ref_phase2 = None
        if dist_idx is not None and t in dist_idx.index:
            ref_phase2 = dist_idx.loc[t].get("available_count")

        rows.append({
            "Cible": TARGETS.get(t, {}).get("label", t),
            "target": t,
            "Unité": TARGETS.get(t, {}).get("unit", ""),
            "Lignes NAV": n_total,
            "Absentes (NaN)": absent,
            "Zéros placeholder": artificial,
            "Zéros réels": true_zero,
            "Valeurs mesurées": measured,
            "Réf. non nulles phase 2": int(ref_phase2) if ref_phase2 is not None and pd.notna(ref_phase2) else None,
            "Vérification": check or "—",
            "Après validation phase 2": n_valid,
            "Échantillons modélisation": n_train,
        })
    return pd.DataFrame(rows)


def missingness_by_column(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Alias — tableau de complétude avec distinction absent / zéro."""
    from services.value_semantics import completeness_table

    return completeness_table(df, columns)


def artifact_freshness() -> pd.DataFrame:
    rows = []
    status = file_status()
    for key, exists in status.items():
        path = PATHS.get(key)
        if not path:
            continue
        mtime = ""
        if exists and Path(path).exists():
            ts = datetime.fromtimestamp(Path(path).stat().st_mtime)
            mtime = ts.strftime("%Y-%m-%d %H:%M")
        rows.append({
            "Artefact": key,
            "Disponible": "✓" if exists else "✗",
            "Dernière mise à jour": mtime if exists else "—",
            "Chemin": str(path),
        })
    return pd.DataFrame(rows)


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for _label, col in FILTER_COLUMNS.items():
        if col not in out.columns:
            continue
        sel = st.session_state.get(f"v2_filter_{col}", "Tous")
        if sel and sel != "Tous":
            out = out[out[col].astype(str) == sel]
    return out


def render_filter_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Filtres contexte")
    for label, col in FILTER_COLUMNS.items():
        if col not in df.columns:
            continue
        opts = ["Tous"] + sorted(df[col].dropna().astype(str).unique().tolist())[:200]
        st.sidebar.selectbox(label.capitalize(), opts, key=f"v2_filter_{col}")
    return apply_filters(df)
