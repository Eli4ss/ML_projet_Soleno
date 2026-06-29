"""Page 4 — Validation des modèles (niveau Analyse R&D)."""
import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import show_plotly
from components.layout import expert_mode, page_header, render_expert_toggle_sidebar, show_file_missing
from config import PATHS, TARGETS
from services.maturity import build_target_maturity_table, explain_negative_r2
from services.validation_service import (
    load_ab_comparison,
    load_deployment_full,
    load_master_metrics,
    metrics_for_target,
    validation_summary_table,
    degradation_interpretation,
)

render_expert_toggle_sidebar()
page_header(
    "Validation des modèles",
    "Résultats de validation robuste — comparaison des protocoles sans sur-interprétation du R².",
    "delivered",
)

summary = validation_summary_table()
if summary.empty:
    show_file_missing(
        "Évaluation unifiée phase 5",
        str(PATHS.get("recommended_model", "")),
        "Exécutez `python run_scientific_evaluation.py`.",
    )
else:
    st.subheader("Synthèse par cible (modèle recommandé)")
    st.caption("Source : outputs/26_unified_evaluation/recommended_model_per_target.csv")
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.subheader("Comparaison des protocoles de validation")
master = load_master_metrics()
if master.empty:
    st.info("Métriques longues non disponibles.")
else:
    st.caption("Source : master_metrics_long.csv — priorité aux schémas groupés et temporels")
    targets_avail = sorted(master["target"].astype(str).unique())
    sel_target = st.selectbox(
        "Cible détaillée",
        targets_avail,
        format_func=lambda t: TARGETS.get(t, {}).get("label", t),
    )
    model_ver = st.radio("Version modèle", ["A", "B"], horizontal=True)
    sub = metrics_for_target(sel_target, model_ver)
    if not sub.empty:
        display = sub[[
            c for c in [
                "validation_label", "n_samples", "n_groups",
                "r2_mean", "mae_mean", "validation_priority",
            ] if c in sub.columns
        ]].copy()
        display["R² interprétation"] = display["r2_mean"].apply(
            lambda x: explain_negative_r2(x) if pd.notna(x) and float(x) < 0 else ""
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        chart_sub = sub.sort_values("validation_priority") if "validation_priority" in sub.columns else sub
        for metric_col, metric_label in (("r2_mean", "R²"), ("mae_mean", "MAE")):
            if metric_col not in chart_sub.columns:
                continue
            fig = px.bar(
                chart_sub,
                x="validation_label",
                y=metric_col,
                color="validation_scheme",
                title=(
                    f"{TARGETS.get(sel_target, {}).get('label', sel_target)} — "
                    f"{metric_label} par protocole (modèle {model_ver})"
                ),
                labels={metric_col: metric_label, "validation_label": "Protocole"},
            )
            if metric_col == "r2_mean":
                fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.6)
            fig.update_layout(showlegend=False, height=420, xaxis_tickangle=-25)
            show_plotly(fig)

st.subheader("Modèle A vs Modèle B (validation robuste)")
ab = load_ab_comparison()
if ab.empty:
    st.info("Comparaison A/B non disponible — exécutez phase 3 (step 23).")
else:
    st.caption("Source : outputs/23_model_A_vs_B_robust/model_A_vs_B_robust.csv")
    st.dataframe(ab, use_container_width=True, hide_index=True)

if expert_mode():
    st.subheader("Table complète deployment_status")
    dep = load_deployment_full()
    if not dep.empty:
        st.dataframe(dep, use_container_width=True, hide_index=True)

    mat = build_target_maturity_table()
    for _, row in mat.iterrows():
        r2g = row.get("r2_group_supplier")
        if pd.notna(r2g) and float(r2g) < 0:
            st.warning(
                f"**{row['label']}** — R² fournisseur négatif ({float(r2g):.2f}) : "
                f"{explain_negative_r2(float(r2g))}"
            )
        ratio = row.get("mae_degradation_ratio")
        if pd.notna(ratio):
            st.caption(f"{row['label']} — {degradation_interpretation(ratio)}")
