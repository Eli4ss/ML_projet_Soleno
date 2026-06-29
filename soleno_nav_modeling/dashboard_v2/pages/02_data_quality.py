"""Page 2 — Données et qualité (niveau Analyse R&D)."""
import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import show_plotly
from components.dataset_selector import dataset_selector_block, show_dataset_banner
from components.layout import page_header, render_expert_toggle_sidebar
from services.data_service import (
    artifact_freshness,
    missingness_by_column,
    target_coverage_table,
)
from services.value_semantics import column_value_breakdown, distribution_values
from config import MVP_TARGETS, TARGETS
from src.data_loader import load_target_outlier_summary
from src.outlier_viz import column_summary_stats

OUTLIER_COLUMN_HELP = {
    "n_iqr": (
        "**Outliers IQR** — lots dont la valeur dépasse Q1 − 1,5×IQR ou Q3 + 1,5×IQR "
        "(méthode classique sur les valeurs mesurées de la cible)."
    ),
    "n_robust_z": (
        "**Outliers z robuste** — lots avec |z| > 3,5 où z est calculé via la médiane "
        "et l'écart absolu médian (MAD), moins sensible aux extrêmes que le z-score classique."
    ),
    "n_below_min": (
        "**Sous minimum physique** — lots en dessous du seuil minimal défini dans les règles "
        "de validité polymère (`target_rules.py`) pour cette propriété."
    ),
    "n_above_max": (
        "**Au-dessus du maximum physique** — lots au-dessus du seuil maximal défini "
        "dans les règles de validité polymère pour cette propriété."
    ),
}

render_expert_toggle_sidebar()
page_header(
    "Données et qualité",
    "Aperçu du dataset NAV, complétude et valeurs suspectes — outil d'exploration R&D.",
    "delivered",
)

df, meta = dataset_selector_block(key_prefix="v2_data_quality", in_sidebar=True)
if df is None:
    st.stop()

show_dataset_banner(meta)

st.markdown("### Vue d'ensemble")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Lots (jeu sélectionné)", f"{len(df):,}")
c2.metric("Colonnes", len(df.columns))
c3.metric(
    "Fournisseurs",
    df["supplier_code"].nunique() if "supplier_code" in df.columns else "—",
)
c4.metric(
    "Grades",
    df["description_fr"].nunique() if "description_fr" in df.columns else "—",
)


st.subheader("Couverture par propriété cible (référence multi-modèles)")
st.caption(
    "Table comparative sur le **NAV complet (étape 4)** — indépendante du jeu sélectionné ci-dessus."
    "La colonne **Vérification** signale les écarts avec `target_distribution_summary.csv` (phase 2)."
)
cov = target_coverage_table()
st.dataframe(cov, use_container_width=True, hide_index=True)

key_cols = ["mi", "hlmi", "density_g_cm3", "carbon_black", "ash", "pp", "onset", "peak", "delta_h"]
key_cols += [t for t in MVP_TARGETS if t in df.columns]
miss = missingness_by_column(df, key_cols)
if not miss.empty:
    st.subheader("Complétude — features et cibles")
    fig = px.bar(
        miss.head(20),
        x="Colonne",
        y="Taux absent %",
        title=f"Données absentes (NaN) — {meta.label if meta else ''}",
    )
    show_plotly(fig)
    st.dataframe(miss, use_container_width=True, hide_index=True)

st.subheader("Couverture temporelle")
if "reception_year" in df.columns:
    yr = df.groupby("reception_year").size().reset_index(name="n_lots")
    yr["reception_year"] = pd.to_numeric(yr["reception_year"], errors="coerce")
    yr = yr.dropna()
    if not yr.empty:
        show_plotly(px.bar(yr, x="reception_year", y="n_lots", title="Lots par année de réception"))

st.subheader("Couverture par fournisseur")
if "supplier_code" in df.columns:
    top_sup = df["supplier_code"].value_counts().head(15).reset_index()
    top_sup.columns = ["Fournisseur", "Lots"]
    st.dataframe(top_sup, use_container_width=True, hide_index=True)

outlier_sum = load_target_outlier_summary()
if outlier_sum is not None and not outlier_sum.empty:
    st.subheader("Résumé validation cibles (phase 2 — agrégat)")
    st.caption("Source : `outputs/14_target_validation/target_outlier_summary.csv`")

    with st.expander("Comment lire les colonnes de comptage ?", expanded=True):
        for col, text in OUTLIER_COLUMN_HELP.items():
            st.markdown(text)
        

    display_o = outlier_sum.rename(
        columns={
            "target_name": "Cible",
            "n_iqr": "Outliers IQR",
            "n_robust_z": "Outliers z robuste",
            "n_below_min": "Sous min. physique",
            "n_above_max": "Au-dessus max. physique",
        }
    )
    if "Cible" in display_o.columns:
        display_o["Cible"] = display_o["Cible"].map(
            lambda t: TARGETS.get(str(t), {}).get("label", t)
        )
    st.dataframe(display_o, use_container_width=True, hide_index=True)
else:
    st.info("Résumé outliers phase 2 non disponible — exécutez `run_phase2.py`.")

st.subheader("Distribution et valeurs suspectes")
prop = st.selectbox(
    "Propriété ou feature",
    [c for c in key_cols if c in df.columns],
    format_func=lambda x: TARGETS.get(x, {}).get("label", x),
)
exclude_zeros = st.checkbox(
    "Exclure les zéros placeholder des graphiques",
    value=True,
    help="Retire les 0 interprétés comme « non mesuré » (ex. MI=0, cibles où zéro non admis).",
)

brk = column_value_breakdown(df[prop], prop, len(df))
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Absentes", f"{brk['n_absent']:,}")
s2.metric("Zéros placeholder", f"{brk['n_artificial_zero']:,}")
s3.metric("Zéros réels", f"{brk['n_true_zero']:,}")
s4.metric("Valeurs mesurées", f"{brk['n_measured']:,}")
dist_vals = distribution_values(df, prop, exclude_artificial_zeros=exclude_zeros)
col_stats = column_summary_stats(dist_vals, prop) if not dist_vals.empty else {"n": 0}
s5.metric("Outliers IQR", col_stats.get("iqr_outliers", "—"))

if not dist_vals.empty:
    fig = px.histogram(
        dist_vals,
        nbins=40,
        title=f"Distribution — {TARGETS.get(prop, {}).get('label', prop)} (valeurs mesurées, n={len(dist_vals):,})",
    )
    show_plotly(fig)
else:
    st.info("Aucune valeur mesurée à afficher pour cette colonne (toutes absentes ou placeholders).")

if st.session_state.get("v2_expert_mode"):
    st.subheader("État des artefacts")
    st.dataframe(artifact_freshness(), use_container_width=True, hide_index=True)
