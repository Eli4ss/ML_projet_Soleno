"""Page 3 — Exploration R&D (niveau Analyse R&D)."""
import _bootstrap  # noqa: F401

import pandas as pd
import plotly.express as px
import streamlit as st

from components.charts import show_plotly
from components.dataset_selector import dataset_selector_block, show_dataset_banner
from components.layout import page_header, render_expert_toggle_sidebar
from config import FILTER_COLUMNS, MVP_TARGETS, TARGETS
from services.data_service import render_filter_sidebar
from services.value_semantics import filter_for_distribution
from src.visualization import (
    correlation_heatmap,
    distribution_dual_histogram,
    scatter_colored,
    violin_boxplot_by_group,
)

render_expert_toggle_sidebar()
page_header(
    "Exploration R&D",
    "Comparaisons par fournisseur, grade, site et période — aide à l'analyse.",
    "delivered",
)

prop = st.selectbox(
    "Propriété étudiée",
    MVP_TARGETS,
    format_func=lambda t: f"{TARGETS[t]['label']} ({TARGETS[t]['unit']})",
    key="v2_explore_prop",
)

df, meta = dataset_selector_block(
    key_prefix="v2_exploration",
    default_target=prop,
    in_sidebar=True,
)
if df is None:
    st.stop()

df = render_filter_sidebar(df)
show_dataset_banner(meta)
st.caption(f"{len(df):,} lots après filtres contexte")

exclude_zeros = st.sidebar.checkbox(
    "Exclure zéros placeholder",
    value=True,
    key="v2_explore_exclude_zeros",
    help="Retire des graphiques les 0 = « non mesuré » (distincts des vraies mesures à 0).",
)

group_opts = {k: v for k, v in FILTER_COLUMNS.items() if v in df.columns}
group_by = st.selectbox("Comparer par", list(group_opts.keys()) if group_opts else ["—"])
group_col = group_opts.get(group_by)

viz_df = df
if prop not in df.columns:
    st.warning(
        f"La propriété `{prop}` est absente du jeu sélectionné. "
        "Choisissez « NAV complet » ou « Modélisation — cible valide physiquement »."
    )

prop_df = filter_for_distribution(viz_df, prop, exclude_artificial_zeros=exclude_zeros) if prop in viz_df.columns else viz_df.iloc[0:0]

num_cols = viz_df.select_dtypes(include="number").columns.tolist()
default_corr = [
    c for c in ["mi", "hlmi", "density_g_cm3", "oit_min", "izod", "flexion", "carbon_black", "ash", prop]
    if c in num_cols
]
default_corr = list(dict.fromkeys(default_corr))

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Distribution", "Par groupe", "Nuage de points", "Tendances", "Corrélations"]
)

with tab1:
    if prop in viz_df.columns and not prop_df.empty:
        show_plotly(distribution_dual_histogram(prop_df, prop))
    elif prop in viz_df.columns:
        st.info("Aucune valeur mesurée pour cette propriété (absentes ou zéros placeholder uniquement).")
    else:
        st.info("Sélectionnez une propriété présente dans le jeu de données.")

with tab2:
    if group_col and prop in viz_df.columns and not prop_df.empty:
        show_plotly(violin_boxplot_by_group(prop_df, prop, group_col))
    elif group_col and prop in viz_df.columns:
        st.info("Pas de valeurs mesurées à comparer par groupe.")
    else:
        st.info("Colonne de groupement ou propriété non disponible.")

with tab3:
    st.caption(
        "Relation entre deux variables numériques (zéros placeholder exclus sur chaque axe si option active)."
    )
    if len(num_cols) < 2:
        st.info("Pas assez de colonnes numériques dans ce jeu de données.")
    else:
        sc1, sc2 = st.columns(2)
        with sc1:
            x_sc = st.selectbox(
                "Axe X",
                num_cols,
                index=num_cols.index("mi") if "mi" in num_cols else 0,
                key="v2_scatter_x",
            )
        with sc2:
            y_default = prop if prop in num_cols else min(1, len(num_cols) - 1)
            y_idx = num_cols.index(y_default) if y_default in num_cols else min(1, len(num_cols) - 1)
            y_sc = st.selectbox("Axe Y", num_cols, index=y_idx, key="v2_scatter_y")
        scatter_df = viz_df
        if exclude_zeros:
            scatter_df = filter_for_distribution(scatter_df, x_sc, exclude_artificial_zeros=True)
            scatter_df = filter_for_distribution(scatter_df, y_sc, exclude_artificial_zeros=True)
        hover = [c for c in ("lot_id", "supplier_code", "description_fr", "location") if c in scatter_df.columns]
        if scatter_df.empty:
            st.info("Données insuffisantes pour le nuage de points.")
        elif show_plotly(scatter_colored(scatter_df, x_sc, y_sc, hover_cols=hover)):
            st.caption(
                "Couleurs : bleu = normal, orange = hors plage plausible, rouge = suspect, "
                "violet = outlier IQR. Axes zoomés P1–P99."
            )

with tab4:
    if "reception_year" in viz_df.columns and prop in viz_df.columns and not prop_df.empty:
        trend = prop_df.groupby("reception_year")[prop].median().reset_index()
        trend["reception_year"] = pd.to_numeric(trend["reception_year"], errors="coerce")
        trend = trend.dropna()
        if len(trend) >= 2:
            show_plotly(
                px.line(
                    trend,
                    x="reception_year",
                    y=prop,
                    markers=True,
                    title=f"Médiane annuelle — {TARGETS[prop]['label']} (valeurs mesurées)",
                )
            )
    if "recycled_virgin" in viz_df.columns and prop in viz_df.columns and not prop_df.empty:
        show_plotly(violin_boxplot_by_group(prop_df, prop, "recycled_virgin"))

with tab5:
    st.caption(
        "Matrice paramétrable : choisissez les variables, la méthode et le winsorisage. "
        "Spearman est plus robuste aux valeurs extrêmes que Pearson."
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        corr_method = st.selectbox("Méthode", ["pearson", "spearman"], key="v2_corr_method")
    with cc2:
        winsor_corr = st.checkbox("Winsoriser (P1–P99)", value=True, key="v2_corr_winsor")

    corr_cols = st.multiselect(
        "Variables à inclure",
        num_cols,
        default=default_corr or num_cols[:8],
        key="v2_corr_cols",
        format_func=lambda c: TARGETS.get(c, {}).get("label", c),
    )
    if len(corr_cols) < 2:
        st.info("Sélectionnez au moins deux variables numériques.")
    else:
        corr_df = viz_df
        if exclude_zeros:
            for c in corr_cols:
                corr_df = filter_for_distribution(corr_df, c, exclude_artificial_zeros=True)
        if len(corr_df) < 2:
            st.info("Pas assez de lignes communes après exclusion des placeholders.")
        else:
            show_plotly(
                correlation_heatmap(
                    corr_df,
                    corr_cols,
                    method=corr_method,
                    winsorize=winsor_corr,
                )
            )
