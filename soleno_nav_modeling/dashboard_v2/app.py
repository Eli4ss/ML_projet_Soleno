"""Point d'entrée Polymer AI — Dashboard maturité scientifique."""
import _bootstrap  # noqa: F401

import streamlit as st

st.set_page_config(
    page_title="Polymer AI — Maturité",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

from components.layout import maturity_badge, render_expert_toggle_sidebar
from components.maturity_guide import render_maturity_logic
from v2_config.settings import MATURITY_LEVELS, VALIDATION_JOURNEY
from services.data_service import artifact_freshness, nav_dataset_stats
from services.maturity import build_target_maturity_table, count_by_level

st.sidebar.markdown("## Soleno-udes Polymer.AI")
render_expert_toggle_sidebar()

st.title("Polymer.AI")
st.markdown(
    "Plateforme R&D Soleno NAV pour explorer les résines recyclées, "
    "évaluer les modèles et tester des estimations dans un **pilote contrôlé**."
)

st.info(
    "Polymer AI ne remplace pas encore les essais en laboratoire. "
    "Il agit comme un outil d'aide à la décision pour estimer les propriétés, repérer les valeurs "
    "atypiques et orienter les analyses. Les essais physiques restent essentiels pour confirmer "
    "les résultats."
)

# Trois cartes maturité
c1, c2, c3 = st.columns(3)
counts = count_by_level()
for col, key in zip((c1, c2, c3), ("delivered", "pilot", "experimental")):
    cfg = MATURITY_LEVELS[key]
    with col:
        st.markdown(
            f'<div style="background:{cfg["bg"]};border-left:4px solid {cfg["color"]};'
            f'padding:16px;border-radius:8px;min-height:140px;">'
            f'<h3 style="color:{cfg["color"]};margin-top:0;">{cfg["label"]}</h3>'
            f'<p style="font-size:0.95em;">{cfg["description"]}</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if key == "pilot":
            st.caption(f"{counts.get('pilot', 0)} propriété(s) éligibles au pilote")
        elif key == "experimental":
            st.caption(f"{counts.get('experimental', 0)} propriété(s) en recherche")

st.markdown("---")
st.subheader("Indicateurs dataset NAV")
stats = nav_dataset_stats()
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Lots (lignes NAV)", f"{stats.get('n_rows', 0):,}" if stats.get("n_rows") else "—")
m2.metric("Fournisseurs", stats.get("n_suppliers", "—"))
m3.metric("Grades", stats.get("n_grades", "—"))
period = "—"
if stats.get("year_min") and stats.get("year_max"):
    period = f"{stats['year_min']} – {stats['year_max']}"
m4.metric("Période couverte", period)
m5.metric("Propriétés étudiées", len(build_target_maturity_table()))

st.subheader("Parcours projet")
st.markdown(" → ".join(f"**{s}**" for s in VALIDATION_JOURNEY))

st.subheader("Maturité par cible")
mat_df = build_target_maturity_table()
display_cols = [
    "label", "level_label", "n_samples", "deployment_label",
    "r2_group_supplier", "allowed_for_prediction", "status_reason",
]
show = mat_df[[c for c in display_cols if c in mat_df.columns]].copy()
show = show.rename(columns={
    "label": "Propriété",
    "level_label": "Niveau",
    "n_samples": "n obs.",
    "deployment_label": "Statut validation",
    "r2_group_supplier": "R² fournisseur",
    "allowed_for_prediction": "Pilote autorisé",
    "status_reason": "Commentaire",
})
st.dataframe(show, use_container_width=True, hide_index=True)

with st.expander("Disponibilité des fichiers sources"):
    st.dataframe(artifact_freshness(), use_container_width=True, hide_index=True)

st.markdown("---")
render_maturity_logic()
