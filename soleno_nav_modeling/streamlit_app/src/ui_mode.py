"""Mode Simple / Expert (sidebar partagée)."""
from __future__ import annotations

import streamlit as st


def ensure_ui_mode() -> str:
    if "ui_mode" not in st.session_state:
        st.session_state.ui_mode = "simple"
    return st.session_state.ui_mode


def render_mode_sidebar() -> str:
    with st.sidebar:
        st.markdown("### Paramètres")
        mode = st.radio(
            "Mode d'affichage",
            options=["simple", "expert"],
            format_func=lambda x: "Simple" if x == "simple" else "Expert",
            index=0 if st.session_state.get("ui_mode", "simple") == "simple" else 1,
            key="ui_mode_radio",
        )
        st.session_state.ui_mode = mode
        if mode == "simple":
            st.caption("Prédiction, confiance et recommandation uniquement.")
        else:
            st.caption("Détails techniques, comparaison A/B, qualité des données, SHAP.")
    return mode


def is_expert() -> bool:
    return ensure_ui_mode() == "expert"
