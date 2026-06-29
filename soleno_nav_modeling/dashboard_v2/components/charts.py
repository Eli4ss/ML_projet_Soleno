"""Affichage Plotly — masque les graphiques vides."""
from __future__ import annotations

import streamlit as st


def show_plotly(fig, *, caption: str = "") -> bool:
    """Affiche un graphique Plotly uniquement s'il est défini. Retourne True si affiché."""
    if fig is None:
        return False
    st.plotly_chart(fig, use_container_width=True)
    if caption:
        st.caption(caption)
    return True
