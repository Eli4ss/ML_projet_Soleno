"""Soleno NAV MVP — point d'entrée Streamlit (page d'accueil)."""
import _bootstrap  # noqa: F401

import streamlit as st

st.set_page_config(
    page_title="Soleno NAV",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.pages_content import render_home

render_home()
