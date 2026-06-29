import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_model_comparison

st.set_page_config(page_title="Comparaison des modèles — Soleno NAV", layout="wide")
render_model_comparison()
