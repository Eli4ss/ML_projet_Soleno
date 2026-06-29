import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_single_prediction

st.set_page_config(page_title="Prédiction unitaire — Soleno NAV", layout="wide")
render_single_prediction()
