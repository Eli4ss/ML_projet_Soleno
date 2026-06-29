import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_batch_prediction

st.set_page_config(page_title="Prédiction par lot — Soleno NAV", layout="wide")
render_batch_prediction()
