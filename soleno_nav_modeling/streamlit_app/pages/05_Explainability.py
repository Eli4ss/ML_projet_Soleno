import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_explainability

st.set_page_config(page_title="Explicabilité — Soleno NAV", layout="wide")
render_explainability()
