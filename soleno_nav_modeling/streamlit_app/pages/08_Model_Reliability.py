import _bootstrap  # noqa: F401

import streamlit as st

from src.pages_content import render_model_reliability



st.set_page_config(page_title="Fiabilité des modèles — Soleno NAV", layout="wide")

render_model_reliability()

