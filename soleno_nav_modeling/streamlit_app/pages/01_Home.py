import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_home

st.set_page_config(page_title="Accueil — Soleno NAV", layout="wide")
render_home()
