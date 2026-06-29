import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_cascade_comparison

st.set_page_config(page_title="Cascade / PSPP — Soleno NAV", layout="wide")
render_cascade_comparison()
