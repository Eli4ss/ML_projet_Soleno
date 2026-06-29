import _bootstrap  # noqa: F401
import streamlit as st
from src.pages_content import render_data_explorer

st.set_page_config(page_title="Explorateur — Soleno NAV", layout="wide")
render_data_explorer()
