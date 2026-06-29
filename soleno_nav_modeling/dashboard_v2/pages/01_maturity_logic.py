"""Page 1 — Logique de maturité (navigation dédiée)."""
import _bootstrap  # noqa: F401

import streamlit as st

from components.layout import page_header, render_expert_toggle_sidebar
from components.maturity_guide import render_maturity_logic

render_expert_toggle_sidebar()

render_maturity_logic()
