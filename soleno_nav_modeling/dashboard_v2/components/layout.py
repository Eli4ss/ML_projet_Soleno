"""Composants UI partagés — badges et mise en page."""
from __future__ import annotations

import streamlit as st

from v2_config.settings import MATURITY_LEVELS


def maturity_badge(level: str, *, inline: bool = True) -> None:
    cfg = MATURITY_LEVELS.get(level, MATURITY_LEVELS["delivered"])
    st.markdown(
        f'<span style="background:{cfg["bg"]};color:{cfg["color"]};'
        f'padding:4px 10px;border-radius:6px;font-size:0.85em;font-weight:600;'
        f'border:1px solid {cfg["color"]}33;">{cfg["short"]} — {cfg["label"]}</span>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str, level: str) -> None:
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title(title)
        st.caption(subtitle)
    with col2:
        maturity_badge(level)


def info_tooltip(label: str, help_text: str) -> None:
    st.markdown(f"**{label}** ", help=help_text)


def metric_with_source(label: str, value, source: str, *, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or f"Source : {source}")


def show_file_missing(name: str, path: str, pipeline_hint: str) -> None:
    st.warning(
        f"**{name}** non disponible (`{path}`). "
        f"{pipeline_hint}"
    )


def expert_mode() -> bool:
    return st.session_state.get("v2_expert_mode", False)


def render_expert_toggle_sidebar() -> None:
    st.sidebar.markdown("---")
    st.sidebar.checkbox(
        "Mode expert",
        key="v2_expert_mode",
        help="Affiche métriques détaillées, chemins de fichiers et comparaisons A/B.",
    )
