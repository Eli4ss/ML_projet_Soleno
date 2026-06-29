"""Sélecteur de dataset — composant UI compact."""
from __future__ import annotations

import streamlit as st

from services.dataset_registry import DatasetMeta, render_dataset_selector


def dataset_selector_block(
    *,
    key_prefix: str,
    default_target: str | None = None,
    in_sidebar: bool = False,
) -> tuple:
    """Wrapper pour pages Données / Exploration."""
    return render_dataset_selector(
        key_prefix=key_prefix,
        default_target=default_target,
        location="sidebar" if in_sidebar else "main",
    )


def show_dataset_banner(meta: DatasetMeta | None) -> None:
    if meta and meta.n_rows:
        st.caption(
            f"Visualisations basées sur : **{meta.label}** ({meta.n_rows:,} lots) — `{meta.path}`"
        )
