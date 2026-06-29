"""Utilitaires Streamlit MVP."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from config import PATHS


def safe_read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False, **kwargs)
    except Exception as e:
        st.warning(f"Impossible de charger `{path.name}` : {e}")
        return None


def safe_read_excel(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    try:
        return pd.read_excel(path, **kwargs)
    except Exception as e:
        st.warning(f"Impossible de charger `{path.name}` : {e}")
        return None


def file_status() -> dict[str, bool]:
    return {k: Path(v).exists() for k, v in PATHS.items() if v}


def show_missing_file(name: str, path: Path) -> None:
    st.info(f"Fichier `{name}` non disponible : `{path}` — exécutez le pipeline correspondant.")


def add_nav_root_to_path() -> None:
    import sys
    from config import NAV_ROOT
    if str(NAV_ROOT) not in sys.path:
        sys.path.insert(0, str(NAV_ROOT))
