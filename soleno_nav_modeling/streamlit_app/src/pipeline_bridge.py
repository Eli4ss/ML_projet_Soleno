"""Import du package pipeline soleno_nav_modeling (évite conflit avec streamlit_app/src)."""
from __future__ import annotations

import importlib
import sys
from functools import lru_cache
from pathlib import Path

from config import NAV_ROOT

STREAMLIT_APP_ROOT = Path(__file__).resolve().parents[1]
STREAMLIT_SRC = STREAMLIT_APP_ROOT / "src"


def _is_streamlit_src_module(name: str) -> bool:
    mod = sys.modules.get(name)
    if mod is None:
        return False
    path = getattr(mod, "__file__", "") or ""
    return "streamlit_app" in path.replace("\\", "/")


def _snapshot_streamlit_src_modules() -> dict:
    """Sauvegarde les modules src.* de l'app Streamlit (pas ceux du pipeline)."""
    saved = {}
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            if _is_streamlit_src_module(key):
                saved[key] = sys.modules.pop(key)
    return saved


def _clear_pipeline_src_modules() -> None:
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            if not _is_streamlit_src_module(key):
                sys.modules.pop(key, None)


@lru_cache(maxsize=4)
def _import_nav_module(submodule: str):
    """Importe soleno_nav_modeling.src.<submodule> sans écraser streamlit_app/src."""
    app_root = str(STREAMLIT_APP_ROOT)
    nav_root = str(NAV_ROOT)
    saved_path = sys.path.copy()
    saved_streamlit = _snapshot_streamlit_src_modules()
    try:
        _clear_pipeline_src_modules()
        sys.path = [p for p in saved_path if p != app_root]
        if nav_root not in sys.path:
            sys.path.insert(0, nav_root)
        return importlib.import_module(f"src.{submodule}")
    finally:
        _clear_pipeline_src_modules()
        sys.modules.update(saved_streamlit)
        sys.path = saved_path


def get_modeling():
    return _import_nav_module("modeling")


def get_ood_reference():
    return _import_nav_module("ood_reference")


def get_cascade_predictor_module():
    """Retourne le module cascade_predictor depuis le pipeline (pas le src Streamlit)."""
    return _import_nav_module("cascade_predictor")
