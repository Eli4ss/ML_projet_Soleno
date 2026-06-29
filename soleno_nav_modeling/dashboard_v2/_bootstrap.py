"""Initialise les chemins d'import pour dashboard_v2 + réutilisation streamlit_app."""
from __future__ import annotations

import sys
from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent
MVP_ROOT = V2_ROOT.parent / "streamlit_app"
NAV_ROOT = V2_ROOT.parent


def _clear_foreign_src_package() -> None:
    """Retire le package `src` pipeline s'il a été chargé avant le MVP."""
    mod = sys.modules.get("src")
    if mod is None:
        return
    path = getattr(mod, "__file__", "") or ""
    normalized = path.replace("\\", "/")
    if "streamlit_app" in normalized:
        return
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            sys.modules.pop(key, None)


def _ensure_path_order() -> None:
    """MVP (streamlit_app) avant V2 — `config` et `src.*` résolvent vers le MVP."""
    for root in (MVP_ROOT, V2_ROOT):
        s = str(root)
        if s in sys.path:
            sys.path.remove(s)
    for root in (MVP_ROOT, V2_ROOT):
        sys.path.insert(0, str(root))


_ensure_path_order()
_clear_foreign_src_package()
