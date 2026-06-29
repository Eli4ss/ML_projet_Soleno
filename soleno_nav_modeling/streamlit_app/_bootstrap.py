"""Initialise sys.path pour imports config / src."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _clear_foreign_src_package() -> None:
    """
    Si une page précédente a laissé `src` = package pipeline dans sys.modules,
    le supprimer pour que `from src.pages_content` résolve vers streamlit_app/src.
    """
    mod = sys.modules.get("src")
    if mod is None:
        return
    path = getattr(mod, "__file__", "") or ""
    if "streamlit_app" in path.replace("\\", "/"):
        return
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            sys.modules.pop(key, None)


_clear_foreign_src_package()
