"""Inférence avec modèles sauvegardés (direct, cascade, PSPP)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from config import MVP_TARGETS, PATHS, TARGETS
from src.data_loader import load_registry
from src.feature_engineering import add_derived_features
from src.pipeline_bridge import get_modeling


@st.cache_resource
def load_model_cached(path: str):
    return joblib.load(path)


def resolve_model_path(target: str, model_version: str) -> Path | None:
    reg = load_registry()
    if reg.empty:
        for folder in (PATHS["models_saved_phase2"], PATHS["models_saved_phase1"]):
            if not folder.exists():
                continue
            for pattern in (
                f"best_{target}_{model_version}.joblib",
                f"best_{target}_{model_version}_valid.joblib",
                f"best_{target}_{model_version}_log.joblib",
            ):
                p = folder / pattern
                if p.exists():
                    return p
        return None

    ver_col = "model_A_or_B" if "model_A_or_B" in reg.columns else "model_version"
    sub = reg[(reg["target"] == target) & (reg[ver_col] == model_version)]
    if sub.empty:
        sub = reg[reg["target"] == target]
    if sub.empty:
        return None
    path = sub.iloc[0].get("model_path", "")
    if path and Path(str(path)).exists():
        return Path(str(path))
    return None


def predict_single(input_row: dict, target: str, model_version: str) -> dict:
    build_inference_x = get_modeling().build_inference_x

    path = resolve_model_path(target, model_version)
    if path is None:
        return {"status": "model_not_available", "prediction": None, "path": None}

    row = add_derived_features(pd.DataFrame([input_row])).iloc[0].to_dict()
    try:
        model = load_model_cached(str(path))
        X = build_inference_x(row, model)
        pred = float(model.predict(X)[0])
        return {"status": "OK", "prediction": pred, "path": str(path)}
    except Exception as e:
        return {"status": "error", "prediction": None, "path": str(path), "error": str(e)}


def predict_batch(df: pd.DataFrame, targets: list[str], model_version: str) -> pd.DataFrame:
    build_inference_x = get_modeling().build_inference_x

    out = df.copy()
    out["model_version"] = model_version
    out["prediction_status"] = "OK"

    for target in targets:
        col = f"predicted_{target}"
        out[col] = pd.NA
        path = resolve_model_path(target, model_version)
        if path is None:
            out[col] = pd.NA
            mask = out["prediction_status"] == "OK"
            out.loc[mask, "prediction_status"] = "model_not_available"
            continue
        try:
            model = load_model_cached(str(path))
            preds = []
            statuses = []
            for _, row in df.iterrows():
                row_d = add_derived_features(pd.DataFrame([row.to_dict()])).iloc[0].to_dict()
                try:
                    X = build_inference_x(row_d, model)
                    preds.append(float(model.predict(X)[0]))
                    statuses.append("OK")
                except Exception:
                    preds.append(pd.NA)
                    statuses.append("error")
            out[col] = preds
            for i, st_val in enumerate(statuses):
                if st_val != "OK" and out.iloc[i]["prediction_status"] == "OK":
                    out.at[out.index[i], "prediction_status"] = st_val
        except Exception:
            out[col] = pd.NA
            out.loc[out["prediction_status"] == "OK", "prediction_status"] = "error"

    core = ["mi", "density_g_cm3"]
    miss = [c for c in core if c not in df.columns or df[c].isna().all()]
    if miss:
        out.loc[out["prediction_status"] == "OK", "prediction_status"] = "missing_features"

    return out


def target_unit(target: str) -> str:
    return TARGETS.get(target, {}).get("unit", "")


# ---------------------------------------------------------------------------
# Cascade / PSPP prediction
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_cascade_predictor(approach: str, models_dir_str: str):
    """Cache le CascadePredictor Streamlit par approche.

    Utilise pipeline_bridge pour éviter le conflit entre streamlit_app/src
    et soleno_nav_modeling/src.
    """
    try:
        from pathlib import Path as _Path
        from src.pipeline_bridge import _import_nav_module
        cascade_mod = _import_nav_module("cascade_predictor")
        return cascade_mod.CascadePredictor.load(
            approach=approach,
            models_dir=_Path(models_dir_str),
        )
    except Exception as exc:
        st.warning(f"[cascade] Impossible de charger les modèles '{approach}': {exc}")
        return None


def cascade_models_dir() -> Path:
    """Retourne le dossier des modèles phase 4."""
    nav_root = Path(__file__).resolve().parents[2]
    return nav_root / "outputs" / "24_cascade_modeling" / "models"


def predict_cascade_single(
    input_row: dict[str, Any],
    approach: str = "cascade",
) -> dict[str, Any]:
    """Prédit toutes les cibles via l'approche en cascade pour un seul lot.

    Parameters
    ----------
    input_row : dict avec les features matière
    approach  : "cascade" | "pspp" | "direct"

    Returns
    -------
    dict avec "status", "predictions" (dict target→float), "approach"
    """
    mdir = cascade_models_dir()
    predictor = _load_cascade_predictor(approach, str(mdir))

    if predictor is None:
        return {
            "status": "model_not_available",
            "predictions": {},
            "approach": approach,
            "message": f"Modèles '{approach}' introuvables. Exécutez run_phase4.py.",
        }

    row = add_derived_features(pd.DataFrame([input_row])).iloc[0].to_dict()
    try:
        preds = predictor.predict(row)
        return {
            "status": "OK",
            "predictions": preds,
            "approach": approach,
        }
    except Exception as exc:
        return {
            "status": "error",
            "predictions": {},
            "approach": approach,
            "error": str(exc),
        }


def predict_cascade_batch(
    df: pd.DataFrame,
    approach: str = "cascade",
) -> pd.DataFrame:
    """Prédit toutes les cibles en cascade pour un DataFrame de lots.

    Retourne le DataFrame d'entrée enrichi avec des colonnes `cascade_{target}`.
    """
    mdir = cascade_models_dir()
    predictor = _load_cascade_predictor(approach, str(mdir))

    out = df.copy()
    if predictor is None:
        return out

    for idx, row in df.iterrows():
        row_d = add_derived_features(pd.DataFrame([row.to_dict()])).iloc[0].to_dict()
        try:
            preds = predictor.predict(row_d)
            for target, val in preds.items():
                col = f"{approach}_{target}"
                if col not in out.columns:
                    out[col] = np.nan
                out.at[idx, col] = val
        except Exception:
            pass

    return out


def cascade_approaches_available() -> list[str]:
    """Retourne la liste des approches pour lesquelles des modèles existent.

    Utilise des fichiers sentinelles représentatifs pour chaque approche.
    PSPP : oit_min est traité comme direct (pspp_oit_min n'existe pas),
    on vérifie donc pspp_ncls.joblib qui est toujours généré.
    """
    mdir = cascade_models_dir()
    if not mdir.exists():
        return []
    # Fichiers sentinelles : un par approche, représentatif
    sentinels = {
        "direct":  "direct_oit_min.joblib",
        "cascade": "cascade_thermal_oit_min.joblib",
        "pspp":    "pspp_ncls.joblib",
    }
    return [approach for approach, fname in sentinels.items() if (mdir / fname).exists()]
