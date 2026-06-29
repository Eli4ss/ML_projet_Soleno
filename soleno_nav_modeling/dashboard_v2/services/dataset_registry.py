"""Sélection et chargement des jeux de données NAV pour visualisations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from config import MVP_TARGETS, OUTPUTS, PATHS, TARGETS
from src.data_loader import load_cleaned, load_feature_engineered
from src.utils import safe_read_csv


@dataclass
class DatasetMeta:
    key: str
    label: str
    path: str
    description: str
    n_rows: int = 0
    n_cols: int = 0
    target: str | None = None


DATASET_OPTIONS: dict[str, dict] = {
    "feature_engineered": {
        "label": "NAV complet — toutes les lignes + features dérivées",
        "tagline": "Étape 4 · exploration générale",
        "description": (
            "Fichier `04_NAV_feature_engineered.csv` : **tous les lots NAV**, colonnes brutes "
            "et dérivées (FRR, log_mi, charge_total…). "
            "Référence pour comparer fournisseurs, grades et périodes sur l’historique entier."
        ),
        "path_key": "feature_engineered",
        "needs_target": False,
    },
    "cleaned": {
        "label": "NAV nettoyé — standardisation sans features dérivées",
        "tagline": "Étape 3 · avant feature engineering",
        "description": (
            "Fichier `03_NAV_cleaned.csv` : lots après audit et nettoyage (noms de colonnes, "
            "doublons, flags qualité), **sans** variables dérivées. "
            "Utile pour mesurer l’impact du feature engineering (étape 4)."
        ),
        "path_key": "cleaned",
        "needs_target": False,
    },
    "target_valid": {
        "label": "Modélisation — cible valide physiquement (phase 2)",
        "tagline": "Outliers invalides retirés · entraînement modèles",
        "description": (
            "Fichier `target_valid_{cible}.csv` : lots où la propriété sélectionnée est "
            "**non nulle et physiquement valide** (hors plage ou incohérentes exclues). "
            "Même jeu que celui utilisé pour entraîner et valider les modèles."
        ),
        "needs_target": True,
        "filename": "target_valid_{target}.csv",
    },
    "target_raw": {
        "label": "Exploration cible — mesure présente, sans filtre physique",
        "tagline": "Phase 2 · voir les lots exclus par validation",
        "description": (
            "Fichier `target_raw_{cible}.csv` : lots avec **valeur non nulle** pour la cible, "
            "y compris mesures suspectes ou hors plage. "
            "Permet de comparer avec le jeu « modélisation » et comprendre les exclusions phase 2."
        ),
        "needs_target": True,
        "filename": "target_raw_{target}.csv",
    },
}


def dataset_option_label(key: str) -> str:
    """Libellé du sélecteur : nom explicite + contexte court."""
    opt = DATASET_OPTIONS[key]
    return f"{opt['label']} · {opt.get('tagline', '')}"


def _target_valid_dir() -> Path:
    return OUTPUTS / "14_target_validation" / "corrected_target_datasets"


def _resolve_path(key: str, target: str | None) -> Path | None:
    opt = DATASET_OPTIONS.get(key)
    if not opt:
        return None
    if opt.get("needs_target"):
        if not target:
            return None
        return _target_valid_dir() / opt["filename"].format(target=target)
    path_key = opt.get("path_key")
    if path_key:
        return PATHS.get(path_key)
    return None


@st.cache_data
def _load_csv_cached(path_str: str) -> pd.DataFrame | None:
    return safe_read_csv(Path(path_str))


def load_dataset(key: str, target: str | None = None) -> tuple[pd.DataFrame | None, DatasetMeta | None]:
    """Charge un jeu de données et retourne (dataframe, métadonnées)."""
    opt = DATASET_OPTIONS.get(key)
    if not opt:
        return None, None

    path = _resolve_path(key, target)
    if path is None:
        return None, None

    df: pd.DataFrame | None
    if key == "feature_engineered":
        df = load_feature_engineered()
    elif key == "cleaned":
        df = load_cleaned()
    else:
        df = _load_csv_cached(str(path)) if path.exists() else None
        if df is not None and target and target in df.columns:
            df = df[df[target].notna()].copy()

    if df is None or df.empty:
        meta = DatasetMeta(
            key=key,
            label=opt["label"],
            path=str(path),
            description=opt["description"],
            target=target,
        )
        return None, meta

    meta = DatasetMeta(
        key=key,
        label=opt["label"] + (f" — {TARGETS.get(target, {}).get('label', target)}" if target else ""),
        path=str(path),
        description=opt["description"],
        n_rows=len(df),
        n_cols=len(df.columns),
        target=target,
    )
    return df, meta


def render_dataset_selector(
    *,
    key_prefix: str = "v2_dataset",
    default_target: str | None = None,
    location: str = "main",
) -> tuple[pd.DataFrame | None, DatasetMeta | None]:
    """
    Affiche le sélecteur de dataset et retourne le dataframe chargé.

    location: 'main' | 'sidebar'
    """
    option_keys = list(DATASET_OPTIONS.keys())

    container = st.sidebar if location == "sidebar" else st
    container.markdown("### Jeu de données")

    ds_key = container.selectbox(
        "Source pour les visualisations",
        option_keys,
        format_func=dataset_option_label,
        key=f"{key_prefix}_source",
        help=(
            "Quatre niveaux de preprocessing NAV : complet (étape 4), nettoyé (étape 3), "
            "cible brute ou cible valide pour modélisation (phase 2)."
        ),
    )

    target: str | None = None
    if DATASET_OPTIONS[ds_key].get("needs_target"):
        targets_avail = [t for t in MVP_TARGETS if (_target_valid_dir() / f"target_valid_{t}.csv").exists()]
        if not targets_avail:
            targets_avail = list(MVP_TARGETS)
        init = default_target if default_target in targets_avail else targets_avail[0]
        idx = targets_avail.index(init) if init in targets_avail else 0
        target = container.selectbox(
            "Propriété cible du jeu phase 2",
            targets_avail,
            index=idx,
            format_func=lambda t: TARGETS.get(t, {}).get("label", t),
            key=f"{key_prefix}_target",
            help="Un fichier par propriété (OIT, NCLS, flexion…).",
        )

    opt = DATASET_OPTIONS[ds_key]
    container.caption(opt["description"])

    df, meta = load_dataset(ds_key, target)
    path = _resolve_path(ds_key, target)

    if df is None:
        container.error(
            f"Fichier indisponible : `{path}` — "
            + ("exécutez `run_phase2.py`." if opt.get("needs_target") else "exécutez `run_pipeline.py`.")
        )
        return None, meta

    container.info(
        f"**{meta.label if meta else opt['label']}** — {len(df):,} lots × {len(df.columns)} colonnes  \n"
        f"`{path}`"
    )
    return df, meta
