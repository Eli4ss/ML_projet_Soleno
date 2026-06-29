"""Préparation des fichiers uploadés (batch)."""
from __future__ import annotations

import pandas as pd

from src.feature_engineering import add_derived_features

# Mapping noms courants -> standard
COLUMN_ALIASES = {
    "mi": ["mi", "MI", "Mi"],
    "hlmi": ["hlmi", "HLMI"],
    "density_g_cm3": ["density_g_cm3", "densite", "densité", "density", "Densité g cm3"],
    "carbon_black": ["carbon_black", "noir de carbone", "Noir de carbone"],
    "ash": ["ash", "cendres", "Cendres"],
    "onset": ["onset", "Onset"],
    "peak": ["peak", "Peak"],
    "delta_h": ["delta_h", "Delta H", "delta h"],
    "pp": ["pp", "PP"],
    "supplier_code": ["supplier_code", "fournisseur", "Fournisseur"],
    "supplier_name": ["supplier_name", "nom fourn"],
    "location": ["location", "Location"],
    "description_fr": ["description_fr", "Description FR"],
    "reception_year": ["reception_year", "annee", "année"],
    "recycled_virgin": ["recycled_virgin", "Recyclé Vierge"],
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    rename = {}
    cols_lower = {str(c).strip().lower(): c for c in out.columns}
    for std, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            key = a.strip().lower()
            if key in cols_lower and cols_lower[key] not in rename:
                rename[cols_lower[key]] = std
                break
    return out.rename(columns=rename)


def prepare_uploaded_df(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = normalize_columns(df)
    df = add_derived_features(df)
    core = ["mi", "hlmi", "density_g_cm3"]
    missing = [c for c in core if c not in df.columns]
    return df, missing
