"""Étape 3 — Nettoyage et standardisation."""
from __future__ import annotations

import pandas as pd
import numpy as np

from .config import REPORTS, TABLES, TARGET_COLUMNS
from .utils import ensure_dirs, coerce_numeric_columns, load_raw_excel, standardize_columns

NUMERIC_COLS = [
    "fluidity_g_10min", "mi", "hlmi", "density_g_cm3", "density_plaque_g_cm3",
    "carbon_black", "ash", "onset", "peak", "delta_h", "pp",
    "oit_min", "izod", "flexion", "cell_class", "quantity_kg",
    "ncls", "pct_elongation", "traction", "temp_c", "ucls",
]


def _flag_outliers(series: pd.Series, iqr_mult: float = 3.0) -> pd.Series:
    s = series.dropna()
    if len(s) < 10:
        return pd.Series(False, index=series.index)
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(False, index=series.index)
    low, high = q1 - iqr_mult * iqr, q3 + iqr_mult * iqr
    return (series < low) | (series > high)


def run_step03(df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(REPORTS, TABLES)
    log = ["# Journal de nettoyage", ""]
    if df is None:
        df = standardize_columns(load_raw_excel())

    n0 = len(df)
    df = coerce_numeric_columns(df, NUMERIC_COLS)

    for dc in ["reception_date", "completion_date"]:
        if dc in df.columns:
            df[dc] = pd.to_datetime(df[dc], errors="coerce")

    if "recycled_virgin" in df.columns:
        df["recycled_virgin"] = df["recycled_virgin"].astype(str).str.strip().str.upper()

    if "supplier_name" in df.columns:
        df["supplier_name"] = df["supplier_name"].astype(str).str.strip()

    dup_mask = df.duplicated(subset=["numero"], keep=False) if "numero" in df.columns else pd.Series(False, index=df.index)
    df["is_duplicate_lot"] = dup_mask

    outlier_any = pd.Series(False, index=df.index)
    for c in NUMERIC_COLS:
        if c in df.columns:
            outlier_any |= _flag_outliers(df[c]).fillna(False)
    df["is_outlier_feature"] = outlier_any

    suspicious = pd.Series(False, index=df.index)
    if "density_g_cm3" in df.columns:
        suspicious |= (df["density_g_cm3"] < 0.5) | (df["density_g_cm3"] > 1.5)
    if "mi" in df.columns:
        suspicious |= (df["mi"] < 0) | (df["mi"] > 500)
    df["is_suspicious_value"] = suspicious

    rheo_cols = ["mi", "hlmi", "fluidity_g_10min"]
    dens_cols = ["density_g_cm3", "density_plaque_g_cm3"]
    charge_cols = ["carbon_black", "ash", "pp"]
    thermal_cols = ["onset", "peak", "delta_h"]

    df["has_rheology_data"] = df[rheo_cols].notna().any(axis=1) if all(c in df.columns for c in rheo_cols) else False
    df["has_density_data"] = df[dens_cols].notna().any(axis=1) if all(c in df.columns for c in dens_cols) else False
    df["has_charge_data"] = df[charge_cols].notna().any(axis=1) if all(c in df.columns for c in charge_cols) else False
    df["has_thermal_data"] = df[thermal_cols].notna().any(axis=1) if all(c in df.columns for c in thermal_cols) else False
    core = ["mi", "density_g_cm3", "carbon_black"]
    df["has_complete_core_features"] = df[[c for c in core if c in df.columns]].notna().all(axis=1)

    for t in TARGET_COLUMNS:
        if t in df.columns:
            df[f"is_missing_target_{t}"] = df[t].isna()

    df["is_missing_target"] = False
    if TARGET_COLUMNS:
        present = [t for t in TARGET_COLUMNS if t in df.columns]
        if present:
            df["is_missing_target"] = df[present].isna().all(axis=1)

    log.append(f"- Lignes initiales : {n0}")
    log.append(f"- Doublons numero signalés : {dup_mask.sum()}")
    log.append(f"- Valeurs aberrantes (IQR×3) : {outlier_any.sum()}")
    log.append(f"- Valeurs suspectes : {suspicious.sum()}")
    log.append("- Aucune ligne supprimée (politique flags).")

    flags_cols = [c for c in df.columns if c.startswith("is_") or c.startswith("has_")]
    flags_df = df[flags_cols + (["numero"] if "numero" in df.columns else [])].copy()
    flags_df.to_csv(TABLES / "03_data_quality_flags.csv", index=False)
    df.to_csv(TABLES / "03_NAV_cleaned.csv", index=False)
    (REPORTS / "03_cleaning_log.md").write_text("\n".join(log), encoding="utf-8")
    return df, flags_df
