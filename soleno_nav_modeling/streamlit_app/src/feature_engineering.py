"""Features dérivées pour inférence Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    mi = pd.to_numeric(out.get("mi"), errors="coerce")
    hlmi = pd.to_numeric(out.get("hlmi"), errors="coerce")
    density = pd.to_numeric(out.get("density_g_cm3"), errors="coerce")
    carbon = pd.to_numeric(out.get("carbon_black"), errors="coerce").fillna(0)
    ash = pd.to_numeric(out.get("ash"), errors="coerce").fillna(0)
    onset = pd.to_numeric(out.get("onset"), errors="coerce")
    peak = pd.to_numeric(out.get("peak"), errors="coerce")
    delta_h = pd.to_numeric(out.get("delta_h"), errors="coerce")

    out["frr"] = np.where(mi > 0, hlmi / mi, np.nan)
    out["log_mi"] = np.log1p(mi.fillna(0))
    out["log_hlmi"] = np.log1p(hlmi.fillna(0))
    out["charge_total"] = carbon + ash
    out["thermal_window"] = peak - onset
    if "pp" in out.columns:
        out["pp_indicator"] = pd.to_numeric(out["pp"], errors="coerce").notna().astype(int)
    if density.notna().any() and mi.notna().any():
        out["mi_density_ratio"] = np.where(density > 0, mi / density, np.nan)
        out["hlmi_density_ratio"] = np.where(density > 0, hlmi / density, np.nan)
    if "delta_h" in out.columns:
        out["crystallinity_estimated"] = np.clip(
            pd.to_numeric(out["delta_h"], errors="coerce") / 200.0, 0, 1
        )
    if carbon.notna().any() and ash.notna().any():
        out["carbon_black_to_ash_ratio"] = np.where(ash > 0, carbon / ash, np.nan)
    return out


def row_to_input_dict(row: dict) -> dict:
    df = add_derived_features(pd.DataFrame([row]))
    return df.iloc[0].to_dict()
