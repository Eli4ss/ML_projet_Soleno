"""Étape 4 — Feature engineering."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REPORTS, TABLES
from .utils import ensure_dirs


def _bin_class(series: pd.Series, labels: list[str]) -> pd.Series:
    s = series.dropna()
    if len(s) < 30:
        return pd.Series(index=series.index, dtype="object")
    try:
        return pd.qcut(series, q=len(labels), labels=labels, duplicates="drop")
    except ValueError:
        return pd.Series(index=series.index, dtype="object")


def run_step04(df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(REPORTS, TABLES)
    out = df.copy()
    log = ["# Rapport de feature engineering", ""]

    if "hlmi" in out.columns and "mi" in out.columns:
        out["frr"] = np.where(out["mi"] > 0, out["hlmi"] / out["mi"], np.nan)
        log.append("- FRR = HLMI / MI")

    for col, name in [("mi", "log_mi"), ("hlmi", "log_hlmi")]:
        if col in out.columns:
            out[name] = np.log1p(out[col].clip(lower=0))
            log.append(f"- {name} = log1p({col})")

    if "mi" in out.columns and "density_g_cm3" in out.columns:
        out["mi_density_ratio"] = np.where(
            out["density_g_cm3"] > 0, out["mi"] / out["density_g_cm3"], np.nan
        )
    if "hlmi" in out.columns and "density_g_cm3" in out.columns:
        out["hlmi_density_ratio"] = np.where(
            out["density_g_cm3"] > 0, out["hlmi"] / out["density_g_cm3"], np.nan
        )

    charge_parts = [c for c in ["carbon_black", "ash", "pp"] if c in out.columns]
    if charge_parts:
        out["charge_total"] = out[charge_parts].sum(axis=1, skipna=True)
        log.append("- charge_total = somme charges")

    if "carbon_black" in out.columns and "ash" in out.columns:
        out["carbon_black_to_ash_ratio"] = np.where(
            out["ash"] > 0, out["carbon_black"] / out["ash"], np.nan
        )

    if "pp" in out.columns:
        out["pp_indicator"] = (out["pp"].fillna(0) > 0).astype(int)

    if "peak" in out.columns and "onset" in out.columns:
        out["thermal_window"] = out["peak"] - out["onset"]
        log.append("- thermal_window = Peak - Onset")

    if "delta_h" in out.columns:
        out["crystallinity_estimated"] = out["delta_h"]  # proxy DSC
        log.append("- crystallinity_estimated ≈ Delta H")

    if "mi" in out.columns:
        out["rheology_class"] = _bin_class(out["mi"], ["basse", "moyenne", "haute"])
    if "density_g_cm3" in out.columns:
        out["density_class"] = _bin_class(out["density_g_cm3"], ["basse", "moyenne", "haute"])
    if "charge_total" in out.columns:
        out["charge_class"] = _bin_class(out["charge_total"], ["faible", "moyenne", "élevée"])

    if "reception_date" in out.columns:
        out["reception_year"] = out["reception_date"].dt.year
        out["reception_month"] = out["reception_date"].dt.month
        out["reception_quarter"] = out["reception_date"].dt.quarter

    out.to_csv(TABLES / "04_NAV_feature_engineered.csv", index=False)
    (REPORTS / "04_feature_engineering_report.md").write_text("\n".join(log), encoding="utf-8")
    return out
