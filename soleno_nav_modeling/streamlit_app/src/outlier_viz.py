"""Détection et filtrage d'outliers pour les visualisations Streamlit."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.quality_check import load_input_rules

# Plages physiques cibles (alignées sur src/target_rules.py)
TARGET_PHYSICAL_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "oit_min": (0, 500),
    "ncls": (0, 200),
    "ucls": (0, 300),
    "izod": (0, 2000),
    "traction": (0, 80),
    "pct_elongation": (0, 2000),
    "flexion": (0, 2000),
    "cell_class": (None, None),
    "temp_c": (-50, 400),
}

LOG_SCALE_SUGGESTED = {"oit_min", "izod", "ncls", "pct_elongation"}

POINT_STATUS_ORDER = ("normal", "hors_plage", "suspect", "outlier_iqr", "outlier_pipeline")
POINT_STATUS_LABELS = {
    "normal": "Normal",
    "hors_plage": "Hors plage plausible",
    "suspect": "Valeur suspecte",
    "outlier_iqr": "Outlier IQR (colonne)",
    "outlier_pipeline": "Outlier pipeline",
}
POINT_STATUS_COLORS = {
    "normal": "#4C78A8",
    "hors_plage": "#F58518",
    "suspect": "#E45756",
    "outlier_iqr": "#B279A2",
    "outlier_pipeline": "#EECA3B",
}


def percentile_bounds(series: pd.Series, low: float = 5, high: float = 95) -> tuple[float, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0, 1.0
    return float(np.percentile(s, low)), float(np.percentile(s, high))


def iqr_outlier_mask(series: pd.Series, mult: float = 1.5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    valid = s.dropna()
    if len(valid) < 10:
        return pd.Series(False, index=series.index)
    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=series.index)
    lo, hi = q1 - mult * iqr, q3 + mult * iqr
    return (s < lo) | (s > hi)


def physical_bounds(column: str) -> tuple[float | None, float | None]:
    if column in TARGET_PHYSICAL_BOUNDS:
        return TARGET_PHYSICAL_BOUNDS[column]
    rules = load_input_rules()
    if rules.empty or column not in rules["feature"].values:
        return None, None
    row = rules.loc[rules["feature"] == column].iloc[0]
    lo = row.get("min_plausible")
    hi = row.get("max_plausible")
    lo = float(lo) if pd.notna(lo) else None
    hi = float(hi) if pd.notna(hi) else None
    return lo, hi


def value_out_of_physical(column: str, value: float) -> bool:
    lo, hi = physical_bounds(column)
    if lo is not None and value < lo:
        return True
    if hi is not None and value > hi:
        return True
    return False


def classify_point_status(
    df: pd.DataFrame,
    column: str,
    index: pd.Index | None = None,
) -> pd.Series:
    """Statut qualité par ligne pour une colonne numérique."""
    idx = index if index is not None else df.index
    status = pd.Series("normal", index=idx, dtype="object")

    if "is_outlier_feature" in df.columns:
        pipe = df.loc[idx, "is_outlier_feature"].fillna(False).astype(bool)
        status = status.mask(pipe, "outlier_pipeline")

    if "is_suspicious_value" in df.columns:
        susp = df.loc[idx, "is_suspicious_value"].fillna(False).astype(bool)
        status = status.mask(susp & (status == "normal"), "suspect")

    if column in df.columns:
        s = pd.to_numeric(df.loc[idx, column], errors="coerce")
        iqr = iqr_outlier_mask(s)
        status = status.mask(iqr & status.isin(["normal", "suspect"]), "outlier_iqr")

        lo, hi = physical_bounds(column)
        if lo is not None or hi is not None:
            physical_bad = pd.Series(False, index=idx)
            if lo is not None:
                physical_bad |= s < lo
            if hi is not None:
                physical_bad |= s > hi
            status = status.mask(physical_bad & (status == "normal"), "hors_plage")

    return status


def classify_row_status(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    """Statut le plus sévère sur plusieurs colonnes (ex. scatter X/Y)."""
    if not columns:
        return pd.Series("normal", index=df.index, dtype="object")
    statuses = [classify_point_status(df, c) for c in columns if c in df.columns]
    if not statuses:
        return pd.Series("normal", index=df.index, dtype="object")

    rank = {k: i for i, k in enumerate(POINT_STATUS_ORDER)}
    out = statuses[0].copy()
    for st in statuses[1:]:
        out = out.where(
            out.map(rank) >= st.map(rank),
            st,
        )
    return out


def apply_view_filter(
    df: pd.DataFrame,
    column: str | None,
    view_mode: str,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if view_mode == "no_pipeline_outliers" and "is_outlier_feature" in out.columns:
        out = out[~out["is_outlier_feature"].fillna(False)]
    elif view_mode == "physical_valid" and column and column in out.columns:
        lo, hi = physical_bounds(column)
        s = pd.to_numeric(out[column], errors="coerce")
        if lo is not None:
            out = out[s >= lo]
        if hi is not None:
            out = out[s <= hi]
    elif view_mode == "zoom_pct" and column and column in out.columns:
        lo, hi = percentile_bounds(out[column], 5, 95)
        s = pd.to_numeric(out[column], errors="coerce")
        out = out[(s >= lo) & (s <= hi)]
    return out


def column_summary_stats(series: pd.Series, column: str) -> dict:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {"n": 0}
    p5, p95 = percentile_bounds(s, 5, 95)
    lo_phys, hi_phys = physical_bounds(column)
    iqr_n = int(iqr_outlier_mask(s).sum())
    phys_n = 0
    if lo_phys is not None:
        phys_n += int((s < lo_phys).sum())
    if hi_phys is not None:
        phys_n += int((s > hi_phys).sum())
    return {
        "n": len(s),
        "min": float(s.min()),
        "max": float(s.max()),
        "median": float(s.median()),
        "p5": p5,
        "p95": p95,
        "iqr_outliers": iqr_n,
        "physical_outliers": phys_n,
        "lo_phys": lo_phys,
        "hi_phys": hi_phys,
    }


def top_extreme_rows(
    df: pd.DataFrame,
    column: str,
    n: int = 10,
    ascending: bool = False,
) -> pd.DataFrame:
    if column not in df.columns:
        return pd.DataFrame()
    sub = df.copy()
    sub["_val"] = pd.to_numeric(sub[column], errors="coerce")
    sub = sub.dropna(subset=["_val"])
    if sub.empty:
        return pd.DataFrame()
    id_cols = [c for c in ("numero", "supplier_code", "description_fr", "location") if c in sub.columns]
    cols = id_cols + [column, "_val"]
    if "is_outlier_feature" in sub.columns:
        cols.append("is_outlier_feature")
    ordered = sub.sort_values("_val", ascending=ascending)
    out = ordered[cols].head(n).rename(columns={"_val": f"{column}_tri"})
    return out
