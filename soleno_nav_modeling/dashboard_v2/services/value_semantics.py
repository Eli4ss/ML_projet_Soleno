"""Sémantique absent / zéro réel / zéro placeholder — aligné pipeline NAV."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from config import MVP_TARGETS
from src.pipeline_bridge import _import_nav_module
from src.quality_check import load_input_rules

# Features où 0 = « non renseigné » (optional_zero=false dans input_validity_rules.csv)
_INPUT_ZERO_PLACEHOLDER: set[str] | None = None


def _load_input_zero_placeholder() -> set[str]:
    global _INPUT_ZERO_PLACEHOLDER
    if _INPUT_ZERO_PLACEHOLDER is not None:
        return _INPUT_ZERO_PLACEHOLDER
    rules = load_input_rules()
    placeholders: set[str] = set()
    if not rules.empty and "feature" in rules.columns:
        opt = rules.get("optional_zero", pd.Series(dtype=object))
        for _, row in rules.iterrows():
            feat = str(row.get("feature", ""))
            oz = row.get("optional_zero")
            if feat and (oz is False or str(oz).lower() == "false"):
                placeholders.add(feat)
    # Valeurs par défaut si fichier incomplet
    placeholders.update({"mi", "hlmi", "density_g_cm3", "density_plaque_g_cm3", "fluidity_g_10min"})
    _INPUT_ZERO_PLACEHOLDER = placeholders
    return placeholders


@lru_cache(maxsize=1)
def _target_zero_allowed() -> dict[str, bool]:
    try:
        tr = _import_nav_module("target_rules")
        df = tr.rules_dataframe()
        return {
            str(r["target_name"]): bool(r.get("zero_allowed", True))
            for _, r in df.iterrows()
        }
    except Exception:
        return {t: t in ("oit_min", "cell_class", "temp_c") for t in MVP_TARGETS}


def zero_allowed_for_column(column: str) -> bool:
    if column in _target_zero_allowed():
        return _target_zero_allowed()[column]
    return column not in _load_input_zero_placeholder()


def is_artificial_zero(column: str, value) -> bool:
    """True si la valeur 0 représente un placeholder (non mesuré), pas une mesure réelle."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return False
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != 0:
        return False
    return not zero_allowed_for_column(column)


def column_value_breakdown(series: pd.Series, column: str, n_total: int | None = None) -> dict:
    """Comptages distincts : absent, zéro placeholder, zéro réel, valeur mesurée."""
    n_total = n_total if n_total is not None else len(series)
    numeric = pd.to_numeric(series, errors="coerce")
    absent = int(numeric.isna().sum())
    artificial = true_zero = measured = 0
    for v in numeric:
        if pd.isna(v):
            continue
        if is_artificial_zero(column, v):
            artificial += 1
        elif float(v) == 0:
            true_zero += 1
            measured += 1
        else:
            measured += 1
    return {
        "n_total": n_total,
        "n_absent": absent,
        "n_artificial_zero": artificial,
        "n_true_zero": true_zero,
        "n_measured": measured,
        "n_non_null_raw": int(numeric.notna().sum()),
    }


def completeness_table(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in columns:
        if col not in df.columns:
            continue
        b = column_value_breakdown(df[col], col, n)
        rows.append({
            "Colonne": col,
            "Absentes (NaN)": b["n_absent"],
            "Taux absent %": round(100 * b["n_absent"] / n, 1) if n else 0,
            "Zéros placeholder": b["n_artificial_zero"],
            "Zéros réels": b["n_true_zero"],
            "Valeurs mesurées": b["n_measured"],
            "Non nulles brutes": b["n_non_null_raw"],
        })
    return pd.DataFrame(rows).sort_values("Taux absent %", ascending=False)


def filter_for_distribution(
    df: pd.DataFrame,
    column: str,
    *,
    exclude_artificial_zeros: bool = True,
) -> pd.DataFrame:
    """Retourne les lignes exploitables pour histogrammes / violons / médianes."""
    if column not in df.columns or df.empty:
        return df.iloc[0:0]
    out = df.copy()
    numeric = pd.to_numeric(out[column], errors="coerce")
    mask = numeric.notna()
    if exclude_artificial_zeros:
        for idx in out.index[mask]:
            if is_artificial_zero(column, numeric.at[idx]):
                mask.at[idx] = False
    return out.loc[mask]


def distribution_values(df: pd.DataFrame, column: str, *, exclude_artificial_zeros: bool = True) -> pd.Series:
    sub = filter_for_distribution(df, column, exclude_artificial_zeros=exclude_artificial_zeros)
    if sub.empty:
        return pd.Series(dtype=float)
    return pd.to_numeric(sub[column], errors="coerce").dropna()


def verify_target_counts(fe: pd.DataFrame, target: str, dist_row: pd.Series | None) -> str | None:
    """Retourne un message d'écart si comptage FE ≠ référence phase 2."""
    if fe is None or target not in fe.columns or dist_row is None:
        return None
    raw = int(fe[target].notna().sum())
    ref = dist_row.get("available_count")
    if ref is None or pd.isna(ref):
        return None
    ref = int(ref)
    if raw != ref:
        return f"écart {raw - ref:+d} vs phase 2 ({ref})"
    return "cohérent phase 2"
