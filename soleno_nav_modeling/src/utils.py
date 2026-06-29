"""Utilitaires partagés : chargement, nettoyage, chemins."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .columns import RAW_TO_STANDARD
from .config import DATA_RAW, PROJECT_ROOT


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def load_raw_excel(path: Path | None = None) -> pd.DataFrame:
    path = path or DATA_RAW
    df = pd.read_excel(path)
    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for col in df.columns:
        if col in RAW_TO_STANDARD:
            rename[col] = RAW_TO_STANDARD[col]
        else:
            for raw, std in RAW_TO_STANDARD.items():
                if col.strip() == raw.strip() or col.replace("?", "é") == raw:
                    rename[col] = std
                    break
    out = df.rename(columns=rename)
    # fallback: normalize unknown cols
    for c in out.columns:
        if c not in rename.values():
            key = str(c).lower().strip()
            for raw, std in RAW_TO_STANDARD.items():
                if raw.lower() in key or std in str(c).lower():
                    if c not in rename:
                        out = out.rename(columns={c: std})
    return out


def parse_numeric_series(s: pd.Series) -> pd.Series:
    """Convertit virgules décimales, <, >, ND, N/A en float."""
    if s.dtype in (np.float64, np.int64):
        return pd.to_numeric(s, errors="coerce")

    def _one(val: Any) -> float | np.nan:
        if pd.isna(val):
            return np.nan
        if isinstance(val, (int, float, np.integer, np.floating)):
            return float(val)
        txt = str(val).strip().upper()
        if txt in ("", "ND", "N/A", "NA", "N.D.", "-", "—"):
            return np.nan
        txt = txt.replace(",", ".")
        m = re.match(r"^[<>]\s*([\d.]+)", txt)
        if m:
            return float(m.group(1))
        m = re.match(r"^([\d.]+)\s*%", txt)
        if m:
            return float(m.group(1))
        m = re.search(r"([\d.]+)", txt)
        if m:
            try:
                return float(m.group(1))
            except ValueError:
                return np.nan
        return np.nan

    return s.map(_one)


def coerce_numeric_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = parse_numeric_series(out[c])
    return out


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
