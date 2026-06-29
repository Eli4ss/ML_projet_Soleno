"""Explicabilité — importance des variables."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src.data_loader import load_importance
from src.pipeline_bridge import get_modeling


def get_importance_table(target: str, model_version: str) -> pd.DataFrame:
    imp = load_importance()
    if imp.empty:
        return pd.DataFrame()
    ver_col = "model_version" if "model_version" in imp.columns else "model_A_or_B"
    sub = imp[imp["target"] == target]
    if ver_col in imp.columns:
        sub = sub[sub[ver_col] == model_version]
    if sub.empty:
        return pd.DataFrame()
    return sub.sort_values("importance", ascending=False)


def compute_live_importance(df: pd.DataFrame, target: str, model_version: str, n: int = 2000) -> pd.DataFrame:
    """Importance RF rapide si pas de fichier."""
    modeling = get_modeling()
    build_xy = modeling.build_xy
    make_preprocessor = modeling.make_preprocessor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import Pipeline

    if target not in df.columns or df[target].notna().sum() < 50:
        return pd.DataFrame()
    sub = df[df[target].notna()].head(n)
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
        pipe = Pipeline([
            ("prep", make_preprocessor(num_cols, cat_cols)),
            ("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(X, y)
        names = num_cols + cat_cols
        imps = pipe.named_steps["model"].feature_importances_
        rows = []
        for i, val in enumerate(imps):
            if i < len(names):
                rows.append({"feature": names[i], "importance": float(val)})
        return pd.DataFrame(rows).sort_values("importance", ascending=False)
    except Exception:
        return pd.DataFrame()
