"""Validation robuste : GroupKFold, split temporel, métriques agrégées."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline

from .config import (
    GROUP_FEATURE_EXCLUSIONS,
    MAX_GROUPK_FOLDS,
    MIN_GROUPS_ROBUST_CV,
    MIN_SAMPLES_ROBUST_CV,
    RANDOM_STATE,
    TEMPORAL_COLUMN,
)
from .modeling import build_xy, make_preprocessor, regression_metrics
from .step15_modeling_after_cleaning import _filter_training_rows, _load_corrected


def load_modeling_frame(target: str, dataset_version: str = "valid") -> pd.DataFrame:
    df = _load_corrected(target, dataset_version)
    if df is None:
        df = _load_corrected(target, "raw")
    if df is None:
        raise FileNotFoundError(f"Dataset cible introuvable pour {target}")
    return _filter_training_rows(df, target, dataset_version)


def make_group_labels(df: pd.DataFrame, group_col: str) -> pd.Series:
    if group_col not in df.columns:
        return pd.Series("_missing_", index=df.index)
    g = df[group_col].astype(str).str.strip().replace({"nan": "_missing_", "": "_missing_"})
    return g.fillna("_missing_")


def _effective_n_splits(n_groups: int) -> int:
    if n_groups < MIN_GROUPS_ROBUST_CV:
        return 0
    return min(MAX_GROUPK_FOLDS, n_groups)


def _default_regressor() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)


def evaluate_group_kfold(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    group_col: str,
    group_key: str,
    algorithm: str = "RandomForest",
) -> tuple[list[dict], pd.DataFrame]:
    """GroupKFold : généralisation à de nouveaux groupes (fournisseur, grade, origine)."""
    exclude = GROUP_FEATURE_EXCLUSIONS.get(group_key, [])
    try:
        X, y, num_cols, cat_cols = build_xy(df, target, model_version, exclude_features=exclude)
    except Exception as e:
        return [{"error": str(e)}], pd.DataFrame()

    groups = make_group_labels(df.loc[y.index], group_col)
    valid_mask = groups != "_missing_"
    X, y, groups = X.loc[valid_mask], y.loc[valid_mask], groups.loc[valid_mask]

    n_samples = len(y)
    n_groups = groups.nunique()
    if n_samples < MIN_SAMPLES_ROBUST_CV:
        return [{"error": f"n_samples={n_samples} < {MIN_SAMPLES_ROBUST_CV}"}], pd.DataFrame()

    n_splits = _effective_n_splits(n_groups)
    if n_splits < 2:
        return [{"error": f"n_groups={n_groups} insuffisant"}], pd.DataFrame()

    pre = make_preprocessor(num_cols, cat_cols)
    est = _default_regressor() if algorithm == "RandomForest" else ExtraTreesRegressor(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    pipe = Pipeline([("prep", pre), ("model", est)])

    fold_metrics: list[dict] = []
    pred_rows: list[dict] = []

    gkf = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        m = regression_metrics(y.iloc[te].values, pred)
        fold_metrics.append({"fold": fold, **m})
        for i, idx in enumerate(y.iloc[te].index):
            pred_rows.append({
                "index": idx,
                "y_true": float(y.loc[idx]),
                "y_pred": float(pred[i]),
                "residual": float(y.loc[idx] - pred[i]),
                "group_value": groups.loc[idx],
                "fold": fold,
            })

    agg = _aggregate_fold_metrics(fold_metrics)
    summary = {
        "validation_scheme": f"group_{group_key}",
        "group_column": group_col,
        "n_samples": n_samples,
        "n_groups": n_groups,
        "n_splits": n_splits,
        "features_excluded": ",".join(exclude) if exclude else "",
        **agg,
    }
    return [summary], pd.DataFrame(pred_rows)


def evaluate_kfold_random(
    df: pd.DataFrame,
    target: str,
    model_version: str,
) -> tuple[list[dict], pd.DataFrame]:
    """KFold classique (référence) sur le même sous-ensemble."""
    try:
        X, y, num_cols, cat_cols = build_xy(df, target, model_version)
    except Exception as e:
        return [{"error": str(e)}], pd.DataFrame()
    if len(y) < MIN_SAMPLES_ROBUST_CV:
        return [{"error": "effectif insuffisant"}], pd.DataFrame()

    n_splits = min(MAX_GROUPK_FOLDS, 5)
    pre = make_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([("prep", pre), ("model", _default_regressor())])

    fold_metrics, pred_rows = [], []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for fold, (tr, te) in enumerate(kf.split(X, y)):
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        m = regression_metrics(y.iloc[te].values, pred)
        fold_metrics.append({"fold": fold, **m})
        for i, idx in enumerate(y.iloc[te].index):
            pred_rows.append({
                "index": idx,
                "y_true": float(y.loc[idx]),
                "y_pred": float(pred[i]),
                "residual": float(y.loc[idx] - pred[i]),
                "group_value": "random_fold",
                "fold": fold,
            })

    summary = {
        "validation_scheme": "kfold_random",
        "group_column": "",
        "n_samples": len(y),
        "n_groups": np.nan,
        "n_splits": n_splits,
        "features_excluded": "",
        **_aggregate_fold_metrics(fold_metrics),
    }
    return [summary], pd.DataFrame(pred_rows)


def evaluate_temporal(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    time_col: str = TEMPORAL_COLUMN,
) -> tuple[list[dict], pd.DataFrame]:
    """Validation temporelle : entraînement sur le passé, test sur le futur."""
    exclude = ["reception_year", "reception_month", "reception_quarter"] if model_version == "B" else []
    try:
        X, y, num_cols, cat_cols = build_xy(df, target, model_version, exclude_features=exclude)
    except Exception as e:
        return [{"error": str(e)}], pd.DataFrame()

    sub = df.loc[y.index].copy()
    if time_col not in sub.columns:
        return [{"error": f"colonne {time_col} absente"}], pd.DataFrame()

    t = pd.to_numeric(sub[time_col], errors="coerce")
    valid = t.notna()
    X, y, t = X.loc[valid], y.loc[valid], t.loc[valid]
    if len(y) < MIN_SAMPLES_ROBUST_CV:
        return [{"error": "effectif insuffisant"}], pd.DataFrame()

    order = np.argsort(t.values)
    X, y, t = X.iloc[order], y.iloc[order], t.iloc[order]

    n_splits = min(3, max(2, len(t.unique()) - 1))
    pre = make_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([("prep", pre), ("model", _default_regressor())])

    fold_metrics, pred_rows = [], []
    tss = TimeSeriesSplit(n_splits=n_splits)
    for fold, (tr, te) in enumerate(tss.split(X)):
        if len(te) < 5:
            continue
        pipe.fit(X.iloc[tr], y.iloc[tr])
        pred = pipe.predict(X.iloc[te])
        m = regression_metrics(y.iloc[te].values, pred)
        fold_metrics.append({"fold": fold, **m})
        for i, idx in enumerate(y.iloc[te].index):
            pred_rows.append({
                "index": idx,
                "y_true": float(y.loc[idx]),
                "y_pred": float(pred[i]),
                "residual": float(y.loc[idx] - pred[i]),
                "group_value": str(t.loc[idx]),
                "fold": fold,
            })

    if not fold_metrics:
        return [{"error": "splits temporels impossibles"}], pd.DataFrame()

    summary = {
        "validation_scheme": "temporal",
        "group_column": time_col,
        "n_samples": len(y),
        "n_groups": t.nunique(),
        "n_splits": len(fold_metrics),
        "features_excluded": ",".join(exclude),
        **_aggregate_fold_metrics(fold_metrics),
    }
    return [summary], pd.DataFrame(pred_rows)


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict[str, float]:
    if not fold_metrics:
        return {}
    keys = ["mae", "rmse", "r2", "mape", "spearman"]
    out: dict[str, float] = {}
    for k in keys:
        vals = [f[k] for f in fold_metrics if k in f and pd.notna(f.get(k))]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"] = float(np.std(vals))
    return out


def error_analysis_by_group(
    pred_df: pd.DataFrame,
    df_full: pd.DataFrame,
    target: str,
    dimension_col: str,
    dimension_name: str,
) -> pd.DataFrame:
    """MAE / biais / effectif par valeur de dimension (fournisseur, grade, etc.)."""
    if pred_df.empty or dimension_col not in df_full.columns:
        return pd.DataFrame()

    dim = make_group_labels(df_full, dimension_col)
    rows = []
    merged = pred_df.copy()
    merged["dimension"] = merged["index"].map(dim)

    for val, g in merged.groupby("dimension"):
        if len(g) < 3:
            continue
        res = g["residual"].values
        rows.append({
            "target": target,
            "dimension_type": dimension_name,
            "dimension_value": val,
            "n_predictions": len(g),
            "mae": float(np.mean(np.abs(res))),
            "bias": float(np.mean(res)),
            "rmse": float(np.sqrt(np.mean(res ** 2))),
        })
    return pd.DataFrame(rows)
