"""Étape 7 — Modèles baseline ML."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .config import METRICS, N_FOLDS, REPORTS, RANDOM_STATE, TARGET_COLUMNS
from .modeling import (
    build_xy,
    classification_metrics,
    get_classification_models,
    get_regression_models,
    make_preprocessor,
    regression_metrics,
    split_data,
)
from .utils import ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)


def _run_regression(df: pd.DataFrame, target: str, model_version: str) -> list[dict]:
    results = []
    try:
        X, y, num_cols, cat_cols = build_xy(df, target, model_version)
    except Exception:
        return results
    if len(y) < 30:
        return results
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    for name, est in get_regression_models().items():
        pipe = Pipeline([("prep", pre), ("model", est)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            m = regression_metrics(y_test, pred)
            cv_mae = np.nan
            if len(y) < 2000:
                try:
                    cv = cross_val_score(
                        pipe, X, y,
                        cv=KFold(n_splits=min(N_FOLDS, 3), shuffle=True, random_state=RANDOM_STATE),
                        scoring="neg_mean_absolute_error", n_jobs=-1,
                    )
                    cv_mae = float(-cv.mean())
                except Exception:
                    pass
            results.append({
                "target": target, "model_version": model_version, "algorithm": name,
                "task": "regression", **m, "cv_mae": cv_mae,
            })
        except Exception as e:
            results.append({
                "target": target, "model_version": model_version, "algorithm": name,
                "task": "regression", "error": str(e),
            })
    return results


def _run_classification(df: pd.DataFrame, target: str, model_version: str) -> list[dict]:
    col = f"{target}_class"
    if col not in df.columns:
        return []
    results = []
    sub = df[df[col].notna()].copy()
    sub[target] = sub[col]
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
    except Exception:
        return results
    if len(y) < 30 or y.nunique() < 2:
        return results
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    for name, est in get_classification_models().items():
        pipe = Pipeline([("prep", pre), ("model", est)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
            m = classification_metrics(y_test, pred, proba)
            results.append({
                "target": target, "model_version": model_version, "algorithm": name,
                "task": "classification", **m,
            })
        except Exception as e:
            results.append({
                "target": target, "model_version": model_version, "algorithm": name,
                "task": "classification", "error": str(e),
            })
    return results


def run_step07(df: pd.DataFrame, df_cls: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs(METRICS, REPORTS)
    reg_rows, cls_rows, rank_rows = [], [], []

    for version in ("A", "B"):
        for target in TARGET_COLUMNS:
            reg_rows.extend(_run_regression(df, target, version))
            cls_rows.extend(_run_classification(df_cls, target, version))
            # ranking: use regression metrics on rank score
            rank_col = f"{target}_rank"
            if rank_col in df.columns:
                tmp = df.copy()
                tmp["_rank_target"] = tmp[rank_col]
                tmp2 = tmp.rename(columns={"_rank_target": target})
                reg_rows.extend(_run_regression(tmp2, target, version))

    reg_df = pd.DataFrame(reg_rows)
    cls_df = pd.DataFrame(cls_rows)
    rank_df = reg_df[reg_df.get("task", "") == "regression"].copy() if not reg_df.empty else pd.DataFrame()

    reg_df.to_csv(METRICS / "07_baseline_results_regression.csv", index=False)
    cls_df.to_csv(METRICS / "07_baseline_results_classification.csv", index=False)
    rank_df.to_csv(METRICS / "07_baseline_results_ranking.csv", index=False)

    best = []
    if not reg_df.empty and "mae" in reg_df.columns:
        valid = reg_df.dropna(subset=["mae"])
        for (t, v), g in valid.groupby(["target", "model_version"]):
            i = g["mae"].idxmin()
            best.append(g.loc[i].to_dict())
    if not cls_df.empty and "f1" in cls_df.columns:
        valid = cls_df.dropna(subset=["f1"])
        for (t, v), g in valid.groupby(["target", "model_version"]):
            i = g["f1"].idxmax()
            row = g.loc[i].to_dict()
            row["task"] = "classification"
            best.append(row)
    best_df = pd.DataFrame(best)
    best_df.to_csv(METRICS / "07_baseline_best_models.csv", index=False)

    report = [
        "# Rapport baseline ML",
        "",
        f"- Expériences régression : {len(reg_df)}",
        f"- Expériences classification : {len(cls_df)}",
        "",
        "Validation : hold-out 80/20 + CV KFold (sans GroupKFold).",
    ]
    (REPORTS / "07_baseline_report.md").write_text("\n".join(report), encoding="utf-8")
    return reg_df, cls_df, best_df
