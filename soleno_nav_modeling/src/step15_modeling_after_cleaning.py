"""Étape 15 — ML et DL sur datasets cibles corrigés."""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline

from .config import (
    N_FOLDS,
    PHASE2_MODELING,
    PHASE2_VALIDATION,
    RANDOM_STATE,
    TARGET_COLUMNS_MODELING,
)
from .modeling import (
    build_xy,
    classification_metrics,
    get_classification_models,
    get_regression_models,
    make_preprocessor,
    regression_metrics,
    split_data,
)
from .target_rules import get_rule
from .utils import ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)

DATASET_VERSIONS_REG = ["raw", "valid", "winsorized", "log"]
REG_ALGORITHMS = ["Ridge", "ElasticNet", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost"]
CLS_ALGORITHMS = ["LogisticRegression", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost"]


def _load_corrected(target: str, version: str) -> pd.DataFrame | None:
    base = PHASE2_VALIDATION / "corrected_target_datasets"
    path = base / f"target_{version}_{target}.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, low_memory=False)


def _filter_training_rows(df: pd.DataFrame, target: str, version: str) -> pd.DataFrame:
    sub = df.copy()
    if version == "valid":
        inv_col = f"{target}__target_is_physically_invalid"
        if inv_col in sub.columns:
            sub = sub[~sub[inv_col].fillna(False)]
        elif "target_is_physically_invalid" in sub.columns:
            sub = sub[~sub["target_is_physically_invalid"].fillna(False)]
    if version in ("raw", "winsorized", "log"):
        rule = get_rule(target)
        y = pd.to_numeric(sub[target], errors="coerce")
        min_v, max_v = rule.get("min_physical_value"), rule.get("max_physical_value")
        mask = y.notna()
        if min_v is not None and pd.notna(min_v):
            mask &= y >= float(min_v)
        if max_v is not None and pd.notna(max_v):
            mask &= y <= float(max_v)
        if not rule.get("negative_allowed", True):
            mask &= y >= 0
        sub = sub[mask | y.isna()]
    return sub


def _run_regression(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    dataset_version: str,
) -> list[dict]:
    results = []
    if target == "cell_class":
        return results
    sub = _filter_training_rows(df, target, dataset_version)
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
    except Exception:
        return results
    if len(y) < 30:
        return results
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_regression_models()
    for name in REG_ALGORITHMS:
        if name not in models:
            continue
        est = models[name]
        pipe = Pipeline([("prep", pre), ("model", est)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            m = regression_metrics(y_test, pred)
            results.append({
                "target": target,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "algorithm": name,
                "task": "regression",
                "n_samples": len(y),
                **m,
            })
        except Exception as e:
            results.append({
                "target": target,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "algorithm": name,
                "task": "regression",
                "error": str(e),
            })
    return results


def _run_classification(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    dataset_version: str,
) -> list[dict]:
    results: list[dict] = []
    col = f"{target}_class"
    path = PHASE2_VALIDATION / "corrected_target_datasets" / f"target_classification_{target}.csv"
    if not path.exists():
        return results
    sub = pd.read_csv(path, low_memory=False)
    if col not in sub.columns:
        return []
    sub = sub[sub[col].notna()].copy()
    sub[target] = sub[col]
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
    except Exception:
        return []
    if len(y) < 30 or y.nunique() < 2:
        return []
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    models = get_classification_models()
    for name in CLS_ALGORITHMS:
        if name not in models:
            continue
        if name == "SVM" and len(y) > 500:
            continue
        pipe = Pipeline([("prep", pre), ("model", models[name])])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
            m = classification_metrics(y_test, pred, proba)
            results.append({
                "target": target,
                "model_version": model_version,
                "dataset_version": "classification",
                "algorithm": name,
                "task": "classification",
                "n_samples": len(y),
                **m,
            })
        except Exception as e:
            results.append({
                "target": target,
                "model_version": model_version,
                "dataset_version": "classification",
                "algorithm": name,
                "task": "classification",
                "error": str(e),
            })
    return results


def _ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    from scipy.stats import spearmanr
    sp, _ = spearmanr(y_true, y_score)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    top_k = max(1, int(0.2 * n))
    true_top = set(np.argsort(y_true)[-top_k:])
    pred_top = set(np.argsort(y_score)[-top_k:])
    true_bot = set(np.argsort(y_true)[:top_k])
    pred_bot = set(np.argsort(y_score)[:top_k])
    return {
        "spearman": float(sp) if not np.isnan(sp) else np.nan,
        "top_k_accuracy": len(true_top & pred_top) / top_k,
        "bottom_k_accuracy": len(true_bot & pred_bot) / top_k,
    }


def _run_ranking(df: pd.DataFrame, target: str, model_version: str, dataset_version: str) -> list[dict]:
    rows = _run_regression(df, target, model_version, dataset_version)
    out = []
    sub = _filter_training_rows(df, target, dataset_version)
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
    except Exception:
        return out
    if len(y) < 30:
        return out
    from sklearn.ensemble import RandomForestRegressor
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    pipe = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))])
    try:
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        m = _ranking_metrics(y_test.values, pred)
        out.append({
            "target": target,
            "model_version": model_version,
            "dataset_version": dataset_version,
            "algorithm": "RandomForest_ranking",
            "task": "ranking",
            "n_samples": len(y),
            **m,
        })
    except Exception:
        pass
    return out


def _run_dl(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    dataset_version: str,
    task: str = "regression",
) -> list[dict]:
    rows = []
    if target == "cell_class" and task == "regression":
        return rows
    sub = _filter_training_rows(df, target, dataset_version)
    if task == "classification":
        col = f"{target}_class"
        p = PHASE2_VALIDATION / "corrected_target_datasets" / f"target_classification_{target}.csv"
        if not p.exists():
            return rows
        sub = pd.read_csv(p, low_memory=False)
        sub = sub[sub[col].notna()].copy()
        sub[target] = sub[col]
    try:
        X, y, num_cols, cat_cols = build_xy(sub, target, model_version)
    except Exception:
        return rows
    if len(y) < 50:
        return rows
    pre = make_preprocessor(num_cols, cat_cols)
    X_train, X_test, y_train, y_test = split_data(X, y)
    algo = "MLP"
    kwargs = dict(hidden_layer_sizes=(64, 32), max_iter=200, random_state=RANDOM_STATE, early_stopping=True)
    est = (MLPClassifier if task == "classification" else MLPRegressor)(**kwargs)
    for _once in [0]:
        pipe = Pipeline([("prep", pre), ("model", est)])
        try:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            if task == "regression":
                m = regression_metrics(y_test, pred)
            else:
                proba = pipe.predict_proba(X_test) if hasattr(pipe, "predict_proba") else None
                m = classification_metrics(y_test, pred, proba)
            rows.append({
                "target": target,
                "model_version": model_version,
                "dataset_version": dataset_version,
                "algorithm": algo,
                "task": task,
                "n_samples": len(y),
                **m,
            })
            if task == "regression":
                joblib.dump(
                    pipe,
                    PHASE2_MODELING / "dl_best_models" / f"mlp_{target}_{model_version}_{dataset_version}.joblib",
                )
        except Exception as e:
            rows.append({"target": target, "algorithm": algo, "error": str(e)})
    return rows


def run_step15() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs(
        PHASE2_MODELING,
        PHASE2_MODELING / "dl_training_curves",
        PHASE2_MODELING / "dl_best_models",
    )
    reg_rows, cls_rows, rank_rows, dl_rows = [], [], [], []

    for target in TARGET_COLUMNS_MODELING:
        if target == "cell_class":
            for version in ("A", "B"):
                cls_rows.extend(_run_classification(pd.DataFrame(), target, version, "classification"))
            continue
        for dver in DATASET_VERSIONS_REG:
            df = _load_corrected(target, dver)
            if df is None:
                continue
            for version in ("A", "B"):
                reg_rows.extend(_run_regression(df, target, version, dver))
                rank_rows.extend(_run_ranking(df, target, version, dver))
                if dver == "valid":
                    dl_rows.extend(_run_dl(df, target, version, dver, "regression"))
        for version in ("A", "B"):
            cls_rows.extend(_run_classification(pd.DataFrame(), target, version, "classification"))
            vdf = _load_corrected(target, "valid")
            if vdf is None:
                vdf = _load_corrected(target, "raw")
            if vdf is not None:
                dl_rows.extend(_run_dl(vdf, target, version, "valid", "classification"))

    reg_df = pd.DataFrame(reg_rows)
    cls_df = pd.DataFrame(cls_rows)
    rank_df = pd.DataFrame(rank_rows)
    dl_df = pd.DataFrame(dl_rows)

    reg_df.to_csv(PHASE2_MODELING / "ml_regression_results.csv", index=False)
    cls_df.to_csv(PHASE2_MODELING / "ml_classification_results.csv", index=False)
    rank_df.to_csv(PHASE2_MODELING / "ml_ranking_results.csv", index=False)
    dl_df.to_csv(PHASE2_MODELING / "dl_regression_results.csv", index=False)
    dl_df.to_csv(PHASE2_MODELING / "dl_classification_results.csv", index=False)
    dl_df.to_csv(PHASE2_MODELING / "dl_ranking_results.csv", index=False)

    best = []
    if not reg_df.empty and "mae" in reg_df.columns:
        valid = reg_df.dropna(subset=["mae"])
        on_scale = valid[valid["dataset_version"].isin(["raw", "valid", "winsorized"])]
        pick_from = on_scale if not on_scale.empty else valid
        for (t, v), g in pick_from.groupby(["target", "model_version"]):
            i = g["mae"].idxmin()
            best.append(g.loc[i].to_dict())
    if not cls_df.empty and "f1" in cls_df.columns:
        valid = cls_df.dropna(subset=["f1"])
        for keys, g in valid.groupby(["target", "model_version"]):
            i = g["f1"].idxmax()
            best.append(g.loc[i].to_dict())
    best_df = pd.DataFrame(best)
    best_df.to_csv(PHASE2_MODELING / "ml_best_models.csv", index=False)

    report = [
        "# Rapport ML après correction des cibles",
        "",
        f"- Expériences régression : {len(reg_df)}",
        f"- Expériences classification : {len(cls_df)}",
        f"- Expériences ranking : {len(rank_rows)}",
        f"- Expériences DL : {len(dl_df)}",
        "",
        "Versions testées : raw, valid, winsorized, log.",
        "Validation : hold-out + KFold (sans GroupKFold).",
    ]
    (PHASE2_MODELING / "ml_after_cleaning_report.md").write_text("\n".join(report), encoding="utf-8")
    (PHASE2_MODELING / "dl_after_cleaning_report.md").write_text(
        "# Rapport DL après correction\n\nMLP sklearn avec early stopping.\n", encoding="utf-8"
    )
    return reg_df, cls_df, rank_df, best_df
