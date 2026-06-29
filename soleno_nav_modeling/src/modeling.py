"""Préparation matrices, pipelines sklearn et métriques."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import MODEL_A_FEATURES, MODEL_B_EXTRA, RANDOM_STATE, TEST_SIZE


def get_feature_lists(model_version: str) -> tuple[list[str], list[str]]:
    """Retourne (numeric_features, categorical_features) pour A ou B."""
    numeric = [f for f in MODEL_A_FEATURES]
    categorical: list[str] = []
    if model_version == "B":
        categorical = ["recycled_virgin", "supplier_code", "supplier_name", "location", "description_fr"]
        for c in ["reception_year", "reception_month", "reception_quarter"]:
            if c not in numeric:
                numeric.append(c)
    else:
        categorical = ["recycled_virgin", "rheology_class", "density_class", "charge_class"]
    return numeric, categorical


def build_xy(
    df: pd.DataFrame,
    target: str,
    model_version: str,
    exclude_features: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    mask = df[target].notna()
    sub = df.loc[mask].copy()
    y = sub[target]
    numeric, categorical = get_feature_lists(model_version)
    exclude = set(exclude_features or [])
    numeric = [c for c in numeric if c not in exclude]
    categorical = [c for c in categorical if c not in exclude]
    cat_avail = [c for c in categorical if c in sub.columns]
    cat_set = set(cat_avail)
    num_avail = [
        c for c in numeric
        if c in sub.columns
        and c not in cat_set
        and pd.api.types.is_numeric_dtype(sub[c])
    ]
    feature_cols = list(dict.fromkeys(num_avail + cat_avail))
    X = sub.loc[:, feature_cols].copy()
    X = X.loc[:, ~X.columns.duplicated()]
    # Retirer colonnes 100 % manquantes sur le sous-ensemble
    keep = [c for c in X.columns if X[c].notna().any()]
    X = X[keep]
    num_avail = [c for c in num_avail if c in X.columns]
    cat_avail = [c for c in cat_avail if c in X.columns]
    return X, y, num_avail, cat_avail


def get_pipeline_feature_columns(pipeline: Pipeline) -> list[str]:
    """Colonnes attendues par le ColumnTransformer d'un pipeline entraîné."""
    prep = pipeline.named_steps.get("prep")
    if prep is None:
        return []
    cols: list[str] = []
    for _name, _trans, columns in prep.transformers_:
        if columns is not None and _name != "remainder":
            cols.extend(list(columns))
    return list(dict.fromkeys(cols))


def build_inference_x(input_row: dict[str, Any], pipeline: Pipeline) -> pd.DataFrame:
    """Construit la matrice X pour l'inférence à partir d'un dict utilisateur."""
    feature_cols = get_pipeline_feature_columns(pipeline)
    row = {col: input_row.get(col, np.nan) for col in feature_cols}
    return pd.DataFrame([row])


def make_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
  transformers = []
  if num_cols:
      transformers.append(
          ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols)
      )
  if cat_cols:
      transformers.append(
          (
              "cat",
              Pipeline([
                  ("imp", SimpleImputer(strategy="most_frequent")),
                  ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
              ]),
              cat_cols,
          )
      )
  return ColumnTransformer(transformers=transformers, remainder="drop")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    if len(y_true) > 1 and np.nanstd(y_true) > 0 and np.nanstd(y_pred) > 0:
        from scipy.stats import spearmanr
        sp, _ = spearmanr(y_true, y_pred)
    else:
        sp = np.nan
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape, "spearman": float(sp)}


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict[str, Any]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
        except Exception:
            metrics["roc_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def split_data(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def get_regression_models() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(random_state=RANDOM_STATE, max_iter=5000),
        "ElasticNet": ElasticNet(random_state=RANDOM_STATE, max_iter=5000),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(random_state=RANDOM_STATE, n_estimators=100, verbosity=0)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(random_state=RANDOM_STATE, n_estimators=100, verbose=-1)
    except ImportError:
        pass
    return models


def get_classification_models() -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(random_state=RANDOM_STATE, n_estimators=100, verbosity=0, use_label_encoder=False, eval_metric="logloss")
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier
        models["LightGBM"] = LGBMClassifier(random_state=RANDOM_STATE, n_estimators=100, verbose=-1)
    except ImportError:
        pass
    return models
