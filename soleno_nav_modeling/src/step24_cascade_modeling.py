"""Étape 24 — Modélisation en cascade (Thermique → Mécanique → Performance).

Trois approches sont entraînées et comparées sur **datasets validés phase 2** :
  - Direct   : Matière → cible
  - Cascade  : blocs 1→2→3 avec intermédiaires **mesurés** (mode lab-assisté)
  - PSPP     : stacking OOF — évaluation proche du mode production

Le mode cascade (lab-assisté) surestime souvent la performance en production sans mesures
intermédiaires ; PSPP sert de référence honnête pour l'inférence en chaîne.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline

from .config import (
    CASCADE_BLOC_ORDER,
    CASCADE_BLOCS,
    CASCADE_FINAL_TARGETS,
    CASCADE_MECHANICAL_TARGETS,
    CASCADE_THERMAL_TARGETS,
    MODEL_A_FEATURES,
    N_FOLDS,
    PHASE4_CASCADE,
    RANDOM_STATE,
    TABLES,
    TARGET_COLUMNS_MODELING,
)
from .evaluation_standard import get_dataset_version
from .modeling import (
    build_xy,
    get_regression_models,
    make_preprocessor,
    regression_metrics,
)
from .robust_validation import load_modeling_frame
from .utils import ensure_dirs

warnings.filterwarnings("ignore", category=UserWarning)

ALGORITHMS = ["Ridge", "RandomForest", "ExtraTrees", "GradientBoosting", "XGBoost"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_feature_engineered() -> pd.DataFrame:
    path = TABLES / "04_NAV_feature_engineered.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable : {path}\n"
            "Exécutez d'abord run_pipeline.py."
        )
    return pd.read_csv(path, low_memory=False)


def _load_target_frame(target: str) -> pd.DataFrame:
    """Dataset canonique par cible (phase 2 validé + version log si recommandée)."""
    dver = get_dataset_version(target)
    try:
        return load_modeling_frame(target, dver)
    except FileNotFoundError:
        print(f"    [{target}] fallback dataset feature-engineered (phase 2 absente)")
        return _load_feature_engineered()


def _numeric_cols_for_bloc(bloc_name: str, df: pd.DataFrame) -> list[str]:
    """Colonnes numériques = MODEL_A_FEATURES + outputs des blocs précédents."""
    extra = CASCADE_BLOCS[bloc_name]["extra_inputs"]
    base = [c for c in MODEL_A_FEATURES if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    extra_avail = [
        c for c in extra
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    return list(dict.fromkeys(base + extra_avail))


def _build_cascade_xy(
    df: pd.DataFrame,
    target: str,
    bloc_name: str,
    *,
    extra_overrides: dict[str, pd.Series] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Construit X, y pour un target de cascade.

    Par défaut utilise les valeurs mesurées des intermédiaires (mode lab-assisté).
    *extra_overrides* permet d'injecter des prédictions OOF (mode PSPP).
    """
    extra = CASCADE_BLOCS[bloc_name]["extra_inputs"]
    mask = df[target].notna()
    sub = df.copy()
    for col in extra:
        if extra_overrides and col in extra_overrides:
            aligned = extra_overrides[col].reindex(sub.index)
            sub[col] = aligned
        if col in sub.columns:
            mask = mask & sub[col].notna()
    sub = sub[mask].copy()
    if len(sub) < 20:
        return pd.DataFrame(), pd.Series(dtype=float)

    num_cols = _numeric_cols_for_bloc(bloc_name, sub)
    cat_cols = ["recycled_virgin"] if "recycled_virgin" in sub.columns else []

    feature_cols = [c for c in num_cols + cat_cols if c in sub.columns and c != target]
    X = sub[feature_cols].copy()
    y = pd.to_numeric(sub[target], errors="coerce")
    valid = y.notna()
    return X[valid], y[valid]


def _train_one_target(
    df: pd.DataFrame,
    target: str,
    bloc_name: str,
    models_dir: Path,
) -> list[dict]:
    """Entraîne plusieurs algorithmes sur un target d'un bloc.

    Retourne la liste des métriques (une ligne par algorithme).
    """
    X, y = _build_cascade_xy(df, target, bloc_name)
    if X.empty or len(y) < 20:
        print(f"    [{bloc_name}] {target}: not enough data ({len(y)} rows), skipped.")
        return []

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    pre = make_preprocessor(num_cols, cat_cols)
    all_models = get_regression_models()

    results = []
    best_r2 = -np.inf
    best_pipe: Pipeline | None = None

    for algo_name in ALGORITHMS:
        if algo_name not in all_models:
            continue
        estimator = all_models[algo_name]
        pipe = Pipeline([("prep", pre), ("model", estimator)])
        try:
            kf = KFold(n_splits=min(N_FOLDS, len(y) // 5 + 1), shuffle=True, random_state=RANDOM_STATE)
            y_pred_cv = cross_val_predict(pipe, X, y, cv=kf)
            metrics = regression_metrics(y.values, y_pred_cv)
            metrics.update({
                "target": target,
                "bloc": bloc_name,
                "approach": "cascade",
                "eval_mode": "lab_assisted",
                "dataset_version": get_dataset_version(target),
                "algorithm": algo_name,
                "n_samples": len(y),
            })
            results.append(metrics)

            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                pipe_fitted = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", all_models[algo_name])])
                pipe_fitted.fit(X, y)
                best_pipe = pipe_fitted
                best_algo = algo_name
        except Exception as exc:
            print(f"    [{bloc_name}] {target} / {algo_name} : {exc}")

    if best_pipe is not None:
        out_path = models_dir / f"cascade_{bloc_name}_{target}.joblib"
        joblib.dump(best_pipe, out_path)
        print(f"    [{bloc_name}] {target} -> best={best_algo} R2={best_r2:.3f} | saved: {out_path.name}")

    return results


# ---------------------------------------------------------------------------
# Direct baseline (pour comparaison)
# ---------------------------------------------------------------------------

def _train_direct_target(
    target: str,
    models_dir: Path,
) -> list[dict]:
    """Entraîne un modèle direct Matière → target sur dataset validé."""
    dver = get_dataset_version(target)
    try:
        df = load_modeling_frame(target, dver)
        X, y, num_cols, cat_cols = build_xy(df, target, model_version="A")
    except Exception:
        return []
    if len(y) < 20:
        return []

    pre = make_preprocessor(num_cols, cat_cols)
    all_models = get_regression_models()
    results = []
    best_r2 = -np.inf
    best_pipe: Pipeline | None = None
    best_algo = "RandomForest"

    for algo_name in ALGORITHMS:
        if algo_name not in all_models:
            continue
        pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", all_models[algo_name])])
        try:
            kf = KFold(n_splits=min(N_FOLDS, len(y) // 5 + 1), shuffle=True, random_state=RANDOM_STATE)
            y_pred_cv = cross_val_predict(pipe, X, y, cv=kf)
            metrics = regression_metrics(y.values, y_pred_cv)
            metrics.update({
                "target": target,
                "bloc": "direct",
                "approach": "direct",
                "eval_mode": "production",
                "dataset_version": dver,
                "algorithm": algo_name,
                "n_samples": len(y),
            })
            results.append(metrics)
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                p2 = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", all_models[algo_name])])
                p2.fit(X, y)
                best_pipe = p2
                best_algo = algo_name
        except Exception as exc:
            print(f"    [direct] {target} / {algo_name} : {exc}")

    if best_pipe is not None:
        out_path = models_dir / f"direct_{target}.joblib"
        joblib.dump(best_pipe, out_path)
        print(f"    [direct] {target} -> best={best_algo} R2={best_r2:.3f}")

    return results


# ---------------------------------------------------------------------------
# Cross-val stacking propre (PSPP)
# ---------------------------------------------------------------------------

def _train_pspp_target(
    target: str,
    models_dir: Path,
    oof_thermal: pd.DataFrame | None = None,
    oof_mechanical: pd.DataFrame | None = None,
) -> list[dict]:
    """Entraîne le modèle PSPP avec injection des OOF prédictions intermédiaires."""
    dver = get_dataset_version(target)
    if target in CASCADE_THERMAL_TARGETS:
        return _train_direct_target(target, models_dir)

    try:
        df_enriched = load_modeling_frame(target, dver)
    except FileNotFoundError:
        return []

    if target in CASCADE_MECHANICAL_TARGETS and oof_thermal is not None:
        bloc_name = "mechanical_pspp"
        extra_overrides = {col: oof_thermal[col] for col in oof_thermal.columns if col in oof_thermal.columns}
    elif target in CASCADE_FINAL_TARGETS:
        bloc_name = "final_pspp"
        extra_overrides = {}
        if oof_thermal is not None:
            extra_overrides.update({col: oof_thermal[col] for col in oof_thermal.columns})
        if oof_mechanical is not None:
            extra_overrides.update({col: oof_mechanical[col] for col in oof_mechanical.columns})
    else:
        bloc_name = "direct_pspp"
        extra_overrides = None

    bloc_for_xy = "mechanical" if target in CASCADE_MECHANICAL_TARGETS else (
        "final" if target in CASCADE_FINAL_TARGETS else "thermal"
    )
    X, y = _build_cascade_xy(
        df_enriched,
        target,
        bloc_for_xy,
        extra_overrides=extra_overrides,
    )
    if len(y) < 20:
        return []

    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    pre = make_preprocessor(num_cols, cat_cols)
    all_models = get_regression_models()
    results = []
    best_r2 = -np.inf
    best_pipe: Pipeline | None = None
    best_algo = "RandomForest"

    for algo_name in ALGORITHMS:
        if algo_name not in all_models:
            continue
        pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", all_models[algo_name])])
        try:
            kf = KFold(n_splits=min(N_FOLDS, len(y) // 5 + 1), shuffle=True, random_state=RANDOM_STATE)
            y_pred_cv = cross_val_predict(pipe, X, y, cv=kf)
            metrics = regression_metrics(y.values, y_pred_cv)
            metrics.update({
                "target": target,
                "bloc": bloc_name,
                "approach": "pspp",
                "eval_mode": "production_oof",
                "dataset_version": dver,
                "algorithm": algo_name,
                "n_samples": len(y),
            })
            results.append(metrics)
            if metrics["r2"] > best_r2:
                best_r2 = metrics["r2"]
                p2 = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", all_models[algo_name])])
                p2.fit(X, y)
                best_pipe = p2
                best_algo = algo_name
        except Exception as exc:
            print(f"    [pspp] {target} / {algo_name} : {exc}")

    if best_pipe is not None:
        out_path = models_dir / f"pspp_{target}.joblib"
        joblib.dump(best_pipe, out_path)
        print(f"    [pspp] {target} -> best={best_algo} R2={best_r2:.3f}")

    return results


def _compute_oof_predictions(
    targets: list[str],
    bloc_name: str,
    oof_thermal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calcule les prédictions OOF pour une liste de cibles (stacking propre)."""
    series_map: dict[str, pd.Series] = {}
    all_models = get_regression_models()
    rf = all_models.get("RandomForest", RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1))

    for target in targets:
        df = _load_target_frame(target)
        extra_overrides = None
        if bloc_name == "mechanical" and oof_thermal is not None:
            extra_overrides = {col: oof_thermal[col] for col in oof_thermal.columns}

        X, y = _build_cascade_xy(
            df,
            target,
            bloc_name,
            extra_overrides=extra_overrides,
        )
        if X.empty or len(y) < 20:
            series_map[target] = pd.Series(np.nan, index=df.index, name=target)
            continue
        num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        cat_cols = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
        pipe = Pipeline([("prep", make_preprocessor(num_cols, cat_cols)), ("model", rf)])
        kf = KFold(n_splits=min(N_FOLDS, len(y) // 5 + 1), shuffle=True, random_state=RANDOM_STATE)
        try:
            preds = cross_val_predict(pipe, X, y, cv=kf)
            oof_series = pd.Series(np.nan, index=df.index, name=target)
            oof_series.loc[y.index] = preds
            series_map[target] = oof_series
        except Exception:
            series_map[target] = pd.Series(np.nan, index=df.index, name=target)

    if not series_map:
        return pd.DataFrame()
    return pd.DataFrame(series_map)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_step24() -> pd.DataFrame:
    """Entraîne les 3 approches (Direct, Cascade, PSPP) sur datasets validés."""
    models_dir = PHASE4_CASCADE / "models"
    metrics_dir = PHASE4_CASCADE / "metrics"
    ensure_dirs(PHASE4_CASCADE, models_dir, metrics_dir)

    print("Datasets : phase 2 validés (version log/valid par cible)")

    all_results: list[dict] = []

    # ------ 1. Approche DIRECTE ------
    print("\n--- Approach: DIRECT ---")
    direct_targets = [t for t in TARGET_COLUMNS_MODELING if t != "cell_class"]
    for target in direct_targets:
        print(f"  {target} ({get_dataset_version(target)})")
        all_results.extend(_train_direct_target(target, models_dir))

    # ------ 2. Approche CASCADE (lab-assisté) ------
    print("\n--- Approach: CASCADE lab-assisté (Blocs 1->2->3) ---")
    for bloc_name in CASCADE_BLOC_ORDER:
        bloc_cfg = CASCADE_BLOCS[bloc_name]
        print(f"  Bloc [{bloc_name}] - targets: {bloc_cfg['targets']}")
        for target in bloc_cfg["targets"]:
            print(f"    {target}")
            df_t = _load_target_frame(target)
            all_results.extend(_train_one_target(df_t, target, bloc_name, models_dir))

    # ------ 3. Approche PSPP (production OOF) ------
    print("\n--- Approach: PSPP (production OOF) ---")

    print("  Computing thermal OOF predictions...")
    oof_thermal = _compute_oof_predictions(CASCADE_THERMAL_TARGETS, "thermal")

    print("  Computing mechanical OOF predictions (with thermal OOF)...")
    oof_mechanical = _compute_oof_predictions(
        CASCADE_MECHANICAL_TARGETS,
        "mechanical",
        oof_thermal=oof_thermal if not oof_thermal.empty else None,
    )

    all_pspp_targets = (
        CASCADE_THERMAL_TARGETS
        + CASCADE_MECHANICAL_TARGETS
        + CASCADE_FINAL_TARGETS
    )
    for target in all_pspp_targets:
        print(f"  {target}")
        all_results.extend(
            _train_pspp_target(
                target,
                models_dir,
                oof_thermal if not oof_thermal.empty else None,
                oof_mechanical if not oof_mechanical.empty else None,
            )
        )

    # ------ Sauvegarde ------
    results_df = pd.DataFrame(all_results)
    out_path = metrics_dir / "cascade_all_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nResults saved: {out_path}")

    if not results_df.empty and "r2" in results_df.columns:
        best_df = (
            results_df.sort_values("r2", ascending=False)
            .groupby(["target", "approach"], as_index=False)
            .first()
        )
        best_path = metrics_dir / "cascade_best_per_approach.csv"
        best_df.to_csv(best_path, index=False)
        print(f"Best models per approach: {best_path}")

    return results_df
