"""Étape 19 — Registre des modèles après correction."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .config import MODELS, PHASE2_MODELING, PHASE2_REGISTRY, TARGET_COLUMNS_MODELING
from .modeling import build_xy, make_preprocessor
from .step15_modeling_after_cleaning import _filter_training_rows, _load_corrected
from .utils import ensure_dirs


def run_step19(best_ml: pd.DataFrame) -> pd.DataFrame:
    models_dir = MODELS / "19_saved_models"
    ensure_dirs(PHASE2_REGISTRY, models_dir)
    registry = []

    for target in TARGET_COLUMNS_MODELING:
        for version in ("A", "B"):
            sub = best_ml[
                (best_ml["target"] == target)
                & (best_ml["model_version"] == version)
            ] if not best_ml.empty else pd.DataFrame()
            if sub.empty:
                continue
            if "mae" in sub.columns and sub["mae"].notna().any():
                row_best = sub.loc[sub["mae"].idxmin()]
            else:
                row_best = sub.iloc[0]
            dver = row_best.get("dataset_version", "valid")
            task = row_best.get("task", "regression")
            if target == "cell_class":
                task = "classification"
                dver = "classification"

            dver_load = dver if dver != "classification" else "valid"
            df = _load_corrected(target, str(dver_load))
            if df is None:
                df = _load_corrected(target, "raw")
            if df is None:
                continue
            if task == "regression" and target != "cell_class":
                df = _filter_training_rows(df, target, str(dver))
                try:
                    X, y, num_cols, cat_cols = build_xy(df, target, version)
                except Exception:
                    continue
                if len(y) < 30:
                    continue
                pre = make_preprocessor(num_cols, cat_cols)
                pipe = Pipeline([
                    ("prep", pre),
                    ("model", RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                ])
                try:
                    pipe.fit(X, y)
                    path = models_dir / f"best_{target}_{version}_{dver}.joblib"
                    joblib.dump(pipe, path)
                    registry.append({
                        "target": target,
                        "task_type": task,
                        "dataset_version": dver,
                        "model_family": "sklearn",
                        "model_name": row_best.get("algorithm", "RandomForest"),
                        "model_version": version,
                        "features_version": "phase2_validated",
                        "model_A_or_B": version,
                        "metrics": f"mae={row_best.get('mae', '')}, r2={row_best.get('r2', '')}",
                        "main_score": row_best.get("mae"),
                        "secondary_scores": f"spearman={row_best.get('spearman', '')}",
                        "n_train": len(y),
                        "n_test": int(len(y) * 0.2),
                        "advantages": "Données cibles validées",
                        "limitations": "Sans validation GroupKFold",
                        "recommended_use": "retenu avec prudence" if len(y) < 200 else "retenu",
                        "model_path": str(path),
                        "decision": "retenu" if len(y) >= 100 else "exploratoire",
                    })
                except Exception:
                    continue

    reg_df = pd.DataFrame(registry)
    reg_df.to_csv(PHASE2_REGISTRY / "updated_best_model_registry.csv", index=False)
    return reg_df
