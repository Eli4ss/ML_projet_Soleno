"""Étape 12 — Sélection et registre des meilleurs modèles."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .config import METRICS, MODELS, REPORTS, TARGET_COLUMNS
from .modeling import build_xy, make_preprocessor
from .utils import ensure_dirs


def run_step12(df: pd.DataFrame, best_baseline: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(METRICS, REPORTS, MODELS / "12_saved_models")
    registry = []

    for target in TARGET_COLUMNS:
        for version in ("A", "B"):
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
                path = MODELS / "12_saved_models" / f"best_{target}_{version}.joblib"
                joblib.dump(pipe, path)
                row = {
                    "target": target,
                    "model_version": version,
                    "algorithm": "RandomForest",
                    "task": "regression",
                    "n_train": len(y),
                    "model_path": str(path),
                    "feature_count": X.shape[1],
                }
                if not best_baseline.empty:
                    sub = best_baseline[(best_baseline["target"] == target) & (best_baseline["model_version"] == version)]
                    if not sub.empty and "mae" in sub.columns:
                        row["holdout_mae"] = sub["mae"].min()
                        row["holdout_r2"] = sub.loc[sub["mae"].idxmin()].get("r2", None)
                registry.append(row)
            except Exception:
                continue

    reg_df = pd.DataFrame(registry)
    reg_df.to_csv(METRICS / "12_preliminary_best_model_registry.csv", index=False)
    (REPORTS / "12_preliminary_model_selection_report.md").write_text(
        "# Sélection préliminaire des modèles\n\n"
        f"{len(registry)} modèles enregistrés (RF par défaut, réentraînés sur toutes les données labellisées).\n",
        encoding="utf-8",
    )
    return reg_df
