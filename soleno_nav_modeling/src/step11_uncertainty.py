"""Étape 11 — Incertitude (quantiles, bootstrap)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .config import REPORTS, TABLES, TARGET_COLUMNS
from .modeling import build_xy, make_preprocessor, split_data
from .utils import ensure_dirs


def run_step11(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(REPORTS, TABLES)
    intervals = []
    reliability = []

    for target in TARGET_COLUMNS:
        try:
            X, y, num_cols, cat_cols = build_xy(df, target, "A")
        except Exception:
            continue
        if len(y) < 40:
            continue
        pre = make_preprocessor(num_cols, cat_cols)
        X_train, X_test, y_train, y_test = split_data(X, y)
        preds_boot = []
        rng = np.random.RandomState(42)
        for b in range(20):
            idx = rng.choice(len(X_train), len(X_train), replace=True)
            pipe = Pipeline([
                ("prep", pre),
                ("model", GradientBoostingRegressor(n_estimators=50, random_state=b)),
            ])
            try:
                pipe.fit(X_train.iloc[idx], y_train.iloc[idx])
                preds_boot.append(pipe.predict(X_test))
            except Exception:
                continue
        if not preds_boot:
            continue
        arr = np.array(preds_boot)
        low, med, high = np.percentile(arr, [10, 50, 90], axis=0)
        for i in range(min(50, len(y_test))):
            intervals.append({
                "target": target,
                "y_true": float(y_test.iloc[i]),
                "y_pred_median": float(med[i]),
                "pi_low_10": float(low[i]),
                "pi_high_90": float(high[i]),
            })
        coverage = np.mean((y_test.values[: len(med)] >= low[: len(y_test)]) & (y_test.values[: len(med)] <= high[: len(y_test)]))
        reliability.append({
            "target": target,
            "bootstrap_models": len(preds_boot),
            "nominal_coverage": 0.8,
            "empirical_coverage": float(coverage),
            "confidence_score": float(min(1.0, len(y) / 500)),
        })

    int_df = pd.DataFrame(intervals)
    rel_df = pd.DataFrame(reliability)
    int_df.to_csv(TABLES / "11_prediction_intervals.csv", index=False)
    rel_df.to_csv(TABLES / "11_reliability_scores.csv", index=False)
    (REPORTS / "11_uncertainty_preliminary_report.md").write_text(
        "# Rapport incertitude préliminaire\n\nBootstrap (n=20) + intervalles 10-90 %.\n",
        encoding="utf-8",
    )
    return int_df, rel_df
