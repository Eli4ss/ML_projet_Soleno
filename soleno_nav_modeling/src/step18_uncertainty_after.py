"""Étape 18 — Incertitude après correction des cibles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from .config import PHASE2_UNCERTAINTY, TARGET_COLUMNS_MODELING
from .modeling import build_xy, make_preprocessor, split_data
from .step15_modeling_after_cleaning import _filter_training_rows, _load_corrected
from .utils import ensure_dirs


def run_step18() -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dirs(PHASE2_UNCERTAINTY)
    intervals, reliability = [], []

    for target in TARGET_COLUMNS_MODELING:
        if target in ("cell_class",):
            continue
        df = _load_corrected(target, "valid")
        if df is None:
            df = _load_corrected(target, "raw")
        if df is None:
            continue
        df = _filter_training_rows(df, target, "valid")
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
        for b in range(25):
            idx = rng.choice(len(X_train), len(X_train), replace=True)
            pipe = Pipeline([
                ("prep", pre),
                ("model", GradientBoostingRegressor(n_estimators=60, random_state=b)),
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
        widths = high - low
        for i in range(min(80, len(y_test))):
            w = float(widths[i])
            conf = "élevée" if w < np.median(widths) * 0.7 else ("moyenne" if w < np.median(widths) * 1.3 else "faible")
            intervals.append({
                "target": target,
                "dataset_version": "valid",
                "y_true": float(y_test.iloc[i]),
                "prediction": float(med[i]),
                "pi_low": float(low[i]),
                "pi_high": float(high[i]),
                "interval_width": w,
                "confidence_level": conf,
                "comment": f"Bootstrap n={len(preds_boot)}",
            })
        coverage = np.mean(
            (y_test.values[: len(med)] >= low[: len(y_test)])
            & (y_test.values[: len(med)] <= high[: len(y_test)])
        )
        reliability.append({
            "target": target,
            "bootstrap_models": len(preds_boot),
            "empirical_coverage": float(coverage),
            "confidence_score": float(min(1.0, len(y) / 800)),
            "mean_interval_width": float(np.mean(widths)),
        })

    int_df = pd.DataFrame(intervals)
    rel_df = pd.DataFrame(reliability)
    int_df.to_csv(PHASE2_UNCERTAINTY / "prediction_intervals.csv", index=False)
    rel_df.to_csv(PHASE2_UNCERTAINTY / "reliability_scores.csv", index=False)
    (PHASE2_UNCERTAINTY / "uncertainty_after_cleaning_report.md").write_text(
        "# Incertitude après correction\n\nBootstrap 25 répliques, intervalles 10–90 %.\n",
        encoding="utf-8",
    )
    return int_df, rel_df
