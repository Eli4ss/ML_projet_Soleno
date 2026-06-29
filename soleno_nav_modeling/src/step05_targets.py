"""Étape 5 — Jeux de données par cible."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REPORTS, TABLES, TARGET_COLUMNS, TARGET_DATASETS, MIN_TARGET_SAMPLES
from .utils import ensure_dirs


def _task_recommendation(n: int, unique: int) -> str:
    if n < MIN_TARGET_SAMPLES:
        return "régression_faible_effectif (explorer quand même)"
    if unique <= 5:
        return "classification"
    if unique <= 15:
        return "classification_ou_régression"
    return "régression"


def run_step05(df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(TARGET_DATASETS, REPORTS, TABLES)
    rows = []
    strategy_lines = ["# Stratégie de modélisation par cible", ""]

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        sub = df[df[target].notna()].copy()
        n = len(sub)
        miss_feat = sub.isna().mean().mean() * 100
        desc = sub[target].describe()
        path = TARGET_DATASETS / f"05_target_{target}.csv"
        sub.to_csv(path, index=False)
        uq = int(sub[target].nunique())
        task = _task_recommendation(n, uq)
        rows.append({
            "target": target,
            "row_count": n,
            "feature_columns": sub.shape[1],
            "target_missing_rate_pct": round((1 - n / len(df)) * 100, 2),
            "target_unique": uq,
            "target_mean": desc.get("mean", np.nan) if n else np.nan,
            "target_std": desc.get("std", np.nan) if n else np.nan,
            "target_min": desc.get("min", np.nan) if n else np.nan,
            "target_max": desc.get("max", np.nan) if n else np.nan,
            "recommended_task": task,
        })
        strategy_lines.append(
            f"## {target}\n- Effectif : {n}\n- Tâche recommandée : {task}\n"
            f"- Distribution : min={desc.get('min', 'NA'):.3g}, max={desc.get('max', 'NA'):.3g}\n"
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "05_target_availability_summary.csv", index=False)
    (REPORTS / "05_target_modeling_strategy.md").write_text("\n".join(strategy_lines), encoding="utf-8")
    return summary
