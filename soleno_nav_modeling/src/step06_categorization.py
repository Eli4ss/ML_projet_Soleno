"""Étape 6 — Catégorisation des cibles (régression, classification, ranking)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import REPORTS, TABLES, TARGET_COLUMNS, TARGET_DATASETS
from .utils import ensure_dirs


def _tercile_bins(series: pd.Series) -> tuple[np.ndarray, list[str]]:
    vals = series.dropna()
    if len(vals) < 9:
        q = [0, 0.5, 1.0]
        labels = ["bas", "haut"]
    else:
        q = [0, 1 / 3, 2 / 3, 1.0]
        labels = ["bas", "moyen", "haut"]
    bins = vals.quantile(q).values
    bins = np.unique(bins)
    if len(bins) < 2:
        bins = np.array([vals.min(), vals.max()])
    return bins, labels[: max(1, len(bins) - 1)]


def run_step06(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_dirs(TARGET_DATASETS, REPORTS, TABLES)
    cls_dir = TARGET_DATASETS / "06_target_classification_datasets"
    rank_dir = TARGET_DATASETS / "06_target_ranking_datasets"
    ensure_dirs(cls_dir, rank_dir)

    rules = []
    df_cls = df.copy()
    df_rank = df.copy()

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        s = df[target].dropna()
        if len(s) < 5:
            continue
        bins, labels = _tercile_bins(s)
        col_cls = f"{target}_class"
        col_rank = f"{target}_rank"
        try:
            df_cls.loc[df[target].notna(), col_cls] = pd.cut(
                df.loc[df[target].notna(), target], bins=bins, labels=labels, include_lowest=True, duplicates="drop"
            )
        except Exception:
            med = s.median()
            df_cls.loc[df[target].notna(), col_cls] = np.where(df.loc[df[target].notna(), target] <= med, "bas", "haut")

        df_rank.loc[df[target].notna(), col_rank] = df.loc[df[target].notna(), target].rank(pct=True)

        rules.append({
            "target": target,
            "method": "terciles" if len(labels) == 3 else "médiane",
            "bins": str(bins.tolist()),
            "labels": str(labels),
        })
        sub_cls = df_cls[df_cls[target].notna()].copy()
        sub_rank = df_rank[df_rank[target].notna()].copy()
        sub_cls.to_csv(cls_dir / f"{target}_classification.csv", index=False)
        sub_rank.to_csv(rank_dir / f"{target}_ranking.csv", index=False)

    rules_df = pd.DataFrame(rules)
    rules_df.to_excel(TABLES / "06_target_classes_rules.xlsx", index=False)
    df_cls.to_csv(TABLES / "06_NAV_classification_ready.csv", index=False)
    df_rank.to_csv(TABLES / "06_NAV_ranking_ready.csv", index=False)
    return rules_df, df_cls, df_rank
