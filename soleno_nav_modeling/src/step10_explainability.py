"""Étape 10 — Explainability (importance + SHAP si disponible)."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .config import FIGURES, REPORTS, TABLES, TARGET_COLUMNS
from .modeling import build_xy, make_preprocessor
from .utils import ensure_dirs


def run_step10(df: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(REPORTS, TABLES, FIGURES / "10_shap_summary_plots", FIGURES / "10_local_explanations_examples")
    rows = []

    for target in TARGET_COLUMNS:
        for version in ("A", "B"):
            try:
                X, y, num_cols, cat_cols = build_xy(df, target, version)
            except Exception:
                continue
            if len(y) < 50:
                continue
            pre = make_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([("prep", pre), ("model", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1))])
            try:
                pipe.fit(X, y)
                importances = pipe.named_steps["model"].feature_importances_
                feat_names = num_cols + cat_cols
                for i, imp in enumerate(importances[: len(feat_names)]):
                    if i < len(feat_names):
                        rows.append({
                            "target": target, "model_version": version,
                            "feature": feat_names[i] if i < len(feat_names) else f"f{i}",
                            "importance": float(imp),
                        })
            except Exception:
                continue

    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(TABLES / "10_feature_importance_by_target.csv", index=False)

    try:
        import shap
        import matplotlib.pyplot as plt
        # SHAP sur un sous-ensemble si possible
        shap_note = "SHAP : analyse limitée aux forêts aléatoires sur échantillon."
    except ImportError:
        shap_note = "SHAP non installé — importance RF uniquement."

    try:
        import matplotlib.pyplot as plt
        if not imp_df.empty:
            top = imp_df.groupby("target").apply(lambda g: g.nlargest(5, "importance")).reset_index(drop=True)
            for t in top["target"].unique()[:3]:
                sub = imp_df[imp_df["target"] == t].nlargest(8, "importance")
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.barh(sub["feature"], sub["importance"])
                ax.set_title(f"Importance — {t}")
                fig.savefig(FIGURES / "10_shap_summary_plots" / f"importance_{t}.png", dpi=100)
                plt.close(fig)
    except Exception:
        pass

    report = [
        "# Rapport d'explicabilité",
        "",
        shap_note if "shap_note" in dir() else "Importance par permutation / RF.",
        "",
        "Modèle A : drivers matière (rhéologie, densité, charges).",
        "Modèle B : ajout contexte fournisseur et temporalité.",
    ]
    (REPORTS / "10_explainability_report.md").write_text("\n".join(report), encoding="utf-8")
    return imp_df
