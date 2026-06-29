"""Étape 17 — Explicabilité après correction des cibles."""
from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .config import PHASE2_EXPLAIN, PHASE2_MODELING, PHASE2_VALIDATION, TARGET_COLUMNS_MODELING
from .modeling import build_xy, make_preprocessor
from .step15_modeling_after_cleaning import _filter_training_rows, _load_corrected
from .utils import ensure_dirs


def run_step17() -> pd.DataFrame:
    ensure_dirs(PHASE2_EXPLAIN, PHASE2_EXPLAIN / "shap_summary_plots")
    rows = []
    best_path = PHASE2_MODELING / "ml_best_models.csv"
    best = pd.read_csv(best_path) if best_path.exists() else pd.DataFrame()

    for target in TARGET_COLUMNS_MODELING:
        if target == "cell_class":
            continue
        dver = "valid"
        if not best.empty:
            sub = best[(best["target"] == target) & (best.get("task", "regression") == "regression")]
            if not sub.empty and "dataset_version" in sub.columns:
                dver = sub.iloc[0]["dataset_version"]
        df = _load_corrected(target, dver)
        if df is None:
            continue
        df = _filter_training_rows(df, target, dver)
        for version in ("A", "B"):
            try:
                X, y, num_cols, cat_cols = build_xy(df, target, version)
            except Exception:
                continue
            if len(y) < 50:
                continue
            pre = make_preprocessor(num_cols, cat_cols)
            pipe = Pipeline([
                ("prep", pre),
                ("model", RandomForestRegressor(n_estimators=80, random_state=42, n_jobs=-1)),
            ])
            try:
                pipe.fit(X, y)
                imp = pipe.named_steps["model"].feature_importances_
                names = (num_cols + cat_cols)[: len(imp)]
                for i, val in enumerate(imp):
                    fname = names[i] if i < len(names) else f"f{i}"
                    rows.append({
                        "target": target,
                        "model_version": version,
                        "dataset_version": dver,
                        "feature": fname,
                        "importance": float(val),
                    })
            except Exception:
                continue

    imp_df = pd.DataFrame(rows)
    imp_df.to_csv(PHASE2_EXPLAIN / "feature_importance_by_target.csv", index=False)

    try:
        import matplotlib.pyplot as plt
        for target in imp_df["target"].unique()[:6]:
            sub = imp_df[imp_df["target"] == target].nlargest(10, "importance")
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(sub["feature"], sub["importance"])
            ax.set_title(f"Importance — {target} (après correction)")
            fig.savefig(PHASE2_EXPLAIN / "shap_summary_plots" / f"importance_{target}.png", dpi=100)
            plt.close(fig)
    except Exception:
        pass

    report = [
        "# Explicabilité après correction",
        "",
        "Importance RF sur datasets validés/winsorisés.",
        "Modèle A : drivers matière dominants attendus.",
        "Modèle B : vérifier si fournisseur/grade dominent.",
        "",
        f"Features analysées : {len(imp_df)} entrées.",
    ]
    (PHASE2_EXPLAIN / "explainability_after_cleaning_report.md").write_text("\n".join(report), encoding="utf-8")
    return imp_df
