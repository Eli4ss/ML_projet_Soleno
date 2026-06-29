"""Étape 21 — Validation robuste (GroupKFold, temporelle)."""
from __future__ import annotations

import pandas as pd

from .config import (
    GROUP_COLUMNS,
    PHASE3_COMPARISON,
    PHASE3_ROBUST,
    TARGET_COLUMNS_MODELING,
    TEMPORAL_COLUMN,
    TEMPORAL_PERIOD_COLUMN,
)
from .robust_validation import (
    evaluate_group_kfold,
    evaluate_kfold_random,
    evaluate_temporal,
    load_modeling_frame,
)
from .utils import ensure_dirs


def run_step21() -> pd.DataFrame:
    ensure_dirs(PHASE3_ROBUST, PHASE3_ROBUST / "fold_details")
    all_rows: list[dict] = []
    all_preds: list[pd.DataFrame] = []

    for target in TARGET_COLUMNS_MODELING:
        if target == "cell_class":
            continue
        try:
            df = load_modeling_frame(target, "valid")
        except FileNotFoundError:
            continue

        for model_version in ("A", "B"):
            # Référence KFold aléatoire
            rows, preds = evaluate_kfold_random(df, target, model_version)
            for r in rows:
                r.update({"target": target, "model_version": model_version})
                all_rows.append(r)
            if not preds.empty:
                preds = preds.assign(target=target, model_version=model_version, validation_scheme="kfold_random")
                all_preds.append(preds)

            # GroupKFold fournisseur / grade / origine
            for group_key, group_col in GROUP_COLUMNS.items():
                rows, preds = evaluate_group_kfold(df, target, model_version, group_col, group_key)
                for r in rows:
                    r.update({"target": target, "model_version": model_version})
                    all_rows.append(r)
                if not preds.empty:
                    preds = preds.assign(
                        target=target,
                        model_version=model_version,
                        validation_scheme=f"group_{group_key}",
                    )
                    all_preds.append(preds)

            # Validation temporelle (année)
            rows, preds = evaluate_temporal(df, target, model_version, TEMPORAL_COLUMN)
            for r in rows:
                r.update({"target": target, "model_version": model_version})
                all_rows.append(r)
            if not preds.empty:
                preds = preds.assign(target=target, model_version=model_version, validation_scheme="temporal")
                all_preds.append(preds)

    results = pd.DataFrame(all_rows)
    results.to_csv(PHASE3_ROBUST / "robust_validation_results.csv", index=False)

    if all_preds:
        pred_all = pd.concat(all_preds, ignore_index=True)
        pred_all.to_csv(PHASE3_ROBUST / "robust_cv_predictions.csv", index=False)
    else:
        pred_all = pd.DataFrame()

    report = _build_report(results)
    (PHASE3_ROBUST / "robust_generalization_report.md").write_text(report, encoding="utf-8")
    return results


def _build_report(results: pd.DataFrame) -> str:
    lines = [
        "# Rapport — Généralisation industrielle (validation robuste)",
        "",
        "## Schémas testés",
        "- `kfold_random` : référence (même protocole que phase 1/2)",
        "- `group_supplier` : GroupKFold sur `supplier_code` (features fournisseur exclues du modèle B)",
        "- `group_grade` : GroupKFold sur `description_fr` (grade / description exclue du modèle B)",
        "- `group_origin` : GroupKFold sur `location` (site exclu du modèle B)",
        "- `temporal` : TimeSeriesSplit sur `reception_year` (features date exclues du modèle B)",
        "",
        "## Interprétation",
        "- Si le MAE group >> MAE random : faible généralisation à de nouveaux groupes.",
        "- Modèle A : test matière pure sur nouveaux fournisseurs/sites/grades.",
        "- Modèle B : gain doit persister sans features de la dimension testée.",
        "",
    ]
    if not results.empty and "mae_mean" in results.columns:
        ok = results.dropna(subset=["mae_mean"])
        lines.append("## Synthèse MAE moyen (CV)\n")
        lines.append(ok[["target", "model_version", "validation_scheme", "mae_mean", "r2_mean", "n_samples"]].to_string(index=False))
    return "\n".join(lines)
