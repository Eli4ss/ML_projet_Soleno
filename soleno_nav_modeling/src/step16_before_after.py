"""Étape 16 — Comparaison avant/après correction des cibles."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import METRICS, PHASE1_METRICS, PHASE2_COMPARISON, PHASE2_MODELING
from .utils import df_to_markdown, ensure_dirs


def _best_phase1(reg_old: pd.DataFrame) -> pd.DataFrame:
    if reg_old.empty or "mae" not in reg_old.columns:
        return pd.DataFrame()
    v = reg_old.dropna(subset=["mae"])
    rows = []
    for (t, mv), g in v.groupby(["target", "model_version"]):
        i = g["mae"].idxmin()
        r = g.loc[i].to_dict()
        r["dataset_version"] = "phase1_raw"
        r["phase"] = "before"
        rows.append(r)
    return pd.DataFrame(rows)


def _best_phase2(reg_new: pd.DataFrame) -> pd.DataFrame:
    if reg_new.empty or "mae" not in reg_new.columns:
        return pd.DataFrame()
    v = reg_new.dropna(subset=["mae"])
    rows = []
    for (t, mv, dv), g in v.groupby(["target", "model_version", "dataset_version"]):
        i = g["mae"].idxmin()
        r = g.loc[i].to_dict()
        r["phase"] = "after"
        rows.append(r)
    return pd.DataFrame(rows)


def run_step16() -> pd.DataFrame:
    ensure_dirs(PHASE2_COMPARISON, PHASE2_COMPARISON / "before_after_plots")
    old_path = PHASE1_METRICS / "07_baseline_results_regression.csv"
    new_path = PHASE2_MODELING / "ml_regression_results.csv"
    reg_old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    reg_new = pd.read_csv(new_path) if new_path.exists() else pd.DataFrame()

    b1 = _best_phase1(reg_old)
    b2 = _best_phase2(reg_new)
    summary_rows = []

    for target in sorted(set(b1.get("target", pd.Series()).tolist() + b2.get("target", pd.Series()).tolist())):
        o = b1[b1["target"] == target] if not b1.empty else pd.DataFrame()
        n = b2[b2["target"] == target] if not b2.empty else pd.DataFrame()
        for mv in ("A", "B"):
            oa = o[o["model_version"] == mv] if not o.empty else pd.DataFrame()
            na = n[n["model_version"] == mv] if not n.empty else pd.DataFrame()
            mae_before = float(oa["mae"].min()) if not oa.empty and "mae" in oa.columns else np.nan
            r2_before = float(oa["r2"].max()) if not oa.empty and "r2" in oa.columns else np.nan
            na_scale = na[na["dataset_version"].isin(["raw", "valid", "winsorized"])] if "dataset_version" in na.columns else na
            na_pick = na_scale if not na_scale.empty else na
            best_after = na_pick.loc[na_pick["mae"].idxmin()] if not na_pick.empty and "mae" in na_pick.columns else None
            mae_after = float(best_after["mae"]) if best_after is not None else np.nan
            r2_after = float(best_after["r2"]) if best_after is not None and "r2" in best_after else np.nan
            dv_after = best_after.get("dataset_version", "") if best_after is not None else ""
            summary_rows.append({
                "target": target,
                "model_version": mv,
                "mae_before": mae_before,
                "mae_after": mae_after,
                "mae_delta": mae_after - mae_before if pd.notna(mae_before) and pd.notna(mae_after) else np.nan,
                "r2_before": r2_before,
                "r2_after": r2_after,
                "r2_improved": r2_after > r2_before if pd.notna(r2_before) and pd.notna(r2_after) else None,
                "best_dataset_version_after": dv_after,
                "rmse_improved": mae_after < mae_before if pd.notna(mae_before) and pd.notna(mae_after) else None,
            })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(PHASE2_COMPARISON / "before_after_summary_by_target.csv", index=False)

    with pd.ExcelWriter(PHASE2_COMPARISON / "before_after_metrics_comparison.xlsx") as xl:
        summary.to_excel(xl, sheet_name="summary", index=False)
        if not b1.empty:
            b1.to_excel(xl, sheet_name="phase1_best", index=False)
        if not b2.empty:
            b2.to_excel(xl, sheet_name="phase2_best", index=False)

    try:
        import matplotlib.pyplot as plt
        if not summary.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            x = np.arange(len(summary))
            w = 0.35
            ax.bar(x - w / 2, summary["mae_before"], w, label="Avant")
            ax.bar(x + w / 2, summary["mae_after"], w, label="Après")
            ax.set_xticks(x)
            ax.set_xticklabels(summary["target"] + "_" + summary["model_version"], rotation=45, ha="right")
            ax.set_ylabel("MAE")
            ax.legend()
            fig.tight_layout()
            fig.savefig(PHASE2_COMPARISON / "before_after_plots" / "mae_before_after.png", dpi=100)
            plt.close(fig)
    except Exception:
        pass

    report = [
        "# Comparaison avant / après correction des cibles",
        "",
        "## Questions clés",
        "",
        "1. **R²** : voir colonne `r2_improved` dans le résumé.",
        "2. **RMSE/MAE** : voir `mae_delta` (négatif = amélioration).",
        "3. **Log-transform** : comparer `dataset_version=log` dans phase 2.",
        "4. **Classification** : voir résultats phase 2 classification.",
        "5. **Ranking** : voir `ml_ranking_results.csv`.",
        "6. **Modèle B vs A** : comparer par `model_version`.",
        "",
        df_to_markdown(summary, index=False),
    ]
    (PHASE2_COMPARISON / "before_after_comparison_report.md").write_text("\n".join(report), encoding="utf-8")
    return summary
