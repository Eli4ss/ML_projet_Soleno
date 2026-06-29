"""Étape 22 — Analyse des erreurs par fournisseur, grade, origine, période."""
from __future__ import annotations

import pandas as pd

from .config import (
    GROUP_COLUMNS,
    PHASE3_ERRORS,
    PHASE3_ROBUST,
    TARGET_COLUMNS_MODELING,
    TEMPORAL_COLUMN,
    TEMPORAL_PERIOD_COLUMN,
)
from .robust_validation import error_analysis_by_group, load_modeling_frame
from .utils import ensure_dirs


def run_step22() -> pd.DataFrame:
    ensure_dirs(PHASE3_ERRORS, PHASE3_ERRORS / "figures")
    pred_path = PHASE3_ROBUST / "robust_cv_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()

    pred_all = pd.read_csv(pred_path)
    error_frames: list[pd.DataFrame] = []

    dimensions = {
        **GROUP_COLUMNS,
        "period_year": TEMPORAL_COLUMN,
        "period_quarter": TEMPORAL_PERIOD_COLUMN,
    }

    for target in TARGET_COLUMNS_MODELING:
        if target == "cell_class":
            continue
        try:
            df = load_modeling_frame(target, "valid")
        except FileNotFoundError:
            continue

        for scheme in pred_all["validation_scheme"].unique():
            sub_pred = pred_all[(pred_all["target"] == target) & (pred_all["validation_scheme"] == scheme)]
            if sub_pred.empty:
                continue
            for dim_name, dim_col in dimensions.items():
                if dim_col not in df.columns:
                    continue
                err = error_analysis_by_group(sub_pred, df, target, dim_col, dim_name)
                if not err.empty:
                    err["validation_scheme"] = scheme
                    error_frames.append(err)

    err_df = pd.concat(error_frames, ignore_index=True) if error_frames else pd.DataFrame()
    err_df.to_csv(PHASE3_ERRORS / "error_by_supplier_grade_origin_period.csv", index=False)

    # Top erreurs par dimension
    if not err_df.empty:
        worst = err_df.nlargest(50, "mae")
        worst.to_csv(PHASE3_ERRORS / "worst_groups_by_mae.csv", index=False)

    _plot_worst_suppliers(err_df)

    report = [
        "# Analyse des erreurs par groupe industriel",
        "",
        f"- Fichier détaillé : `error_by_supplier_grade_origin_period.csv`",
        f"- Pires groupes (MAE) : `worst_groups_by_mae.csv`",
        "",
        "## Dimensions",
        "- `supplier` : code fournisseur",
        "- `grade` : description FR (type matière / grade)",
        "- `origin` : site (`location`)",
        "- `period_year` / `period_quarter` : période de réception",
    ]
    (PHASE3_ERRORS / "error_analysis_report.md").write_text("\n".join(report), encoding="utf-8")
    return err_df


def _plot_worst_suppliers(err_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    sub = err_df[(err_df["dimension_type"] == "supplier") & (err_df["validation_scheme"] == "group_supplier")]
    if sub.empty:
        return
    top = sub.nlargest(15, "mae")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(top["dimension_value"].astype(str), top["mae"])
    ax.set_xlabel("MAE")
    ax.set_title("Pires fournisseurs — GroupKFold supplier")
    fig.tight_layout()
    fig.savefig(PHASE3_ERRORS / "figures" / "worst_suppliers_mae.png", dpi=100)
    plt.close(fig)
