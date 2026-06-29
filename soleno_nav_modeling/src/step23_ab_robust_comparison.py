"""Étape 23 — Comparaison modèle A vs B sous validation robuste."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PHASE3_COMPARISON, PHASE3_ROBUST
from .utils import ensure_dirs


def run_step23(results: pd.DataFrame) -> pd.DataFrame:
    ensure_dirs(PHASE3_COMPARISON, PHASE3_COMPARISON / "figures")
    if results.empty or "mae_mean" not in results.columns:
        return pd.DataFrame()

    valid = results.dropna(subset=["mae_mean"]).copy()
    rows = []
    for (target, scheme), g in valid.groupby(["target", "validation_scheme"]):
        a = g[g["model_version"] == "A"]
        b = g[g["model_version"] == "B"]
        if a.empty or b.empty:
            continue
        ma = float(a["mae_mean"].iloc[0])
        mb = float(b["mae_mean"].iloc[0])
        ra = float(a["r2_mean"].iloc[0]) if "r2_mean" in a.columns else np.nan
        rb = float(b["r2_mean"].iloc[0]) if "r2_mean" in b.columns else np.nan
        rows.append({
            "target": target,
            "validation_scheme": scheme,
            "mae_A": ma,
            "mae_B": mb,
            "mae_delta_B_minus_A": mb - ma,
            "B_better_mae": mb < ma,
            "r2_A": ra,
            "r2_B": rb,
            "r2_delta_B_minus_A": rb - ra if pd.notna(ra) and pd.notna(rb) else np.nan,
            "B_better_r2": rb > ra if pd.notna(ra) and pd.notna(rb) else None,
            "interpretation": _interpret(ma, mb, scheme),
        })

    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(PHASE3_COMPARISON / "model_A_vs_B_robust.csv", index=False)
    cmp_df.to_excel(PHASE3_COMPARISON / "model_A_vs_B_robust.xlsx", index=False)

    _plot_ab_delta(cmp_df)

    report = [
        "# Comparaison Modèle A vs B — validation robuste",
        "",
        "- **MAE plus bas = meilleur**",
        "- `B_better_mae` : True si le contexte opérationnel aide encore sans fuite sur la dimension testée",
        "",
        cmp_df.to_string(index=False) if not cmp_df.empty else "Aucun résultat",
        "",
        "## Lecture industrielle",
        "- Sur `group_supplier`, un B nettement meilleur qu’A suggère un effet fournisseur résiduel non capturé par la matière.",
        "- Sur `group_grade` / `group_origin`, même logique pour grade et site.",
        "- Si B ≈ A en group CV mais B >> A en KFold random : le B sur-apprenait le contexte (phase 1-2).",
    ]
    (PHASE3_COMPARISON / "model_A_vs_B_robust_report.md").write_text("\n".join(report), encoding="utf-8")
    return cmp_df


def _interpret(mae_a: float, mae_b: float, scheme: str) -> str:
    if mae_b < mae_a * 0.95:
        return "B meilleur (contexte utile)"
    if mae_a < mae_b * 0.95:
        return "A meilleur (matière suffit / B sur-apprend)"
    return "Équivalent"


def _plot_ab_delta(cmp_df: pd.DataFrame) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if cmp_df.empty:
        return
    cmp_df = cmp_df.copy()
    cmp_df["label"] = cmp_df["target"] + "\n" + cmp_df["validation_scheme"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(cmp_df))
    w = 0.35
    ax.bar(x - w / 2, cmp_df["mae_A"], w, label="Modèle A")
    ax.bar(x + w / 2, cmp_df["mae_B"], w, label="Modèle B")
    ax.set_xticks(x)
    ax.set_xticklabels(cmp_df["label"], rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("MAE moyen (CV)")
    ax.legend()
    ax.set_title("Modèle A vs B — validation robuste")
    fig.tight_layout()
    fig.savefig(PHASE3_COMPARISON / "figures" / "mae_A_vs_B_by_scheme.png", dpi=100)
    plt.close(fig)
