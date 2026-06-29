"""Étape 14 — Validation des cibles, flags, datasets corrigés."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .config import PHASE2_VALIDATION, TABLES, TARGET_COLUMNS
from .target_rules import RULE_COLUMNS, get_rule, rules_dataframe
from .utils import ensure_dirs


def _robust_zscore(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = np.median(np.abs(s - med))
    if mad < 1e-9:
        return pd.Series(0.0, index=s.index)
    return 0.6745 * (s - med) / mad


def _iqr_outlier_mask(s: pd.Series) -> pd.Series:
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr <= 0:
        return pd.Series(False, index=s.index)
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < lo) | (s > hi)


def audit_target_series(y: pd.Series, target: str) -> dict:
    s = pd.to_numeric(y, errors="coerce").dropna()
    n = len(s)
    row: dict = {
        "target_name": target,
        "available_count": n,
        "missing_count": int(y.isna().sum()),
    }
    if n == 0:
        return row
    row.update({
        "min": float(s.min()),
        "max": float(s.max()),
        "mean": float(s.mean()),
        "median": float(s.median()),
        "std": float(s.std()),
        "skewness": float(stats.skew(s)),
        "kurtosis": float(stats.kurtosis(s)),
        "n_negative": int((s < 0).sum()),
        "n_zero": int((s == 0).sum()),
        "n_iqr_outliers": int(_iqr_outlier_mask(s).sum()),
        "n_robust_z_outliers": int((_robust_zscore(s).abs() > 3.5).sum()),
    })
    for p in (1, 5, 25, 50, 75, 95, 99):
        row[f"p{p}"] = float(np.percentile(s, p))
    return row


def _plot_target(y: pd.Series, target: str, fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    s = pd.to_numeric(y, errors="coerce").dropna()
    if len(s) < 5:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].hist(s, bins=min(50, max(10, len(s) // 20)), edgecolor="k", alpha=0.7)
    axes[0].set_title(f"{target} — histogramme")
    axes[1].boxplot(s, vert=True)
    axes[1].set_title("boxplot")
    pos = s[s > 0]
    if len(pos) > 5:
        axes[2].hist(np.log1p(pos), bins=30, edgecolor="k", alpha=0.7)
        axes[2].set_title("log1p(y>0)")
    else:
        axes[2].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / f"dist_{target}.png", dpi=100)
    plt.close(fig)


def apply_target_flags(df: pd.DataFrame, target: str, rule: dict) -> pd.DataFrame:
    out = df.copy()
    is_cat = rule.get("outlier_method") == "categorical" or target == "cell_class"
    if target not in out.columns:
        y = pd.Series(np.nan, index=out.index)
        y_num = y
    elif is_cat:
        y = out[target]
        y_num = pd.to_numeric(y, errors="coerce")
    else:
        y = pd.to_numeric(out[target], errors="coerce")
        y_num = y
    out["target_is_missing"] = y.isna()
    out["target_is_negative"] = y_num < 0
    out["target_is_zero"] = y_num == 0

    min_v = rule.get("min_physical_value")
    max_v = rule.get("max_physical_value")
    phys_invalid = pd.Series(False, index=out.index)
    if not is_cat:
        if min_v is not None and pd.notna(min_v):
            phys_invalid |= y_num < float(min_v)
        if max_v is not None and pd.notna(max_v):
            phys_invalid |= y_num > float(max_v)
        if not rule.get("negative_allowed", True):
            phys_invalid |= y_num < 0
    out["target_is_physically_invalid"] = phys_invalid & y.notna()
    out["target_is_extreme_iqr"] = False
    valid = y_num.dropna()
    if not is_cat and len(valid) >= 10:
        mask = _iqr_outlier_mask(valid)
        out.loc[mask.index, "target_is_extreme_iqr"] = mask
    out["target_is_extreme_percentile"] = False
    if not is_cat and len(valid) >= 20:
        p1, p99 = np.percentile(valid, [1, 99])
        out["target_is_extreme_percentile"] = y_num.notna() & ((y_num < p1) | (y_num > p99))
    rz = _robust_zscore(y_num.fillna(y_num.median() if y_num.notna().any() else 0))
    out["target_is_suspicious"] = (
        out["target_is_physically_invalid"]
        | out["target_is_extreme_iqr"]
        | (rz.abs() > 4)
    )
    out["target_needs_unit_check"] = out["target_is_suspicious"] | out["target_is_extreme_percentile"]

    if target == "temp_c" or is_cat:
        out["target_use_for_regression"] = False
    else:
        out["target_use_for_regression"] = y.notna() & ~out["target_is_physically_invalid"]
    out["target_use_for_classification"] = y.notna() & (target != "temp_c")
    if target in ("temp_c", "cell_class"):
        out["target_use_for_ranking"] = False
    else:
        out["target_use_for_ranking"] = y.notna() & ~out["target_is_physically_invalid"]
    return out


def winsorize_series(y: pd.Series, low_p: float = 1, high_p: float = 99) -> pd.Series:
    s = y.copy()
    valid = s.dropna()
    if len(valid) < 10:
        return s
    lo, hi = np.percentile(valid, [low_p, high_p])
    return s.clip(lower=lo, upper=hi)


def build_corrected_versions(df: pd.DataFrame, target: str, rule: dict, out_dir: Path) -> dict[str, int]:
    """Crée raw, valid, winsorized, log, classification, ranking."""
    counts: dict[str, int] = {}
    base_cols = [c for c in df.columns if c != target or c == target]
    sub = df.copy()

    # raw
    raw_path = out_dir / f"target_raw_{target}.csv"
    sub.to_csv(raw_path, index=False)
    counts["raw"] = int(sub[target].notna().sum()) if target in sub.columns else 0

    flags = apply_target_flags(sub, target, rule)
    train_mask = flags["target_use_for_regression"] if target != "cell_class" else flags[target].notna()

    invalid_labeled = flags["target_is_physically_invalid"] & sub[target].notna()
    valid = sub.loc[~invalid_labeled].copy()
    valid_path = out_dir / f"target_valid_{target}.csv"
    valid.to_csv(valid_path, index=False)
    counts["valid"] = int(valid.loc[valid[target].notna(), target].shape[0]) if target in valid.columns else 0

    if target != "cell_class" and target != "temp_c":
        y = pd.to_numeric(sub[target], errors="coerce")
        win = sub.copy()
        win[target] = winsorize_series(y, 1, 99)
        win_path = out_dir / f"target_winsorized_{target}.csv"
        win.to_csv(win_path, index=False)
        counts["winsorized"] = int(win[target].notna().sum())

        log_df = sub.copy()
        log_df[f"{target}_log"] = np.log1p(y.clip(lower=0))
        log_df[target] = log_df[f"{target}_log"]
        log_path = out_dir / f"target_log_{target}.csv"
        log_df.to_csv(log_path, index=False)
        counts["log"] = int(log_df[target].notna().sum())

    # classification terciles
    cls_df = sub.copy()
    yv = pd.to_numeric(cls_df[target], errors="coerce")
    labeled = yv.dropna()
    if len(labeled) >= 9:
        q33, q66 = np.percentile(labeled, [33.33, 66.67])
        cls_df[f"{target}_class"] = pd.cut(
            yv,
            bins=[-np.inf, q33, q66, np.inf],
            labels=["faible", "moyen", "élevé"],
        )
    cls_path = out_dir / f"target_classification_{target}.csv"
    cls_df.to_csv(cls_path, index=False)

    # ranking top/mid/bottom
    rank_df = sub.copy()
    if len(labeled) >= 5:
        p20, p80 = np.percentile(labeled, [20, 80])
        rank_df[f"{target}_rank"] = "middle"
        rank_df.loc[yv <= p20, f"{target}_rank"] = "bottom"
        rank_df.loc[yv >= p80, f"{target}_rank"] = "top"
    rank_path = out_dir / f"target_ranking_{target}.csv"
    rank_df.to_csv(rank_path, index=False)

    # excluded tracking
    excl = sub.loc[flags["target_is_physically_invalid"] & sub[target].notna()]
    if len(excl) > 0:
        excl.to_csv(out_dir / f"target_excluded_invalid_{target}.csv", index=False)

    return counts


def modeling_decision_row(
    target: str,
    counts: dict[str, int],
    flags_summary: dict,
    rule: dict,
) -> dict:
    raw_n = counts.get("raw", 0)
    valid_n = counts.get("valid", 0)
    invalid = flags_summary.get("n_physically_invalid", 0)
    extreme = flags_summary.get("n_extreme_iqr", 0) + flags_summary.get("n_extreme_pct", 0)

    if target == "temp_c":
        decision, priority, task = "à exclure temporairement", "N/A", "input_variable"
        just = "Variable procédé — reclassée comme entrée descriptive, pas cible."
    elif target == "cell_class":
        decision, priority, task = "priorité moyenne", "moyenne", "classification"
        just = "Code classe PE — classification uniquement."
    elif raw_n < 80:
        decision, priority, task = "exploratoire", "basse", "classification+ranking"
        just = f"Effectif faible (n={raw_n}) — privilégier classification/ranking."
    elif valid_n >= 200:
        decision, priority, task = "priorité élevée", "élevée", "regression+classification"
        just = "Effectif suffisant après validation — régression et classification."
    else:
        decision, priority, task = "priorité moyenne", "moyenne", "regression+ranking"
        just = "Effectif modéré — tester winsorisation et ranking."

    return {
        "target_name": target,
        "available_count_raw": raw_n,
        "available_count_after_validation": valid_n,
        "invalid_count": invalid,
        "extreme_count": extreme,
        "recommended_task": task,
        "regression_allowed": target not in ("temp_c", "cell_class"),
        "classification_allowed": target != "temp_c",
        "ranking_allowed": target not in ("temp_c", "cell_class"),
        "log_transform_recommended": rule.get("transformation_recommended") == "log1p",
        "winsorization_recommended": rule.get("action_for_extreme_values", "").startswith("winsorize"),
        "modeling_priority": priority,
        "decision": decision,
        "justification": just,
    }


def run_step14(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    ensure_dirs(
        PHASE2_VALIDATION,
        PHASE2_VALIDATION / "figures",
        PHASE2_VALIDATION / "corrected_target_datasets",
    )
    rules = rules_dataframe()
    rules.to_csv(PHASE2_VALIDATION / "target_validity_rules.csv", index=False)
    rules.to_excel(PHASE2_VALIDATION / "target_validity_rules.xlsx", index=False)

    dist_rows, outlier_rows = [], []
    flag_parts = []
    corrected_dir = PHASE2_VALIDATION / "corrected_target_datasets"
    decision_rows = []

    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        y = df[target]
        dist_rows.append(audit_target_series(y, target))
        s = pd.to_numeric(y, errors="coerce").dropna()
        rule = get_rule(target)
        if len(s) > 0:
            outlier_rows.append({
                "target_name": target,
                "n_iqr": int(_iqr_outlier_mask(s).sum()),
                "n_robust_z": int((_robust_zscore(s).abs() > 3.5).sum()),
                "n_below_min": int((s < float(rule.get("min_physical_value") or -np.inf)).sum()) if rule.get("min_physical_value") is not None else 0,
                "n_above_max": int((s > float(rule.get("max_physical_value") or np.inf)).sum()) if rule.get("max_physical_value") is not None else 0,
            })
        _plot_target(y, target, PHASE2_VALIDATION / "figures")
        flagged = apply_target_flags(df, target, rule)
        flag_parts.append(flagged[[c for c in flagged.columns if c.startswith("target_") or c in df.columns]].assign(_target=target))
        counts = build_corrected_versions(df, target, rule, corrected_dir)
        fs = {
            "n_physically_invalid": int(flagged["target_is_physically_invalid"].sum()),
            "n_extreme_iqr": int(flagged["target_is_extreme_iqr"].sum()),
            "n_extreme_pct": int(flagged["target_is_extreme_percentile"].sum()),
        }
        decision_rows.append(modeling_decision_row(target, counts, fs, rule))

    dist_df = pd.DataFrame(dist_rows)
    outlier_df = pd.DataFrame(outlier_rows)
    dist_df.to_csv(PHASE2_VALIDATION / "target_distribution_summary.csv", index=False)
    outlier_df.to_csv(PHASE2_VALIDATION / "target_outlier_summary.csv", index=False)

    # flags globaux (une ligne par index, colonnes par cible simplifiées)
    flags_global = df.copy()
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        fl = apply_target_flags(df, target, get_rule(target))
        for c in fl.columns:
            if c.startswith("target_"):
                flags_global[f"{target}__{c}"] = fl[c]
    flags_global.to_csv(PHASE2_VALIDATION / "target_quality_flags.csv", index=False)

    summary = []
    for target in TARGET_COLUMNS:
        if target not in df.columns:
            continue
        fl = apply_target_flags(df, target, get_rule(target))
        summary.append({
            "target_name": target,
            "n_rows": len(df),
            "n_labeled": int(df[target].notna().sum()),
            "n_physically_invalid": int(fl["target_is_physically_invalid"].sum()),
            "n_suspicious": int(fl["target_is_suspicious"].sum()),
            "n_regression_ok": int(fl["target_use_for_regression"].sum()),
            "n_classification_ok": int(fl["target_use_for_classification"].sum()),
        })
    pd.DataFrame(summary).to_csv(PHASE2_VALIDATION / "target_quality_flags_summary.csv", index=False)

    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(PHASE2_VALIDATION / "target_modeling_decision_table.csv", index=False)
    decision_df.to_excel(PHASE2_VALIDATION / "target_modeling_decision_table.xlsx", index=False)

    report = [
        "# Rapport de validation des cibles (étape 14)",
        "",
        f"Dataset source : `{TABLES / '04_NAV_feature_engineered.csv'}`",
        "",
        "## Synthèse",
        "",
        dist_df.to_markdown(index=False) if hasattr(dist_df, "to_markdown") else str(dist_df.head()),
        "",
        "## Décisions",
        "",
        decision_df.to_markdown(index=False) if hasattr(decision_df, "to_markdown") else "",
        "",
        "## Notes",
        "- `temp_c` : reclassée variable procédé, hors modélisation cible.",
        "- `cell_class` : codes PE — classification uniquement.",
        "- Règles physiques : **à valider par expert Soleno**.",
        "- Données invalides conservées dans `target_excluded_invalid_*.csv`.",
    ]
    (PHASE2_VALIDATION / "target_validation_report.md").write_text("\n".join(report), encoding="utf-8")
    return dist_df, decision_df, corrected_dir
