"""Orchestration qualité → prédiction → confiance → recommandation."""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.confidence import compute_confidence
from src.pipeline_bridge import get_ood_reference
from src.prediction import predict_single
from src.quality_check import QualityReport, check_lot_quality
from src.recommendations import build_global_recommendation, recommend_for_target
from src.target_readiness import readiness_for_target

TARGET_THRESHOLDS = {
    "oit_min": {"good": 30, "bad": 10},
    "ncls": {"good": 100, "bad": 40},
    "izod": {"good": 80, "bad": 30},
    "traction": {"good": 20, "bad": 10},
}


def _row_dict(full_row) -> dict:
    if isinstance(full_row, dict):
        return full_row
    return full_row.to_dict()


def _ood_trust(input_row: dict, target: str, model_version: str) -> float | None:
    try:
        ood_mod = get_ood_reference()
        report = ood_mod.check_ood(
            input_row,
            target,
            model_version,
            ood_mod.load_reference(),
            ood_mod.load_risky_suppliers_by_target(),
        )
        return float(report.trust_score)
    except Exception:
        return None


def _safe_predict(input_row: dict, target: str, version: str):
    res = predict_single(input_row, target, version)
    if res.get("status") == "OK":
        return res["prediction"], True
    return None, False


def predict_lot_with_metadata(
    full_row: dict | pd.Series,
    targets: list[str],
    model_version: str = "A",
    compare_ab: bool = False,
) -> tuple[pd.DataFrame, str, QualityReport]:
    input_dict = _row_dict(full_row)
    quality = check_lot_quality(input_dict, model_version)

    rows = []
    for target in targets:
        pred_a, ok_a = _safe_predict(input_dict, target, "A")
        pred_b, ok_b = _safe_predict(input_dict, target, "B") if compare_ab else (None, False)
        pred, ok = _safe_predict(input_dict, target, model_version)

        trust = _ood_trust(input_dict, target, model_version)
        conf_score, conf_level = compute_confidence(target, quality, trust, ok)

        th = TARGET_THRESHOLDS.get(target, {})
        rec = recommend_for_target(
            target,
            pred,
            conf_level,
            quality,
            th.get("good"),
            th.get("bad"),
        )

        gap = None
        comment = ""
        if compare_ab and pred_a is not None and pred_b is not None:
            try:
                gap = float(pred_b) - float(pred_a)
                if abs(gap) > 10:
                    comment = "écart important — prudence"
                elif abs(gap) > 3:
                    comment = "contexte opérationnel influence la prédiction"
                else:
                    comment = "modèles cohérents"
            except (TypeError, ValueError):
                comment = "comparaison non numérique"

        ready = readiness_for_target(target)
        rows.append(
            {
                "target": target,
                "prediction": pred,
                "pred_model_a": pred_a,
                "pred_model_b": pred_b,
                "gap_a_b": gap,
                "ab_comment": comment,
                "confidence_score": conf_score,
                "confidence_level": conf_level,
                "recommendation": rec,
                "data_quality": quality.quality_level,
                "readiness_status": ready.get("status", ""),
                "prediction_status": "ok" if ok else "failed",
                "ood_trust": trust,
            }
        )

    df = pd.DataFrame(rows)
    global_rec = build_global_recommendation(rows)
    return df, global_rec, quality


def enrich_batch_predictions(
    df_batch: pd.DataFrame,
    targets: list[str],
    model_version: str,
) -> pd.DataFrame:
    """Ajoute confiance, statut et recommandation par cible."""
    from src.feature_engineering import add_derived_features

    out = df_batch.copy()
    out["generated_at"] = datetime.now(timezone.utc).isoformat()
    if "model_version" not in out.columns:
        out["model_version"] = model_version

    for target in targets:
        pred_col = f"predicted_{target}"
        if pred_col not in out.columns:
            pred_col = f"pred_{target}"

        confs, levels, recs, stats = [], [], [], []
        for _, row in out.iterrows():
            row_d = add_derived_features(pd.DataFrame([row.to_dict()])).iloc[0].to_dict()
            q = check_lot_quality(row_d, model_version)
            pred = row.get(pred_col, row.get(target))
            trust = _ood_trust(row_d, target, model_version)
            ok = pred is not None and str(pred) not in ("nan", "None", "<NA>")
            cs, cl = compute_confidence(target, q, trust, ok)
            th = TARGET_THRESHOLDS.get(target, {})
            rec = recommend_for_target(target, pred, cl, q, th.get("good"), th.get("bad"))
            confs.append(cs)
            levels.append(cl)
            recs.append(rec)
            stats.append("ok" if ok else "failed")

        out[f"confidence_{target}"] = confs
        out[f"confidence_level_{target}"] = levels
        out[f"recommendation_{target}"] = recs
        out[f"status_{target}"] = stats

    out["prediction_status"] = out.apply(
        lambda r: "ok"
        if any(str(r.get(f"status_{t}", "")) == "ok" for t in targets)
        else "partial",
        axis=1,
    )
    out["recommendation"] = out.apply(lambda r: _batch_row_summary(r, targets), axis=1)
    return out


def _batch_row_summary(row, targets):
    recs = [str(row.get(f"recommendation_{t}", "")) for t in targets]
    if any("intéressant" in x.lower() for x in recs):
        return "Lot intéressant — confirmer cibles à faible confiance"
    if any("labo" in x.lower() or "screening" in x.lower() for x in recs):
        return "Test laboratoire recommandé"
    if any("insuffisant" in x.lower() for x in recs):
        return "Données insuffisantes"
    return "Prédiction prudente"


def export_column_rename(df: pd.DataFrame, targets: list[str]) -> pd.DataFrame:
    """Alias lisibles pour export Soleno."""
    out = df.copy()
    alias = {"oit_min": "oit", "pct_elongation": "allongement"}
    for t in targets:
        short = alias.get(t, t)
        for suffix in ("predicted_", "pred_"):
            old = f"{suffix}{t}"
            if old in out.columns:
                out.rename(columns={old: f"pred_{short}"}, inplace=True)
        if f"confidence_{t}" in out.columns:
            out.rename(columns={f"confidence_{t}": f"confidence_{short}"}, inplace=True)
    return out
