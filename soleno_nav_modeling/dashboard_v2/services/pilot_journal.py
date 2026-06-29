"""Journal du pilote contrôlé — validation terrain."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from v2_config.settings import PILOT_JOURNAL_DIR, PILOT_JOURNAL_FILE

PILOT_COLUMNS = [
    "prediction_id",
    "lot_id",
    "prediction_date",
    "target",
    "model_used",
    "model_version",
    "inputs_json",
    "predicted_value",
    "prediction_unit",
    "lab_result",
    "lab_result_date",
    "abs_error",
    "rel_error_pct",
    "supplier_code",
    "grade",
    "site",
    "user_comment",
    "actual_decision",
    "maturity_level",
    "data_quality_level",
    "quality_index",
    "prediction_status",
    "updated_at",
]


def _ensure_dir() -> None:
    PILOT_JOURNAL_DIR.mkdir(parents=True, exist_ok=True)


def load_pilot_journal() -> pd.DataFrame:
    _ensure_dir()
    if not PILOT_JOURNAL_FILE.exists():
        return pd.DataFrame(columns=PILOT_COLUMNS)
    try:
        df = pd.read_csv(PILOT_JOURNAL_FILE)
        for col in PILOT_COLUMNS:
            if col not in df.columns:
                df[col] = pd.NA
        return df[PILOT_COLUMNS]
    except Exception:
        return pd.DataFrame(columns=PILOT_COLUMNS)


def append_pilot_prediction(entry: dict) -> str:
    _ensure_dir()
    now = datetime.now(timezone.utc).isoformat()
    pid = entry.get("prediction_id") or f"PILOT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
    row = {c: entry.get(c) for c in PILOT_COLUMNS}
    row["prediction_id"] = pid
    row["prediction_date"] = row.get("prediction_date") or now
    row["updated_at"] = now
    if isinstance(row.get("inputs_json"), dict):
        row["inputs_json"] = json.dumps(row["inputs_json"], default=str)[:8000]

    df = pd.DataFrame([row])
    if PILOT_JOURNAL_FILE.exists():
        df.to_csv(PILOT_JOURNAL_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(PILOT_JOURNAL_FILE, index=False)
    return pid


def update_lab_feedback(
    prediction_id: str,
    lab_result: float,
    *,
    user_comment: str = "",
    actual_decision: str = "",
) -> bool:
    df = load_pilot_journal()
    if df.empty or prediction_id not in df["prediction_id"].astype(str).values:
        return False

    idx = df.index[df["prediction_id"].astype(str) == str(prediction_id)][0]
    pred = pd.to_numeric(df.at[idx, "predicted_value"], errors="coerce")
    lab = float(lab_result)
    df.at[idx, "lab_result"] = lab
    df.at[idx, "lab_result_date"] = datetime.now(timezone.utc).isoformat()
    if pd.notna(pred):
        df.at[idx, "abs_error"] = abs(lab - float(pred))
        if float(pred) != 0:
            df.at[idx, "rel_error_pct"] = abs(lab - float(pred)) / abs(float(pred)) * 100
    if user_comment:
        df.at[idx, "user_comment"] = user_comment
    if actual_decision:
        df.at[idx, "actual_decision"] = actual_decision
    df.at[idx, "updated_at"] = datetime.now(timezone.utc).isoformat()
    df.to_csv(PILOT_JOURNAL_FILE, index=False)
    return True


def pilot_summary_stats(df: pd.DataFrame | None = None) -> dict:
    df = df if df is not None else load_pilot_journal()
    if df.empty:
        return {
            "n_predictions": 0,
            "n_lab_received": 0,
            "n_pending": 0,
            "mae_field": None,
            "mean_rel_error": None,
        }
    lab_mask = pd.to_numeric(df["lab_result"], errors="coerce").notna()
    abs_err = pd.to_numeric(df.loc[lab_mask, "abs_error"], errors="coerce")
    rel_err = pd.to_numeric(df.loc[lab_mask, "rel_error_pct"], errors="coerce")
    return {
        "n_predictions": len(df),
        "n_lab_received": int(lab_mask.sum()),
        "n_pending": int((~lab_mask).sum()),
        "mae_field": float(abs_err.mean()) if not abs_err.empty else None,
        "mean_rel_error": float(rel_err.mean()) if not rel_err.empty else None,
    }


def is_test_prediction(row: pd.Series) -> bool:
    """Prédiction de test / démo (lot par défaut du formulaire pilote)."""
    lot = str(row.get("lot_id", "")).strip().lower()
    if lot in {"lot_pilote", "lot_manuel", "lot_test", "test"}:
        return True
    if lot.startswith("test-") or lot.startswith("test_"):
        return True
    pid = str(row.get("prediction_id", ""))
    if pid.startswith("TEST-"):
        return True
    comment = str(row.get("user_comment", "") or "").lower()
    if "test" in comment and ("supprimer" in comment or "demo" in comment or "démo" in comment):
        return True
    return False


def delete_pilot_predictions(prediction_ids: list[str]) -> int:
    """Supprime des entrées par prediction_id. Retourne le nombre supprimé."""
    if not prediction_ids:
        return 0
    df = load_pilot_journal()
    if df.empty:
        return 0
    ids = {str(i) for i in prediction_ids}
    before = len(df)
    df = df[~df["prediction_id"].astype(str).isin(ids)]
    removed = before - len(df)
    if removed:
        df.to_csv(PILOT_JOURNAL_FILE, index=False)
    return removed


def delete_test_predictions() -> int:
    """Supprime toutes les prédictions identifiées comme tests."""
    df = load_pilot_journal()
    if df.empty:
        return 0
    test_ids = df.loc[df.apply(is_test_prediction, axis=1), "prediction_id"].astype(str).tolist()
    return delete_pilot_predictions(test_ids)
