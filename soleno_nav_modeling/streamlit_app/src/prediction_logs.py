"""Journal local des prédictions."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import NAV_ROOT

LOG_DIR = NAV_ROOT / "outputs" / "prediction_logs"
LOG_FILE = LOG_DIR / "prediction_history.csv"

COLUMNS = [
  "timestamp",
  "lot_id",
  "target",
  "model_version",
  "features_json",
  "prediction",
  "confidence_score",
  "confidence_level",
  "recommendation",
  "data_quality_level",
  "prediction_status",
  "mode",
]


def append_prediction_log(
  lot_id: str,
  target: str,
  model_version: str,
  features: dict,
  prediction,
  confidence_score: float,
  confidence_level: str,
  recommendation: str,
  data_quality_level: str,
  prediction_status: str = "ok",
  mode: str = "single",
) -> None:
  LOG_DIR.mkdir(parents=True, exist_ok=True)
  row = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "lot_id": lot_id,
    "target": target,
    "model_version": model_version,
    "features_json": json.dumps(features, default=str)[:4000],
    "prediction": prediction,
    "confidence_score": confidence_score,
    "confidence_level": confidence_level,
    "recommendation": recommendation,
    "data_quality_level": data_quality_level,
    "prediction_status": prediction_status,
    "mode": mode,
  }
  df = pd.DataFrame([row])
  if LOG_FILE.exists():
    df.to_csv(LOG_FILE, mode="a", header=False, index=False)
  else:
    df.to_csv(LOG_FILE, index=False)
