"""Configuration centrale des niveaux de maturité — dérivée de la phase 5."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from v2_config.settings import MATURITY_LEVELS, MATURITY_OVERRIDES_YAML
from config import PATHS, TARGETS  # streamlit_app config

_DEPLOYMENT_TO_LEVEL = {
    "deploy": "pilot",
    "caution": "pilot",
    "lab_only": "experimental",
    "blocked": "experimental",
}


def _load_overrides() -> dict[str, dict[str, Any]]:
    if not MATURITY_OVERRIDES_YAML.exists():
        return {}
    try:
        data = yaml.safe_load(MATURITY_OVERRIDES_YAML.read_text(encoding="utf-8")) or {}
        return dict(data.get("targets") or {})
    except Exception:
        return {}


@lru_cache(maxsize=1)
def load_deployment_table() -> pd.DataFrame:
    path = PATHS.get("deployment_status")
    if path and Path(path).exists():
        return pd.read_csv(path)
    rec = PATHS.get("recommended_model")
    if rec and Path(rec).exists():
        return pd.read_csv(rec)
    return pd.DataFrame()


def _recommended_row(target: str) -> pd.Series | None:
    rec_path = PATHS.get("recommended_model")
    if not rec_path or not Path(rec_path).exists():
        df = load_deployment_table()
        if df.empty:
            return None
        sub = df[df["target"].astype(str) == target]
        if sub.empty:
            return None
        if "recommended_model" in sub.columns:
            best = sub.iloc[0].get("recommended_model", "A")
            match = sub[sub.get("model_version", sub.get("model_A_or_B", pd.Series())) == best]
            return match.iloc[0] if not match.empty else sub.iloc[0]
        return sub.iloc[0]
    df = pd.read_csv(rec_path)
    sub = df[df["target"].astype(str) == target]
    return sub.iloc[0] if not sub.empty else None


def derive_target_maturity(target: str) -> dict[str, Any]:
    """Retourne la config maturité pour une cible."""
    overrides = _load_overrides()
    base = {
        "target": target,
        "label": TARGETS.get(target, {}).get("label", target),
        "unit": TARGETS.get(target, {}).get("unit", ""),
        "level": "experimental",
        "level_label": MATURITY_LEVELS["experimental"]["label"],
        "allowed_for_prediction": False,
        "requires_lab_confirmation": True,
        "deployment_status": "unknown",
        "deployment_label": "Non évalué",
        "n_samples": None,
        "r2_random": None,
        "r2_group_supplier": None,
        "mae_degradation_ratio": None,
        "recommended_model": None,
        "model_version": None,
        "tier": None,
        "polymer_rationale": "",
        "status_reason": "",
        "source": "défaut",
    }

    row = _recommended_row(target)
    if row is not None:
        status = str(row.get("deployment_status", "caution"))
        level_key = _DEPLOYMENT_TO_LEVEL.get(status, "experimental")
        base.update({
            "level": level_key,
            "level_label": MATURITY_LEVELS[level_key]["label"],
            "allowed_for_prediction": level_key == "pilot",
            "requires_lab_confirmation": True,
            "deployment_status": status,
            "deployment_label": row.get("deployment_label", status),
            "n_samples": row.get("n_samples"),
            "r2_random": row.get("r2_random"),
            "r2_group_supplier": row.get("r2_group_supplier"),
            "mae_degradation_ratio": row.get("mae_degradation_ratio"),
            "recommended_model": row.get("recommended_model"),
            "model_version": row.get("model_version"),
            "tier": row.get("tier"),
            "polymer_rationale": row.get("polymer_rationale", ""),
            "status_reason": row.get("status_reason", ""),
            "source": str(PATHS.get("recommended_model", "")),
        })

    if target in overrides:
        ov = overrides[target]
        if ov.get("level") in MATURITY_LEVELS:
            base["level"] = ov["level"]
            base["level_label"] = MATURITY_LEVELS[ov["level"]]["label"]
        if "allowed_for_prediction" in ov:
            base["allowed_for_prediction"] = bool(ov["allowed_for_prediction"])
        if "requires_lab_confirmation" in ov:
            base["requires_lab_confirmation"] = bool(ov["requires_lab_confirmation"])
        if ov.get("note"):
            base["status_reason"] = f"{base.get('status_reason', '')} | Override: {ov['note']}".strip(" |")
        base["source"] = str(MATURITY_OVERRIDES_YAML)

    return base


@lru_cache(maxsize=1)
def build_target_maturity_table() -> pd.DataFrame:
    from config import MVP_TARGETS

    rows = [derive_target_maturity(t) for t in MVP_TARGETS]
    return pd.DataFrame(rows)


def pilot_targets() -> list[str]:
    df = build_target_maturity_table()
    return df[df["allowed_for_prediction"] == True]["target"].tolist()  # noqa: E712


def count_by_level() -> dict[str, int]:
    df = build_target_maturity_table()
    counts = {k: 0 for k in MATURITY_LEVELS}
    for level in df["level"]:
        if level in counts:
            counts[level] += 1
    return counts


def explain_negative_r2(r2: float | None) -> str:
    if r2 is None or (isinstance(r2, float) and pd.isna(r2)):
        return "Métrique non disponible pour ce protocole."
    if float(r2) < 0:
        return (
            "Sur ce protocole, le modèle fait moins bien qu'une prédiction basée "
            "sur la moyenne du jeu de test."
        )
    return ""
