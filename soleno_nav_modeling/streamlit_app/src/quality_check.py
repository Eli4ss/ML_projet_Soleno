"""Contrôle qualité des inputs avant prédiction."""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import pandas as pd

from config import APP_ROOT, NAV_ROOT

RULES_PATH = APP_ROOT / "assets" / "input_validity_rules.csv"
if not RULES_PATH.exists():
    RULES_PATH = NAV_ROOT / "streamlit_app" / "assets" / "input_validity_rules.csv"

CORE_FEATURES = ["mi", "hlmi", "density_g_cm3", "carbon_black", "ash", "pp", "onset", "peak", "delta_h"]
OPTIONAL_B = ["supplier_code", "location", "description_fr", "reception_year"]


@dataclass
class QualityReport:
    completeness_pct: float
    quality_level: str  # élevée | moyenne | faible
    missing_features: list[str] = field(default_factory=list)
    out_of_range: list[str] = field(default_factory=list)
    suspicious: list[str] = field(default_factory=list)
    derived_issues: list[str] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def summary_text(self) -> str:
        parts = [f"Qualité des données : **{self.quality_level}**."]
        parts.append(f"Complétude : {self.completeness_pct:.0f} % des features clés renseignées.")
        if self.missing_features:
            parts.append(f"Manquants : {', '.join(self.missing_features[:8])}.")
        if self.out_of_range:
            parts.append(f"Hors plage : {', '.join(self.out_of_range[:5])}.")
        if self.derived_issues:
            parts.append(f"Dérivées : {'; '.join(self.derived_issues[:3])}.")
        return " ".join(parts)


@lru_cache(maxsize=1)
def load_input_rules() -> pd.DataFrame:
    if RULES_PATH.exists():
        return pd.read_csv(RULES_PATH)
    return pd.DataFrame(
        columns=["feature", "min_plausible", "max_plausible", "unit", "optional_zero"]
    )


def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, str) and val.strip() == "":
        return True
    try:
        return bool(pd.isna(val))
    except Exception:
        return False


def check_lot_quality(input_row: dict, model_version: str = "A") -> QualityReport:
    rules = load_input_rules()
    rules_idx = rules.set_index("feature") if not rules.empty else {}

    check_feats = list(CORE_FEATURES)
    if model_version == "B":
        check_feats += [c for c in OPTIONAL_B if c not in ("reception_year",)]

    present = 0
    missing = []
    out_of_range = []
    suspicious = []
    derived_issues = []

    for feat in check_feats:
        val = input_row.get(feat)
        if _is_missing(val):
            missing.append(feat)
            continue
        present += 1
        try:
            v = float(val)
        except (TypeError, ValueError):
            suspicious.append(f"{feat} (non numérique)")
            continue

        if feat in rules_idx.index:
            r = rules_idx.loc[feat]
            opt_zero = str(r.get("optional_zero", False)).lower() in ("true", "1", "yes")
            if v == 0 and opt_zero:
                continue
            lo, hi = r.get("min_plausible"), r.get("max_plausible")
            unit = r.get("unit", "")
            if pd.notna(lo) and v < float(lo):
                out_of_range.append(f"{feat}={v} < {lo} {unit}")
            if pd.notna(hi) and v > float(hi):
                out_of_range.append(f"{feat}={v} > {hi} {unit}")

    mi = input_row.get("mi")
    hlmi = input_row.get("hlmi")
    try:
        mi_f = float(mi) if not _is_missing(mi) else np.nan
        hlmi_f = float(hlmi) if not _is_missing(hlmi) else np.nan
        if pd.notna(mi_f) and mi_f == 0:
            derived_issues.append("FRR non calculable (MI = 0)")
        elif pd.notna(mi_f) and pd.notna(hlmi_f) and mi_f > 0:
            frr = hlmi_f / mi_f
            if frr > 500:
                suspicious.append(f"FRR={frr:.1f} très élevé")
    except Exception:
        pass

    completeness = 100.0 * present / max(len(check_feats), 1)
    if completeness >= 75 and not out_of_range and not derived_issues:
        level = "élevée"
    elif completeness >= 50 or (out_of_range and completeness >= 40):
        level = "moyenne"
    else:
        level = "faible"

    messages = []
    if missing:
        messages.append(f"Features manquantes : {', '.join(missing)}")
    if out_of_range:
        messages.append("Valeurs hors plage plausible détectées")
    if derived_issues:
        messages.append("Features dérivées impossibles ou incertaines")

    return QualityReport(
        completeness_pct=completeness,
        quality_level=level,
        missing_features=missing,
        out_of_range=out_of_range,
        suspicious=suspicious,
        derived_issues=derived_issues,
        messages=messages,
    )
