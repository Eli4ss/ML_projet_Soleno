"""Références OOD et fournisseurs pour alertes Streamlit (phase 3)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import GROUP_COLUMNS, PHASE3_ERRORS, PROJECT_ROOT, TABLES, TARGET_COLUMNS_MODELING

REFERENCE_PATH = PROJECT_ROOT / "outputs" / "app" / "ood_reference.parquet"
META_PATH = PROJECT_ROOT / "outputs" / "app" / "ood_meta.json"


NUMERIC_FEATURES = [
    "mi", "hlmi", "density_g_cm3", "carbon_black", "ash",
    "onset", "peak", "frr", "charge_total", "fluidity_g_10min",
]


@dataclass
class OODAlert:
    level: str  # danger | warning | info
    category: str
    message: str


@dataclass
class OODReport:
    alerts: list[OODAlert] = field(default_factory=list)
    trust_score: float = 1.0
    summary: str = ""

    @property
    def has_critical(self) -> bool:
        return any(a.level == "danger" for a in self.alerts)


def _norm(s: str) -> str:
    return str(s).strip().upper().replace(" ", "")


def build_reference(df: pd.DataFrame) -> dict:
    """Construit les ensembles et stats de référence depuis le NAV engineeré."""
    ref: dict = {}

    for key, col in GROUP_COLUMNS.items():
        if col in df.columns:
            vals = df[col].dropna().astype(str).str.strip()
            ref[f"known_{key}"] = sorted(vals[vals != ""].unique().tolist())

    if "supplier_name" in df.columns:
        names = df["supplier_name"].dropna().astype(str).str.strip().unique().tolist()
        ref["known_supplier_names"] = sorted(names)

    stats = {}
    for c in NUMERIC_FEATURES:
        if c not in df.columns:
            continue
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s) < 20:
            continue
        stats[c] = {
            "p01": float(np.percentile(s, 1)),
            "p05": float(np.percentile(s, 5)),
            "p50": float(np.percentile(s, 50)),
            "p95": float(np.percentile(s, 95)),
            "p99": float(np.percentile(s, 99)),
        }
    ref["numeric_stats"] = stats

    if "supplier_code" in df.columns:
        counts = df["supplier_code"].astype(str).str.strip().value_counts()
        ref["supplier_counts"] = counts[counts.index != ""].head(500).to_dict()

    return ref


def load_risky_suppliers_by_target() -> dict[str, set[str]]:
    """Fournisseurs à fort MAE en GroupKFold (phase 3)."""
    path = PHASE3_ERRORS / "worst_groups_by_mae.csv"
    if not path.exists():
        path = PHASE3_ERRORS / "error_by_supplier_grade_origin_period.csv"
    if not path.exists():
        return {}

    err = pd.read_csv(path)
    sub = err[
        (err["dimension_type"] == "supplier")
        & (err["validation_scheme"] == "group_supplier")
    ]
    if sub.empty:
        return {}

    out: dict[str, set[str]] = {}
    for target in TARGET_COLUMNS_MODELING:
        tsub = sub[sub["target"] == target]
        if tsub.empty:
            continue
        # Top 15 MAE avec au moins 3 prédictions dans le fold
        top = tsub[tsub["n_predictions"] >= 3].nlargest(15, "mae")
        out[target] = set(top["dimension_value"].astype(str))
    return out


def load_reference(df: pd.DataFrame | None = None) -> dict:
    if df is not None:
        return build_reference(df)
    fe_path = TABLES / "04_NAV_feature_engineered.csv"
    if not fe_path.exists():
        return {}
    df = pd.read_csv(fe_path, low_memory=False)
    return build_reference(df)


def check_ood(
    input_row: dict,
    target: str,
    model_version: str,
    ref: dict | None = None,
    risky_by_target: dict[str, set[str]] | None = None,
) -> OODReport:
    ref = ref or load_reference()
    risky_by_target = risky_by_target or load_risky_suppliers_by_target()
    alerts: list[OODAlert] = []

    supplier = str(input_row.get("supplier_code") or input_row.get("supplier_name") or "").strip()
    location = str(input_row.get("location") or "").strip()
    grade = str(input_row.get("description_fr") or "").strip()

    known_sup = set(ref.get("known_supplier", [])) | set(ref.get("known_supplier_names", []))
    known_sup_norm = {_norm(x) for x in known_sup}

    if model_version == "B" and supplier:
        if _norm(supplier) not in known_sup_norm:
            alerts.append(OODAlert(
                "danger",
                "nouveau_fournisseur",
                f"Fournisseur « {supplier} » absent du NAV d'entraînement — prédiction peu fiable (GroupKFold fournisseur dégradé).",
            ))
        else:
            counts = ref.get("supplier_counts", {})
            n = counts.get(supplier, counts.get(supplier.upper(), 0))
            if n and n < 10:
                alerts.append(OODAlert(
                    "warning",
                    "fournisseur_rare",
                    f"Fournisseur connu mais rare (n={int(n)} lots) — généralisation incertaine.",
                ))
            risky = risky_by_target.get(target, set())
            sup_n = _norm(supplier)
            if supplier in risky or sup_n in {_norm(s) for s in risky}:
                alerts.append(OODAlert(
                    "warning",
                    "fournisseur_risque",
                    f"Fournisseur identifié comme à fort MAE pour « {target} » en validation GroupKFold.",
                ))

    if model_version == "B" and location:
        known_loc = {_norm(x) for x in ref.get("known_origin", [])}
        if _norm(location) not in known_loc:
            alerts.append(OODAlert(
                "warning",
                "nouvelle_origine",
                f"Site / location « {location} » hors des {len(known_loc)} sites connus.",
            ))

    if model_version == "B" and grade:
        known_gr = {str(x).strip() for x in ref.get("known_grade", [])}
        if grade not in known_gr:
            alerts.append(OODAlert(
                "warning",
                "nouveau_grade",
                f"Description matière « {grade[:80]}{'…' if len(grade) > 80 else ''} » non vue ou nouvelle.",
            ))

    optional_zero = {"onset", "peak", "carbon_black", "ash", "fluidity_g_10min"}

    stats = ref.get("numeric_stats", {})
    for feat, bounds in stats.items():
        val = input_row.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        try:
            v = float(val)
        except (TypeError, ValueError):
            continue
        if feat in optional_zero and v == 0:
            continue
        p01, p99 = bounds.get("p01"), bounds.get("p99")
        p05, p95 = bounds.get("p05"), bounds.get("p95")
        if p01 is not None and v < p01:
            alerts.append(OODAlert(
                "danger",
                "hors_distribution",
                f"{feat}={v:.4g} sous le percentile 1 % NAV ({p01:.4g}).",
            ))
        elif p99 is not None and v > p99:
            alerts.append(OODAlert(
                "danger",
                "hors_distribution",
                f"{feat}={v:.4g} au-dessus du percentile 99 % NAV ({p99:.4g}).",
            ))
        elif p05 is not None and v < p05:
            alerts.append(OODAlert(
                "warning",
                "hors_distribution",
                f"{feat}={v:.4g} sous le percentile 5 % ({p05:.4g}).",
            ))
        elif p95 is not None and v > p95:
            alerts.append(OODAlert(
                "warning",
                "hors_distribution",
                f"{feat}={v:.4g} au-dessus du percentile 95 % ({p95:.4g}).",
            ))

    missing_core = [f for f in ("mi", "density_g_cm3") if not input_row.get(f) or pd.isna(input_row.get(f))]
    if missing_core:
        alerts.append(OODAlert(
            "warning",
            "donnees_manquantes",
            f"Variables cœur manquantes : {', '.join(missing_core)}.",
        ))

    penalty = sum(0.25 if a.level == "danger" else 0.1 for a in alerts)
    trust = max(0.0, min(1.0, 1.0 - penalty))

    if trust >= 0.75:
        summary = "Confiance élevée — profil proche du NAV d'entraînement."
    elif trust >= 0.45:
        summary = "Confiance moyenne — vérifier les alertes avant décision."
    else:
        summary = "Confiance faible — lot hors distribution ou nouveau contexte."

    return OODReport(alerts=alerts, trust_score=trust, summary=summary)
