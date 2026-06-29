"""Étape 2 — Table de décision des features."""
from __future__ import annotations

import pandas as pd

from .columns import FEATURE_METADATA
from .config import REPORTS, TABLES
from .utils import ensure_dirs, load_raw_excel, standardize_columns


def run_step02(df: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_dirs(REPORTS, TABLES)
    if df is None:
        df = standardize_columns(load_raw_excel())

    rows = []
    for meta in FEATURE_METADATA:
        std = meta["standard"]
        if std not in df.columns:
            avail = 0
            miss_rate = 100.0
            uq = 0
        else:
            avail = int(df[std].notna().sum())
            miss_rate = round(df[std].isna().mean() * 100, 2)
            uq = int(df[std].nunique(dropna=True))

        leakage = "élevé" if meta["role"] == "id" else ("moyen" if meta["role"] == "target" else "faible")
        if meta["role"] == "target":
            decision = "exclure_entrée"
            just = "Cible de prédiction — risque de fuite si utilisée en entrée."
        elif meta["role"] == "id":
            decision = "exclure"
            just = "Identifiant pur — non généralisable."
        elif meta["A"] and meta["B"]:
            decision = "inclure_A_et_B"
            just = "Mesure matière ou dérivée ; utile scientifique et opérationnel."
        elif meta["B"] is True:
            decision = "inclure_B_uniquement"
            just = "Contexte fournisseur/grade/date — modèle opérationnel uniquement."
        elif meta["A"]:
            decision = "inclure_A_et_B"
            just = "Propriété physique mesurée en laboratoire."
        else:
            decision = "exclure"
            just = "Métadonnée administrative ou redondante."

        rows.append({
            "raw_feature_name": meta["raw"],
            "standard_feature_name": std,
            "family_pspp": meta["family"],
            "data_type": meta["dtype"],
            "role": meta["role"],
            "use_in_material_model_A": bool(meta["A"]),
            "use_in_operational_model_B": bool(meta["B"]) if meta["B"] is not False else False,
            "missing_rate": miss_rate,
            "available_count": avail,
            "unique_count": uq,
            "leakage_risk": leakage,
            "transformation_needed": "oui" if meta["dtype"] in ("category", "text", "datetime") else "non",
            "decision": decision,
            "justification": just,
        })

    fdt = pd.DataFrame(rows)
    fdt.to_csv(TABLES / "02_feature_decision_table.csv", index=False)
    fdt.to_excel(TABLES / "02_feature_decision_table.xlsx", index=False)

    summary = [
        "# Synthèse des rôles de features",
        "",
        f"**Features modèle A (matière) :** {fdt['use_in_material_model_A'].sum()}",
        f"**Features modèle B (opérationnel) :** {fdt['use_in_operational_model_B'].sum()}",
        f"**Cibles :** {(fdt['role'] == 'target').sum()}",
        f"**Identifiants exclus :** {(fdt['role'] == 'id').sum()}",
        "",
        "Voir `02_feature_decision_table.xlsx` pour le détail par colonne.",
    ]
    (REPORTS / "02_feature_roles_summary.md").write_text("\n".join(summary), encoding="utf-8")
    return fdt
