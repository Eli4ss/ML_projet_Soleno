"""Étape 1 — Audit des données NAV."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import REPORTS, TABLES
from .utils import ensure_dirs, load_raw_excel, standardize_columns


def run_step01() -> pd.DataFrame:
    ensure_dirs(REPORTS, TABLES)
    df_raw = load_raw_excel()
    df = standardize_columns(df_raw)

    n_rows, n_cols = df.shape
    lines = [
        "# Rapport d'audit des données NAV — Soleno",
        "",
        f"**Source :** `data/raw/Resine.xlsx`",
        f"**Dimensions :** {n_rows:,} lignes × {n_cols} colonnes",
        "",
        "## Synthèse",
        "",
        "Jeu de données NAV (Business Central) des lots de résine recyclée avec mesures",
        "rheologiques, densité, charges, DSC et propriétés mécaniques/oxydatives.",
        "",
        "### Colonnes principales",
        "",
    ]
    for c in df.columns:
        miss = df[c].isna().mean() * 100
        lines.append(f"- `{c}` : {df[c].dtype}, manquants {miss:.1f}%")

    lines.extend([
        "",
        "## Cibles identifiées",
        "",
        "Propriétés cibles avec taux de disponibilité variable : OIT, NCLS, UCLS, IZod,",
        "traction, allongement, flexion, cell class, température.",
        "",
        "## Recommandations",
        "",
        "1. Conserver toutes les lignes ; utiliser des flags qualité.",
        "2. Ne pas utiliser les identifiants (lot, PO) comme features.",
        "3. Modéliser chaque cible séparément (fort déséquilibre des mesures).",
        "4. Valider les seuils industriels de classification avec l'équipe R&D.",
        "",
    ])

    (REPORTS / "01_NAV_data_audit_report.md").write_text("\n".join(lines), encoding="utf-8")

    inv = pd.DataFrame({
        "column_name": df.columns,
        "dtype": [str(df[c].dtype) for c in df.columns],
        "non_null_count": [df[c].notna().sum() for c in df.columns],
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_rate_pct": [df[c].isna().mean() * 100 for c in df.columns],
    })
    inv.to_csv(TABLES / "01_NAV_columns_inventory.csv", index=False)

    miss = pd.DataFrame({
        "column_name": df.columns,
        "missing_count": [df[c].isna().sum() for c in df.columns],
        "missing_rate_pct": [round(df[c].isna().mean() * 100, 2) for c in df.columns],
    }).sort_values("missing_rate_pct", ascending=False)
    miss.to_csv(TABLES / "01_NAV_missing_values_report.csv", index=False)

    uniq = pd.DataFrame({
        "column_name": df.columns,
        "unique_count": [df[c].nunique(dropna=True) for c in df.columns],
        "sample_values": [str(df[c].dropna().head(3).tolist())[:200] for c in df.columns],
    })
    uniq.to_csv(TABLES / "01_NAV_unique_values_report.csv", index=False)

    return df
