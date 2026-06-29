"""Prototype Streamlit — prédiction + alertes OOD (phase 3)."""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import METRICS, TABLES, TARGET_COLUMNS
from src.modeling import build_inference_x, get_pipeline_feature_columns
from src.ood_reference import OODAlert, check_ood, load_reference, load_risky_suppliers_by_target

st.set_page_config(page_title="Soleno NAV — Prédiction résine", layout="wide")
st.title("Soleno — Prédiction des propriétés (résine recyclée)")

registry_path = ROOT / "outputs" / "19_updated_model_registry" / "updated_best_model_registry.csv"
if not registry_path.exists():
    registry_path = METRICS / "12_preliminary_best_model_registry.csv"
reliability_path = ROOT / "outputs" / "18_uncertainty_after_cleaning" / "reliability_scores.csv"
if not reliability_path.exists():
    reliability_path = TABLES / "11_reliability_scores.csv"
importance_path = ROOT / "outputs" / "17_explainability_after_cleaning" / "feature_importance_by_target.csv"
if not importance_path.exists():
    importance_path = TABLES / "10_feature_importance_by_target.csv"


@st.cache_data
def load_registry():
    if registry_path.exists():
        return pd.read_csv(registry_path)
    return pd.DataFrame()


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def get_ood_reference():
    return load_reference()


@st.cache_data
def get_risky_suppliers():
    return load_risky_suppliers_by_target()


def render_alerts(report) -> None:
    st.subheader("Fiabilité & alertes (généralisation industrielle)")
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        pct = int(report.trust_score * 100)
        if report.trust_score >= 0.75:
            st.metric("Score de confiance", f"{pct} %", delta="Élevée")
        elif report.trust_score >= 0.45:
            st.metric("Score de confiance", f"{pct} %", delta="Moyenne")
        else:
            st.metric("Score de confiance", f"{pct} %", delta="Faible", delta_color="inverse")
    with col_t2:
        st.caption(report.summary)

    if not report.alerts:
        st.success("Aucune alerte : profil dans les plages NAV habituelles.")
        return

    for alert in report.alerts:
        if alert.level == "danger":
            st.error(f"**{alert.category}** — {alert.message}")
        elif alert.level == "warning":
            st.warning(f"**{alert.category}** — {alert.message}")
        else:
            st.info(f"**{alert.category}** — {alert.message}")


registry = load_registry()
if not registry.empty and "target" in registry.columns:
    targets_available = registry["target"].unique().tolist()
else:
    targets_available = TARGET_COLUMNS

st.sidebar.header("Configuration")
target = st.sidebar.selectbox("Propriété cible", targets_available)
model_version = st.sidebar.radio(
    "Modèle",
    ["A", "B"],
    format_func=lambda x: f"Modèle {x} ({'matière' if x == 'A' else 'opérationnel'})",
)

ref = get_ood_reference()
risky_map = get_risky_suppliers()

st.subheader(f"Paramètres du lot — cible : {target}")

col1, col2 = st.columns(2)
with col1:
    mi = st.number_input("MI (g/10min)", min_value=0.0, value=0.5, step=0.01)
    hlmi = st.number_input("HLMI", min_value=0.0, value=4.0, step=0.1)
    density = st.number_input("Densité (g/cm³)", min_value=0.5, max_value=1.5, value=0.95, step=0.001)
    carbon = st.number_input("Noir de carbone (%)", min_value=0.0, value=0.0, step=0.1)
with col2:
    ash = st.number_input("Cendres (%)", min_value=0.0, value=0.0, step=0.1)
    onset = st.number_input("Onset DSC", value=0.0, step=0.1)
    peak = st.number_input("Peak DSC", value=0.0, step=0.1)
    recycled = st.selectbox("Recyclé / Vierge", ["RECYCLE", "VIERGE", ""])

supplier = ""
location = ""
grade = ""
year = 2023

if model_version == "B":
    st.markdown("#### Contexte opérationnel (modèle B)")
    c3, c4 = st.columns(2)
    known_sup = sorted(ref.get("known_supplier", [])) if ref else []
    supplier_options = ["— Sélectionner —"] + known_sup + ["➕ Nouveau fournisseur (hors NAV)"]
    with c3:
        supplier_choice = st.selectbox("Fournisseur (code NAV)", supplier_options, index=0)
        if supplier_choice == "➕ Nouveau fournisseur (hors NAV)":
            supplier = st.text_input("Code / nom du nouveau fournisseur", "")
        elif supplier_choice != "— Sélectionner —":
            supplier = supplier_choice
        else:
            supplier = st.text_input("Ou saisir un code fournisseur", "")

        known_loc = sorted(ref.get("known_origin", [])) if ref else []
        loc_opts = ["— Sélectionner —"] + known_loc + ["➕ Nouveau site"]
        loc_choice = st.selectbox("Origine / site (location)", loc_opts)
        if loc_choice == "➕ Nouveau site":
            location = st.text_input("Code site", "")
        elif loc_choice != "— Sélectionner —":
            location = loc_choice
        else:
            location = st.text_input("Site (location)", "")

    with c4:
        known_gr = sorted(ref.get("known_grade", [])) if ref else []
        gr_opts = ["— Sélectionner —"] + known_gr + ["➕ Nouveau grade / description"]
        gr_choice = st.selectbox("Grade / description matière", gr_opts)
        if gr_choice == "➕ Nouveau grade / description":
            grade = st.text_input("Description FR", "")
        elif gr_choice != "— Sélectionner —":
            grade = gr_choice
        else:
            grade = st.text_input("Description matière (optionnel)", "")

        year = st.number_input("Année réception", min_value=2010, max_value=2030, value=2023)

input_row = {
    "mi": mi,
    "hlmi": hlmi,
    "density_g_cm3": density,
    "carbon_black": carbon,
    "ash": ash,
    "onset": onset,
    "peak": peak,
    "recycled_virgin": recycled,
    "frr": hlmi / mi if mi > 0 else np.nan,
    "log_mi": np.log1p(mi),
    "log_hlmi": np.log1p(hlmi),
    "charge_total": carbon + ash,
    "thermal_window": peak - onset if peak and onset else np.nan,
}
if model_version == "B":
    input_row.update({
        "supplier_code": supplier,
        "supplier_name": supplier,
        "location": location,
        "description_fr": grade,
        "reception_year": year,
        "reception_month": 6,
        "reception_quarter": 2,
    })

ood_report = check_ood(input_row, target, model_version, ref, risky_map)
render_alerts(ood_report)

st.divider()
st.subheader("Prédiction")

if not registry.empty:
    if "model_A_or_B" in registry.columns:
        row = registry[(registry["target"] == target) & (registry["model_A_or_B"] == model_version)]
    elif "model_version" in registry.columns:
        row = registry[(registry["target"] == target) & (registry["model_version"] == model_version)]
    else:
        row = registry[registry["target"] == target]
else:
    row = pd.DataFrame()
if row.empty and not registry.empty:
    row = registry[registry["target"] == target].head(1)

if not row.empty and Path(row.iloc[0]["model_path"]).exists():
    model = load_model(row.iloc[0]["model_path"])
    try:
        X = build_inference_x(input_row, model)
        pred = model.predict(X)
        ds_ver = row.iloc[0].get("dataset_version", "")
        st.success(f"Prédiction **{target}** : **{pred[0]:.4f}**")
        if ds_ver:
            st.caption(f"Modèle entraîné sur dataset : `{ds_ver}` — {row.iloc[0].get('model_name', 'RF')}")

        if ood_report.has_critical:
            st.warning(
                "Prédiction affichée à titre indicatif : au moins une alerte critique "
                "(nouveau fournisseur ou hors distribution). Validation labo recommandée."
            )

        if reliability_path.exists():
            rel = pd.read_csv(reliability_path)
            r = rel[rel["target"] == target]
            if not r.empty and "confidence_score" in r.columns:
                st.info(f"Couverture bootstrap (phase 2) : {r.iloc[0]['confidence_score']:.2f}")

        expected_cols = get_pipeline_feature_columns(model)
        missing = [
            c for c in expected_cols
            if c not in input_row or pd.isna(input_row.get(c)) or input_row.get(c) == ""
        ]
        if missing:
            st.warning(
                "Features non renseignées (imputation automatique) : "
                + ", ".join(missing[:12])
            )
    except Exception as e:
        st.error(f"Erreur prédiction : {e}")
else:
    st.warning("Modèle non trouvé — exécutez `python run_pipeline.py` puis `python run_phase2.py`.")

with st.sidebar.expander("Aide — alertes"):
    st.markdown(
        """
        - **nouveau_fournisseur** : code absent du NAV (phase 3 GroupKFold).
        - **fournisseur_rare** : moins de 10 lots historiques.
        - **fournisseur_risque** : MAE élevé en CV par fournisseur pour cette cible.
        - **hors_distribution** : MI, densité, etc. hors percentiles NAV.
        - **nouvelle_origine / nouveau_grade** : site ou grade jamais vu.
        """
    )

if importance_path.exists():
    imp = pd.read_csv(importance_path)
    ver_col = "model_version" if "model_version" in imp.columns else "model_A_or_B"
    sub = imp[(imp["target"] == target)]
    if ver_col in imp.columns:
        sub = sub[sub[ver_col] == model_version]
    if not sub.empty:
        st.subheader("Importance des variables")
        st.bar_chart(sub.set_index("feature")["importance"].head(10))

st.caption("Soleno NAV — Streamlit avec alertes OOD (phase 3)")
