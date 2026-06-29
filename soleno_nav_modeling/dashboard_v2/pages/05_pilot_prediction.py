"""Page 5 — Prédiction pilote (niveau Pilote contrôlé)."""
import _bootstrap  # noqa: F401

import pandas as pd
import streamlit as st
from components.layout import page_header, render_expert_toggle_sidebar
from config import TARGETS
from v2_config.settings import PILOT_WARNING
from services.maturity import derive_target_maturity, pilot_targets
from services.pilot_journal import append_pilot_prediction
from src.data_loader import load_target_valid
from src.lot_workflow import predict_lot_with_metadata
from src.pipeline_bridge import get_ood_reference
from src.quality_check import check_lot_quality

render_expert_toggle_sidebar()
page_header(
    "Prédiction pilote",
    "Estimation supervisée dans un protocole de validation terrain — confirmation laboratoire recommandée.",
    "pilot",
)

st.warning(PILOT_WARNING)

allowed = pilot_targets()
if not allowed:
    st.error("Aucune cible éligible au pilote selon la configuration de maturité (phase 5).")
    st.stop()

target = st.selectbox(
    "Propriété étudiée",
    allowed,
    format_func=lambda t: TARGETS.get(t, {}).get("label", t),
)
mat = derive_target_maturity(target)
st.caption(
    f"Niveau : {mat['level_label']} | Modèle recommandé : {mat.get('recommended_model', 'A')} | "
    f"n obs. validation : {mat.get('n_samples', '—')}"
)

compare_ab = st.checkbox(
    "Comparer modèle A et modèle B",
    help="Un écart important entre A et B signale une sensibilité au contexte opérationnel.",
)
model_version = st.radio(
    "Modèle utilisé",
    ["A", "B"],
    index=0 if mat.get("recommended_model") == "A" else 1,
    horizontal=True,
)

st.markdown("#### Caractéristiques matière")
c1, c2, c3 = st.columns(3)
with c1:
    mi = st.number_input("MI (g/10min)", 0.0, value=0.5, step=0.01)
    hlmi = st.number_input("HLMI (g/10min)", 0.0, value=4.0, step=0.1)
    density = st.number_input("Densité (g/cm³)", 0.5, 1.5, 0.95, 0.001)
with c2:
    carbon = st.number_input("Noir de carbone %", 0.0, value=0.0)
    ash = st.number_input("Cendres %", 0.0, value=0.0)
    pp = st.number_input("PP %", 0.0, value=0.0)
with c3:
    onset = st.number_input("DSC Onset (°C)", value=0.0)
    peak = st.number_input("DSC Peak (°C)", value=0.0)
    delta_h = st.number_input("Delta H (J/g)", value=0.0)

input_row = {
    "mi": mi, "hlmi": hlmi,
    "density_g_cm3": density, "density_plaque_g_cm3": density,
    "carbon_black": carbon, "ash": ash, "pp": pp,
    "onset": onset, "peak": peak, "delta_h": delta_h,
    "fluidity_g_10min": mi,
}

if model_version == "B" or compare_ab:
    st.markdown("#### Contexte opérationnel (modèle B)")
    b1, b2 = st.columns(2)
    with b1:
        input_row["supplier_code"] = st.text_input("Code fournisseur")
        input_row["location"] = st.text_input("Site")
    with b2:
        input_row["description_fr"] = st.text_input("Grade")
        input_row["reception_year"] = st.number_input("Année réception", 2026)

lot_id = st.text_input("Identifiant lot", value="lot_pilote")
quality = check_lot_quality(input_row, model_version)

st.markdown("#### Contrôle des entrées")
st.markdown(quality.summary_text())
st.caption(
    "L'indice de qualité décrit les conditions de la prédiction — "
    "il ne garantit pas l'exactitude du résultat."
)

if st.button("Calculer l'estimation", type="primary"):
    results, global_rec, quality = predict_lot_with_metadata(
        input_row, [target], model_version=model_version, compare_ab=compare_ab,
    )
    row = results.iloc[0]
    unit = TARGETS.get(target, {}).get("unit", "")

    st.success(f"Estimation {TARGETS[target]['label']} : **{row['prediction']:.3g} {unit}**")
    st.caption(f"Recommandation prudente : {row['recommendation']}")
    st.caption(f"Indice confiance : {row['confidence_level']} (score {row['confidence_score']:.0f})")

    if compare_ab and row.get("pred_model_a") is not None and row.get("pred_model_b") is not None:
        st.markdown(
            f"| Modèle A | Modèle B | Écart |\n"
            f"|----------|----------|-------|\n"
            f"| {row['pred_model_a']:.3g} | {row['pred_model_b']:.3g} | {row.get('gap_a_b', '—')} |"
        )
        st.caption(row.get("ab_comment", ""))

    vdf = load_target_valid(target)
    if vdf is not None and pd.notna(row["prediction"]):
        hist = pd.to_numeric(vdf[target], errors="coerce").dropna()
        pred = float(row["prediction"])
        pct = (hist < pred).mean() * 100
        med = hist.median()
        if pred < hist.quantile(0.25):
            pos = "estimation dans la plage basse"
        elif pred > hist.quantile(0.75):
            pos = "estimation dans la plage haute"
        else:
            pos = "estimation proche de la médiane historique"
        st.info(f"{pos} (percentile ≈ {pct:.0f} %, n={len(hist):,}, médiane={med:.3g} {unit})")

    try:
        ood = get_ood_reference()
        report = ood.check_ood(
            input_row, target, model_version,
            ood.load_reference(), ood.load_risky_suppliers_by_target(),
        )
        st.caption(f"Indicateur de domaine : confiance OOD {report.trust_score:.0f}/100")
        if report.trust_score < 50:
            st.warning(
                "Profil peu représenté dans les données historiques — "
                "confirmation laboratoire recommandée."
            )
    except Exception:
        pass

    if st.session_state.get("save_pilot", True):
        pid = append_pilot_prediction({
            "lot_id": lot_id,
            "target": target,
            "model_used": f"direct_{model_version}",
            "model_version": model_version,
            "inputs_json": input_row,
            "predicted_value": row["prediction"],
            "prediction_unit": unit,
            "supplier_code": input_row.get("supplier_code", ""),
            "grade": input_row.get("description_fr", ""),
            "site": input_row.get("location", ""),
            "maturity_level": mat["level"],
            "data_quality_level": quality.quality_level,
            "quality_index": row["confidence_score"],
            "prediction_status": row["prediction_status"],
        })
        st.caption(f"Enregistré dans le journal pilote (id : {pid})")

st.checkbox("Enregistrer automatiquement dans le journal pilote", value=True, key="save_pilot")
