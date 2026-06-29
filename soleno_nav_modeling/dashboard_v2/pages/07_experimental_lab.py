"""Page 7 — Laboratoire expérimental (niveau Expérimental)."""
import _bootstrap  # noqa: F401

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from components.layout import page_header, render_expert_toggle_sidebar
from config import CASCADE_APPROACH_HELP, CASCADE_APPROACH_LABELS, CASCADE_FINAL_TARGETS, CASCADE_MECHANICAL_TARGETS, CASCADE_THERMAL_TARGETS, PATHS, TARGETS, cascade_approach_label
from v2_config.settings import EXPERIMENTAL_WARNING
from services.maturity import build_target_maturity_table
from src.prediction import cascade_approaches_available

render_expert_toggle_sidebar()
page_header(
    "Laboratoire expérimental",
    "PSPP, cascade, modèles avancés — recherche non validée pour usage industriel seul.",
    "experimental",
)

st.error(EXPERIMENTAL_WARNING)

st.subheader("Architecture PSPP")
st.markdown(
    """
```
Procédé → Structure → Propriétés → Performance

Bloc 1 (thermique) : OIT
Bloc 2 (mécanique) : Traction, Flexion, IZOD, Allongement
Bloc 3 (durabilité) : NCLS, UCLS
```
    """
)

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Bloc 1 — thermique**")
    for t in CASCADE_THERMAL_TARGETS:
        st.write(f"- {TARGETS.get(t, {}).get('label', t)}")
with c2:
    st.markdown("**Bloc 2 — mécanique**")
    for t in CASCADE_MECHANICAL_TARGETS:
        st.write(f"- {TARGETS.get(t, {}).get('label', t)}")
with c3:
    st.markdown("**Bloc 3 — durabilité**")
    for t in CASCADE_FINAL_TARGETS:
        st.write(f"- {TARGETS.get(t, {}).get('label', t)}")

st.subheader("Direct vs Cascade lab-assistée vs PSPP")
pspp_lbl = CASCADE_APPROACH_LABELS["pspp"]
st.markdown(
    f"""
| Approche | Intermédiaires | Usage |
|----------|----------------|-------|
| {CASCADE_APPROACH_LABELS["direct"]} | — | Matière → cible |
| {CASCADE_APPROACH_LABELS["cascade"]} | **Mesurés** | Meilleure référence si labo disponible |
| Cascade entièrement prédite | **Prédits en chaîne** | Inférence réaliste sans labo — erreurs propagées |
    """
)
st.caption(CASCADE_APPROACH_HELP["pspp"])

pivot_r2 = PATHS.get("cascade_pivot_r2")
if pivot_r2 and Path(pivot_r2).exists():
    r2 = pd.read_csv(pivot_r2, index_col=0)
    r2_display = r2.rename(columns=cascade_approach_label)
    st.caption(f"Source : {pivot_r2}")
    st.dataframe(r2_display, use_container_width=True)
    melted = r2.reset_index().melt(id_vars=r2.index.name or "target", var_name="approche", value_name="R2")
    if "target" not in melted.columns and r2.index.name:
        melted = melted.rename(columns={r2.index.name: "target"})
    melted["approche"] = melted["approche"].map(cascade_approach_label)
    fig = px.bar(melted, x="target", y="R2", color="approche", barmode="group", title="R² par approche (phase 4)")
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Comparaison cascade non disponible — exécutez `run_phase4.py` puis `run_scientific_evaluation.py --with-cascade`.")
st.markdown("- Propagation d'erreurs entre blocs PSPP sans mesures intermédiaires")

st.subheader("Travaux futurs et hypothèses de recherche")
st.markdown(
    """
- Score hors domaine avancé et incertitude prédictive calibrée
- Intégration variables procédé et modèles de recettes
- Deep Learning sur séries DSC / profils complets
- Recommandations automatiques de formulation 
- Apprentissage continu avec gouvernance des données
    """
)
