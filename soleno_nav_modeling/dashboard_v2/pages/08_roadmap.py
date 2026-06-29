"""Page 8 — Feuille de route."""
import _bootstrap  # noqa: F401

import streamlit as st

from components.layout import page_header, render_expert_toggle_sidebar
from services.maturity import build_target_maturity_table, pilot_targets

render_expert_toggle_sidebar()
page_header(
    "Feuille de route",
    "Progression vers une utilisation industrielle fiable.",
    "delivered",
)

st.subheader("Étape actuelle")
st.markdown(
    """
- Exploration des données NAV et contrôle qualité
- Validation scientifique multi-protocoles (phase 3–5)
- Préparation du **pilote contrôlé** pour les cibles éligibles
    """
)
pilots = pilot_targets()
if pilots:
    st.success(f"Cibles pilote actives : {', '.join(pilots)}")
else:
    st.warning("Aucune cible pilote — vérifiez phase 5 ou overrides maturité.")

st.subheader("Étape pilote")
st.markdown(
    """
- Tests sur **nouveaux lots** avec journalisation
- Comparaison estimation / résultat laboratoire
- Collecte des retours utilisateurs et décisions réelles
- Définition de seuils métier acceptables (MAE, erreur relative)
    """
)

st.subheader("Étape d'amélioration")
st.markdown(
    """
- Collecte plus de données  
- Amélioration des modèles faibles en généralisation
- Ajout de variables procédé non capturées aujourd'hui
- Calibration des incertitudes prédictives
    """
)


st.subheader("Étape d'industrialisation")
st.markdown(
    """
| Domaine | Actions prévues |
|---------|-----------------|
| Accès | Authentification, rôles utilisateurs |
| Intégration | Connexion NAV, flux lots temps réel |
| Modèles | Versionnement, traçabilité, réentraînement contrôlé |
| Surveillance | Détection de dérive, alertes qualité données |
| Gouvernance | Comité validation, documentation limites |
    """
)

