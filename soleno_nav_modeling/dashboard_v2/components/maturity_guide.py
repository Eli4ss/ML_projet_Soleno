"""Section « Logique de maturité » — parcours de validation scientifique."""
from __future__ import annotations

import streamlit as st

from config import CASCADE_APPROACH_HELP, CASCADE_APPROACH_LABELS
from v2_config.settings import VALIDATION_JOURNEY


def render_maturity_logic() -> None:

    st.subheader("Parcours de validation")
    journey = " → ".join(f"**{s}**" for s in VALIDATION_JOURNEY)
    st.markdown(journey)

    with st.expander("1. Développement technique ≠ validation industrielle", expanded=True):
        st.markdown(
            """
Le niveau de maturité dépend notamment de :
- la qualité et la complétude des données ;
- le nombre d'échantillons et la représentativité fournisseur / grade ;
- la stabilité temporelle ;
- la performance sur des données jamais vues (GroupKFold, test temporel) ;
- la cohérence physique et la validation par les utilisateurs ;
- la comparaison avec les résultats laboratoire.
            """
        )

    with st.expander("2. La validation aléatoire ne suffit pas"):
        st.markdown(
            """
Un bon R² en KFold **aléatoire** peut être optimiste si des lots similaires se retrouvent
à la fois en entraînement et en test.

**Priorité aux protocoles robustes :**
1. GroupKFold par fournisseur
2. GroupKFold par grade
3. GroupKFold par site
4. Validation temporelle
5. KFold aléatoire (référence interne uniquement)
6. Test terrain sur nouveaux lots (pilote)

Le dashboard compare ces protocoles **sans présenter le meilleur chiffre comme unique vérité**.
            """
        )

    with st.expander("3. Modèle A vs Modèle B — questions différentes"):
        st.markdown(
            """
| | Modèle A | Modèle B |
|---|----------|----------|
| Entrées | Propriétés matière | Matière + contexte (fournisseur, grade, site, période) |
| Usage | Signal intrinsèque matière | Performance sur contextes connus |
| Risque | Sous-estime effets opérationnels | Peut mémoriser l'historique |

Présentation recommandée : prédiction A, prédiction B, écart A/B, sensibilité au contexte.
Un **grand écart** est un signal à examiner — pas une preuve que B est « meilleur ».
            """
        )

    with st.expander("4. Cascade lab-assistée vs PSPP"):
        pspp_lbl = CASCADE_APPROACH_LABELS["pspp"]
        st.markdown(
            f"""
| Mode | Intermédiaires | Interprétation |
|------|----------------|----------------|
| {CASCADE_APPROACH_LABELS["cascade"]} | **Mesures réelles** (OIT, traction, flexion…) | Blocs suivants reçoivent des infos labo fiables |
| {pspp_lbl} | **Valeurs prédites** (chaîne complète) | Erreurs propagées — pas de mesure labo intermédiaire |

{CASCADE_APPROACH_HELP["pspp"]}

Ne pas conclure que la cascade est « supérieure » uniquement parce qu'elle utilise de vraies mesures intermédiaires.
            """
        )

    with st.expander("5. Indice de qualité ≠ probabilité de vérité"):
        st.markdown(
            """
Les scores combinant complétude, hors plage, OOD et statut modèle décrivent les **conditions**
de la prédiction — pas une garantie d'exactitude.

Vocabulaire recommandé :
- indice de qualité de la prédiction ;
- indice de fiabilité des entrées ;
- indicateur de domaine ;
- niveau de prudence.
            """
        )
