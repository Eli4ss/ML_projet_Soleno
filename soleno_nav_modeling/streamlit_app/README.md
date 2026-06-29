# Soleno NAV — MVP Streamlit



Application R&D pour explorer le dataset NAV, prédire les propriétés des résines et évaluer la **fiabilité** des modèles (généralisation fournisseur/grade).



## Prérequis



Exécuter le pipeline à la racine `soleno_nav_modeling/` :



```bash

python run_pipeline.py

python run_phase2.py

python run_phase3.py

python run_scientific_evaluation.py   # recommandé — statuts déploiement dashboard

# ou avec cascade :

python run_scientific_evaluation.py --with-cascade

```



## Installation



```bash

cd streamlit_app

pip install -r requirements.txt

```



## Lancement



```bash

streamlit run app.py

```



Navigation : sidebar Streamlit (pages multipages).



## Fonctionnalités



- **Fiabilité des modèles** (phase 5) — statuts déploiement, GroupKFold, parity plots

- **Contrôle qualité** avant prédiction (`assets/input_validity_rules.csv`)

- **Score de confiance** et **recommandation** par cible

- **Cascade lab-assisté** / **PSPP production** (phase 4)

- **Comparaison A vs B**, export batch CSV/Excel, journal des prédictions

- **Mode Simple / Expert** (sidebar)



Guide complet : [`../GUIDE_UTILISATION.md`](../GUIDE_UTILISATION.md)



## Pages



| Page | Fichier |

|------|---------|

| Accueil | `app.py` / `pages/01_Home.py` |

| Explorateur de données | `pages/02_Data_Explorer.py` |

| Prédiction unitaire | `pages/03_Single_Lot_Prediction.py` |

| Prédiction par lot | `pages/04_Batch_Prediction.py` |

| Explicabilité | `pages/05_Explainability.py` |

| Comparaison des modèles | `pages/06_Model_Comparison.py` |

| **Fiabilité des modèles** | `pages/08_Model_Reliability.py` |

| Cascade / PSPP | `pages/07_Cascade_Comparison.py` |



Données et modèles lus depuis `../outputs/` et `../data/`.

