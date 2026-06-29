# Dashboard V2 — Polymer AI (maturité scientifique)

Deuxième dashboard Streamlit, distinct du MVP (`streamlit_app/`), organisé autour de trois niveaux de maturité : **Analyse R&D**, **Pilote contrôlé**, **Expérimental**.

## Prérequis

- Python 3.10+
- Pipeline exécuté au minimum jusqu'à l'étape 4 (`run_pipeline.py`)
- Pour validation complète : `run_scientific_evaluation.py`
- Pour prédictions pilote : modèles `.joblib` (phase 2)

## Installation

```bash
cd soleno_nav_modeling/dashboard_v2
pip install -r requirements.txt
```

## Lancement

```bash
cd soleno_nav_modeling/dashboard_v2
streamlit run app.py
```

Le dashboard démarre même si certains fichiers secondaires sont absents (message informatif par composant).

## Pages

| Page | Niveau | Rôle |
|------|--------|------|
| Accueil | — | Vision, cartes maturité, indicateurs NAV |
| Comment lire la plateforme | L1 | Logique scientifique de séparation |
| Données et qualité | L1 | Complétude, outliers, couverture |
| Exploration R&D | L1 | Filtres, distributions, corrélations |
| Validation modèles | L1 | Métriques multi-protocoles (phase 5) |
| Prédiction pilote | L2 | Estimation supervisée + journal |
| Suivi pilote | L2 | Retours laboratoire, MAE terrain |
| Laboratoire expérimental | L3 | PSPP, cascade, cibles bloquées |
| Feuille de route | — | Progression vers industrialisation |

## Configuration maturité

- Dérivation automatique : `outputs/26_unified_evaluation/recommended_model_per_target.csv`
- Overrides manuels : `v2_config/maturity_overrides.yaml`

## Documentation

- Conception : `../DASHBOARD_V2_DESIGN.md`
- Architecture : `ARCHITECTURE.md`
- Rapport d'implémentation : `../DASHBOARD_V2_IMPLEMENTATION_REPORT.md`

## Tests

```bash
cd soleno_nav_modeling/dashboard_v2
python -m unittest discover -s tests -v
```

## Dashboard MVP existant

Le MVP reste disponible :

```bash
cd soleno_nav_modeling/streamlit_app
streamlit run app.py
```
