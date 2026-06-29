# Architecture — Dashboard V2

## Structure

```
dashboard_v2/
├── app.py                 # Accueil Streamlit
├── _bootstrap.py          # sys.path + nettoyage namespace src
├── v2_config/
│   ├── settings.py        # Couleurs maturité, chemins journal pilote
│   └── maturity_overrides.yaml
├── services/
│   ├── maturity.py        # Dérivation niveaux depuis phase 5
│   ├── data_service.py    # Wrap data_loader MVP
│   ├── validation_service.py
│   └── pilot_journal.py
├── components/
│   ├── layout.py          # Badges, en-têtes, mode expert
│   └── maturity_guide.py  # Texte pédagogique
├── pages/                 # 8 pages numérotées
└── tests/
```

## Dépendances externes (réutilisées)

| Module MVP | Usage V2 |
|------------|----------|
| `streamlit_app/config.py` | PATHS, TARGETS, filtres |
| `streamlit_app/src/data_loader.py` | Chargement CSV |
| `streamlit_app/src/prediction.py` | Inférence A/B |
| `streamlit_app/src/lot_workflow.py` | Orchestration pilote |
| `streamlit_app/src/visualization.py` | Graphiques Plotly |
| `streamlit_app/src/pipeline_bridge.py` | Import pipeline sans conflit |

## Flux de données

```
Phase 5 CSV ──► services/maturity.py ──► UI badges + filtre pilote
Phase 5 CSV ──► validation_service.py ──► page Validation
04_NAV_*.csv ──► data_service.py ──► pages Données / Exploration
.joblib ──► prediction.py ──► page Pilote (si disponibles)
pilot_journal.csv ──► pilot_journal.py ──► Suivi pilote
```

## Règles de conception

1. **Aucune métrique inventée** — tout vient des CSV `outputs/`
2. **Fichier absent** — composant masqué + hint pipeline
3. **Vocabulaire prudent** — pas de « lot accepté/rejeté »
4. **Séparation L1/L2/L3** visible dès l'accueil

## Extension

Pour forcer le niveau d'une cible sans modifier le code :

```yaml
# v2_config/maturity_overrides.yaml
targets:
  traction:
    level: pilot
    allowed_for_prediction: true
```
