# Dashboard V2 — Conception orientée maturité scientifique

**Projet :** Polymer AI / Soleno NAV  
**Date :** 2026-06-18  
**Dossier :** `dashboard_v2/`

---

## 1. Audit du projet existant

### Architecture

| Élément | Emplacement | Rôle |
|---------|-------------|------|
| Pipeline ML | `src/` (steps 01–26) | Audit → modélisation → validation robuste → cascade/PSPP |
| Dashboard MVP | `streamlit_app/` | 8 pages Streamlit (exploration, prédiction, cascade, fiabilité) |
| Données | `outputs/tables/`, `data/target_datasets/` | NAV feature-engineered, datasets par cible |
| Métriques | `outputs/26_unified_evaluation/` | **Source de vérité** pour maturité et validation |
| Modèles | `outputs/models/` | `.joblib` (A/B par cible) — peuvent être absents localement |

### Dashboards existants

1. **Streamlit MVP** (`streamlit_app/app.py`) — application principale, ton « produit ».
2. **Prototype legacy** (`app/app.py`) — obsolète.

Le dashboard V2 **ne remplace pas** le MVP : il propose une **vision par maturité** (livré / pilote / expérimental).

### Modules réutilisables (Tier 1)

- `streamlit_app/src/data_loader.py` — chargement artefacts
- `streamlit_app/src/prediction.py` — inférence A/B
- `streamlit_app/src/pipeline_bridge.py` — résolution conflit namespace `src`
- `streamlit_app/src/quality_check.py`, `confidence.py`, `recommendations.py`
- `streamlit_app/src/lot_workflow.py` — orchestration prédiction
- `streamlit_app/src/visualization.py`, `outlier_viz.py`
- `streamlit_app/config.py` — registre `PATHS`, `TARGETS`

### Risques techniques

| Risque | Mitigation V2 |
|--------|----------------|
| Modèles `.joblib` absents | Message clair ; pilote dégradé mais app démarre |
| Métriques optimistes (KFold aléatoire) | Priorité phase 5 ; comparaison multi-protocoles |
| Collision namespace `src` | `_bootstrap.py` + import depuis `streamlit_app` |
| Monolithe `pages_content.py` | Pages modulaires dans `dashboard_v2/pages/` |

---

## 2. Vision et utilisateurs

### Message central

> Polymer AI n'est pas encore un système autonome remplaçant le laboratoire. C'est une plateforme R&D structurée : explorer → valider → tester en pilote → industrialiser.

### Utilisateurs

| Profil | Niveau principal | Besoin |
|--------|------------------|--------|
| R&D / Qualité Soleno | Analyse R&D + Pilote | Comprendre données, limites, tri préliminaire |
| Data scientist / IA | Expérimental | PSPP, cascade, XAI, recherche |
| Direction / comité | Accueil + Feuille de route | Progression crédible, pas de sur-promesse |

---

## 3. Trois niveaux de maturité

| Niveau | Couleur | Contenu |
|--------|---------|---------|
| **1 — Analyse R&D** | Bleu `#2563eb` | Exploration NAV, qualité, validation modèles, documentation |
| **2 — Pilote contrôlé** | Ambre `#d97706` | Prédiction unitaire supervisée, journal terrain, retours labo |
| **3 — Expérimental** | Violet `#6b21a8` | NCLS, IZOD, UCLS, cascade, PSPP, DL, XAI avancée |

### Dérivation dynamique des cibles (phase 5)

Statuts lus depuis `outputs/26_unified_evaluation/target_deployment_status.csv` :

| `deployment_status` | Niveau prédiction | `allowed_for_prediction` |
|---------------------|-------------------|--------------------------|
| `deploy` | Pilote | Oui |
| `caution` | Pilote (avec confirmation labo) | Oui |
| `lab_only` | Expérimental | Non |
| `blocked` | Expérimental | Non |

Overrides optionnels : `dashboard_v2/v2_config/maturity_overrides.yaml`

**État actuel des données (juin 2026) :**

| Cible | Niveau pilote | Justification (phase 5) |
|-------|---------------|-------------------------|
| OIT | Oui | `caution`, tier A, n≈7588 |
| Flexion | Oui | `caution`, tier A |
| % Allongement | Oui | `caution`, tier A |
| Traction | Oui | `caution` (modèle B recommandé) |
| NCLS | Non | `lab_only`, R² group fournisseur négatif |
| UCLS | Non | `blocked`, n≈86 |
| IZOD | Non | `blocked`, généralisation fournisseur faible |

---

## 4. Pages et navigation

```
Accueil ──┬── Données & qualité (L1)
          ├── Exploration R&D (L1)
          ├── Validation modèles (L1)
          ├── Logique de maturité (L1 — section dédiée)
          ├── Prédiction pilote (L2)
          ├── Suivi pilote (L2)
          ├── Laboratoire expérimental (L3)
          └── Feuille de route
```

Sidebar : badge de niveau par page, mode Simple / Expert, date de mise à jour des fichiers.

---

## 5. Sources de données par page

| Page | Fichiers principaux |
|------|---------------------|
| Accueil | `04_NAV_feature_engineered.csv`, phase 5 recommended + deployment |
| Données | cleaned, feature_engineered, `14_target_validation/*` |
| Exploration | feature_engineered, target_valid_* |
| Validation | `master_metrics_long.csv`, `target_deployment_status.csv`, `ab_robust` |
| Pilote | modèles joblib, input_validity_rules, pilot_journal |
| Suivi pilote | `pilot_journal.csv`, `prediction_history.csv` |
| Expérimental | `25_approach_comparison/*`, cascade metrics |
| Roadmap | statique + statuts phase 5 |

---

## 6. Différences vs dashboard MVP

| Aspect | MVP | Dashboard V2 |
|--------|-----|--------------|
| Organisation | Par fonction (prédiction, cascade…) | Par **maturité scientifique** |
| Prédiction | Toutes cibles MVP par défaut | **Uniquement cibles pilote** autorisées |
| Cascade / PSPP | Page dédiée comparatif | Section **expérimentale** avec avertissement |
| Fiabilité | Page Model Reliability | Intégrée à Validation + explication protocoles |
| Vocabulaire | Mix produit / R&D | **Prudent** (estimation, tri préliminaire) |
| Journal | prediction_history | **pilot_journal** enrichi (labo, décision réelle) |
| Architecture | `pages_content.py` monolithique | Pages + services + composants |

---

## 7. Composants réutilisés vs nouveaux

### Réutilisés

Chargement, prédiction, qualité, visualisations Plotly, bootstrap pipeline.

### Nouveaux

- `services/maturity.py` — configuration centrale maturité
- `services/pilot_journal.py` — protocole validation terrain
- `services/validation_service.py` — agrégation métriques multi-protocoles
- `components/maturity_badges.py`, `maturity_guide.py`
- Pages modulaires orientées parcours de validation
- Tests de chargement et prédiction pilote

---

## 8. Hypothèses et limites

- Les statuts pilote suivent la phase 5 ; un override YAML permet l'ajustement sans redeploy code.
- Les modèles peuvent être indisponibles : la prédiction pilote affiche un message, pas une erreur fatale.
- Pas d'authentification ni intégration NAV temps réel (feuille de route future).
- Intervalles de prédiction affichés seulement si le pipeline les produit (sinon masqués).
