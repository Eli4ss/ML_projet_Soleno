# Rapport d'implémentation — Dashboard V2 (maturité scientifique)

**Date :** 2026-06-18  
**Emplacement :** `soleno_nav_modeling/dashboard_v2/`

---

## 1. Ce qui a été créé

| Livrable | Fichier / dossier |
|----------|-------------------|
| Application Streamlit | `dashboard_v2/app.py` + 8 pages dans `pages/` |
| Configuration maturité | `services/maturity.py`, `v2_config/maturity_overrides.yaml` |
| Journal pilote | `services/pilot_journal.py` → `outputs/prediction_logs/pilot_journal.csv` |
| Composants UI | `components/layout.py`, `components/maturity_guide.py` |
| Services | `data_service.py`, `validation_service.py` |
| Tests | `tests/test_loading.py`, `tests/test_prediction.py` (8 tests, OK) |
| Documentation | `README.md`, `ARCHITECTURE.md`, `CHANGELOG.md`, `DASHBOARD_V2_DESIGN.md` |

---

## 2. Réutilisé du dashboard MVP

- `streamlit_app/src/data_loader.py` — chargement datasets et phase 5
- `streamlit_app/src/prediction.py` — inférence modèles A/B
- `streamlit_app/src/lot_workflow.py` — prédiction + confiance + recommandations prudentes
- `streamlit_app/src/quality_check.py`, `confidence.py`, `recommendations.py`
- `streamlit_app/src/visualization.py`, `outlier_viz.py`
- `streamlit_app/src/pipeline_bridge.py`, `_bootstrap.py` (pattern)
- `streamlit_app/config.py` — PATHS, TARGETS, labels cascade

**Non modifié :** le dashboard MVP (`streamlit_app/`) reste intact.

---

## 3. Nouvelles visions proposées

1. **Navigation par maturité** (L1 bleu / L2 ambre / L3 violet) au lieu de navigation par fonction produit.
2. **Page pédagogique** « Comment lire la plateforme » — validation aléatoire vs robuste, A vs B, cascade vs PSPP.
3. **Pilote comme protocole** — journal enrichi (labo, décision réelle, erreurs terrain).
4. **Filtrage des cibles** — seules les cibles `caution`/`deploy` phase 5 sont proposées en prédiction pilote.
5. **Section expérimentale isolée** — NCLS, IZOD, UCLS, PSPP avec avertissement explicite.

---

## 4. Décisions de conception

| Décision | Justification |
|----------|---------------|
| Maturité dérivée de phase 5 | Évite statuts codés en dur ; cohérent avec `evaluation_standard.py` |
| Overrides YAML | Permet ajustement métier sans redeploy |
| Dossier `v2_config/` vs `config/` | Évite collision avec `streamlit_app/config.py` |
| Pas de duplication prediction.py | Un seul moteur d'inférence maintenu |
| Vocabulaire « estimation » / « tri préliminaire » | Exigence présentation industrielle prudente |

---

## 5. Maturité des cibles (données actuelles phase 5)

| Cible | Niveau | Pilote autorisé | Source |
|-------|--------|-----------------|--------|
| OIT | L2 Pilote | Oui | `caution`, tier A, n=7588 |
| Flexion | L2 Pilote | Oui | `caution`, tier A |
| % Allongement | L2 Pilote | Oui | `caution`, tier A |
| Traction | L2 Pilote | Oui | `caution` (modèle B) |
| NCLS | L3 Expérimental | Non | `lab_only`, R² group négatif |
| UCLS | L3 Expérimental | Non | `blocked`, n≈86 |
| IZOD | L3 Expérimental | Non | `blocked`, tier C |

---

## 6. Tests réalisés

```
python -m unittest discover -s tests -v
→ 8 tests OK
```

- Chargement table maturité et stats NAV
- Cibles pilote cohérentes avec phase 5
- Journal pilote append/load
- Prédiction gracieuse si modèle absent
- NCLS bloquée en prédiction pilote

---

## 7. Fonctionnalités

### Fonctionnelles

- Accueil avec 3 cartes maturité et parcours validation
- Données & qualité (missingness, couverture, outliers phase 2)
- Exploration R&D (filtres, distributions, ECDF, corrélations)
- Validation modèles (synthèse phase 5, détail protocoles, A/B)
- Prédiction pilote (qualité entrées, A/B, journal)
- Suivi pilote (stats terrain, retour labo)
- Laboratoire expérimental (PSPP, pivot R² cascade si disponible)
- Feuille de route

### Partielles

- **Prédiction pilote** — fonctionne si `.joblib` présents ; sinon message `model_not_available`
- **Intervalle de prédiction** — non affiché (pipeline ne produit pas d'intervalles calibrés exportés)
- **Comparaison cascade** — métriques OK ; inférence cascade non exposée en pilote (volontairement en L3)

### Reportées

- Mode simple vs expert complet (toggle expert basique implémenté)
- Matrice cas validés/non validés interactive avancée
- Export PDF / rapport automatique
- Authentification et intégration NAV temps réel
- Surveillance de dérive et réentraînement

---

## 8. Hypothèses

- `recommended_model_per_target.csv` reflète la dernière exécution de `run_scientific_evaluation.py`
- Les colonnes NAV (`supplier_code`, `reception_year`, etc.) sont stables
- Le journal pilote local suffit pour la phase de validation terrain initiale

---

## 9. Limites connues

- Modèles `.joblib` absents dans l'environnement audité — prédiction pilote dégradée
- `data/raw/` peut être vide — rerun pipeline impossible sans Excel
- Chemins absolus dans certains registres phase 2 (machine locale)
- Pas de tests E2E Streamlit automatisés (browser)

---

## 10. Description des pages

| Page | Contenu principal |
|------|-------------------|
| **Accueil** | Message vision, 3 cartes L1/L2/L3, KPIs NAV, table maturité par cible, logique intégrée |
| **Comment lire** | 5 expanders pédagogiques (validation, A/B, cascade, indice qualité) |
| **Données** | Lignes vs valeurs validées vs entraînement ; missingness ; distributions |
| **Exploration** | Filtres fournisseur/grade/site/année ; boxplots ; tendances ; corrélations |
| **Validation** | Table synthèse ; détail protocoles ; barres R²/MAE ; comparaison A/B |
| **Pilote prédiction** | Avertissement ; formulaire matière ; contrôle qualité ; estimation ; journal |
| **Suivi pilote** | KPIs terrain ; erreurs par fournisseur/cible ; formulaire retour labo |
| **Expérimental** | Architecture PSPP ; pivot R² ; cibles bloquées ; travaux futurs |
| **Roadmap** | 4 étapes vers industrialisation |

---

## 11. Fichiers sources utilisés

Voir `v2_config/settings.py` → `SOURCE_FILES_DOC` et `ARCHITECTURE.md`.

Principaux :

- `outputs/tables/04_NAV_feature_engineered.csv`
- `outputs/26_unified_evaluation/*.csv`
- `outputs/14_target_validation/*.csv`
- `outputs/21_robust_generalization/robust_validation_results.csv`
- `outputs/25_approach_comparison/pivot_r2.csv` (optionnel)

---

## 12. Prochaines étapes recommandées

1. Régénérer les modèles phase 2 (`run_phase2.py`) pour activer les prédictions pilote.
2. Lancer le pilote sur 20–50 lots avec saisie systématique retour labo.
3. Définir seuils MAE métier par cible à partir du suivi pilote.
4. Enrichir `maturity_overrides.yaml` si comité validation ajuste les statuts.
5. Ajouter tests E2E Playwright si CI requis.

---

## 13. Lancement

```bash
cd soleno_nav_modeling/dashboard_v2
pip install -r requirements.txt
streamlit run app.py
```
