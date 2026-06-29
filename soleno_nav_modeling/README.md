# Soleno NAV — Pipeline ML résines recyclées

## Structure

Voir arborescence `data/`, `src/`, `notebooks/`, `outputs/`, `app/`.

## Installation

```bash
cd soleno_nav_modeling
pip install -r requirements.txt
```

## Exécution du pipeline (étapes 1-12)

```bash
python run_pipeline.py
```

## Phase 2 — Validation des cibles et re-modélisation (étapes 14-20)

```bash
python run_phase2.py
```

Livrables dans `outputs/14_target_validation/` … `outputs/19_updated_model_registry/` et `outputs/reports/20_improvement_final_report.md`.

## Phase 3 — Généralisation industrielle (étapes 21-23)

```bash
python run_phase3.py
```

- GroupKFold fournisseur (`supplier_code`), grade (`description_fr`), origine (`location`)
- Validation temporelle (`reception_year`)
- Analyse erreurs par fournisseur / grade / origine / période
- Comparaison modèle A vs B : `outputs/23_model_A_vs_B_robust/`

## Phase 4 — Cascade / PSPP (étapes 24-25)

```bash
python run_phase4.py
```

## Phase 5 — Évaluation scientifique unifiée (référence dashboard)

```bash
python run_scientific_evaluation.py
# ou avec re-calcul cascade :
python run_scientific_evaluation.py --with-cascade
```

Livrables dans `outputs/26_unified_evaluation/` — statuts de déploiement et métriques maîtres.

## Application Streamlit

**MVP multipages (recommandé)** :

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

**Prototype initial (étape 13)** :

```bash
streamlit run app/app.py
```

L’app affiche des **alertes OOD** (phase 3) : nouveau fournisseur, site/grade inconnus, variables hors percentiles NAV, fournisseurs à risque selon GroupKFold. Nécessite `run_phase2.py` et idéalement `run_phase3.py`.

## Modèles

- **Modèle A** : features matière (rhéologie, densité, charges, thermique, dérivées).
- **Modèle B** : A + fournisseur, location, description, dates (année/mois/trimestre).

Les identifiants (lot, PO, numéros internes) ne sont jamais utilisés en entrée.
