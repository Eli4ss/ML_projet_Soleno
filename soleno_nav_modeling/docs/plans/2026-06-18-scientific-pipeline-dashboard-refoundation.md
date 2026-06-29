# Refonte scientifique pipeline + dashboard — Plan d'exécution

> **Objectif :** aligner toutes les phases sur un protocole d'évaluation unique, crédible industriellement, et refondre le dashboard autour de la fiabilité plutôt que du R² optimiste.

**Architecture :** `evaluation_standard.py` devient la source de vérité (tiers cibles, versions dataset, statuts déploiement). `step26` agrège phase 3 + registre phase 2 en tables maîtres consommées par Streamlit. Phase 4 est réalignée sur datasets validés phase 2 avec distinction lab-assisté vs production OOF.

**Stack :** Python, sklearn, pandas, Streamlit, Plotly.

---

## Lot 1 — Protocole scientifique unifié

- [x] `src/evaluation_standard.py` — tiers, priorités CV, statuts déploiement
- [x] `src/step26_unified_evaluation.py` — tables maîtres + rapport
- [x] `run_scientific_evaluation.py` — orchestrateur
- [x] Aligner `step24` sur `load_modeling_frame` (datasets validés)
- [x] Documenter modes cascade : `cascade` (lab-assisté) vs `pspp` (production OOF)

## Lot 2 — Dashboard fiabilité

- [x] Page `08_Model_Reliability.py`
- [x] `load_unified_evaluation()` dans data_loader
- [x] `target_readiness.py` alimenté par step26
- [x] Visualisations parity / dégradation random vs group CV
- [x] Corriger textes interprétatifs page 06 + accueil

## Lot 3 — Validation & gouvernance

- [x] Exécuter `run_scientific_evaluation.py --with-cascade`
- [ ] Revue métier des seuils `deployment_status`
- [ ] Journalisation drift (phase ultérieure)

## Critères de succès

1. Une seule table `target_deployment_status.csv` référencée par le dashboard
2. Chaque cible affiche random CV **et** group supplier côte à côte
3. Phase 4 n'utilise plus le dataset brut non validé
4. Messages dashboard cohérents avec phase 3 (pas de « izod robuste » si group CV négatif)
