# Guide d'utilisation — Soleno NAV Dashboard

**Plateforme R&D de prédiction des propriétés des résines recyclées**

---

## Table des matières

1. [Présentation générale](#1-présentation-générale)
2. [Lancer l'application](#2-lancer-lapplication)
3. [Navigation et mode d'affichage](#3-navigation-et-mode-daffichage)
4. [Page Accueil](#4-page-accueil)
5. [Page Explorateur de données](#5-page-explorateur-de-données)
6. [Page Prédiction unitaire](#6-page-prédiction-unitaire)
7. [Page Prédiction par lot](#7-page-prédiction-par-lot)
8. [Page Explicabilité](#8-page-explicabilité)
9. [Page Comparaison des modèles](#9-page-comparaison-des-modèles)
10. [Page Fiabilité des modèles](#10-page-fiabilité-des-modèles)
11. [Page Cascade / PSPP](#11-page-cascade--pspp)
12. [Scores de confiance et recommandations](#12-scores-de-confiance-et-recommandations)
13. [Référence technique : features et cibles](#13-référence-technique--features-et-cibles)
14. [Workflow typique d'utilisation](#14-workflow-typique-dutilisation)
15. [Limites connues et conseils](#15-limites-connues-et-conseils)

---

## 1. Présentation générale

L'application **Soleno NAV Dashboard** est une plateforme R&D construite avec [Streamlit](https://streamlit.io). Elle exploite les données **NAV (Business Central)** de Soleno pour prédire, à partir de mesures physico-chimiques réalisées à la réception d'un lot de résine recyclée, ses propriétés mécaniques et oxydatives finales — **avant** que le test laboratoire complet ne soit effectué.

### Objectif principal

> Permettre à l'équipe R&D de **trier, classer et orienter** les lots de résine dès la réception, en s'appuyant sur des modèles ML entraînés sur l'historique de **15 013 lots NAV** (dataset mis à jour mai 2026).

L'application est un **assistant de pré-qualification** : elle indique quelles propriétés sont prédictibles, dans quelles conditions, et quand une prédiction ne doit pas être utilisée seule (nouveau fournisseur, cible bloquée, score OOD faible).

### Ce que l'application peut prédire


| Cible       | Symbole          | Unité | Description                                             |
| ----------- | ---------------- | ----- | ------------------------------------------------------- |
| OIT         | `oit_min`        | min   | Temps d'induction d'oxydation — durabilité oxydative    |
| NCLS        | `ncls`           | h     | Résistance à la fissuration sous contrainte notchée     |
| UCLS        | `ucls`           | h     | Résistance à la fissuration sous contrainte non-notchée |
| IZOD        | `izod`           | J/m   | Résistance aux chocs                                    |
| Traction    | `traction`       | MPa   | Module de traction                                      |
| Allongement | `pct_elongation` | %     | Allongement à la rupture                                |
| Flexion     | `flexion`        | MPa   | Module de flexion                                       |


> **Note :** `cell_class` (code classe PE) est disponible en classification uniquement. `temp_c` a été reclassée en variable d'entrée descriptive et n'est pas prédite.

### Approches de modélisation disponibles

| Approche | Logique | Usage recommandé |
|---|---|---|
| **Modèle A** | Matière → cible | Référence matière pure, généralisation fournisseur |
| **Modèle B** | Matière + contexte → cible | Meilleur si fournisseur/grade déjà vu dans NAV |
| **Cascade lab-assisté** | Blocs avec intermédiaires **mesurés** | R&D avec essais partiels (OIT, traction…) |
| **PSPP (OOF)** | Chaîne prédite sans fuite de données | **Référence honnête** pour l'inférence sans labo |

> **Lecture scientifique :** le R² en KFold aléatoire surestime souvent la généralisation. La **validation GroupKFold fournisseur** (phase 3) et le **statut de déploiement** (phase 5) priment pour les décisions industrielles.

---

## 2. Lancer l'application

### Prérequis

- Python 3.10 ou supérieur
- Dépendances installées : `pip install -r streamlit_app/requirements.txt`
- Pipeline exécuté au minimum jusqu'à la **phase 3** (recommandé : **phase 5** pour le dashboard fiabilité)

### Démarrage

```bash
cd soleno_nav_modeling/streamlit_app
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

### Fichiers nécessaires au démarrage


| Fichier | Rôle | Généré par |
| ------- | ---- | ---------- |
| `outputs/tables/04_NAV_feature_engineered.csv` | Dataset principal (15 013 lots × 75 colonnes) | `run_pipeline.py` (étape 4) |
| `outputs/tables/03_NAV_cleaned.csv` | Dataset nettoyé pré-engineering | `run_pipeline.py` (étape 3) |
| `outputs/14_target_validation/corrected_target_datasets/target_valid_*.csv` | Données validées par cible (Phase 2) | `run_phase2.py` (étape 14) |
| `outputs/21_robust_generalization/robust_validation_results.csv` | Validation robuste GroupKFold / temporelle | `run_phase3.py` |
| `outputs/26_unified_evaluation/target_deployment_status.csv` | **Statuts de déploiement par cible** | `run_scientific_evaluation.py` |
| `outputs/26_unified_evaluation/recommended_model_per_target.csv` | Modèle recommandé par cible | `run_scientific_evaluation.py` |
| `outputs/models/` | Modèles ML classiques (`.joblib`) | `run_pipeline.py` ou `run_phase2.py` |
| `outputs/24_cascade_modeling/models/` | Modèles cascade / PSPP (`.joblib`) | `run_phase4.py` ou `run_scientific_evaluation.py --with-cascade` |


Si un fichier est absent, l'application l'indique avec une icône ❌ sur la page Accueil et affiche un message d'avertissement.

### Séquence complète d'exécution du pipeline

```bash
# Phase 1 — Prétraitement + modèles de base (étapes 1-12)
python run_pipeline.py

# Phase 2 — Validation des cibles + re-modélisation (étapes 14-20)
python run_phase2.py

# Phase 3 — Généralisation robuste (GroupKFold, temporel)
python run_phase3.py

# Phase 4 — Modélisation cascade / PSPP (étapes 24-25)
python run_phase4.py

# Phase 5 — Évaluation scientifique unifiée (référence dashboard)
python run_scientific_evaluation.py

# Ou tout-en-un (phase 4 + 5) :
python run_scientific_evaluation.py --with-cascade
```

---

## 3. Navigation et mode d'affichage

### Pages disponibles

La barre de navigation latérale gauche donne accès à **8 pages** :

| Page | Icône | Usage |
| ---- | ----- | ----- |
| **Accueil** | 🏠 | Vue d'ensemble du projet et statut des fichiers |
| **Explorateur de données** | 📊 | Analyse visuelle du dataset NAV |
| **Prédiction unitaire** | 🔬 | Prédire les propriétés d'un lot saisi manuellement |
| **Prédiction par lot** | 📁 | Prédire sur un fichier CSV ou Excel entier |
| **Explicabilité** | 💡 | Comprendre les drivers de chaque prédiction (mode Expert) |
| **Comparaison des modèles** | 📈 | Métriques historiques par phase du pipeline |
| **Fiabilité des modèles** | 🛡️ | **Référence scientifique** — statuts déploiement, GroupKFold |
| **Cascade / PSPP** | 🔗 | Comparer direct / cascade lab-assisté / PSPP production |


### Mode Simple / Expert

Chaque page dispose d'un **sélecteur de mode** dans la barre latérale gauche :

- **Mode Simple** (par défaut) : affiche uniquement la prédiction, le niveau de confiance et la recommandation. Adapté à une utilisation quotidienne en R&D ou production.

- **Mode Expert** : affiche en plus les détails techniques — qualité des données, comparaison Modèle A vs B, scores OOD (hors-domaine), SHAP/importance des features, et tableaux complets.

> Basculer en mode **Expert** ne modifie aucune donnée — c'est uniquement un filtre d'affichage.

---

## 4. Page Accueil

### Ce qu'elle affiche

- **4 métriques en haut** : nombre de lots NAV chargés, nombre de colonnes, nombre de modèles enregistrés, nombre de cibles MVP.
- **Description des approches** (A, B, Cascade, PSPP).
- **Cibles principales** avec leur nombre de valeurs disponibles dans le dataset.
- **Statut du pipeline** : liste des étapes avec ✅ / ⬜ selon les fichiers détectés.
- **Disponibilité des modèles par cible** : tableau alimenté par la **phase 5** (statut déploiement, usage recommandé, confiance a priori).

### Tableau de disponibilité des modèles

| Colonne | Signification |
| ------- | ------------- |
| Statut déploiement | `Déployable` / `Prudence` / `Labo requis` / `Bloqué` — référence phase 5 |
| Statut scientifique | Synthèse qualitative (ex. « Fragile en généralisation ») |
| Usage recommandé | Tri préliminaire, screening, essai labo obligatoire… |
| Confiance a priori | `élevée` / `moyenne` / `faible` — indépendamment du lot saisi |
| Notes | Justification (ex. ratio MAE group/random, R² group fournisseur négatif) |

**Explication technique :** Les statuts proviennent de `outputs/26_unified_evaluation/target_deployment_status.csv`, calculés à partir de la validation robuste phase 3 (GroupKFold fournisseur, grade, site, temporel). Consultez la page **Fiabilité des modèles** pour le détail par protocole.

---

## 5. Page Explorateur de données

### Source des données

Un sélecteur en haut permet de choisir entre :

- **Données enrichies** : `04_NAV_feature_engineered.csv` — 75 colonnes avec toutes les features dérivées (FRR, log_mi, charge_total, etc.)
- **Données nettoyées** : `03_NAV_cleaned.csv` — données brutes après nettoyage, avec flags qualité

### Onglet 1 — Aperçu

Affiche les 200 premières lignes du dataset sélectionné ainsi que la liste des colonnes et leurs types.

### Onglet 2 — Qualité

Contient :

- **Tableau des valeurs manquantes** : taux de manquants par colonne, triés du plus élevé au plus faible.
- **Flags qualité pipeline** (étape 3) :
  - `is_duplicate_lot` — lot dont le numéro apparaît plusieurs fois
  - `is_outlier_feature` — valeur IQR×3 sur au moins une feature d'entrée
  - `is_suspicious_value` — valeur suspecte selon règles métier
- **Outliers par cible (Phase 2)** : comptage IQR, z-score robuste et valeurs hors plage physique pour chaque cible.
- **Statistiques de distribution** : min, max, médiane, P5, P95, skewness par cible.

### Onglet 3 — Visualisations

C'est la section principale d'analyse graphique. Elle contient quatre blocs :

#### Filtres contexte (en haut)

Permet de filtrer l'ensemble des graphes par :

- **Fournisseur** (`supplier_code`)
- **Origine / site** (`location`)
- **Grade** (`description_fr`)
- **Année de réception** (`reception_year`)

#### Options d'affichage

- **Jeu affiché** : filtre supplémentaire sur les données affichées dans les graphes
  - *Toutes les données* : aucun filtre supplémentaire
  - *Zoom P5–P95* : exclut visuellement les 5 % les plus extrêmes (garde les données, zoom l'axe)
  - *Sans outliers pipeline* : retire les lignes avec `is_outlier_feature = True`
  - *Plage physique valide* : retire les valeurs hors bornes physiques connues
- **Corrélation** : méthode Pearson ou Spearman
- **Winsoriser** : applique une winsorisation P1–P99 avant le calcul de corrélation

#### Bloc Distribution & ECDF

**Important :** Quand une **cible** est sélectionnée (OIT, IZOD, Flexion, etc.), les graphes utilisent automatiquement les **données validées Phase 2** (`target_valid_<nom>.csv`) au lieu du dataset complet.

Graphes produits :

- **Histogramme double** : à gauche toutes les valeurs, à droite un zoom P5–P95 avec lignes de percentiles et bornes physiques.
- **ECDF** (fonction de répartition cumulée empirique) : montre quelle proportion de lots a une valeur ≤ x.
- **Répartition qualité** : diagramme en barres par statut.
- **Top 10 valeurs extrêmes** : tableau des 10 lots avec les valeurs les plus hautes et les plus basses.

#### Bloc Comparaison par groupe

Violon + boxplot d'une cible ou feature, groupé par fournisseur, site ou grade. Limité aux 15 groupes les plus fréquents si plus de 20 groupes existent.

#### Bloc Nuage de points

Scatter plot de deux variables numériques quelconques. Chaque point est coloré par son **statut qualité** : bleu (normal), orange (hors plage), rouge (suspect), violet (outlier IQR), jaune (flag pipeline).

#### Bloc Corrélation

Heatmap de corrélation (Pearson ou Spearman) entre les variables sélectionnées.

---

## 6. Page Prédiction unitaire

### Objectif

Saisir manuellement les caractéristiques d'un lot et obtenir immédiatement les prédictions pour les cibles souhaitées.

### Étape 1 — Choisir l'approche de modélisation

Depuis la Phase 4, **4 approches** sont disponibles :


| Approche | Données d'entrée | Usage typique |
| -------- | ---------------- | ------------- |
| **Modèle A** | Physico-chimie uniquement | Quand seules les analyses matière sont disponibles |
| **Modèle B** | Physico-chimie + contexte opérationnel | Quand le contexte fournisseur est connu |
| **Cascade lab-assisté** | Physico-chimie (+ mesures intermédiaires optionnelles) | Quand des essais partiels (OIT, traction…) sont disponibles |
| **PSPP production** | Chaîne prédite (OOF) | Référence pour inférence sans mesures labo intermédiaires |


La case **"Comparer modèle A et modèle B"** (modèles classiques uniquement) affiche les deux prédictions côte à côte avec l'écart entre elles.

### Étape 2 — Sélectionner les cibles (modèles classiques)

Choisir une ou plusieurs cibles à prédire via le widget multiselect. Par défaut : OIT, IZOD, Flexion.

> Pour les approches **Cascade** et **PSPP**, toutes les cibles sont prédites simultanément — il n'y a pas de sélection individuelle.

### Étape 3 — Saisir les paramètres matière


| Champ | Variable | Unité | Plage plausible |
| ----- | -------- | ----- | --------------- |
| MI | `mi` | g/10min | 0 – 100 |
| HLMI | `hlmi` | g/10min | 0 – 1 000 |
| Densité | `density_g_cm3` | g/cm³ | 0.90 – 1.10 |
| Noir de carbone | `carbon_black` | % | 0 – 10 |
| Cendres | `ash` | % | 0 – 50 |
| PP | `pp` | % | 0 – 100 |
| DSC Onset | `onset` | °C | 100 – 300 |
| DSC Peak | `peak` | °C | 110 – 320 |
| Delta H | `delta_h` | J/g | 0 – 200 |
| Recyclé / Vierge | `recycled_virgin` | — | RECYCLE / VIERGE |


### Étape 4 — Mesures intermédiaires (Cascade / PSPP uniquement)

Un encart déroulant **"Mesures intermédiaires disponibles (optionnel)"** permet de fournir des valeurs laboratoire déjà mesurées pour les propriétés intermédiaires :

- **Bloc 1 — Thermique** : OIT (min)
- **Bloc 2 — Mécanique** : Traction (MPa), Flexion (MPa), IZOD (J/m), % Allongement

> **Règle de priorité :** Si vous entrez une valeur mesurée, elle est utilisée **à la place** de la prédiction du modèle pour ce bloc. Cela améliore systématiquement la précision des blocs suivants, car les modèles ont été entraînés sur des valeurs réelles.
>
> **Exemple :** Si vous avez mesuré OIT = 45 min au labo, entrez-le. Le modèle Bloc 2 utilisera cette valeur réelle plutôt que son estimation pour calculer NCLS et UCLS.

### Étape 5 — Lancer la prédiction

Cliquer sur **"Prédire"**.

L'application calcule automatiquement les **features dérivées** avant la prédiction :

- `FRR = HLMI / MI` (Flow Rate Ratio — indicateur de distribution des masses molaires)
- `log_mi = log1p(mi)`
- `log_hlmi = log1p(hlmi)`
- `charge_total = carbon_black + ash + pp`
- `thermal_window = peak - onset`

### Résultats affichés

#### Modèles classiques (A / B)

- **Qualité des données** : encadré vert (élevée) / orange (moyenne) / rouge (faible).
- **Recommandation globale** : phrase synthèse sur l'ensemble des cibles prédites.
- **Tableau des prédictions** : cible, valeur prédite avec unité, niveau de confiance, recommandation spécifique.

#### Cascade / PSPP

Les résultats sont affichés **par bloc** en 3 colonnes :

```
Bloc 1 — Thermique    |  Bloc 2 — Mécanique    |  Bloc 3 — Performance
──────────────────────|────────────────────────|──────────────────────
OIT : 40.95 min       |  Traction : 7.77 MPa   |  NCLS : 27.1 h
  ~ prédit            |  Flexion : 468.94 MPa  |  UCLS : 1.12 h
                      |  IZOD : 323.26 J/m     |
                      |  Allongement : 35.8 %  |
```

Chaque valeur est étiquetée **✓ mesuré** (vert, si vous l'avez fournie) ou **~ prédit** (bleu, si estimée par le modèle). Un graphe en barres distingue visuellement les deux sources.

#### Mode Expert (en plus)

- **Contrôle qualité avant prédiction** : liste des features manquantes, valeurs hors plage.
- **Détails techniques** : chemin du modèle chargé par cible, valeurs des features dérivées.
- **Score OOD (hors-domaine)** : indique si le lot est proche ou éloigné de la distribution d'entraînement.
- **Tableau comparaison A vs B** : prédictions des deux modèles, écart absolu, commentaire interprétatif.

### Identifiant de lot

Le champ **"Identifiant lot"** est optionnel. Il est utilisé uniquement pour le journal des prédictions (`prediction_history.csv`).

---

## 7. Page Prédiction par lot

### Objectif

Charger un fichier CSV ou Excel contenant plusieurs lots et lancer les prédictions en masse.

### Format du fichier d'entrée

Le fichier doit contenir au minimum les colonnes du Modèle A. Les noms de colonnes doivent correspondre aux noms standardisés (voir section 12).

**Colonnes minimales recommandées :**

```
mi, hlmi, density_g_cm3, carbon_black, ash, pp, onset, peak, delta_h, recycled_virgin
```

Pour le Modèle B, ajouter :

```
supplier_code, location, description_fr, reception_year
```

Pour la Cascade / PSPP avec mesures intermédiaires, ajouter si disponibles :

```
oit_min, traction, flexion, izod, pct_elongation
```

Une colonne `lot_id` ou `id` permet d'identifier chaque ligne dans les résultats.

### Étapes

1. **Charger le fichier** via le bouton d'upload (CSV ou Excel).
2. **Vérifier l'aperçu** : les 20 premières lignes sont affichées.
3. **Choisir l'approche** (A, B, Cascade ou PSPP) et **les cibles** si applicable.
4. Cliquer sur **"Lancer les prédictions"**.

> En mode Expert, les plages plausibles par feature sont affichées dans un encart déroulant avant la prédiction.

### Résultats

#### Modèles classiques (A / B)

Un tableau est produit avec, pour chaque lot et chaque cible :

- `predicted_<cible>` — valeur prédite
- `confidence_<cible>` — score de confiance 0–1
- `confidence_level_<cible>` — élevée / moyenne / faible
- `recommendation_<cible>` — recommandation spécifique
- `prediction_status` — `ok` ou code d'erreur

#### Cascade / PSPP

Les résultats sont présentés en **3 expandeurs par bloc** :

- **Bloc 1 — Structure thermique** : colonnes `cascade_oit_min`
- **Bloc 2 — Structure mécanique** : colonnes `cascade_traction`, `cascade_flexion`, `cascade_izod`, `cascade_pct_elongation`
- **Bloc 3 — Performance finale** : colonnes `cascade_ncls`, `cascade_ucls`

Un graphe de distribution est disponible pour chaque cible prédite.

### Export

Deux boutons permettent d'exporter les résultats :

- **Exporter CSV** : fichier `.csv` encodé UTF-8 avec BOM (compatible Excel).
- **Exporter Excel** : fichier `.xlsx` (nécessite `openpyxl`).

Le nom de fichier inclut automatiquement l'horodatage : `predictions_cascade_YYYYMMDD_HHMMSS.csv`.

### Journal des prédictions

Toutes les prédictions (unitaires et par lot, toutes approches) sont automatiquement ajoutées à :

```
outputs/prediction_logs/prediction_history.csv
```

Ce journal contient : lot_id, cible, modèle/approche, features d'entrée, prédiction, confiance, recommandation, horodatage.

---

## 8. Page Explicabilité

> **Disponible en mode Expert uniquement.**

### Objectif

Comprendre **quelles features ont le plus influencé** la prédiction d'une cible donnée.

### Utilisation

1. Sélectionner une **cible** (OIT, IZOD, etc.).
2. Sélectionner le **modèle** (A ou B).
3. Le graphe d'importance des features s'affiche automatiquement (top 15 features).

### Source des importances

L'application cherche les importances dans cet ordre :

1. `outputs/17_explainability_after_cleaning/feature_importance_by_target.csv` (Phase 2, recommandé)
2. `outputs/tables/10_feature_importance_by_target.csv` (Phase 1, fallback)
3. Si aucun fichier n'est disponible : **calcul à la volée** sur un échantillon du dataset NAV.

### Lecture du graphe

Le graphe en barres horizontales montre le **score d'importance** de chaque feature (plus la barre est longue, plus la feature a de poids dans le modèle). Pour les forêts aléatoires et Extra Trees, il s'agit de l'importance MDI (Mean Decrease in Impurity).

### Comparaison A vs B

Un encart déroulant "Comparaison A vs B (top 10)" affiche les importances des deux modèles côte à côte. Cela permet de voir quelles features opérationnelles (fournisseur, site, grade) ajoutent de l'information par rapport aux features matière seules.

---

## 9. Page Comparaison des modèles

### Objectif

Consulter les métriques historiques des différentes phases du pipeline (baselines, phase 2, validation robuste).

> **Pour les décisions industrielles**, privilégiez la page **Fiabilité des modèles** (phase 5) plutôt que les sources « random CV » de cette page.

### Sélectionner la source de métriques

| Source | Fichier | Ce qu'elle montre |
| ------ | ------- | ----------------- |
| Référence ML (phase 1) | `metrics/07_baseline_results_regression.csv` | Performance des modèles de base (RF, GB, ET, Ridge…) |
| Apprentissage profond | `metrics/08_dl_results_regression.csv` | Performance des réseaux de neurones |
| ML après correction (phase 2) | `15_modeling_after_target_cleaning/ml_regression_results.csv` | Performance après validation et correction des cibles |
| Avant / après correction | `16_before_after_comparison/before_after_summary_by_target.csv` | Gain de la correction Phase 2 par cible |
| **Validation robuste (phase 3)** | `21_robust_generalization/robust_validation_results.csv` | **Généralisation GroupKFold et temporelle** |
| A vs B robuste | `23_model_A_vs_B_robust/model_A_vs_B_robust.csv` | Comparaison Modèle A et B sur validation robuste |

### Interprétation des résultats (alignée phase 3 / 5)

| Cible | Tier | Statut déploiement | Lecture |
| ----- | ---- | ------------------ | ------- |
| `oit_min` | A | Prudence | Bon en random CV, dégradation sur nouveau grade/site |
| `flexion`, `pct_elongation` | A | Prudence | Utiles en tri préliminaire, confirmer sur nouveau fournisseur |
| `traction` | B | Prudence | Signal matière partiel, fort effet process |
| `ncls` | B | Labo requis | R² group fournisseur fortement négatif — screening uniquement |
| `izod` | C | Bloqué | Non déployable sans revalidation |
| `ucls` | C | Bloqué | Effectif insuffisant (~86 lots) |

---

## 10. Page Fiabilité des modèles

### Objectif

**Page de référence scientifique** — consolide la validation robuste (phase 3) et les statuts de déploiement (phase 5). À consulter **avant** d'utiliser une prédiction pour une décision d'achat ou de formulation.

### Prérequis

```bash
python run_phase2.py
python run_phase3.py
python run_scientific_evaluation.py
```

### Onglets

| Onglet | Contenu |
| ------ | ------- |
| **Vue d'ensemble** | Statuts déploiement par cible (modèle A), tableau recommandé |
| **Protocoles CV** | R² moyen par schéma (random, group fournisseur, grade, temporel) + heatmap |
| **Par cible** | Comparaison R² et MAE random vs group fournisseur + justification métier |
| **Diagnostics** | Parity plot (prédit vs réel) depuis les prédictions CV phase 3 |

### Statuts de déploiement

| Statut | Signification | Action recommandée |
| ------ | ------------- | ------------------ |
| **Déployable** | Généralisation fournisseur acceptable | Tri préliminaire autorisé avec surveillance |
| **Prudence** | Performance interne OK, généralisation limitée | Utiliser avec vérification labo ciblée |
| **Labo requis** | R² group négatif ou dégradation MAE > 2× | Ne pas décider seul sur la prédiction |
| **Bloqué** | Données ou métriques insuffisantes | Collecter données / revalidation |

### Ordre de lecture des métriques

1. **GroupKFold fournisseur** — peut-on généraliser à un nouveau fournisseur ?
2. **GroupKFold grade / site** — sensibilité au grade ou au site
3. **Validation temporelle** — dérive dans le temps
4. **KFold aléatoire** — référence interne uniquement (souvent optimiste)

### Statuts actuels (référence mai 2026)

| Cible | Modèle reco. | Statut | R² random | R² group fournisseur |
| ----- | ------------ | ------ | --------- | -------------------- |
| `oit_min` | B | Prudence | 0.80 | 0.52 |
| `flexion` | A | Prudence | 0.46 | 0.11 |
| `pct_elongation` | B | Prudence | 0.73 | 0.23 |
| `traction` | B | Prudence | 0.58 | 0.02 |
| `ncls` | B | Labo requis | 0.81 | **−5.1** |
| `izod` | B | Bloqué | 0.89 | **−7.2** |
| `ucls` | A | Bloqué | −0.21 | N/A |

---

## 11. Page Cascade / PSPP

### Objectif

Comparer les **3 approches de modélisation** et permettre une prédiction en cascade en temps réel.

> **Important :** la cascade *lab-assistée* utilise les vraies mesures intermédiaires en évaluation — elle est **optimiste**. **PSPP (OOF)** est la référence pour l'inférence sans mesures labo.

### Architecture de la cascade

```
Bloc 1 — Structure thermique
  Entrées  : Matière (MI, HLMI, Densité, CB, DSC…)
  Sortie   : OIT (min)
       ↓
Bloc 2 — Structure mécanique
  Entrées  : Matière + OIT (Bloc 1)
  Sorties  : Traction (MPa), Flexion (MPa), IZOD (J/m), % Allongement
       ↓
Bloc 3 — Performance finale
  Entrées  : Matière + OIT + Traction + Flexion + IZOD + % Allongement
  Sorties  : NCLS (h), UCLS (h)
```

### Onglets disponibles

| Onglet | Contenu |
|---|---|
| **Architecture** | Schéma de la cascade + distinction training / inférence |
| **R² par approche** | Direct / Cascade lab-assisté / PSPP production |
| **MAE par approche** | Idem — plus bas = mieux |
| **Gain cascade** | Gain lab-assisté vs direct sur NCLS/UCLS ; comparer aussi PSPP |
| **Prédiction live** | Formulaire simultané par les 3 approches |

### Résultats de performance (datasets validés phase 2)

| Cible | Direct R² | Cascade lab-assisté R² | PSPP production R² |
|---|---|---|---|
| `oit_min` | 0.721 | 0.719 | 0.721 |
| `traction` | 0.556 | 0.531 | 0.483 |
| `flexion` | 0.459 | 0.488 | 0.428 |
| `izod` | 0.873 | 0.918 | 0.905 |
| `pct_elongation` | 0.698 | 0.743 | 0.729 |
| `ncls` | 0.758 | **0.820** | **−0.151** |
| `ucls` | −0.060 | −0.022 | 0.071 |

**Gain NCLS/UCLS (cascade lab-assisté vs direct) :** NCLS +0.062 R², UCLS +0.038 R².

**Lecture polymères :** le gain cascade sur NCLS disparaît en mode PSPP production (−0.91 R² vs direct). La cascade n'apporte un avantage net que si des **mesures intermédiaires réelles** (OIT, traction…) sont injectées — comme en R&D avec essais partiels.

### Différence Training vs Inference

| Phase | Valeurs intermédiaires utilisées |
|---|---|
| **Entraînement cascade** | Valeurs **réelles mesurées** en laboratoire |
| **Évaluation PSPP (OOF)** | Prédictions out-of-fold — **référence production** |
| **Inférence sans mesures** | Prédictions du bloc précédent (propagation d'erreur) |
| **Inférence avec mesures** | Valeurs labo fournies — condition optimale |

> Pour NCLS/UCLS : **fournir OIT mesuré** améliore la chaîne ; sans mesures, préférer le modèle **direct** ou **PSPP** selon la page Cascade.

### Lancer les modèles cascade

```bash
# Option 1 — Phase 4 seule
python run_phase4.py

# Option 2 — Phase 4 + consolidation dashboard (recommandé)
python run_scientific_evaluation.py --with-cascade
```

Les modèles sont sauvegardés dans `outputs/24_cascade_modeling/models/` (datasets **validés phase 2**).

---

## 12. Scores de confiance et recommandations

### Comment est calculé le score de confiance ?

Le score de confiance est un nombre entre 0 et 1 calculé automatiquement pour chaque prédiction. Il combine plusieurs facteurs :

```
score = f(complétude des features, valeurs hors plage, confiance a priori du modèle, score OOD)
```

**Détail du calcul :**


| Facteur | Impact |
| ------- | ------ |
| Complétude des features (`completeness_pct`) | score × (0.3 + 0.7 × complétude) |
| Valeur hors plage physique | score × 0.75 |
| Feature dérivée impossible (ex. FRR si MI=0) | score × 0.85 |
| Trop de features manquantes (≥ 4) | score × 0.80 |
| Confiance a priori `faible` (cible peu fiable / statut lab_only) | score × 0.70 |
| Confiance a priori `moyenne` | score × 0.90 |
| Score OOD < 45 % (lot hors-domaine) | score × 0.75 |


**Niveaux finaux :**

- `élevée` : score ≥ 0.70
- `moyenne` : score ≥ 0.45
- `faible` : score < 0.45

> **Note :** Pour les approches Cascade et PSPP, le score de confiance n'est pas calculé au niveau des blocs individuels — les prédictions sont toujours indiquées avec la source (mesuré / prédit).

### Règles de recommandation automatique


| Situation | Recommandation |
| --------- | -------------- |
| Qualité données < 40 % ou niveau faible | "Compléter les analyses avant décision" |
| Confiance faible + cible NCLS, UCLS ou IZOD | "Labo obligatoire — cible non déployable en autonomie" |
| Confiance faible + autre cible | "Test laboratoire recommandé" |
| Prédiction haute + confiance élevée | "Lot intéressant" |
| Prédiction haute + confiance moyenne | "Lot prometteur — confirmer au labo" |
| Prédiction basse + confiance élevée | "Lot à éviter ou reformuler" |
| Confiance moyenne, cas général | "Prédiction prudente" |


---

## 13. Référence technique : features et cibles

### Features Modèle A (physico-chimie matière)


| Variable | Description | Famille |
| -------- | ----------- | ------- |
| `mi` | Melt Index (indice de fluidité standard) | Rhéologie |
| `hlmi` | High Load Melt Index | Rhéologie |
| `frr` | Flow Rate Ratio = HLMI / MI | Rhéologie (dérivée) |
| `log_mi` | log1p(MI) | Rhéologie (dérivée) |
| `log_hlmi` | log1p(HLMI) | Rhéologie (dérivée) |
| `density_g_cm3` | Densité du granulé (g/cm³) | Densité |
| `density_plaque_g_cm3` | Densité de la plaque moulée | Densité |
| `carbon_black` | Teneur en noir de carbone (%) | Charges |
| `ash` | Teneur en cendres (%) | Charges |
| `pp` | Teneur en polypropylène (%) | Charges |
| `charge_total` | carbon_black + ash + pp | Charges (dérivée) |
| `onset` | Température d'onset DSC (°C) | Analyse thermique |
| `peak` | Température de pic DSC (°C) | Analyse thermique |
| `delta_h` | Enthalpie de fusion DSC (J/g) | Analyse thermique |
| `thermal_window` | peak - onset (°C) | Analyse thermique (dérivée) |
| `recycled_virgin` | Nature du matériau (RECYCLE / VIERGE) | Composition |


### Features supplémentaires Modèle B (contexte opérationnel)


| Variable | Description |
| -------- | ----------- |
| `supplier_code` | Code fournisseur |
| `supplier_name` | Nom du fournisseur |
| `location` | Site / entrepôt de réception |
| `description_fr` | Grade commercial (désignation) |
| `reception_year` | Année de réception |
| `reception_month` | Mois de réception |
| `reception_quarter` | Trimestre de réception |


### Features intermédiaires injectées par la cascade


| Variable | Rôle dans la cascade | Bloc qui la produit |
| -------- | -------------------- | ------------------- |
| `oit_min` | Feature de Blocs 2 et 3 | Bloc 1 (Thermique) |
| `traction` | Feature de Bloc 3 | Bloc 2 (Mécanique) |
| `flexion` | Feature de Bloc 3 | Bloc 2 (Mécanique) |
| `izod` | Feature de Bloc 3 | Bloc 2 (Mécanique) |
| `pct_elongation` | Feature de Bloc 3 | Bloc 2 (Mécanique) |


### Nouveaux champs du dataset mai 2026


| Variable | Description |
| -------- | ----------- |
| `escr` | ESCR (Environmental Stress Cracking Resistance) |
| `flag_density` | Flag qualité densité |
| `lot_search` | Référence de recherche lot |


### Bornes physiques des features d'entrée


| Feature | Min plausible | Max plausible | Unité |
| ------- | ------------- | ------------- | ----- |
| `mi` | 0 | 100 | g/10min |
| `hlmi` | 0 | 1 000 | g/10min |
| `density_g_cm3` | 0.90 | 1.10 | g/cm³ |
| `carbon_black` | 0 | 10 | % |
| `ash` | 0 | 50 | % |
| `pp` | 0 | 100 | % |
| `onset` | 100 | 300 | °C |
| `peak` | 110 | 320 | °C |
| `delta_h` | 0 | 200 | J/g |
| `frr` | 0 | 500 | ratio |


### Flags qualité dans le dataset


| Flag | Signification |
| ---- | ------------- |
| `is_duplicate_lot` | Le numéro de lot apparaît plusieurs fois dans NAV |
| `is_outlier_feature` | Au moins une feature d'entrée est hors IQR×3 |
| `is_suspicious_value` | Valeur suspecte selon règles métier |
| `has_rheology_data` | MI ou HLMI disponible |
| `has_density_data` | Densité disponible |
| `has_charge_data` | Au moins une charge disponible |
| `has_thermal_data` | Données DSC disponibles |
| `has_complete_core_features` | Toutes les features cœur disponibles |


---

## 14. Workflow typique d'utilisation

### Cas 0 — Vérifier si une cible est déployable (nouveau)

```
1. Ouvrir "Fiabilité des modèles"
2. Consulter l'onglet "Vue d'ensemble" — statuts par cible
3. Pour une cible précise : onglet "Par cible" → comparer random vs group fournisseur
4. Si statut "Labo requis" ou "Bloqué" → ne pas utiliser la prédiction seule pour décider
```

### Cas 1 — Évaluation rapide d'un nouveau lot à la réception

```
1. Ouvrir "Prédiction unitaire"
2. Sélectionner Modèle A
3. Saisir MI, HLMI, densité, noir de carbone, cendres
   (les autres champs peuvent rester à 0 si non disponibles)
4. Sélectionner les cibles OIT + IZOD + Flexion
5. Cliquer "Prédire"
6. Lire la recommandation globale et le niveau de confiance
   → Confiance élevée + "Lot intéressant" = prioriser ce lot
   → Confiance faible = attendre les analyses labo complètes
```

### Cas 2 — Prédire NCLS/UCLS avec OIT mesuré (cascade)

```
1. Ouvrir "Prédiction unitaire"
2. Sélectionner l'approche **"Cascade lab-assisté"**
3. Saisir les paramètres matière habituels
4. Ouvrir l'encart "Mesures intermédiaires disponibles"
   → Entrer la valeur OIT mesurée en laboratoire
5. Cliquer "Prédire"
6. Lire les résultats Bloc 3 (NCLS, UCLS) — étiquetés
   "✓ mesuré" pour OIT, "~ prédit" pour les autres
```

### Cas 3 — Analyse d'un fournisseur

```
1. Ouvrir "Explorateur de données" → onglet Visualisations
2. Filtrer par fournisseur (menu "Fournisseur" en haut)
3. Sélectionner "Comparaison par groupe" → Variable = OIT, Grouper par = Grade
4. Observer la dispersion et la médiane par grade de ce fournisseur
5. Comparer avec d'autres fournisseurs en changeant le filtre
```

### Cas 4 — Prédiction en masse sur un fichier de réception

```
1. Préparer un fichier CSV avec les colonnes MI, HLMI, density_g_cm3,
   carbon_black, ash, pp, onset, peak, delta_h (et un ID de lot)
2. Ouvrir "Prédiction par lot"
3. Charger le fichier → vérifier l'aperçu
4. Choisir l'approche (A, B, Cascade ou PSPP)
5. Cliquer "Lancer les prédictions"
6. Consulter les résultats groupés par bloc (Cascade)
7. Exporter en CSV ou Excel → partager avec l'équipe
```

### Cas 5 — Comparer les 3 approches sur une formulation

```
1. Ouvrir "Cascade / PSPP" → onglet "Prédiction live"
2. Saisir les paramètres matière
3. Cliquer "Prédire avec les 3 approches"
4. Comparer les prédictions Direct vs Cascade vs PSPP
   dans le tableau résultats
```

### Cas 6 — Comprendre pourquoi un lot a un OIT prédit faible

```
1. Mode Expert : activer dans la barre latérale
2. Ouvrir "Prédiction unitaire" → saisir les données du lot
3. Après prédiction : consulter "Détails techniques"
   → Vérifier les valeurs des features dérivées (FRR, charge_total)
4. Ouvrir "Explicabilité" → sélectionner OIT, Modèle A
   → Identifier les features avec la plus forte importance
5. Vérifier si les valeurs du lot sont dans les zones à risque
   via "Explorateur de données" → Nuage de points (feature vs OIT)
```

---

## 15. Limites connues et conseils

### Limites du modèle (référence phase 5)

| Cible | Statut | Limite principale | Conseil |
| ----- | ------ | ----------------- | ------- |
| `oit_min` | Prudence | Dégradation sur nouveau grade/site | Confirmer au labo si OIT critique ; Modèle B si fournisseur connu |
| `flexion`, `pct_elongation` | Prudence | Généralisation fournisseur limitée | Tri préliminaire, pas spec finale |
| `traction` | Prudence | Fort effet process non mesuré | Modèle B si contexte connu |
| `ncls` | Labo requis | R² group fournisseur négatif | Screening uniquement |
| `izod` | Bloqué | Variabilité extrême, non généralisable | Revalidation requise avant tout usage |
| `ucls` | Bloqué | ~86 échantillons labellisés | Collecter données avant modélisation |
| `cell_class` | Classification | Pas de régression continue | Classification PE uniquement |
| `temp_c` | Hors scope | Variable procédé | Entrée descriptive, pas cible |


### Limites spécifiques à la cascade

- **Cascade lab-assistée vs production** : les métriques cascade utilisent des intermédiaires *mesurés* — elles surestiment la performance sans essais labo. **PSPP (OOF)** est la référence honnête.
- **NCLS en PSPP** : R² fortement négatif vs direct — ne pas déployer la cascade seule pour NCLS sans mesures intermédiaires.
- **Dépendance en chaîne** : une erreur au Bloc 1 se propage ; atténuer en injectant OIT/traction mesurés.
- **Effectif Bloc 3** : seules les lignes avec toutes les mesures intermédiaires sont utilisées à l'entraînement.

### Bonnes pratiques

- Consulter **Fiabilité des modèles** avant toute décision sur une nouvelle cible ou un nouveau fournisseur.
- Toujours **vérifier le niveau de confiance** avant d'utiliser une prédiction pour achat ou formulation.
- Le **Modèle B** aide quand le fournisseur est **déjà connu** dans NAV — pas pour un fournisseur totalement nouveau.
- **Cascade avec OIT mesuré** : utile en R&D pour NCLS/UCLS ; sans mesures, préférer direct ou PSPP.
- Si le **score OOD est faible** (< 45 %), le lot est hors distribution — prudence maximale.
- Cibles **lab_only** ou **blocked** : outil de tri au mieux, jamais décision finale seule.

### Données manquantes

L'application tolère des features manquantes et les signale dans le contrôle qualité. Cependant :

- Moins de 50 % de features renseignées → qualité **faible**, confiance **faible**
- Entre 50 et 75 % → qualité **moyenne**
- Plus de 75 % sans valeur hors plage → qualité **élevée**

---

*Guide mis à jour — Soleno NAV Dashboard v5 (Phases 1–5 + évaluation scientifique unifiée) — Dataset mai 2026 (15 013 lots)*
