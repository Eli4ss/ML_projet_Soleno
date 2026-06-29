"""Configuration centralisée du pipeline Soleno NAV."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Priorité au dataset mis à jour mai 2026
_RAW_UPDATED = PROJECT_ROOT / "data" / "raw" / "Resine_update_mai2026.xlsx"
_RAW_LEGACY  = PROJECT_ROOT / "data" / "raw" / "Resine.xlsx"
DATA_RAW = _RAW_UPDATED if _RAW_UPDATED.exists() else _RAW_LEGACY
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
TARGET_DATASETS = PROJECT_ROOT / "data" / "target_datasets"
OUTPUTS = PROJECT_ROOT / "outputs"
REPORTS = OUTPUTS / "reports"
FIGURES = OUTPUTS / "figures"
METRICS = OUTPUTS / "metrics"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
NOTEBOOKS = PROJECT_ROOT / "notebooks"
APP_DIR = PROJECT_ROOT / "app"

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_FOLDS = 5
MIN_TARGET_SAMPLES = 30

# Cibles physiques (propriétés à prédire)
TARGET_COLUMNS = [
    "oit_min",
    "ncls",
    "ucls",
    "izod",
    "traction",
    "pct_elongation",
    "flexion",
    "cell_class",
    "temp_c",
]

# Cibles pour modélisation après validation phase 2 (temp_c = variable procédé)
TARGET_COLUMNS_MODELING = [c for c in TARGET_COLUMNS if c != "temp_c"]

# Dossiers phase 2 (validation cibles + re-modélisation)
PHASE2_VALIDATION = OUTPUTS / "14_target_validation"
PHASE2_MODELING = OUTPUTS / "15_modeling_after_target_cleaning"
PHASE2_COMPARISON = OUTPUTS / "16_before_after_comparison"
PHASE2_EXPLAIN = OUTPUTS / "17_explainability_after_cleaning"
PHASE2_UNCERTAINTY = OUTPUTS / "18_uncertainty_after_cleaning"
PHASE2_REGISTRY = OUTPUTS / "19_updated_model_registry"
PHASE2_FINAL_REPORT = OUTPUTS / "reports" / "20_improvement_final_report.md"

# Résultats phase 1 (référence avant/après)
PHASE1_METRICS = METRICS

# Phase 3 — généralisation industrielle
PHASE3_ROBUST = OUTPUTS / "21_robust_generalization"
PHASE3_ERRORS = OUTPUTS / "22_error_analysis_by_group"
PHASE3_COMPARISON = OUTPUTS / "23_model_A_vs_B_robust"

# Colonnes de groupement (origine = site / location NAV)
GROUP_COLUMNS = {
    "supplier": "supplier_code",
    "grade": "description_fr",
    "origin": "location",
}
TEMPORAL_COLUMN = "reception_year"
TEMPORAL_PERIOD_COLUMN = "reception_quarter"

# Features modèle B à exclure lors du GroupKFold sur la même dimension (anti-fuite)
GROUP_FEATURE_EXCLUSIONS = {
    "supplier": ["supplier_code", "supplier_name"],
    "grade": ["description_fr"],
    "origin": ["location"],
}

MIN_SAMPLES_ROBUST_CV = 50
MIN_GROUPS_ROBUST_CV = 4
MAX_GROUPK_FOLDS = 5

# ===== Phase 4 — Cascade / PSPP modeling =====

# Bloc 1 : Structure thermique (prédit uniquement depuis matière brute)
CASCADE_THERMAL_TARGETS = ["oit_min"]

# Bloc 2 : Structure mécanique (matière + outputs thermiques mesurés/prédits)
CASCADE_MECHANICAL_TARGETS = ["traction", "flexion", "izod", "pct_elongation"]

# Bloc 3 : Performance finale (matière + thermique + mécanique)
CASCADE_FINAL_TARGETS = ["ncls", "ucls"]

# Topologie complète : pour chaque bloc, quels outputs des blocs précédents injecter
CASCADE_BLOCS: dict = {
    "thermal": {
        "targets": CASCADE_THERMAL_TARGETS,
        "extra_inputs": [],
    },
    "mechanical": {
        "targets": CASCADE_MECHANICAL_TARGETS,
        "extra_inputs": CASCADE_THERMAL_TARGETS,
    },
    "final": {
        "targets": CASCADE_FINAL_TARGETS,
        "extra_inputs": CASCADE_THERMAL_TARGETS + CASCADE_MECHANICAL_TARGETS,
    },
}

# Ordre d'exécution des blocs
CASCADE_BLOC_ORDER = ["thermal", "mechanical", "final"]

# Dossiers Phase 4
PHASE4_CASCADE = OUTPUTS / "24_cascade_modeling"
PHASE4_COMPARISON = OUTPUTS / "25_approach_comparison"

# Libellés UI — comparaison direct / cascade / PSPP (clé technique inchangée : pspp)
CASCADE_APPROACH_LABELS = {
    "direct": "Direct (Matière → cible)",
    "cascade": "Cascade lab-assistée (intermédiaires mesurés)",
    "pspp": "PSPP sans laboratoire (chaîne entièrement prédite)",
}

CASCADE_APPROACH_HELP = {
    "direct": "Une seule étape : propriétés matière → propriété cible.",
    "cascade": (
        "Les blocs suivants reçoivent des mesures intermédiaires réelles "
        "(OIT, traction, flexion…). Référence optimiste si le labo est disponible."
    ),
    "pspp": (
        "Chaque bloc utilise les prédictions du bloc précédent — sans mesures labo intermédiaires. "
        "Les erreurs se propagent dans toute la chaîne. L'évaluation out-of-fold (OOF) est la "
        "référence honnête pour ce mode d'inférence."
    ),
}


def cascade_approach_label(key: str) -> str:
    return CASCADE_APPROACH_LABELS.get(str(key), str(key))

# Phase 5 — évaluation scientifique unifiée (référence dashboard)
PHASE5_UNIFIED = OUTPUTS / "26_unified_evaluation"

# Features Model A (matière / scientifique)
MODEL_A_FEATURES = [
    "recycled_virgin",
    "fluidity_g_10min",
    "mi",
    "hlmi",
    "density_g_cm3",
    "density_plaque_g_cm3",
    "carbon_black",
    "ash",
    "onset",
    "peak",
    "delta_h",
    "pp",
    "frr",
    "log_mi",
    "log_hlmi",
    "mi_density_ratio",
    "hlmi_density_ratio",
    "charge_total",
    "carbon_black_to_ash_ratio",
    "pp_indicator",
    "thermal_window",
    "crystallinity_estimated",
    "rheology_class",
    "density_class",
    "charge_class",
]

# Features additionnelles Model B (opérationnel)
MODEL_B_EXTRA = [
    "supplier_code",
    "supplier_name",
    "location",
    "description_fr",
    "reception_year",
    "reception_month",
    "reception_quarter",
]

# Identifiants exclus des modèles
ID_COLUMNS = [
    "auto_no",
    "numero",
    "item_no",
    "no2",
    "purchase_order",
]

# Colonnes de flags qualité
FLAG_COLUMNS = [
    "is_missing_target",
    "is_outlier_feature",
    "is_suspicious_value",
    "is_duplicate_lot",
    "has_complete_core_features",
    "has_rheology_data",
    "has_density_data",
    "has_charge_data",
    "has_thermal_data",
]
