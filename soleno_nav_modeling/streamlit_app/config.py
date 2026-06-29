"""Configuration centralisée de l'app Streamlit MVP Soleno NAV."""
from __future__ import annotations

from pathlib import Path

# Racine streamlit_app/
APP_ROOT = Path(__file__).resolve().parent
# Racine projet pipeline (soleno_nav_modeling/)
NAV_ROOT = APP_ROOT.parent
if not (NAV_ROOT / "outputs").exists():
    NAV_ROOT = APP_ROOT


def _load_pipeline_config():
    """Charge src/config.py du pipeline sans enregistrer le package `src` pipeline."""
    import importlib.util

    path = NAV_ROOT / "src" / "config.py"
    spec = importlib.util.spec_from_file_location("_soleno_nav_pipeline_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Impossible de charger {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_pc = _load_pipeline_config()
CASCADE_APPROACH_HELP = _pc.CASCADE_APPROACH_HELP
CASCADE_APPROACH_LABELS = _pc.CASCADE_APPROACH_LABELS
CASCADE_FINAL_TARGETS = _pc.CASCADE_FINAL_TARGETS
CASCADE_MECHANICAL_TARGETS = _pc.CASCADE_MECHANICAL_TARGETS
CASCADE_THERMAL_TARGETS = _pc.CASCADE_THERMAL_TARGETS
cascade_approach_label = _pc.cascade_approach_label

OUTPUTS = NAV_ROOT / "outputs"
DATA = NAV_ROOT / "data"
MODELS = OUTPUTS / "models"
TABLES = OUTPUTS / "tables"
METRICS = OUTPUTS / "metrics"
REPORTS = OUTPUTS / "reports"

# Fichiers données
PATHS = {
    "feature_engineered": TABLES / "04_NAV_feature_engineered.csv",
    "cleaned": TABLES / "03_NAV_cleaned.csv",
    "feature_decision": TABLES / "02_feature_decision_table.csv",
    "registry_phase2": OUTPUTS / "19_updated_model_registry" / "updated_best_model_registry.csv",
    "registry_phase1": METRICS / "12_preliminary_best_model_registry.csv",
    "importance_phase2": OUTPUTS / "17_explainability_after_cleaning" / "feature_importance_by_target.csv",
    "importance_phase1": TABLES / "10_feature_importance_by_target.csv",
    "baseline_reg": METRICS / "07_baseline_results_regression.csv",
    "baseline_cls": METRICS / "07_baseline_results_classification.csv",
    "dl_reg": METRICS / "08_dl_results_regression.csv",
    "ml_after_cleaning": OUTPUTS / "15_modeling_after_target_cleaning" / "ml_regression_results.csv",
    "before_after": OUTPUTS / "16_before_after_comparison" / "before_after_summary_by_target.csv",
    "robust_validation": OUTPUTS / "21_robust_generalization" / "robust_validation_results.csv",
    "ab_robust": OUTPUTS / "23_model_A_vs_B_robust" / "model_A_vs_B_robust.csv",
    "models_saved_phase2": MODELS / "19_saved_models",
    "models_saved_phase1": MODELS / "12_saved_models",
    "target_outlier_summary": OUTPUTS / "14_target_validation" / "target_outlier_summary.csv",
    "target_distribution_summary": OUTPUTS / "14_target_validation" / "target_distribution_summary.csv",
    "target_quality_flags_summary": OUTPUTS / "14_target_validation" / "target_quality_flags_summary.csv",
}

TARGETS = {
    "oit_min": {"label": "OIT (min)", "unit": "min"},
    "ncls": {"label": "NCLS", "unit": "h"},
    "ucls": {"label": "UCLS", "unit": "h"},
    "izod": {"label": "IZOD", "unit": "J/m"},
    "traction": {"label": "Traction", "unit": "MPa"},
    "pct_elongation": {"label": "% Allongement", "unit": "%"},
    "flexion": {"label": "Flexion", "unit": "MPa"},
    "cell_class": {"label": "Cell Class", "unit": "code"},
}

MVP_TARGETS = ["oit_min", "ncls", "ucls", "izod", "traction", "pct_elongation", "flexion"]

FILTER_COLUMNS = {
    "fournisseur": "supplier_code",
    "origine": "location",
    "grade": "description_fr",
    "année": "reception_year",
}

TARGET_DECISION_TABLE_CSV = OUTPUTS / "14_target_validation" / "target_modeling_decision_table.csv"
UPDATED_REGISTRY_CSV = PATHS["registry_phase2"]
ROBUST_VALIDATION_CSV = PATHS["robust_validation"]
PREDICTION_LOGS_DIR = OUTPUTS / "prediction_logs"

PIPELINE_STEPS = [
    "Audit NAV",
    "Nettoyage & standardisation",
    "Feature engineering",
    "Datasets par cible",
    "Baselines ML",
    "Deep Learning",
    "Évaluation standard",
    "Validation des cibles (phase 2)",
    "Modélisation après correction",
    "Validation robuste (phase 3)",
    "Cascade / PSPP (phase 4)",
    "Évaluation unifiée (phase 5)",
]

# Phase 4 — Cascade modeling
PHASE4_MODELS_DIR = OUTPUTS / "24_cascade_modeling" / "models"
PHASE4_METRICS_DIR = OUTPUTS / "24_cascade_modeling" / "metrics"
PHASE4_COMPARISON_DIR = OUTPUTS / "25_approach_comparison"
PHASE5_UNIFIED_DIR = OUTPUTS / "26_unified_evaluation"

PATHS["cascade_all_results"] = PHASE4_METRICS_DIR / "cascade_all_results.csv"
PATHS["cascade_best_per_approach"] = PHASE4_COMPARISON_DIR / "best_per_approach.csv"
PATHS["cascade_pivot_r2"] = PHASE4_COMPARISON_DIR / "pivot_r2.csv"
PATHS["cascade_pivot_mae"] = PHASE4_COMPARISON_DIR / "pivot_mae.csv"
PATHS["cascade_winner"] = PHASE4_COMPARISON_DIR / "winner_per_target.csv"
PATHS["cascade_gain"] = PHASE4_COMPARISON_DIR / "cascade_gain.csv"

PATHS["deployment_status"] = PHASE5_UNIFIED_DIR / "target_deployment_status.csv"
PATHS["recommended_model"] = PHASE5_UNIFIED_DIR / "recommended_model_per_target.csv"
PATHS["master_metrics_long"] = PHASE5_UNIFIED_DIR / "master_metrics_long.csv"
PATHS["robust_predictions"] = OUTPUTS / "21_robust_generalization" / "robust_cv_predictions.csv"

DEPLOYMENT_STATUS_COLORS = {
    "deploy": "#2ca02c",
    "caution": "#ff7f0e",
    "lab_only": "#d62728",
    "blocked": "#7f7f7f",
}

DEPLOYMENT_STATUS_LABELS = {
    "deploy": "Déployable (domaine connu)",
    "caution": "Prudence",
    "lab_only": "Labo requis",
    "blocked": "Bloqué",
}
