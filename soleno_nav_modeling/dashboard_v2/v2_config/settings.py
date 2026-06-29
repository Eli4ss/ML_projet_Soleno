"""Configuration UI et chemins spécifiques dashboard_v2."""
from __future__ import annotations

from pathlib import Path

V2_ROOT = Path(__file__).resolve().parent.parent
NAV_ROOT = V2_ROOT.parent
CONFIG_DIR = Path(__file__).resolve().parent

MATURITY_OVERRIDES_YAML = CONFIG_DIR / "maturity_overrides.yaml"
PILOT_JOURNAL_DIR = NAV_ROOT / "outputs" / "prediction_logs"
PILOT_JOURNAL_FILE = PILOT_JOURNAL_DIR / "pilot_journal.csv"

MATURITY_LEVELS = {
    "delivered": {
        "key": "delivered",
        "label": "Analyse R&D",
        "short": "L1",
        "color": "#2563eb",
        "bg": "#eff6ff",
        "description": "Exploration des données, qualité et validation des modèles — outils d'aide à l'analyse.",
    },
    "pilot": {
        "key": "pilot",
        "label": "Pilote contrôlé",
        "short": "L2",
        "color": "#d97706",
        "bg": "#fffbeb",
        "description": "Prédictions supervisées avec journalisation et comparaison laboratoire.",
    },
    "experimental": {
        "key": "experimental",
        "label": "Expérimental",
        "short": "L3",
        "color": "#6b21a8",
        "bg": "#faf5ff",
        "description": "Recherche avancée — résultats non validés pour usage industriel seul.",
    },
}

EXPERIMENTAL_WARNING = (
    "Les résultats présentés dans cette section sont expérimentaux. "
    "Ils servent à orienter les travaux de recherche et ne doivent pas être "
    "utilisés seuls pour une décision industrielle."
)

PILOT_WARNING = (
    "Protocole pilote contrôlé : les estimations servent au tri préliminaire "
    "et doivent être confirmées par essai laboratoire lorsque la propriété "
    "intervient dans une décision technique."
)

VALIDATION_JOURNEY = [
    "Données",
    "Analyse",
    "Modélisation",
    "Validation robuste",
    "Pilote terrain",
    "Industrialisation",
]

SOURCE_FILES_DOC = [
    ("Dataset NAV", "outputs/tables/04_NAV_feature_engineered.csv"),
    ("Statuts déploiement", "outputs/26_unified_evaluation/target_deployment_status.csv"),
    ("Métriques longues", "outputs/26_unified_evaluation/master_metrics_long.csv"),
    ("Modèle recommandé", "outputs/26_unified_evaluation/recommended_model_per_target.csv"),
    ("Validation robuste", "outputs/21_robust_generalization/robust_validation_results.csv"),
    ("Comparaison A/B", "outputs/23_model_A_vs_B_robust/model_A_vs_B_robust.csv"),
    ("Validation cibles", "outputs/14_target_validation/target_outlier_summary.csv"),
    ("Comparaison cascade", "outputs/25_approach_comparison/pivot_r2.csv"),
    ("Journal pilote", "outputs/prediction_logs/pilot_journal.csv"),
]
