"""Génère les notebooks 01-13 (appels au pipeline)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
NB.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("01_data_audit", "from src.step01_audit import run_step01\nrun_step01()"),
    ("02_feature_decision", "from src.step02_features import run_step02\nfrom src.utils import load_raw_excel, standardize_columns\nrun_step02(standardize_columns(load_raw_excel()))"),
    ("03_cleaning", "from src.step03_cleaning import run_step03\nfrom src.utils import load_raw_excel, standardize_columns\ndf,_=run_step03(standardize_columns(load_raw_excel()))"),
    ("04_feature_engineering", "from src.step04_engineering import run_step04\nimport pandas as pd\ndf=pd.read_csv('../outputs/tables/03_NAV_cleaned.csv')\nrun_step04(df)"),
    ("05_target_datasets", "from src.step05_targets import run_step05\nimport pandas as pd\ndf=pd.read_csv('../outputs/tables/04_NAV_feature_engineered.csv')\nrun_step05(df)"),
    ("06_target_categorization", "from src.step06_categorization import run_step06\nimport pandas as pd\ndf=pd.read_csv('../outputs/tables/04_NAV_feature_engineered.csv')\nrun_step06(df)"),
    ("07_baseline_ml", "print('Exécuter run_pipeline.py ou importer step07')"),
    ("08_deep_learning", "print('Étape 8 via run_pipeline.py')"),
    ("09_standard_evaluation", "print('Étape 9 via run_pipeline.py')"),
    ("10_explainability", "print('Étape 10 via run_pipeline.py')"),
    ("11_uncertainty", "print('Étape 11 via run_pipeline.py')"),
    ("12_model_selection", "print('Étape 12 via run_pipeline.py')"),
    ("13_streamlit_prototype", "print('streamlit run app/app.py')"),
]

def make_nb(name: str, code: str) -> dict:
    return {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {name}\n"]},
            {"cell_type": "code", "metadata": {}, "source": ["import sys\nfrom pathlib import Path\nsys.path.insert(0, str(Path('.').resolve().parent))\n", code], "outputs": [], "execution_count": None},
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }

for i, (name, code) in enumerate(STEPS, 1):
    path = NB / f"{i:02d}_{name}.ipynb"
    path.write_text(json.dumps(make_nb(name, code), indent=1), encoding="utf-8")
    print("wrote", path)
