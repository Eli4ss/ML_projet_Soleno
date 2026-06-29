"""Moteur d'inférence en cascade — propage les prédictions entre blocs.

Usage :
    predictor = CascadePredictor.load(approach="cascade")  # ou "pspp" / "direct"
    results = predictor.predict(input_row_dict)
    # → {"oit_min": 32.1, "traction": 22.4, ..., "ncls": 1800.0, "ucls": 4500.0}

La classe est instanciable depuis le pipeline Streamlit et depuis des scripts.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .config import (
    CASCADE_BLOC_ORDER,
    CASCADE_BLOCS,
    CASCADE_THERMAL_TARGETS,
    CASCADE_MECHANICAL_TARGETS,
    CASCADE_FINAL_TARGETS,
    MODEL_A_FEATURES,
    PHASE4_CASCADE,
)
from .modeling import get_pipeline_feature_columns


def _build_row_df(input_row: dict[str, Any], expected_cols: list[str]) -> pd.DataFrame:
    """Construit un DataFrame 1-ligne avec les colonnes attendues par le pipeline."""
    row = {col: input_row.get(col, np.nan) for col in expected_cols}
    return pd.DataFrame([row])


class CascadePredictor:
    """Prédicateur en cascade pour les 3 approches."""

    def __init__(self, models: dict[str, Any], approach: str):
        """
        Parameters
        ----------
        models : dict mapping target → sklearn Pipeline
        approach : "cascade" | "pspp" | "direct"
        """
        self.models = models
        self.approach = approach

    @classmethod
    def load(
        cls,
        approach: str = "cascade",
        models_dir: Path | None = None,
    ) -> "CascadePredictor":
        """Charge tous les modèles d'une approche depuis le dossier sauvegardé.

        Parameters
        ----------
        approach : "cascade", "pspp", ou "direct"
        models_dir : chemin vers le dossier des modèles (défaut : PHASE4_CASCADE/models)
        """
        if models_dir is None:
            models_dir = PHASE4_CASCADE / "models"

        prefix_map = {
            "cascade": "cascade_{bloc}_{target}",
            "pspp": "pspp_{target}",
            "direct": "direct_{target}",
        }
        if approach not in prefix_map:
            raise ValueError(f"approach doit être parmi {list(prefix_map.keys())}")

        all_targets = (
            CASCADE_THERMAL_TARGETS
            + CASCADE_MECHANICAL_TARGETS
            + CASCADE_FINAL_TARGETS
        )
        models: dict[str, Any] = {}

        for target in all_targets:
            if approach == "cascade":
                bloc = _target_to_bloc(target)
                pattern = f"cascade_{bloc}_{target}.joblib"
            elif approach == "pspp":
                pattern = f"pspp_{target}.joblib"
            else:
                pattern = f"direct_{target}.joblib"

            path = models_dir / pattern
            # Fallback : PSPP thermal targets use the direct model file
            if not path.exists() and approach == "pspp":
                fallback = models_dir / f"direct_{target}.joblib"
                if fallback.exists():
                    path = fallback
            if path.exists():
                try:
                    models[target] = joblib.load(path)
                except Exception as exc:
                    print(f"[CascadePredictor] Impossible de charger {path}: {exc}")

        if not models:
            raise FileNotFoundError(
                f"Aucun modèle '{approach}' trouvé dans {models_dir}.\n"
                "Exécutez run_phase4.py d'abord."
            )

        return cls(models=models, approach=approach)

    def predict(self, input_row: dict[str, Any]) -> dict[str, float]:
        """Prédit toutes les cibles disponibles pour un lot donné.

        Pour l'approche 'cascade', la prédiction est séquentielle :
        les outputs des blocs précédents sont injectés comme features
        dans les blocs suivants.

        Si une valeur intermédiaire (ex: oit_min mesuré) est déjà présente
        dans input_row et est non-NaN, elle est utilisée telle quelle comme
        feature pour les blocs suivants — aucune prédiction n'est effectuée
        pour cette cible.

        Parameters
        ----------
        input_row : dict avec les features matière (MODEL_A_FEATURES).
                    Peut aussi contenir des mesures intermédiaires optionnelles
                    (oit_min, traction, flexion…) qui seront prioritaires sur
                    les prédictions de leurs blocs respectifs.

        Returns
        -------
        dict target → valeur (mesurée si fournie, prédite sinon ; nan si absent)
        """
        row = dict(input_row)
        predictions: dict[str, float] = {}
        all_intermediate = (
            CASCADE_THERMAL_TARGETS
            + CASCADE_MECHANICAL_TARGETS
            + CASCADE_FINAL_TARGETS
        )

        if self.approach == "cascade":
            for bloc_name in CASCADE_BLOC_ORDER:
                for target in CASCADE_BLOCS[bloc_name]["targets"]:
                    # Priorité aux valeurs mesurées fournies dans input_row
                    measured = row.get(target)
                    if measured is not None and not np.isnan(float(measured)):
                        predictions[target] = float(measured)
                        # déjà dans row, pas besoin de réinjecter
                    else:
                        pred = self._predict_one(target, row)
                        predictions[target] = pred
                        if not np.isnan(pred):
                            row[target] = pred  # injection pour le bloc suivant
        else:
            # Direct et PSPP : prédiction indépendante de chaque cible
            for target, model in self.models.items():
                measured = row.get(target)
                if measured is not None and not np.isnan(float(measured)):
                    predictions[target] = float(measured)
                else:
                    predictions[target] = self._predict_one(target, row)

        return predictions

    def _predict_one(self, target: str, row: dict[str, Any]) -> float:
        """Prédit une seule cible à partir d'un dict de features."""
        model = self.models.get(target)
        if model is None:
            return np.nan
        try:
            expected = get_pipeline_feature_columns(model)
            X = _build_row_df(row, expected)
            return float(model.predict(X)[0])
        except Exception as exc:
            print(f"[CascadePredictor] Erreur prediction {target}: {exc}")
            return np.nan

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prédit toutes les cibles pour un DataFrame de lots.

        Returns
        -------
        DataFrame avec une colonne par cible prédite.
        """
        records = []
        for _, row in df.iterrows():
            preds = self.predict(row.to_dict())
            records.append(preds)
        return pd.DataFrame(records, index=df.index)

    def available_targets(self) -> list[str]:
        return list(self.models.keys())

    def __repr__(self) -> str:
        return (
            f"CascadePredictor(approach='{self.approach}', "
            f"targets={self.available_targets()})"
        )


def _target_to_bloc(target: str) -> str:
    """Retourne le nom du bloc pour un target donné."""
    for bloc_name, cfg in CASCADE_BLOCS.items():
        if target in cfg["targets"]:
            return bloc_name
    return "thermal"


def load_all_predictors(
    models_dir: Path | None = None,
) -> dict[str, CascadePredictor]:
    """Charge les 3 prédicateurs (direct, cascade, pspp) en une fois.

    Les approches non disponibles sont omises silencieusement.
    """
    predictors: dict[str, CascadePredictor] = {}
    for approach in ("direct", "cascade", "pspp"):
        try:
            predictors[approach] = CascadePredictor.load(approach, models_dir)
        except FileNotFoundError:
            pass
    return predictors
