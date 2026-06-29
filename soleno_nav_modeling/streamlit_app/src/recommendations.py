"""Recommandations automatiques après prédiction."""
from __future__ import annotations

from src.quality_check import QualityReport


def recommend_for_target(
  target: str,
  prediction: float | str | None,
  confidence_level: str,
  quality: QualityReport,
  good_threshold: float | None = None,
  bad_threshold: float | None = None,
) -> str:
  """Une recommandation courte par cible."""
  if prediction is None or (isinstance(prediction, float) and str(prediction) == "nan"):
    return "Données insuffisantes"

  if quality.quality_level == "faible" or quality.completeness_pct < 40:
    return "Compléter les analyses avant décision"

  if confidence_level == "faible":
    if target in ("ncls", "ucls"):
      return "Screening seulement — test labo recommandé"
    return "Test laboratoire recommandé"

  pred_f = None
  try:
    pred_f = float(prediction)
  except (TypeError, ValueError):
    return "Prédiction prudente"

  if good_threshold is not None and pred_f >= good_threshold:
    if confidence_level == "élevée":
      return "Lot intéressant"
    return "Lot prometteur — confirmer au labo"

  if bad_threshold is not None and pred_f <= bad_threshold:
    if confidence_level == "élevée":
      return "Lot à éviter ou reformuler"
    return "Prédiction prudente — vérifier au labo"

  if confidence_level == "élevée":
    return "Prédiction exploitable"
  if confidence_level == "moyenne":
    return "Prédiction prudente"
  return "Test laboratoire recommandé"


def build_global_recommendation(
  rows: list[dict],
) -> str:
  """Phrase synthèse multi-cibles."""
  if not rows:
    return "Aucune prédiction disponible."

  good_hi = [r for r in rows if "intéressant" in r.get("recommendation", "").lower()]
  weak = [r for r in rows if r.get("confidence_level") == "faible"]
  lab = [r for r in rows if "labo" in r.get("recommendation", "").lower() or "screening" in r.get("recommendation", "").lower()]

  parts = []
  if good_hi:
    tgts = ", ".join(r["target"] for r in good_hi[:3])
    parts.append(f"Ce lot semble intéressant pour {tgts}")
  if weak:
    tgts = ", ".join(r["target"] for r in weak[:3])
    parts.append(f"la prédiction {tgts} est incertaine")
  if lab:
    parts.append("Un test laboratoire est recommandé avant validation")
  elif not parts:
    parts.append("Interpréter les prédictions avec prudence selon la qualité des données")

  return ". ".join(parts) + "." if parts else "Prédiction terminée — vérifier la qualité des données."
