"""Score de confiance simple (MVP)."""
from __future__ import annotations

from src.quality_check import QualityReport
from src.target_readiness import readiness_for_target


def _level(score: float) -> str:
  if score >= 0.7:
    return "élevée"
  if score >= 0.45:
    return "moyenne"
  return "faible"


def compute_confidence(
  target: str,
  quality: QualityReport,
  ood_score: float | None = None,
  prediction_ok: bool = True,
) -> tuple[float, str]:
  """
  Retourne (score 0-1, niveau élevée/moyenne/faible).
  """
  score = 1.0

  if not prediction_ok:
    return 0.0, "faible"

  comp = quality.completeness_pct / 100.0
  score *= 0.3 + 0.7 * comp

  if quality.out_of_range:
    score *= 0.75
  if quality.derived_issues:
    score *= 0.85
  if len(quality.missing_features) >= 4:
    score *= 0.8

  prior = readiness_for_target(target).get("confidence_prior", "moyenne")
  if prior == "faible":
    score *= 0.7
  elif prior == "moyenne":
    score *= 0.9

  # ood_score = trust_score OOD (1 = proche du NAV)
  if ood_score is not None:
    if ood_score < 0.45:
      score *= 0.75
    elif ood_score > 0.75:
      score = min(1.0, score * 1.05)

  score = max(0.0, min(1.0, score))
  return round(score, 3), _level(score)
