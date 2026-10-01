"""Fusion rules for combining predictions from views of one canopy."""

from __future__ import annotations

from collections import Counter
from typing import Any


CLASS_NAMES = ("Healthy", "Mild Stress", "Severe Stress")
HEALTH_VALUES = (100.0, 50.0, 0.0)


def health_score(probabilities: list[float]) -> float:
    """Return expected canopy health on a 0-100 scale."""
    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(f"Expected {len(CLASS_NAMES)} class probabilities")
    return sum(probability * value for probability, value in zip(probabilities, HEALTH_VALUES))


def health_category(score: float) -> str:
    """Map a health score to the nearest named health band."""
    if score >= 75:
        return "Healthy"
    if score >= 25:
        return "Mild Stress"
    return "Severe Stress"


def _normalize(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise ValueError("Cannot normalize an empty probability distribution")
    return [value / total for value in values]


def fuse_predictions(
    predictions: list[dict[str, Any]],
    quality_weighted: bool = False,
) -> dict[str, dict[str, Any]]:
    """Compare majority vote, mean probabilities, and confidence-weighted fusion."""
    if not predictions:
        raise ValueError("At least one image prediction is required")

    for prediction in predictions:
        probabilities = prediction["probabilities"]
        if len(probabilities) != len(CLASS_NAMES):
            raise ValueError("Every prediction must contain three class probabilities")

    indices = {name: index for index, name in enumerate(CLASS_NAMES)}
    votes = Counter(indices[prediction["predicted_class"]] for prediction in predictions)
    majority = [votes[index] / len(predictions) for index in range(len(CLASS_NAMES))]
    average = [
        sum(prediction["probabilities"][index] for prediction in predictions) / len(predictions)
        for index in range(len(CLASS_NAMES))
    ]

    weighted_totals = [0.0] * len(CLASS_NAMES)
    weight_sum = 0.0
    for prediction in predictions:
        probabilities = prediction["probabilities"]
        confidence = max(probabilities)
        quality = prediction.get("quality", {}).get("quality_score", 1.0) if quality_weighted else 1.0
        weight = confidence * max(0.0, min(1.0, float(quality)))
        for index, probability in enumerate(probabilities):
            weighted_totals[index] += probability * weight
        weight_sum += weight
    if weight_sum == 0:
        weighted = average
    else:
        weighted = [value / weight_sum for value in weighted_totals]

    fused: dict[str, dict[str, Any]] = {}
    for name, probabilities in (
        ("majority_vote", majority),
        ("average_probability", average),
        ("confidence_weighted", weighted),
    ):
        probabilities = _normalize(probabilities)
        predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        score = health_score(probabilities)
        fused[name] = {
            "probabilities": dict(zip(CLASS_NAMES, probabilities)),
            "predicted_class": CLASS_NAMES[predicted_index],
            "confidence": probabilities[predicted_index],
            "health_score": round(score, 1),
            "health_category": health_category(score),
        }
    return fused