"""Checkpoint-backed image prediction and an explicitly separate demo estimate."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from explainability.gradcam import create_gradcam
from inference.fusion import CLASS_NAMES, health_category, health_score
from models.network import build_model
from preprocessing.quality import assess_image_quality
from training.dataset import image_transforms


class CanopyPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str | None = None):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        if checkpoint.get("class_names") != list(CLASS_NAMES):
            raise ValueError("Checkpoint classes do not match the project class order")
        self.model = build_model(pretrained=False).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.transform = image_transforms(training=False)

    def predict(self, image_path: str | Path, gradcam_path: str | None = None) -> dict[str, Any]:
        image = Image.open(image_path).convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self.model(tensor), dim=1)[0].cpu().tolist()
        predicted_index = int(np.argmax(probabilities))
        score = health_score(probabilities)
        result = {
            "mode": "trained_model",
            "predicted_class": CLASS_NAMES[predicted_index],
            "probabilities": probabilities,
            "confidence": probabilities[predicted_index],
            "health_score": round(score, 1),
            "health_category": health_category(score),
            "quality": assess_image_quality(image_path),
            "gradcam_path": None,
        }
        if gradcam_path:
            result["gradcam_path"] = create_gradcam(
                self.model, tensor, predicted_index, gradcam_path, str(image_path)
            )
        return result


def demo_predict(image_path: str | Path) -> dict[str, Any]:
    """Return a deterministic heuristic estimate for interface demonstrations only."""
    quality = assess_image_quality(image_path)
    coverage = quality["vegetation_coverage"]
    brightness = quality["brightness"]
    if coverage >= 0.35 and 35 <= brightness <= 225:
        probabilities = [0.75, 0.25]
    elif coverage >= 0.12:
        probabilities = [0.4, 0.6]
    else:
        probabilities = [0.15, 0.85]
    index = int(np.argmax(probabilities))
    score = health_score(probabilities)
    return {
        "mode": "demo",
        "predicted_class": CLASS_NAMES[index],
        "probabilities": probabilities,
        "confidence": probabilities[index],
        "health_score": round(score, 1),
        "health_category": health_category(score),
        "quality": quality,
        "gradcam_path": None,
    }