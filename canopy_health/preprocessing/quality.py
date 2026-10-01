"""Simple, interpretable image-quality checks for canopy photographs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def vegetation_mask(rgb_image: np.ndarray) -> np.ndarray:
    """Create an optional vegetation mask from HSV green and excess-green cues."""
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    hsv_green = cv2.inRange(hsv, np.array([25, 30, 25]), np.array([100, 255, 255])) > 0
    red, green, blue = cv2.split(rgb_image.astype(np.float32))
    exg = 2 * green - red - blue
    exg_green = exg > 20
    return np.logical_or(hsv_green, exg_green)


def assess_image_quality(image_path: str | Path) -> dict[str, Any]:
    """Measure sharpness, brightness, resolution, and vegetation coverage."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 2]))
    coverage = float(np.mean(vegetation_mask(rgb)))

    sharpness_score = min(1.0, sharpness / 250.0)
    brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
    resolution_score = min(1.0, (width * height) / (1024.0 * 768.0))
    coverage_score = min(1.0, coverage / 0.2)
    quality_score = (
        0.35 * sharpness_score
        + 0.25 * brightness_score
        + 0.2 * resolution_score
        + 0.2 * coverage_score
    )
    flags = []
    if sharpness < 30:
        flags.append("blurry")
    if brightness < 35 or brightness > 225:
        flags.append("very dark or overexposed")
    if width < 256 or height < 256:
        flags.append("low resolution")
    if coverage < 0.03:
        flags.append("little detected vegetation")
    return {
        "sharpness": round(sharpness, 2),
        "brightness": round(brightness, 1),
        "resolution": f"{width} x {height}",
        "width": width,
        "height": height,
        "vegetation_coverage": round(coverage, 4),
        "quality_score": round(quality_score, 3),
        "flags": flags,
        "is_low_quality": bool(flags),
    }