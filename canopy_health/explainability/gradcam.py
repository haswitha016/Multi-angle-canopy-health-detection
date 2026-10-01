"""Grad-CAM overlays for the final MobileNetV3 feature block."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def create_gradcam(model, input_tensor, class_index: int, output_path: str, original_path: str) -> str:
    """Save a Grad-CAM overlay. It visualizes model cues, not biological proof."""
    activations = []
    gradients = []

    def capture_activation(_module, _inputs, output):
        activations.append(output)
        output.register_hook(lambda gradient: gradients.append(gradient))

    target_layer = model.features[-1]
    hook = target_layer.register_forward_hook(capture_activation)
    model.zero_grad(set_to_none=True)
    logits = model(input_tensor)
    logits[0, class_index].backward()
    hook.remove()

    feature_map = activations[0].detach()[0]
    gradient = gradients[0].detach()[0]
    weights = gradient.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * feature_map).sum(dim=0)).cpu().numpy()
    original = cv2.imread(original_path)
    if original is None:
        raise ValueError("Could not read the original image for Grad-CAM overlay")
    height, width = original.shape[:2]
    cam = cv2.resize(cam, (width, height))
    cam = cam / (float(cam.max()) + 1e-8)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.58, heatmap, 0.42, 0)
    cv2.imwrite(output_path, overlay)
    return output_path