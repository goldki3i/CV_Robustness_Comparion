"""Deterministic image corruption functions for robustness evaluation."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def corrupt_image(image: Image.Image, corruption: str, severity: int, seed: int = 0) -> Image.Image:
    """Return a corrupted RGB PIL image.

    Args:
        image: Input image.
        corruption: One of ``noise``, ``blur``, or ``brightness``.
        severity: Integer from 1 to 3.
        seed: Per-image seed for reproducible Gaussian noise.
    """
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1, 2, or 3")
    image = image.convert("RGB")

    if corruption == "noise":
        sigma = {1: 8.0, 2: 16.0, 3: 25.0}[severity]
        arr = np.asarray(image).astype(np.float32)
        rng = np.random.default_rng(seed)
        noisy = np.clip(arr + rng.normal(0.0, sigma, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy, mode="RGB")

    if corruption == "blur":
        radius = {1: 0.6, 2: 1.2, 3: 1.8}[severity]
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

    if corruption == "brightness":
        # Values move progressively away from the original brightness (1.0).
        factor = {1: 0.75, 2: 0.50, 3: 0.30}[severity]
        return ImageEnhance.Brightness(image).enhance(factor)

    raise ValueError(f"Unknown corruption: {corruption!r}")


CORRUPTIONS = ("noise", "blur", "brightness")
