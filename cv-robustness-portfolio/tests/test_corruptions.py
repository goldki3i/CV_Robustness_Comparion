from PIL import Image
import numpy as np
import pytest

from src.corruptions import corrupt_image
from src.datasets import stratified_indices


def test_corruptions_preserve_image_shape_and_rgb():
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), mode="RGB")
    for kind in ("noise", "blur", "brightness"):
        result = corrupt_image(image, kind, 2, seed=123)
        assert result.mode == "RGB"
        assert result.size == image.size


def test_noise_is_reproducible():
    image = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8), mode="RGB")
    first = np.asarray(corrupt_image(image, "noise", 2, seed=7))
    second = np.asarray(corrupt_image(image, "noise", 2, seed=7))
    assert np.array_equal(first, second)


def test_invalid_severity_raises():
    image = Image.new("RGB", (32, 32))
    with pytest.raises(ValueError):
        corrupt_image(image, "blur", 4)


def test_stratified_indices_balances_classes():
    targets = [0] * 20 + [1] * 20 + [2] * 20
    indices = stratified_indices(targets, 15, seed=11)
    counts = np.bincount(np.asarray(targets)[indices])
    assert len(indices) == 15
    assert counts.tolist() == [5, 5, 5]
