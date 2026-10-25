"""CIFAR-10 loading and deterministic, approximately stratified sampling."""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10


class CIFAR10PIL(Dataset):
    """CIFAR-10 wrapper returning PIL images and integer labels."""
    def __init__(self, root: str, train: bool, download: bool = True):
        self.dataset = CIFAR10(root=root, train=train, download=download, transform=None)
        self.targets = np.asarray(self.dataset.targets, dtype=np.int64)
        self.classes = self.dataset.classes

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, label = self.dataset[index]
        return image.convert("RGB"), int(label)


def stratified_indices(targets: Sequence[int], limit: int | None, seed: int) -> list[int]:
    """Choose a deterministic sample while keeping class counts balanced.

    When limit is None or >= dataset length, return all indices in original order.
    """
    n = len(targets)
    if limit is None or limit >= n:
        return list(range(n))
    if limit <= 0:
        raise ValueError("limit must be positive")

    labels = np.asarray(targets)
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    base, remainder = divmod(limit, len(classes))
    chosen: list[int] = []
    for class_position, cls in enumerate(classes):
        candidates = np.flatnonzero(labels == cls)
        take = base + (1 if class_position < remainder else 0)
        if take > len(candidates):
            raise ValueError("Requested sample exceeds available examples in a class")
        chosen.extend(rng.choice(candidates, size=take, replace=False).tolist())
    rng.shuffle(chosen)
    return chosen
