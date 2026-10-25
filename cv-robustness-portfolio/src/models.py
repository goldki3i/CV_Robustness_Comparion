"""Frozen pretrained feature extractors for DINOv2, CLIP, and ResNet-50."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


@dataclass
class FeatureBatch:
    features: np.ndarray
    labels: np.ndarray


class ImageListDataset(Dataset):
    def __init__(self, images: list[Image.Image], labels: list[int], processor_kind: str, processor=None):
        self.images = images
        self.labels = labels
        self.processor_kind = processor_kind
        self.processor = processor

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index].convert("RGB")
        if self.processor_kind == "hf":
            pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        else:
            pixels = self.processor(image)
        return pixels, int(self.labels[index])


class FrozenFeatureExtractor:
    """Wrap a pretrained model and return fixed image embeddings."""
    def __init__(self, name: str, device: torch.device, cache_dir: str | None = None):
        self.name = name.lower()
        self.device = device
        self.processor_kind = "hf"
        self.processor = None

        if self.name == "dinov2":
            from transformers import AutoImageProcessor, AutoModel
            model_id = "facebook/dinov2-base"
            self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
            self.feature_method = "dinov2"
        elif self.name == "clip":
            from transformers import CLIPModel, CLIPProcessor
            model_id = "openai/clip-vit-base-patch32"
            self.processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir)
            self.model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir)
            self.feature_method = "clip"
        elif self.name == "resnet50":
            weights = ResNet50_Weights.DEFAULT
            self.processor = weights.transforms()
            self.processor_kind = "torchvision"
            self.model = resnet50(weights=weights)
            # Remove the ImageNet classifier and use pooled penultimate features.
            self.model.fc = torch.nn.Identity()
            self.feature_method = "resnet50"
        else:
            raise ValueError(f"Unsupported model {name!r}; choose dinov2, clip, resnet50")

        self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.inference_mode()
    def extract(self, images: list[Image.Image], labels: list[int], batch_size: int = 32) -> FeatureBatch:
        dataset = ImageListDataset(images, labels, self.processor_kind, self.processor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        feature_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []

        for pixels, batch_labels in loader:
            pixels = pixels.to(self.device)
            if self.feature_method == "dinov2":
                output = self.model(pixel_values=pixels)
                feats = output.last_hidden_state[:, 0, :]  # CLS token
            elif self.feature_method == "clip":
                feats = self.model.get_image_features(pixel_values=pixels)
            else:
                feats = self.model(pixels)  # pooled 2048-D ResNet features
            feats = torch.nn.functional.normalize(feats.float(), p=2, dim=1)
            feature_chunks.append(feats.cpu().numpy().astype(np.float32))
            label_chunks.append(batch_labels.numpy().astype(np.int64))

        if not feature_chunks:
            raise ValueError("Cannot extract features from an empty image list")
        return FeatureBatch(np.concatenate(feature_chunks), np.concatenate(label_chunks))
