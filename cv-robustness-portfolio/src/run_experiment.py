"""Run paired corruption robustness experiments on CIFAR-10."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.corruptions import CORRUPTIONS, corrupt_image
from src.datasets import CIFAR10PIL, stratified_indices
from src.evaluation import finalize_metrics, metric_row, save_summaries, train_linear_probe
from src.models import FrozenFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["dinov2", "clip", "resnet50"], default=["dinov2", "clip", "resnet50"])
    parser.add_argument("--train-limit", type=int, default=10000, help="Number of CIFAR-10 train images; use 50000 for all")
    parser.add_argument("--test-limit", type=int, default=1000, help="Number of CIFAR-10 test images; use 10000 for all")
    parser.add_argument("--severities", nargs="+", type=int, choices=[1, 2, 3], default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--cache-dir", default="hf_cache")
    parser.add_argument("--output-dir", default="results/run_001")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was selected, but CUDA is not available")
    return torch.device(choice)


def cache_key(model: str, split: str, condition: str, severity: int, indices: list[int], seed: int) -> str:
    payload = json.dumps({"model": model, "split": split, "condition": condition, "severity": severity,
                          "indices": indices, "seed": seed}, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:20]


def load_or_extract(extractor, images, labels, model: str, split: str, condition: str, severity: int,
                    indices: list[int], seed: int, batch_size: int, feature_dir: Path):
    key = cache_key(model, split, condition, severity, indices, seed)
    path = feature_dir / f"{key}.npz"
    if path.exists():
        saved = np.load(path)
        return saved["features"], saved["labels"]
    batch = extractor.extract(images, labels, batch_size=batch_size)
    np.savez_compressed(path, features=batch.features, labels=batch.labels)
    return batch.features, batch.labels


def main():
    args = parse_args()
    if args.train_limit <= 0 or args.test_limit <= 0 or args.batch_size <= 0:
        raise ValueError("train-limit, test-limit, and batch-size must be positive")
    seed_everything(args.seed)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    feature_dir = output_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    train_ds = CIFAR10PIL(args.data_dir, train=True)
    test_ds = CIFAR10PIL(args.data_dir, train=False)
    train_indices = stratified_indices(train_ds.targets, args.train_limit, args.seed)
    test_indices = stratified_indices(test_ds.targets, args.test_limit, args.seed + 1)

    train_images, train_labels = zip(*(train_ds[i] for i in train_indices))
    test_images, test_labels = zip(*(test_ds[i] for i in test_indices))
    train_images, train_labels = list(train_images), list(train_labels)
    test_images, test_labels = list(test_images), list(test_labels)

    config = vars(args).copy()
    config.update({
        "device_resolved": str(device),
        "train_samples_used": len(train_images),
        "test_samples_used": len(test_images),
        "classes": train_ds.classes,
        "feature_extractor_frozen": True,
        "classifier": "StandardScaler + LogisticRegression(solver=lbfgs, C=1.0, max_iter=2000)",
        "corruptions": list(CORRUPTIONS),
        "note": "Train probe on clean train features; evaluate on paired clean/corrupted test images.",
    })
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows: list[dict] = []
    for model_name in args.models:
        print(f"\n=== Model: {model_name} ===")
        extractor = FrozenFeatureExtractor(model_name, device, cache_dir=args.cache_dir)

        x_train, y_train = load_or_extract(extractor, train_images, train_labels, model_name, "train", "clean", 0,
                                           train_indices, args.seed, args.batch_size, feature_dir)
        classifier = train_linear_probe(x_train, y_train)

        # Clean test condition
        x_clean, y_clean = load_or_extract(extractor, test_images, test_labels, model_name, "test", "clean", 0,
                                           test_indices, args.seed, args.batch_size, feature_dir)
        clean_pred = classifier.predict(x_clean)
        rows.append(metric_row(model_name, "clean", 0, y_clean, clean_pred))
        print(f"clean accuracy: {rows[-1]['accuracy']:.4f}")

        for corruption in CORRUPTIONS:
            for severity in args.severities:
                corrupted_images = [corrupt_image(img, corruption, severity, seed=args.seed + int(idx))
                                    for img, idx in zip(test_images, test_indices)]
                x_corrupt, y_corrupt = load_or_extract(
                    extractor, corrupted_images, test_labels, model_name, "test", corruption, severity,
                    test_indices, args.seed, args.batch_size, feature_dir
                )
                predictions = classifier.predict(x_corrupt)
                rows.append(metric_row(model_name, corruption, severity, y_corrupt, predictions))
                print(f"{corruption} severity={severity}: accuracy={rows[-1]['accuracy']:.4f}")

        del extractor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    metrics = finalize_metrics(rows)
    save_summaries(metrics, output_dir)
    print(f"\nSaved metrics and plots to: {output_dir.resolve()}")
    print(metrics[["model", "condition", "severity", "accuracy_pct", "accuracy_drop_pp"]].to_string(index=False))


if __name__ == "__main__":
    main()
