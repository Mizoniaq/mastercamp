"""COULD extension — fine-tune a lightweight image classifier on real chest X-rays.

Transfer learning: a pretrained ResNet18 is fine-tuned to separate `normal` from
`suspected_opacity` (Kaggle chest-xray-pneumonia: NORMAL / PNEUMONIA). This adds a
real, MEASURED training phase and a more controllable confidence score than the
rule-based toy baseline. A low softmax confidence maps to `uncertain`, keeping the
project's three-class, cautious contract.

Measured on the held-out test split and compared to the baselines in the report.
Weights are saved under models/ (gitignored); only the metrics are committed.

Usage:
    python finetuning/train_cxr_classifier.py --epochs 3 --per-class-train 500
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image

from src.metrics import summarize_metrics
from src.guardrails import WARNING_TEXT

# Folder name -> project class.
CLASS_OF_FOLDER = {"NORMAL": "normal", "PNEUMONIA": "suspected_opacity"}
CLASSES = ["normal", "suspected_opacity"]
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}

_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),  # CXR are grayscale
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def collect(split_dir: Path, per_class: int | None, seed: int) -> list[tuple[Path, int]]:
    rng = random.Random(seed)
    items: list[tuple[Path, int]] = []
    for folder, cls in CLASS_OF_FOLDER.items():
        d = split_dir / folder
        if not d.exists():
            continue
        imgs = [p for p in d.iterdir()
                if p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith("._")]
        rng.shuffle(imgs)
        if per_class:
            imgs = imgs[:per_class]
        items += [(p, CLASSES.index(cls)) for p in imgs]
    rng.shuffle(items)
    return items


class CXRDataset(Dataset):
    def __init__(self, items: list[tuple[Path, int]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        return _TF(Image.open(path).convert("RGB")), label


def build_model(device: str) -> nn.Module:
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model.to(device)


def train(model, loader, device, epochs: int) -> None:
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {epoch + 1}/{epochs}  loss={total / max(len(loader), 1):.4f}")


@torch.no_grad()
def evaluate(model, items, device, uncertain_threshold: float) -> list[dict]:
    model.eval()
    rows = []
    for path, label_idx in items:
        x = _TF(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
        start = time.perf_counter()
        probs = torch.softmax(model(x), dim=1)[0]
        latency_ms = int((time.perf_counter() - start) * 1000)
        conf, idx = float(probs.max()), int(probs.argmax())
        pred = "uncertain" if conf < uncertain_threshold else CLASSES[idx]
        rows.append({
            "label": CLASSES[label_idx],
            "predicted_class": pred,
            "confidence": round(conf, 3),
            "json_valid": True,
            "warning": WARNING_TEXT,
            "latency_ms": latency_ms,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "real" / "chest_xray" / "chest_xray")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--per-class-train", type=int, default=500)
    parser.add_argument("--per-class-test", type=int, default=0, help="0 = all test images")
    parser.add_argument("--uncertain-threshold", type=float, default=0.60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-out", type=Path, default=ROOT / "models" / "cxr_classifier.pt")
    parser.add_argument("--metrics-out", type=Path, default=ROOT / "eval" / "results" / "classifier_metrics.json")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Dataset introuvable: {args.data_dir}. Voir data/real/README.md (Kaggle).")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    train_items = collect(args.data_dir / "train", args.per_class_train, seed=0)
    test_items = collect(args.data_dir / "test", args.per_class_test or None, seed=1)
    print(f"train={len(train_items)}  test={len(test_items)}")

    model = build_model(device)
    loader = DataLoader(CXRDataset(train_items), batch_size=args.batch_size, shuffle=True, num_workers=0)
    print("Entraînement…")
    train(model, loader, device, args.epochs)

    print("Évaluation sur le test set…")
    rows = evaluate(model, test_items, device, args.uncertain_threshold)
    metrics = summarize_metrics(rows)
    metrics["model"] = "resnet18-finetuned"
    metrics["n_train"] = len(train_items)
    metrics["uncertain_threshold"] = args.uncertain_threshold

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.model_out)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(f"Modèle: {args.model_out}  |  Métriques: {args.metrics_out}")


if __name__ == "__main__":
    main()
