"""Build the input-gate detector: "is this image a chest X-ray?".

A colour check alone (and a one-class distance model) let non-radiographs through —
especially grayscale photos. This builds a **discriminative** detector that
generalises to *anything that is not a chest X-ray*:

    frozen ResNet18(ImageNet) features -> standardise -> LogisticRegression

trained on chest X-rays (positives) vs a **diverse** negative set:
    - real cat/dog photos (HF `microsoft/cats_vs_dogs`, streamed)
    - natural images (HF `uoft-cs/cifar10`)
    - clothing (FashionMNIST) and digits (MNIST) (torchvision)
    - random noise / gradient patterns

The negatives span low- and high-resolution, simple and textured content, so the
classifier learns the chest-X-ray region rather than a single cue (e.g. sharpness).

Needs the Kaggle CXR data (data/real/, gitignored) plus build-time libs:
    pip install datasets scikit-learn
Produces a **small** artifact `src/artifacts/cxr_ood.npz` (~7 KB, committed) so the
detector works at runtime on any clone (torch + numpy only, no datasets).

Usage:
    python finetuning/build_ood_detector.py
"""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import numpy as np
import torch
from torchvision import models, transforms, datasets
from torchvision.models import ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from PIL import Image

IMG = {".jpeg", ".jpg", ".png", ".bmp"}
_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_cxr(folder: Path, n: int, skip: int = 0) -> list[Image.Image]:
    if not folder.exists():
        return []
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMG and not p.name.startswith("._"))
    return [Image.open(p) for p in files[skip:skip + n]]


def synthetic(n: int, rng) -> list[Image.Image]:
    return [Image.fromarray(rng.integers(0, 255, (rng.integers(80, 256), rng.integers(80, 256))).astype("uint8"))
            for _ in range(n)]


def stream_pets(n: int) -> list[Image.Image]:
    from datasets import load_dataset
    ds = load_dataset("microsoft/cats_vs_dogs", split="train", streaming=True).shuffle(seed=1, buffer_size=1000)
    out = []
    for ex in ds:
        try:
            out.append(ex["image"].convert("RGB"))
        except Exception:
            continue
        if len(out) >= n:
            break
    return out


def make_extractor(device: str):
    net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(device)

    @torch.no_grad()
    def feats(images: list[Image.Image]) -> np.ndarray:
        return np.stack([net(_TF(im.convert("RGB")).unsqueeze(0).to(device))[0].cpu().numpy() for im in images])

    return feats


def main() -> None:
    from datasets import load_dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = ROOT / "data" / "real" / "chest_xray" / "chest_xray"
    if not real.exists():
        raise SystemExit(f"Dataset CXR introuvable ({real}). Voir data/real/README.md (Kaggle).")
    feats = make_extractor(device)
    rng = np.random.default_rng(0)

    # Positives: chest X-rays (real + synthetic).
    pos = (load_cxr(real / "train" / "NORMAL", 120) + load_cxr(real / "train" / "PNEUMONIA", 120)
           + load_cxr(ROOT / "data" / "sample_images", 30))
    # Negatives: diverse non-radiographs.
    print("Chargement des négatifs (pets HF stream, CIFAR, FashionMNIST, MNIST, synth)…")
    pets = stream_pets(280)
    cifar = list(load_dataset("uoft-cs/cifar10", split="test[:300]")["img"])
    fm = datasets.FashionMNIST("/tmp/fm", train=False, download=True)
    mn = datasets.MNIST("/tmp/mn", train=False, download=True)
    neg = (pets + cifar + [fm[i][0] for i in range(120)] + [mn[i][0] for i in range(80)] + synthetic(60, rng))
    print(f"positifs={len(pos)}  négatifs={len(neg)}  device={device}")

    X = np.vstack([feats(pos), feats(neg)])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    scaler = StandardScaler().fit(X)
    lr = LogisticRegression(max_iter=5000, C=0.3, class_weight="balanced").fit(scaler.transform(X), y)

    out = ROOT / "src" / "artifacts" / "cxr_ood.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        feat_mean=scaler.mean_.astype(np.float32),
        feat_scale=scaler.scale_.astype(np.float32),
        lr_coef=lr.coef_[0].astype(np.float32),
        lr_intercept=np.float32(lr.intercept_[0]),
        threshold=np.float32(0.5),
    )
    print(f"Artefact écrit : {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
