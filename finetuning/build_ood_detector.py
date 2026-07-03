"""Build the out-of-distribution (OOD) input detector.

Goal: reject inputs that are not chest X-rays — even *grayscale* ones (a colour
check alone would miss a grayscale cat). Method = classic Mahalanobis OOD:

  ImageNet ResNet18 features -> PCA(40) -> Gaussian (mean + Ledoit-Wolf covariance)
  fitted on real + synthetic chest X-rays. A test image whose Mahalanobis distance
  to that distribution exceeds a threshold (calibrated on held-out CXR) is rejected.

This script needs the Kaggle CXR data (data/real/, gitignored) to build a **small**
reference artifact `src/artifacts/cxr_ood.npz` (~100 KB) that IS committed, so the
detector works at runtime on any clone without the dataset.

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
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from PIL import Image

IMG = {".jpeg", ".jpg", ".png", ".bmp"}
_TF = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def list_images(folder: Path, n: int, skip: int = 0) -> list[Path]:
    if not folder.exists():
        return []
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMG and not p.name.startswith("._"))
    return files[skip:skip + n]


def extractor(device: str):
    net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    net.fc = torch.nn.Identity()
    net.eval().to(device)

    @torch.no_grad()
    def feat(paths: list[Path]) -> np.ndarray:
        out = []
        for p in paths:
            x = _TF(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            out.append(net(x)[0].cpu().numpy())
        return np.stack(out)

    return feat


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    real = ROOT / "data" / "real" / "chest_xray" / "chest_xray"
    if not real.exists():
        raise SystemExit(f"Dataset réel introuvable ({real}). Voir data/real/README.md (Kaggle).")

    feat = extractor(device)

    # Reference = real normal + real pneumonia + synthetic CXR.
    ref_paths = (list_images(real / "train" / "NORMAL", 120)
                 + list_images(real / "train" / "PNEUMONIA", 120)
                 + list_images(ROOT / "data" / "sample_images", 30))
    # Calibration = held-out real CXR (not in reference) to set the threshold.
    cal_paths = (list_images(real / "test" / "NORMAL", 60)
                 + list_images(real / "test" / "PNEUMONIA", 60))
    print(f"reference={len(ref_paths)}  calibration={len(cal_paths)}  device={device}")

    F = feat(ref_paths)
    pca = PCA(n_components=40).fit(F)
    Z = pca.transform(F)
    cov = LedoitWolf().fit(Z)            # location_ = Z.mean(0), precision_ = inv cov
    mu = cov.location_
    prec = cov.precision_

    def maha(paths: list[Path]) -> np.ndarray:
        z = pca.transform(feat(paths))
        v = z - mu
        return np.einsum("ij,jk,ik->i", v, prec, v)  # squared Mahalanobis

    cal_scores = maha(cal_paths)
    # Threshold just above the held-out CXR maximum (small headroom): accepts real
    # + synthetic X-rays with ~0 false rejects, while non-radiographs score higher.
    threshold = float(cal_scores.max() * 1.1)
    print(f"held-out CXR maha: median={np.median(cal_scores):.1f} "
          f"p99={np.percentile(cal_scores, 99):.1f} max={cal_scores.max():.1f}  -> threshold={threshold:.1f}")

    out = ROOT / "src" / "artifacts" / "cxr_ood.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out,
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        ref_mean=mu.astype(np.float32),
        precision=prec.astype(np.float32),
        threshold=np.float32(threshold),
    )
    size_kb = out.stat().st_size / 1024
    print(f"Artefact écrit : {out} ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
