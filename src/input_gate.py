"""Input gate: decide whether an uploaded image is plausibly a chest X-ray.

Two layers, cheapest first:
  1. **Colour check** (`preprocessing.is_probably_cxr`) — torch-free, instant;
     rejects colour photos (a cat, a selfie).
  2. **Out-of-distribution detector** (this module) — a ResNet18(ImageNet) feature
     + PCA + Mahalanobis model fitted on real & synthetic chest X-rays
     (`src/artifacts/cxr_ood.npz`, built by `finetuning/build_ood_detector.py`).
     Catches non-radiographs that the colour check misses, e.g. a *grayscale* cat.

The deep layer degrades gracefully: if torch/torchvision or the artifact are
unavailable, the gate falls back to the colour check only.
"""

from __future__ import annotations

from pathlib import Path

from .preprocessing import is_probably_cxr

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "cxr_ood.npz"

_OOD: dict | None = None  # cached model + params; {} means "unavailable"


def _load_ood() -> dict:
    global _OOD
    if _OOD is not None:
        return _OOD
    try:
        import numpy as np
        import torch
        from torchvision import models, transforms
        from torchvision.models import ResNet18_Weights

        if not ARTIFACT.exists():
            _OOD = {}
            return _OOD

        a = np.load(ARTIFACT)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        net = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
        net.eval().to(device)
        tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        _OOD = {
            "np": np, "torch": torch, "net": net, "tf": tf, "device": device,
            "pca_mean": a["pca_mean"], "pca_components": a["pca_components"],
            "ref_mean": a["ref_mean"], "precision": a["precision"],
            "threshold": float(a["threshold"]),
        }
    except Exception:  # torch/torchvision missing, bad artifact, etc.
        _OOD = {}
    return _OOD


def cxr_ood_score(path) -> tuple[float, float] | None:
    """Return (mahalanobis_score, threshold) or None if the detector is unavailable."""
    ood = _load_ood()
    if not ood:
        return None
    from PIL import Image

    torch, np = ood["torch"], ood["np"]
    with torch.no_grad():
        x = ood["tf"](Image.open(path).convert("RGB")).unsqueeze(0).to(ood["device"])
        feat = ood["net"](x)[0].cpu().numpy()
    z = (feat - ood["pca_mean"]) @ ood["pca_components"].T
    v = z - ood["ref_mean"]
    return float(v @ ood["precision"] @ v), ood["threshold"]


def gate_input(path) -> tuple[bool, str]:
    """(accepted, reason). Rejects colour photos and out-of-distribution images."""
    ok, reason = is_probably_cxr(path)          # colour check (always available)
    if not ok:
        return False, reason
    scored = cxr_ood_score(path)                 # deep OOD check (optional)
    if scored is not None:
        score, threshold = scored
        if score > threshold:
            return False, (
                f"l'image ne ressemble pas à une radiographie thoracique "
                f"(score de distribution {score:.0f} > seuil {threshold:.0f})"
            )
    return True, ""
