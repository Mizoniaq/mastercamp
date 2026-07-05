"""Input gate: decide whether an uploaded image is plausibly a chest X-ray.

Two layers, cheapest first:
  1. **Colour check** (`preprocessing.is_probably_cxr`) — torch-free, instant;
     rejects colour photos.
  2. **Discriminative detector** (this module) — a logistic regression on frozen
     ResNet18(ImageNet) features, trained to separate chest X-rays from a diverse
     negative set (natural photos incl. cats/dogs, CIFAR, clothing, digits, noise).
     Artifact: `src/artifacts/cxr_ood.npz` (~7 KB, built by
     `finetuning/build_ood_detector.py`). Rejects anything that is not a chest
     X-ray — including a *grayscale* cat.

The deep layer degrades gracefully: if torch/torchvision or the artifact are
unavailable, the gate falls back to the colour check only.
"""

from __future__ import annotations

import math
from pathlib import Path

from .preprocessing import is_probably_cxr

ARTIFACT = Path(__file__).resolve().parent / "artifacts" / "cxr_ood.npz"

_MODEL: dict | None = None  # cached extractor + LR params; {} means "unavailable"


def _load_model() -> dict:
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    try:
        import numpy as np
        import torch
        from torchvision import models, transforms
        from torchvision.models import ResNet18_Weights

        if not ARTIFACT.exists():
            _MODEL = {}
            return _MODEL

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
        _MODEL = {
            "np": np, "torch": torch, "net": net, "tf": tf, "device": device,
            "feat_mean": a["feat_mean"], "feat_scale": a["feat_scale"],
            "lr_coef": a["lr_coef"], "lr_intercept": float(a["lr_intercept"]),
            "threshold": float(a["threshold"]),
        }
    except Exception:  # torch/torchvision missing, bad artifact, etc.
        _MODEL = {}
    return _MODEL


def cxr_probability(path) -> tuple[float, float] | None:
    """Return (P(chest X-ray), threshold) or None if the detector is unavailable."""
    m = _load_model()
    if not m:
        return None
    from PIL import Image

    torch, np = m["torch"], m["np"]
    with torch.no_grad():
        x = m["tf"](Image.open(path).convert("RGB")).unsqueeze(0).to(m["device"])
        feat = m["net"](x)[0].cpu().numpy()
    z = (feat - m["feat_mean"]) / m["feat_scale"]
    logit = float(z @ m["lr_coef"] + m["lr_intercept"])
    prob = 1.0 / (1.0 + math.exp(-logit))
    return prob, m["threshold"]


def detector_available() -> bool:
    """Cheap check (no model download): are torch/torchvision + the artifact present?

    Lets the UI warn when the deep detector is unavailable and the gate is running
    on the colour check only (less precise: grayscale non-radiographs slip through).
    """
    import importlib.util

    if not ARTIFACT.exists():
        return False
    return all(importlib.util.find_spec(m) is not None for m in ("torch", "torchvision"))


def gate_input(path) -> tuple[bool, str]:
    """(accepted, reason). Rejects colour photos and any non-chest-X-ray image."""
    ok, reason = is_probably_cxr(path)          # colour check (always available)
    if not ok:
        return False, reason
    scored = cxr_probability(path)               # discriminative check (optional)
    if scored is not None:
        prob, threshold = scored
        if prob < threshold:
            return False, (
                f"l'image ne ressemble pas à une radiographie thoracique "
                f"(probabilité radio {prob:.0%} < seuil {threshold:.0%})"
            )
    return True, ""
