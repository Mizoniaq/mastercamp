from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Contrast (grayscale std) thresholds used to flag image quality.
# Derived from the synthetic toy set: uncertain/limited images have visibly
# lower global contrast than well-exposed normal or opacity images.
POOR_CONTRAST_STD = 8.0
LIMITED_CONTRAST_STD = 13.0

# A frontal chest X-ray is essentially grayscale (R≈G≈B). A colour photo (a cat,
# a selfie, a screenshot) has a much higher mean channel spread. Above this
# saturation, the input is rejected as "not a radiograph".
MAX_CXR_SATURATION = 18.0


def is_probably_cxr(path: str | Path) -> tuple[bool, str]:
    """Cheap, transparent gate: does this image plausibly look like a CXR?

    Primary signal = grayscale-ness (colour images are rejected). This is a
    heuristic, not an out-of-distribution detector: it reliably rejects colour
    photos but cannot catch a grayscale non-medical image. A dedicated
    "is-this-a-chest-X-ray?" classifier would be the robust production solution.
    """
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return True, ""  # already single-channel -> treat as grayscale/plausible
    spread = arr.max(axis=2) - arr.min(axis=2)  # 0 for a pure-gray pixel
    mean_saturation = float(spread.mean())
    if mean_saturation > MAX_CXR_SATURATION:
        return False, (
            f"image en couleur détectée (saturation {mean_saturation:.0f}) — "
            "une radiographie thoracique est en niveaux de gris"
        )
    return True, ""


def load_image(path: str | Path, size: tuple[int, int] = (512, 512)) -> Image.Image:
    """Load an image safely for the educational prototype.

    This function intentionally keeps preprocessing minimal. For real CXR work,
    DICOM metadata, windowing, projection and acquisition details should be handled
    explicitly and documented.
    """
    path = Path(path)
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        raise ValueError(f"Unsupported image format: {path.suffix}")
    img = Image.open(path).convert("RGB")
    return img.resize(size)


def extract_features(path: str | Path) -> dict[str, float]:
    """Extract a few simple, interpretable grayscale features from an image.

    These are deliberately transparent statistics (not a learned model) so the
    toy classifier and its uncertainty rule stay fully explainable:

    - ``contrast``: standard deviation of pixel intensities. Low contrast is a
      proxy for a poorly exposed / limited-quality acquisition.
    - ``bright_peak``: 99th percentile intensity. A high peak signals a focal
      bright region compatible with the toy "opacity" marker.
    - ``bright_fraction``: fraction of pixels far above the mean, i.e. the size
      of any focal bright area.
    - ``mean``: mean intensity, kept for reference/debugging.
    """
    arr = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std())
    threshold = mean + 3.0 * std
    bright_fraction = float((arr > threshold).mean())
    return {
        "contrast": std,
        "bright_peak": float(np.percentile(arr, 99)),
        "bright_fraction": bright_fraction,
        "mean": mean,
    }


def quality_from_features(
    features: dict[str, float],
    limited_std: float = LIMITED_CONTRAST_STD,
    poor_std: float = POOR_CONTRAST_STD,
) -> str:
    """Map contrast to an interpretable image-quality label.

    Thresholds are overridable so the sensitivity analysis can sweep them.
    """
    contrast = features["contrast"]
    if contrast < poor_std:
        return "poor"
    if contrast < limited_std:
        return "limited"
    return "good"


def basic_quality_flag(path: str | Path) -> str:
    """Image-based quality flag (contrast proxy).

    Replaces the earlier filename-based heuristic so the flag reflects the
    actual pixels. Real CXR quality control would inspect exposure, inspiration,
    rotation and projection explicitly.
    """
    return quality_from_features(extract_features(path))
