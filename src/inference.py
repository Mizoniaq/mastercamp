from __future__ import annotations

from pathlib import Path
import os
import time
import re
import json
from typing import Any

from .preprocessing import extract_features, quality_from_features

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

MODEL_ID = "google/medgemma-4b-it"
WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Decision thresholds for the transparent toy classifier.
# A focal bright region covering more than this fraction of the image is the
# toy proxy for a "suspected opacity".
OPACITY_BRIGHT_FRACTION = 0.008
# Improved mode abstains ("uncertain") below this confidence, matching the rule
# stated in prompts/improved_prompt.txt.
CONFIDENCE_ABSTAIN_THRESHOLD = 0.60

BASE_LIMITATIONS = ["synthetic toy image", "no clinical context", "not a validated medical model"]

_PIPE = None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _raw_decision(features: dict[str, float]) -> tuple[str, float, list[str]]:
    """Return the naive (baseline) class, confidence and visual evidence.

    This is a deliberately simple, fully interpretable rule based on the size of
    any focal bright region. It has no notion of image quality and never
    abstains — that limitation is exactly what the improved mode fixes.
    """
    bright_fraction = features["bright_fraction"]
    bright_peak = features["bright_peak"]

    if bright_fraction >= OPACITY_BRIGHT_FRACTION:
        confidence = _clip(0.60 + bright_fraction * 10.0, 0.60, 0.90)
        evidence = [
            f"focal bright region detected (99th-percentile intensity {bright_peak:.0f}, "
            f"bright-area fraction {bright_fraction:.1%})"
        ]
        return "suspected_opacity", round(confidence, 3), evidence

    confidence = _clip(0.85 - bright_fraction * 20.0, 0.60, 0.85)
    evidence = [
        f"no focal bright region (bright-area fraction {bright_fraction:.1%}); "
        f"global contrast within expected range"
    ]
    return "normal", round(confidence, 3), evidence


def toy_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Transparent image-feature classifier used to validate the repo pipeline.

    It reads simple grayscale statistics from the image (not the filename) and
    applies one of two rules:

    - ``baseline``: focal-bright-region rule only, never abstains.
    - ``improved``: adds an explicit uncertainty rule — abstain to "uncertain"
      when image quality is limited/poor or when confidence is below
      ``CONFIDENCE_ABSTAIN_THRESHOLD``.

    This is a reproducible software validation, not medical inference. It runs
    with no GPU and no network so it works in CI and in the smoke test.
    """
    start = time.perf_counter()
    features = extract_features(image_path)
    quality = quality_from_features(features)

    predicted_class, confidence, evidence = _raw_decision(features)
    limitations = list(BASE_LIMITATIONS)

    if mode == "improved":
        limitations.append("confidence is a heuristic proxy, not calibrated")
        if quality in {"limited", "poor"}:
            predicted_class = "uncertain"
            confidence = round(min(confidence, 0.50), 3)
            evidence = [
                f"low global contrast ({features['contrast']:.1f}) indicates "
                f"{quality} image quality; abstaining is safer than forcing a class"
            ]
        elif confidence < CONFIDENCE_ABSTAIN_THRESHOLD:
            predicted_class = "uncertain"
            evidence.append(
                f"decision confidence below the {CONFIDENCE_ABSTAIN_THRESHOLD:.2f} threshold"
            )

    justification = _justify(predicted_class, mode)

    latency_ms = int((time.perf_counter() - start) * 1000)
    return {
        "image_quality": quality,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "visual_evidence": evidence,
        "justification": justification,
        "limitations": limitations,
        "warning": WARNING,
        "model_name": f"toy-imgfeat-{mode}",
        "prompt_version": f"{mode}_v1",
        "latency_ms": latency_ms,
    }


def _justify(predicted_class: str, mode: str) -> str:
    """Return a cautious, evidence-based justification for the toy output."""
    if predicted_class == "suspected_opacity":
        return (
            "A localized brighter region compatible with the toy opacity marker is present. "
            "This is a pipeline validation result on a synthetic image, not a medical interpretation."
        )
    if predicted_class == "normal":
        return (
            "No focal bright region matching the toy opacity marker was found and contrast is adequate. "
            "This conclusion is limited to the synthetic validation setting."
        )
    return (
        "Image quality or decision confidence is insufficient to conclude, so the safe output is "
        "uncertainty rather than a forced class."
    )


def load_prompt(mode: str = "baseline") -> str:
    """Load the baseline or improved prompt text used for real VLM calls."""
    filename = "improved_prompt.txt" if mode == "improved" else "baseline_prompt.txt"
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8")


def get_pipeline():
    """Lazily load the MedGemma pipeline once and reuse it for all predictions.

    Requires the HF_TOKEN environment variable to be set (never hardcode a
    token in source). Picks CUDA if available, otherwise falls back to CPU;
    "mps" is Apple-Silicon-only and must not be assumed here.
    """
    global _PIPE

    if _PIPE is not None:
        return _PIPE

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. "
            "Set it before starting the app, for example: "
            "export HF_TOKEN='<your-huggingface-token>'"
        )

    from transformers import pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _PIPE = pipeline(
        "image-text-to-text",
        model=MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device=device,
        token=hf_token,
    )

    return _PIPE


def medgemma_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Run real MedGemma inference on a chest X-ray image.

    Optional live path: not used by the default toy smoke test, eval, API or
    Streamlit flows, since it requires HF_TOKEN, network access and a
    multi-GB model download. Uses the prompt file matching ``mode``.
    """
    from PIL import Image

    start = time.perf_counter()
    pipe = get_pipeline()
    prompt_text = load_prompt(mode)

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt_text}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this X-ray"},
            ],
        },
    ]

    output = pipe(text=messages, max_new_tokens=2000)
    text = output[0]["generated_text"][-1]["content"]
    latency_ms = int((time.perf_counter() - start) * 1000)

    data = parse_model_json(text)
    data.setdefault("model_name", MODEL_ID)
    data.setdefault("prompt_version", f"{mode}_v1")
    data["latency_ms"] = latency_ms
    return data


def parse_model_json(text: str) -> dict[str, Any]:
    """Parse the model response into JSON.

    Supports both raw JSON and JSON wrapped in Markdown fences.
    """
    text = text.strip()

    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    json_str = fenced_match.group(1) if fenced_match else text

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Model did not return valid JSON. Raw response was:\n{text}"
        ) from exc


def vlm_predict_placeholder(image_path: str | Path, prompt: str) -> dict[str, Any]:
    """Placeholder for a Hugging Face / MedGemma / Gemma 4 VLM call.

    Students should keep the same output schema as toy_predict.
    """
    return toy_predict(image_path, mode="baseline")
