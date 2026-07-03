from __future__ import annotations

from pathlib import Path
import os
import time
import re
import json
from typing import Any

from .preprocessing import extract_features, quality_from_features, is_probably_cxr

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"

# Intended medical model (gated: needs HF_TOKEN + accepted licence). Override with
# the MODEL_ID env var to point at any open image-text-to-text VLM.
MODEL_ID = os.environ.get("MODEL_ID", "google/medgemma-4b-it")
WARNING = "Prototype pédagogique. Non destiné au diagnostic. Validation par un professionnel qualifié requise."

# Decision thresholds for the transparent toy classifier.
# A focal bright region covering more than this fraction of the image is the
# toy proxy for a "suspected opacity".
OPACITY_BRIGHT_FRACTION = 0.008
# Improved mode abstains ("uncertain") below this confidence, matching the rule
# stated in prompts/improved_prompt.txt.
CONFIDENCE_ABSTAIN_THRESHOLD = 0.60

BASE_LIMITATIONS = ["synthetic toy image", "no clinical context", "not a validated medical model"]

# Pipelines are cached per model id so the harness can reuse a loaded model.
_PIPES: dict[str, Any] = {}


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


def toy_predict(
    image_path: str | Path,
    mode: str = "baseline",
    limited_contrast_std: float | None = None,
) -> dict[str, Any]:
    """Transparent image-feature classifier used to validate the repo pipeline.

    It reads simple grayscale statistics from the image (not the filename) and
    applies one of two rules:

    - ``baseline``: focal-bright-region rule only, never abstains.
    - ``improved``: adds an explicit uncertainty rule — abstain to "uncertain"
      when image quality is limited/poor or when confidence is below
      ``CONFIDENCE_ABSTAIN_THRESHOLD``.

    ``limited_contrast_std`` overrides the quality threshold (for the sensitivity
    analysis). This is a reproducible software validation, not medical inference.
    It runs with no GPU and no network so it works in CI and in the smoke test.
    """
    start = time.perf_counter()
    features = extract_features(image_path)
    if limited_contrast_std is None:
        quality = quality_from_features(features)
    else:
        quality = quality_from_features(features, limited_std=limited_contrast_std)

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


def rejection_result(reason: str, mode: str = "improved") -> dict[str, Any]:
    """Schema-valid response for an out-of-scope input (not a chest X-ray)."""
    return {
        "image_quality": "poor",
        "predicted_class": "uncertain",
        "confidence": 0.0,
        "visual_evidence": [reason],
        "justification": (
            "L'image fournie ne semble pas être une radiographie thoracique frontale ; "
            "l'analyse est refusée par sécurité."
        ),
        "limitations": BASE_LIMITATIONS + ["entrée hors périmètre (non-radiographie)"],
        "warning": WARNING,
        "model_name": "input-gate",
        "prompt_version": f"{mode}_v1",
        "latency_ms": 0,
        "input_rejected": True,
        "reject_reason": reason,
    }


def robust_predict(image_path: str | Path, mode: str = "improved") -> dict[str, Any]:
    """Toy prediction that never raises on a bad upload.

    Rejects inputs that are not plausibly a chest X-ray (colour photos), and falls
    back to a safe "uncertain" output on a corrupt/undecodable file — always with
    the mandatory warning, never a crash.
    """
    try:
        ok, reason = is_probably_cxr(image_path)
        if not ok:
            return rejection_result(reason, mode=mode)
        return toy_predict(image_path, mode=mode)
    except Exception as exc:  # unreadable / unsupported input
        return {
            "image_quality": "poor",
            "predicted_class": "uncertain",
            "confidence": 0.0,
            "visual_evidence": ["input could not be read as an image"],
            "justification": (
                "The uploaded file could not be decoded as an image, so no analysis is possible "
                "and the safe output is uncertainty."
            ),
            "limitations": BASE_LIMITATIONS + ["unreadable or unsupported input"],
            "warning": WARNING,
            "model_name": f"toy-imgfeat-{mode}",
            "prompt_version": f"{mode}_v1",
            "latency_ms": 0,
            "input_error": str(exc),
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


def get_pipeline(model_id: str | None = None):
    """Lazily load an image-text-to-text VLM pipeline and cache it per model id.

    ``HF_TOKEN`` is only required for gated models (e.g. MedGemma); open models
    load without it. Picks CUDA if available, otherwise CPU; "mps" is
    Apple-Silicon-only and must not be assumed here.
    """
    model_id = model_id or MODEL_ID
    if model_id in _PIPES:
        return _PIPES[model_id]

    from transformers import pipeline
    import torch

    use_cuda = torch.cuda.is_available()
    pipe = pipeline(
        "image-text-to-text",
        model=model_id,
        torch_dtype=torch.bfloat16 if use_cuda else torch.float32,
        device="cuda" if use_cuda else "cpu",
        token=os.environ.get("HF_TOKEN"),  # None is fine for open models
    )
    _PIPES[model_id] = pipe
    return pipe


def vlm_predict(image_path: str | Path, mode: str = "baseline", model_id: str | None = None) -> dict[str, Any]:
    """Run a real VLM on a chest X-ray using the prompt file matching ``mode``.

    Optional live path (not used by the default toy smoke test/eval/API/Streamlit):
    it requires a model download and, for gated models, HF_TOKEN. Works with any
    open image-text-to-text model via ``model_id`` / the MODEL_ID env var.
    """
    from PIL import Image

    model_id = model_id or MODEL_ID
    start = time.perf_counter()
    pipe = get_pipeline(model_id)
    prompt_text = load_prompt(mode)

    image = Image.open(image_path).convert("RGB")
    # Fold the prompt into a single user turn (image first) for broad model
    # compatibility: some chat templates (e.g. Gemma family) reject a system turn.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text + "\n\nAnswer with the required JSON only."},
            ],
        },
    ]

    max_new_tokens = int(os.environ.get("VLM_MAX_NEW_TOKENS", "512"))
    output = pipe(text=messages, max_new_tokens=max_new_tokens)
    text = output[0]["generated_text"][-1]["content"]
    latency_ms = int((time.perf_counter() - start) * 1000)

    try:
        data = parse_model_json(text)
    except ValueError:
        # Real models do not always return valid JSON; the guardrails downstream
        # turn this into a safe "uncertain". We keep the raw text for auditing.
        data = {"predicted_class": "uncertain", "raw_text": text[:500], "json_error": True}

    data.setdefault("model_name", model_id)
    data["prompt_version"] = f"{mode}_v1"
    data["latency_ms"] = latency_ms
    return data


# Backwards-compatible alias for the intended medical model.
def medgemma_predict(image_path: str | Path, mode: str = "baseline") -> dict[str, Any]:
    """Convenience wrapper: real inference with the MedGemma model id."""
    return vlm_predict(image_path, mode=mode, model_id=MODEL_ID)


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
