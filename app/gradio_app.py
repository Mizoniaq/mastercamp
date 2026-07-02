from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails


def analyze(image_path, mode):
    if image_path is None:
        return {"error": "no image"}
    return apply_safety_guardrails(toy_predict(image_path, mode=mode))


demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Image(type="filepath", label="Radiographie thoracique"), gr.Radio(["baseline", "improved"], value="improved")],
    outputs=gr.JSON(label="Sortie structurée"),
    title="Assistant radiologue virtuel — prototype pédagogique",
    description="Non destiné au diagnostic. Validation par un professionnel qualifié requise.",
)

if __name__ == "__main__":
    demo.launch()
