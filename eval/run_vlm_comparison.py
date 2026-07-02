"""Prompt comparison on a REAL vision-language model (baseline vs improved prompt).

This closes the "measured improvement" and "hallucination metric" goals with real
inference instead of the toy rule. For each prompt it runs the model over a subset
of cases, applies the same guardrails, and reports accuracy, macro-F1, JSON
validity, an unfounded-justification (over-claim) rate and latency.

Model selection:
  - Intended medical model (gated, needs HF_TOKEN + accepted licence):
        python eval/run_vlm_comparison.py --model google/medgemma-4b-it
  - Accessible open stand-in used to validate the harness end to end:
        python eval/run_vlm_comparison.py --model HuggingFaceTB/SmolVLM-256M-Instruct

The stand-in is a general (non-medical) VLM: it demonstrates the engineering
harness (two prompts, guardrails, JSON validation, hallucination check, metrics),
not medical performance.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import vlm_predict
from src.guardrails import apply_safety_guardrails, validate_prediction, detect_overclaim
from src.metrics import summarize_metrics
from src.database import insert_run


def read_cases(limit: int | None) -> list[dict]:
    with (ROOT / "data" / "synthetic_cases.csv").open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))
    return cases[:limit] if limit else cases


def run_mode(mode: str, model_id: str, cases: list[dict], db_path: Path) -> tuple[list[dict], dict]:
    rows = []
    overclaim_hits = 0
    json_errors = 0
    for case in cases:
        image_path = ROOT / case["image_path"]
        pred = apply_safety_guardrails(vlm_predict(image_path, mode=mode, model_id=model_id))
        valid, _ = validate_prediction(pred)
        overclaim = detect_overclaim(pred)
        overclaim_hits += bool(overclaim)
        json_errors += bool(pred.get("json_error"))
        rows.append(
            {
                "case_id": case["case_id"],
                "label": case["label"],
                "predicted_class": pred["predicted_class"],
                "confidence": pred["confidence"],
                "json_valid": valid,
                "overclaim_terms": ";".join(overclaim),
                "warning": pred.get("warning", ""),
                "latency_ms": pred.get("latency_ms", 0),
            }
        )
        insert_run(db_path, case["case_id"], str(image_path), pred)

    metrics = summarize_metrics(rows)
    metrics["overclaim_rate"] = round(overclaim_hits / len(rows), 4) if rows else 0.0
    metrics["json_error_rate"] = round(json_errors / len(rows), 4) if rows else 0.0
    metrics["model"] = model_id
    return rows, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="HuggingFaceTB/SmolVLM-256M-Instruct",
                        help="HF image-text-to-text model id (use google/medgemma-4b-it with HF_TOKEN)")
    parser.add_argument("--limit", type=int, default=9, help="number of cases (0 = all)")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--db-path", type=Path, default=ROOT / "medical_ai_evidence.sqlite")
    args = parser.parse_args()

    cases = read_cases(args.limit or None)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for mode in ("baseline", "improved"):
        rows, metrics = run_mode(mode, args.model, cases, args.db_path)
        with (args.out_dir / f"vlm_{mode}_predictions.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        (args.out_dir / f"vlm_{mode}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        summary.append({"mode": mode, **metrics})

    with (args.out_dir / "vlm_before_after_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader(); w.writerows(summary)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
