"""Generate a commented error register from the toy evaluation.

Runs the baseline and improved classifiers on every synthetic case, tags each
baseline outcome with the project error taxonomy (FN/FP/UA/JF/HT/OK) and records
the corrective effect of the improved uncertainty rule. The output is a single
CSV of 20-30 commented cases used for the error analysis and the oral defense.

Usage:
    python eval/build_error_register.py [--out eval/error_register.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails, validate_prediction


def classify_error(ground_truth: str, prediction: str, json_valid: bool) -> tuple[str, str]:
    """Return (error_type, severity) following the project taxonomy.

    FN  false negative  - an abnormal/unreadable case is called reassuringly normal
    FP  false positive   - a normal case is flagged as suspected opacity
    UA  acceptable unc.  - correctly abstains on an ambiguous/limited case
    JF  JSON format error - output not exploitable
    HT  hallucination     - assessed manually (never auto-flagged here)
    OK  correct           - prediction matches ground truth
    """
    if not json_valid:
        return "JF", "high"
    if ground_truth == prediction:
        if prediction == "uncertain":
            return "UA", "low"
        return "OK", "none"
    # Mismatches.
    if ground_truth == "suspected_opacity":
        return "FN", "high"  # missed a (toy) opacity
    if ground_truth == "normal" and prediction == "suspected_opacity":
        return "FP", "medium"
    if ground_truth == "uncertain" and prediction == "normal":
        # Forcing a confident "normal" on a limited-quality image is falsely reassuring.
        return "FN", "high"
    if ground_truth == "uncertain" and prediction == "suspected_opacity":
        return "FP", "medium"
    return "OTHER", "medium"


COMMENTS = {
    "OK": "Baseline classe correctement ce cas.",
    "UA": "Incertitude correctement signalée sur une image de qualité limitée.",
    "FN": "Baseline force une classe rassurante alors que l'image est anormale ou illisible.",
    "FP": "Baseline sur-signale une opacité sur un cas non pathologique.",
    "JF": "Sortie JSON non conforme au schéma attendu.",
    "OTHER": "Écart de classification à examiner manuellement.",
}


def corrective_action(baseline_pred: str, improved_pred: str, ground_truth: str) -> str:
    if baseline_pred == improved_pred:
        return "Aucune correction nécessaire (comportement identique)."
    if improved_pred == ground_truth:
        return "La règle d'incertitude (qualité + seuil 0.60) corrige le cas en mode amélioré."
    return "Le mode amélioré modifie la sortie ; vérifier le seuil et la qualité image."


def build(out_path: Path) -> list[dict]:
    cases_csv = ROOT / "data" / "synthetic_cases.csv"
    with cases_csv.open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    register: list[dict] = []
    for case in cases:
        image_path = ROOT / case["image_path"]
        gt = case["label"]

        base = apply_safety_guardrails(toy_predict(image_path, mode="baseline"))
        impr = apply_safety_guardrails(toy_predict(image_path, mode="improved"))
        base_valid, _ = validate_prediction(base)

        error_type, severity = classify_error(gt, base["predicted_class"], base_valid)
        register.append(
            {
                "case_id": case["case_id"],
                "ground_truth": gt,
                "baseline_prediction": base["predicted_class"],
                "improved_prediction": impr["predicted_class"],
                "baseline_confidence": base["confidence"],
                "improved_confidence": impr["confidence"],
                "error_type": error_type,
                "severity": severity,
                "comment": COMMENTS.get(error_type, ""),
                "corrective_action": corrective_action(
                    base["predicted_class"], impr["predicted_class"], gt
                ),
            }
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(register[0].keys()))
        writer.writeheader()
        writer.writerows(register)
    return register


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "error_register.csv")
    args = parser.parse_args()
    register = build(args.out)

    # Small console summary for the oral defense.
    from collections import Counter

    counts = Counter(row["error_type"] for row in register)
    print(f"Registre écrit : {args.out} ({len(register)} cas)")
    print("Répartition par type :", dict(counts))


if __name__ == "__main__":
    main()
