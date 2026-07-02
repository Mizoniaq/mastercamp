"""Sensitivity analysis: how the improved-mode quality threshold affects outcomes.

The improved uncertainty rule abstains ("uncertain") when image contrast falls
below a threshold. Sweeping that threshold shows the trade-off between abstaining
too little (no benefit over baseline) and too much (over-abstention that destroys
accuracy). This supports the "impact des seuils de confiance" analysis.

Usage:
    python eval/threshold_sweep.py [--out eval/results/threshold_sweep.csv]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.inference import toy_predict
from src.guardrails import apply_safety_guardrails
from src.metrics import summarize_metrics


def sweep(thresholds: list[float]) -> list[dict]:
    with (ROOT / "data" / "synthetic_cases.csv").open(newline="", encoding="utf-8") as f:
        cases = list(csv.DictReader(f))

    results = []
    for thr in thresholds:
        rows = []
        for case in cases:
            pred = apply_safety_guardrails(
                toy_predict(ROOT / case["image_path"], mode="improved", limited_contrast_std=thr)
            )
            rows.append({
                "label": case["label"],
                "predicted_class": pred["predicted_class"],
                "json_valid": True,
                "warning": pred.get("warning"),
                "latency_ms": pred.get("latency_ms", 0),
            })
        m = summarize_metrics(rows)
        results.append({
            "limited_contrast_std": thr,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "uncertain_rate": m["uncertain_rate"],
            "sensitivity": m["sensitivity"],
            "specificity": m["specificity"],
        })
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results" / "threshold_sweep.csv")
    args = parser.parse_args()

    thresholds = [round(8.0 + 0.5 * i, 1) for i in range(0, 25)]  # 8.0 .. 20.0
    results = sweep(thresholds)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader(); writer.writerows(results)

    best = max(results, key=lambda r: (r["accuracy"], r["macro_f1"]))
    print(f"Sweep écrit : {args.out} ({len(results)} points)")
    print(f"Meilleur seuil (accuracy): {best['limited_contrast_std']} "
          f"-> accuracy={best['accuracy']} uncertain_rate={best['uncertain_rate']}")


if __name__ == "__main__":
    main()
