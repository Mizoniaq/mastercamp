from __future__ import annotations

import statistics
from collections import Counter
from typing import Iterable

CLASSES = ["normal", "suspected_opacity", "uncertain"]


def accuracy(y_true: Iterable[str], y_pred: Iterable[str]) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def recall_for(y_true: Iterable[str], y_pred: Iterable[str], target: str) -> float:
    """Recall (sensitivity) for a single class: TP / (TP + FN)."""
    y_true = list(y_true); y_pred = list(y_pred)
    tp = sum(t == target and p == target for t, p in zip(y_true, y_pred))
    fn = sum(t == target and p != target for t, p in zip(y_true, y_pred))
    return tp / (tp + fn) if tp + fn else 0.0


def macro_f1(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> float:
    y_true = list(y_true); y_pred = list(y_pred)
    scores = []
    for c in classes:
        tp = sum(t == c and p == c for t, p in zip(y_true, y_pred))
        fp = sum(t != c and p == c for t, p in zip(y_true, y_pred))
        fn = sum(t == c and p != c for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        scores.append(f1)
    return sum(scores) / len(scores)


def confusion_counts(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, int]:
    counts = Counter()
    for t, p in zip(y_true, y_pred):
        counts[f"{t}__{p}"] += 1
    return dict(counts)


def confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str], classes: list[str] = CLASSES) -> list[dict]:
    """Confusion matrix as a list of rows suitable for CSV export."""
    y_true = list(y_true); y_pred = list(y_pred)
    rows = []
    for t in classes:
        row = {"true_label": t}
        for p in classes:
            row[f"pred_{p}"] = sum(a == t and b == p for a, b in zip(y_true, y_pred))
        rows.append(row)
    return rows


def summarize_metrics(rows: list[dict]) -> dict[str, float]:
    y_true = [r["label"] for r in rows]
    y_pred = [r["predicted_class"] for r in rows]
    json_valid = [r.get("json_valid", True) for r in rows]
    warnings = [bool(r.get("warning")) for r in rows]
    latencies = [float(r["latency_ms"]) for r in rows if r.get("latency_ms") not in (None, "")]
    return {
        "n": len(rows),
        "accuracy": round(accuracy(y_true, y_pred), 4),
        "macro_f1": round(macro_f1(y_true, y_pred), 4),
        # Sensitivity = recall on the class we least want to miss (suspected_opacity).
        "sensitivity": round(recall_for(y_true, y_pred, "suspected_opacity"), 4),
        # Specificity here = recall on the normal class (protocol wording).
        "specificity": round(recall_for(y_true, y_pred, "normal"), 4),
        "json_valid_rate": round(sum(json_valid) / len(json_valid), 4) if rows else 0,
        "warning_rate": round(sum(warnings) / len(warnings), 4) if rows else 0,
        "uncertain_rate": round(sum(p == "uncertain" for p in y_pred) / len(y_pred), 4) if rows else 0,
        "latency_ms_median": round(statistics.median(latencies), 1) if latencies else 0,
    }
