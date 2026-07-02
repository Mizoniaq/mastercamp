"""Build a project-schema cases CSV from a folder of real (authorized) images.

Produces `data/real/real_cases.csv` with the same columns as
`data/synthetic_cases.csv` so `eval/run_evaluation.py --cases ...` can consume it.

Usage:
    python scripts/prepare_real_dataset.py --images data/real/images [--labels data/real/labels.csv]

`labels.csv` (optional) must have columns: filename,label
(label in normal | suspected_opacity | uncertain). Missing labels are left empty
(the pipeline still runs, but accuracy is undefined without ground truth).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LABELS = {"normal", "suspected_opacity", "uncertain", ""}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}


def load_labels(labels_csv: Path | None) -> dict[str, str]:
    if not labels_csv or not labels_csv.exists():
        return {}
    with labels_csv.open(newline="", encoding="utf-8") as f:
        mapping = {}
        for row in csv.DictReader(f):
            label = (row.get("label") or "").strip()
            if label not in ALLOWED_LABELS:
                raise ValueError(f"Invalid label '{label}' for {row.get('filename')}")
            mapping[row["filename"].strip()] = label
        return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "real" / "images")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "real" / "real_cases.csv")
    args = parser.parse_args()

    if not args.images.exists():
        raise SystemExit(f"Images folder not found: {args.images}. See data/real/README.md.")

    labels = load_labels(args.labels)
    images = sorted(p for p in args.images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images in {args.images}. Add authorized, de-identified files first.")

    rows = []
    for i, img in enumerate(images, start=1):
        rel = img.relative_to(ROOT).as_posix()
        rows.append({
            "case_id": f"REAL_{i:03d}",
            "image_path": rel,
            "source": "real_public_authorized",
            "label": labels.get(img.name, ""),
            "split": "real",
            "quality": "",
            "notes": "user-provided authorized sample",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    labelled = sum(1 for r in rows if r["label"])
    print(f"Écrit {args.out} : {len(rows)} images ({labelled} avec label).")


if __name__ == "__main__":
    main()
