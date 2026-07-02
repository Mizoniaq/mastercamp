"""Build a project-schema cases CSV from a folder of real (authorized) images.

Produces `data/real/real_cases.csv` with the same columns as
`data/synthetic_cases.csv` so `eval/run_evaluation.py --cases ...` and
`eval/run_vlm_comparison.py --cases ...` can consume it.

Images are scanned **recursively**, so you can point --images straight at an
unzipped Kaggle folder (e.g. data/real/chest_xray). Labels are resolved in this
order:
  1. an explicit --labels CSV (columns: filename,label), else
  2. inferred from a parent folder name (NORMAL -> normal, PNEUMONIA -> suspected_opacity).

Usage:
    # Kaggle chest-xray-pneumonia, balanced 15 per class:
    python scripts/prepare_real_dataset.py --images data/real/chest_xray --per-class 15

    # Flat folder + explicit labels:
    python scripts/prepare_real_dataset.py --images data/real/images --labels data/real/labels.csv
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_LABELS = {"normal", "suspected_opacity", "uncertain", ""}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}

# Folder-name -> project class (Kaggle chest-xray-pneumonia and similar layouts).
FOLDER_LABELS = {
    "normal": "normal",
    "pneumonia": "suspected_opacity",
    "opacity": "suspected_opacity",
    "suspected_opacity": "suspected_opacity",
    "uncertain": "uncertain",
}


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


def infer_label_from_path(path: Path) -> str:
    """Derive a class from any parent folder name (case-insensitive)."""
    for part in reversed(path.parts[:-1]):
        key = part.strip().lower()
        if key in FOLDER_LABELS:
            return FOLDER_LABELS[key]
    return ""


def resolve_label(img: Path, explicit: dict[str, str]) -> str:
    return explicit.get(img.name) or infer_label_from_path(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=ROOT / "data" / "real" / "images")
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "real" / "real_cases.csv")
    parser.add_argument("--per-class", type=int, default=None,
                        help="keep at most N labelled images per class (balanced sample)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.images.exists():
        raise SystemExit(f"Images folder not found: {args.images}. See data/real/README.md.")

    explicit = load_labels(args.labels)
    all_images = sorted(
        p for p in args.images.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES
        and "__MACOSX" not in p.parts          # zip/macOS junk folder
        and not p.name.startswith("._")         # AppleDouble resource forks
    )
    if not all_images:
        raise SystemExit(f"No images under {args.images}. Add authorized, de-identified files first.")

    labelled = [(img, resolve_label(img, explicit)) for img in all_images]

    # Optional balanced subsampling per class (deterministic with --seed).
    if args.per_class:
        rng = random.Random(args.seed)
        by_class: dict[str, list[Path]] = {}
        for img, lab in labelled:
            by_class.setdefault(lab, []).append(img)
        selected = []
        for lab, imgs in by_class.items():
            rng.shuffle(imgs)
            selected += [(img, lab) for img in imgs[: args.per_class]]
        labelled = sorted(selected, key=lambda t: str(t[0]))

    def as_case_path(img: Path) -> str:
        # Relative to repo root when possible (portable); absolute otherwise.
        try:
            return img.relative_to(ROOT).as_posix()
        except ValueError:
            return img.resolve().as_posix()

    rows = []
    for i, (img, label) in enumerate(labelled, start=1):
        rows.append({
            "case_id": f"REAL_{i:03d}",
            "image_path": as_case_path(img),
            "source": "real_public_authorized",
            "label": label,
            "split": "real",
            "quality": "",
            "notes": "user-provided authorized sample",
        })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)

    n_labelled = sum(1 for r in rows if r["label"])
    from collections import Counter
    dist = Counter(r["label"] or "(unlabelled)" for r in rows)
    print(f"Écrit {args.out} : {len(rows)} images ({n_labelled} avec label).")
    print("Répartition :", dict(dist))


if __name__ == "__main__":
    main()
