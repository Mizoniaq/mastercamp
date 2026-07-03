from __future__ import annotations

import compileall
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from fastapi.testclient import TestClient

from api.main import app
from api.main import health
from src.database import fetch_recent_runs, log_prediction
from src.guardrails import WARNING_TEXT, apply_safety_guardrails, detect_overclaim, validate_prediction
from src.inference import robust_predict, toy_predict
from src.metrics import summarize_metrics


ROOT = Path(__file__).resolve().parents[1]


def _evaluate_mode(mode: str) -> dict:
    rows = []
    with (ROOT / "data" / "synthetic_cases.csv").open(encoding="utf-8", newline="") as f:
        for case in csv.DictReader(f):
            pred = apply_safety_guardrails(toy_predict(ROOT / case["image_path"], mode=mode))
            rows.append(
                {
                    "label": case["label"],
                    "predicted_class": pred["predicted_class"],
                    "json_valid": True,
                    "warning": pred.get("warning"),
                    "latency_ms": pred.get("latency_ms", 0),
                }
            )
    return summarize_metrics(rows)


def test_repository_student_contract_is_present() -> None:
    required_paths = [
        "README.md",
        "requirements.txt",
        "requirements-test.txt",
        ".github/workflows/ci.yml",
        "docs/appel_offre.md",
        "docs/architecture.md",
        "docs/ethique_et_limites.md",
        "docs/evaluation_protocol.md",
        "data/synthetic_cases.csv",
        "src/inference.py",
        "src/guardrails.py",
        "api/main.py",
        "eval/run_evaluation.py",
        "eval/run_vlm_comparison.py",
        "eval/build_error_register.py",
        "eval/threshold_sweep.py",
        "eval/error_register.csv",
        "scripts/prepare_real_dataset.py",
        "docs/rapport.md",
        "docs/error_analysis.md",
        "prompts/json_schema.md",
    ]
    forbidden_paths = [
        ".rollback_appel_offre_cleanup_20260516_205745",
        "VALIDATION_REPORT.md",
        "create_remote_repo.sh",
        "docs/expert_review_integration.md",
        "docs/github_push_instructions.md",
        "eval/outputs",
        "medical_ai_evidence.sqlite",
        "assets/assistant_radiologue_v3_notes_professeur_fr.pptx",
        "assets/notes_orales_assistant_radiologue_v3_style_professeur_fr.md",
    ]

    missing = [path for path in required_paths if not (ROOT / path).exists()]
    forbidden = [path for path in forbidden_paths if (ROOT / path).exists()]

    assert missing == []
    assert forbidden == []


def test_synthetic_dataset_contract_is_valid() -> None:
    path = ROOT / "data" / "synthetic_cases.csv"
    required_columns = {"case_id", "image_path", "source", "label", "split", "quality", "notes"}
    allowed_labels = {"normal", "suspected_opacity", "uncertain"}

    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) >= 20
    assert required_columns <= set(rows[0])
    assert {row["label"] for row in rows} <= allowed_labels
    for row in rows:
        assert row["source"] == "synthetic_toy"
        assert (ROOT / row["image_path"]).exists()


def test_prediction_schema_warning_and_guardrails() -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"
    pred = apply_safety_guardrails(toy_predict(image_path, mode="improved"))
    valid, errors = validate_prediction(pred)

    assert valid, errors
    assert pred["predicted_class"] in {"normal", "suspected_opacity", "uncertain"}
    assert pred["warning"] == WARNING_TEXT
    assert "not a validated medical model" in pred["limitations"]


def test_python_source_tree_compiles() -> None:
    for folder in ("src", "api", "app", "eval", "finetuning", "tests", "scripts"):
        assert compileall.compile_dir(ROOT / folder, quiet=1)


def test_invalid_model_output_falls_back_to_uncertain() -> None:
    pred = apply_safety_guardrails({"predicted_class": "diagnosis", "confidence": 0.99})

    assert pred["predicted_class"] == "uncertain"
    assert pred["confidence"] <= 0.5
    assert pred["warning"] == WARNING_TEXT
    assert pred["guardrail_errors"]


def test_metrics_and_api_health_contract() -> None:
    rows = [
        {"label": "normal", "predicted_class": "normal", "json_valid": True, "warning": WARNING_TEXT},
        {"label": "suspected_opacity", "predicted_class": "uncertain", "json_valid": True, "warning": WARNING_TEXT},
    ]
    metrics = summarize_metrics(rows)

    assert health()["status"] == "ok"
    assert health()["scope"] == "educational prototype, not diagnosis"
    assert metrics["n"] == 2
    assert metrics["json_valid_rate"] == 1.0
    assert metrics["warning_rate"] == 1.0


def test_api_predict_preserves_uploaded_case_signal(tmp_path, monkeypatch) -> None:
    # Redirect evidence logging to a temp DB so the test never pollutes the repo root.
    monkeypatch.setenv("EVIDENCE_DB_PATH", str(tmp_path / "evidence.sqlite"))
    client = TestClient(app)
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_002_suspected_opacity.png"

    with image_path.open("rb") as file:
        response = client.post(
            "/predict",
            files={"file": (image_path.name, file, "image/png")},
        )

    payload = response.json()
    assert response.status_code == 200
    assert payload["predicted_class"] == "suspected_opacity"
    assert payload["warning"] == WARNING_TEXT

    # The prediction must be traceable via the history endpoint (logging contract).
    history = client.get("/history").json()["runs"]
    assert history and history[0]["predicted_class"] == "suspected_opacity"
    shutil.rmtree(ROOT / "tmp_uploads", ignore_errors=True)


def test_evaluation_command_runs_and_preserves_warning_contract(tmp_path: Path) -> None:
    db_path = tmp_path / "medical_ai_evidence.sqlite"
    out_dir = tmp_path / "outputs"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)

    result = subprocess.run(
        [
            sys.executable,
            "eval/run_evaluation.py",
            "--mode",
            "toy",
            "--out-dir",
            str(out_dir),
            "--db-path",
            str(db_path),
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert {row["mode"] for row in summary} == {"baseline", "improved"}
    assert all(row["json_valid_rate"] == 1.0 for row in summary)
    assert all(row["warning_rate"] == 1.0 for row in summary)
    assert (out_dir / "before_after_summary.csv").exists()
    assert db_path.exists()


def test_improved_mode_measurably_beats_baseline() -> None:
    baseline = _evaluate_mode("baseline")
    improved = _evaluate_mode("improved")

    # The improved uncertainty rule must abstain more and score at least as well.
    assert improved["uncertain_rate"] > baseline["uncertain_rate"]
    assert improved["accuracy"] >= baseline["accuracy"]
    assert improved["macro_f1"] >= baseline["macro_f1"]
    # New metrics are exposed for the report/dashboard.
    for key in ("sensitivity", "specificity", "latency_ms_median"):
        assert key in improved


def test_improved_mode_abstains_on_limited_quality_image() -> None:
    image_path = ROOT / "data" / "sample_images" / "CXR_SYN_003_uncertain.png"
    pred = apply_safety_guardrails(toy_predict(image_path, mode="improved"))

    assert pred["predicted_class"] == "uncertain"
    assert pred["confidence"] <= 0.6
    assert pred["image_quality"] in {"limited", "poor"}


def test_prediction_logging_round_trip(tmp_path) -> None:
    db_path = tmp_path / "evidence.sqlite"
    pred = apply_safety_guardrails(toy_predict(
        ROOT / "data" / "sample_images" / "CXR_SYN_001_normal.png", mode="baseline"
    ))
    log_prediction(case_id="CXR_SYN_001", image_path="x.png", prediction=pred, db_path=db_path)

    runs = fetch_recent_runs(db_path=db_path)
    assert len(runs) == 1
    assert runs[0]["case_id"] == "CXR_SYN_001"
    assert runs[0]["predicted_class"] == pred["predicted_class"]


def test_robust_predict_handles_corrupt_input(tmp_path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"this is not an image")
    pred = apply_safety_guardrails(robust_predict(bad, mode="improved"))

    assert pred["predicted_class"] == "uncertain"
    assert pred["warning"] == WARNING_TEXT
    assert any("unreadable" in item for item in pred["limitations"])


def test_guardrails_survive_non_numeric_confidence() -> None:
    # A real VLM can emit a non-numeric confidence; guardrails must not crash.
    pred = apply_safety_guardrails({"predicted_class": "normal", "confidence": "high"})
    assert pred["predicted_class"] == "uncertain"
    assert isinstance(pred["confidence"], float)
    assert pred["warning"] == WARNING_TEXT


def test_overclaim_detector_flags_clinical_language() -> None:
    over = detect_overclaim({"justification": "Diagnosis confirmed: pneumonia with pleural effusion.", "visual_evidence": []})
    cautious = detect_overclaim({"justification": "A focal bright region compatible with the toy marker.", "visual_evidence": ["bright area"]})

    assert {"pneumonia", "pleural", "effusion", "diagnosis", "confirmed"} <= set(over)
    assert cautious == []


def test_quality_threshold_override_changes_abstention() -> None:
    normal_image = ROOT / "data" / "sample_images" / "CXR_SYN_001_normal.png"
    # A very high quality threshold over-flags even a normal image as limited.
    pred = apply_safety_guardrails(toy_predict(normal_image, mode="improved", limited_contrast_std=25.0))
    assert pred["predicted_class"] == "uncertain"
