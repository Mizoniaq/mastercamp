from __future__ import annotations

import re
import shutil
from pathlib import Path
from fastapi import FastAPI, File, Query, UploadFile

from src.inference import robust_predict, rejection_result
from src.guardrails import apply_safety_guardrails
from src.input_gate import gate_input
from src.database import fetch_recent_runs, log_prediction

app = FastAPI(title="Assistant radiologue virtuel EFREI", version="0.1.0")
UPLOAD_DIR = Path("tmp_uploads")


@app.get("/")
def health() -> dict:
    return {"status": "ok", "scope": "educational prototype, not diagnosis"}


@app.post("/predict")
async def predict(file: UploadFile = File(...), mode: str = Query("improved")) -> dict:
    UPLOAD_DIR.mkdir(exist_ok=True)
    filename = Path(file.filename or "image.png").name
    suffix = Path(filename).suffix or ".png"
    stem = Path(filename).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    target = UPLOAD_DIR / f"uploaded_{safe_stem}{suffix}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    # Input gate (colour + discriminative detector): refuse non-radiographs before
    # any model. A corrupt file falls back to robust_predict (safe "uncertain").
    try:
        accepted, reason = gate_input(target)
    except Exception:
        pred = apply_safety_guardrails(robust_predict(target, mode=mode))
    else:
        pred = apply_safety_guardrails(
            robust_predict(target, mode=mode) if accepted else rejection_result(reason, mode=mode)
        )
    # Trace every prediction (best effort) so the demo satisfies the logging contract.
    log_prediction(case_id=safe_stem, image_path=str(target), prediction=pred)
    return pred


@app.get("/history")
def history(limit: int = Query(20, ge=1, le=200)) -> dict:
    """Return the most recent logged predictions for auditability."""
    return {"runs": fetch_recent_runs(limit=limit)}
