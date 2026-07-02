from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "sql" / "schema.sql"


def default_db_path() -> str:
    """Resolve the evidence DB path at call time.

    Reading the environment lazily (rather than at import) keeps the web demo
    configurable via EVIDENCE_DB_PATH and lets tests redirect logging to a
    temporary file without polluting the repository root.
    """
    return os.environ.get("EVIDENCE_DB_PATH", str(ROOT / "medical_ai_evidence.sqlite"))


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or default_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    conn = connect(db_path)
    conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit(); conn.close()


def insert_run(db_path: str | Path | None, case_id: str, image_path: str, prediction: dict) -> None:
    init_db(db_path)
    conn = connect(db_path)
    conn.execute(
        """
        INSERT INTO runs(case_id, image_path, model_name, prompt_version, prediction_json, predicted_class, confidence, latency_ms)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            case_id,
            image_path,
            prediction.get("model_name"),
            prediction.get("prompt_version"),
            json.dumps(prediction, ensure_ascii=False),
            prediction.get("predicted_class"),
            float(prediction.get("confidence", 0.0)),
            int(prediction.get("latency_ms", 0)),
        ),
    )
    conn.commit(); conn.close()


def log_prediction(case_id: str, image_path: str, prediction: dict, db_path: str | Path | None = None) -> None:
    """Best-effort logging helper for the web demo.

    Persisting evidence must never break the user-facing response, so any
    storage error is swallowed after being surfaced on stderr.
    """
    try:
        insert_run(db_path, case_id, image_path, prediction)
    except Exception as exc:  # pragma: no cover - defensive path
        print(f"[log_prediction] could not persist run: {exc}", file=sys.stderr)


def fetch_recent_runs(db_path: str | Path | None = None, limit: int = 50) -> list[dict]:
    """Return the most recent logged runs (for the history view)."""
    path = db_path or default_db_path()
    if not Path(path).exists():
        return []
    conn = connect(path)
    try:
        cursor = conn.execute(
            """
            SELECT created_at, case_id, predicted_class, confidence, model_name, prompt_version, latency_ms
            FROM runs
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()
