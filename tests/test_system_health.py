"""Unit and security verification tests for Unified System Health Pre-Flight Audit."""

import argparse
import json
from pathlib import Path
import sqlite3
import pytest

from src.cli import system_health_command
from src.data.db import get_connection, init_db, save_prediction, upsert_stock
from src.security.audit import sign_model_artifact
from src.security.health import run_system_health_audit


def test_system_health_audit_healthy(tmp_path: Path):
    """Verify clean system passes all health checks."""
    db_file = tmp_path / "health_clean.db"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_file)

    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_path=db_file)
    pred = {
        "prediction_id": "pred-hlth-1",
        "ticker": "JPM",
        "exchange": "XNYS",
        "currency": "USD",
        "forecast_origin_timestamp": "2026-04-01T16:00:00",
        "forecast_origin_session": "2026-04-01",
        "origin_price": 200.0,
        "horizon_type": "1_SESSION",
        "horizon_sessions": 1,
        "target_session_date": "2026-04-02",
        "predicted_price": 202.0,
        "model_name": "ConvexEnsemble",
        "model_version": "1.0.0",
    }
    save_prediction(pred, db_path=db_file)

    # Create and sign a dummy model file
    dummy_model = models_dir / "test_model.joblib"
    dummy_model.write_bytes(b"dummy_model_bytes_12345")
    sign_model_artifact(dummy_model)

    receipt = run_system_health_audit(db_path=db_file, models_dir=models_dir)
    assert receipt.is_healthy is True
    assert receipt.database_integrity == "PASS"
    assert receipt.foreign_keys_integrity == "PASS"
    assert receipt.prediction_audit == "PASS"
    assert receipt.model_signatures == "PASS"
    assert receipt.verified_models_count == 1
    assert len(receipt.tampered_models) == 0


def test_system_health_detects_tampered_model(tmp_path: Path):
    """Verify system health flags tampered model artifacts."""
    db_file = tmp_path / "health_tamper_mod.db"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_file)

    dummy_model = models_dir / "tampered_model.joblib"
    dummy_model.write_bytes(b"original_model_weights")
    sign_model_artifact(dummy_model)

    # Tamper with the model file bytes
    dummy_model.write_bytes(b"tampered_malicious_weights")

    receipt = run_system_health_audit(db_path=db_file, models_dir=models_dir)
    assert receipt.is_healthy is False
    assert receipt.model_signatures == "FAIL"
    assert len(receipt.tampered_models) == 1
    assert "tampered_model.joblib" in receipt.tampered_models[0]


def test_system_health_detects_db_tampering(tmp_path: Path):
    """Verify system health flags tampered prediction records."""
    db_file = tmp_path / "health_tamper_db.db"
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    init_db(db_file)

    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_path=db_file)
    pred = {
        "prediction_id": "pred-tamper-1",
        "ticker": "JPM",
        "exchange": "XNYS",
        "currency": "USD",
        "forecast_origin_timestamp": "2026-04-01T16:00:00",
        "forecast_origin_session": "2026-04-01",
        "origin_price": 200.0,
        "horizon_type": "1_SESSION",
        "horizon_sessions": 1,
        "target_session_date": "2026-04-02",
        "predicted_price": 202.0,
        "model_name": "ConvexEnsemble",
        "model_version": "1.0.0",
    }
    save_prediction(pred, db_path=db_file)

    # Tamper with prediction record directly in SQLite
    conn = get_connection(db_file)
    with conn:
        conn.execute("UPDATE predictions SET predicted_price = 9999.0 WHERE prediction_id = 'pred-tamper-1';")
    conn.close()

    receipt = run_system_health_audit(db_path=db_file, models_dir=models_dir)
    assert receipt.is_healthy is False
    assert receipt.prediction_audit == "FAIL"
    assert receipt.tampered_predictions_count == 1


def test_cli_system_health_command(monkeypatch, tmp_path: Path, capsys):
    """Verify CLI system-health command runs in both formatted and JSON modes."""
    db_file = tmp_path / "cli_health.db"
    init_db(db_file)
    monkeypatch.setattr("src.security.health.DATABASE_PATH", db_file)

    # Formatted run
    system_health_command(argparse.Namespace(json=False))
    captured = capsys.readouterr()
    assert "NextOmni Stock: Unified Security & Health Audit" in captured.out
    assert "Overall Health Status:" in captured.out

    # JSON run
    system_health_command(argparse.Namespace(json=True))
    json_captured = capsys.readouterr()
    parsed = json.loads(json_captured.out)
    assert "is_healthy" in parsed
    assert "database_integrity" in parsed
    assert "prediction_audit" in parsed
