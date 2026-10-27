"""Unit tests for cryptographic audit hashing and tamper detection."""

import sqlite3
import pytest
import pandas as pd
from pathlib import Path

from src.data.db import (
    get_connection,
    init_db,
    save_prediction,
    save_prediction_evaluation,
    upsert_stock,
    verify_database_integrity,
)
from src.security.audit import (
    compute_evaluation_audit_hash,
    compute_prediction_audit_hash,
    verify_evaluation_hash,
    verify_prediction_hash,
)


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_audit.db"
    init_db(db_file)
    upsert_stock("JPM", "XNYS", "JPMorgan Chase & Co.", "USD", "America/New_York", db_path=db_file)
    return db_file


def test_audit_hash_determinism():
    """Verify that identical payloads produce identical deterministic SHA-256 digests."""
    pred = {
        "prediction_id": "pred-001",
        "ticker": "JPM",
        "exchange": "XNYS",
        "forecast_origin_session": "2026-10-20",
        "origin_price": 220.50,
        "horizon_type": "5_SESSIONS",
        "horizon_sessions": 5,
        "target_session_date": "2026-10-27",
        "predicted_price": 224.75,
        "model_name": "HistGradientBoosting",
        "model_version": "1.0.0",
    }
    hash1 = compute_prediction_audit_hash(pred)
    hash2 = compute_prediction_audit_hash(pred)
    assert hash1 == hash2
    assert len(hash1) == 64
    assert verify_prediction_hash(pred, hash1) is True

    # Altering field fails verification
    pred_altered = dict(pred)
    pred_altered["predicted_price"] = 225.00
    assert verify_prediction_hash(pred_altered, hash1) is False


def test_database_integrity_verification_and_tamper_detection(temp_db: Path):
    """Verify save_prediction stores hash and detect unauthorized row modifications."""
    # 1. Save valid prediction
    pred = {
        "prediction_id": "pred-test-1",
        "ticker": "JPM",
        "exchange": "XNYS",
        "currency": "USD",
        "forecast_origin_timestamp": "2026-10-20T10:00:00",
        "forecast_origin_session": "2026-10-20",
        "origin_price": 220.0,
        "horizon_type": "1_SESSION",
        "horizon_sessions": 1,
        "target_session_date": "2026-10-21",
        "predicted_price": 221.5,
        "lower_bound": 219.0,
        "upper_bound": 224.0,
        "confidence_level": 0.80,
        "model_name": "HistGradientBoosting",
        "model_version": "1.0.0",
    }
    save_prediction(pred, db_path=temp_db)

    # 2. Check initial integrity
    audit_report = verify_database_integrity(temp_db)
    assert audit_report["is_intact"] is True
    assert audit_report["predictions_audited"] == 1
    assert audit_report["predictions_valid"] == 1
    assert len(audit_report["tampered_predictions"]) == 0

    # 3. Simulate unauthorized database tampering on predicted_price
    conn = get_connection(temp_db)
    with conn:
        conn.execute(
            "UPDATE predictions SET predicted_price = 999.99 WHERE prediction_id = 'pred-test-1';"
        )
    conn.close()

    # 4. Verify tampering is immediately detected
    tamper_report = verify_database_integrity(temp_db)
    assert tamper_report["is_intact"] is False
    assert tamper_report["predictions_valid"] == 0
    assert "pred-test-1" in tamper_report["tampered_predictions"]


def test_evaluation_integrity_and_tamper_detection(temp_db: Path):
    """Verify save_prediction_evaluation stores hash and detects tampering."""
    pred = {
        "prediction_id": "pred-eval-1",
        "ticker": "JPM",
        "exchange": "XNYS",
        "currency": "USD",
        "forecast_origin_timestamp": "2026-10-20T10:00:00",
        "forecast_origin_session": "2026-10-20",
        "origin_price": 220.0,
        "horizon_type": "1_SESSION",
        "horizon_sessions": 1,
        "target_session_date": "2026-10-21",
        "predicted_price": 221.5,
        "model_name": "LinearRidge",
        "model_version": "1.0.0",
    }
    save_prediction(pred, db_path=temp_db)

    eval_rec = {
        "prediction_id": "pred-eval-1",
        "actual_price": 222.0,
        "actual_source": "YahooFinance",
        "actual_observed_at": "2026-10-21T16:00:00",
        "absolute_error": 0.5,
        "percentage_error": 0.225,
        "actual_direction": 1,
        "predicted_direction": 1,
        "direction_correct": True,
    }
    save_prediction_evaluation(eval_rec, db_path=temp_db)

    # 1. Initial report is intact
    report = verify_database_integrity(temp_db)
    assert report["is_intact"] is True
    assert report["evaluations_audited"] == 1
    assert report["evaluations_valid"] == 1

    # 2. Tamper evaluation record
    conn = get_connection(temp_db)
    with conn:
        conn.execute(
            "UPDATE prediction_evaluations SET actual_price = 300.0 WHERE prediction_id = 'pred-eval-1';"
        )
    conn.close()

    tamper_report = verify_database_integrity(temp_db)
    assert tamper_report["is_intact"] is False
    assert "pred-eval-1" in tamper_report["tampered_evaluations"]
