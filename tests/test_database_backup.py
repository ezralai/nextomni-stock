"""Unit tests for atomic online SQLite backup and disaster recovery in Iteration 6."""

import sqlite3
from pathlib import Path
import pandas as pd
import pytest

from src.data.db import (
    backup_database,
    init_db,
    save_observations,
    upsert_stock,
    verify_database_integrity,
)
from src.evaluation.tracker import PredictionTracker


@pytest.fixture
def test_db_env(tmp_path: Path):
    """Fixture to provision a temporary isolated SQLite database with initial schema and data."""
    db_file = tmp_path / "test_stock.db"
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)

    init_db(db_path=db_file)
    upsert_stock("AAPL", "XNYS", "Apple Inc.", "USD", "Technology", db_path=db_file)
    obs_df = pd.DataFrame([{
        "ticker": "AAPL",
        "exchange": "XNYS",
        "session_date": "2026-10-20",
        "open": 230.0,
        "high": 235.0,
        "low": 229.0,
        "close": 234.0,
        "adj_close": 234.0,
        "volume": 1000000,
        "data_status": "FINAL",
        "provider": "Mock",
    }])
    save_observations(obs_df, db_path=db_file)

    tracker = PredictionTracker(db_path=db_file)
    pred_id = tracker.record_prediction(
        ticker="AAPL",
        exchange="XNYS",
        forecast_origin_session="2026-10-20",
        origin_price=234.0,
        horizon_sessions=1,
        target_session_date="2026-10-21",
        predicted_price=236.0,
        model_name="Linear_Direct_1",
        model_version="1.0.0",
    )

    return {"db_file": db_file, "backup_dir": backup_dir, "pred_id": pred_id}


def test_atomic_backup_creation_and_integrity(test_db_env):
    """Verify backup_database successfully creates snapshot and confirms audit integrity."""
    db_file = test_db_env["db_file"]
    dst_backup = test_db_env["backup_dir"] / "snapshot.db"

    target_path = backup_database(destination_path=dst_backup, db_path=db_file)

    assert target_path == dst_backup
    assert dst_backup.exists()
    assert dst_backup.stat().st_size > 0

    integrity = verify_database_integrity(dst_backup)
    assert integrity["is_intact"] is True
    assert integrity["predictions_valid"] == 1


def test_backup_database_tamper_detection(test_db_env):
    """Verify that if a backup snapshot is tampered with post-backup, audit flags it."""
    db_file = test_db_env["db_file"]
    dst_backup = test_db_env["backup_dir"] / "tampered_snapshot.db"

    backup_database(destination_path=dst_backup, db_path=db_file)

    # Tamper with the snapshot
    conn = sqlite3.connect(dst_backup)
    conn.execute("UPDATE predictions SET predicted_price = 9999.0 WHERE ticker = 'AAPL'")
    conn.commit()
    conn.close()

    tamper_report = verify_database_integrity(db_path=dst_backup)
    assert tamper_report["is_intact"] is False
    assert tamper_report["predictions_valid"] == 0
