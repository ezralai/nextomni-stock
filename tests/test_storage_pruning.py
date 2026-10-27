"""Unit and security tests for SQLite raw provider payload storage pruning.

Verifies that:
1. Expired raw JSON payloads older than retention_days are pruned.
2. Fresh raw JSON payloads within the retention window are preserved.
3. Observations, predictions, and evaluation audit records are untouched.
4. Cryptographic prediction audit trail integrity (SHA-256) is 100% preserved.
5. Invalid retention parameters are rejected with validation errors.
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.data.db import (
    get_connection,
    init_db,
    prune_raw_payloads,
    save_observations,
    save_raw_payload,
    upsert_stock,
    verify_database_integrity,
)
from src.evaluation.tracker import PredictionTracker
from src.security.validators import validate_ticker


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate 60 sessions of deterministic synthetic price data."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=60)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    volumes = [10000 + int(abs(r) * 100000) for r in returns]
    return pd.DataFrame(
        {
            "ticker": ["JPM"] * 60,
            "exchange": ["XNYS"] * 60,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "data_status": ["END_OF_DAY"] * 60,
            "provider": ["Mock"] * 60,
        }
    )



def test_prune_raw_payloads_lifecycle(tmp_path: Path):
    """Test pruning deletes expired payloads and preserves recent ones."""
    db_file = tmp_path / "prune_test.db"
    init_db(db_file)

    # Save a fresh payload using standard helper
    save_raw_payload(
        ticker="JPM",
        provider="twelve_data",
        payload={"sample": "recent_data"},
        db_path=db_file,
    )

    # Manually backdate an old payload to 120 days ago
    conn = get_connection(db_file)
    with conn:
        conn.execute(
            """
            INSERT INTO raw_provider_payloads (
                ticker, provider, period_start, period_end, payload_json, retrieved_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now', '-120 days'));
            """,
            ("JPM", "twelve_data", "2026-01-01", "2026-01-10", '{"sample": "old_data"}'),
        )
        conn.execute(
            """
            INSERT INTO raw_provider_payloads (
                ticker, provider, period_start, period_end, payload_json, retrieved_at
            ) VALUES (?, ?, ?, ?, ?, datetime('now', '-95 days'));
            """,
            ("1155.KL", "yahoo_finance", "2026-01-01", "2026-01-10", '{"sample": "old_data_2"}'),
        )
    conn.close()

    # Verify total 3 records before pruning
    conn = get_connection(db_file)
    total_before = conn.execute("SELECT COUNT(*) FROM raw_provider_payloads;").fetchone()[0]
    conn.close()
    assert total_before == 3

    # Prune with 90 days retention
    pruned_count = prune_raw_payloads(retention_days=90, db_path=db_file)
    assert pruned_count == 2

    # Verify remaining records
    conn = get_connection(db_file)
    remaining_rows = conn.execute("SELECT * FROM raw_provider_payloads;").fetchall()
    conn.close()
    assert len(remaining_rows) == 1
    assert remaining_rows[0]["ticker"] == "JPM"


def test_prune_raw_payloads_validation(tmp_path: Path):
    """Verify retention_days input validation."""
    db_file = tmp_path / "prune_val.db"
    init_db(db_file)

    with pytest.raises(ValueError, match="retention_days must be >= 1"):
        prune_raw_payloads(retention_days=0, db_path=db_file)

    with pytest.raises(ValueError, match="retention_days must be >= 1"):
        prune_raw_payloads(retention_days=-15, db_path=db_file)


def test_prune_preserves_audit_trail_and_observations(tmp_path: Path, sample_timeseries_df):
    """Verify that pruning raw payloads does NOT touch audit records or market observations."""
    db_file = tmp_path / "prune_audit.db"
    init_db(db_file)

    # 1. Upsert stock and insert observations
    upsert_stock("JPM", "XNYS", "JPMorgan Chase & Co.", "USD", "America/New_York", db_path=db_file)
    obs_df = sample_timeseries_df.copy()
    obs_df["ticker"] = "JPM"
    obs_df["exchange"] = "XNYS"
    save_observations(obs_df, db_path=db_file)

    # 2. Record a signed prediction
    tracker = PredictionTracker(db_path=db_file)
    pred_res = tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2026-04-01",
        origin_price=195.0,
        horizon_sessions=1,
        target_session_date="2026-04-02",
        predicted_price=200.0,
        model_name="ConvexEnsemble",
        model_version="1.0.0",
        lower_bound=190.0,
        upper_bound=210.0,
    )
    assert pred_res["prediction_id"] is not None

    # Verify initial database integrity passes
    integrity_init = verify_database_integrity(db_file)
    assert integrity_init["is_intact"] is True
    assert integrity_init["predictions_valid"] == 1

    # 3. Insert and prune raw payloads
    conn = get_connection(db_file)
    with conn:
        conn.execute(
            """
            INSERT INTO raw_provider_payloads (
                ticker, provider, payload_json, retrieved_at
            ) VALUES (?, ?, ?, datetime('now', '-200 days'));
            """,
            ("JPM", "twelve_data", '{"old": true}'),
        )
    conn.close()

    deleted = prune_raw_payloads(retention_days=30, db_path=db_file)
    assert deleted == 1

    # 4. Confirm observations and predictions are 100% intact
    conn = get_connection(db_file)
    obs_count = conn.execute("SELECT COUNT(*) FROM market_observations;").fetchone()[0]
    pred_count = conn.execute("SELECT COUNT(*) FROM predictions;").fetchone()[0]
    conn.close()
    assert obs_count == len(obs_df)
    assert pred_count == 1

    # 5. Confirm audit trail cryptographic hashes remain 100% valid
    integrity_after = verify_database_integrity(db_file)
    assert integrity_after["is_intact"] is True
    assert integrity_after["predictions_valid"] == 1
    assert len(integrity_after["tampered_predictions"]) == 0


def test_prune_db_command_cli(monkeypatch, tmp_path: Path):
    """Verify CLI prune_db_command execution."""
    import argparse
    from src.cli import prune_db_command

    db_file = tmp_path / "cli_prune.db"
    init_db(db_file)
    monkeypatch.setattr("src.data.db.DATABASE_PATH", db_file)

    save_raw_payload("JPM", "twelve_data", {"msg": "test"}, db_path=db_file)
    args = argparse.Namespace(days=90)
    prune_db_command(args)

