"""Unit and integration tests for Information Coefficient drift and alpha decay monitor."""

from datetime import datetime, timezone
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.drift import (
    ICDriftReceipt,
    calculate_information_coefficient_drift,
)
from src.evaluation.daemon import EvaluationDaemon


def test_ic_drift_insufficient_data():
    """Verify handling when fewer than 5 evaluated predictions exist."""
    df = pd.DataFrame({
        "predicted_price": [105.0, 106.0],
        "origin_price": [100.0, 100.0],
        "actual_price": [104.0, 107.0],
    })
    receipt = calculate_information_coefficient_drift(df)
    assert isinstance(receipt, ICDriftReceipt)
    assert receipt.sample_size == 2
    assert receipt.decay_status == "INSUFFICIENT_DATA"
    assert not receipt.is_decaying
    assert receipt.decay_warning is not None
    assert receipt.mean_rank_ic == 0.0


def test_ic_drift_positive_monotonic_skill():
    """Verify strong positive ranking skill produces high positive rank IC and STABLE_ALPHA."""
    np.random.seed(42)
    n = 30
    origin_price = 100.0
    # Expected returns with high positive correlation with realized returns
    true_signal = np.linspace(-0.05, 0.05, n)
    pred_returns = true_signal + np.random.normal(0, 0.005, n)
    real_returns = true_signal + np.random.normal(0, 0.005, n)

    df = pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "horizon_type": ["5d"] * n,
        "target_session_date": [f"2024-01-{i+1:02d}" for i in range(n)],
        "origin_price": [origin_price] * n,
        "predicted_price": origin_price * (1.0 + pred_returns),
        "actual_price": origin_price * (1.0 + real_returns),
    })

    receipt = calculate_information_coefficient_drift(df, window=15)
    assert receipt.sample_size == n
    assert receipt.mean_rank_ic > 0.8
    assert receipt.mean_ic > 0.8
    assert receipt.information_ratio > 0.0
    assert receipt.t_statistic > 0.0
    assert receipt.decay_status == "STABLE_ALPHA"
    assert not receipt.is_decaying
    assert receipt.decay_warning is None
    assert len(receipt.history_df) == n


def test_ic_drift_acute_alpha_decay():
    """Verify inverted predictions trigger ACUTE_DECAY alert."""
    n = 25
    origin_price = 100.0
    # Consistently inverted returns (negative correlation)
    pred_returns = np.array([0.05, 0.04, 0.03, 0.02, 0.01] * 5)
    real_returns = -pred_returns

    df = pd.DataFrame({
        "ticker": ["MSFT"] * n,
        "horizon_type": ["1d"] * n,
        "target_session_date": [f"2024-02-{i+1:02d}" for i in range(n)],
        "origin_price": [origin_price] * n,
        "predicted_price": origin_price * (1.0 + pred_returns),
        "actual_price": origin_price * (1.0 + real_returns),
    })

    receipt = calculate_information_coefficient_drift(df, window=10)
    assert receipt.is_decaying
    assert receipt.decay_status == "ACUTE_DECAY"
    assert "Acute alpha decay alert" in receipt.decay_warning
    assert receipt.trailing_rank_ic < 0.0


def test_daemon_integration_includes_alpha_drift(tmp_path):
    """Verify EvaluationDaemon.run_once computes and returns alpha drift in receipt."""
    db_file = tmp_path / "test_daemon_drift.db"
    from src.data.db import init_db
    init_db(db_file)

    daemon = EvaluationDaemon(db_path=db_file)
    with patch.object(daemon, "check_exchange_sessions", return_value={"XNYS": {"is_open": False}}), \
         patch.object(daemon.tracker, "evaluate_pending_predictions", return_value={"evaluated": 0, "awaiting": 0, "pending": 0}):
        receipt = daemon.run_once()

    assert "alpha_drift" in receipt
    drift = receipt["alpha_drift"]
    assert drift["decay_status"] == "INSUFFICIENT_DATA"
    assert drift["sample_size"] == 0


def test_cli_ic_drift_command(capsys):
    """Verify CLI nextomni ic-drift executes cleanly with JSON output."""
    from src.cli import main
    import sys

    dummy_df = pd.DataFrame({
        "ticker": ["AAPL"] * 10,
        "horizon_type": ["5d"] * 10,
        "origin_price": [100.0] * 10,
        "predicted_price": [102.0] * 10,
        "actual_price": [103.0] * 10,
        "target_session_date": [f"2024-01-{i+1:02d}" for i in range(10)],
    })

    with patch("src.data.db.get_predictions", return_value=dummy_df), \
         patch.object(sys, "argv", ["nextomni", "ic-drift", "--ticker", "AAPL", "--horizon", "5", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["sample_size"] == 10
    assert "decay_status" in data
