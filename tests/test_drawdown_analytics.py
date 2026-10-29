"""Test suite for Burke Ratio, Becker Pain Index, and Ulcer Performance Index Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.drawdown_analytics import (
    DrawdownAnalyticsReceipt,
    compute_drawdown_metrics,
    evaluate_drawdown_profile,
    identify_drawdown_valleys,
)
from src.cli import pain_index_command


def test_identify_drawdown_valleys():
    """Verify identification of discrete drawdown episode peak depths."""
    # 0 -> 0.10 -> 0.20 -> 0 -> 0.05 -> 0.15 -> 0
    dd_series = np.array([0.0, 0.10, 0.20, 0.05, 0.0, 0.05, 0.15, 0.0])
    valleys = identify_drawdown_valleys(dd_series)
    assert len(valleys) == 2
    assert abs(valleys[0] - 0.20) < 1e-4
    assert abs(valleys[1] - 0.15) < 1e-4


def test_compute_drawdown_metrics_monotonic_zero_drawdown():
    """Verify zero-drawdown edge case on strictly increasing equity curve."""
    equity = np.array([1.0, 1.05, 1.10, 1.15, 1.20])
    metrics = compute_drawdown_metrics(equity, annualized_return=0.20, risk_free_rate=0.04)

    assert metrics["max_drawdown_pct"] == 0.0
    assert metrics["ulcer_index_pct"] == 0.0
    assert metrics["pain_index_pct"] == 0.0
    assert metrics["is_in_drawdown"] is False
    assert metrics["num_discrete_drawdowns"] == 0


def test_compute_drawdown_metrics_fluctuating_path():
    """Verify drawdown, Burke, Ulcer, and Pain metrics on fluctuating equity curve."""
    # Start at 1.0, drop to 0.8 (20% DD), recover to 1.1, drop to 0.99 (10% DD), finish at 1.2
    equity = np.array([1.0, 0.90, 0.80, 0.95, 1.10, 1.00, 0.99, 1.15, 1.20])
    metrics = compute_drawdown_metrics(equity, annualized_return=0.15, risk_free_rate=0.04)

    assert metrics["max_drawdown_pct"] == 20.0
    assert metrics["ulcer_index_pct"] > 0.0
    assert metrics["pain_index_pct"] > 0.0
    assert metrics["burke_ratio"] > 0.0
    assert metrics["pain_ratio"] > 0.0
    assert metrics["ulcer_performance_index"] > 0.0
    assert metrics["calmar_ratio"] > 0.0
    assert metrics["num_discrete_drawdowns"] == 2
    assert metrics["is_in_drawdown"] is False


def test_evaluate_drawdown_profile_receipt_and_tamper():
    """Verify full drawdown receipt generation and cryptographic HMAC tamper detection."""
    np.random.seed(42)
    ret = np.random.normal(0.0005, 0.015, size=80)
    prices = pd.Series(100.0 * np.exp(np.cumsum(ret)))

    receipt = evaluate_drawdown_profile("AAPL", prices, risk_free_rate=0.04)

    assert isinstance(receipt, DrawdownAnalyticsReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == 80
    assert receipt.max_drawdown_pct >= 0.0
    assert receipt.ulcer_index_pct >= 0.0
    assert receipt.pain_index_pct >= 0.0
    assert receipt.verify_integrity() is True

    # Tamper with Pain Index
    tampered_dict = receipt.to_dict()
    tampered_dict["pain_index_pct"] += 5.0
    tampered_receipt = DrawdownAnalyticsReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False


def test_drawdown_security_and_edge_cases():
    """Verify input validation and insufficient sample handling."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_drawdown_profile("INVALID_TICKER$$$", pd.Series([10.0] * 30))

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_drawdown_profile("AAPL", pd.Series([10.0, 11.0, 12.0]))


def test_cli_pain_index_command_stdout_and_json(monkeypatch):
    """Verify CLI pain-index command in stdout and JSON output modes."""
    np.random.seed(42)
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, size=50))))
    df_obs = pd.DataFrame({"close": prices})

    monkeypatch.setattr("src.data.db.get_observations", lambda ticker, db_path=None: df_obs)

    # 1. Standard stdout mode
    args_std = argparse.Namespace(ticker="AAPL", risk_free_rate=0.04, json=False)
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        pain_index_command(args_std)
        output = fake_out.getvalue()
        assert "Burke Ratio, Becker Pain Index & Drawdown Engine" in output
        assert "Security / Ticker:            AAPL" in output
        assert "Becker (2001) Pain Index:" in output
        assert "Burke (1994) Ratio:" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(ticker="AAPL", risk_free_rate=0.04, json=True)
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        pain_index_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert payload["ticker"] == "AAPL"
        assert "pain_index_pct" in payload
        assert "burke_ratio" in payload
        assert "ulcer_index_pct" in payload
        assert "signature" in payload
