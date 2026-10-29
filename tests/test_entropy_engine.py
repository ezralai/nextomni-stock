"""Tests for Market Information Entropy & Ordinal Complexity Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.entropy import (
    calculate_permutation_entropy,
    calculate_sample_entropy,
    evaluate_market_entropy_profile,
    MarketEntropyReceipt,
)
from src.cli import entropy_report_command


def test_permutation_entropy_mathematical_bounds():
    """Verify Permutation Entropy theoretical bounds: 0 for monotonic, ~1 for white noise."""
    # 1. Perfectly deterministic monotonic trend
    monotonic_series = np.arange(100.0)
    pe_mono = calculate_permutation_entropy(monotonic_series, m=3, tau=1)
    assert np.isclose(pe_mono, 0.0, atol=1e-8)

    # 2. Gaussian white noise (maximum entropy)
    np.random.seed(42)
    noise_series = np.random.normal(0, 1, size=1000)
    pe_noise = calculate_permutation_entropy(noise_series, m=3, tau=1)
    assert 0.95 <= pe_noise <= 1.0

    # 3. Degenerate very short series
    assert calculate_permutation_entropy([1.0], m=3) == 1.0


def test_sample_entropy_properties():
    """Verify Sample Entropy measures regularity: periodic sine wave has lower entropy than noise."""
    t = np.linspace(0, 20 * np.pi, 500)
    periodic_series = np.sin(t)
    sampen_periodic = calculate_sample_entropy(periodic_series, m=2, r_factor=0.2)

    np.random.seed(42)
    noise_series = np.random.normal(0, 1, 500)
    sampen_noise = calculate_sample_entropy(noise_series, m=2, r_factor=0.2)

    # Periodic wave is far more regular (lower SampEn) than noise
    assert sampen_periodic < sampen_noise
    assert sampen_periodic >= 0.0


def test_predictability_index_and_regime_classification():
    """Verify Predictability Index mapping and regime state labels."""
    # Monotonic trend: PI = 100% -> HIGH_PREDICTABILITY_ORDER
    mono_prices = pd.Series(np.arange(1.0, 100.0))
    rec_mono = evaluate_market_entropy_profile("MONO", mono_prices, window=60, m=3)
    assert rec_mono.entropy_regime == "HIGH_PREDICTABILITY_ORDER"
    assert rec_mono.predictability_index_pct > 90.0

    # Gaussian random walk: high entropy -> HIGH_ENTROPY_RANDOM_WALK or MODERATE_STRUCTURE
    np.random.seed(42)
    rw_prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 300)))
    rec_rw = evaluate_market_entropy_profile("RW", rw_prices, window=60, m=3)
    assert rec_rw.entropy_regime in ["HIGH_ENTROPY_RANDOM_WALK", "MODERATE_STRUCTURE"]
    assert rec_rw.permutation_entropy > 0.80


def test_entropy_receipt_signature_and_tamper_detection():
    """Verify deterministic HMAC-SHA256 signature and tamper resilience."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1, 100)))
    receipt = evaluate_market_entropy_profile("AAPL", prices, window=50, m=3)

    assert receipt.ticker == "AAPL"
    assert len(receipt.signature) == 64
    assert receipt.verify_integrity() is True

    # Tampering with permutation_entropy must invalidate signature
    receipt.permutation_entropy = 0.1234
    assert receipt.verify_integrity() is False


def test_evaluate_market_entropy_profile_integration():
    """Verify end-to-end evaluation on realistic stock prices."""
    np.random.seed(42)
    n = 150
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    prices = pd.Series(
        150.0 + np.cumsum(np.random.normal(0.05, 1.2, size=n)),
        index=dates,
    )

    receipt = evaluate_market_entropy_profile("MSFT", prices, window=60, m=3)

    assert receipt.sample_sessions == 150
    assert 0.0 <= receipt.permutation_entropy <= 1.0
    assert 0.0 <= receipt.predictability_index_pct <= 100.0
    assert len(receipt.entropy_history) > 0
    assert "predictability_index" in receipt.entropy_history[-1]
    assert "MSFT" in receipt.summary


def test_cli_entropy_report_command(capsys):
    """Verify CLI entropy-report output formatting and JSON output."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    mock_df = pd.DataFrame({
        "session_date": dates,
        "adj_close": 100.0 + np.cumsum(np.random.normal(0.02, 1.0, size=n)),
        "close": 100.0 + np.cumsum(np.random.normal(0.02, 1.0, size=n)),
    })

    with patch("src.data.db.get_observations", return_value=mock_df):
        # 1. Text output
        args_text = argparse.Namespace(
            ticker="NVDA",
            window=60,
            embedding_dim=3,
            json=False,
        )
        entropy_report_command(args_text)
        out = capsys.readouterr().out
        assert "Market Information Entropy & Ordinal Complexity Engine" in out
        assert "Security / Ticker:            NVDA" in out
        assert "Permutation Entropy (H_P):" in out
        assert "Market Predictability Index:" in out

        # 2. JSON output
        args_json = argparse.Namespace(
            ticker="NVDA",
            window=60,
            embedding_dim=3,
            json=True,
        )
        entropy_report_command(args_json)
        json_out = capsys.readouterr().out
        data = json.loads(json_out)
        assert data["ticker"] == "NVDA"
        assert "permutation_entropy" in data
        assert "predictability_index_pct" in data
        assert "signature" in data


def test_entropy_profile_insufficient_data(capsys):
    """Verify graceful handling when data is too short or empty."""
    # 1. Short series (< 30)
    short_prices = pd.Series([10.0, 10.5, 11.0])
    receipt = evaluate_market_entropy_profile("TINY", short_prices)
    assert receipt.entropy_regime == "INSUFFICIENT_DATA"
    assert receipt.sample_sessions == 3

    # 2. CLI empty observation
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()):
        args_empty = argparse.Namespace(
            ticker="AAPL",
            window=60,
            embedding_dim=3,
            json=False,
        )
        entropy_report_command(args_empty)
        out = capsys.readouterr().out
        assert "No observation history found" in out
