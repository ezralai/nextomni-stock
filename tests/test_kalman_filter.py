"""Tests for Online Kalman Filter Dynamic Hedge Ratio & Adaptive Spread Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.kalman import (
    OnlineKalmanFilterPairsTracker,
    compute_half_life,
    estimate_kalman_spread_series,
    evaluate_kalman_pairs_profile,
    KalmanSpreadReceipt,
)
from src.cli import kalman_spread_command


def test_online_kalman_filter_tracker_convergence_to_known_synthetic_slope():
    """Verify recursive Kalman filter converges accurately to synthetic alpha and beta."""
    np.random.seed(42)
    n = 300
    true_alpha = 5.0
    true_beta = 1.75

    x = np.random.normal(50.0, 5.0, size=n)
    noise = np.random.normal(0.0, 0.2, size=n)
    y = true_alpha + true_beta * x + noise

    tracker = OnlineKalmanFilterPairsTracker(delta=1e-4, measurement_noise_r=0.2**2)

    for t in range(n):
        a, b, e, z = tracker.update(y[t], x[t])

    # After 300 updates, parameters should be close to true values
    assert np.isclose(a, true_alpha, atol=1.0)
    assert np.isclose(b, true_beta, atol=0.05)


def test_kalman_filter_dynamic_adaptation_to_structural_break():
    """Verify that Kalman filter adapts to structural shifts in hedge ratio."""
    np.random.seed(42)
    n_half = 200
    n = n_half * 2

    x = np.random.normal(100.0, 10.0, size=n)
    noise = np.random.normal(0.0, 0.5, size=n)

    # Regime 1: beta = 1.2
    # Regime 2: beta = 2.4 (abrupt structural break)
    y = np.zeros(n)
    y[:n_half] = 10.0 + 1.2 * x[:n_half] + noise[:n_half]
    y[n_half:] = 10.0 + 2.4 * x[n_half:] + noise[n_half:]

    df_spread = estimate_kalman_spread_series(y, x, delta=5e-4)

    beta_regime1 = df_spread["beta"].iloc[n_half - 1]
    beta_regime2 = df_spread["beta"].iloc[-1]

    # Must track the structural shift
    assert np.isclose(beta_regime1, 1.2, atol=0.15)
    assert np.isclose(beta_regime2, 2.4, atol=0.15)
    assert beta_regime2 > beta_regime1 * 1.5


def test_compute_half_life_mean_reversion():
    """Verify Ornstein-Uhlenbeck half-life estimation."""
    np.random.seed(42)
    n = 200

    # 1. Mean-reverting process: s_t = 0.8 * s_{t-1} + eps
    # lambda = 0.8 - 1.0 = -0.2 -> half-life = -ln(2)/(-0.2) ~= 3.46 sessions
    s_mr = np.zeros(n)
    for t in range(1, n):
        s_mr[t] = 0.8 * s_mr[t - 1] + np.random.normal(0, 1)

    hl_mr = compute_half_life(pd.Series(s_mr))
    assert 2.0 <= hl_mr <= 6.0

    # 2. Pure random walk (lambda >= 0)
    s_rw = np.cumsum(np.random.normal(0, 1, size=n))
    hl_rw = compute_half_life(pd.Series(s_rw))
    assert hl_rw >= 10.0


def test_kalman_receipt_signature_and_tamper_detection():
    """Verify HMAC-SHA256 signature integrity and tamper resilience."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    x = 100.0 + np.cumsum(np.random.normal(0, 1, n))
    y = 5.0 + 1.5 * x + np.random.normal(0, 0.5, n)

    df_y = pd.DataFrame({"session_date": dates, "adj_close": y, "close": y})
    df_x = pd.DataFrame({"session_date": dates, "adj_close": x, "close": x})

    receipt = evaluate_kalman_pairs_profile("JPM", "BAC", df_y, df_x)

    assert receipt.y_ticker == "JPM"
    assert receipt.x_ticker == "BAC"
    assert len(receipt.signature) == 64
    assert receipt.verify_integrity() is True

    # Tampering with latest_beta must invalidate signature
    receipt.latest_beta = 9.999
    assert receipt.verify_integrity() is False


def test_evaluate_kalman_pairs_profile_integration():
    """Verify end-to-end pairs estimation with signal categorization and spread history."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    x = 80.0 + np.cumsum(np.random.normal(0.01, 1.0, n))
    y = 0.8 * x + np.random.normal(0, 0.3, n)

    df_y = pd.DataFrame({"session_date": dates, "adj_close": y, "close": y})
    df_x = pd.DataFrame({"session_date": dates, "adj_close": x, "close": x})

    receipt = evaluate_kalman_pairs_profile("MSFT", "AAPL", df_y, df_x)

    assert receipt.sample_sessions == 100
    assert np.isclose(receipt.latest_beta, 0.8, atol=0.1)
    assert receipt.current_signal in ["LONG_SPREAD", "SHORT_SPREAD", "EXIT_POSITION", "NEUTRAL"]
    assert len(receipt.spread_history) == 60
    assert "zscore" in receipt.spread_history[-1]
    assert "beta" in receipt.spread_history[-1]


def test_cli_kalman_spread_command(capsys):
    """Verify CLI kalman-spread command output formatting."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    x = 50.0 + np.cumsum(np.random.normal(0, 1, n))
    y = 10.0 + 1.2 * x + np.random.normal(0, 0.5, n)

    df_y = pd.DataFrame({"session_date": dates, "adj_close": y, "close": y})
    df_x = pd.DataFrame({"session_date": dates, "adj_close": x, "close": x})

    def mock_get_obs(ticker):
        if ticker == "JPM":
            return df_y
        return df_x

    with patch("src.data.db.get_observations", side_effect=mock_get_obs):
        # 1. Text output
        args_text = argparse.Namespace(
            y_ticker="JPM",
            x_ticker="BAC",
            delta=1e-4,
            json=False,
        )
        kalman_spread_command(args_text)
        out = capsys.readouterr().out
        assert "Online Kalman Filter Dynamic Hedge Ratio & Spread Tracker" in out
        assert "Paired Assets:                Y=JPM vs X=BAC" in out
        assert "Latest Dynamic Hedge Ratio" in out
        assert "Current Arbitrage Signal" in out

        # 2. JSON output
        args_json = argparse.Namespace(
            y_ticker="JPM",
            x_ticker="BAC",
            delta=1e-4,
            json=True,
        )
        kalman_spread_command(args_json)
        json_out = capsys.readouterr().out
        data = json.loads(json_out)
        assert data["y_ticker"] == "JPM"
        assert data["x_ticker"] == "BAC"
        assert "latest_beta" in data
        assert "signature" in data


def test_kalman_pairs_profile_insufficient_data(capsys):
    """Verify graceful handling when data is empty or too short."""
    # 1. Insufficient matching dates
    short_df = pd.DataFrame({"session_date": ["2024-01-01"], "adj_close": [10.0], "close": [10.0]})
    receipt = evaluate_kalman_pairs_profile("A", "B", short_df, short_df)
    assert receipt.current_signal == "INSUFFICIENT_DATA"
    assert receipt.sample_sessions == 1

    # 2. CLI missing observation
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()):
        args_missing = argparse.Namespace(
            y_ticker="XYZ",
            x_ticker="ABC",
            delta=1e-4,
            json=False,
        )
        kalman_spread_command(args_missing)
        out = capsys.readouterr().out
        assert "No observation history found" in out
