"""Test suite for Hamilton (1989) Two-State Markov-Switching Dynamic Regime Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.markov_regime import (
    MarkovRegimeReceipt,
    evaluate_markov_regime_profile,
    fit_two_state_markov_switching,
    run_hamilton_filter,
    run_kim_smoother,
)
from src.cli import markov_regime_command


@pytest.fixture
def synthetic_regime_switching_returns():
    """Create reproducible synthetic return series with distinct low-vol and high-vol regimes."""
    np.random.seed(42)
    # Phase 1: Calm Low-Vol Expansion (70 sessions, daily std 0.008)
    ret_low = np.random.normal(0.001, 0.008, size=70)
    # Phase 2: Turbulent High-Vol Crisis (50 sessions, daily std 0.035)
    ret_high = np.random.normal(-0.002, 0.035, size=50)

    returns = np.concatenate([ret_low, ret_high])
    return returns


def test_hamilton_filter_and_kim_smoother_probabilities(synthetic_regime_switching_returns):
    """Verify forward filter and backward smoother mathematical properties."""
    returns = synthetic_regime_switching_returns
    t_len = len(returns)

    mu = np.array([0.001, -0.002])
    var = np.array([0.008 ** 2, 0.035 ** 2])
    trans_matrix = np.array([
        [0.95, 0.05],
        [0.08, 0.92],
    ])

    filtered, predicted, log_lik = run_hamilton_filter(returns, mu, var, trans_matrix)
    assert filtered.shape == (t_len, 2)
    assert predicted.shape == (t_len, 2)
    assert np.all(filtered >= 0.0)
    assert np.all(np.isfinite(filtered))
    assert np.all(abs(np.sum(filtered, axis=1) - 1.0) < 1e-4)
    assert np.isfinite(log_lik)

    smoothed = run_kim_smoother(filtered, predicted, trans_matrix)
    assert smoothed.shape == (t_len, 2)
    assert np.all(smoothed >= 0.0)
    assert np.all(abs(np.sum(smoothed, axis=1) - 1.0) < 1e-4)


def test_fit_two_state_markov_switching_state_identification(synthetic_regime_switching_returns):
    """Verify EM parameter estimation recovers distinct volatility states and transitions."""
    returns = synthetic_regime_switching_returns
    mu, var, trans_mat, filtered, smoothed, log_lik = fit_two_state_markov_switching(returns, max_iter=40)

    # Identification: State 0 is Low Volatility, State 1 is High Volatility
    assert var[0] < var[1]
    assert trans_mat[0, 0] >= 0.50
    assert trans_mat[1, 1] >= 0.50
    assert np.isfinite(log_lik)

    # In Phase 2 (tail observations), smoothed crisis state probability should be high
    tail_crisis_prob = np.mean(smoothed[-30:, 1])
    assert tail_crisis_prob > 0.60


def test_evaluate_markov_regime_receipt_and_tamper(synthetic_regime_switching_returns):
    """Verify receipt creation, expected durations, and cryptographic HMAC tamper detection."""
    returns = synthetic_regime_switching_returns
    prices = pd.Series(100.0 * np.exp(np.cumsum(returns)))

    receipt = evaluate_markov_regime_profile("AAPL", prices, window=100)

    assert isinstance(receipt, MarkovRegimeReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == 99  # returns length
    assert receipt.current_regime in ("LOW_VOLATILITY_EXPANSION", "HIGH_VOLATILITY_CRISIS")
    assert 0.0 <= receipt.current_crisis_probability <= 1.0
    assert receipt.state_1_volatility_ann_pct > receipt.state_0_volatility_ann_pct
    assert receipt.expected_low_vol_duration_sessions > 1.0
    assert receipt.expected_high_vol_duration_sessions > 1.0
    assert receipt.verify_integrity() is True

    # Tamper with crisis probability
    tampered_dict = receipt.to_dict()
    tampered_dict["current_crisis_probability"] = 0.1234
    tampered_receipt = MarkovRegimeReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False


def test_markov_regime_security_and_edge_cases():
    """Verify input validation, insufficient data, and constant prices."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_markov_regime_profile("INVALID_TICKER$$$", pd.Series([10.0] * 50))

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_markov_regime_profile("AAPL", pd.Series([10.0, 11.0, 12.0]))


def test_cli_markov_regime_command_stdout_and_json(synthetic_regime_switching_returns, monkeypatch):
    """Verify CLI markov-regime command in stdout and JSON output modes."""
    prices = pd.Series(100.0 * np.exp(np.cumsum(synthetic_regime_switching_returns)))
    df_obs = pd.DataFrame({"close": prices})

    monkeypatch.setattr("src.data.db.get_observations", lambda ticker, db_path=None: df_obs)

    # 1. Standard stdout mode
    args_std = argparse.Namespace(ticker="AAPL", window=80, json=False)
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        markov_regime_command(args_std)
        output = fake_out.getvalue()
        assert "Hamilton (1989) Two-State Markov-Switching Regime Engine" in output
        assert "Security / Ticker:            AAPL" in output
        assert "Current Inferred Regime:" in output
        assert "Crisis State Probability:" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(ticker="AAPL", window=80, json=True)
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        markov_regime_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert payload["ticker"] == "AAPL"
        assert "current_regime" in payload
        assert "current_crisis_probability" in payload
        assert "p00_persistence" in payload
        assert "signature" in payload
