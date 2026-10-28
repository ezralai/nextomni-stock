"""Unit and integration tests for Bayesian Online Change-Point Detection (BOCPD) engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.changepoints import (
    ChangePointReceipt,
    compute_bocpd_series,
    detect_bayesian_changepoints,
)
from src.cli import changepoint_detect_command


@pytest.fixture
def synthetic_step_df() -> pd.DataFrame:
    """Generate price history with an abrupt structural regime break at session 60."""
    np.random.seed(42)
    n1 = 60
    n2 = 40
    n = n1 + n2

    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    # Regime 1: Mean 0, std 0.005 (calm low-vol regime)
    r1 = np.random.normal(0.0, 0.005, n1)
    # Regime 2: Mean -0.04, std 0.035 (abrupt crisis/crash regime)
    r2 = np.random.normal(-0.04, 0.035, n2)

    returns = np.concatenate([r1, r2])
    prices = 100.0 * np.exp(np.cumsum(returns))

    return pd.DataFrame(
        {
            "session_date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.uniform(1e6, 2e6, n),
        }
    )


def test_bocpd_insufficient_or_empty_data() -> None:
    """Verify handling of empty or insufficient observations."""
    empty_res = compute_bocpd_series([], hazard_lambda=100)
    assert len(empty_res[0]) == 0
    assert len(empty_res[1]) == 0
    assert empty_res[2] == []

    tiny_res = compute_bocpd_series([0.01, -0.02, 0.005], hazard_lambda=100)
    assert len(tiny_res[0]) == 3
    assert len(tiny_res[2]) == 0

    receipt = detect_bayesian_changepoints(pd.DataFrame(), ticker="AAPL")
    assert receipt.regime_stability_status == "INSUFFICIENT_DATA"
    assert receipt.sample_sessions == 0
    assert receipt.current_run_length == 0


def test_bocpd_abrupt_step_shift_detection(synthetic_step_df: pd.DataFrame) -> None:
    """Verify BOCPD detects abrupt structural regime shift at session ~60."""
    prices = synthetic_step_df["adj_close"].values
    returns = np.diff(np.log(prices))

    mode_r, cp_probs, break_indices = compute_bocpd_series(
        returns, hazard_lambda=100, max_run_length=200
    )

    assert len(mode_r) == len(returns)
    assert len(cp_probs) == len(returns)
    # A break should be detected near session 60 (±5 sessions)
    assert len(break_indices) > 0
    detected_near_60 = any(55 <= idx <= 65 for idx in break_indices)
    assert detected_near_60, f"Expected break near index 60, got: {break_indices}"

    # Verify probability mass increases around break point
    max_prob_around_60 = float(np.max(cp_probs[58:65]))
    assert max_prob_around_60 > 0.40, f"Expected high break probability near 60, got: {max_prob_around_60}"


def test_bocpd_bounded_memory() -> None:
    """Verify run-length vector truncation bounds memory to max_run_length."""
    np.random.seed(123)
    long_returns = np.random.normal(0.0005, 0.01, 300)
    max_rl = 50

    mode_r, cp_probs, _ = compute_bocpd_series(long_returns, max_run_length=max_rl)

    assert len(mode_r) == 300
    assert np.all(mode_r <= max_rl)
    assert np.all(np.isfinite(cp_probs))


def test_detect_bayesian_changepoints_receipt(synthetic_step_df: pd.DataFrame) -> None:
    """Verify detect_bayesian_changepoints returns complete ChangePointReceipt."""
    receipt = detect_bayesian_changepoints(synthetic_step_df, ticker="TEST_TICKER", hazard_lambda=100)

    assert isinstance(receipt, ChangePointReceipt)
    assert receipt.ticker == "TEST_TICKER"
    assert receipt.sample_sessions == len(synthetic_step_df) - 1
    assert receipt.hazard_lambda == 100
    assert receipt.current_run_length >= 0
    assert 0.0 <= receipt.current_change_point_prob <= 1.0
    assert receipt.regime_stability_status in [
        "STRUCTURAL_BREAK_DETECTED",
        "REGIME_UNCERTAINTY",
        "STABLE_PERSISTENCE",
    ]
    assert len(receipt.detected_break_dates) > 0

    d = receipt.to_dict()
    assert d["ticker"] == "TEST_TICKER"
    assert "summary" in d
    assert "current_change_point_prob" in d


def test_changepoint_detect_cli_json_and_stdout(synthetic_step_df: pd.DataFrame) -> None:
    """Verify CLI changepoint-detect command outputs valid JSON and formatted audit text."""
    args_json = argparse.Namespace(ticker="JPM", hazard=100, json=True)

    with patch("src.data.db.get_observations", return_value=synthetic_step_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        changepoint_detect_command(args_json)
        output = mock_out.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["ticker"] == "JPM"
        assert "current_run_length" in parsed
        assert "regime_stability_status" in parsed

    args_txt = argparse.Namespace(ticker="JPM", hazard=100, json=False)
    with patch("src.data.db.get_observations", return_value=synthetic_step_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        changepoint_detect_command(args_txt)
        output = mock_out.getvalue()
        assert "Bayesian Online Change-Point" in output
        assert "Ticker Symbol:" in output
        assert "Regime Stability Status:" in output


def test_changepoint_detect_cli_empty_data() -> None:
    """Verify CLI handles missing or empty observations gracefully."""
    args = argparse.Namespace(ticker="EMPTY", hazard=100, json=False)
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        changepoint_detect_command(args)
        output = mock_out.getvalue()
        assert "No observations found for EMPTY" in output
