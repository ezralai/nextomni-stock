"""Unit and integration test suite for Bailey & López de Prado Deflated Sharpe Ratio Engine."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli import deflated_sharpe_command
from src.evaluation.deflated_sharpe import (
    DeflatedSharpeReceipt,
    compute_deflated_sharpe_ratio,
    compute_expected_maximum_sharpe,
    compute_min_track_record_length,
    compute_probabilistic_sharpe_ratio,
    compute_sharpe_standard_error,
    evaluate_deflated_sharpe_profile,
)


@pytest.fixture
def synthetic_positive_returns() -> pd.Series:
    """Generate 252 sessions of positive returns with mild negative skew and fat tails."""
    np.random.seed(42)
    n = 252
    # Gaussian mixture to induce negative skew and excess kurtosis
    norm_part = np.random.normal(0.001, 0.01, size=n)
    shock_indices = np.random.choice(n, size=5, replace=False)
    norm_part[shock_indices] -= 0.035  # Tail crash shocks
    return pd.Series(norm_part)


def test_sharpe_standard_error_non_normality_penalties():
    """Verify that negative skewness and excess kurtosis widen the Sharpe ratio standard error."""
    sr_daily = 0.08
    n_sessions = 252

    # Baseline Gaussian (skew = 0, kurt = 3)
    se_gaussian = compute_sharpe_standard_error(
        sr_daily=sr_daily,
        n_sessions=n_sessions,
        skewness=0.0,
        kurtosis=3.0,
    )

    # Fat-tailed + crash-skewed (skew = -1.5, kurt = 7.0)
    se_fat_tailed = compute_sharpe_standard_error(
        sr_daily=sr_daily,
        n_sessions=n_sessions,
        skewness=-1.5,
        kurtosis=7.0,
    )

    assert se_gaussian > 0.0
    assert se_fat_tailed > se_gaussian  # Severe penalty on uncertainty


def test_probabilistic_sharpe_ratio_monotonicity_and_bounds():
    """Verify PSR properties across varying observed Sharpe ratios and non-normality."""
    n_sessions = 150
    psr_low, _ = compute_probabilistic_sharpe_ratio(
        observed_sr_daily=0.01,
        benchmark_sr_daily=0.0,
        n_sessions=n_sessions,
        skewness=0.0,
        kurtosis=3.0,
    )
    psr_high, _ = compute_probabilistic_sharpe_ratio(
        observed_sr_daily=0.10,
        benchmark_sr_daily=0.0,
        n_sessions=n_sessions,
        skewness=0.0,
        kurtosis=3.0,
    )

    assert 0.0 <= psr_low <= 1.0
    assert 0.0 <= psr_high <= 1.0
    assert psr_high > psr_low  # Monotonic in Sharpe


def test_expected_maximum_sharpe_multiple_testing_penalty():
    """Verify that expected maximum Sharpe ratio increases strictly with number of trials N."""
    v = 0.001  # Variance across trials

    exp_max_1 = compute_expected_maximum_sharpe(num_trials=1, trial_variance=v)
    exp_max_5 = compute_expected_maximum_sharpe(num_trials=5, trial_variance=v)
    exp_max_50 = compute_expected_maximum_sharpe(num_trials=50, trial_variance=v)
    exp_max_500 = compute_expected_maximum_sharpe(num_trials=500, trial_variance=v)

    assert exp_max_1 == 0.0
    assert 0.0 < exp_max_5 < exp_max_50 < exp_max_500


def test_deflated_sharpe_ratio_strictly_deflates_psr():
    """Verify that DSR is strictly less than or equal to PSR due to data snooping penalty."""
    sr_daily = 0.06
    n_sessions = 200
    skew = -0.5
    kurt = 4.2
    v = 0.0008

    psr, _ = compute_probabilistic_sharpe_ratio(
        observed_sr_daily=sr_daily,
        benchmark_sr_daily=0.0,
        n_sessions=n_sessions,
        skewness=skew,
        kurtosis=kurt,
    )

    dsr_10, exp_max_10, _ = compute_deflated_sharpe_ratio(
        observed_sr_daily=sr_daily,
        num_trials=10,
        trial_variance=v,
        n_sessions=n_sessions,
        skewness=skew,
        kurtosis=kurt,
    )

    dsr_100, exp_max_100, _ = compute_deflated_sharpe_ratio(
        observed_sr_daily=sr_daily,
        num_trials=100,
        trial_variance=v,
        n_sessions=n_sessions,
        skewness=skew,
        kurtosis=kurt,
    )

    assert exp_max_100 > exp_max_10
    assert dsr_10 <= psr
    assert dsr_100 < dsr_10  # Harder deflation with 100 trials


def test_min_track_record_length_properties():
    """Verify MinTRL finite sessions calculation and infinite fallback for underperformance."""
    # When observed SR <= benchmark SR, track record length is infinite
    inf_sessions = compute_min_track_record_length(
        observed_sr_daily=0.0,
        benchmark_sr_daily=0.02,
        skewness=0.0,
        kurtosis=3.0,
    )
    assert math.isinf(inf_sessions)

    # When observed SR > benchmark SR
    valid_sessions = compute_min_track_record_length(
        observed_sr_daily=0.08,
        benchmark_sr_daily=0.0,
        skewness=-0.5,
        kurtosis=4.0,
    )
    assert 10.0 <= valid_sessions <= 5000.0


def test_deflated_sharpe_receipt_and_cryptographic_tamper(synthetic_positive_returns):
    """Verify DeflatedSharpeReceipt creation and HMAC-SHA256 tamper resilience."""
    returns = synthetic_positive_returns
    receipt = evaluate_deflated_sharpe_profile(
        ticker="AAPL",
        returns=returns,
        num_trials=20,
        benchmark_sharpe_ann=0.0,
    )

    assert isinstance(receipt, DeflatedSharpeReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.num_trials == 20
    assert receipt.sample_sessions == 252
    assert receipt.overfitting_status in ("ROBUST_GENUINE_ALPHA", "SUSPECT_MULTIPLE_TESTING", "SEVERELY_OVERFIT")
    assert 0.0 <= receipt.probabilistic_sharpe_ratio <= 1.0
    assert 0.0 <= receipt.deflated_sharpe_ratio <= 1.0
    assert receipt.verify_integrity() is True

    # Tamper with DSR value
    tampered_dict = receipt.to_dict()
    tampered_dict["deflated_sharpe_ratio"] = 0.9999
    tampered_receipt = DeflatedSharpeReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with status
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["overfitting_status"] = "ROBUST_GENUINE_ALPHA" if receipt.overfitting_status != "ROBUST_GENUINE_ALPHA" else "SEVERELY_OVERFIT"
    tampered_receipt2 = DeflatedSharpeReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_deflated_sharpe_security_and_edge_cases():
    """Verify ticker allowlist validation and short return history exceptions."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_deflated_sharpe_profile("INVALID$$$", pd.Series([0.01] * 30))

    # Insufficient observations (< 10)
    with pytest.raises(ValueError, match="Insufficient return history"):
        evaluate_deflated_sharpe_profile("AAPL", pd.Series([0.01, 0.02, 0.03]))


def test_cli_deflated_sharpe_command_stdout_and_json(synthetic_positive_returns, monkeypatch):
    """Verify CLI deflated-sharpe command execution across stdout and JSON output modes."""
    df_obs = pd.DataFrame({"close": 100.0 * np.exp(np.cumsum(synthetic_positive_returns))})
    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda *args, **kwargs: df_obs,
    )

    # 1. Stdout execution
    args_stdout = argparse.Namespace(
        ticker="AAPL",
        trials=15,
        benchmark_sr=0.0,
        risk_free_rate=0.04,
        json=False,
    )
    deflated_sharpe_command(args_stdout)

    # 2. JSON execution
    args_json = argparse.Namespace(
        ticker="AAPL",
        trials=15,
        benchmark_sr=0.0,
        risk_free_rate=0.04,
        json=True,
    )
    deflated_sharpe_command(args_json)
