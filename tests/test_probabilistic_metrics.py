"""Unit and integration tests for probabilistic proper scoring rules (CRPS, Pinball Loss)."""

import argparse
import json
import math
import numpy as np
import pandas as pd
import pytest

from src.cli import crps_eval_command
from src.evaluation.probabilistic import (
    ProbabilisticEvaluationReceipt,
    calculate_continuous_ranked_probability_score,
    calculate_pinball_loss,
    evaluate_probabilistic_forecasts,
)


def test_crps_reduction_to_mae_when_zero_sigma():
    """Verify CRPS converges smoothly to deterministic MAE as sigma -> 0."""
    actual = np.array([100.0, 105.0, 95.0])
    pred = np.array([102.0, 103.0, 98.0])
    sigma_zero = np.array([0.0, 0.0, 0.0])

    crps_val = calculate_continuous_ranked_probability_score(actual, pred, sigma_zero)
    mae_val = float(np.mean(np.abs(actual - pred)))

    assert np.isclose(crps_val, mae_val, atol=1e-6)


def test_crps_analytical_positive_sigma():
    """Verify Gaussian CRPS matches analytical calculation for standard normal case."""
    # When y = 0, mu = 0, sigma = 1:
    # z = 0, Phi(0) = 0.5, phi(0) = 1 / sqrt(2*pi)
    # CRPS = 1 * [0 * 0 + 2 / sqrt(2*pi) - 1 / sqrt(pi)] = sqrt(2/pi) - 1/sqrt(pi) = (sqrt(2) - 1) / sqrt(pi)
    expected = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    crps_val = calculate_continuous_ranked_probability_score(0.0, 0.0, 1.0)
    assert np.isclose(crps_val, expected, atol=1e-5)


def test_crps_empty_and_nan_inputs():
    """Verify safe fallback on empty or non-finite inputs."""
    assert calculate_continuous_ranked_probability_score([], [], []) == 0.0
    assert calculate_continuous_ranked_probability_score([np.nan], [100.0], [1.0]) == 0.0


def test_pinball_loss_median_equals_half_mae():
    """Verify quantile check loss at q=0.50 exactly equals 0.5 * MAE."""
    actual = np.array([10.0, 20.0, 30.0])
    pred = np.array([12.0, 17.0, 35.0])

    pinball_50 = calculate_pinball_loss(actual, pred, quantile=0.50)
    mae = float(np.mean(np.abs(actual - pred)))

    assert np.isclose(pinball_50, 0.5 * mae)


def test_pinball_loss_asymmetry_and_bounds():
    """Verify asymmetric penalization of over- vs under-prediction."""
    # Under-prediction: y = 10, q_hat = 8 (err = +2)
    # At q=0.90, loss = 0.90 * 2 = 1.80
    assert np.isclose(calculate_pinball_loss(10.0, 8.0, quantile=0.90), 1.80)

    # Over-prediction: y = 10, q_hat = 12 (err = -2)
    # At q=0.90, loss = (0.90 - 1) * (-2) = 0.20
    assert np.isclose(calculate_pinball_loss(10.0, 12.0, quantile=0.90), 0.20)

    with pytest.raises(ValueError):
        calculate_pinball_loss(10.0, 10.0, quantile=1.5)


@pytest.fixture
def sample_predictions_df():
    """Generate realistic reconciled predictions dataframe."""
    np.random.seed(42)
    n = 20
    actual = 150.0 + np.random.normal(0, 5, n)
    pred = actual + np.random.normal(0, 2, n)
    lower = pred - 3.0
    upper = pred + 3.0

    return pd.DataFrame({
        "ticker": ["AAPL"] * n,
        "model_family": ["ensemble"] * (n // 2) + ["gradient_boosting"] * (n // 2),
        "horizon_sessions": [5] * n,
        "actual_target_price": actual,
        "predicted_target_price": pred,
        "prediction_interval_lower": lower,
        "prediction_interval_upper": upper,
        "status": ["EVALUATED"] * n,
    })


def test_evaluate_probabilistic_forecasts_comprehensive(sample_predictions_df):
    """Verify evaluate_probabilistic_forecasts generates complete valid receipt."""
    receipt = evaluate_probabilistic_forecasts(
        sample_predictions_df,
        horizon_sessions=5,
        model_family="ensemble",
        nominal_confidence=0.80,
    )

    assert isinstance(receipt, ProbabilisticEvaluationReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.model_family == "ensemble"
    assert receipt.horizon_sessions == 5
    assert receipt.sample_size == 10
    assert receipt.mean_crps > 0.0
    assert receipt.mean_mae > 0.0
    assert receipt.mean_pinball_loss_q10 > 0.0
    assert receipt.mean_pinball_loss_q90 > 0.0
    assert receipt.mean_winkler_score_80 > 0.0
    assert 0.0 <= receipt.empirical_coverage_80_pct <= 100.0

    d = receipt.to_dict()
    assert "mean_crps" in d
    assert d["sample_size"] == 10


def test_evaluate_probabilistic_forecasts_empty_and_insufficient():
    """Verify graceful handling when data is empty or insufficient."""
    empty_df = pd.DataFrame()
    rec_empty = evaluate_probabilistic_forecasts(empty_df)
    assert rec_empty.sample_size == 0
    assert rec_empty.mean_crps == 0.0

    short_df = pd.DataFrame({
        "actual_target_price": [100.0, 101.0],
        "predicted_target_price": [100.5, 101.5],
        "prediction_interval_lower": [98.0, 99.0],
        "prediction_interval_upper": [102.0, 103.0],
    })
    rec_short = evaluate_probabilistic_forecasts(short_df)
    assert "Insufficient" in rec_short.summary


def test_cli_crps_eval_command(sample_predictions_df, monkeypatch, capsys):
    """Verify CLI crps-eval execution in standard and JSON modes."""
    monkeypatch.setattr("src.data.db.get_predictions", lambda ticker=None: sample_predictions_df)

    # Standard human-readable mode
    args = argparse.Namespace(ticker="AAPL", horizon=5, model="ensemble", json=False)
    crps_eval_command(args)
    captured = capsys.readouterr()
    assert "Probabilistic Forecast Quality & CRPS Scorecard" in captured.out
    assert "Continuous Ranked Prob (CRPS)" in captured.out
    assert "Winkler Sharpness" in captured.out

    # JSON mode
    args_json = argparse.Namespace(ticker="AAPL", horizon=5, model="ensemble", json=True)
    crps_eval_command(args_json)
    captured_json = capsys.readouterr()
    parsed = json.loads(captured_json.out)
    assert parsed["ticker"] == "AAPL"
    assert "mean_crps" in parsed
    assert "empirical_coverage_80_pct" in parsed
