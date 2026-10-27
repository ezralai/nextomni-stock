"""Unit and integration tests for conformal prediction intervals and Winkler sharpness calibration."""

import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.calibration import (
    CalibrationReceipt,
    calculate_winkler_score,
    compute_conformal_prediction_interval,
    compute_conformal_quantile,
    evaluate_interval_calibration,
)


def test_winkler_score_in_bounds():
    """Verify Winkler score equals interval width when actual price falls within bounds."""
    lower = 95.0
    upper = 105.0
    actual = 100.0
    alpha = 0.20  # 80% CI

    score = calculate_winkler_score(lower, upper, actual, alpha=alpha)
    assert np.isclose(score, 10.0)


def test_winkler_score_breaches_penalty():
    """Verify severe penalty is applied when actual price breaches interval bounds."""
    lower = 95.0
    upper = 105.0
    alpha = 0.20
    penalty_multiplier = 2.0 / alpha  # 10.0

    # Lower breach by 2 units
    score_lower = calculate_winkler_score(lower, upper, 93.0, alpha=alpha)
    expected_lower = 10.0 + penalty_multiplier * 2.0
    assert np.isclose(score_lower, expected_lower)

    # Upper breach by 3 units
    score_upper = calculate_winkler_score(lower, upper, 108.0, alpha=alpha)
    expected_upper = 10.0 + penalty_multiplier * 3.0
    assert np.isclose(score_upper, expected_upper)

    # Vectorized execution
    v_scores = calculate_winkler_score(
        np.array([lower, lower, lower]),
        np.array([upper, upper, upper]),
        np.array([100.0, 93.0, 108.0]),
        alpha=alpha,
    )
    assert len(v_scores) == 3
    assert np.isclose(v_scores[0], 10.0)
    assert np.isclose(v_scores[1], expected_lower)
    assert np.isclose(v_scores[2], expected_upper)


def test_conformal_quantile_coverage_guarantee():
    """Verify non-parametric conformal quantile achieves >= 80% empirical coverage."""
    np.random.seed(42)
    # Calibration fold
    train_residuals = np.random.normal(0, 0.02, 100)
    # Test fold
    test_residuals = np.random.normal(0, 0.02, 200)

    q_hat = compute_conformal_quantile(train_residuals, confidence_level=0.80)
    assert q_hat > 0.0

    empirical_coverage = np.mean(np.abs(test_residuals) <= q_hat)
    # Nominal 80% coverage should be approximately satisfied (e.g. >= 75%)
    assert empirical_coverage >= 0.75


def test_compute_conformal_prediction_interval():
    """Verify conformal price interval calculation adheres to physical bounds."""
    origin_price = 150.0
    expected_return = 0.02
    residuals = np.random.normal(0, 0.015, 50)

    lower_p, upper_p, q_hat = compute_conformal_prediction_interval(
        origin_price=origin_price,
        expected_return=expected_return,
        train_residuals=residuals,
        confidence_level=0.80,
    )

    assert lower_p > 0.0
    assert upper_p > lower_p
    assert lower_p < origin_price * (1.0 + expected_return) < upper_p
    assert q_hat > 0.0


def test_evaluate_interval_calibration_insufficient_data():
    """Verify graceful handling when fewer than 5 evaluated predictions exist."""
    df = pd.DataFrame({
        "lower_bound": [90.0, 91.0],
        "upper_bound": [110.0, 111.0],
        "actual_price": [100.0, 102.0],
    })
    receipt = evaluate_interval_calibration(df, confidence_level=0.80)
    assert isinstance(receipt, CalibrationReceipt)
    assert receipt.sample_size == 2
    assert receipt.calibration_status == "INSUFFICIENT_DATA"
    assert not receipt.is_well_calibrated


def test_evaluate_interval_calibration_well_calibrated():
    """Verify accurate calculation of empirical coverage, Winkler score, and WELL_CALIBRATED status."""
    n = 20
    # 16 within interval (80%), 2 under-breaches (10%), 2 over-breaches (10%)
    lower_bounds = [95.0] * n
    upper_bounds = [105.0] * n
    actuals = [100.0] * 16 + [90.0, 92.0] + [108.0, 110.0]

    df = pd.DataFrame({
        "ticker": ["JPM"] * n,
        "horizon_type": ["5d"] * n,
        "lower_bound": lower_bounds,
        "upper_bound": upper_bounds,
        "actual_price": actuals,
    })

    receipt = evaluate_interval_calibration(df, confidence_level=0.80)
    assert receipt.sample_size == 20
    assert np.isclose(receipt.empirical_coverage_pct, 80.0)
    assert np.isclose(receipt.coverage_error_pct, 0.0)
    assert receipt.is_well_calibrated
    assert receipt.calibration_status == "WELL_CALIBRATED"
    assert receipt.mean_interval_width == 10.0
    assert receipt.mean_winkler_score > 10.0  # width + breach penalties


def test_cli_calibration_report_command(capsys):
    """Verify CLI nextomni calibration-report executes cleanly with JSON output."""
    from src.cli import main
    import sys

    dummy_df = pd.DataFrame({
        "ticker": ["JPM"] * 10,
        "horizon_type": ["5d"] * 10,
        "lower_bound": [95.0] * 10,
        "upper_bound": [105.0] * 10,
        "actual_price": [100.0] * 8 + [90.0, 110.0],
    })

    with patch("src.data.db.get_predictions", return_value=dummy_df), \
         patch.object(sys, "argv", ["nextomni", "calibration-report", "--ticker", "JPM", "--horizon", "5", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["sample_size"] == 10
    assert data["calibration_status"] == "WELL_CALIBRATED"
    assert "mean_winkler_score" in data
