"""Unit tests for extended quantitative risk and performance metrics in Iteration 6."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    calculate_all_metrics,
    calculate_information_ratio,
    calculate_max_error,
)


def test_calculate_max_error_basic():
    """Verify max_error computes the Chebyshev infinity norm ||y - y_hat||_inf."""
    y_true = [100.0, 105.0, 110.0]
    y_pred = [102.0, 101.0, 115.0]  # Absolute errors: [2.0, 4.0, 5.0]
    max_err = calculate_max_error(y_true, y_pred)
    assert max_err == 5.0


def test_calculate_max_error_empty():
    """Verify max_error returns 0.0 on empty sequence."""
    assert calculate_max_error([], []) == 0.0


def test_calculate_information_ratio_positive_alpha():
    """Verify Information Ratio is positive when model directional trades outperform benchmark."""
    # Price steadily rises; model correctly predicts rise every step
    origins = [100.0, 102.0, 104.0, 106.0, 108.0]
    targets = [102.0, 104.0, 106.0, 108.0, 110.0]
    preds = [103.0, 105.0, 107.0, 109.0, 112.0]

    ir = calculate_information_ratio(origins, targets, preds)
    # Active return is sign(pred - orig)*r_bench - r_bench = 1*r_b - r_b = 0 in pure long match
    # Let's test with market where benchmark goes down, but model goes short (sign < 0)
    # causing active return = (-1)*(-0.05) - (-0.05) = +0.05 - (-0.05) = +0.10!
    origins_down = [100.0, 95.0, 90.0, 85.0]
    targets_down = [95.0, 90.0, 85.0, 80.0]
    preds_down = [90.0, 85.0, 80.0, 75.0]  # Model accurately anticipates downturns

    ir_short = calculate_information_ratio(origins_down, targets_down, preds_down)
    assert ir_short > 0.0


def test_calculate_information_ratio_zero_variance():
    """Verify zero division is guarded when tracking error is zero."""
    origins = [100.0, 100.0, 100.0]
    targets = [100.0, 100.0, 100.0]
    preds = [100.0, 100.0, 100.0]
    ir = calculate_information_ratio(origins, targets, preds)
    assert ir == 0.0


def test_calculate_all_metrics_includes_extended():
    """Verify calculate_all_metrics incorporates max_error and information_ratio."""
    origins = [100.0, 102.0, 104.0, 106.0]
    targets = [101.0, 103.0, 102.0, 107.0]
    preds = [102.0, 104.0, 101.0, 108.0]

    metrics = calculate_all_metrics(origins, targets, preds)
    assert "max_error" in metrics
    assert "information_ratio" in metrics
    assert isinstance(metrics["max_error"], float)
    assert isinstance(metrics["information_ratio"], float)
