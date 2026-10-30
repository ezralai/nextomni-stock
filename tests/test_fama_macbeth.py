"""Unit and integration test suite for Fama & MacBeth (1973) Two-Pass Asset Pricing Engine."""

import argparse
import io
import json
import math
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.fama_macbeth import (
    FactorPremiumStat,
    FamaMacBethReceipt,
    compute_fama_macbeth_statistics,
    estimate_time_series_betas,
    evaluate_fama_macbeth_profile,
    run_cross_sectional_pass,
)
from src.cli import fama_macbeth_command


@pytest.fixture
def synthetic_factor_market():
    """Generate 100 sessions across 6 assets and 2 factors with known factor premia."""
    np.random.seed(42)
    T = 100
    N = 6
    K = 2

    # Factor returns (MKT, SMB)
    f1 = np.random.normal(0.0004, 0.010, T)
    f2 = np.random.normal(0.0002, 0.008, T)
    F = np.column_stack([f1, f2])

    # Known true betas (N x K)
    true_betas = np.array([
        [1.2, 0.5],
        [0.8, -0.4],
        [1.5, 0.8],
        [0.6, 0.2],
        [1.0, -0.2],
        [1.1, 0.4],
    ])

    # True risk premia (lambda_1, lambda_2)
    true_lambdas = np.array([0.0005, 0.0003])
    lambda_zero = 0.0001

    # Cross-sectional expected returns: E[R_i] = lambda_0 + beta_i1 * lambda_1 + beta_i2 * lambda_2
    Y = np.zeros((T, N))
    for t in range(T):
        idio_noise = np.random.normal(0.0, 0.005, N)
        Y[t, :] = lambda_zero + (true_betas @ F[t, :]) + idio_noise

    return Y, F, true_betas, true_lambdas


def test_estimate_time_series_betas_recovery(
    synthetic_factor_market: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify Pass 1: Multivariate time-series regression accurately recovers true factor betas."""
    Y, F, true_betas, _ = synthetic_factor_market

    betas_hat, alphas_hat, residuals = estimate_time_series_betas(Y, F)

    assert betas_hat.shape == true_betas.shape
    # Check that estimated betas are close to true betas
    np.testing.assert_allclose(betas_hat, true_betas, atol=0.15)
    assert residuals.shape == Y.shape


def test_run_cross_sectional_pass_dimensions_and_r2(
    synthetic_factor_market: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """Verify Pass 2: Cross-sectional regression produces valid time-series of premia and R^2."""
    Y, F, true_betas, _ = synthetic_factor_market
    betas_hat, _, _ = estimate_time_series_betas(Y, F)

    l0, l_premia, r2_series = run_cross_sectional_pass(Y, betas_hat)

    assert len(l0) == len(Y)
    assert l_premia.shape == (len(Y), 2)
    assert len(r2_series) == len(Y)
    assert np.all(r2_series >= 0.0)
    assert np.all(r2_series <= 1.0)


def test_compute_fama_macbeth_statistics_shanken_adjustment() -> None:
    """Verify Pass 3: Shanken (1992) correction strictly widens standard errors (t_shanken <= t_fm)."""
    np.random.seed(123)
    T = 80
    K = 2
    l_premia = np.column_stack([
        np.random.normal(0.0004, 0.002, T),
        np.random.normal(0.0002, 0.0015, T),
    ])
    F = np.column_stack([
        np.random.normal(0.0005, 0.01, T),
        np.random.normal(0.0001, 0.008, T),
    ])

    stats = compute_fama_macbeth_statistics(["MKT", "SMB"], l_premia, F)

    assert len(stats) == 2
    for s in stats:
        assert isinstance(s, FactorPremiumStat)
        assert s.shanken_correction_factor >= 1.0
        # Shanken corrected t-stat magnitude must be less than or equal to classic FM t-stat
        assert abs(s.t_stat_shanken) <= abs(s.t_stat_fm) + 1e-9
        assert 0.0 <= s.p_value_shanken <= 1.0


def test_evaluate_fama_macbeth_profile_integration() -> None:
    """Verify end-to-end Fama-MacBeth workflow with mock returns matrix."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    np.random.seed(999)
    T = 60
    # Generate mock aligned return matrix for all tickers including SPY
    cols = universe + ["SPY"]
    rets = np.random.normal(0.0005, 0.015, (T, len(cols)))
    df_mock = pd.DataFrame(rets, columns=cols)

    with patch("src.analytics.fama_macbeth.build_aligned_returns_matrix", return_value=df_mock):
        receipt = evaluate_fama_macbeth_profile(
            universe=universe,
            benchmark_ticker="SPY",
            lookback_window=60,
        )

        assert isinstance(receipt, FamaMacBethReceipt)
        assert receipt.universe == universe
        assert receipt.sample_sessions == T
        assert len(receipt.factors_tested) == 4
        assert len(receipt.factor_premia) == 4
        assert 0.0 <= receipt.cross_sectional_r_squared <= 1.0
        assert receipt.pricing_error_rmse_ann_pct >= 0.0
        assert isinstance(receipt.is_model_rejected_5pct, bool)
        assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    cols = universe + ["SPY"]
    df_mock = pd.DataFrame(np.random.normal(0.0003, 0.012, (50, len(cols))), columns=cols)

    with patch("src.analytics.fama_macbeth.build_aligned_returns_matrix", return_value=df_mock):
        receipt = evaluate_fama_macbeth_profile(universe=universe, benchmark_ticker="SPY")

        assert receipt.verify_integrity() is True

        # Tamper with cross-sectional R^2
        receipt.cross_sectional_r_squared += 0.05
        assert receipt.verify_integrity() is False

        # Reset and tamper with pricing RMSE
        receipt.cross_sectional_r_squared -= 0.05
        receipt.pricing_error_rmse_ann_pct += 1.0
        assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify defense-in-depth against short universes, non-finite inputs, and invalid tickers."""
    # Universe < 4 assets
    with pytest.raises(ValueError, match="at least 4 distinct assets"):
        evaluate_fama_macbeth_profile(universe=["AAPL", "MSFT", "NVDA"])

    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_fama_macbeth_profile(universe=["AAPL", "MSFT", "NVDA", "INVALID!TICKER"])

    # Non-finite parameter
    with pytest.raises(ValueError, match="lookback_window must be finite"):
        evaluate_fama_macbeth_profile(
            universe=["AAPL", "MSFT", "NVDA", "GOOGL"],
            lookback_window=float("nan"),
        )


def test_cli_fama_macbeth_command_stdout() -> None:
    """Verify CLI command formatted table stdout output."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    cols = universe + ["SPY"]
    df_mock = pd.DataFrame(np.random.normal(0.0004, 0.014, (60, len(cols))), columns=cols)

    args = argparse.Namespace(
        universe="AAPL,MSFT,NVDA,GOOGL",
        benchmark="SPY",
        lookback=60,
        json=False,
    )

    with patch("src.analytics.fama_macbeth.build_aligned_returns_matrix", return_value=df_mock):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fama_macbeth_command(args)
            output = mock_stdout.getvalue()

            assert "NextOmni Stock: Fama & MacBeth (1973) Two-Pass Asset Pricing Engine" in output
            assert "Cross-Sectional Factor Risk Premia" in output
            assert "MKT" in output
            assert "SMB" in output
            assert "Audit Digest (HMAC)" in output


def test_cli_fama_macbeth_command_json() -> None:
    """Verify CLI command JSON output serialization."""
    universe = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    cols = universe + ["SPY"]
    df_mock = pd.DataFrame(np.random.normal(0.0004, 0.014, (60, len(cols))), columns=cols)

    args = argparse.Namespace(
        universe="AAPL,MSFT,NVDA,GOOGL,AMZN,META",
        benchmark="SPY",
        lookback=60,
        json=True,
    )

    with patch("src.analytics.fama_macbeth.build_aligned_returns_matrix", return_value=df_mock):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            fama_macbeth_command(args)
            output = mock_stdout.getvalue()

            data = json.loads(output)
            assert data["universe"] == universe
            assert "cross_sectional_r_squared" in data
            assert len(data["factor_premia"]) == 4
            assert "signature" in data
