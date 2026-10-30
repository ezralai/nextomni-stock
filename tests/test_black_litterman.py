"""Unit and integration test suite for Black-Litterman (1992) Bayesian Portfolio Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.black_litterman import (
    AssetAllocation,
    BlackLittermanReceipt,
    blend_black_litterman_views,
    compute_black_litterman_optimal_weights,
    compute_implied_equilibrium_returns,
    evaluate_black_litterman_portfolio,
)
from src.cli import black_litterman_command


@pytest.fixture
def sample_universe_data():
    """Generate 100 sessions of synthetic daily returns across 3 assets."""
    np.random.seed(42)
    n = 100
    tickers = ["AAPL", "MSFT", "NVDA"]
    # 3-asset correlated returns
    cov = np.array([
        [0.04, 0.02, 0.015],
        [0.02, 0.05, 0.025],
        [0.015, 0.025, 0.08],
    ]) / 252.0
    mean = np.array([0.0005, 0.0006, 0.001])
    raw_rets = np.random.multivariate_normal(mean, cov, size=n)
    return pd.DataFrame(raw_rets, columns=tickers)


def test_compute_implied_equilibrium_returns() -> None:
    """Verify reverse-engineering of CAPM implied equilibrium prior returns: Pi = delta * Sigma * w."""
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    w = np.array([0.6, 0.4])
    delta = 2.5

    pi = compute_implied_equilibrium_returns(cov, w, risk_aversion_delta=delta)

    # Expected: 2.5 * ([0.04*0.6 + 0.01*0.4, 0.01*0.6 + 0.09*0.4]) = 2.5 * [0.028, 0.042] = [0.070, 0.105]
    expected_pi = delta * (cov @ w)
    assert np.allclose(pi, expected_pi)
    assert pi[0] == pytest.approx(0.070, rel=1e-3)
    assert pi[1] == pytest.approx(0.105, rel=1e-3)


def test_zero_views_convergence() -> None:
    """Verify that when no views are expressed, posterior returns equal prior returns (mu_BL == Pi)."""
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    w_mkt = np.array([0.5, 0.5])
    delta = 2.5
    pi = compute_implied_equilibrium_returns(cov, w_mkt, risk_aversion_delta=delta)

    mu_bl, sigma_bl = blend_black_litterman_views(prior_returns=pi, cov_matrix=cov, pick_matrix=None, view_returns=None)

    assert np.allclose(mu_bl, pi)
    assert np.allclose(sigma_bl, (1.0 + 0.05) * cov)

    # Optimal unconstrained weights should match baseline market weights
    w_opt = compute_black_litterman_optimal_weights(mu_bl, sigma_bl, risk_aversion_delta=delta, long_only=False)
    assert np.allclose(w_opt, w_mkt, atol=1e-3)


def test_bullish_view_shifts_weight() -> None:
    """Verify that expressing a bullish view on asset 0 increases its posterior return and allocation weight."""
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.04],
    ])
    w_mkt = np.array([0.5, 0.5])
    delta = 2.5
    pi = compute_implied_equilibrium_returns(cov, w_mkt, risk_aversion_delta=delta)

    # Express strong bullish view on asset 0: expected return 25% (well above equilibrium prior)
    P = np.array([[1.0, 0.0]])
    Q = np.array([0.25])

    mu_bl, sigma_bl = blend_black_litterman_views(
        prior_returns=pi,
        cov_matrix=cov,
        pick_matrix=P,
        view_returns=Q,
        tau=0.05,
    )

    # Asset 0 posterior return should be pulled upward
    assert mu_bl[0] > pi[0]

    # Constrained optimal weights
    w_opt = compute_black_litterman_optimal_weights(mu_bl, sigma_bl, risk_aversion_delta=delta, long_only=True)
    assert w_opt[0] > w_mkt[0]
    assert w_opt[1] < w_mkt[1]
    assert np.sum(w_opt) == pytest.approx(1.0, abs=1e-5)


def test_long_only_and_sum_to_one_constraints() -> None:
    """Verify that optimal weights strictly observe long-only and sum-to-one constraints."""
    mu = np.array([0.15, -0.05, 0.08, 0.22])
    sigma = np.diag([0.04, 0.09, 0.05, 0.12])

    w_opt = compute_black_litterman_optimal_weights(mu, sigma, risk_aversion_delta=2.5, long_only=True)

    assert len(w_opt) == 4
    assert np.all(w_opt >= -1e-6)  # Non-negative
    assert np.sum(w_opt) == pytest.approx(1.0, abs=1e-5)
    # The negative return asset should have near-zero weight
    assert w_opt[1] == pytest.approx(0.0, abs=1e-3)


def test_evaluate_black_litterman_portfolio_integration(sample_universe_data: pd.DataFrame) -> None:
    """Verify end-to-end Black-Litterman evaluation workflow with synthetic returns and RMT covariance denoising."""
    universe = ["AAPL", "MSFT", "NVDA"]

    with patch("src.analytics.black_litterman.build_aligned_returns_matrix", return_value=sample_universe_data):
        receipt = evaluate_black_litterman_portfolio(
            universe=universe,
            views_dict={"NVDA": 0.30},
            tau=0.05,
            risk_aversion_delta=2.5,
            lookback_window=100,
            denoise_cov=True,
        )

        assert isinstance(receipt, BlackLittermanReceipt)
        assert receipt.universe == universe
        assert receipt.sample_sessions == 100
        assert receipt.views_count == 1
        assert receipt.expected_portfolio_return_ann_pct > 0.0
        assert receipt.expected_portfolio_vol_ann_pct > 0.0
        assert len(receipt.allocations) == 3
        assert receipt.verify_integrity() is True

        # Check that NVDA received positive shift due to high view
        nvda_alloc = next(a for a in receipt.allocations if a["ticker"] == "NVDA")
        assert nvda_alloc["posterior_return_ann_pct"] > nvda_alloc["prior_return_ann_pct"]
        assert nvda_alloc["weight_shift"] > 0.0


def test_receipt_hmac_tamper_detection(sample_universe_data: pd.DataFrame) -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    universe = ["AAPL", "MSFT", "NVDA"]

    with patch("src.analytics.black_litterman.build_aligned_returns_matrix", return_value=sample_universe_data):
        receipt = evaluate_black_litterman_portfolio(universe=universe, views_dict=None)

        assert receipt.verify_integrity() is True

        # Tamper with expected return
        receipt.expected_portfolio_return_ann_pct += 5.0
        assert receipt.verify_integrity() is False

        # Reset and tamper with volatility
        receipt.expected_portfolio_return_ann_pct -= 5.0
        receipt.expected_portfolio_vol_ann_pct += 2.0
        assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify input validation and defense-in-depth against malicious parameters."""
    # Universe < 2 assets
    with pytest.raises(ValueError, match="at least 2 distinct assets"):
        evaluate_black_litterman_portfolio(universe=["AAPL"])

    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_black_litterman_portfolio(universe=["AAPL", "INVALID!TICKER"])

    # Non-finite parameter
    with pytest.raises(ValueError, match="tau must be finite"):
        evaluate_black_litterman_portfolio(universe=["AAPL", "MSFT"], tau=float("nan"))


def test_cli_black_litterman_command_stdout(sample_universe_data: pd.DataFrame) -> None:
    """Verify CLI command standard formatted output."""
    args = argparse.Namespace(
        universe="AAPL,MSFT,NVDA",
        views="NVDA:0.25,AAPL:0.12",
        tau=0.05,
        risk_aversion=2.5,
        lookback=100,
        no_denoise=False,
        json=False,
    )

    with patch("src.analytics.black_litterman.build_aligned_returns_matrix", return_value=sample_universe_data):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            black_litterman_command(args)
            output = mock_stdout.getvalue()

            assert "NextOmni Stock: Black-Litterman (1992) Bayesian Portfolio Engine" in output
            assert "AAPL" in output
            assert "MSFT" in output
            assert "NVDA" in output
            assert "Bayesian Return & Constrained Allocation Decomposition" in output
            assert "Audit Digest (HMAC)" in output


def test_cli_black_litterman_command_json(sample_universe_data: pd.DataFrame) -> None:
    """Verify CLI command JSON output serialization."""
    args = argparse.Namespace(
        universe="AAPL,MSFT,NVDA",
        views='{"NVDA": 0.25}',
        tau=0.05,
        risk_aversion=2.5,
        lookback=100,
        no_denoise=False,
        json=True,
    )

    with patch("src.analytics.black_litterman.build_aligned_returns_matrix", return_value=sample_universe_data):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            black_litterman_command(args)
            output = mock_stdout.getvalue()

            data = json.loads(output)
            assert data["universe"] == ["AAPL", "MSFT", "NVDA"]
            assert data["views_count"] == 1
            assert "signature" in data
            assert len(data["allocations"]) == 3
