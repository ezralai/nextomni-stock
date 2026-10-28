"""Unit and integration tests for Ledoit-Wolf shrinkage covariance and portfolio optimizer."""

import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.portfolio import (
    LedoitWolfReceipt,
    PortfolioReceipt,
    calculate_portfolio_risk_contributions,
    compute_ledoit_wolf_covariance,
    optimize_alpha_portfolio,
)
from src.analytics.screener import ScreenerRecord


@pytest.fixture
def sample_returns_df():
    """Generate 60 sessions of synthetic daily returns across 4 assets."""
    np.random.seed(42)
    n_sessions = 60
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN"]
    # Correlated returns
    factors = np.random.normal(0, 0.015, (n_sessions, 2))
    loadings = np.array([
        [0.8, 0.2],
        [0.7, 0.3],
        [0.5, 0.5],
        [0.3, 0.7],
    ])
    idiosyncratic = np.random.normal(0, 0.008, (n_sessions, 4))
    rets = factors @ loadings.T + idiosyncratic
    return pd.DataFrame(rets, columns=tickers)


@pytest.fixture
def sample_screener_records():
    """Generate mock ScreenerRecord objects for 4 assets."""
    return [
        ScreenerRecord(
            rank=1,
            ticker="AAPL",
            exchange="XNYS",
            company_name="Apple Inc",
            currency="USD",
            latest_price=180.0,
            expected_return_pct=3.5,
            directional_win_probability=0.68,
            conviction_label="STRONG_BULLISH",
            cornish_fisher_var_pct=4.2,
            predicted_sharpe=1.8,
            horizon_sessions=5,
            model_family="ensemble",
        ),
        ScreenerRecord(
            rank=2,
            ticker="MSFT",
            exchange="XNYS",
            company_name="Microsoft Corp",
            currency="USD",
            latest_price=410.0,
            expected_return_pct=2.8,
            directional_win_probability=0.62,
            conviction_label="LEAN_BULLISH",
            cornish_fisher_var_pct=3.8,
            predicted_sharpe=1.5,
            horizon_sessions=5,
            model_family="ensemble",
        ),
        ScreenerRecord(
            rank=3,
            ticker="GOOGL",
            exchange="XNYS",
            company_name="Alphabet Inc",
            currency="USD",
            latest_price=175.0,
            expected_return_pct=1.9,
            directional_win_probability=0.55,
            conviction_label="LEAN_BULLISH",
            cornish_fisher_var_pct=4.5,
            predicted_sharpe=0.9,
            horizon_sessions=5,
            model_family="ensemble",
        ),
        ScreenerRecord(
            rank=4,
            ticker="AMZN",
            exchange="XNYS",
            company_name="Amazon.com Inc",
            currency="USD",
            latest_price=185.0,
            expected_return_pct=0.5,
            directional_win_probability=0.51,
            conviction_label="NEUTRAL",
            cornish_fisher_var_pct=5.0,
            predicted_sharpe=0.3,
            horizon_sessions=5,
            model_family="ensemble",
        ),
    ]


def test_compute_ledoit_wolf_covariance_properties(sample_returns_df):
    """Verify Ledoit-Wolf shrinkage produces positive-definite matrix and bounded shrinkage intensity."""
    receipt = compute_ledoit_wolf_covariance(sample_returns_df)

    assert isinstance(receipt, LedoitWolfReceipt)
    assert receipt.sample_size == 60
    assert 0.0 <= receipt.shrinkage_intensity <= 1.0
    assert receipt.shrunk_covariance.shape == (4, 4)

    # Positive-definite check: all eigenvalues > 0
    eigenvals = np.linalg.eigvals(receipt.shrunk_covariance.values)
    assert np.all(eigenvals > 0)

    # Correlation diagonal == 1.0
    np.testing.assert_allclose(np.diag(receipt.shrunk_correlation.values), 1.0)


def test_optimize_alpha_portfolio_modes_and_constraints(sample_screener_records, sample_returns_df):
    """Verify weight constraints (sum=1.0, 0 <= w <= max_weight) across optimization modes."""
    modes = ["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp"]

    for m in modes:
        receipt = optimize_alpha_portfolio(
            screener_records=sample_screener_records,
            returns_df=sample_returns_df,
            mode=m,
            max_weight=0.40,
        )

        assert isinstance(receipt, PortfolioReceipt)
        assert receipt.mode == m
        weights = list(receipt.weights.values())

        # Weights sum to 1.0
        assert np.isclose(sum(weights), 1.0, atol=1e-4)

        # Bounds: all weights >= 0 and <= max_weight + tolerance
        for w in weights:
            assert w >= -1e-5
            assert w <= 0.40 + 1e-4

        # Risk metrics validity
        assert receipt.annualized_volatility_pct > 0.0
        assert receipt.effective_number_of_assets >= 1.0
        assert len(receipt.risk_contributions_pct) == 4


def test_calculate_portfolio_risk_contributions_identity():
    """Verify risk contributions sum exactly to 1.0."""
    cov = np.array([
        [0.04, 0.01],
        [0.01, 0.09],
    ])
    weights = np.array([0.6, 0.4])
    rc = calculate_portfolio_risk_contributions(weights, cov)

    assert len(rc) == 2
    assert np.isclose(np.sum(rc), 1.0)


def test_compute_hrp_weights_mathematical_properties():
    """Verify HRP tree clustering, inverse-variance properties, and singularity robustness."""
    from src.analytics.portfolio import compute_hrp_weights

    # 1. 3-asset known covariance with asset 0 having lowest risk
    cov = np.array([
        [0.01, 0.002, 0.001],
        [0.002, 0.04, 0.005],
        [0.001, 0.005, 0.09],
    ])
    w = compute_hrp_weights(cov)
    assert len(w) == 3
    assert np.isclose(np.sum(w), 1.0)
    assert (w >= 0.0).all()
    # Lowest variance asset should receive the largest weight in HRP
    assert w[0] > w[1] > w[2]

    # 2. Maximum weight constraint and redistribution
    w_capped = compute_hrp_weights(cov, max_weight=0.45)
    assert np.isclose(np.sum(w_capped), 1.0)
    assert (w_capped <= 0.45 + 1e-6).all()

    # 3. Collinear / singular covariance matrix (rank-deficient where Markowitz fails)
    cov_singular = np.array([
        [0.04, 0.04, 0.01],
        [0.04, 0.04, 0.01],
        [0.01, 0.01, 0.09],
    ])
    w_sing = compute_hrp_weights(cov_singular)
    assert len(w_sing) == 3
    assert np.isclose(np.sum(w_sing), 1.0)
    assert (w_sing >= 0.0).all()


def test_optimize_alpha_portfolio_single_asset_fallback(sample_screener_records):
    """Verify single-asset input produces 100% allocation without errors."""
    receipt = optimize_alpha_portfolio(
        screener_records=[sample_screener_records[0]],
        mode="max_sharpe",
    )
    assert receipt.weights == {"AAPL": 1.0}
    assert receipt.risk_contributions_pct == {"AAPL": 100.0}
    assert receipt.effective_number_of_assets == 1.0


def test_cli_portfolio_command(capsys, sample_screener_records, sample_returns_df):
    """Verify CLI nextomni portfolio command outputs structured JSON and table."""
    from src.cli import main
    import sys

    with patch("src.analytics.screener.screen_asset_universe", return_value=sample_screener_records), \
         patch("src.analytics.portfolio.build_aligned_returns_matrix", return_value=sample_returns_df), \
         patch.object(sys, "argv", ["nextomni", "portfolio", "--mode", "hrp", "--json"]):
        main()

    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert data["mode"] == "hrp"
    assert "weights" in data
    assert "sharpe_ratio" in data

