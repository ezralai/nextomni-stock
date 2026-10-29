"""Comprehensive test suite for Michaud (1998) Resampled Efficiency and robust frontier engine."""

import argparse
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.resampled_portfolio import (
    FrontierPoint,
    ResampledFrontierReceipt,
    _calculate_effective_assets,
    compute_resampled_efficient_frontier,
    evaluate_resampled_frontier_profile,
    trace_frontier,
)
from src.cli import resampled_frontier_command


@pytest.fixture
def synthetic_returns_df() -> pd.DataFrame:
    """Generate reproducible synchronized multi-asset returns matrix."""
    np.random.seed(42)
    n_obs = 120
    tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"]
    # Correlated returns
    means = [0.0008, 0.0006, 0.0012, 0.0007]
    cov_structure = np.array([
        [0.0004, 0.0002, 0.00025, 0.00018],
        [0.0002, 0.00035, 0.00022, 0.0002],
        [0.00025, 0.00022, 0.0006, 0.00024],
        [0.00018, 0.0002, 0.00024, 0.00038],
    ])
    draws = np.random.multivariate_normal(means, cov_structure, size=n_obs)
    dates = pd.date_range("2024-01-01", periods=n_obs, freq="B").strftime("%Y-%m-%d")
    df = pd.DataFrame(draws, index=dates, columns=tickers)
    return df


def test_effective_assets_calculation():
    """Verify Inverse Herfindahl-Hirschman index properties."""
    # 1 asset gets 100%
    w_single = np.array([1.0, 0.0, 0.0, 0.0])
    assert abs(_calculate_effective_assets(w_single) - 1.0) < 1e-4

    # 4 equal-weighted assets get 4.0
    w_equal = np.array([0.25, 0.25, 0.25, 0.25])
    assert abs(_calculate_effective_assets(w_equal) - 4.0) < 1e-4

    # 2 assets split 50/50 get 2.0
    w_half = np.array([0.5, 0.5, 0.0, 0.0])
    assert abs(_calculate_effective_assets(w_half) - 2.0) < 1e-4


def test_trace_frontier_single_and_multi_asset():
    """Verify classical frontier tracing constraints and weight normalization."""
    # Single asset
    mu_1 = np.array([0.15])
    cov_1 = np.array([[0.04]])
    pts_1, w_1 = trace_frontier(mu_1, cov_1, ["AAPL"], num_points=5)
    assert len(pts_1) == 5
    assert pts_1[0].weights == {"AAPL": 1.0}
    assert w_1.shape == (5, 1)

    # Multi asset
    mu_multi = np.array([0.12, 0.18, 0.25])
    cov_multi = np.array([
        [0.04, 0.01, 0.02],
        [0.01, 0.06, 0.025],
        [0.02, 0.025, 0.09],
    ])
    pts_m, w_m = trace_frontier(mu_multi, cov_multi, ["AAPL", "MSFT", "NVDA"], num_points=8)
    assert len(pts_m) == 8
    assert w_m.shape == (8, 3)

    for k in range(8):
        # Weights non-negative and sum to 1.0
        assert np.all(w_m[k, :] >= -1e-6)
        assert abs(np.sum(w_m[k, :]) - 1.0) < 1e-4
        assert pts_m[k].effective_assets >= 1.0


def test_compute_resampled_efficient_frontier_properties(synthetic_returns_df):
    """Verify Michaud resampled frontier mathematical properties and receipt structure."""
    receipt = compute_resampled_efficient_frontier(
        returns_df=synthetic_returns_df,
        num_simulations=30,
        num_frontier_points=10,
        risk_free_rate=0.04,
        random_state=42,
    )

    assert isinstance(receipt, ResampledFrontierReceipt)
    assert receipt.tickers == ["AAPL", "MSFT", "NVDA", "GOOGL"]
    assert receipt.sample_sessions == 120
    assert receipt.num_simulations == 30
    assert receipt.num_frontier_points == 10
    assert len(receipt.classical_frontier) == 10
    assert len(receipt.resampled_frontier) == 10

    # Verify resampled max sharpe
    assert receipt.resampled_max_sharpe.sharpe_ratio > 0.0
    assert receipt.classical_max_sharpe.sharpe_ratio > 0.0
    assert receipt.resampled_max_sharpe.effective_assets >= 1.0

    # Verify weights sum to 1.0 across resampled points
    for pt in receipt.resampled_frontier:
        w_sum = sum(pt.weights.values())
        assert abs(w_sum - 1.0) < 1e-3

    # Check weight stability dispersion dictionary
    assert set(receipt.weight_stability_dispersion.keys()) == set(receipt.tickers)
    for t, s in receipt.weight_stability_dispersion.items():
        assert s >= 0.0


def test_resampled_frontier_receipt_tamper_detection(synthetic_returns_df):
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    receipt = compute_resampled_efficient_frontier(
        returns_df=synthetic_returns_df,
        num_simulations=25,
        num_frontier_points=8,
        risk_free_rate=0.04,
    )

    assert receipt.verify_integrity() is True

    # Tamper with Sharpe ratio
    tampered_dict = receipt.to_dict()
    tampered_dict["resampled_max_sharpe"]["sharpe_ratio"] += 2.5
    tampered_pt = FrontierPoint(**tampered_dict["resampled_max_sharpe"])
    tampered_receipt = ResampledFrontierReceipt(
        tickers=receipt.tickers,
        sample_sessions=receipt.sample_sessions,
        num_simulations=receipt.num_simulations,
        num_frontier_points=receipt.num_frontier_points,
        risk_free_rate_pct=receipt.risk_free_rate_pct,
        shrinkage_intensity=receipt.shrinkage_intensity,
        classical_max_sharpe=receipt.classical_max_sharpe,
        resampled_max_sharpe=tampered_pt,
        classical_min_variance=receipt.classical_min_variance,
        resampled_min_variance=receipt.resampled_min_variance,
        classical_frontier=receipt.classical_frontier,
        resampled_frontier=receipt.resampled_frontier,
        michaud_diversification_gain_pct=receipt.michaud_diversification_gain_pct,
        weight_stability_dispersion=receipt.weight_stability_dispersion,
        summary=receipt.summary,
        signature=receipt.signature,
    )
    assert tampered_receipt.verify_integrity() is False


def test_evaluate_resampled_frontier_profile_insufficient_or_invalid(monkeypatch):
    """Verify security validation and edge-case error bounds."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_resampled_frontier_profile(["INVALID_TICKER_123$$$"])

    # Less than 2 tickers
    with pytest.raises(ValueError, match="At least 2 validated ticker symbols"):
        evaluate_resampled_frontier_profile(["AAPL"])

    # Insufficient DB data
    monkeypatch.setattr(
        "src.analytics.resampled_portfolio.build_aligned_returns_matrix",
        lambda tickers, db_path=None, min_sessions=30: pd.DataFrame(),
    )
    with pytest.raises(ValueError, match="Insufficient synchronized session observations"):
        evaluate_resampled_frontier_profile(["AAPL", "MSFT"])


def test_cli_resampled_frontier_command_stdout_and_json(synthetic_returns_df, monkeypatch):
    """Test CLI resampled-frontier command formatting in stdout and JSON modes."""
    monkeypatch.setattr(
        "src.analytics.resampled_portfolio.build_aligned_returns_matrix",
        lambda tickers, db_path=None, min_sessions=30: synthetic_returns_df,
    )

    # 1. Standard stdout format
    args_std = argparse.Namespace(
        tickers="AAPL,MSFT,NVDA,GOOGL",
        simulations=20,
        points=8,
        risk_free_rate=0.04,
        json=False,
    )
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        resampled_frontier_command(args_std)
        output = fake_out.getvalue()
        assert "NextOmni Stock: Michaud (1998) Resampled Efficiency" in output
        assert "Maximum Sharpe Ratio Allocations Comparison:" in output
        assert "Michaud Resampled" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(
        tickers="AAPL,MSFT,NVDA,GOOGL",
        simulations=20,
        points=8,
        risk_free_rate=0.04,
        json=True,
    )
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        resampled_frontier_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert "tickers" in payload
        assert "resampled_max_sharpe" in payload
        assert "resampled_frontier" in payload
        assert "signature" in payload
        assert payload["num_simulations"] == 20
