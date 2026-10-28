"""Unit and integration tests for Multi-Factor Risk Attribution & Style Exposure Engine."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.factor_attribution import (
    FactorAttributionReceipt,
    construct_style_factor_returns,
    decompose_portfolio_factor_risk,
    estimate_factor_betas,
    evaluate_factor_attribution,
)
from src.cli import factor_attribution_command


def test_construct_style_factor_returns_properties() -> None:
    """Verify leak-free style factor returns generation and mathematical bounds."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=60, freq="B")
    tickers = ["AAPL", "MSFT", "GOOG", "AMZN"]
    
    # Generate synthetic daily returns
    data = np.random.normal(0.0005, 0.015, size=(60, 4))
    df_returns = pd.DataFrame(data, index=dates, columns=tickers)

    factor_df = construct_style_factor_returns(df_returns, window_mom=10, window_vol=10, window_rev=5)
    
    assert not factor_df.empty
    assert list(factor_df.columns) == ["MKT", "MOM", "VOL", "REV"]
    assert len(factor_df) == 60 - (10 + 1)
    assert not factor_df.isna().any().any()

    # Empty and degenerate single-asset input
    assert construct_style_factor_returns(pd.DataFrame()).empty
    single_asset = pd.DataFrame({"AAPL": [0.01, -0.02, 0.03]})
    assert construct_style_factor_returns(single_asset).empty


def test_estimate_factor_betas_mathematical_properties() -> None:
    """Verify OLS beta estimation, Newey-West HAC errors, and Euler risk decomposition."""
    np.random.seed(101)
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    # Generate synthetic orthogonal factors
    f_mkt = np.random.normal(0.0004, 0.012, size=n)
    f_mom = np.random.normal(0.0001, 0.008, size=n)
    f_vol = np.random.normal(0.0002, 0.006, size=n)

    factors_df = pd.DataFrame(
        {"MKT": f_mkt, "MOM": f_mom, "VOL": f_vol},
        index=dates,
    )

    # True data generating process: y = 0.0002 + 1.25 * MKT - 0.60 * MOM + 0.30 * VOL + noise
    noise = np.random.normal(0.0, 0.004, size=n)
    true_alpha = 0.0002
    y_returns = pd.Series(
        true_alpha + 1.25 * f_mkt - 0.60 * f_mom + 0.30 * f_vol + noise,
        index=dates,
    )

    res = estimate_factor_betas(y_returns, factors_df)

    assert res["sample_sessions"] == n
    assert 0.0 <= res["r_squared"] <= 1.0
    assert 0.0 <= res["adj_r_squared"] <= 1.0

    # Betas should be close to true values
    assert abs(res["factor_betas"]["MKT"] - 1.25) < 0.15
    assert abs(res["factor_betas"]["MOM"] - (-0.60)) < 0.15
    assert abs(res["factor_betas"]["VOL"] - 0.30) < 0.15

    # Newey-West HAC standard errors should yield significant t-stats for true factors
    assert abs(res["factor_t_stats"]["MKT"]) > 2.0
    assert res["factor_p_values"]["MKT"] < 0.05
    assert abs(res["factor_t_stats"]["MOM"]) > 2.0
    assert res["factor_p_values"]["MOM"] < 0.05

    # Euler exact risk decomposition
    assert abs(res["systemic_risk_share_pct"] + res["specific_risk_share_pct"] - 100.0) < 1e-2
    assert res["systemic_risk_share_pct"] > 60.0  # Dominant systematic factor risk


def test_decompose_portfolio_factor_risk() -> None:
    """Verify multi-asset portfolio factor risk decomposition linearity."""
    np.random.seed(202)
    n = 80
    dates = pd.date_range("2025-01-01", periods=n, freq="B")

    factors_df = pd.DataFrame({
        "MKT": np.random.normal(0.0003, 0.01, size=n),
        "MOM": np.random.normal(0.0001, 0.007, size=n),
    }, index=dates)

    returns_matrix = pd.DataFrame({
        "AAPL": np.random.normal(0.0004, 0.015, size=n),
        "MSFT": np.random.normal(0.0003, 0.014, size=n),
    }, index=dates)

    weights = {"AAPL": 0.60, "MSFT": 0.40}
    asset_betas = {
        "AAPL": {"MKT": 1.20, "MOM": 0.50},
        "MSFT": {"MKT": 0.90, "MOM": -0.20},
    }

    port_decomp = decompose_portfolio_factor_risk(weights, asset_betas, factors_df, returns_matrix)

    # Portfolio beta = 0.60 * 1.20 + 0.40 * 0.90 = 0.72 + 0.36 = 1.08
    expected_mkt_beta = 0.60 * 1.20 + 0.40 * 0.90
    assert abs(port_decomp["portfolio_betas"]["MKT"] - expected_mkt_beta) < 1e-4

    # Portfolio MOM beta = 0.60 * 0.50 + 0.40 * (-0.20) = 0.30 - 0.08 = 0.22
    expected_mom_beta = 0.60 * 0.50 + 0.40 * (-0.20)
    assert abs(port_decomp["portfolio_betas"]["MOM"] - expected_mom_beta) < 1e-4

    assert abs(port_decomp["systemic_risk_share_pct"] + port_decomp["specific_risk_share_pct"] - 100.0) < 1e-2


def test_factor_attribution_receipt_signature_and_tamper() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    receipt = FactorAttributionReceipt(
        ticker="JPM",
        exchange="XNYS",
        sample_sessions=120,
        r_squared=0.6543,
        adj_r_squared=0.6432,
        annualized_alpha_pct=4.50,
        alpha_t_stat=1.85,
        alpha_p_value=0.067,
        factor_betas={"MKT": 1.15, "MOM": -0.25},
        factor_t_stats={"MKT": 8.5, "MOM": -1.9},
        factor_p_values={"MKT": 0.0001, "MOM": 0.058},
        factor_risk_contributions_pct={"MKT": 62.0, "MOM": 3.4},
        systemic_risk_share_pct=65.4,
        specific_risk_share_pct=34.6,
        total_volatility_ann_pct=22.5,
        idiosyncratic_volatility_ann_pct=13.2,
        summary="Test attribution receipt",
    )

    assert receipt.signature != ""
    assert receipt.verify_integrity() is True

    # Tamper with numerical metrics
    receipt.r_squared = 0.9999
    assert receipt.verify_integrity() is False


def test_evaluate_factor_attribution_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify full end-to-end evaluation with mocked universe return matrix."""
    np.random.seed(77)
    dates = pd.date_range("2025-01-01", periods=80, freq="B")
    mock_returns = pd.DataFrame({
        "JPM": np.random.normal(0.0005, 0.015, size=80),
        "BAC": np.random.normal(0.0004, 0.016, size=80),
        "WFC": np.random.normal(0.0003, 0.014, size=80),
    }, index=dates)

    monkeypatch.setattr(
        "src.analytics.factor_attribution.get_all_stocks",
        lambda db_path=None: [
            {"ticker": "JPM", "exchange": "XNYS"},
            {"ticker": "BAC", "exchange": "XNYS"},
            {"ticker": "WFC", "exchange": "XNYS"},
        ],
    )
    monkeypatch.setattr(
        "src.analytics.factor_attribution.build_aligned_returns_matrix",
        lambda tickers, exchange, db_path=None, min_sessions=60: mock_returns,
    )

    receipt = evaluate_factor_attribution(ticker="JPM", exchange="XNYS", min_sessions=30)

    assert receipt.ticker == "JPM"
    assert receipt.exchange == "XNYS"
    assert receipt.sample_sessions > 0
    assert "MKT" in receipt.factor_betas
    assert receipt.verify_integrity() is True


def test_cli_factor_attribution_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify CLI factor-attribution command execution and JSON serialization."""
    fake_receipt = FactorAttributionReceipt(
        ticker="AAPL",
        exchange="XNYS",
        sample_sessions=100,
        r_squared=0.7200,
        adj_r_squared=0.7100,
        annualized_alpha_pct=5.20,
        alpha_t_stat=2.15,
        alpha_p_value=0.033,
        factor_betas={"MKT": 1.10, "MOM": 0.40},
        factor_t_stats={"MKT": 12.0, "MOM": 3.2},
        factor_p_values={"MKT": 0.0001, "MOM": 0.002},
        factor_risk_contributions_pct={"MKT": 60.0, "MOM": 12.0},
        systemic_risk_share_pct=72.0,
        specific_risk_share_pct=28.0,
        total_volatility_ann_pct=24.0,
        idiosyncratic_volatility_ann_pct=12.7,
        summary="AAPL exhibits 72% systematic factor risk.",
    )

    monkeypatch.setattr(
        "src.analytics.factor_attribution.evaluate_factor_attribution",
        lambda ticker, exchange: fake_receipt,
    )

    # 1. Standard stdout tabular report
    args_std = argparse.Namespace(ticker="AAPL", exchange="XNYS", json=False)
    factor_attribution_command(args_std)
    out_std = capsys.readouterr().out
    assert "Multi-Factor Risk Attribution & Style Exposure Engine" in out_std
    assert "AAPL (XNYS)" in out_std
    assert "72.0%" in out_std

    # 2. JSON structured output
    args_json = argparse.Namespace(ticker="AAPL", exchange="XNYS", json=True)
    factor_attribution_command(args_json)
    out_json = capsys.readouterr().out
    parsed = json.loads(out_json)
    assert parsed["ticker"] == "AAPL"
    assert parsed["systemic_risk_share_pct"] == 72.0
    assert parsed["factor_betas"]["MKT"] == 1.10
    assert "signature" in parsed
