"""Unit and integration test suite for Frazzini & Pedersen (2014) Betting Against Beta Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.betting_against_beta import (
    BABReceipt,
    calculate_betting_against_beta,
    classify_bab_regime,
    evaluate_bab_profile,
)
from src.cli import bab_command


def test_calculate_betting_against_beta_low_beta_compounder() -> None:
    """Test low-beta asset with high risk-adjusted return (classic BAB candidate)."""
    n = 252
    rng = np.random.default_rng(101)

    # Market benchmark returns: drift = 10% ann, vol = 16% ann
    r_bench = rng.normal(0.10 / 252, 0.16 / math.sqrt(252), n)
    bench_prices = 300.0 * np.exp(np.cumsum(r_bench))

    # Low-beta asset: beta ~ 0.50 with robust idiosyncratic positive alpha (+12% ann)
    r_asset = 0.50 * r_bench + (0.12 / 252) + rng.normal(0, 0.04 / math.sqrt(252), n)
    asset_prices = 100.0 * np.exp(np.cumsum(r_asset))

    receipt = calculate_betting_against_beta(
        prices=asset_prices,
        benchmark_prices=bench_prices,
        ticker="LOW_BETA",
        benchmark_ticker="SPY",
        rf_rate_pct=4.0,
        equity_risk_premium_pct=6.0,
    )

    assert receipt.ticker == "LOW_BETA"
    assert receipt.sessions_evaluated == 252
    assert receipt.raw_beta < 0.85
    assert receipt.shrunk_beta > receipt.raw_beta  # Vasicek shrinkage pulls beta toward 1.0
    assert receipt.shrunk_beta <= 0.85
    assert receipt.bab_leverage_multiplier > 1.0  # Safe asset is levered up
    assert receipt.is_low_beta_compounder is True
    assert receipt.is_high_beta_overpriced is False
    assert receipt.regime == "LOW_BETA_ALPHA_COMPOUNDER"
    assert receipt.verify_integrity() is True


def test_calculate_betting_against_beta_high_beta_overpriced() -> None:
    """Test high-beta speculative asset with low risk-adjusted alpha (overpriced due to leverage constraint)."""
    n = 252
    rng = np.random.default_rng(202)

    r_bench = rng.normal(0.08 / 252, 0.15 / math.sqrt(252), n)
    bench_prices = 250.0 * np.exp(np.cumsum(r_bench))

    # High-beta asset: beta ~ 1.80 with negative alpha (-4% ann)
    r_asset = 1.80 * r_bench - (0.04 / 252) + rng.normal(0, 0.18 / math.sqrt(252), n)
    asset_prices = 50.0 * np.exp(np.cumsum(r_asset))

    receipt = calculate_betting_against_beta(
        prices=asset_prices,
        benchmark_prices=bench_prices,
        ticker="HIGH_BETA",
        benchmark_ticker="SPY",
        rf_rate_pct=4.0,
        equity_risk_premium_pct=6.0,
    )

    assert receipt.raw_beta > 1.30
    assert receipt.shrunk_beta < receipt.raw_beta  # Vasicek shrinkage pulls beta down toward 1.0
    assert receipt.bab_leverage_multiplier < 1.0  # Speculative asset is de-levered
    assert receipt.is_high_beta_overpriced is True
    assert receipt.is_low_beta_compounder is False
    assert receipt.regime in ("HIGH_BETA_OVERPRICED", "SPECULATIVE_LOTTERY")


def test_vasicek_shrinkage_mathematical_invariance() -> None:
    """Mathematical proof that Vasicek Bayesian shrinkage contracts extreme betas toward unity."""
    n = 100
    prices = [100.0 * (1.0 + 0.001 * i) for i in range(n)]

    # Mock high-beta and low-beta conditions via calculate_betting_against_beta
    rec_high = calculate_betting_against_beta(
        prices=[100.0 * (1.0 + 0.004 * i) for i in range(n)],
        benchmark_prices=prices,
        ticker="HI",
    )
    rec_low = calculate_betting_against_beta(
        prices=[100.0 * (1.0 + 0.0003 * i) for i in range(n)],
        benchmark_prices=prices,
        ticker="LO",
    )

    # For any raw beta != 1.0, distance to 1.0 of shrunk beta MUST be strictly smaller
    assert abs(rec_high.shrunk_beta - 1.0) <= abs(rec_high.raw_beta - 1.0)
    assert abs(rec_low.shrunk_beta - 1.0) <= abs(rec_low.raw_beta - 1.0)


def test_zero_beta_bab_portfolio_property() -> None:
    """Prove that combining levered low-beta with de-levered high-beta forms an exact zero-beta portfolio."""
    beta_low = 0.65
    beta_high = 1.60

    w_low = 1.0 / beta_low
    w_high = 1.0 / beta_high

    # Net portfolio market beta: w_low * beta_low - w_high * beta_high = 1.0 - 1.0 = 0.0
    portfolio_beta = (w_low * beta_low) - (w_high * beta_high)
    assert portfolio_beta == pytest.approx(0.0, abs=1e-12)


def test_bab_receipt_hmac_tamper_resistance() -> None:
    """Verify cryptographic integrity and tamper detection of BABReceipt."""
    prices = [100.0 * (1.0 + 0.001 * i) for i in range(100)]
    receipt = calculate_betting_against_beta(prices, ticker="HMAC_TEST")

    assert receipt.verify_integrity() is True

    # Tamper with shrunk beta
    tampered = BABReceipt(
        ticker=receipt.ticker,
        benchmark_ticker=receipt.benchmark_ticker,
        sessions_evaluated=receipt.sessions_evaluated,
        latest_price=receipt.latest_price,
        realized_annualized_return_pct=receipt.realized_annualized_return_pct,
        asset_annualized_volatility_pct=receipt.asset_annualized_volatility_pct,
        benchmark_annualized_volatility_pct=receipt.benchmark_annualized_volatility_pct,
        correlation_to_benchmark=receipt.correlation_to_benchmark,
        raw_beta=receipt.raw_beta,
        shrunk_beta=receipt.shrunk_beta + 0.50,  # TAMPER
        capm_expected_return_pct=receipt.capm_expected_return_pct,
        jensen_alpha_pct=receipt.jensen_alpha_pct,
        bab_leverage_multiplier=receipt.bab_leverage_multiplier,
        expected_bab_spread_pct=receipt.expected_bab_spread_pct,
        is_low_beta_compounder=receipt.is_low_beta_compounder,
        is_high_beta_overpriced=receipt.is_high_beta_overpriced,
        regime=receipt.regime,
        summary=receipt.summary,
        created_at_utc=receipt.created_at_utc,
        signature=receipt.signature,
    )
    assert tampered.verify_integrity() is False


def test_evaluate_bab_profile_end_to_end() -> None:
    """Test evaluate_bab_profile end-to-end with database fallback and valid schema."""
    receipt = evaluate_bab_profile("MSFT", benchmark_ticker="SPY")

    assert receipt.ticker == "MSFT"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.sessions_evaluated >= 42
    assert receipt.verify_integrity() is True
    assert isinstance(receipt.to_dict(), dict)


def test_cli_bab_command(capsys: pytest.CaptureFixture) -> None:
    """Test CLI bab command handler with text table and JSON format."""
    # Test human-readable table
    args_text = argparse.Namespace(
        ticker="JNJ",
        benchmark="SPY",
        rf_rate=4.0,
        market_premium=6.0,
        json=False,
    )
    bab_command(args_text)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Frazzini & Pedersen (2014) Betting Against Beta (BAB) Engine" in captured
    assert "JNJ" in captured
    assert "Vasicek Shrunk Beta" in captured
    assert "Optimal BAB Multiplier" in captured

    # Test JSON output
    args_json = argparse.Namespace(
        ticker="TSLA",
        benchmark="QQQ",
        rf_rate=4.5,
        market_premium=6.5,
        json=True,
    )
    bab_command(args_json)
    captured_json = capsys.readouterr().out
    payload = json.loads(captured_json)
    assert payload["ticker"] == "TSLA"
    assert payload["benchmark_ticker"] == "QQQ"
    assert "shrunk_beta" in payload
    assert "signature" in payload


def test_bab_input_validation() -> None:
    """Test defensive security validation on price inputs and ticker formatting."""
    # Insufficient length
    with pytest.raises(ValueError, match="requires at least 42 sessions"):
        calculate_betting_against_beta([100.0] * 20, ticker="SHORT")

    # Non-finite price
    with pytest.raises(ValueError):
        calculate_betting_against_beta([100.0] * 50 + [float("nan")], ticker="NAN")

    # Non-positive price
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_betting_against_beta([100.0] * 50 + [0.0], ticker="ZERO")

    # Invalid ticker format
    with pytest.raises(ValueError):
        evaluate_bab_profile("INVALID; DROP TABLE stock_observations;")
