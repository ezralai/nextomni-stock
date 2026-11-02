"""Unit and integration test suite for Campbell, Hilscher & Szilagyi (2008) Distress Risk Engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pytest

from src.analytics.distress_risk import (
    DistressRiskReceipt,
    calculate_distress_risk,
    classify_distress_regime,
    evaluate_distress_risk_profile,
)
from src.cli import distress_command


def test_calculate_distress_risk_solvent_safe_haven() -> None:
    """Test high-profitability, low-leverage safe solvent compounder."""
    receipt = calculate_distress_risk(
        ticker="SOLVENT_CORP",
        latest_price=150.0,
        market_cap=100_000_000_000.0,  # $100B
        total_liabilities=15_000_000_000.0,  # $15B (TLMTA = 13%)
        net_income=12_000_000_000.0,  # $12B (NIMTA = 10.4%)
        cash_equivalents=20_000_000_000.0,  # $20B
        annualized_volatility_pct=18.0,
        excess_trailing_return_pct=12.0,
    )

    assert receipt.ticker == "SOLVENT_CORP"
    assert receipt.market_adjusted_assets == 115_000_000_000.0
    assert receipt.tlmta_pct < 15.0
    assert receipt.nimta_pct > 8.0
    assert receipt.chs_linear_index < -9.0
    assert receipt.distress_probability_12m_pct < 0.25
    assert receipt.is_solvent_safe_haven is True
    assert receipt.is_distress_trap is False
    assert receipt.regime == "SOLVENT_SAFE_HAVEN"
    assert receipt.expected_distress_alpha_pct == +3.5
    assert receipt.verify_integrity() is True


def test_calculate_distress_risk_acute_insolvency_trap() -> None:
    """Test highly leveraged, chronically unprofitable zombie firm facing imminent distress."""
    receipt = calculate_distress_risk(
        ticker="ZOMBIE_DISTRESS",
        latest_price=0.75,  # Penny stock penalty
        market_cap=25_000_000.0,  # $25M
        total_liabilities=350_000_000.0,  # $350M (TLMTA = 93.3%)
        net_income=-65_000_000.0,  # -$65M heavy losses
        cash_equivalents=1_500_000.0,  # Depleted cash
        annualized_volatility_pct=85.0,  # Extreme volatility
        excess_trailing_return_pct=-45.0,  # Severe stock crash
    )

    assert receipt.tlmta_pct > 85.0
    assert receipt.nimta_pct < -10.0
    assert receipt.chs_linear_index > -1.0
    assert receipt.distress_probability_12m_pct >= 8.0
    assert receipt.is_distress_trap is True
    assert receipt.is_solvent_safe_haven is False
    assert receipt.regime == "ACUTE_INSOLVENCY_CRISIS"
    assert receipt.expected_distress_alpha_pct == -14.2
    assert receipt.verify_integrity() is True


def test_distress_probability_monotonicity() -> None:
    """Mathematical proof that increasing net income decreases default probability and increasing debt increases it."""
    base_mcap = 10_000_000_000.0
    base_price = 50.0

    # Monotonicity with respect to Net Income
    rec_low_profit = calculate_distress_risk(
        ticker="TEST",
        latest_price=base_price,
        market_cap=base_mcap,
        total_liabilities=4_000_000_000.0,
        net_income=-200_000_000.0,  # Losses
        cash_equivalents=500_000_000.0,
    )
    rec_high_profit = calculate_distress_risk(
        ticker="TEST",
        latest_price=base_price,
        market_cap=base_mcap,
        total_liabilities=4_000_000_000.0,
        net_income=+800_000_000.0,  # Profits
        cash_equivalents=500_000_000.0,
    )

    assert rec_high_profit.distress_probability_12m_pct < rec_low_profit.distress_probability_12m_pct

    # Monotonicity with respect to Debt Liabilities
    rec_low_debt = calculate_distress_risk(
        ticker="TEST",
        latest_price=base_price,
        market_cap=base_mcap,
        total_liabilities=1_000_000_000.0,
        net_income=500_000_000.0,
        cash_equivalents=500_000_000.0,
    )
    rec_high_debt = calculate_distress_risk(
        ticker="TEST",
        latest_price=base_price,
        market_cap=base_mcap,
        total_liabilities=8_000_000_000.0,
        net_income=500_000_000.0,
        cash_equivalents=500_000_000.0,
    )

    assert rec_low_debt.distress_probability_12m_pct < rec_high_debt.distress_probability_12m_pct


def test_distress_risk_receipt_hmac_tamper() -> None:
    """Verify cryptographic authenticity and tamper detection of DistressRiskReceipt."""
    receipt = calculate_distress_risk(
        ticker="HMAC_CORP",
        latest_price=80.0,
        market_cap=5_000_000_000.0,
        total_liabilities=1_000_000_000.0,
        net_income=400_000_000.0,
        cash_equivalents=600_000_000.0,
    )

    assert receipt.verify_integrity() is True

    # Tamper with distress probability
    tampered = DistressRiskReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        market_cap=receipt.market_cap,
        total_liabilities=receipt.total_liabilities,
        net_income=receipt.net_income,
        cash_equivalents=receipt.cash_equivalents,
        market_adjusted_assets=receipt.market_adjusted_assets,
        nimta_pct=receipt.nimta_pct,
        tlmta_pct=receipt.tlmta_pct,
        cashmta_pct=receipt.cashmta_pct,
        annualized_volatility_pct=receipt.annualized_volatility_pct,
        excess_trailing_return_pct=receipt.excess_trailing_return_pct,
        chs_linear_index=receipt.chs_linear_index,
        distress_probability_12m_pct=receipt.distress_probability_12m_pct + 12.0,  # TAMPER
        expected_distress_alpha_pct=receipt.expected_distress_alpha_pct,
        is_solvent_safe_haven=receipt.is_solvent_safe_haven,
        is_distress_trap=receipt.is_distress_trap,
        regime=receipt.regime,
        summary=receipt.summary,
        created_at_utc=receipt.created_at_utc,
        signature=receipt.signature,
    )
    assert tampered.verify_integrity() is False


def test_evaluate_distress_risk_profile_end_to_end() -> None:
    """Test evaluate_distress_risk_profile end-to-end with database fallback and valid dict schema."""
    receipt = evaluate_distress_risk_profile("AAPL")

    assert receipt.ticker == "AAPL"
    assert receipt.latest_price > 0
    assert receipt.market_cap > 0
    assert receipt.verify_integrity() is True
    assert isinstance(receipt.to_dict(), dict)


def test_cli_distress_command(capsys: pytest.CaptureFixture) -> None:
    """Test CLI distress command handler with table formatting and JSON output."""
    # Test human-readable table
    args_text = argparse.Namespace(
        ticker="IBM",
        market_cap=120_000_000_000.0,
        debt=40_000_000_000.0,
        net_income=8_000_000_000.0,
        cash=10_000_000_000.0,
        json=False,
    )
    distress_command(args_text)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Campbell, Hilscher & Szilagyi (2008) Distress Risk Engine" in captured
    assert "IBM" in captured
    assert "12-Month Failure Hazard" in captured

    # Test JSON output
    args_json = argparse.Namespace(
        ticker="UBER",
        market_cap=150_000_000_000.0,
        debt=30_000_000_000.0,
        net_income=5_000_000_000.0,
        cash=7_000_000_000.0,
        json=True,
    )
    distress_command(args_json)
    captured_json = capsys.readouterr().out
    payload = json.loads(captured_json)
    assert payload["ticker"] == "UBER"
    assert "distress_probability_12m_pct" in payload
    assert "signature" in payload


def test_distress_risk_input_validation() -> None:
    """Test defensive security validation on price, market cap, and debt inputs."""
    # Non-positive price
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_distress_risk(
            ticker="FAIL",
            latest_price=-10.0,
            market_cap=1_000_000.0,
            total_liabilities=100_000.0,
            net_income=50_000.0,
            cash_equivalents=50_000.0,
        )

    # Non-positive market cap
    with pytest.raises(ValueError, match="strictly positive"):
        calculate_distress_risk(
            ticker="FAIL",
            latest_price=10.0,
            market_cap=0.0,
            total_liabilities=100_000.0,
            net_income=50_000.0,
            cash_equivalents=50_000.0,
        )

    # Negative liabilities
    with pytest.raises(ValueError, match="cannot be negative"):
        calculate_distress_risk(
            ticker="FAIL",
            latest_price=10.0,
            market_cap=1_000_000.0,
            total_liabilities=-50_000.0,
            net_income=50_000.0,
            cash_equivalents=50_000.0,
        )

    # Invalid ticker format
    with pytest.raises(ValueError):
        evaluate_distress_risk_profile("INVALID; DROP TABLE stock_observations;")
