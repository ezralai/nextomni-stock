"""Unit, boundary, security, and integration tests for J. Bradford De Long, Andrei Shleifer,

Lawrence H. Summers & Robert J. Waldmann (1990) Noise Trader Risk & Limits of Arbitrage Engine.
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.dssw_noise_trader_risk import (
    DSSWReceipt,
    calculate_dssw_equilibrium,
    classify_dssw_regime,
    evaluate_dssw_profile,
    persist_dssw_record,
    verify_dssw_audit_digest,
)
from src.cli import dssw_command
from src.security.validators import SecurityValidationError


def test_calculate_dssw_sentiment_euphoria() -> None:
    """Verify extreme bullish sentiment triggers sentiment euphoria regime."""
    receipt = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.10,
        sentiment_shock=0.25,
        sentiment_variance=0.01,
        risk_aversion=1.0,
        risk_free_rate=0.045,
        ticker="EUPHOR",
    )

    assert receipt.ticker == "EUPHOR"
    assert receipt.mispricing_percentage_pct > 15.0
    assert receipt.is_sentiment_euphoria is True
    assert receipt.regime == "NOISE_TRADER_SENTIMENT_EUPHORIA"
    assert verify_dssw_audit_digest(receipt) is True


def test_calculate_dssw_panic_depression() -> None:
    """Verify extreme bearish sentiment shock triggers panic depression regime."""
    receipt = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=-0.08,
        sentiment_shock=-0.25,
        sentiment_variance=0.01,
        risk_aversion=1.0,
        risk_free_rate=0.045,
        ticker="PANIC",
    )

    assert receipt.ticker == "PANIC"
    assert receipt.mispricing_percentage_pct < -15.0
    assert receipt.is_panic_depression is True
    assert receipt.regime == "NOISE_TRADER_PANIC_DEPRESSION"
    assert verify_dssw_audit_digest(receipt) is True


def test_calculate_dssw_arbitrage_risk_dominated() -> None:
    """Verify high sentiment variance triggers arbitrage risk dominated regime via create-space effect."""
    receipt = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.02,
        sentiment_shock=0.02,
        sentiment_variance=0.05,
        risk_aversion=2.0,
        risk_free_rate=0.04,
        ticker="ARBRISK",
    )

    assert receipt.create_space_discount_dollar > (0.10 * receipt.fundamental_price)
    assert receipt.is_arbitrage_risk_dominated is True
    assert receipt.regime == "ARBITRAGE_RISK_DOMINATED"
    assert verify_dssw_audit_digest(receipt) is True


def test_calculate_dssw_rational_fundamental_equilibrium() -> None:
    """Verify mild sentiment shocks and low variance maintain fundamental equilibrium."""
    receipt = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.01,
        sentiment_shock=0.02,
        sentiment_variance=0.005,
        risk_aversion=1.0,
        risk_free_rate=0.045,
        ticker="RATION",
    )

    assert abs(receipt.mispricing_percentage_pct) <= 15.0
    assert receipt.is_fundamental_equilibrium is True
    assert receipt.regime == "RATIONAL_FUNDAMENTAL_EQUILIBRIUM"
    assert verify_dssw_audit_digest(receipt) is True


def test_create_space_effect_monotonicity() -> None:
    """Verify fundamental DSSW theorem: increasing sentiment variance strictly depresses market price."""
    variances = [0.005, 0.015, 0.030, 0.050, 0.080]
    prices = []
    discounts = []

    for v in variances:
        rec = calculate_dssw_equilibrium(
            fundamental_price=100.0,
            mean_sentiment=0.03,
            sentiment_shock=0.03,
            sentiment_variance=v,
            risk_aversion=1.5,
            risk_free_rate=0.045,
        )
        prices.append(rec.dssw_market_price)
        discounts.append(rec.create_space_discount_dollar)

    # Market price must strictly decrease as sentiment variance increases
    for i in range(len(prices) - 1):
        assert prices[i] > prices[i + 1]

    # Create space discount must strictly increase
    for i in range(len(discounts) - 1):
        assert discounts[i] < discounts[i + 1]


def test_noise_traders_can_outperform() -> None:
    """Verify DSSW proposition: noise traders earn higher expected return by bearing sentiment risk."""
    rec_outperform = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.08,
        sentiment_shock=0.05,
        sentiment_variance=0.01,
        risk_aversion=1.0,
        risk_free_rate=0.045,
    )
    assert rec_outperform.expected_return_differential_pct > 0.0
    assert rec_outperform.is_noise_trader_outperforming is True

    # Under very high risk aversion or extreme variance, Friedman effect dominates
    rec_underperform = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.02,
        sentiment_shock=0.02,
        sentiment_variance=0.06,
        risk_aversion=3.0,
        risk_free_rate=0.045,
    )
    assert rec_underperform.expected_return_differential_pct < 0.0
    assert rec_underperform.is_noise_trader_outperforming is False


def test_audit_digest_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature detects tampering in DSSW receipt."""
    receipt = calculate_dssw_equilibrium(
        fundamental_price=100.0,
        mean_sentiment=0.05,
        sentiment_shock=0.08,
        ticker="AUTHEN",
    )
    assert verify_dssw_audit_digest(receipt) is True

    # Tamper with market price
    tampered_dict = receipt.to_dict()
    tampered_dict["dssw_market_price"] = 999.99
    tampered_receipt = DSSWReceipt(**tampered_dict)
    assert verify_dssw_audit_digest(tampered_receipt) is False

    # Tamper with mispricing percentage
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["mispricing_percentage_pct"] = 88.88
    tampered_receipt2 = DSSWReceipt(**tampered_dict2)
    assert verify_dssw_audit_digest(tampered_receipt2) is False


def test_invalid_inputs_raise_errors() -> None:
    """Verify invalid numerical inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        calculate_dssw_equilibrium(fundamental_price=-50.0)

    with pytest.raises(ValueError):
        calculate_dssw_equilibrium(sentiment_variance=-0.01)

    with pytest.raises(ValueError):
        calculate_dssw_equilibrium(risk_aversion=0.0)

    with pytest.raises(ValueError):
        calculate_dssw_equilibrium(risk_free_rate=0.0)

    with pytest.raises(ValueError):
        calculate_dssw_equilibrium(fundamental_price=float("nan"))


def test_security_validation_ticker() -> None:
    """Verify malicious ticker symbols are rejected by security validators."""
    with pytest.raises(SecurityValidationError):
        evaluate_dssw_profile(ticker="INVAL$$ID")


def test_evaluate_dssw_profile_with_prices() -> None:
    """Verify evaluate_dssw_profile functions with explicit price arrays."""
    prices = [100.0 + (5.0 * math.sin(i / 3.0)) for i in range(50)]

    receipt = evaluate_dssw_profile(
        ticker="AAPL",
        prices=prices,
        risk_free_rate=0.045,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.fundamental_price > 0.0
    assert verify_dssw_audit_digest(receipt) is True


def test_persist_dssw_record(tmp_path: Path) -> None:
    """Verify SQLite persistence of DSSW evaluation records."""
    db_file = tmp_path / "test_dssw.db"

    receipt = calculate_dssw_equilibrium(
        fundamental_price=110.0,
        mean_sentiment=0.04,
        sentiment_shock=0.06,
        ticker="TESTDB",
    )

    success = persist_dssw_record(receipt, db_path=db_file)
    assert success is True

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, fundamental_price, dssw_market_price, audit_digest FROM dssw_noise_trader_records")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "TESTDB"
    assert abs(row[1] - 110.0) < 1e-4
    assert abs(row[2] - receipt.dssw_market_price) < 1e-4
    assert row[3] == receipt.audit_digest


def test_cli_dssw_command_and_json(capsys: pytest.CaptureFixture) -> None:
    """Verify dssw-noise CLI handler runs cleanly in human-readable and JSON modes."""
    # Test human readable
    args = argparse.Namespace(
        ticker="AAPL",
        mean_sentiment=0.05,
        shock=0.08,
        variance=0.015,
        risk_aversion=1.25,
        rf=0.0425,
        json=False,
    )
    dssw_command(args)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: De Long, Shleifer, Summers & Waldmann (1990)" in captured
    assert "DSSW Equilibrium Market Price" in captured
    assert "Create-Space Volatility Discount" in captured

    # Test JSON
    args.json = True
    dssw_command(args)
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["ticker"] == "AAPL"
    assert "dssw_market_price" in data
    assert "create_space_discount_dollar" in data
    assert "audit_digest" in data
