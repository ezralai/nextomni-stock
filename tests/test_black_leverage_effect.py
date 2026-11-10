"""Unit, boundary, security, and integration tests for Fischer Black (1976)

& Campbell-Hentschel (1992) Financial Leverage Effect & Volatility Asymmetry Engine.
"""

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.black_leverage_effect import (
    BlackLeverageReceipt,
    calculate_black_leverage_effect,
    classify_leverage_effect_regime,
    evaluate_black_leverage_profile,
    persist_black_leverage_record,
    verify_black_leverage_audit_digest,
)
from src.cli import black_leverage_command
from src.security.validators import SecurityValidationError


def test_calculate_black_leverage_acute_amplification() -> None:
    """Verify high debt-to-equity ratio triggers acute leverage amplification regime."""
    receipt = calculate_black_leverage_effect(
        market_price=100.0,
        debt_per_share=180.0,
        asset_volatility=0.15,
        ticker="ACUTELEV",
    )

    assert receipt.ticker == "ACUTELEV"
    assert receipt.debt_to_equity_ratio == 1.80
    assert receipt.leverage_volatility_multiplier == 2.80
    assert receipt.equity_volatility_pct > (2.5 * receipt.asset_volatility_pct)
    assert receipt.is_acute_leverage_amplification is True
    assert receipt.regime == "ACUTE_LEVERAGE_AMPLIFICATION"
    assert verify_black_leverage_audit_digest(receipt) is True


def test_calculate_black_leverage_unlevered_insulation() -> None:
    """Verify minimal debt produces unlevered insulation regime with equity vol ~ asset vol."""
    receipt = calculate_black_leverage_effect(
        market_price=100.0,
        debt_per_share=8.0,
        asset_volatility=0.16,
        ticker="UNLEV",
    )

    assert receipt.debt_to_equity_ratio <= 0.15
    assert receipt.is_unlevered_insulation is True
    assert receipt.regime == "UNLEVERED_INSULATION"
    assert abs(receipt.equity_volatility_pct - receipt.asset_volatility_pct) < 2.0
    assert verify_black_leverage_audit_digest(receipt) is True


def test_calculate_black_leverage_volatility_feedback_cascade() -> None:
    """Verify strong downside return asymmetry and large feedback discount trigger cascade regime."""
    # Synthetic returns with heavy negative skewness
    np.random.seed(42)
    rets = list(np.random.normal(0.002, 0.005, 50)) + list(np.random.normal(-0.025, 0.035, 30))

    receipt = calculate_black_leverage_effect(
        market_price=80.0,
        debt_per_share=75.0,
        asset_volatility=0.20,
        feedback_elasticity=0.45,
        returns=rets,
        ticker="CASCADE",
    )

    assert receipt.volatility_asymmetry_ratio > 1.35
    assert receipt.volatility_feedback_discount_pct > 8.0
    assert receipt.is_volatility_feedback_cascade is True
    assert receipt.regime == "VOLATILITY_FEEDBACK_CASCADE"
    assert verify_black_leverage_audit_digest(receipt) is True


def test_calculate_black_leverage_mild_asymmetry_stability() -> None:
    """Verify standard baseline leverage produces mild asymmetry stability regime."""
    receipt = calculate_black_leverage_effect(
        market_price=100.0,
        debt_per_share=25.0,
        asset_volatility=0.18,
        ticker="MILD",
    )

    assert receipt.is_mild_asymmetry_stability is True
    assert receipt.regime == "MILD_ASYMMETRY_STABILITY"
    assert verify_black_leverage_audit_digest(receipt) is True


def test_leverage_effect_monotonicity_with_price() -> None:
    """Verify Black (1976) core theorem: as stock price drops, equity volatility strictly increases."""
    prices = [150.0, 120.0, 100.0, 80.0, 60.0]
    vols = []
    sensitivities = []

    for p in prices:
        rec = calculate_black_leverage_effect(
            market_price=p,
            debt_per_share=40.0,
            asset_volatility=0.18,
        )
        vols.append(rec.equity_volatility_pct)
        sensitivities.append(rec.volatility_price_sensitivity)

    # Equity volatility must strictly monotonically increase as price drops
    for i in range(len(vols) - 1):
        assert vols[i] < vols[i + 1]

    # Sensitivity d\sigma_E/dP must be strictly negative everywhere
    for s in sensitivities:
        assert s < 0.0


def test_leverage_effect_monotonicity_with_debt() -> None:
    """Verify that holding price fixed, increasing debt monotonically increases equity volatility."""
    debts = [0.0, 20.0, 50.0, 100.0, 150.0]
    vols = []

    for d in debts:
        rec = calculate_black_leverage_effect(
            market_price=100.0,
            debt_per_share=d,
            asset_volatility=0.18,
        )
        vols.append(rec.equity_volatility_pct)

    for i in range(len(vols) - 1):
        assert vols[i] < vols[i + 1]


def test_audit_digest_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature detects tampering in Black leverage receipt."""
    receipt = calculate_black_leverage_effect(
        market_price=100.0,
        debt_per_share=30.0,
        asset_volatility=0.18,
        ticker="AUTHEN",
    )
    assert verify_black_leverage_audit_digest(receipt) is True

    # Tamper with equity volatility
    tampered_dict = receipt.to_dict()
    tampered_dict["equity_volatility_pct"] = 99.99
    tampered_receipt = BlackLeverageReceipt(**tampered_dict)
    assert verify_black_leverage_audit_digest(tampered_receipt) is False

    # Tamper with feedback price
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["feedback_adjusted_price"] = 55.55
    tampered_receipt2 = BlackLeverageReceipt(**tampered_dict2)
    assert verify_black_leverage_audit_digest(tampered_receipt2) is False


def test_invalid_inputs_raise_errors() -> None:
    """Verify invalid numerical inputs raise appropriate errors."""
    with pytest.raises(ValueError):
        calculate_black_leverage_effect(market_price=-10.0)

    with pytest.raises(ValueError):
        calculate_black_leverage_effect(debt_per_share=-5.0)

    with pytest.raises(ValueError):
        calculate_black_leverage_effect(asset_volatility=0.0)

    with pytest.raises(ValueError):
        calculate_black_leverage_effect(feedback_elasticity=-0.1)

    with pytest.raises(ValueError):
        calculate_black_leverage_effect(market_price=float("nan"))


def test_security_validation_ticker() -> None:
    """Verify malicious ticker symbols are rejected by security validators."""
    with pytest.raises(SecurityValidationError):
        evaluate_black_leverage_profile(ticker="INVAL$$ID")


def test_evaluate_black_leverage_profile_with_prices() -> None:
    """Verify evaluate_black_leverage_profile functions with explicit price arrays."""
    prices = [100.0 * (1.001 ** i) for i in range(50)]

    receipt = evaluate_black_leverage_profile(
        ticker="AAPL",
        prices=prices,
        debt_per_share_override=20.0,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.market_price == round(prices[-1], 4)
    assert verify_black_leverage_audit_digest(receipt) is True


def test_persist_black_leverage_record(tmp_path: Path) -> None:
    """Verify SQLite persistence of Black leverage evaluation records."""
    db_file = tmp_path / "test_black_leverage.db"

    receipt = calculate_black_leverage_effect(
        market_price=105.0,
        debt_per_share=35.0,
        asset_volatility=0.18,
        ticker="TESTDB",
    )

    success = persist_black_leverage_record(receipt, db_path=db_file)
    assert success is True

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, market_price, equity_volatility_pct, audit_digest FROM black_leverage_records")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "TESTDB"
    assert abs(row[1] - 105.0) < 1e-4
    assert abs(row[2] - receipt.equity_volatility_pct) < 1e-4
    assert row[3] == receipt.audit_digest


def test_cli_black_leverage_command_and_json(capsys: pytest.CaptureFixture) -> None:
    """Verify black-leverage CLI handler runs cleanly in human-readable and JSON modes."""
    # Test human readable
    args = argparse.Namespace(
        ticker="AAPL",
        debt=25.0,
        asset_vol=0.18,
        feedback=0.25,
        json=False,
    )
    black_leverage_command(args)
    captured = capsys.readouterr().out
    assert "NextOmni Stock: Black (1976) & Campbell-Hentschel (1992)" in captured
    assert "Equity Volatility" in captured
    assert "Volatility Asymmetry Ratio (VAR)" in captured

    # Test JSON
    args.json = True
    black_leverage_command(args)
    captured_json = capsys.readouterr().out
    data = json.loads(captured_json)
    assert data["ticker"] == "AAPL"
    assert "equity_volatility_pct" in data
    assert "volatility_asymmetry_ratio" in data
    assert "audit_digest" in data
