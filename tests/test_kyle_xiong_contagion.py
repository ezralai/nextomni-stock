"""Unit, boundary, security, and integration tests for Albert S. Kyle & Wei Xiong (2001)

Contagion as a Wealth Effect & Forced Liquidation Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.kyle_xiong_contagion import (
    KyleXiongReceipt,
    calculate_kyle_xiong_contagion,
    classify_contagion_regime,
    evaluate_kyle_xiong_profile,
    persist_kyle_xiong_record,
    verify_kyle_xiong_audit_digest,
)
from src.cli import kyle_xiong_command
from src.security.validators import SecurityValidationError


def test_calculate_kyle_xiong_systemic_contagion() -> None:
    """Verify severe wealth loss triggers systemic liquidation contagion regime."""
    receipt = calculate_kyle_xiong_contagion(
        ticker="SYSCON",
        latest_price=60.0,
        arbitrageur_wealth_ratio=0.45,  # 55% wealth drawdown
        risk_aversion_gamma=3.0,
        fundamental_volatility=0.25,
    )

    assert receipt.ticker == "SYSCON"
    assert receipt.wealth_drawdown_pct == 55.0
    assert receipt.is_systemic_contagion is True
    assert receipt.regime == "SYSTEMIC_LIQUIDATION_CONTAGION"
    assert receipt.contagion_spillover_beta > 0.40
    assert receipt.non_fundamental_excess_corr > 0.30
    assert receipt.forced_liquidation_pressure > 50.0
    assert verify_kyle_xiong_audit_digest(receipt) is True


def test_calculate_kyle_xiong_moderate_spillover() -> None:
    """Verify moderate wealth loss triggers moderate wealth spillover regime."""
    receipt = calculate_kyle_xiong_contagion(
        ticker="MODSPI",
        latest_price=90.0,
        arbitrageur_wealth_ratio=0.75,  # 25% wealth drawdown
        risk_aversion_gamma=2.5,
    )

    assert receipt.is_moderate_spillover is True
    assert receipt.regime == "MODERATE_WEALTH_SPILLOVER"
    assert verify_kyle_xiong_audit_digest(receipt) is True


def test_calculate_kyle_xiong_capital_surplus() -> None:
    """Verify elevated wealth triggers arbitrage capital surplus regime."""
    receipt = calculate_kyle_xiong_contagion(
        ticker="SURPLS",
        latest_price=150.0,
        arbitrageur_wealth_ratio=1.35,  # Surplus capital
        risk_aversion_gamma=2.0,
    )

    assert receipt.is_capital_surplus is True
    assert receipt.regime == "ARBITRAGE_CAPITAL_SURPLUS"
    assert receipt.wealth_drawdown_pct == 0.0
    assert receipt.forced_liquidation_pressure <= 15.0
    assert verify_kyle_xiong_audit_digest(receipt) is True


def test_calculate_kyle_xiong_benign_decoupled() -> None:
    """Verify minor drawdown triggers benign decoupled regime."""
    receipt = calculate_kyle_xiong_contagion(
        ticker="BENIGN",
        latest_price=105.0,
        arbitrageur_wealth_ratio=0.92,  # 8% wealth drawdown
    )

    assert receipt.is_benign_decoupled is True
    assert receipt.regime == "BENIGN_DECOUPLED"
    assert verify_kyle_xiong_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: lower wealth contracts risk capacity and elevates spillover."""
    wealth_ratios = [0.40, 0.60, 0.80, 1.00, 1.30]
    capacities = []
    spillovers = []
    excess_corrs = []

    for w in wealth_ratios:
        rec = calculate_kyle_xiong_contagion(
            ticker="MONO",
            arbitrageur_wealth_ratio=w,
            risk_aversion_gamma=2.5,
            fundamental_volatility=0.20,
        )
        capacities.append(rec.risk_bearing_capacity)
        spillovers.append(rec.contagion_spillover_beta)
        excess_corrs.append(rec.non_fundamental_excess_corr)

    # Higher wealth -> strictly higher risk capacity kappa
    for i in range(len(capacities) - 1):
        assert capacities[i] < capacities[i + 1]

    # Higher wealth -> strictly lower contagion spillover beta
    for i in range(len(spillovers) - 1):
        assert spillovers[i] > spillovers[i + 1]

    # Higher wealth -> strictly lower non-fundamental excess correlation
    for i in range(len(excess_corrs) - 1):
        assert excess_corrs[i] > excess_corrs[i + 1]


def test_classify_contagion_regime() -> None:
    """Verify regime boundaries for wealth contagion."""
    assert classify_contagion_regime(45.0, 70.0) == "SYSTEMIC_LIQUIDATION_CONTAGION"
    assert classify_contagion_regime(25.0, 40.0) == "MODERATE_WEALTH_SPILLOVER"
    assert classify_contagion_regime(0.0, 10.0) == "ARBITRAGE_CAPITAL_SURPLUS"
    assert classify_contagion_regime(12.0, 20.0) == "BENIGN_DECOUPLED"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_kyle_xiong_contagion(ticker="AUDIT")
    assert verify_kyle_xiong_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = KyleXiongReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 15.0}
    )
    assert verify_kyle_xiong_audit_digest(tampered_price) is False

    # Tampered spillover beta
    tampered_beta = KyleXiongReceipt(
        **{**receipt.to_dict(), "contagion_spillover_beta": receipt.contagion_spillover_beta + 0.20}
    )
    assert verify_kyle_xiong_audit_digest(tampered_beta) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_kyle_xiong_contagion(ticker="INVALIDTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_kyle_xiong_contagion(ticker="AAPL", latest_price=-5.0)

    with pytest.raises(SecurityValidationError):
        calculate_kyle_xiong_contagion(ticker="AAPL", arbitrageur_wealth_ratio=0.01)

    with pytest.raises(SecurityValidationError):
        calculate_kyle_xiong_contagion(ticker="AAPL", risk_aversion_gamma=30.0)


def test_evaluate_kyle_xiong_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 120 daily price observations with an end-of-series pullback
    base = 120.0
    returns = np.random.normal(0.0003, 0.016, 120)
    returns[-10:] = -0.025  # Drawdown at the end
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_kyle_xiong_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.05 <= receipt.arbitrageur_wealth_ratio <= 3.00
    assert receipt.risk_bearing_capacity > 0.0
    assert verify_kyle_xiong_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table kyle_xiong_contagion_records."""
    db_file = tmp_path / "test_kyle_xiong.db"
    receipt = calculate_kyle_xiong_contagion(
        ticker="DBTEST",
        latest_price=85.0,
        arbitrageur_wealth_ratio=0.80,
    )

    persist_kyle_xiong_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, contagion_spillover_beta FROM kyle_xiong_contagion_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 85.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.contagion_spillover_beta) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        wealth_drawdown=25.0,
        volatility=0.22,
        gamma=2.5,
        json=False,
    )

    kyle_xiong_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Kyle & Xiong (2001) Wealth Contagion Engine" in captured
    assert "AAPL" in captured
    assert "Contagion Spillover Beta:" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="NVDA",
        wealth_drawdown=15.0,
        volatility=0.35,
        gamma=3.0,
        json=True,
    )

    kyle_xiong_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "NVDA"
    assert "contagion_spillover_beta" in payload
    assert "audit_digest" in payload
    assert "regime" in payload

