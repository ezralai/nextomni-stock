"""Unit, boundary, security, and integration tests for Markus K. Brunnermeier & Lasse Heje Pedersen (2005)

Predatory Trading & Strategic Liquidation Overshoot Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.predatory_trading import (
    PredatoryTradingReceipt,
    calculate_predatory_trading,
    classify_predatory_regime,
    evaluate_predatory_trading_profile,
    persist_predatory_trading_record,
    verify_predatory_audit_digest,
)
from src.cli import predatory_trading_command
from src.security.validators import SecurityValidationError


def test_calculate_predatory_trading_aggressive_run() -> None:
    """Verify large distressed liquidation with high predator share triggers aggressive predatory run."""
    receipt = calculate_predatory_trading(
        ticker="AGPRD",
        latest_price=80.0,
        distressed_liquidation_size_pct_adv=5.50,
        predator_concentration_ratio=0.65,
        market_depth_multiplier=0.90,
    )

    assert receipt.ticker == "AGPRD"
    assert receipt.distressed_liquidation_size_pct_adv == 5.50
    assert receipt.is_aggressive_predatory_run is True
    assert receipt.regime == "AGGRESSIVE_PREDATORY_RUN"
    assert receipt.price_overshoot_ratio >= 1.45
    assert receipt.predatory_liquidity_drain_pct > 50.0
    assert receipt.distressed_slippage_inflation_pct > 35.0
    assert verify_predatory_audit_digest(receipt) is True


def test_calculate_predatory_trading_coordinated_front_running() -> None:
    """Verify moderate distressed liquidation triggers coordinated front-running regime."""
    receipt = calculate_predatory_trading(
        ticker="COORDF",
        latest_price=105.0,
        distressed_liquidation_size_pct_adv=2.80,
        predator_concentration_ratio=0.45,
    )

    assert receipt.is_coordinated_front_running is True
    assert receipt.regime == "COORDINATED_FRONT_RUNNING"
    assert receipt.price_overshoot_ratio >= 1.20
    assert verify_predatory_audit_digest(receipt) is True


def test_calculate_predatory_trading_competitive_absorption() -> None:
    """Verify dispersed strategic order flow triggers competitive absorption regime."""
    receipt = calculate_predatory_trading(
        ticker="CMPABS",
        latest_price=140.0,
        distressed_liquidation_size_pct_adv=2.50,
        predator_concentration_ratio=0.15,
    )

    assert receipt.is_competitive_absorption is True
    assert receipt.regime == "COMPETITIVE_ABSORPTION"
    assert receipt.predatory_shorting_volume_pct_adv < 1.0
    assert verify_predatory_audit_digest(receipt) is True


def test_calculate_predatory_trading_benign_liquidity() -> None:
    """Verify minimal distressed liquidation triggers benign liquidity provision regime."""
    receipt = calculate_predatory_trading(
        ticker="BENIGN",
        latest_price=120.0,
        distressed_liquidation_size_pct_adv=0.60,
        predator_concentration_ratio=0.20,
    )

    assert receipt.is_benign_liquidity is True
    assert receipt.regime == "BENIGN_LIQUIDITY_PROVISION"
    assert receipt.distressed_slippage_inflation_pct <= 20.0
    assert verify_predatory_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher predator share strictly increases overshoot and slippage."""
    predator_ratios = [0.15, 0.35, 0.55, 0.75, 0.90]
    overshoots = []
    slippages = []
    short_volumes = []

    for pr in predator_ratios:
        rec = calculate_predatory_trading(
            ticker="MONO",
            distressed_liquidation_size_pct_adv=3.50,
            predator_concentration_ratio=pr,
        )
        overshoots.append(rec.price_overshoot_ratio)
        slippages.append(rec.distressed_slippage_inflation_pct)
        short_volumes.append(rec.predatory_shorting_volume_pct_adv)

    # Higher predator share -> strictly higher predatory short volume
    for i in range(len(short_volumes) - 1):
        assert short_volumes[i] < short_volumes[i + 1]

    # Higher predator share -> strictly higher price overshoot
    for i in range(len(overshoots) - 1):
        assert overshoots[i] < overshoots[i + 1]

    # Higher price overshoot -> strictly higher distressed slippage inflation
    for i in range(len(slippages) - 1):
        assert slippages[i] < slippages[i + 1]


def test_classify_predatory_regime() -> None:
    """Verify regime boundaries for predatory trading."""
    assert classify_predatory_regime(5.0, 0.65, 1.50) == "AGGRESSIVE_PREDATORY_RUN"
    assert classify_predatory_regime(2.5, 0.40, 1.25) == "COORDINATED_FRONT_RUNNING"
    assert classify_predatory_regime(2.5, 0.15, 1.10) == "COMPETITIVE_ABSORPTION"
    assert classify_predatory_regime(0.5, 0.20, 1.05) == "BENIGN_LIQUIDITY_PROVISION"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_predatory_trading(ticker="AUDIT")
    assert verify_predatory_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = PredatoryTradingReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 20.0}
    )
    assert verify_predatory_audit_digest(tampered_price) is False

    # Tampered price overshoot
    tampered_overshoot = PredatoryTradingReceipt(
        **{**receipt.to_dict(), "price_overshoot_ratio": receipt.price_overshoot_ratio + 0.30}
    )
    assert verify_predatory_audit_digest(tampered_overshoot) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_predatory_trading(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_predatory_trading(ticker="AAPL", latest_price=-15.0)

    with pytest.raises(SecurityValidationError):
        calculate_predatory_trading(ticker="AAPL", distressed_liquidation_size_pct_adv=0.01)

    with pytest.raises(SecurityValidationError):
        calculate_predatory_trading(ticker="AAPL", predator_concentration_ratio=0.99)


def test_evaluate_predatory_trading_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price and volume observations."""
    np.random.seed(42)
    # Generate 100 daily price observations with elevated volatility
    base = 95.0
    returns = np.random.normal(0.0004, 0.022, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))
    volumes = list(np.random.uniform(500_000, 2_000_000, 101))

    receipt = evaluate_predatory_trading_profile(
        ticker="SYNTH",
        prices=prices,
        volumes=volumes,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.10 <= receipt.distressed_liquidation_size_pct_adv <= 30.0
    assert receipt.price_overshoot_ratio >= 1.00
    assert verify_predatory_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table predatory_trading_records."""
    db_file = tmp_path / "test_predatory_trading.db"
    receipt = calculate_predatory_trading(
        ticker="DBTEST",
        latest_price=115.0,
        distressed_liquidation_size_pct_adv=3.80,
    )

    persist_predatory_trading_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, price_overshoot_ratio FROM predatory_trading_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 115.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.price_overshoot_ratio) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        distressed_size=4.0,
        predator_concentration=0.60,
        market_depth=1.10,
        json=False,
    )

    predatory_trading_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Brunnermeier & Pedersen (2005) Predatory Trading Engine" in captured
    assert "AAPL" in captured
    assert "Predatory Price Overshoot Ratio:" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="AMD",
        distressed_size=3.0,
        predator_concentration=0.50,
        market_depth=1.00,
        json=True,
    )

    predatory_trading_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "AMD"
    assert "price_overshoot_ratio" in payload
    assert "audit_digest" in payload
    assert "regime" in payload
