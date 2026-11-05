"""Unit, boundary, security, and integration tests for Franklin Allen & Douglas Gale (2000)

Financial Contagion & Incomplete Network Cascades Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.allen_gale_network import (
    AllenGaleNetworkReceipt,
    calculate_allen_gale_network,
    classify_network_regime,
    evaluate_allen_gale_network_profile,
    persist_allen_gale_network_record,
    verify_allen_gale_audit_digest,
)
from src.cli import allen_gale_command
from src.security.validators import SecurityValidationError


def test_calculate_allen_gale_network_systemic_cascade() -> None:
    """Verify incomplete sparse network under severe shock triggers systemic cascade collapse regime."""
    receipt = calculate_allen_gale_network(
        ticker="CASCAD",
        latest_price=80.0,
        network_completeness_ratio=0.20,
        local_liquidity_shock_ratio=0.65,
        aggregate_liquidity_buffer=1.00,
        asset_volatility=0.25,
    )

    assert receipt.ticker == "CASCAD"
    assert receipt.network_completeness_ratio == 0.20
    assert receipt.unabsorbed_liquidity_deficit > 0.30
    assert receipt.contagion_cascade_multiplier > 2.0
    assert receipt.is_systemic_cascade is True
    assert receipt.regime == "SYSTEMIC_CASCADE_COLLAPSE"
    assert receipt.network_fire_sale_markdown_pct > 3.0
    assert verify_allen_gale_audit_digest(receipt) is True


def test_calculate_allen_gale_network_incomplete_fragile() -> None:
    """Verify moderate completeness with deficit triggers incomplete fragile network regime."""
    receipt = calculate_allen_gale_network(
        ticker="FRAGIL",
        latest_price=105.0,
        network_completeness_ratio=0.45,
        local_liquidity_shock_ratio=0.70,
        aggregate_liquidity_buffer=1.00,
    )

    assert receipt.is_incomplete_fragile is True
    assert receipt.regime == "INCOMPLETE_FRAGILE_NETWORK"
    assert receipt.unabsorbed_liquidity_deficit > 0.0
    assert verify_allen_gale_audit_digest(receipt) is True


def test_calculate_allen_gale_network_complete_fortress() -> None:
    """Verify highly complete network absorbs shock with zero cascade multiplier."""
    receipt = calculate_allen_gale_network(
        ticker="FORTRS",
        latest_price=150.0,
        network_completeness_ratio=0.92,
        local_liquidity_shock_ratio=0.40,
        aggregate_liquidity_buffer=1.00,
    )

    assert receipt.is_complete_fortress is True
    assert receipt.regime == "COMPLETE_NETWORK_FORTRESS"
    assert receipt.unabsorbed_liquidity_deficit == 0.0
    assert receipt.contagion_cascade_multiplier == 1.00
    assert receipt.network_fire_sale_markdown_pct == 0.0
    assert verify_allen_gale_audit_digest(receipt) is True


def test_calculate_allen_gale_network_resilient_sharing() -> None:
    """Verify strong connectivity with minor unabsorbed deficit triggers resilient partial sharing regime."""
    receipt = calculate_allen_gale_network(
        ticker="RESIL",
        latest_price=125.0,
        network_completeness_ratio=0.75,
        local_liquidity_shock_ratio=0.85,
        aggregate_liquidity_buffer=1.00,
    )

    assert receipt.is_resilient_sharing is True
    assert receipt.regime == "RESILIENT_PARTIAL_SHARING"
    assert verify_allen_gale_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: lower completeness strictly increases cascade multiplier and contagion index."""
    completenesses = [0.25, 0.38, 0.50, 0.62]
    multipliers = []
    indices = []

    for comp in completenesses:
        rec = calculate_allen_gale_network(
            ticker="MONO",
            network_completeness_ratio=comp,
            local_liquidity_shock_ratio=0.70,
            aggregate_liquidity_buffer=1.00,
        )
        multipliers.append(rec.contagion_cascade_multiplier)
        indices.append(rec.systemic_contagion_index)

    # Lower completeness -> strictly higher cascade multiplier
    for i in range(len(multipliers) - 1):
        assert multipliers[i] > multipliers[i + 1]

    # Lower completeness -> strictly higher systemic contagion index
    for i in range(len(indices) - 1):
        assert indices[i] > indices[i + 1]


def test_classify_network_regime() -> None:
    """Verify regime boundaries for network contagion."""
    assert classify_network_regime(0.25, 0.20, 70.0) == "SYSTEMIC_CASCADE_COLLAPSE"
    assert classify_network_regime(0.50, 0.10, 45.0) == "INCOMPLETE_FRAGILE_NETWORK"
    assert classify_network_regime(0.92, 0.00, 10.0) == "COMPLETE_NETWORK_FORTRESS"
    assert classify_network_regime(0.75, 0.05, 25.0) == "RESILIENT_PARTIAL_SHARING"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_allen_gale_network(ticker="AUDIT")
    assert verify_allen_gale_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = AllenGaleNetworkReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 25.0}
    )
    assert verify_allen_gale_audit_digest(tampered_price) is False

    # Tampered cascade multiplier
    tampered_mult = AllenGaleNetworkReceipt(
        **{**receipt.to_dict(), "contagion_cascade_multiplier": receipt.contagion_cascade_multiplier + 0.50}
    )
    assert verify_allen_gale_audit_digest(tampered_mult) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_allen_gale_network(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_allen_gale_network(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_allen_gale_network(ticker="AAPL", network_completeness_ratio=1.50)

    with pytest.raises(SecurityValidationError):
        calculate_allen_gale_network(ticker="AAPL", local_liquidity_shock_ratio=-0.10)


def test_evaluate_allen_gale_network_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 100 daily price observations
    base = 100.0
    returns = np.random.normal(0.0004, 0.016, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_allen_gale_network_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.05 <= receipt.network_completeness_ratio <= 1.00
    assert receipt.contagion_cascade_multiplier >= 1.00
    assert verify_allen_gale_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table allen_gale_network_records."""
    db_file = tmp_path / "test_allen_gale.db"
    receipt = calculate_allen_gale_network(
        ticker="DBTEST",
        latest_price=115.0,
        network_completeness_ratio=0.45,
        local_liquidity_shock_ratio=0.55,
    )

    persist_allen_gale_network_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, contagion_cascade_multiplier FROM allen_gale_network_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 115.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.contagion_cascade_multiplier) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        network_completeness=0.50,
        liquidity_shock=0.40,
        aggregate_liquidity=1.00,
        json=False,
    )

    allen_gale_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Allen & Gale (2000) Financial Network Contagion Engine" in captured
    assert "AAPL" in captured
    assert "Contagion Cascade Multiplier:" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="BAC",
        network_completeness=0.35,
        liquidity_shock=0.60,
        aggregate_liquidity=0.80,
        json=True,
    )

    allen_gale_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "BAC"
    assert "contagion_cascade_multiplier" in payload
    assert "audit_digest" in payload
    assert "regime" in payload
