"""Unit, boundary, security, and integration tests for John Geanakoplos (2010)

The Leverage Cycle, Collateral Margins & De-leveraging Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.geanakoplos_leverage_cycle import (
    GeanakoplosLeverageReceipt,
    calculate_geanakoplos_leverage,
    classify_leverage_regime,
    evaluate_geanakoplos_leverage_profile,
    persist_geanakoplos_leverage_record,
    verify_leverage_audit_digest,
)
from src.cli import leverage_cycle_command
from src.security.validators import SecurityValidationError


def test_calculate_geanakoplos_leverage_acute_margin_squeeze() -> None:
    """Verify severe haircut tightening triggers acute margin squeeze panic regime."""
    receipt = calculate_geanakoplos_leverage(
        ticker="ACUTE",
        latest_price=85.0,
        collateral_haircut_m=0.50,
        baseline_haircut_m0=0.18,
        asset_volatility=0.25,
    )

    assert receipt.ticker == "ACUTE"
    assert receipt.collateral_haircut_m == 0.50
    assert receipt.haircut_tightening_ratio > 2.5
    assert receipt.forced_deleveraging_pressure > 30.0
    assert receipt.is_acute_margin_squeeze is True
    assert receipt.regime == "ACUTE_MARGIN_SQUEEZE_PANIC"
    assert receipt.collateral_squeeze_markdown_pct > 1.0
    assert verify_leverage_audit_digest(receipt) is True


def test_calculate_geanakoplos_leverage_elevated_deleveraging() -> None:
    """Verify moderate haircut tightening triggers elevated de-leveraging phase regime."""
    receipt = calculate_geanakoplos_leverage(
        ticker="ELEVAT",
        latest_price=105.0,
        collateral_haircut_m=0.35,
        baseline_haircut_m0=0.20,
    )

    assert receipt.is_elevated_deleveraging is True
    assert receipt.regime == "ELEVATED_DELEVERAGING_PHASE"
    assert receipt.forced_deleveraging_pressure > 0.0
    assert verify_leverage_audit_digest(receipt) is True


def test_calculate_geanakoplos_leverage_euphoric_expansion() -> None:
    """Verify loose collateral requirements trigger euphoric high leverage expansion regime."""
    receipt = calculate_geanakoplos_leverage(
        ticker="EUPHOR",
        latest_price=160.0,
        collateral_haircut_m=0.08,
        baseline_haircut_m0=0.08,
    )

    assert receipt.is_euphoric_expansion is True
    assert receipt.regime == "EUPHORIC_HIGH_LEVERAGE_EXPANSION"
    assert receipt.effective_leverage_ratio >= 10.0
    assert receipt.forced_deleveraging_pressure == 0.0
    assert verify_leverage_audit_digest(receipt) is True


def test_calculate_geanakoplos_leverage_prudent_stability() -> None:
    """Verify normal baseline margin requirement triggers prudent stability regime."""
    receipt = calculate_geanakoplos_leverage(
        ticker="PRUDNT",
        latest_price=120.0,
        collateral_haircut_m=0.22,
        baseline_haircut_m0=0.22,
    )

    assert receipt.is_prudent_stability is True
    assert receipt.regime == "PRUDENT_LOW_LEVERAGE_STABILITY"
    assert receipt.forced_deleveraging_pressure == 0.0
    assert verify_leverage_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher haircut strictly lowers leverage, while tighter haircut increases pressure."""
    # 1. Leverage decreasing in haircut
    haircuts = [0.08, 0.15, 0.25, 0.40, 0.60]
    leverages = []
    for m in haircuts:
        rec = calculate_geanakoplos_leverage(
            ticker="MONO",
            collateral_haircut_m=m,
            baseline_haircut_m0=0.20,
        )
        leverages.append(rec.effective_leverage_ratio)

    for i in range(len(leverages) - 1):
        assert leverages[i] > leverages[i + 1]

    # 2. De-leveraging pressure strictly increasing as haircut expands above baseline
    squeeze_haircuts = [0.22, 0.28, 0.35, 0.45, 0.60]
    pressures = []
    markdowns = []
    for m in squeeze_haircuts:
        rec = calculate_geanakoplos_leverage(
            ticker="MONO",
            collateral_haircut_m=m,
            baseline_haircut_m0=0.20,
            asset_volatility=0.22,
        )
        pressures.append(rec.forced_deleveraging_pressure)
        markdowns.append(rec.collateral_squeeze_markdown_pct)

    for i in range(len(pressures) - 1):
        assert pressures[i] < pressures[i + 1]

    for i in range(len(markdowns) - 1):
        assert markdowns[i] < markdowns[i + 1]


def test_classify_leverage_regime() -> None:
    """Verify regime boundaries for the leverage cycle."""
    assert classify_leverage_regime(0.50, 70.0) == "ACUTE_MARGIN_SQUEEZE_PANIC"
    assert classify_leverage_regime(0.32, 40.0) == "ELEVATED_DELEVERAGING_PHASE"
    assert classify_leverage_regime(0.10, 5.0) == "EUPHORIC_HIGH_LEVERAGE_EXPANSION"
    assert classify_leverage_regime(0.20, 0.0) == "PRUDENT_LOW_LEVERAGE_STABILITY"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_geanakoplos_leverage(ticker="AUDIT")
    assert verify_leverage_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = GeanakoplosLeverageReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 20.0}
    )
    assert verify_leverage_audit_digest(tampered_price) is False

    # Tampered haircut
    tampered_haircut = GeanakoplosLeverageReceipt(
        **{**receipt.to_dict(), "collateral_haircut_m": receipt.collateral_haircut_m + 0.15}
    )
    assert verify_leverage_audit_digest(tampered_haircut) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_geanakoplos_leverage(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_geanakoplos_leverage(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_geanakoplos_leverage(ticker="AAPL", collateral_haircut_m=0.01)

    with pytest.raises(SecurityValidationError):
        calculate_geanakoplos_leverage(ticker="AAPL", baseline_haircut_m0=0.95)


def test_evaluate_geanakoplos_leverage_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 100 daily price observations
    base = 100.0
    returns = np.random.normal(0.0004, 0.018, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_geanakoplos_leverage_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.04 <= receipt.collateral_haircut_m <= 0.90
    assert receipt.effective_leverage_ratio >= 1.10
    assert verify_leverage_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table geanakoplos_leverage_records."""
    db_file = tmp_path / "test_geanakoplos.db"
    receipt = calculate_geanakoplos_leverage(
        ticker="DBTEST",
        latest_price=110.0,
        collateral_haircut_m=0.30,
        baseline_haircut_m0=0.20,
    )

    persist_geanakoplos_leverage_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, collateral_haircut_m FROM geanakoplos_leverage_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 110.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.collateral_haircut_m) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        haircut=0.35,
        baseline_haircut=0.20,
        volatility=0.25,
        json=False,
    )

    leverage_cycle_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Geanakoplos (2010) Leverage Cycle Engine" in captured
    assert "AAPL" in captured
    assert "Collateral Margin Haircut (m):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="JPM",
        haircut=0.25,
        baseline_haircut=0.20,
        volatility=0.22,
        json=True,
    )

    leverage_cycle_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "JPM"
    assert "collateral_haircut_m" in payload
    assert "audit_digest" in payload
    assert "regime" in payload
