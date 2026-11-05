"""Unit, boundary, security, and integration tests for Nicholas Barberis, Ming Huang & Tano Santos (2001)

Prospect Theory, Dynamic Loss Aversion & Prior Gain Cushion Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.barberis_huang_santos import (
    BHSProspectReceipt,
    calculate_bhs_prospect,
    classify_prospect_regime,
    evaluate_bhs_prospect_profile,
    persist_bhs_prospect_record,
    verify_bhs_audit_digest,
)
from src.cli import bhs_prospect_command
from src.security.validators import SecurityValidationError


def test_calculate_bhs_prospect_acute_loss_aversion() -> None:
    """Verify deep prior loss triggers acute loss aversion panic regime."""
    receipt = calculate_bhs_prospect(
        ticker="ACUTE",
        latest_price=70.0,
        reference_price=100.0,
        baseline_loss_aversion=2.25,
        asset_volatility=0.25,
    )

    assert receipt.ticker == "ACUTE"
    assert receipt.prior_cushion_state_z > 1.30
    assert receipt.dynamic_loss_aversion_lambda >= 3.50
    assert receipt.is_acute_loss_aversion is True
    assert receipt.regime == "ACUTE_LOSS_AVERSION_PANIC"
    assert receipt.prospective_loss_surcharge_pct > 2.0
    assert receipt.total_prospect_hurdle_pct > receipt.capm_benchmark_return_pct
    assert verify_bhs_audit_digest(receipt) is True


def test_calculate_bhs_prospect_moderate_sensitivity() -> None:
    """Verify moderate prior loss triggers moderate recession sensitivity regime."""
    receipt = calculate_bhs_prospect(
        ticker="MODER",
        latest_price=90.0,
        reference_price=100.0,
    )

    assert receipt.is_moderate_sensitivity is True
    assert receipt.regime == "MODERATE_RECESSION_SENSITIVITY"
    assert receipt.dynamic_loss_aversion_lambda > 2.25
    assert verify_bhs_audit_digest(receipt) is True


def test_calculate_bhs_prospect_house_money_cushion() -> None:
    """Verify strong prior gains trigger house money cushion regime."""
    receipt = calculate_bhs_prospect(
        ticker="CUSHN",
        latest_price=135.0,
        reference_price=100.0,
    )

    assert receipt.is_house_money_cushion is True
    assert receipt.regime == "HOUSE_MONEY_CUSHION"
    assert receipt.house_money_cushion_pct > 15.0
    assert receipt.dynamic_loss_aversion_lambda < 2.25
    assert verify_bhs_audit_digest(receipt) is True


def test_calculate_bhs_prospect_benign_neutral() -> None:
    """Verify baseline near-zero prior return triggers benign neutral prospect regime."""
    receipt = calculate_bhs_prospect(
        ticker="NEUTR",
        latest_price=100.0,
        reference_price=100.0,
    )

    assert receipt.is_benign_neutral is True
    assert receipt.regime == "BENIGN_NEUTRAL_PROSPECT"
    assert receipt.dynamic_loss_aversion_lambda == 2.25
    assert verify_bhs_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher z_state strictly increases loss aversion and surcharge."""
    reference_prices = [80.0, 95.0, 100.0, 115.0, 135.0]
    lambdas = []
    surcharges = []
    hurdles = []

    for ref in reference_prices:
        rec = calculate_bhs_prospect(
            ticker="MONO",
            latest_price=100.0,
            reference_price=ref,
            asset_volatility=0.20,
        )
        lambdas.append(rec.dynamic_loss_aversion_lambda)
        surcharges.append(rec.prospective_loss_surcharge_pct)
        hurdles.append(rec.total_prospect_hurdle_pct)

    # Higher reference price (worse prior loss / higher z) -> strictly higher lambda
    for i in range(len(lambdas) - 1):
        assert lambdas[i] < lambdas[i + 1]

    # Higher z -> strictly higher prospective loss surcharge
    for i in range(len(surcharges) - 1):
        assert surcharges[i] < surcharges[i + 1]

    # Higher z -> strictly higher total prospect hurdle rate
    for i in range(len(hurdles) - 1):
        assert hurdles[i] < hurdles[i + 1]


def test_classify_prospect_regime() -> None:
    """Verify regime boundaries for prospect theory."""
    assert classify_prospect_regime(1.40, 5.0) == "ACUTE_LOSS_AVERSION_PANIC"
    assert classify_prospect_regime(1.15, 3.0) == "MODERATE_RECESSION_SENSITIVITY"
    assert classify_prospect_regime(0.75, 1.8) == "HOUSE_MONEY_CUSHION"
    assert classify_prospect_regime(1.00, 2.25) == "BENIGN_NEUTRAL_PROSPECT"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_bhs_prospect(ticker="AUDIT")
    assert verify_bhs_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = BHSProspectReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 20.0}
    )
    assert verify_bhs_audit_digest(tampered_price) is False

    # Tampered dynamic lambda
    tampered_lambda = BHSProspectReceipt(
        **{**receipt.to_dict(), "dynamic_loss_aversion_lambda": receipt.dynamic_loss_aversion_lambda + 0.80}
    )
    assert verify_bhs_audit_digest(tampered_lambda) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_bhs_prospect(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_bhs_prospect(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_bhs_prospect(ticker="AAPL", reference_price=0.0)

    with pytest.raises(SecurityValidationError):
        calculate_bhs_prospect(ticker="AAPL", baseline_loss_aversion=8.0)


def test_evaluate_bhs_prospect_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 100 daily price observations
    base = 100.0
    returns = np.random.normal(0.0003, 0.015, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_bhs_prospect_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert receipt.reference_price > 0.0
    assert receipt.dynamic_loss_aversion_lambda >= 1.0
    assert verify_bhs_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table barberis_huang_santos_records."""
    db_file = tmp_path / "test_bhs.db"
    receipt = calculate_bhs_prospect(
        ticker="DBTEST",
        latest_price=95.0,
        reference_price=100.0,
    )

    persist_bhs_prospect_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, dynamic_loss_aversion_lambda FROM barberis_huang_santos_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 95.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.dynamic_loss_aversion_lambda) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        reference_price=90.0,
        lambda_zero=2.25,
        json=False,
    )

    bhs_prospect_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Barberis, Huang & Santos (2001) Prospect Theory Engine" in captured
    assert "AAPL" in captured
    assert "Dynamic Loss Aversion (lambda):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="GOOGL",
        reference_price=120.0,
        lambda_zero=2.50,
        json=True,
    )

    bhs_prospect_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "GOOGL"
    assert "dynamic_loss_aversion_lambda" in payload
    assert "audit_digest" in payload
    assert "regime" in payload

