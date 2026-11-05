"""Unit, boundary, security, and integration tests for Douglas T. Breeden (1979)

Consumption-Based Capital Asset Pricing Model (CCAPM) Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.breeden_ccapm import (
    BreedenCCAPMReceipt,
    calculate_breeden_ccapm,
    classify_ccapm_regime,
    evaluate_breeden_ccapm_profile,
    persist_breeden_ccapm_record,
    verify_breeden_audit_digest,
)
from src.cli import breeden_ccapm_command
from src.security.validators import SecurityValidationError


def test_calculate_breeden_ccapm_high_exposure() -> None:
    """Verify high consumption beta triggers high consumption exposure regime."""
    receipt = calculate_breeden_ccapm(
        ticker="HIGHEX",
        latest_price=95.0,
        consumption_beta=2.20,
        consumption_risk_premium=0.0450,
        risk_free_rate=0.0425,
    )

    assert receipt.ticker == "HIGHEX"
    assert receipt.consumption_beta == 2.20
    assert receipt.is_high_consumption_exposure is True
    assert receipt.regime == "HIGH_CONSUMPTION_EXPOSURE"
    assert receipt.ccapm_required_return_pct >= 14.0
    assert receipt.intertemporal_hedge_elasticity > 2.0
    assert verify_breeden_audit_digest(receipt) is True


def test_calculate_breeden_ccapm_moderate_procyclical() -> None:
    """Verify normal procyclical beta triggers moderate procyclical consumption regime."""
    receipt = calculate_breeden_ccapm(
        ticker="MODPRO",
        latest_price=110.0,
        consumption_beta=1.20,
    )

    assert receipt.is_moderate_procyclical is True
    assert receipt.regime == "MODERATE_PROCYCLICAL_CONSUMPTION"
    assert verify_breeden_audit_digest(receipt) is True


def test_calculate_breeden_ccapm_defensive_hedge() -> None:
    """Verify low beta triggers defensive consumption hedge regime."""
    receipt = calculate_breeden_ccapm(
        ticker="DEFHED",
        latest_price=130.0,
        consumption_beta=0.45,
    )

    assert receipt.is_defensive_hedge is True
    assert receipt.regime == "DEFENSIVE_CONSUMPTION_HEDGE"
    assert receipt.ccapm_required_return_pct < receipt.capm_benchmark_return_pct
    assert verify_breeden_audit_digest(receipt) is True


def test_calculate_breeden_ccapm_countercyclical() -> None:
    """Verify negative beta triggers countercyclical asset regime."""
    receipt = calculate_breeden_ccapm(
        ticker="COUNTR",
        latest_price=160.0,
        consumption_beta=-0.50,
        risk_free_rate=0.0425,
    )

    assert receipt.is_countercyclical_asset is True
    assert receipt.regime == "COUNTERCYCLICAL_ASSET"
    assert receipt.ccapm_required_return_pct < 4.25
    assert verify_breeden_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher consumption beta strictly increases CCAPM hurdle and hedge elasticity."""
    betas = [0.20, 0.60, 1.00, 1.50, 2.10]
    returns = []
    elasticities = []

    for b in betas:
        rec = calculate_breeden_ccapm(
            ticker="MONO",
            consumption_beta=b,
            consumption_risk_premium=0.0400,
        )
        returns.append(rec.ccapm_required_return_pct)
        elasticities.append(rec.intertemporal_hedge_elasticity)

    # Higher beta_C -> strictly higher required return
    for i in range(len(returns) - 1):
        assert returns[i] < returns[i + 1]

    # Higher beta_C -> strictly higher hedge elasticity
    for i in range(len(elasticities) - 1):
        assert elasticities[i] < elasticities[i + 1]


def test_classify_ccapm_regime() -> None:
    """Verify regime boundaries for consumption asset pricing."""
    assert classify_ccapm_regime(-0.20, 3.5) == "COUNTERCYCLICAL_ASSET"
    assert classify_ccapm_regime(1.80, 12.0) == "HIGH_CONSUMPTION_EXPOSURE"
    assert classify_ccapm_regime(1.10, 8.5) == "MODERATE_PROCYCLICAL_CONSUMPTION"
    assert classify_ccapm_regime(0.50, 6.0) == "DEFENSIVE_CONSUMPTION_HEDGE"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_breeden_ccapm(ticker="AUDIT")
    assert verify_breeden_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = BreedenCCAPMReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 30.0}
    )
    assert verify_breeden_audit_digest(tampered_price) is False

    # Tampered consumption beta
    tampered_beta = BreedenCCAPMReceipt(
        **{**receipt.to_dict(), "consumption_beta": receipt.consumption_beta + 0.50}
    )
    assert verify_breeden_audit_digest(tampered_beta) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_breeden_ccapm(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_breeden_ccapm(ticker="AAPL", latest_price=-8.0)

    with pytest.raises(SecurityValidationError):
        calculate_breeden_ccapm(ticker="AAPL", consumption_beta=10.0)

    with pytest.raises(SecurityValidationError):
        calculate_breeden_ccapm(ticker="AAPL", consumption_growth_std=0.35)


def test_evaluate_breeden_ccapm_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 100 daily price observations
    base = 100.0
    returns = np.random.normal(0.0004, 0.016, 100)
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_breeden_ccapm_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert -3.0 <= receipt.consumption_beta <= 6.0
    assert receipt.ccapm_required_return_pct > 0.0
    assert verify_breeden_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table breeden_ccapm_records."""
    db_file = tmp_path / "test_breeden.db"
    receipt = calculate_breeden_ccapm(
        ticker="DBTEST",
        latest_price=125.0,
        consumption_beta=1.35,
    )

    persist_breeden_ccapm_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, consumption_beta FROM breeden_ccapm_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 125.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.consumption_beta) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        consumption_beta=1.25,
        consumption_premium=0.040,
        consumption_vol=0.016,
        json=False,
    )

    breeden_ccapm_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Breeden (1979) Consumption CAPM Engine" in captured
    assert "AAPL" in captured
    assert "Consumption Beta (beta_C):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="MSFT",
        consumption_beta=1.05,
        consumption_premium=0.038,
        consumption_vol=0.015,
        json=True,
    )

    breeden_ccapm_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "MSFT"
    assert "consumption_beta" in payload
    assert "audit_digest" in payload
    assert "regime" in payload

