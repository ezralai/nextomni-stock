"""Unit, boundary, security, and integration tests for Harrison Hong & Jeremy C. Stein (2003)

Differences of Opinion, Short-Sales Constraints & Market Crashes Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.hong_stein_crash import (
    HongSteinCrashReceipt,
    calculate_hong_stein_crash,
    classify_crash_regime,
    evaluate_hong_stein_crash_profile,
    persist_hong_stein_crash_record,
    verify_hong_stein_audit_digest,
)
from src.cli import hong_stein_crash_command
from src.security.validators import SecurityValidationError


def test_calculate_hong_stein_crash_acute_vulnerability() -> None:
    """Verify high dispersion and strict short constraint trigger acute crash vulnerability regime."""
    receipt = calculate_hong_stein_crash(
        ticker="ACUTE",
        latest_price=85.0,
        opinion_dispersion_ratio=0.28,
        short_sale_constraint_tightness=0.75,
        turnover_ratio=1.50,
        baseline_volatility=0.25,
    )

    assert receipt.ticker == "ACUTE"
    assert receipt.opinion_dispersion_ratio == 0.28
    assert receipt.short_sale_constraint_tightness == 0.75
    assert receipt.is_acute_crash_vulnerability is True
    assert receipt.regime == "ACUTE_CRASH_VULNERABILITY"
    assert receipt.latent_information_overhang > 0.80
    assert receipt.conditional_crash_skewness < -0.80
    assert receipt.crash_vulnerability_index > 30.0
    assert verify_hong_stein_audit_digest(receipt) is True


def test_calculate_hong_stein_crash_elevated_overhang() -> None:
    """Verify moderate dispersion and constraint trigger elevated latent overhang regime."""
    receipt = calculate_hong_stein_crash(
        ticker="ELEVAT",
        latest_price=110.0,
        opinion_dispersion_ratio=0.16,
        short_sale_constraint_tightness=0.45,
    )

    assert receipt.is_elevated_latent_overhang is True
    assert receipt.regime == "ELEVATED_LATENT_OVERHANG"
    assert receipt.conditional_crash_skewness < 0.0
    assert verify_hong_stein_audit_digest(receipt) is True


def test_calculate_hong_stein_crash_symmetric_unconstrained() -> None:
    """Verify low short-sale impediment triggers symmetric unconstrained regime."""
    receipt = calculate_hong_stein_crash(
        ticker="SYMM",
        latest_price=135.0,
        opinion_dispersion_ratio=0.15,
        short_sale_constraint_tightness=0.10,
    )

    assert receipt.is_symmetric_unconstrained is True
    assert receipt.regime == "SYMMETRIC_UNCONSTRAINED"
    assert receipt.latent_information_overhang < 0.25
    assert verify_hong_stein_audit_digest(receipt) is True


def test_calculate_hong_stein_crash_consensus_stability() -> None:
    """Verify low belief dispersion triggers consensus stability regime."""
    receipt = calculate_hong_stein_crash(
        ticker="CONSNS",
        latest_price=160.0,
        opinion_dispersion_ratio=0.03,
        short_sale_constraint_tightness=0.30,
    )

    assert receipt.is_consensus_stability is True
    assert receipt.regime == "CONSENSUS_STABILITY"
    assert receipt.crash_vulnerability_index <= 20.0
    assert verify_hong_stein_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher dispersion/constraint strictly elevates overhang and negative skewness."""
    dispersions = [0.08, 0.14, 0.20, 0.26, 0.32]
    overhangs = []
    skewnesses = []
    indices = []

    for disp in dispersions:
        rec = calculate_hong_stein_crash(
            ticker="MONO",
            opinion_dispersion_ratio=disp,
            short_sale_constraint_tightness=0.60,
        )
        overhangs.append(rec.latent_information_overhang)
        skewnesses.append(rec.conditional_crash_skewness)
        indices.append(rec.crash_vulnerability_index)

    # Higher dispersion -> strictly higher latent overhang
    for i in range(len(overhangs) - 1):
        assert overhangs[i] < overhangs[i + 1]

    # Higher dispersion -> strictly more negative conditional crash skewness
    for i in range(len(skewnesses) - 1):
        assert skewnesses[i] > skewnesses[i + 1]

    # Higher dispersion -> strictly higher crash vulnerability index
    for i in range(len(indices) - 1):
        assert indices[i] < indices[i + 1]


def test_classify_crash_regime() -> None:
    """Verify regime boundaries for crash vulnerability."""
    assert classify_crash_regime(0.25, 0.65, 70.0) == "ACUTE_CRASH_VULNERABILITY"
    assert classify_crash_regime(0.15, 0.45, 40.0) == "ELEVATED_LATENT_OVERHANG"
    assert classify_crash_regime(0.12, 0.12, 15.0) == "SYMMETRIC_UNCONSTRAINED"
    assert classify_crash_regime(0.04, 0.25, 10.0) == "CONSENSUS_STABILITY"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_hong_stein_crash(ticker="AUDIT")
    assert verify_hong_stein_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = HongSteinCrashReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 25.0}
    )
    assert verify_hong_stein_audit_digest(tampered_price) is False

    # Tampered crash skewness
    tampered_skew = HongSteinCrashReceipt(
        **{**receipt.to_dict(), "conditional_crash_skewness": receipt.conditional_crash_skewness - 0.50}
    )
    assert verify_hong_stein_audit_digest(tampered_skew) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_hong_stein_crash(ticker="TOOLONGTICKERNAME")

    with pytest.raises(SecurityValidationError):
        calculate_hong_stein_crash(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_hong_stein_crash(ticker="AAPL", opinion_dispersion_ratio=0.001)

    with pytest.raises(SecurityValidationError):
        calculate_hong_stein_crash(ticker="AAPL", short_sale_constraint_tightness=1.50)


def test_evaluate_hong_stein_crash_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price and volume observations."""
    np.random.seed(42)
    # Generate 120 daily price observations with negative skewness
    base = 100.0
    returns = np.random.normal(0.0005, 0.018, 120)
    returns[40:43] = -0.05  # Negative shocks causing negative skewness
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))
    volumes = list(np.random.uniform(800_000, 2_500_000, 121))

    receipt = evaluate_hong_stein_crash_profile(
        ticker="SYNTH",
        prices=prices,
        volumes=volumes,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.005 <= receipt.opinion_dispersion_ratio <= 1.50
    assert receipt.latent_information_overhang >= 0.0
    assert verify_hong_stein_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table hong_stein_crash_records."""
    db_file = tmp_path / "test_hong_stein.db"
    receipt = calculate_hong_stein_crash(
        ticker="DBTEST",
        latest_price=120.0,
        opinion_dispersion_ratio=0.20,
    )

    persist_hong_stein_crash_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, regime, crash_vulnerability_index FROM hong_stein_crash_records WHERE ticker = ?",
        ("DBTEST",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 120.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.crash_vulnerability_index) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        opinion_dispersion=0.22,
        short_constraint=0.65,
        turnover=1.30,
        json=False,
    )

    hong_stein_crash_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Hong & Stein (2003) Crash Asymmetry Engine" in captured
    assert "AAPL" in captured
    assert "Conditional Crash Skewness:" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="TSLA",
        opinion_dispersion=0.25,
        short_constraint=0.70,
        turnover=2.00,
        json=True,
    )

    hong_stein_crash_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "TSLA"
    assert "conditional_crash_skewness" in payload
    assert "audit_digest" in payload
    assert "regime" in payload

