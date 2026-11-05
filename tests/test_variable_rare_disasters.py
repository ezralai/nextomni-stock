"""Unit, boundary, security, and integration tests for Xavier Gabaix (2012)

Variable Rare Disasters, Macroeconomic Tail Risk & Firm Resilience Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.variable_rare_disasters import (
    RareDisasterReceipt,
    calculate_variable_rare_disasters,
    classify_disaster_regime,
    evaluate_rare_disasters_profile,
    persist_rare_disasters_record,
    verify_disaster_audit_digest,
)
from src.cli import rare_disasters_command
from src.security.validators import SecurityValidationError


def test_calculate_variable_rare_disasters_vulnerable() -> None:
    """Verify vulnerable firm with low recovery factor triggers disaster vulnerable regime."""
    receipt = calculate_variable_rare_disasters(
        ticker="VULN",
        latest_price=45.0,
        disaster_arrival_rate_lambda=0.05,
        firm_dividend_recovery_factor_f=0.35,
    )

    assert receipt.ticker == "VULN"
    assert receipt.disaster_arrival_rate_lambda == 0.05
    assert receipt.firm_dividend_recovery_factor_f == 0.35
    assert receipt.is_disaster_vulnerable is True
    assert receipt.regime == "DISASTER_VULNERABLE_HIGH_PREMIUM"
    assert receipt.disaster_resilience_premium_pct >= 5.5
    assert receipt.total_disaster_hurdle_pct > receipt.capm_required_return_pct
    assert verify_disaster_audit_digest(receipt) is True


def test_calculate_variable_rare_disasters_cyclical() -> None:
    """Verify cyclical firm with moderate recovery factor triggers cyclical tail risk regime."""
    receipt = calculate_variable_rare_disasters(
        ticker="CYCLIC",
        latest_price=75.0,
        disaster_arrival_rate_lambda=0.035,
        firm_dividend_recovery_factor_f=0.68,
    )

    assert receipt.is_cyclical_exposure is True
    assert receipt.regime == "CYCLICAL_TAIL_RISK_EXPOSURE"
    assert receipt.disaster_resilience_premium_pct >= 2.5
    assert verify_disaster_audit_digest(receipt) is True


def test_calculate_variable_rare_disasters_fortress() -> None:
    """Verify defensive firm with high recovery factor triggers resilient fortress regime."""
    receipt = calculate_variable_rare_disasters(
        ticker="FORTR",
        latest_price=160.0,
        disaster_arrival_rate_lambda=0.035,
        firm_dividend_recovery_factor_f=0.96,
    )

    assert receipt.is_resilient_fortress is True
    assert receipt.regime == "DISASTER_RESILIENT_FORTRESS"
    assert receipt.disaster_resilience_premium_pct <= 1.0
    assert receipt.implied_price_dividend_multiple > 10.0
    assert verify_disaster_audit_digest(receipt) is True


def test_calculate_variable_rare_disasters_benign() -> None:
    """Verify intermediate firm triggers benign tail stability regime."""
    receipt = calculate_variable_rare_disasters(
        ticker="BENIGN",
        latest_price=110.0,
        disaster_arrival_rate_lambda=0.020,
        firm_dividend_recovery_factor_f=0.85,
    )

    assert receipt.is_benign_stability is True
    assert receipt.regime == "BENIGN_TAIL_STABILITY"
    assert verify_disaster_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: lower recovery F or higher lambda increases premium and compresses P/D."""
    # 1. Recovery factor F monotonicity
    f_values = [0.30, 0.50, 0.70, 0.90]
    premiums_f = []
    multiples_f = []

    for f in f_values:
        rec = calculate_variable_rare_disasters(
            ticker="MONOF",
            firm_dividend_recovery_factor_f=f,
        )
        premiums_f.append(rec.disaster_resilience_premium_pct)
        multiples_f.append(rec.implied_price_dividend_multiple)

    # Lower F -> strictly higher premium H
    for i in range(len(premiums_f) - 1):
        assert premiums_f[i] > premiums_f[i + 1]

    # Lower F -> strictly lower P/D multiple (valuation compression)
    for i in range(len(multiples_f) - 1):
        assert multiples_f[i] < multiples_f[i + 1]

    # 2. Disaster arrival rate lambda monotonicity
    lambda_values = [0.01, 0.03, 0.05, 0.08]
    premiums_l = []

    for lam in lambda_values:
        rec = calculate_variable_rare_disasters(
            ticker="MONOL",
            disaster_arrival_rate_lambda=lam,
            firm_dividend_recovery_factor_f=0.70,
        )
        premiums_l.append(rec.disaster_resilience_premium_pct)

    # Higher lambda -> strictly higher premium H
    for i in range(len(premiums_l) - 1):
        assert premiums_l[i] < premiums_l[i + 1]


def test_classify_disaster_regime() -> None:
    """Verify regime boundaries for disaster resilience."""
    assert classify_disaster_regime(0.50, 6.0) == "DISASTER_VULNERABLE_HIGH_PREMIUM"
    assert classify_disaster_regime(0.70, 3.5) == "CYCLICAL_TAIL_RISK_EXPOSURE"
    assert classify_disaster_regime(0.95, 0.5) == "DISASTER_RESILIENT_FORTRESS"
    assert classify_disaster_regime(0.85, 1.8) == "BENIGN_TAIL_STABILITY"


def test_audit_digest_security() -> None:
    """Verify cryptographic audit digest generation and tamper rejection."""
    receipt = calculate_variable_rare_disasters(ticker="AUDIT")
    assert verify_disaster_audit_digest(receipt) is True

    # Tampered latest price
    tampered_price = RareDisasterReceipt(
        **{**receipt.to_dict(), "latest_price": receipt.latest_price + 25.0}
    )
    assert verify_disaster_audit_digest(tampered_price) is False

    # Tampered disaster premium
    tampered_premium = RareDisasterReceipt(
        **{**receipt.to_dict(), "disaster_resilience_premium_pct": receipt.disaster_resilience_premium_pct + 1.0}
    )
    assert verify_disaster_audit_digest(tampered_premium) is False


def test_security_input_validation() -> None:
    """Verify strict input validation against malicious inputs and boundary violations."""
    with pytest.raises(SecurityValidationError):
        calculate_variable_rare_disasters(ticker="TOOLONGTICKER")

    with pytest.raises(SecurityValidationError):
        calculate_variable_rare_disasters(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(SecurityValidationError):
        calculate_variable_rare_disasters(ticker="AAPL", disaster_arrival_rate_lambda=0.001)

    with pytest.raises(SecurityValidationError):
        calculate_variable_rare_disasters(ticker="AAPL", firm_dividend_recovery_factor_f=1.80)


def test_evaluate_rare_disasters_profile_synthetic() -> None:
    """Verify profile evaluation with synthetic historical price observations."""
    np.random.seed(42)
    # Generate 150 daily price observations with intermittent simulated macro dip
    base = 100.0
    returns = np.random.normal(0.0005, 0.015, 150)
    returns[75:80] = -0.06  # Simulate severe drawdown event
    prices = [base]
    for r in returns:
        prices.append(prices[-1] * (1.0 + r))

    receipt = evaluate_rare_disasters_profile(
        ticker="SYNTH",
        prices=prices,
    )

    assert receipt.ticker == "SYNTH"
    assert receipt.latest_price > 0.0
    assert 0.005 <= receipt.disaster_arrival_rate_lambda <= 0.20
    assert 0.05 <= receipt.firm_dividend_recovery_factor_f <= 1.50
    assert verify_disaster_audit_digest(receipt) is True


def test_database_persistence(tmp_path: Path) -> None:
    """Verify parameterized persistence into SQLite table rare_disasters."""
    db_file = tmp_path / "test_rare_disasters.db"
    receipt = calculate_variable_rare_disasters(
        ticker="DBTEST",
        latest_price=125.0,
        firm_dividend_recovery_factor_f=0.72,
    )

    persist_rare_disasters_record(receipt, db_path=db_file)

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, latest_price, regime, disaster_resilience_premium_pct FROM variable_rare_disaster_records WHERE ticker = ?", ("DBTEST",))
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "DBTEST"
    assert row[1] == 125.0
    assert row[2] == receipt.regime
    assert abs(row[3] - receipt.disaster_resilience_premium_pct) < 1e-4


def test_cli_command_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command generates formatted human-readable output."""
    args = argparse.Namespace(
        ticker="AAPL",
        disaster_lambda=0.04,
        recovery_f=0.70,
        gamma=4.0,
        json=False,
    )

    rare_disasters_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Gabaix (2012) Variable Rare Disasters Engine" in captured
    assert "AAPL" in captured
    assert "Disaster Resilience Premium (H):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command outputs valid parseable JSON receipt."""
    args = argparse.Namespace(
        ticker="MSFT",
        disaster_lambda=0.03,
        recovery_f=0.85,
        gamma=3.5,
        json=True,
    )

    rare_disasters_command(args)
    captured = capsys.readouterr().out

    payload = json.loads(captured)
    assert payload["ticker"] == "MSFT"
    assert "disaster_resilience_premium_pct" in payload
    assert "audit_digest" in payload
    assert "regime" in payload
