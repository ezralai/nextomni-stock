"""Tests for Lu Zhang (2005) Costly Reversibility of Capital & Value Beta Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.costly_reversibility import (
    CostlyReversibilityReceipt,
    calculate_costly_reversibility,
    classify_reversibility_regime,
    evaluate_costly_reversibility_profile,
)
from src.cli import reversibility_command
from src.security.validators import SecurityValidationError


def test_calculate_costly_reversibility_compensated_value_leader() -> None:
    """Verify compensated irreversible value leader when B/M >= 0.85 and LZVI >= 0.40."""
    receipt = calculate_costly_reversibility(
        ticker="XOM",
        latest_price=118.0,
        book_to_market_ratio=1.15,
        downward_upward_cost_asymmetry=9.5,
        fixed_capital_intensity=0.58,
        macro_state_zscore=-0.40,
    )
    assert receipt.ticker == "XOM"
    assert receipt.capital_irreversibility_index > 5.0
    assert receipt.disinvestment_friction_mult > 1.2
    assert receipt.countercyclical_value_beta > 1.3
    assert receipt.lu_zhang_value_impulse >= 0.40
    assert receipt.regime == "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM"
    assert receipt.is_compensated_irreversible_value_leader is True
    assert receipt.is_stranded_capital_recession_trap is False
    assert receipt.expected_reversibility_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_costly_reversibility_stranded_capital_recession_trap() -> None:
    """Verify stranded capital recession trap in deep contraction with high fixed capital."""
    receipt = calculate_costly_reversibility(
        ticker="F",
        latest_price=11.0,
        book_to_market_ratio=0.70,
        downward_upward_cost_asymmetry=12.0,
        fixed_capital_intensity=0.85,
        macro_state_zscore=-1.80,
    )
    assert receipt.lu_zhang_value_impulse <= -0.35
    assert receipt.regime == "STRANDED_CAPITAL_RECESSION_TRAP"
    assert receipt.is_stranded_capital_recession_trap is True
    assert receipt.is_compensated_irreversible_value_leader is False
    assert receipt.expected_reversibility_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_costly_reversibility_flexible_growth_expansion_option() -> None:
    """Verify flexible growth expansion option when B/M <= 0.45 and f_K <= 0.35."""
    receipt = calculate_costly_reversibility(
        ticker="CRM",
        latest_price=290.0,
        book_to_market_ratio=0.25,
        downward_upward_cost_asymmetry=3.0,
        fixed_capital_intensity=0.20,
        macro_state_zscore=0.50,
    )
    assert receipt.regime == "FLEXIBLE_GROWTH_EXPANSION_OPTION"
    assert receipt.is_compensated_irreversible_value_leader is False
    assert receipt.is_stranded_capital_recession_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_costly_reversibility_neutral_equilibrium() -> None:
    """Verify neutral adjustment cost equilibrium for moderate B/M and fixed capital."""
    receipt = calculate_costly_reversibility(
        ticker="JPM",
        latest_price=195.0,
        book_to_market_ratio=0.65,
        downward_upward_cost_asymmetry=5.0,
        fixed_capital_intensity=0.45,
        macro_state_zscore=0.10,
    )
    assert receipt.regime == "NEUTRAL_ADJUSTMENT_COST_EQUILIBRIUM"
    assert receipt.is_compensated_irreversible_value_leader is False
    assert receipt.is_stranded_capital_recession_trap is False
    assert receipt.verify_integrity() is True


def test_lu_zhang_countercyclical_value_beta_rises_in_recessions() -> None:
    """Verify Lu Zhang Theorem: value stock conditional beta rises in bad macroeconomic states."""
    expansion = calculate_costly_reversibility(
        ticker="XOM",
        latest_price=110.0,
        book_to_market_ratio=1.20,
        downward_upward_cost_asymmetry=10.0,
        fixed_capital_intensity=0.65,
        macro_state_zscore=1.20,
    )
    recession = calculate_costly_reversibility(
        ticker="XOM",
        latest_price=110.0,
        book_to_market_ratio=1.20,
        downward_upward_cost_asymmetry=10.0,
        fixed_capital_intensity=0.65,
        macro_state_zscore=-1.20,
    )
    assert recession.disinvestment_friction_mult > expansion.disinvestment_friction_mult
    assert recession.countercyclical_value_beta > expansion.countercyclical_value_beta


def test_classify_reversibility_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_reversibility_regime(0.80, 0.80, -1.20, -0.50)
        == "STRANDED_CAPITAL_RECESSION_TRAP"
    )
    assert (
        classify_reversibility_regime(0.95, 0.55, -0.20, 0.55)
        == "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM"
    )
    assert (
        classify_reversibility_regime(0.30, 0.25, 0.40, -0.10)
        == "FLEXIBLE_GROWTH_EXPANSION_OPTION"
    )
    assert (
        classify_reversibility_regime(0.65, 0.50, 0.0, 0.15)
        == "NEUTRAL_ADJUSTMENT_COST_EQUILIBRIUM"
    )


def test_costly_reversibility_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_costly_reversibility(
        ticker="AAPL",
        latest_price=220.0,
    )
    assert receipt.verify_integrity() is True

    tampered = CostlyReversibilityReceipt(
        **{
            **receipt.to_dict(),
            "countercyclical_value_beta": 9.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_costly_reversibility_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_costly_reversibility("AAPL", 0.0)

    with pytest.raises(ValueError, match="book_to_market_ratio"):
        calculate_costly_reversibility("AAPL", 150.0, book_to_market_ratio=0.01)

    with pytest.raises(ValueError, match="downward_upward_cost_asymmetry"):
        calculate_costly_reversibility("AAPL", 150.0, downward_upward_cost_asymmetry=0.5)

    with pytest.raises(ValueError, match="fixed_capital_intensity"):
        calculate_costly_reversibility("AAPL", 150.0, fixed_capital_intensity=1.2)

    with pytest.raises(ValueError, match="macro_state_zscore"):
        calculate_costly_reversibility("AAPL", 150.0, macro_state_zscore=-4.0)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_costly_reversibility(malicious, 150.0)


def test_evaluate_costly_reversibility_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_rev.db"
    db.init_db(db_file)
    db.upsert_stock(
        ticker="AAPL",
        exchange="XNAS",
        company_name="Apple Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=db_file,
    )

    records = []
    for i in range(30):
        p = 180.0 - i * 0.5
        records.append(
            {
                "ticker": "AAPL",
                "exchange": "XNAS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 50_000_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    receipt = evaluate_costly_reversibility_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(165.5, abs=1e-2)
    assert receipt.countercyclical_value_beta > 0.40
    assert receipt.verify_integrity() is True


def test_evaluate_costly_reversibility_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_rev.db"
    db.init_db(db_file)
    receipt = evaluate_costly_reversibility_profile(
        ticker="AAPL",
        db_path=db_file,
        book_to_market_ratio=0.70,
        downward_upward_cost_asymmetry=14.0,
        fixed_capital_intensity=0.85,
        macro_state_zscore=-1.90,
    )
    assert receipt.regime == "STRANDED_CAPITAL_RECESSION_TRAP"
    assert receipt.is_stranded_capital_recession_trap is True


def test_cli_reversibility_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI reversibility_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        book_to_market=1.15,
        cost_asymmetry=9.0,
        fixed_capital_intensity=0.55,
        macro_state_zscore=-0.35,
        json=False,
    )
    reversibility_command(args_text)
    captured = capsys.readouterr()
    assert "Lu Zhang (2005) Costly Reversibility & Value Beta Engine" in captured.out
    assert "Countercyclical Value Beta:" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        book_to_market=1.15,
        cost_asymmetry=9.0,
        fixed_capital_intensity=0.55,
        macro_state_zscore=-0.35,
        json=True,
    )
    reversibility_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "COMPENSATED_IRREVERSIBLE_VALUE_PREMIUM"
    assert len(payload["signature"]) == 64

