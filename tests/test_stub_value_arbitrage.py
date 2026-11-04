"""Tests for Lamont & Thaler (2003) & Mitchell-Pulvino-Stafford (2002) Stub Value Arbitrage Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.stub_value_arbitrage import (
    StubValueArbitrageReceipt,
    calculate_stub_value_arbitrage,
    classify_stub_regime,
    evaluate_stub_value_arbitrage_profile,
)
from src.cli import stub_command
from src.security.validators import SecurityValidationError


def test_calculate_stub_value_arbitrage_negative_stub_violation() -> None:
    """Verify negative stub violation (Lamont & Thaler 2003 3Com/Palm benchmark)."""
    receipt = calculate_stub_value_arbitrage(
        parent_ticker="COMS",
        latest_parent_price=104.0,
        subsidiary_ticker="PALM",
        subsidiary_price=95.0,
        shares_per_parent_ratio=1.50,
        annual_borrow_fee_pct=25.0,
    )
    assert receipt.parent_ticker == "COMS"
    assert receipt.subsidiary_ticker == "PALM"
    assert receipt.implied_subsidiary_value_per_share == pytest.approx(142.5, abs=1e-2)
    assert receipt.implied_stub_value == pytest.approx(-38.5, abs=1e-2)
    assert receipt.stub_fraction_of_parent < -0.05
    assert receipt.short_sale_friction_multiplier > 1.40
    assert receipt.lamont_thaler_arbitrage_impulse <= -0.40
    assert receipt.regime == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP"
    assert receipt.is_negative_stub_violation is True
    assert receipt.is_compressed_stub_opportunity is False
    assert receipt.expected_convergence_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_stub_value_arbitrage_compressed_stub_opportunity() -> None:
    """Verify compressed stub arbitrage opportunity when 0.0 <= stub_frac <= 0.15."""
    receipt = calculate_stub_value_arbitrage(
        parent_ticker="PARENT",
        latest_parent_price=100.0,
        subsidiary_ticker="SUB",
        subsidiary_price=92.0,
        shares_per_parent_ratio=1.0,
        annual_borrow_fee_pct=3.0,
    )
    assert 0.0 <= receipt.stub_fraction_of_parent <= 0.15
    assert receipt.lamont_thaler_arbitrage_impulse >= 0.35
    assert receipt.regime == "COMPRESSED_STUB_ARBITRAGE_CONVERGENCE"
    assert receipt.is_compressed_stub_opportunity is True
    assert receipt.is_negative_stub_violation is False
    assert receipt.expected_convergence_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_stub_value_arbitrage_normal_operating_equilibrium() -> None:
    """Verify normal operating stub equilibrium when stub_frac > 0.15."""
    receipt = calculate_stub_value_arbitrage(
        parent_ticker="GOOG",
        latest_parent_price=180.0,
        subsidiary_ticker="WAYMO",
        subsidiary_price=50.0,
        shares_per_parent_ratio=0.80,
        annual_borrow_fee_pct=2.0,
    )
    assert receipt.stub_fraction_of_parent > 0.15
    assert receipt.regime == "NORMAL_STUB_OPERATING_EQUILIBRIUM"
    assert receipt.is_negative_stub_violation is False
    assert receipt.is_compressed_stub_opportunity is False
    assert receipt.verify_integrity() is True


def test_borrow_fee_increases_short_sale_friction_multiplier() -> None:
    """Verify higher borrowing fee increases the short-sale friction multiplier."""
    low_fee = calculate_stub_value_arbitrage(
        parent_ticker="COMS",
        latest_parent_price=100.0,
        subsidiary_price=110.0,
        shares_per_parent_ratio=1.0,
        annual_borrow_fee_pct=2.0,
    )
    high_fee = calculate_stub_value_arbitrage(
        parent_ticker="COMS",
        latest_parent_price=100.0,
        subsidiary_price=110.0,
        shares_per_parent_ratio=1.0,
        annual_borrow_fee_pct=40.0,
    )
    assert high_fee.short_sale_friction_multiplier > low_fee.short_sale_friction_multiplier


def test_classify_stub_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_stub_regime(-0.15, 20.0, -1.5)
        == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP"
    )
    assert (
        classify_stub_regime(0.08, 4.0, 0.65)
        == "COMPRESSED_STUB_ARBITRAGE_CONVERGENCE"
    )
    assert (
        classify_stub_regime(0.45, 2.0, 0.10)
        == "NORMAL_STUB_OPERATING_EQUILIBRIUM"
    )


def test_stub_value_arbitrage_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_stub_value_arbitrage(
        parent_ticker="COMS",
        latest_parent_price=104.0,
    )
    assert receipt.verify_integrity() is True

    tampered = StubValueArbitrageReceipt(
        **{
            **receipt.to_dict(),
            "implied_stub_value": 999.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_stub_value_arbitrage_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_parent_price must be strictly positive"):
        calculate_stub_value_arbitrage("COMS", 0.0)

    with pytest.raises(ValueError, match="subsidiary_price must be strictly positive"):
        calculate_stub_value_arbitrage("COMS", 100.0, subsidiary_price=-10.0)

    with pytest.raises(ValueError, match="shares_per_parent_ratio"):
        calculate_stub_value_arbitrage("COMS", 100.0, shares_per_parent_ratio=0.001)

    with pytest.raises(ValueError, match="annual_borrow_fee_pct"):
        calculate_stub_value_arbitrage("COMS", 100.0, annual_borrow_fee_pct=150.0)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in parent and subsidiary tickers are rejected."""
    for malicious in ["../etc/passwd", "COMS'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_stub_value_arbitrage(malicious, 100.0)

    with pytest.raises(SecurityValidationError):
        calculate_stub_value_arbitrage("COMS", 100.0, subsidiary_ticker="BAD'; DROP--")


def test_evaluate_stub_value_arbitrage_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_stub.db"
    db.init_db(db_file)
    db.upsert_stock(
        ticker="COMS",
        exchange="XNAS",
        company_name="3Com Corporation",
        currency="USD",
        timezone="America/New_York",
        db_path=db_file,
    )

    records = []
    for i in range(30):
        p = 100.0 + i * 0.5
        records.append(
            {
                "ticker": "COMS",
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

    receipt = evaluate_stub_value_arbitrage_profile(ticker="COMS", db_path=db_file)
    assert receipt.parent_ticker == "COMS"
    assert receipt.latest_parent_price == pytest.approx(114.5, abs=1e-2)
    assert receipt.verify_integrity() is True


def test_evaluate_stub_value_arbitrage_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_stub.db"
    db.init_db(db_file)
    receipt = evaluate_stub_value_arbitrage_profile(
        ticker="COMS",
        db_path=db_file,
        subsidiary_ticker="PALM",
        subsidiary_price=120.0,
        shares_per_parent_ratio=1.50,
        annual_borrow_fee_pct=30.0,
    )
    assert receipt.subsidiary_ticker == "PALM"
    assert receipt.regime == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP"
    assert receipt.is_negative_stub_violation is True


def test_stub_value_arbitrage_to_dict_structure() -> None:
    """Verify StubValueArbitrageReceipt serialization dictionary keys and structure."""
    receipt = calculate_stub_value_arbitrage(
        parent_ticker="COMS",
        latest_parent_price=104.0,
        subsidiary_ticker="PALM",
        subsidiary_price=95.0,
    )
    d = receipt.to_dict()
    assert d["parent_ticker"] == "COMS"
    assert d["subsidiary_ticker"] == "PALM"
    assert "implied_stub_value" in d
    assert "signature" in d
    assert "created_at_utc" in d


def test_cli_stub_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI stub_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="COMS",
        subsidiary_ticker="PALM",
        subsidiary_price=95.0,
        shares_per_parent=1.5,
        borrow_fee=25.0,
        json=False,
    )
    stub_command(args_text)
    captured = capsys.readouterr()
    assert "Lamont & Thaler (2003) Stub Value Arbitrage Engine" in captured.out
    assert "Implied Stub Value (S_stub):" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="COMS",
        subsidiary_ticker="PALM",
        subsidiary_price=95.0,
        shares_per_parent=1.5,
        borrow_fee=25.0,
        json=True,
    )
    stub_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["parent_ticker"] == "COMS"
    assert payload["subsidiary_ticker"] == "PALM"
    assert payload["regime"] == "NEGATIVE_STUB_SHORT_SALE_IMPEDIMENT_TRAP"
    assert len(payload["signature"]) == 64
