"""Tests for Brunnermeier & Pedersen (2009) Market & Funding Liquidity Spirals Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.liquidity_spiral import (
    LiquiditySpiralReceipt,
    calculate_liquidity_spiral,
    classify_liquidity_spiral_regime,
    evaluate_liquidity_spiral_profile,
)
from src.cli import liquidity_spiral_command
from src.security.validators import SecurityValidationError


def test_calculate_liquidity_spiral_fire_sale_trap() -> None:
    """Verify destabilizing liquidity spiral fire-sale trap when M_spiral >= 1.70 and B_fund < -0.05."""
    receipt = calculate_liquidity_spiral(
        ticker="LEHM",
        latest_price=45.0,
        current_margin_haircut=0.55,
        baseline_margin_haircut=0.15,
        funding_margin_buffer=-0.45,
        annualized_volatility=0.65,
        baseline_volatility=0.20,
    )
    assert receipt.ticker == "LEHM"
    assert receipt.margin_tightness_ratio > 3.0
    assert receipt.funding_fragility_score > 4.0
    assert receipt.liquidity_spiral_multiplier >= 1.70
    assert receipt.brunnermeier_pedersen_impulse <= -0.50
    assert receipt.regime == "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE"
    assert receipt.is_fire_sale_spiral_trap is True
    assert receipt.is_funding_liquidity_safe_haven is False
    assert receipt.expected_liquidity_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_liquidity_spiral_safe_haven() -> None:
    """Verify robust funding liquidity safe haven when M_spiral <= 1.15 and B_fund >= 0.40."""
    receipt = calculate_liquidity_spiral(
        ticker="JNJ",
        latest_price=160.0,
        current_margin_haircut=0.12,
        baseline_margin_haircut=0.12,
        funding_margin_buffer=0.50,
        annualized_volatility=0.14,
        baseline_volatility=0.14,
    )
    assert receipt.liquidity_spiral_multiplier <= 1.15
    assert receipt.funding_margin_buffer >= 0.40
    assert receipt.brunnermeier_pedersen_impulse >= 0.35
    assert receipt.regime == "ROBUST_FUNDING_LIQUIDITY_SAFE_HAVEN"
    assert receipt.is_funding_liquidity_safe_haven is True
    assert receipt.is_fire_sale_spiral_trap is False
    assert receipt.expected_liquidity_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_liquidity_spiral_margin_tightening_watch() -> None:
    """Verify margin tightening watch when kappa >= 1.35 but funding buffer >= 0.0."""
    receipt = calculate_liquidity_spiral(
        ticker="AAPL",
        latest_price=220.0,
        current_margin_haircut=0.28,
        baseline_margin_haircut=0.18,
        funding_margin_buffer=0.15,
        annualized_volatility=0.32,
        baseline_volatility=0.22,
    )
    assert receipt.margin_tightness_ratio >= 1.35
    assert receipt.funding_margin_buffer >= 0.0
    assert receipt.regime == "MARGIN_TIGHTENING_WATCH"
    assert receipt.is_fire_sale_spiral_trap is False
    assert receipt.is_funding_liquidity_safe_haven is False
    assert receipt.verify_integrity() is True


def test_calculate_liquidity_spiral_normal_equilibrium() -> None:
    """Verify normal liquidity buffer equilibrium when parameters are near baseline."""
    receipt = calculate_liquidity_spiral(
        ticker="MSFT",
        latest_price=415.0,
        current_margin_haircut=0.16,
        baseline_margin_haircut=0.15,
        funding_margin_buffer=0.25,
        annualized_volatility=0.22,
        baseline_volatility=0.20,
    )
    assert receipt.regime == "NORMAL_LIQUIDITY_BUFFER_EQUILIBRIUM"
    assert receipt.is_fire_sale_spiral_trap is False
    assert receipt.is_funding_liquidity_safe_haven is False
    assert receipt.verify_integrity() is True


def test_margin_haircut_tightening_amplifies_spiral_multiplier() -> None:
    """Verify Brunnermeier & Pedersen Theorem: higher margin haircut strictly increases spiral multiplier."""
    normal_margin = calculate_liquidity_spiral(
        ticker="AAPL",
        latest_price=200.0,
        current_margin_haircut=0.15,
        baseline_margin_haircut=0.15,
        funding_margin_buffer=-0.30,
    )
    spiked_margin = calculate_liquidity_spiral(
        ticker="AAPL",
        latest_price=200.0,
        current_margin_haircut=0.45,
        baseline_margin_haircut=0.15,
        funding_margin_buffer=-0.30,
    )
    assert spiked_margin.liquidity_spiral_multiplier > normal_margin.liquidity_spiral_multiplier


def test_classify_liquidity_spiral_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_liquidity_spiral_regime(2.10, -0.25, 2.50)
        == "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE"
    )
    assert (
        classify_liquidity_spiral_regime(1.10, 0.45, 1.05)
        == "ROBUST_FUNDING_LIQUIDITY_SAFE_HAVEN"
    )
    assert (
        classify_liquidity_spiral_regime(1.30, 0.15, 1.45)
        == "MARGIN_TIGHTENING_WATCH"
    )
    assert (
        classify_liquidity_spiral_regime(1.20, 0.20, 1.10)
        == "NORMAL_LIQUIDITY_BUFFER_EQUILIBRIUM"
    )


def test_liquidity_spiral_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_liquidity_spiral(
        ticker="AAPL",
        latest_price=220.0,
    )
    assert receipt.verify_integrity() is True

    tampered = LiquiditySpiralReceipt(
        **{
            **receipt.to_dict(),
            "liquidity_spiral_multiplier": 9.99,
        }
    )
    assert tampered.verify_integrity() is False


def test_liquidity_spiral_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_liquidity_spiral("AAPL", 0.0)

    with pytest.raises(ValueError, match="current_margin_haircut"):
        calculate_liquidity_spiral("AAPL", 150.0, current_margin_haircut=0.95)

    with pytest.raises(ValueError, match="baseline_margin_haircut"):
        calculate_liquidity_spiral("AAPL", 150.0, baseline_margin_haircut=0.01)

    with pytest.raises(ValueError, match="funding_margin_buffer"):
        calculate_liquidity_spiral("AAPL", 150.0, funding_margin_buffer=-1.5)

    with pytest.raises(ValueError, match="annualized_volatility"):
        calculate_liquidity_spiral("AAPL", 150.0, annualized_volatility=2.5)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_liquidity_spiral(malicious, 150.0)


def test_evaluate_liquidity_spiral_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_spiral.db"
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
        p = 150.0 + i * 1.5
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

    receipt = evaluate_liquidity_spiral_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(193.5, abs=1e-2)
    assert receipt.liquidity_spiral_multiplier >= 1.0
    assert receipt.verify_integrity() is True


def test_evaluate_liquidity_spiral_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_spiral.db"
    db.init_db(db_file)
    receipt = evaluate_liquidity_spiral_profile(
        ticker="AAPL",
        db_path=db_file,
        current_margin_haircut=0.60,
        baseline_margin_haircut=0.15,
        funding_margin_buffer=-0.50,
        annualized_volatility=0.70,
        baseline_volatility=0.20,
    )
    assert receipt.regime == "DESTABILIZING_LIQUIDITY_SPIRAL_FIRE_SALE"
    assert receipt.is_fire_sale_spiral_trap is True


def test_cli_liquidity_spiral_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI liquidity_spiral_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        margin_haircut=0.25,
        baseline_margin=0.15,
        funding_buffer=0.30,
        volatility=0.25,
        baseline_volatility=0.20,
        json=False,
    )
    liquidity_spiral_command(args_text)
    captured = capsys.readouterr()
    assert "Brunnermeier & Pedersen (2009) Liquidity Spiral Engine" in captured.out
    assert "Liquidity Spiral Multiplier:" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        margin_haircut=0.25,
        baseline_margin=0.15,
        funding_buffer=0.30,
        volatility=0.25,
        baseline_volatility=0.20,
        json=True,
    )
    liquidity_spiral_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["liquidity_spiral_multiplier"] >= 1.0
    assert len(payload["signature"]) == 64

