"""Tests for George & Hwang (2004) 52-Week High Psychological Anchoring Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import src.data.db as db
from src.analytics.fifty_two_week_high import (
    FiftyTwoWeekHighReceipt,
    calculate_fifty_two_week_high,
    classify_52w_high_regime,
    evaluate_fifty_two_week_high_profile,
)
from src.cli import high52_command
from src.security.validators import SecurityValidationError


def test_calculate_fifty_two_week_high_near_high_leader() -> None:
    """Verify anchored 52-week-high breakout leader when NH_52 >= 0.90 and GHAI >= 0.45."""
    receipt = calculate_fifty_two_week_high(
        ticker="AAPL",
        latest_price=245.0,
        high_52w=250.0,
        low_52w=180.0,
        past_return_20d=0.28,
        breakout_volume_ratio=1.35,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.nearness_to_52w_high == pytest.approx(0.98, abs=1e-4)
    assert receipt.range_position_52w > 0.90
    assert receipt.george_hwang_anchoring_impulse >= 0.45
    assert receipt.regime == "ANCHORED_52W_HIGH_BREAKOUT_LEADER"
    assert receipt.is_52w_high_breakout_leader is True
    assert receipt.is_deep_anchor_overhang_trap is False
    assert receipt.expected_anchoring_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_fifty_two_week_high_dead_cat_bounce_divergence() -> None:
    """Verify dead-cat bounce divergence when PRET >= 0.08 but NH_52 < 0.70."""
    receipt = calculate_fifty_two_week_high(
        ticker="MSFT",
        latest_price=60.0,
        high_52w=100.0,
        low_52w=45.0,
        past_return_20d=0.22,
        breakout_volume_ratio=1.10,
    )
    assert receipt.nearness_to_52w_high == pytest.approx(0.60, abs=1e-4)
    assert receipt.regime == "DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE"
    assert receipt.is_52w_high_breakout_leader is False
    assert receipt.is_deep_anchor_overhang_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_fifty_two_week_high_deep_anchor_overhang_trap() -> None:
    """Verify deep 52-week anchor overhang trap when NH_52 <= 0.58 and past_return < 0.08."""
    receipt = calculate_fifty_two_week_high(
        ticker="INTC",
        latest_price=52.0,
        high_52w=100.0,
        low_52w=50.0,
        past_return_20d=-0.25,
        breakout_volume_ratio=0.90,
    )
    assert receipt.nearness_to_52w_high == pytest.approx(0.52, abs=1e-4)
    assert receipt.george_hwang_anchoring_impulse < -0.50
    assert receipt.regime == "DEEP_52W_ANCHOR_OVERHANG_TRAP"
    assert receipt.is_deep_anchor_overhang_trap is True
    assert receipt.is_52w_high_breakout_leader is False
    assert receipt.expected_anchoring_alpha_pct < 0.0
    assert receipt.verify_integrity() is True


def test_calculate_fifty_two_week_high_upper_range_accumulation() -> None:
    """Verify constructive upper-range accumulation when NH_52 is in [0.70, 0.90)."""
    receipt = calculate_fifty_two_week_high(
        ticker="GOOG",
        latest_price=170.0,
        high_52w=200.0,
        low_52w=130.0,
        past_return_20d=0.14,
        breakout_volume_ratio=1.05,
    )
    assert 0.70 <= receipt.nearness_to_52w_high < 0.90
    assert receipt.george_hwang_anchoring_impulse > 0.0
    assert receipt.regime == "CONSTRUCTIVE_UPPER_RANGE_ACCUMULATION"
    assert receipt.is_52w_high_breakout_leader is False
    assert receipt.is_deep_anchor_overhang_trap is False
    assert receipt.verify_integrity() is True


def test_calculate_fifty_two_week_high_equal_span_handling() -> None:
    """Verify degenerate 52-week span (H_52 == L_52 == P_t) sets range_position_52w to 0.50."""
    receipt = calculate_fifty_two_week_high(
        ticker="NVDA",
        latest_price=150.0,
        high_52w=150.0,
        low_52w=150.0,
        past_return_20d=0.02,
        breakout_volume_ratio=1.0,
    )
    assert receipt.nearness_to_52w_high == pytest.approx(1.0, abs=1e-4)
    assert receipt.range_position_52w == pytest.approx(0.50, abs=1e-4)
    assert receipt.is_52w_high_breakout_leader is True


def test_classify_52w_high_regime_boundaries() -> None:
    """Verify exact regime classification boundaries."""
    assert (
        classify_52w_high_regime(nearness_ratio=0.92, past_return=0.15, anchoring_impulse=0.50)
        == "ANCHORED_52W_HIGH_BREAKOUT_LEADER"
    )
    assert (
        classify_52w_high_regime(nearness_ratio=0.84, past_return=0.10, anchoring_impulse=0.30)
        == "CONSTRUCTIVE_UPPER_RANGE_ACCUMULATION"
    )
    assert (
        classify_52w_high_regime(nearness_ratio=0.66, past_return=0.18, anchoring_impulse=-0.20)
        == "DEAD_CAT_BOUNCE_ANCHOR_DIVERGENCE"
    )
    assert (
        classify_52w_high_regime(nearness_ratio=0.55, past_return=-0.20, anchoring_impulse=-0.60)
        == "DEEP_52W_ANCHOR_OVERHANG_TRAP"
    )


def test_fifty_two_week_high_hmac_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity fails if any receipt field is mutated."""
    receipt = calculate_fifty_two_week_high(
        ticker="AAPL",
        latest_price=245.0,
        high_52w=250.0,
        low_52w=180.0,
        past_return_20d=0.25,
        breakout_volume_ratio=1.20,
    )
    assert receipt.verify_integrity() is True

    tampered = FiftyTwoWeekHighReceipt(
        **{
            **receipt.to_dict(),
            "nearness_to_52w_high": 0.9999,
        }
    )
    assert tampered.verify_integrity() is False


def test_fifty_two_week_high_input_validation_bounds() -> None:
    """Verify input validation rejects out-of-range parameters."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_fifty_two_week_high("AAPL", 0.0)

    with pytest.raises(ValueError, match="high_52w"):
        calculate_fifty_two_week_high(
            ticker="AAPL",
            latest_price=200.0,
            high_52w=180.0,
            low_52w=150.0,
            past_return_20d=0.10,
            breakout_volume_ratio=1.0,
        )

    with pytest.raises(ValueError, match="low_52w"):
        calculate_fifty_two_week_high(
            ticker="AAPL",
            latest_price=200.0,
            high_52w=220.0,
            low_52w=210.0,
            past_return_20d=0.10,
            breakout_volume_ratio=1.0,
        )

    with pytest.raises(ValueError, match="past_return_20d"):
        calculate_fifty_two_week_high(
            ticker="AAPL",
            latest_price=200.0,
            high_52w=220.0,
            low_52w=150.0,
            past_return_20d=6.5,
            breakout_volume_ratio=1.0,
        )

    with pytest.raises(ValueError, match="breakout_volume_ratio"):
        calculate_fifty_two_week_high(
            ticker="AAPL",
            latest_price=200.0,
            high_52w=220.0,
            low_52w=150.0,
            past_return_20d=0.10,
            breakout_volume_ratio=0.0,
        )


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_fifty_two_week_high(malicious, 200.0, high_52w=220.0, low_52w=150.0)


def test_evaluate_fifty_two_week_high_profile_from_sqlite(tmp_path: Path) -> None:
    """Verify end-to-end empirical evaluation from SQLite price observations."""
    db_file = tmp_path / "test_high52.db"
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
    for i in range(25):
        p = 160.0 + i * 2.0
        records.append(
            {
                "ticker": "AAPL",
                "exchange": "XNAS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 0.5,
                "low": p - 0.5,
                "close": p,
                "adj_close": p,
                "volume": 55_000_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    receipt = evaluate_fifty_two_week_high_profile(ticker="AAPL", db_path=db_file)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(208.0, abs=1e-2)
    assert receipt.nearness_to_52w_high >= 0.98
    assert receipt.is_52w_high_breakout_leader is True
    assert receipt.verify_integrity() is True


def test_evaluate_fifty_two_week_high_profile_override_precedence(tmp_path: Path) -> None:
    """Verify explicit override parameters take precedence over SQLite defaults."""
    db_file = tmp_path / "test_override_high52.db"
    db.init_db(db_file)
    receipt = evaluate_fifty_two_week_high_profile(
        ticker="AAPL",
        db_path=db_file,
        high_52w=200.0,
        low_52w=80.0,
        past_return_20d=-0.20,
        breakout_volume_ratio=0.95,
    )
    assert receipt.nearness_to_52w_high == pytest.approx(0.50, abs=1e-4)
    assert receipt.regime == "DEEP_52W_ANCHOR_OVERHANG_TRAP"
    assert receipt.is_deep_anchor_overhang_trap is True


def test_cli_high52_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI high52_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        high_52w=300.0,
        low_52w=50.0,
        past_return=0.22,
        breakout_volume_ratio=1.35,
        json=False,
    )
    high52_command(args_text)
    captured = capsys.readouterr()
    assert "George & Hwang (2004) 52-Week High Anchoring Engine" in captured.out
    assert "Nearness to 52W High (NH_52,t):" in captured.out
    assert "Verified: True" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        high_52w=300.0,
        low_52w=50.0,
        past_return=0.22,
        breakout_volume_ratio=1.35,
        json=True,
    )
    high52_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert 0.0 < payload["nearness_to_52w_high"] <= 1.0
    assert len(payload["signature"]) == 64
