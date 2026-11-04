"""Unit and integration tests for Christopher Polk & Paola Sapienza (2009)
Catering Theory of Corporate Investment & Valuation Mispricing Engine (Iteration 88).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.catering_investment import (
    CateringInvestmentReceipt,
    calculate_catering_investment,
    classify_catering_regime,
    evaluate_catering_investment_profile,
)
from src.cli import catering_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_disciplined_fundamental_allocator_regime() -> None:
    """Verify restrained abnormal investment and low mispricing classify as DISCIPLINED_FUNDAMENTAL_ALLOCATOR."""
    rec = calculate_catering_investment(
        ticker="AAPL",
        latest_price=190.0,
        capex_amount=320.0,
        lagged_net_ppe=2_500.0,
        tobins_q=2.20,
        cash_flow_to_capital=0.45,
        discretionary_mispricing_score=0.02,
        annual_turnover_pct=95.0,
    )
    assert rec.residual_abnormal_investment <= 0.02
    assert rec.catering_distortion_index <= 0.15
    assert rec.regime == "DISCIPLINED_FUNDAMENTAL_ALLOCATOR"
    assert rec.is_disciplined_allocator is True
    assert rec.is_catering_capex_trap is False
    assert rec.expected_catering_alpha_pct > 2.5
    assert rec.verify_integrity() is True


def test_overvalued_catering_capex_trap_regime() -> None:
    """Verify aggressive CapEx above Q-benchmark coupled with equity overvaluation and high turnover triggers trap."""
    rec = calculate_catering_investment(
        ticker="NVDA",
        latest_price=140.0,
        capex_amount=1_200.0,
        lagged_net_ppe=2_000.0,
        tobins_q=1.50,
        cash_flow_to_capital=0.20,
        discretionary_mispricing_score=0.65,
        annual_turnover_pct=320.0,
    )
    assert rec.investment_to_capital_ratio == 0.60
    assert rec.residual_abnormal_investment >= 0.15
    assert rec.catering_distortion_index >= 1.10
    assert rec.regime == "OVERVALUED_CATERING_CAPEX_TRAP"
    assert rec.is_catering_capex_trap is True
    assert rec.is_disciplined_allocator is False
    assert rec.expected_catering_alpha_pct < -4.0
    assert rec.verify_integrity() is True


def test_neoclassical_q_expansion_regime() -> None:
    """Verify high CapEx justified by high Tobin's Q and cash flow classifies as NEOCLASSICAL_Q_EXPANSION."""
    rec = calculate_catering_investment(
        ticker="MSFT",
        latest_price=420.0,
        capex_amount=650.0,
        lagged_net_ppe=2_500.0,
        tobins_q=3.20,
        cash_flow_to_capital=0.45,
        discretionary_mispricing_score=0.22,
        annual_turnover_pct=110.0,
    )
    assert rec.investment_to_capital_ratio >= 0.20
    assert rec.catering_distortion_index < 0.45
    assert rec.regime == "NEOCLASSICAL_Q_EXPANSION"
    assert rec.is_catering_capex_trap is False


def test_sentiment_catering_watch_regime() -> None:
    """Verify intermediate catering distortion (0.45 <= CDI_t < 1.10) classifies as SENTIMENT_CATERING_WATCH."""
    rec = calculate_catering_investment(
        ticker="TSLA",
        latest_price=250.0,
        capex_amount=720.0,
        lagged_net_ppe=2_000.0,
        tobins_q=1.80,
        cash_flow_to_capital=0.30,
        discretionary_mispricing_score=0.30,
        annual_turnover_pct=200.0,
    )
    assert 0.45 <= rec.catering_distortion_index < 1.10
    assert rec.regime == "SENTIMENT_CATERING_WATCH"
    assert rec.is_catering_capex_trap is False
    assert rec.is_disciplined_allocator is False


def test_turnover_and_mispricing_interaction_monotonicity() -> None:
    """Verify Polk-Sapienza interaction: higher share turnover and mispricing amplify CDI when RAI > 0."""
    rec_low_turn = calculate_catering_investment(
        ticker="AAPL",
        latest_price=180.0,
        capex_amount=800.0,
        lagged_net_ppe=2_000.0,
        tobins_q=1.60,
        cash_flow_to_capital=0.25,
        discretionary_mispricing_score=0.10,
        annual_turnover_pct=50.0,
    )
    rec_high_turn = calculate_catering_investment(
        ticker="AAPL",
        latest_price=180.0,
        capex_amount=800.0,
        lagged_net_ppe=2_000.0,
        tobins_q=1.60,
        cash_flow_to_capital=0.25,
        discretionary_mispricing_score=0.60,
        annual_turnover_pct=350.0,
    )
    assert rec_high_turn.short_horizon_turnover_multiplier > rec_low_turn.short_horizon_turnover_multiplier
    assert rec_high_turn.catering_distortion_index > rec_low_turn.catering_distortion_index
    assert rec_high_turn.expected_catering_alpha_pct < rec_low_turn.expected_catering_alpha_pct


def test_classify_catering_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_catering_regime."""
    assert classify_catering_regime(0.15, 0.01, 0.05, 0.05) == "DISCIPLINED_FUNDAMENTAL_ALLOCATOR"
    assert classify_catering_regime(0.28, 0.04, 0.10, 0.20) == "NEOCLASSICAL_Q_EXPANSION"
    assert classify_catering_regime(0.35, 0.12, 0.25, 0.65) == "SENTIMENT_CATERING_WATCH"
    assert classify_catering_regime(0.55, 0.25, 0.55, 1.35) == "OVERVALUED_CATERING_CAPEX_TRAP"


def test_extreme_ik_clipping_guard() -> None:
    """Verify asset-light firm with very small lagged PPE clips IK_t to 2.50 without numerical overflow."""
    rec = calculate_catering_investment(
        ticker="ADBE",
        latest_price=500.0,
        capex_amount=1_000.0,
        lagged_net_ppe=100.0,
        tobins_q=3.50,
        cash_flow_to_capital=1.20,
    )
    assert rec.investment_to_capital_ratio == 2.50
    assert np.isfinite(rec.catering_distortion_index)
    assert rec.verify_integrity() is True


def test_evaluate_catering_investment_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_catering_investment_profile reads closing prices from SQLite and computes catering metrics."""
    db_file = tmp_path / "catering_test.db"
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
        p = 180.0 + i * 0.5
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

    rec = evaluate_catering_investment_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert abs(rec.latest_price - 192.0) < 1e-2
    assert rec.lagged_net_ppe == 2_400.0
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, capex, lagged_net_ppe, tobins_q, mispricing, and turnover raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_catering_investment("AAPL", 0.0, 300.0, 2000.0)
    with pytest.raises(ValueError, match="capex_amount must be non-negative"):
        calculate_catering_investment("AAPL", 150.0, -50.0, 2000.0)
    with pytest.raises(ValueError, match="lagged_net_ppe must be strictly positive"):
        calculate_catering_investment("AAPL", 150.0, 300.0, 0.0)
    with pytest.raises(ValueError, match="tobins_q must be strictly positive"):
        calculate_catering_investment("AAPL", 150.0, 300.0, 2000.0, tobins_q=0.0)
    with pytest.raises(ValueError, match="discretionary_mispricing_score must be in"):
        calculate_catering_investment(
            "AAPL", 150.0, 300.0, 2000.0, discretionary_mispricing_score=1.5
        )
    with pytest.raises(ValueError, match="annual_turnover_pct must be non-negative"):
        calculate_catering_investment("AAPL", 150.0, 300.0, 2000.0, annual_turnover_pct=-10.0)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_catering_investment(malicious, 150.0, 300.0, 2000.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_catering_investment(
        ticker="GOOG",
        latest_price=175.0,
        capex_amount=400.0,
        lagged_net_ppe=2_200.0,
    )
    assert rec.verify_integrity() is True
    tampered = CateringInvestmentReceipt(
        **{**rec.to_dict(), "expected_catering_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_catering_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI catering_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        capex=320.0,
        ppent_lag=2500.0,
        tobins_q=2.20,
        cash_flow_ratio=0.45,
        mispricing_score=0.02,
        turnover_pct=95.0,
        json=False,
    )
    catering_command(args_text)
    captured = capsys.readouterr()
    assert "Polk & Sapienza (2009) Catering Theory of Investment Engine" in captured.out
    assert "AAPL" in captured.out
    assert "DISCIPLINED_FUNDAMENTAL_ALLOCATOR" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        capex=320.0,
        ppent_lag=2500.0,
        tobins_q=2.20,
        cash_flow_ratio=0.45,
        mispricing_score=0.02,
        turnover_pct=95.0,
        json=True,
    )
    catering_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "DISCIPLINED_FUNDAMENTAL_ALLOCATOR"
    assert payload["is_disciplined_allocator"] is True
    assert len(payload["signature"]) == 64

