"""Unit and integration tests for Tarun Chordia, Richard Roll & Avanidhar Subrahmanyam (2000, 2001, 2002)
Commonality in Liquidity & Order Imbalance Market-Wide Friction Engine (Iteration 89).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.liquidity_commonality import (
    LiquidityCommonalityReceipt,
    calculate_liquidity_commonality,
    classify_commonality_regime,
    evaluate_liquidity_commonality_profile,
)
from src.cli import commonality_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_idiosyncratic_liquidity_haven_regime() -> None:
    """Verify low spread/depth commonality and mild order imbalance classify as IDIOSYNCRATIC_LIQUIDITY_HAVEN."""
    rec = calculate_liquidity_commonality(
        ticker="AAPL",
        latest_price=190.0,
        spread_commonality_beta=0.30,
        depth_commonality_beta=0.25,
        liquidity_market_r2=0.10,
        signed_order_imbalance=0.04,
        oib_autocorrelation=0.10,
        down_market_illiquidity_surge=1.05,
    )
    assert rec.effective_commonality_beta <= 0.45
    assert rec.systemic_inventory_contagion_index <= 0.65
    assert rec.regime == "IDIOSYNCRATIC_LIQUIDITY_HAVEN"
    assert rec.is_idiosyncratic_liquidity_haven is True
    assert rec.is_systemic_contagion_trap is False
    assert rec.expected_commonality_alpha_pct > 1.5
    assert rec.verify_integrity() is True


def test_systemic_liquidity_contagion_trap_regime() -> None:
    """Verify high spread commonality coupled with severe down-market illiquidity surge triggers trap."""
    rec = calculate_liquidity_commonality(
        ticker="NVDA",
        latest_price=140.0,
        spread_commonality_beta=1.60,
        depth_commonality_beta=1.40,
        liquidity_market_r2=0.45,
        signed_order_imbalance=-0.35,
        oib_autocorrelation=0.40,
        down_market_illiquidity_surge=1.65,
    )
    assert rec.effective_commonality_beta >= 1.35
    assert rec.systemic_inventory_contagion_index >= 1.75
    assert rec.regime == "SYSTEMIC_LIQUIDITY_CONTAGION_TRAP"
    assert rec.is_systemic_contagion_trap is True
    assert rec.is_idiosyncratic_liquidity_haven is False
    assert rec.expected_commonality_alpha_pct < -4.0
    assert rec.verify_integrity() is True


def test_order_imbalance_pressure_watch_regime() -> None:
    """Verify persistent institutional order-splitting (|OIB| >= 0.28, rho >= 0.35) classifies as watch."""
    rec = calculate_liquidity_commonality(
        ticker="TSLA",
        latest_price=245.0,
        spread_commonality_beta=0.55,
        depth_commonality_beta=0.50,
        liquidity_market_r2=0.15,
        signed_order_imbalance=0.38,
        oib_autocorrelation=0.48,
        down_market_illiquidity_surge=1.10,
    )
    assert rec.effective_commonality_beta < 1.35
    assert rec.systemic_inventory_contagion_index < 1.75
    assert rec.regime == "ORDER_IMBALANCE_PRESSURE_WATCH"
    assert rec.is_systemic_contagion_trap is False
    assert rec.is_idiosyncratic_liquidity_haven is False


def test_normal_market_comovement_regime() -> None:
    """Verify intermediate commonality beta and mild OIB classify as NORMAL_MARKET_COMOVEMENT."""
    rec = calculate_liquidity_commonality(
        ticker="MSFT",
        latest_price=425.0,
        spread_commonality_beta=0.75,
        depth_commonality_beta=0.65,
        liquidity_market_r2=0.22,
        signed_order_imbalance=0.10,
        oib_autocorrelation=0.15,
        down_market_illiquidity_surge=1.15,
    )
    assert 0.45 < rec.effective_commonality_beta < 1.35
    assert rec.regime == "NORMAL_MARKET_COMOVEMENT"
    assert rec.is_idiosyncratic_liquidity_haven is False
    assert rec.is_systemic_contagion_trap is False


def test_down_market_surge_and_r2_monotonicity() -> None:
    """Verify Chordia-Roll-Subrahmanyam (2001) asymmetric down-market surge increases SICI_t and lowers alpha."""
    rec_mild = calculate_liquidity_commonality(
        ticker="AAPL",
        latest_price=185.0,
        spread_commonality_beta=0.80,
        depth_commonality_beta=0.70,
        liquidity_market_r2=0.12,
        down_market_illiquidity_surge=1.00,
    )
    rec_severe = calculate_liquidity_commonality(
        ticker="AAPL",
        latest_price=185.0,
        spread_commonality_beta=0.80,
        depth_commonality_beta=0.70,
        liquidity_market_r2=0.55,
        down_market_illiquidity_surge=1.90,
    )
    assert rec_severe.systemic_inventory_contagion_index > rec_mild.systemic_inventory_contagion_index
    assert rec_severe.expected_commonality_alpha_pct < rec_mild.expected_commonality_alpha_pct


def test_classify_commonality_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_commonality_regime."""
    assert classify_commonality_regime(0.35, 0.05, 0.10, 0.48) == "IDIOSYNCRATIC_LIQUIDITY_HAVEN"
    assert classify_commonality_regime(0.75, 0.10, 0.15, 0.95) == "NORMAL_MARKET_COMOVEMENT"
    assert classify_commonality_regime(0.75, -0.35, 0.42, 1.20) == "ORDER_IMBALANCE_PRESSURE_WATCH"
    assert classify_commonality_regime(1.45, 0.10, 0.20, 1.90) == "SYSTEMIC_LIQUIDITY_CONTAGION_TRAP"


def test_order_splitting_autocorrelation_amplification() -> None:
    """Verify positive OIB autocorrelation amplifies order_splitting_pressure_score while negative does not."""
    rec_neg_ac = calculate_liquidity_commonality(
        ticker="AMD",
        latest_price=155.0,
        signed_order_imbalance=0.40,
        oib_autocorrelation=-0.30,
    )
    rec_pos_ac = calculate_liquidity_commonality(
        ticker="AMD",
        latest_price=155.0,
        signed_order_imbalance=0.40,
        oib_autocorrelation=0.55,
    )
    assert abs(rec_neg_ac.order_splitting_pressure_score - 0.40) < 1e-6
    assert abs(rec_pos_ac.order_splitting_pressure_score - (0.40 * 1.55)) < 1e-6


def test_evaluate_liquidity_commonality_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_liquidity_commonality_profile reads OHLCV history from SQLite and infers OIB."""
    db_file = tmp_path / "commonality_test.db"
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
        p = 180.0 + (0.6 if i % 2 == 0 else -0.3) * i
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
                "volume": 45_000_000 + i * 500_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    rec = evaluate_liquidity_commonality_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert rec.latest_price > 0
    assert -1.0 <= rec.signed_order_imbalance <= 1.0
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, R^2, OIB, autocorrelation, and down_market_surge raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_liquidity_commonality("AAPL", 0.0)
    with pytest.raises(ValueError, match="liquidity_market_r2 must be in"):
        calculate_liquidity_commonality("AAPL", 150.0, liquidity_market_r2=1.2)
    with pytest.raises(ValueError, match="signed_order_imbalance must be in"):
        calculate_liquidity_commonality("AAPL", 150.0, signed_order_imbalance=1.5)
    with pytest.raises(ValueError, match="oib_autocorrelation must be in"):
        calculate_liquidity_commonality("AAPL", 150.0, oib_autocorrelation=-1.2)
    with pytest.raises(ValueError, match="down_market_illiquidity_surge must be at least 0.5"):
        calculate_liquidity_commonality("AAPL", 150.0, down_market_illiquidity_surge=0.2)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_liquidity_commonality(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_liquidity_commonality(
        ticker="GOOG",
        latest_price=178.0,
        spread_commonality_beta=0.40,
        depth_commonality_beta=0.35,
    )
    assert rec.verify_integrity() is True
    tampered = LiquidityCommonalityReceipt(
        **{**rec.to_dict(), "expected_commonality_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_commonality_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI commonality_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        spread_beta=0.30,
        depth_beta=0.25,
        liquidity_r2=0.10,
        order_imbalance=0.04,
        oib_autocorr=0.10,
        down_market_surge=1.05,
        json=False,
    )
    commonality_command(args_text)
    captured = capsys.readouterr()
    assert "Chordia et al. (2000, 2001) Liquidity Commonality Engine" in captured.out
    assert "AAPL" in captured.out
    assert "IDIOSYNCRATIC_LIQUIDITY_HAVEN" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        spread_beta=0.30,
        depth_beta=0.25,
        liquidity_r2=0.10,
        order_imbalance=0.04,
        oib_autocorr=0.10,
        down_market_surge=1.05,
        json=True,
    )
    commonality_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "IDIOSYNCRATIC_LIQUIDITY_HAVEN"
    assert payload["is_idiosyncratic_liquidity_haven"] is True
    assert len(payload["signature"]) == 64

