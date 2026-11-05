"""Tests for Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis Engine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

import src.data.db as db
from src.analytics.adaptive_markets import (
    AdaptiveMarketsReceipt,
    calculate_adaptive_markets,
    classify_amh_regime,
    evaluate_amh_profile,
)
from src.cli import amh_command
from src.security.validators import SecurityValidationError


def test_calculate_adaptive_markets_competitive_saturation() -> None:
    """Verify high-efficiency competitive saturation when VR is near 1.0 and rho is near 0.0."""
    receipt = calculate_adaptive_markets(
        ticker="SPY",
        latest_price=510.0,
        lo_mackinlay_variance_ratio=1.02,
        serial_autocorrelation_rho1=0.01,
        ecological_stress_drawdown=0.03,
        realized_volatility=0.14,
        baseline_volatility=0.14,
    )
    assert receipt.ticker == "SPY"
    assert receipt.adaptive_market_efficiency_index >= 0.85
    assert receipt.regime == "HIGH_EFFICIENCY_COMPETITIVE_SATURATION"
    assert receipt.is_competitive_saturation is True
    assert receipt.is_punctuated_equilibrium is False
    assert receipt.verify_integrity() is True


def test_calculate_adaptive_markets_punctuated_equilibrium() -> None:
    """Verify behavioral punctuated equilibrium when VR departs significantly and stress is elevated."""
    receipt = calculate_adaptive_markets(
        ticker="PANIC",
        latest_price=35.0,
        lo_mackinlay_variance_ratio=1.95,
        serial_autocorrelation_rho1=0.45,
        ecological_stress_drawdown=0.45,
        realized_volatility=0.65,
        baseline_volatility=0.20,
    )
    assert receipt.adaptive_market_efficiency_index <= 0.40
    assert receipt.evolutionary_adaptation_pressure >= 1.50
    assert receipt.regime == "BEHAVIORAL_PUNCTUATED_EQUILIBRIUM"
    assert receipt.is_punctuated_equilibrium is True
    assert receipt.is_competitive_saturation is False
    assert receipt.expected_adaptive_alpha_pct > 0.0
    assert receipt.verify_integrity() is True


def test_calculate_adaptive_markets_adaptive_recovery() -> None:
    """Verify adaptive arbitrage recovery when efficiency is in moderate range (0.40 < E < 0.70)."""
    receipt = calculate_adaptive_markets(
        ticker="GROW",
        latest_price=120.0,
        lo_mackinlay_variance_ratio=1.45,
        serial_autocorrelation_rho1=0.20,
        ecological_stress_drawdown=0.12,
        realized_volatility=0.28,
        baseline_volatility=0.22,
    )
    assert 0.40 < receipt.adaptive_market_efficiency_index < 0.70
    assert receipt.regime == "ADAPTIVE_ARBITRAGE_RECOVERY"
    assert receipt.is_adaptive_recovery is True
    assert receipt.is_punctuated_equilibrium is False
    assert receipt.verify_integrity() is True


def test_calculate_adaptive_markets_cyclical_neutral() -> None:
    """Verify cyclical ecological equilibrium when efficiency is standard (0.70 <= E < 0.85)."""
    receipt = calculate_adaptive_markets(
        ticker="AAPL",
        latest_price=220.0,
        lo_mackinlay_variance_ratio=1.20,
        serial_autocorrelation_rho1=0.10,
        ecological_stress_drawdown=0.08,
        realized_volatility=0.22,
        baseline_volatility=0.20,
    )
    assert 0.70 <= receipt.adaptive_market_efficiency_index < 0.85
    assert receipt.regime == "CYCLICAL_ECOLOGICAL_EQUILIBRIUM"
    assert receipt.is_competitive_saturation is False
    assert receipt.is_punctuated_equilibrium is False
    assert receipt.verify_integrity() is True


def test_classify_amh_regime_direct() -> None:
    """Directly test classify_amh_regime boundary logic."""
    assert classify_amh_regime(0.90, 1.10, -0.2) == "HIGH_EFFICIENCY_COMPETITIVE_SATURATION"
    assert classify_amh_regime(0.35, 2.50, 1.80) == "BEHAVIORAL_PUNCTUATED_EQUILIBRIUM"
    assert classify_amh_regime(0.60, 1.80, 0.80) == "ADAPTIVE_ARBITRAGE_RECOVERY"
    assert classify_amh_regime(0.78, 1.30, 0.40) == "CYCLICAL_ECOLOGICAL_EQUILIBRIUM"


def test_tamper_signature_verification() -> None:
    """Verify HMAC-SHA256 signature detects any modified field."""
    receipt = calculate_adaptive_markets(
        ticker="MSFT",
        latest_price=420.0,
    )
    assert receipt.verify_integrity() is True

    # Tamper with numerical field
    tampered_dict = receipt.to_dict()
    tampered_dict["adaptive_market_efficiency_index"] = 0.01
    tampered_receipt = AdaptiveMarketsReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with signature
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["signature"] = "deadbeef11111111"
    tampered_receipt2 = AdaptiveMarketsReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_security_validation_ticker_and_bounds() -> None:
    """Ensure invalid tickers and out-of-bound arguments raise errors."""
    for mal in ["../etc/passwd", "AAPL'; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_adaptive_markets(ticker=mal, latest_price=100.0)

    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_adaptive_markets(ticker="AAPL", latest_price=-10.0)

    with pytest.raises(ValueError, match="lo_mackinlay_variance_ratio must be in"):
        calculate_adaptive_markets(ticker="AAPL", latest_price=100.0, lo_mackinlay_variance_ratio=3.50)

    with pytest.raises(ValueError, match="serial_autocorrelation_rho1 must be in"):
        calculate_adaptive_markets(ticker="AAPL", latest_price=100.0, serial_autocorrelation_rho1=0.99)

    with pytest.raises(ValueError, match="ecological_stress_drawdown must be in"):
        calculate_adaptive_markets(ticker="AAPL", latest_price=100.0, ecological_stress_drawdown=0.95)


def test_evaluate_amh_profile_with_observations(tmp_path: Path) -> None:
    """Verify end-to-end evaluation using synthetic database observations."""
    test_db = tmp_path / "test_amh.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="AAPL",
        exchange="XNAS",
        company_name="Apple Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(40):
        p = 150.0 + i * 0.8
        records.append({
            "ticker": "AAPL",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": p,
            "high": p + 1.0,
            "low": p - 1.0,
            "close": p,
            "adj_close": p,
            "volume": 1_200_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    receipt = evaluate_amh_profile(ticker="AAPL", db_path=test_db)
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == pytest.approx(181.2, abs=1e-2)
    assert receipt.adaptive_market_efficiency_index > 0.0
    assert receipt.verify_integrity() is True


def test_evaluate_amh_profile_overrides(tmp_path: Path) -> None:
    """Verify parameter overrides take precedence over inferred defaults."""
    test_db = tmp_path / "test_amh_override.db"
    db.init_db(test_db)

    receipt = evaluate_amh_profile(
        ticker="AAPL",
        db_path=test_db,
        lo_mackinlay_variance_ratio=1.01,
        serial_autocorrelation_rho1=0.01,
        ecological_stress_drawdown=0.02,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.lo_mackinlay_variance_ratio == 1.01
    assert receipt.serial_autocorrelation_rho1 == 0.01
    assert receipt.regime == "HIGH_EFFICIENCY_COMPETITIVE_SATURATION"
    assert receipt.is_competitive_saturation is True


def test_cli_amh_json_output(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    """Verify CLI amh command output in JSON mode."""
    test_db = tmp_path / "test_cli_amh.db"
    db.init_db(test_db)

    args = argparse.Namespace(
        ticker="MSFT",
        variance_ratio=1.10,
        autocorrelation=0.05,
        drawdown=0.08,
        volatility=0.22,
        baseline_volatility=0.20,
        json=True,
    )
    amh_command(args)

    out, _ = capsys.readouterr()
    data = json.loads(out)
    assert data["ticker"] == "MSFT"
    assert data["lo_mackinlay_variance_ratio"] == 1.10
    assert "signature" in data
    assert len(data["signature"]) == 64


def test_cli_amh_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI amh command standard formatted output."""
    args = argparse.Namespace(
        ticker="GOOGL",
        variance_ratio=1.05,
        autocorrelation=0.03,
        drawdown=0.05,
        volatility=0.20,
        baseline_volatility=0.20,
        json=False,
    )
    amh_command(args)

    out, _ = capsys.readouterr()
    assert "NextOmni Stock: Andrew W. Lo (2004, 2005) Adaptive Markets Engine" in out
    assert "Security / Ticker:            GOOGL" in out
    assert "Verified: True" in out


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify parameterized database queries and clean execution without SQL injection."""
    test_db = tmp_path / "test_amh_db.db"
    db.init_db(test_db)
    db.upsert_stock(
        ticker="NVDA",
        exchange="XNAS",
        company_name="Nvidia Corp.",
        currency="USD",
        timezone="America/New_York",
        db_path=test_db,
    )

    records = []
    for i in range(25):
        records.append({
            "ticker": "NVDA",
            "exchange": "XNAS",
            "session_date": f"2026-01-{i+1:02d}",
            "open": 120.0,
            "high": 125.0,
            "low": 118.0,
            "close": 122.0,
            "adj_close": 122.0,
            "volume": 2_000_000,
            "data_status": "END_OF_DAY",
            "provider": "MOCK",
        })
    db.save_observations(pd.DataFrame(records), db_path=test_db)

    obs = db.get_observations("NVDA", db_path=test_db)
    assert len(obs) == 25

