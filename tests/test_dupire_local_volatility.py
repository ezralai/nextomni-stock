"""Tests for Bruno Dupire (1994) Non-Parametric Local Volatility Surface Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.dupire_local_volatility import (
    _build_hmac_payload,
    calculate_dupire_local_volatility,
    evaluate_dupire_profile_from_prices,
    persist_dupire_record,
    verify_hmac_signature,
)


def test_flat_implied_vol_converges_to_constant_local_vol() -> None:
    """Under a flat volatility surface, Dupire local volatility matches implied volatility."""
    atm_vol = 0.20
    rec = calculate_dupire_local_volatility(
        spot_price=100.0,
        strike_price=100.0,
        time_to_maturity=0.50,
        risk_free_rate=0.03,
        dividend_yield=0.0,
        atm_vol=atm_vol,
        skew_slope=0.0,
        vol_of_vol=0.0,
        ticker="SPY_FLAT",
    )
    # Local vol equals implied vol under zero skew
    assert rec.local_vol == pytest.approx(atm_vol, rel=0.03)
    assert rec.local_implied_ratio == pytest.approx(1.0, rel=0.03)
    assert rec.calendar_arbitrage_free is True
    assert rec.butterfly_arbitrage_free is True
    assert rec.risk_neutral_density > 0.0
    assert rec.regime == "STABLE_LOCAL_DIFFUSION"


def test_negative_skew_amplifies_downward_local_volatility() -> None:
    """Negative equity skew amplifies local volatility at lower strikes (Dupire 2x skew rule)."""
    # Downside strike K = 90
    rec_down = calculate_dupire_local_volatility(
        spot_price=100.0,
        strike_price=90.0,
        time_to_maturity=0.50,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        atm_vol=0.20,
        skew_slope=-0.25,
        vol_of_vol=0.20,
        ticker="SPY_SKEW_DOWN",
    )
    # In negative skew regimes, local volatility expands faster than implied vol
    assert rec_down.local_vol > rec_down.implied_vol
    assert rec_down.local_implied_ratio > 1.05
    assert rec_down.regime == "ELEVATED_LOCAL_VOL_SKEW_EXPANSION"

    # Upside strike K = 110
    rec_up = calculate_dupire_local_volatility(
        spot_price=100.0,
        strike_price=110.0,
        time_to_maturity=0.50,
        risk_free_rate=0.04,
        dividend_yield=0.0,
        atm_vol=0.20,
        skew_slope=-0.25,
        vol_of_vol=0.20,
        ticker="SPY_SKEW_UP",
    )
    # Upside local volatility is compressed below implied volatility
    assert rec_up.local_vol < rec_up.implied_vol
    assert rec_up.local_implied_ratio < 0.95


def test_dupire_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_dupire_local_volatility(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Spot"):
        calculate_dupire_local_volatility(spot_price=-100.0, ticker="SPY")

    with pytest.raises(ValueError, match="ATM volatility"):
        calculate_dupire_local_volatility(atm_vol=-0.2, ticker="SPY")

    # Evaluate profile from simulated price sequence
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_dupire_profile_from_prices(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        spot_price=receipt.spot_price,
        strike_price=receipt.strike_price,
        time_to_maturity=receipt.time_to_maturity,
        implied_vol=receipt.implied_vol,
        local_vol=receipt.local_vol,
        local_implied_ratio=receipt.local_implied_ratio,
        dC_dT=receipt.dC_dT,
        d2C_dK2=receipt.d2C_dK2,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "dupire_test.db"
    assert persist_dupire_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, spot_price, local_vol, regime, hmac_signature FROM dupire_local_volatility_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == pytest.approx(receipt.spot_price)
    assert rows[0][4] == receipt.hmac_signature


def test_cli_dupire_vol_json_and_aliases() -> None:
    """Verify CLI dupire-vol and aliases emit valid JSON receipt."""
    for alias in ("dupire-vol", "dupire", "dupire-1994", "local-vol"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--spot",
                "100.0",
                "--strike",
                "95.0",
                "--vol",
                "0.20",
                "--skew",
                "-0.15",
                "--ttm",
                "0.50",
                "--rf",
                "0.04",
                "--div",
                "0.01",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["local_vol"] > 0.0
        assert len(data["hmac_signature"]) == 64

