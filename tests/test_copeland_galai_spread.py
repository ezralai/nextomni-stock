"""Tests for Copeland & Galai (1983) Option-Theoretic Bid-Ask Spread Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from src.analytics.copeland_galai_spread import (
    _build_hmac_payload,
    calculate_copeland_galai_spread,
    evaluate_copeland_galai_profile,
    persist_copeland_galai_record,
    verify_hmac_signature,
)


def test_zero_informed_traders_collapses_spread_and_zero_profit_balance() -> None:
    """Zero informed arrival collapses spread to 0; positive pi_I satisfies exact dealer zero-profit."""
    rec_zero = calculate_copeland_galai_spread(
        spot_price=100.0,
        volatility_annual=0.25,
        informed_prob=0.0,
        ticker="SPY",
    )
    assert rec_zero.equilibrium_spread == pytest.approx(0.0, abs=1e-12)
    assert rec_zero.bid_price == pytest.approx(100.0, abs=1e-12)
    assert rec_zero.ask_price == pytest.approx(100.0, abs=1e-12)
    assert rec_zero.regime == "ZERO_ADVERSE_SELECTION_FRICTIONLESS"

    rec = calculate_copeland_galai_spread(
        spot_price=100.0,
        volatility_annual=0.25,
        informed_prob=0.15,
        ticker="SPY",
    )
    assert rec.equilibrium_residual < 1e-10
    assert rec.ask_price - rec.bid_price == pytest.approx(rec.equilibrium_spread, abs=1e-12)
    assert rec.spread_bps > 1.0


def test_spread_monotonicity_in_informed_prob_and_volatility() -> None:
    """Spread widens strictly with both informed share pi_I and fundamental volatility sigma."""
    low_pi = calculate_copeland_galai_spread(100.0, volatility_annual=0.25, informed_prob=0.10)
    high_pi = calculate_copeland_galai_spread(100.0, volatility_annual=0.25, informed_prob=0.45)
    assert high_pi.equilibrium_spread > low_pi.equilibrium_spread
    assert high_pi.spread_bps > low_pi.spread_bps
    assert high_pi.regime in ("ELEVATED_INFORMATION_SPREAD", "SEVERE_ADVERSE_SELECTION_ILLIQUIDITY")

    low_vol = calculate_copeland_galai_spread(100.0, volatility_annual=0.15, informed_prob=0.20)
    high_vol = calculate_copeland_galai_spread(100.0, volatility_annual=0.45, informed_prob=0.20)
    assert high_vol.equilibrium_spread > low_vol.equilibrium_spread
    assert high_vol.spread_sensitivity_per_1pct_pi > 0.0


def test_empirical_profile_and_validation_guards_hmac_sqlite(tmp_path) -> None:
    """Test empirical calibration, invalid bounds, HMAC verification, and SQLite audit."""
    prices = [100.0 + (i % 5) * 0.4 for i in range(40)]
    emp = evaluate_copeland_galai_profile(prices, ticker="QQQ", informed_prob=0.15)
    assert emp.ticker == "QQQ"
    assert emp.equilibrium_spread > 0.0

    emp_short = evaluate_copeland_galai_profile([100.0, 101.0], ticker="MSFT")
    assert emp_short.spot_price == pytest.approx(100.0, abs=1e-6)

    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_copeland_galai_spread(100.0, ticker="../BAD")
    with pytest.raises(ValueError, match="spot_price"):
        calculate_copeland_galai_spread(-50.0, ticker="SPY")
    with pytest.raises(ValueError, match="informed_prob"):
        calculate_copeland_galai_spread(100.0, informed_prob=0.95, ticker="SPY")

    rec = calculate_copeland_galai_spread(100.0, 0.25, 0.20, ticker="AAPL")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        spot_price=rec.spot_price,
        volatility_annual=rec.volatility_annual,
        informed_prob=rec.informed_prob,
        equilibrium_spread=rec.equilibrium_spread,
        spread_bps=rec.spread_bps,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("AAPL", "NVDA"), rec.hmac_signature) is False

    db_file = str(tmp_path / "cg_test.db")
    assert persist_copeland_galai_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, regime FROM copeland_galai_audits").fetchall()
    assert rows == [("AAPL", rec.regime)]


def test_cli_copeland_galai_json_and_aliases() -> None:
    """Verify CLI copeland-galai and aliases emit valid JSON receipt."""
    for alias in ("copeland-galai", "option-spread", "cg-1983"):
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
                "--vol",
                "0.25",
                "--informed-prob",
                "0.15",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["equilibrium_residual"] < 1e-10
        assert data["spread_bps"] > 1.0

