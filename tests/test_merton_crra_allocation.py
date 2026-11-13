"""Tests for Samuelson (1969) & Merton (1969/1971) CRRA Lifetime Portfolio Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.merton_crra_allocation import (
    _build_hmac_payload,
    calculate_merton_crra_allocation,
    evaluate_merton_crra_profile,
    persist_merton_crra_record,
    verify_hmac_signature,
)


def test_log_utility_kelly_limit_and_myopic_share() -> None:
    """When gamma=1 (Log Utility), hedging demand is zero and c/W equals discount rate rho."""
    rec = calculate_merton_crra_allocation(
        expected_return=0.10,
        volatility=0.20,
        risk_free_rate=0.04,
        gamma_risk_aversion=1.0,
        time_discount_rate=0.03,
        state_correlation=-0.50,
        ticker="KELLY_FUND",
    )
    # w_myopic = (0.10 - 0.04) / (1.0 * 0.04) = 1.50
    assert rec.merton_share_myopic == pytest.approx(1.50, abs=1e-6)
    assert rec.hedging_demand == pytest.approx(0.0, abs=1e-9)
    assert rec.total_risky_weight == pytest.approx(1.50, abs=1e-6)
    assert rec.consumption_wealth_ratio == pytest.approx(0.03, abs=1e-6)
    assert rec.regime == "LEVERED_GROWTH_KELLY"


def test_intertemporal_hedging_demand_and_regimes() -> None:
    """Risk-averse investors (gamma > 1) with negative state correlation have positive hedge demand."""
    rec_hedged = calculate_merton_crra_allocation(
        expected_return=0.08,
        volatility=0.16,
        risk_free_rate=0.03,
        gamma_risk_aversion=4.0,
        time_discount_rate=0.03,
        state_correlation=-0.40,
        state_volatility=0.08,
        horizon_factor=1.5,
        ticker="BALANCED",
    )
    # gamma > 1 and rho_state < 0 implies positive hedging tilt
    assert rec_hedged.hedging_demand > 0.0
    assert rec_hedged.total_risky_weight > rec_hedged.merton_share_myopic
    assert rec_hedged.regime in ("BALANCED_CRRA_INTERTEMPORAL", "CONSERVATIVE_INCOME_PRESERVATION")

    # High risk aversion and low excess return yields defensive cash tilt
    rec_def = calculate_merton_crra_allocation(
        expected_return=0.042,
        volatility=0.20,
        risk_free_rate=0.040,
        gamma_risk_aversion=10.0,
        state_correlation=0.0,
        ticker="DEFENSIVE",
    )
    assert rec_def.total_risky_weight < 0.15
    assert rec_def.regime == "DEFENSIVE_CASH_TILT"


def test_merton_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite audit logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_merton_crra_allocation(ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="gamma_risk_aversion"):
        calculate_merton_crra_allocation(gamma_risk_aversion=-1.0, ticker="SPY")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0008)))
    receipt = evaluate_merton_crra_profile(prices, ticker="SPY", gamma_risk_aversion=3.0)
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        gamma_risk_aversion=receipt.gamma_risk_aversion,
        time_discount_rate=receipt.time_discount_rate,
        expected_return=receipt.expected_return,
        volatility=receipt.volatility,
        risk_free_rate=receipt.risk_free_rate,
        total_risky_weight=receipt.total_risky_weight,
        consumption_wealth_ratio=receipt.consumption_wealth_ratio,
        certainty_equivalent_return=receipt.certainty_equivalent_return,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "merton_crra_test.db"
    assert persist_merton_crra_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, gamma_risk_aversion, regime, hmac_signature FROM merton_crra_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 3.0
    assert rows[0][3] == receipt.hmac_signature


def test_cli_merton_share_json_and_aliases() -> None:
    """Verify CLI merton-share and aliases emit valid JSON receipt."""
    for alias in ("merton-share", "merton-1969", "samuelson-merton", "crra-allocation"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--mu",
                "0.09",
                "--vol",
                "0.16",
                "--rf",
                "0.04",
                "--gamma",
                "3.0",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["gamma_risk_aversion"] == 3.0
        assert data["merton_share_myopic"] > 0.0
        assert len(data["hmac_signature"]) == 64

