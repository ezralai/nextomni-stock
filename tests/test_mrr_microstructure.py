"""Tests for Ananth Madhavan, Matthew Richardson & Mark Roomans (1997) MRR Microstructure Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.mrr_microstructure import (
    _build_hmac_payload,
    calculate_mrr_microstructure,
    evaluate_mrr_from_prices,
    persist_mrr_record,
    verify_hmac_signature,
)


def test_structural_parameter_recovery_from_synthetic_trades() -> None:
    """MRR model successfully recovers information impact theta and order processing friction phi0."""
    np.random.seed(42)
    n = 250

    # 1. Simulate Markov order flow with rho ~ 0.65 (phi = 2*rho - 1 ~ 0.30)
    x = np.ones(n)
    for t in range(1, n):
        x[t] = x[t - 1] if np.random.rand() < 0.65 else -x[t - 1]

    # True structural parameters:
    theta_true = 0.03
    phi0_true = 0.05
    phi_true = 0.30

    # Price change: Delta p_t = (theta + phi0) * x_t - (phi*theta + phi0) * x_{t-1} + noise
    b1_true = theta_true + phi0_true   # 0.08
    b2_true = -(phi_true * theta_true + phi0_true)  # -0.059

    dp = np.zeros(n)
    dp[1:] = b1_true * x[1:] + b2_true * x[:-1] + np.random.normal(0, 0.002, size=n - 1)

    rec = calculate_mrr_microstructure(
        dp,
        x,
        ticker="SPY_MRR",
        reference_price=100.0,
    )

    # Effective spread S = 2 * (theta + phi0) = 2 * 0.08 = $0.16 = 16.0 bps on $100
    assert rec.effective_spread_bps == pytest.approx(16.0, rel=0.15)
    assert rec.theta_dollar == pytest.approx(theta_true, abs=0.015)
    assert rec.phi0_dollar == pytest.approx(phi0_true, abs=0.015)
    assert rec.continuation_prob_rho == pytest.approx(0.65, abs=0.08)
    assert rec.order_autocorrelation_phi == pytest.approx(0.30, abs=0.16)
    assert rec.adverse_selection_ratio + rec.order_processing_ratio == pytest.approx(1.0, abs=1e-6)
    assert rec.r_squared > 0.40


def test_mrr_regime_classification() -> None:
    """Adverse selection dominant vs order processing dominant regimes are classified correctly."""
    np.random.seed(42)
    n = 100
    x = np.sign(np.random.normal(size=n))

    # 1. Information asymmetry dominant (theta=0.08, phi0=0.01)
    dp_info = np.zeros(n)
    dp_info[1:] = 0.09 * x[1:] - 0.01 * x[:-1]
    rec_info = calculate_mrr_microstructure(dp_info, x, ticker="INFO_HEAVY", reference_price=100.0)
    assert rec_info.adverse_selection_ratio >= 0.60
    assert rec_info.regime == "INFORMATION_ASYMMETRY_DOMINANT"

    # 2. Order processing friction dominant (theta=0.01, phi0=0.08)
    dp_friction = np.zeros(n)
    dp_friction[1:] = 0.09 * x[1:] - 0.08 * x[:-1]
    rec_fric = calculate_mrr_microstructure(dp_friction, x, ticker="FRIC_HEAVY", reference_price=100.0)
    assert rec_fric.order_processing_ratio >= 0.65
    assert rec_fric.regime == "ORDER_PROCESSING_FRICTION_DOMINANT"


def test_mrr_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_mrr_microstructure([0.01] * 40, [1] * 40, ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_mrr_microstructure([0.01] * 40, [1] * 30, ticker="SPY")

    with pytest.raises(ValueError, match="At least 30"):
        calculate_mrr_microstructure([0.01] * 20, [1] * 20, ticker="SPY")

    # Evaluate profile from simulated price series
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_mrr_from_prices(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        sample_size=receipt.sample_size,
        theta_dollar=receipt.theta_dollar,
        phi0_dollar=receipt.phi0_dollar,
        effective_spread_bps=receipt.effective_spread_bps,
        adverse_selection_ratio=receipt.adverse_selection_ratio,
        order_processing_ratio=receipt.order_processing_ratio,
        order_autocorrelation_phi=receipt.order_autocorrelation_phi,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "mrr_test.db"
    assert persist_mrr_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, sample_size, effective_spread_bps, regime, hmac_signature FROM mrr_microstructure_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == receipt.sample_size
    assert rows[0][4] == receipt.hmac_signature


def test_cli_mrr_model_json_and_aliases() -> None:
    """Verify CLI mrr-model and aliases emit valid JSON receipt."""
    for alias in ("mrr-model", "mrr", "mrr-1997", "order-flow-impact"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--spread",
                "15.0",
                "--theta-ratio",
                "0.40",
                "--autocorr",
                "0.25",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["effective_spread_bps"] > 0.0
        assert len(data["hmac_signature"]) == 64

