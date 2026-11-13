"""Tests for Roger D. Huang & Hans R. Stoll (1997) Three-Way Spread Decomposition Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.huang_stoll_spread import (
    _build_hmac_payload,
    calculate_huang_stoll_decomposition,
    evaluate_huang_stoll_from_prices,
    persist_huang_stoll_record,
    verify_hmac_signature,
)


def test_structural_parameter_recovery_and_normalization() -> None:
    """Huang & Stoll decomposition accurately recovers effective spread and normalizes components to 100%."""
    np.random.seed(42)
    n = 200
    # Generate Markov order flow with continuation probability pi ~ 0.60
    q = np.ones(n)
    for t in range(1, n):
        if np.random.rand() > 0.60:
            q[t] = -q[t - 1]
        else:
            q[t] = q[t - 1]

    # Prescribe true structural parameters: S = $0.10, alpha = 0.30, beta = 0.20, gamma = 0.50
    s_half = 0.05
    alpha_true = 0.30
    beta_true = 0.20
    gamma_true = 0.50

    # Delta P_t = (S/2)*q_t - gamma*(S/2)*q_{t-1} - beta*(S/2)*q_{t-2} + noise
    dp = np.zeros(n)
    dp[2:] = s_half * q[2:] - gamma_true * s_half * q[1:-1] - beta_true * s_half * q[:-2] + np.random.normal(0, 0.002, size=n - 2)

    rec = calculate_huang_stoll_decomposition(
        dp,
        q,
        ticker="SPY_STRUCTURAL",
        reference_price=100.0,
    )

    # 1. Proportions sum to 1.0 (100%)
    assert rec.adverse_selection_pct + rec.inventory_holding_pct + rec.order_processing_pct == pytest.approx(1.0, abs=1e-6)

    # 2. Recovered spread components
    assert rec.effective_spread_bps == pytest.approx(10.0, rel=0.15)  # $0.10 on $100 = 10 bps
    assert rec.adverse_selection_pct == pytest.approx(alpha_true, abs=0.12)
    assert rec.inventory_holding_pct == pytest.approx(beta_true, abs=0.12)
    assert rec.order_processing_pct == pytest.approx(gamma_true, abs=0.12)
    assert rec.trade_continuation_prob == pytest.approx(0.60, abs=0.08)
    assert rec.r_squared > 0.50


def test_adverse_selection_vs_processing_dominance_regimes() -> None:
    """Extreme cost regimes are correctly categorized."""
    np.random.seed(42)
    n = 100
    q = np.sign(np.random.normal(size=n))
    s_half = 0.08

    # High adverse selection (alpha = 0.70, beta = 0.10, gamma = 0.20)
    dp_toxic = np.zeros(n)
    dp_toxic[2:] = s_half * q[2:] - 0.20 * s_half * q[1:-1] - 0.10 * s_half * q[:-2]

    rec_toxic = calculate_huang_stoll_decomposition(dp_toxic, q, ticker="TOXIC_FLOW", reference_price=100.0)
    assert rec_toxic.adverse_selection_pct >= 0.50
    assert rec_toxic.regime == "ADVERSE_SELECTION_DOMINANT"

    # High order processing profit (alpha = 0.10, beta = 0.10, gamma = 0.80)
    dp_profit = np.zeros(n)
    dp_profit[2:] = s_half * q[2:] - 0.80 * s_half * q[1:-1] - 0.10 * s_half * q[:-2]

    rec_profit = calculate_huang_stoll_decomposition(dp_profit, q, ticker="PROFIT_FLOW", reference_price=100.0)
    assert rec_profit.order_processing_pct >= 0.60
    assert rec_profit.regime == "ORDER_PROCESSING_PROFIT_DOMINANT"


def test_huang_stoll_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate parameter boundaries, HMAC audit sealing, and SQLite logging."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_huang_stoll_decomposition([0.01] * 40, [1] * 40, ticker="BAD TICKER!")

    with pytest.raises(ValueError, match="Length mismatch"):
        calculate_huang_stoll_decomposition([0.01] * 40, [1] * 30, ticker="SPY")

    with pytest.raises(ValueError, match="At least 30"):
        calculate_huang_stoll_decomposition([0.01] * 20, [1] * 20, ticker="SPY")

    # Evaluate profile from simulated price series
    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.0006)))
    receipt = evaluate_huang_stoll_from_prices(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        sample_size=receipt.sample_size,
        effective_spread_bps=receipt.effective_spread_bps,
        adverse_selection_pct=receipt.adverse_selection_pct,
        inventory_holding_pct=receipt.inventory_holding_pct,
        order_processing_pct=receipt.order_processing_pct,
        trade_continuation_prob=receipt.trade_continuation_prob,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "huang_stoll_test.db"
    assert persist_huang_stoll_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, sample_size, effective_spread_bps, regime, hmac_signature FROM huang_stoll_spread_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == receipt.sample_size
    assert rows[0][4] == receipt.hmac_signature


def test_cli_huang_stoll_json_and_aliases() -> None:
    """Verify CLI huang-stoll and aliases emit valid JSON receipt."""
    for alias in ("huang-stoll", "hs-1997", "spread-decomp", "three-way-spread"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--spread",
                "12.5",
                "--alpha",
                "0.40",
                "--beta",
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

