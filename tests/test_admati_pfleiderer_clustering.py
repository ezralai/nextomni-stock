"""Unit, boundary, security, and integration tests for Admati & Pfleiderer (1988)

Volume Clustering & Concentrated Trading Equilibrium Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.admati_pfleiderer_clustering import (
    AdmatiPfleidererReceipt,
    calculate_admati_pfleiderer_clustering,
    classify_clustering_regime,
    evaluate_admati_pfleiderer_profile,
    persist_admati_pfleiderer_record,
    verify_admati_audit_digest,
)
from src.cli import admati_pfleiderer_command
from src.security.validators import SecurityValidationError


def test_calculate_admati_pfleiderer_clustering_concentrated_cluster() -> None:
    """Verify high discretionary ratio yields concentrated liquidity cluster."""
    receipt = calculate_admati_pfleiderer_clustering(
        ticker="CLUST",
        latest_price=100.0,
        discretionary_liquidity_ratio=0.75,
        baseline_spread_bps=16.0,
    )

    assert receipt.ticker == "CLUST"
    assert receipt.clumping_index == 0.75
    assert receipt.market_depth_multiplier >= 1.58
    assert receipt.regime == "CONCENTRATED_LIQUIDITY_CLUSTER"
    assert receipt.is_concentrated_cluster is True
    assert receipt.kyle_lambda_reduction_pct >= 40.0
    assert receipt.execution_slippage_discount_bps > 5.0
    assert receipt.recommended_execution_window == "OPEN_CLOSE_AUCTION_CLUSTER"
    assert verify_admati_audit_digest(receipt) is True


def test_calculate_admati_pfleiderer_clustering_balanced_equilibrium() -> None:
    """Verify moderate discretionary ratio yields balanced intraday equilibrium."""
    receipt = calculate_admati_pfleiderer_clustering(
        ticker="BAL",
        latest_price=50.0,
        discretionary_liquidity_ratio=0.45,
    )

    assert receipt.regime == "BALANCED_INTRADAY_EQUILIBRIUM"
    assert receipt.is_balanced_equilibrium is True
    assert 1.24 <= receipt.market_depth_multiplier < 1.58
    assert receipt.recommended_execution_window == "CONTINUOUS_EQUILIBRIUM_SLICING"
    assert verify_admati_audit_digest(receipt) is True


def test_calculate_admati_pfleiderer_clustering_dispersed_transition() -> None:
    """Verify lower discretionary ratio yields dispersed low depth transition."""
    receipt = calculate_admati_pfleiderer_clustering(
        ticker="DISP",
        latest_price=30.0,
        discretionary_liquidity_ratio=0.25,
    )

    assert receipt.regime == "DISPERSED_LOW_DEPTH_TRANSITION"
    assert 1.08 <= receipt.market_depth_multiplier < 1.24
    assert receipt.recommended_execution_window == "CAUTIOUS_PASSIVE_LIMIT_ORDERS"
    assert verify_admati_audit_digest(receipt) is True


def test_calculate_admati_pfleiderer_clustering_midday_drought() -> None:
    """Verify minimal discretionary ratio yields midday high slippage drought."""
    receipt = calculate_admati_pfleiderer_clustering(
        ticker="DROUG",
        latest_price=20.0,
        discretionary_liquidity_ratio=0.10,
    )

    assert receipt.regime == "HIGH_SLIPPAGE_MIDDAY_DROUGHT"
    assert receipt.is_midday_drought is True
    assert receipt.market_depth_multiplier < 1.08
    assert receipt.recommended_execution_window == "AVOID_MIDDAY_EXECUTION"
    assert verify_admati_audit_digest(receipt) is True


def test_monotonicity_properties() -> None:
    """Verify mathematical monotonicity: higher discretionary ratio increases depth and savings."""
    ratios = [0.10, 0.25, 0.40, 0.55, 0.70, 0.85]
    depths = []
    reductions = []
    savings = []

    for r in ratios:
        rec = calculate_admati_pfleiderer_clustering(ticker="MONO", discretionary_liquidity_ratio=r)
        depths.append(rec.market_depth_multiplier)
        reductions.append(rec.kyle_lambda_reduction_pct)
        savings.append(rec.execution_slippage_discount_bps)

    assert all(depths[i] < depths[i + 1] for i in range(len(depths) - 1))
    assert all(reductions[i] < reductions[i + 1] for i in range(len(reductions) - 1))
    assert all(savings[i] < savings[i + 1] for i in range(len(savings) - 1))


def test_tamper_signature_verification() -> None:
    """Verify cryptographic audit digest detects tampered records."""
    receipt = calculate_admati_pfleiderer_clustering(ticker="AUDIT", discretionary_liquidity_ratio=0.50)
    assert verify_admati_audit_digest(receipt) is True

    tampered = AdmatiPfleidererReceipt(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        discretionary_liquidity_ratio=receipt.discretionary_liquidity_ratio,
        non_discretionary_variance=receipt.non_discretionary_variance,
        informed_trader_count=receipt.informed_trader_count,
        clumping_index=receipt.clumping_index,
        market_depth_multiplier=receipt.market_depth_multiplier + 0.50,  # Tampered!
        kyle_lambda_reduction_pct=receipt.kyle_lambda_reduction_pct,
        execution_slippage_discount_bps=receipt.execution_slippage_discount_bps,
        recommended_execution_window=receipt.recommended_execution_window,
        regime=receipt.regime,
        is_concentrated_cluster=receipt.is_concentrated_cluster,
        is_balanced_equilibrium=receipt.is_balanced_equilibrium,
        is_midday_drought=receipt.is_midday_drought,
        optimal_execution_timing=receipt.optimal_execution_timing,
        summary=receipt.summary,
        audit_digest=receipt.audit_digest,
        calculated_at=receipt.calculated_at,
    )

    assert verify_admati_audit_digest(tampered) is False


def test_security_validation_ticker_and_bounds() -> None:
    """Verify security validation rejects invalid tickers or out-of-bounds parameters."""
    with pytest.raises(SecurityValidationError):
        calculate_admati_pfleiderer_clustering(ticker="INVALID_TICKER_TOOLONG")

    with pytest.raises(SecurityValidationError):
        calculate_admati_pfleiderer_clustering(ticker="AAPL", latest_price=-5.0)

    with pytest.raises(SecurityValidationError):
        calculate_admati_pfleiderer_clustering(ticker="AAPL", discretionary_liquidity_ratio=0.99)

    with pytest.raises(SecurityValidationError):
        calculate_admati_pfleiderer_clustering(ticker="AAPL", non_discretionary_variance=0.0)

    with pytest.raises(SecurityValidationError):
        calculate_admati_pfleiderer_clustering(ticker="AAPL", informed_trader_count=0)


def test_evaluate_admati_pfleiderer_profile_synthetic_observations() -> None:
    """Verify profile evaluation with synthetic price and volume data."""
    np.random.seed(42)
    base_price = 150.0
    returns = np.random.normal(0.0004, 0.016, 70)
    prices = [float(base_price * np.exp(np.sum(returns[:i]))) for i in range(1, 71)]
    volumes = [float(np.random.uniform(800_000, 3_000_000)) for _ in range(70)]

    receipt = evaluate_admati_pfleiderer_profile(
        ticker="SYNTH",
        prices=prices,
        volumes=volumes,
    )

    assert receipt.ticker == "SYNTH"
    assert 0.10 <= receipt.clumping_index <= 0.85
    assert receipt.market_depth_multiplier >= 1.0
    assert verify_admati_audit_digest(receipt) is True


def test_evaluate_admati_pfleiderer_profile_overrides() -> None:
    """Verify profile evaluation honors explicit discretionary ratio override."""
    receipt = evaluate_admati_pfleiderer_profile(
        ticker="OVRD",
        discretionary_ratio_override=0.72,
    )

    assert receipt.ticker == "OVRD"
    assert receipt.clumping_index == 0.72
    assert receipt.is_concentrated_cluster is True


def test_cli_admati_pfleiderer_human_readable(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs human-readable table."""
    args = argparse.Namespace(
        ticker="AAPL",
        discretionary_ratio=0.65,
        informed_count=4,
        spread_bps=12.0,
        json=False,
    )

    admati_pfleiderer_command(args)
    captured = capsys.readouterr()

    assert "NextOmni Stock: Admati-Pfleiderer (1988) Volume Clustering Engine" in captured.out
    assert "AAPL" in captured.out
    assert "Market Depth Multiplier:" in captured.out
    assert "Admati-Pfleiderer Clumping Index:" in captured.out
    assert "Audit Digest (HMAC):" in captured.out


def test_cli_admati_pfleiderer_json_output(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI handler outputs valid JSON."""
    args = argparse.Namespace(
        ticker="MSFT",
        discretionary_ratio=0.55,
        informed_count=3,
        spread_bps=10.0,
        json=True,
    )

    admati_pfleiderer_command(args)
    captured = capsys.readouterr()

    data = json.loads(captured.out)
    assert data["ticker"] == "MSFT"
    assert "clumping_index" in data
    assert "market_depth_multiplier" in data
    assert "kyle_lambda_reduction_pct" in data
    assert "audit_digest" in data


def test_database_persistence_and_parameterized_query(tmp_path: Path) -> None:
    """Verify SQLite persistence and parameterized retrieval."""
    test_db = tmp_path / "test_admati.db"

    receipt = calculate_admati_pfleiderer_clustering(
        ticker="NVDA",
        latest_price=120.0,
        discretionary_liquidity_ratio=0.60,
    )

    persist_admati_pfleiderer_record(receipt, db_path=test_db)

    conn = sqlite3.connect(test_db)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticker, latest_price, clumping_index, market_depth_multiplier, audit_digest FROM admati_pfleiderer_records WHERE ticker = ?",
        ("NVDA",),
    )
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "NVDA"
    assert row[1] == 120.0
    assert row[2] == 0.60
    assert row[4] == receipt.audit_digest
