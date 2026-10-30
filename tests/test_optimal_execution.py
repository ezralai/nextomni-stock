"""Unit test suite for Almgren & Chriss (2000) Optimal Execution Trajectory Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.optimal_execution import (
    ExecutionTrajectoryReceipt,
    compute_almgren_chriss_trajectory,
    estimate_market_impact_parameters,
    evaluate_execution_frontier,
    evaluate_optimal_execution_plan,
)
from src.cli import optimal_execution_command


def test_inventory_conservation_and_monotonicity() -> None:
    """Verify that execution trajectory strictly conserves inventory and decays monotonically."""
    target_shares = 10_000.0
    execution_days = 5

    schedule, kappa, half_life, e_perm, e_temp, e_tot, v_short = compute_almgren_chriss_trajectory(
        target_shares=target_shares,
        execution_days=execution_days,
        initial_price=150.0,
        daily_volatility_pct=2.0,
        daily_volume=1_000_000.0,
        risk_aversion_lambda=1e-6,
    )

    assert len(schedule) == execution_days
    # Sum of trades must equal target shares
    total_traded = sum(step.trade_shares for step in schedule)
    assert total_traded == pytest.approx(target_shares, rel=1e-5)

    # Final inventory remaining must be 0
    assert schedule[-1].inventory_remaining == pytest.approx(0.0, abs=1e-5)

    # Monotonic inventory decrease
    prev_inv = target_shares
    for step in schedule:
        assert step.inventory_start == pytest.approx(prev_inv, rel=1e-5)
        assert step.inventory_remaining < step.inventory_start
        assert step.trade_shares > 0
        prev_inv = step.inventory_remaining

    # Costs and risks must be positive
    assert e_perm > 0.0
    assert e_temp > 0.0
    assert e_tot == pytest.approx(e_perm + e_temp)
    assert v_short > 0.0
    assert kappa > 0.0
    assert half_life > 0.0


def test_risk_neutral_twap_convergence() -> None:
    """Verify that lambda = 0 converges to a pure linear TWAP schedule with equal trade sizes."""
    target_shares = 5_000.0
    execution_days = 5

    schedule, kappa, half_life, e_perm, e_temp, e_tot, v_short = compute_almgren_chriss_trajectory(
        target_shares=target_shares,
        execution_days=execution_days,
        initial_price=100.0,
        daily_volatility_pct=1.5,
        daily_volume=500_000.0,
        risk_aversion_lambda=0.0,  # Risk neutral
    )

    assert kappa == 0.0
    expected_daily_trade = target_shares / float(execution_days)
    for step in schedule:
        assert step.trade_shares == pytest.approx(expected_daily_trade, rel=1e-5)


def test_risk_averse_front_loading() -> None:
    """Verify that elevated risk aversion causes front-loading of trades to minimize timing risk."""
    target_shares = 10_000.0
    execution_days = 5

    schedule, kappa, _, _, _, _, _ = compute_almgren_chriss_trajectory(
        target_shares=target_shares,
        execution_days=execution_days,
        initial_price=200.0,
        daily_volatility_pct=3.0,
        daily_volume=800_000.0,
        risk_aversion_lambda=1e-4,  # High risk aversion
    )

    trades = [step.trade_shares for step in schedule]
    # Trade sizes must be strictly decreasing (front-loaded)
    for i in range(len(trades) - 1):
        assert trades[i] > trades[i + 1]

    assert kappa > 0.15



def test_efficient_frontier_tradeoff() -> None:
    """Verify that higher lambda reduces variance of shortfall at the cost of higher expected shortfall."""
    frontier = evaluate_execution_frontier(
        target_shares=10_000.0,
        execution_days=5,
        initial_price=150.0,
        daily_volatility_pct=2.0,
        daily_volume=1_000_000.0,
        num_points=5,
    )

    assert len(frontier) == 5
    # As lambda increases:
    # shortfall_std_dev_dollars must decrease (lower timing risk)
    # expected_shortfall_dollars must increase (more aggressive early trading incurs higher temporary impact)
    for i in range(len(frontier) - 1):
        assert frontier[i]["shortfall_std_dev_dollars"] >= frontier[i + 1]["shortfall_std_dev_dollars"]
        assert frontier[i]["expected_shortfall_dollars"] <= frontier[i + 1]["expected_shortfall_dollars"]


def test_single_step_execution_boundary() -> None:
    """Verify edge case of a single-step execution horizon (N=1)."""
    schedule, kappa, half_life, e_perm, e_temp, e_tot, v_short = compute_almgren_chriss_trajectory(
        target_shares=1_000.0,
        execution_days=1,
        initial_price=50.0,
        daily_volatility_pct=1.0,
        daily_volume=200_000.0,
    )

    assert len(schedule) == 1
    assert schedule[0].trade_shares == 1_000.0
    assert schedule[0].inventory_remaining == 0.0
    assert v_short == 0.0  # Zero overnight timing risk for immediate execution


def test_receipt_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature verification and tamper detection."""
    receipt = ExecutionTrajectoryReceipt(
        ticker="AAPL",
        target_shares=10_000.0,
        execution_days=5,
        initial_price=175.0,
        total_market_value=1_750_000.0,
        daily_volatility_pct=1.8,
        daily_volume=5_000_000.0,
        risk_aversion_lambda=1e-6,
        urgency_parameter_kappa=0.15,
        half_life_days=4.62,
        expected_permanent_impact_dollars=120.0,
        expected_temporary_impact_dollars=350.0,
        expected_total_shortfall_dollars=470.0,
        expected_total_shortfall_bps=2.69,
        shortfall_variance_dollars_sq=500_000.0,
        shortfall_std_dev_dollars=707.11,
        trajectory_schedule=[],
        summary="Test summary",
    )

    assert receipt.verify_integrity() is True

    # Tamper with expected total shortfall
    receipt.expected_total_shortfall_dollars = 999.0
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify input validation on tickers, shares, days, and lambda."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_optimal_execution_plan("NVDA; DROP TABLE", target_shares=100)

    with pytest.raises(ValueError, match="Target shares must be at least 1"):
        evaluate_optimal_execution_plan("NVDA", target_shares=0.5)

    with pytest.raises(ValueError, match="Execution days must be between 1 and 60"):
        evaluate_optimal_execution_plan("NVDA", target_shares=100, execution_days=0)

    with pytest.raises(ValueError, match="Risk aversion lambda must be >= 0"):
        evaluate_optimal_execution_plan("NVDA", target_shares=100, risk_aversion_lambda=-1.0)


def test_cli_optimal_execution_command(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI optimal-execution command in text and JSON output modes."""
    n = 60
    dates = [f"2026-01-{i+1:02d}" for i in range(n)]
    prices = [180.0 + (i * 0.2) for i in range(n)]
    volumes = [1_000_000.0 for _ in range(n)]
    df_obs = pd.DataFrame({"session_date": dates, "adj_close": prices, "close": prices, "volume": volumes})

    with patch("src.data.db.get_observations", return_value=df_obs):
        # 1. Text mode
        args = argparse.Namespace(
            ticker="AAPL",
            shares=5_000.0,
            days=5,
            risk_aversion=1e-6,
            json=False,
        )
        optimal_execution_command(args)
        out = capsys.readouterr().out
        assert "Almgren & Chriss (2000) Optimal Execution Engine" in out
        assert "Security / Ticker:            AAPL" in out
        assert "Expected Total Shortfall:" in out
        assert "Discrete Execution Trajectory Schedule:" in out

        # 2. JSON mode
        args_json = argparse.Namespace(
            ticker="AAPL",
            shares=5_000.0,
            days=5,
            risk_aversion=1e-6,
            json=True,
        )
        optimal_execution_command(args_json)
        out_json = capsys.readouterr().out
        parsed = json.loads(out_json)
        assert parsed["ticker"] == "AAPL"
        assert parsed["target_shares"] == 5000.0
        assert "urgency_parameter_kappa" in parsed
        assert "signature" in parsed
        assert len(parsed["trajectory_schedule"]) == 5
