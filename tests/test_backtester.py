"""Unit and quantitative verification tests for strategy backtesting engine."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.backtester import (
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    run_forecast_backtest,
)
from src.cli import backtest_command
from src.data.db import init_db, save_observations, upsert_stock


def test_sharpe_ratio_calculations():
    """Verify Sharpe ratio calculation and division-by-zero safeguards."""
    # Constant returns (zero standard deviation)
    const_returns = np.array([0.01, 0.01, 0.01, 0.01])
    assert calculate_sharpe_ratio(const_returns) == 0.0

    # Normal positive returns
    np.random.seed(42)
    pos_returns = np.random.normal(0.002, 0.01, size=100)
    sharpe = calculate_sharpe_ratio(pos_returns, periods_per_year=252)
    assert sharpe > 0.0

    # Negative returns
    neg_returns = np.random.normal(-0.002, 0.01, size=100)
    neg_sharpe = calculate_sharpe_ratio(neg_returns, periods_per_year=252)
    assert neg_sharpe < 0.0


def test_sortino_ratio_calculations():
    """Verify Sortino ratio calculation focusing on downside semivariance."""
    # All positive returns (zero downside deviation)
    all_pos = np.array([0.01, 0.02, 0.015, 0.03])
    assert calculate_sortino_ratio(all_pos) == 0.0

    # Mixed returns
    np.random.seed(42)
    mixed = np.random.normal(0.001, 0.015, size=100)
    sortino = calculate_sortino_ratio(mixed, periods_per_year=252)
    assert np.isfinite(sortino)


def test_calculate_max_drawdown():
    """Verify maximum peak-to-trough drawdown calculation."""
    # Peak at 120, trough at 90 -> (120 - 90) / 120 = 25.0%
    equity = [100.0, 110.0, 120.0, 90.0, 105.0]
    dd = calculate_max_drawdown(equity)
    assert abs(dd - 25.0) < 1e-4

    # Monotonically increasing equity -> 0% drawdown
    up_equity = [100.0, 105.0, 110.0, 115.0]
    assert calculate_max_drawdown(up_equity) == 0.0


def test_run_forecast_backtest_execution():
    """Verify walk-forward backtest execution on synthetic data."""
    dates = pd.date_range("2024-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=65)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    volumes = [10000 + int(abs(r) * 100000) for r in returns]
    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * 65,
            "exchange": ["XNYS"] * 65,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "data_status": ["END_OF_DAY"] * 65,
            "provider": ["Mock"] * 65,
        }
    )

    # Run with 0 bps fee
    res_no_fee = run_forecast_backtest(
        df, horizon_sessions=1, fee_bps=0.0, min_train_sessions=30, model_family="linear"
    )
    # Run with 20 bps fee
    res_high_fee = run_forecast_backtest(
        df, horizon_sessions=1, fee_bps=20.0, min_train_sessions=30, model_family="linear"
    )

    assert len(res_no_fee.equity_df) > 0
    assert "strategy_equity" in res_no_fee.equity_df.columns
    assert "benchmark_equity" in res_no_fee.equity_df.columns

    # High fee backtest must have higher fee drag than no-fee backtest
    assert res_high_fee.metrics.fee_drag_pct >= res_no_fee.metrics.fee_drag_pct
    assert res_high_fee.metrics.total_return_pct <= res_no_fee.metrics.total_return_pct


def test_backtest_insufficient_data():
    """Verify that insufficient data raises ValueError."""
    short_df = pd.DataFrame(
        {
            "session_date": ["2026-01-01", "2026-01-02"],
            "adj_close": [100.0, 101.0],
            "volume": [1000, 1000],
        }
    )
    with pytest.raises(ValueError, match="Insufficient sessions"):
        run_forecast_backtest(short_df, min_train_sessions=120)


def test_cli_backtest_command_execution(monkeypatch, tmp_path: Path):
    """Verify CLI backtest_command runs and outputs scorecard cleanly."""
    db_file = tmp_path / "cli_backtest.db"
    init_db(db_file)
    monkeypatch.setattr("src.data.db.DATABASE_PATH", db_file)

    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_path=db_file)

    dates = pd.date_range("2024-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=65)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    df = pd.DataFrame(
        {
            "ticker": ["JPM"] * 65,
            "exchange": ["XNYS"] * 65,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [10000] * 65,
            "data_status": ["END_OF_DAY"] * 65,
            "provider": ["Mock"] * 65,
        }
    )
    save_observations(df, db_path=db_file)

    args = argparse.Namespace(
        ticker="JPM",
        model="linear",
        horizon=1,
        fee_bps=5.0,
        min_train=30,
    )
    backtest_command(args)
