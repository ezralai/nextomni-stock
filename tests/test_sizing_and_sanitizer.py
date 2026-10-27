"""Unit and quantitative integration tests for OHLC integrity sanitization and dynamic sizing."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.backtester import BacktestMetrics, run_forecast_backtest
from src.cli import backtest_command
from src.data.db import init_db, save_observations, upsert_stock
from src.security.validators import SecurityValidationError, validate_ohlc_integrity


def test_validate_ohlc_integrity_clean_data():
    """Verify clean valid OHLC bars pass through with zero dropped records."""
    df = pd.DataFrame({
        "open": [100.0, 102.0],
        "high": [105.0, 106.0],
        "low": [98.0, 101.0],
        "close": [104.0, 103.0],
        "volume": [1000, 2000],
        "adj_close": [104.0, 103.0],
    })
    cleaned, dropped = validate_ohlc_integrity(df, drop_invalid=True)
    assert dropped == 0
    assert len(cleaned) == 2


def test_validate_ohlc_integrity_detects_corruptions():
    """Verify invalid or physically impossible bars are detected and dropped."""
    df = pd.DataFrame({
        "open": [100.0, -50.0, 100.0, 100.0, 100.0],
        "high": [105.0, 105.0, 95.0, 105.0, 105.0],  # row 2: high < open (100)
        "low": [98.0, 90.0, 90.0, 110.0, 95.0],     # row 3: low > open (100)
        "close": [104.0, 100.0, 98.0, 100.0, np.nan],# row 4: NaN close
        "volume": [1000, 1000, 1000, 1000, 1000],
    })

    # With drop_invalid=True
    cleaned, dropped = validate_ohlc_integrity(df, drop_invalid=True)
    assert dropped == 4
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["open"] == 100.0

    # With drop_invalid=False
    with pytest.raises(SecurityValidationError, match="OHLC integrity violation"):
        validate_ohlc_integrity(df, drop_invalid=False)


def test_validate_ohlc_missing_columns():
    """Verify missing required OHLC columns raises SecurityValidationError."""
    bad_df = pd.DataFrame({"open": [100.0], "close": [105.0]})
    with pytest.raises(SecurityValidationError, match="Missing required OHLC columns"):
        validate_ohlc_integrity(bad_df)


@pytest.fixture
def synthetic_backtest_df():
    """Generate synthetic 65 sessions for fast walk-forward test."""
    dates = pd.date_range("2024-01-01", periods=65, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.002, 0.015, size=65)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    return pd.DataFrame({
        "ticker": ["TEST"] * 65,
        "exchange": ["XNYS"] * 65,
        "session_date": dates,
        "open": prices,
        "high": [p * 1.01 for p in prices],
        "low": [p * 0.99 for p in prices],
        "close": prices,
        "adj_close": prices,
        "volume": [50000] * 65,
        "data_status": ["END_OF_DAY"] * 65,
        "provider": ["Mock"] * 65,
    })


def test_volatility_targeted_backtest(synthetic_backtest_df):
    """Verify volatility-targeted sizing dynamically adjusts portfolio exposure."""
    res_fixed = run_forecast_backtest(
        synthetic_backtest_df,
        horizon_sessions=1,
        min_train_sessions=30,
        model_family="linear",
        sizing_mode="fixed",
    )
    res_vol = run_forecast_backtest(
        synthetic_backtest_df,
        horizon_sessions=1,
        min_train_sessions=30,
        model_family="linear",
        sizing_mode="vol_target",
        target_annual_vol=0.15,
        max_leverage=1.5,
    )

    # Fixed positions are strictly binary {0.0, 1.0}
    unique_fixed = set(res_fixed.equity_df["position"])
    assert unique_fixed.issubset({0.0, 1.0})

    # Vol-targeted positions are dynamic floats
    vol_positions = res_vol.equity_df["position"].tolist()
    assert all(0.0 <= p <= 1.5 for p in vol_positions)
    assert hasattr(res_vol.metrics, "avg_exposure")
    assert res_vol.metrics.avg_exposure >= 0.0


def test_half_kelly_backtest(synthetic_backtest_df):
    """Verify half-kelly sizing adjusts exposure bounded by max_leverage."""
    res_kelly = run_forecast_backtest(
        synthetic_backtest_df,
        horizon_sessions=1,
        min_train_sessions=30,
        model_family="linear",
        sizing_mode="half_kelly",
        max_leverage=1.2,
    )
    positions = res_kelly.equity_df["position"].tolist()
    assert all(0.0 <= p <= 1.2 for p in positions)


def test_cli_backtest_with_sizing_arguments(tmp_path, synthetic_backtest_df, monkeypatch, capsys):
    """Verify CLI backtest parses --sizing and --target-vol successfully."""
    db_file = tmp_path / "test_cli_sizing.db"
    init_db(db_file)
    upsert_stock("TEST", "XNYS", "Test Corp", "USD", "America/New_York", db_path=db_file)
    save_observations(synthetic_backtest_df, db_path=db_file)

    monkeypatch.setattr("src.cli.get_observations", lambda ticker: synthetic_backtest_df)

    args = argparse.Namespace(
        ticker="TEST",
        model="linear",
        horizon=1,
        fee_bps=5.0,
        min_train=30,
        sizing="vol_target",
        target_vol=0.20,
        max_leverage=1.0,
    )
    backtest_command(args)
    captured = capsys.readouterr().out
    assert "Position Sizing:   VOL_TARGET" in captured
    assert "Average Portfolio Exposure:" in captured
