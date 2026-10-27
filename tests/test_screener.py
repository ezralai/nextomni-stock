"""Unit and integration tests for cross-sectional alpha universe screener."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.screener import ScreenerRecord, screen_asset_universe
from src.cli import screen_command
from src.data.db import get_all_stocks, init_db, save_observations, upsert_stock


@pytest.fixture
def screener_db(tmp_path):
    """Create isolated SQLite database with multiple registered stocks."""
    db_file = tmp_path / "test_screener.db"
    init_db(db_file)

    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_path=db_file)
    upsert_stock("AAPL", "XNYS", "Apple Inc", "USD", "America/New_York", db_path=db_file)
    upsert_stock("1155.KL", "XKLS", "Maybank", "MYR", "Asia/Kuala_Lumpur", db_path=db_file)

    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")

    # JPM: Strong upward trend
    p_jpm = [100.0 + i * 1.0 for i in range(60)]
    df_jpm = pd.DataFrame({
        "ticker": ["JPM"] * 60,
        "exchange": ["XNYS"] * 60,
        "session_date": dates,
        "open": p_jpm,
        "high": [p * 1.01 for p in p_jpm],
        "low": [p * 0.99 for p in p_jpm],
        "close": p_jpm,
        "adj_close": p_jpm,
        "volume": [10000] * 60,
        "data_status": ["END_OF_DAY"] * 60,
        "provider": ["Mock"] * 60,
    })
    save_observations(df_jpm, db_path=db_file)

    # AAPL: Moderate trend
    p_aapl = [150.0 + i * 0.2 for i in range(60)]
    df_aapl = pd.DataFrame({
        "ticker": ["AAPL"] * 60,
        "exchange": ["XNYS"] * 60,
        "session_date": dates,
        "open": p_aapl,
        "high": [p * 1.01 for p in p_aapl],
        "low": [p * 0.99 for p in p_aapl],
        "close": p_aapl,
        "adj_close": p_aapl,
        "volume": [20000] * 60,
        "data_status": ["END_OF_DAY"] * 60,
        "provider": ["Mock"] * 60,
    })
    save_observations(df_aapl, db_path=db_file)

    return db_file


def test_get_all_stocks(screener_db):
    """Verify get_all_stocks queries and filters correctly."""
    all_stocks = get_all_stocks(db_path=screener_db)
    assert len(all_stocks) == 3

    nyse_stocks = get_all_stocks(exchange="XNYS", db_path=screener_db)
    assert len(nyse_stocks) == 2
    assert set(nyse_stocks["ticker"]) == {"JPM", "AAPL"}

    bursa_stocks = get_all_stocks(exchange="XKLS", db_path=screener_db)
    assert len(bursa_stocks) == 1
    assert bursa_stocks.iloc[0]["ticker"] == "1155.KL"


def test_screen_asset_universe_ranking(screener_db):
    """Verify screen_asset_universe computes metrics and ranks descending."""
    results = screen_asset_universe(
        db_path=screener_db,
        horizon_sessions=5,
        model_family="linear",
        min_observations=40,
    )

    # 1155.KL has no observations, so only JPM and AAPL should be scored
    assert len(results) == 2

    # Check rank order
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].expected_return_pct >= results[1].expected_return_pct

    # Check metric sanity
    for r in results:
        assert isinstance(r, ScreenerRecord)
        assert 0.0 <= r.directional_win_probability <= 1.0
        assert r.conviction_label in {"STRONG BULLISH", "LEAN BULLISH", "NEUTRAL", "LEAN BEARISH", "STRONG BEARISH"}
        assert r.cornish_fisher_var_pct >= 0.0
        assert np.isfinite(r.predicted_sharpe)


def test_screen_asset_universe_filter_exchange(screener_db):
    """Verify exchange filter isolates results."""
    results_nyse = screen_asset_universe(
        db_path=screener_db,
        exchange="XNYS",
        horizon_sessions=1,
        model_family="linear",
    )
    assert all(r.exchange == "XNYS" for r in results_nyse)

    # XKLS has no observations, returns empty
    results_xkls = screen_asset_universe(
        db_path=screener_db,
        exchange="XKLS",
        horizon_sessions=1,
    )
    assert len(results_xkls) == 0


def test_cli_screen_command_outputs(screener_db, monkeypatch, capsys):
    """Verify nextomni screen CLI command works in formatted and JSON modes."""
    monkeypatch.setattr("src.data.db.DATABASE_PATH", screener_db)

    # Text mode
    args_text = argparse.Namespace(
        exchange="XNYS",
        horizon=5,
        model="linear",
        top=1,
        json=False,
    )
    screen_command(args_text)
    out_text = capsys.readouterr().out
    assert "Cross-Sectional Alpha Universe Screener" in out_text
    assert "Rank" in out_text
    assert "JPM" in out_text

    # JSON mode
    args_json = argparse.Namespace(
        exchange="XNYS",
        horizon=5,
        model="linear",
        top=2,
        json=True,
    )
    screen_command(args_json)
    out_json = capsys.readouterr().out
    parsed = json.loads(out_json)
    assert len(parsed) <= 2
    assert "ticker" in parsed[0]
    assert "directional_win_probability" in parsed[0]
