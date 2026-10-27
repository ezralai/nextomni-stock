"""Unit and integration tests for macro stress-testing and database VACUUM maintenance."""

import argparse
from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import simulate_macro_stress_scenarios
from src.cli import risk_report_command, vacuum_db_command
from src.data.db import (
    get_connection,
    init_db,
    save_observations,
    upsert_stock,
    vacuum_database,
)


@pytest.fixture
def sample_stress_df() -> pd.DataFrame:
    """Generate 60 sessions of deterministic price data."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=60)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    return pd.DataFrame(
        {
            "ticker": ["JPM"] * 60,
            "exchange": ["XNYS"] * 60,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [10000] * 60,
            "data_status": ["END_OF_DAY"] * 60,
            "provider": ["Mock"] * 60,
        }
    )


def test_simulate_macro_stress_scenarios_structure(sample_stress_df: pd.DataFrame):
    """Verify macro stress testing scenarios calculate estimated losses and stressed prices."""
    results = simulate_macro_stress_scenarios(sample_stress_df, position_value=10000.0)
    assert len(results) == 4

    scenario_names = [s["scenario_name"] for s in results]
    assert "2022 Rates & Inflation Shock" in scenario_names
    assert "2008 Lehman Brothers Insolvency" in scenario_names
    assert "2020 COVID Liquidity Freeze" in scenario_names
    assert "1987 Black Monday Tail Event" in scenario_names

    latest_price = float(sample_stress_df["adj_close"].iloc[-1])

    for s in results:
        assert s["estimated_loss_pct"] > 0
        assert s["estimated_loss_dollars"] > 0
        assert s["post_stress_price"] < latest_price
        assert s["severity"] in ["Moderate", "Severe", "Extreme", "Catastrophic"]
        assert s["ticker_vol_beta"] > 0


def test_simulate_macro_stress_scenarios_insufficient_data():
    """Verify empty result returned gracefully when data length is insufficient."""
    short_df = pd.DataFrame({"adj_close": [100.0, 101.0]})
    assert simulate_macro_stress_scenarios(short_df) == []


def test_vacuum_database_clean_execution(tmp_path: Path):
    """Verify VACUUM defragmentation and page reclamation."""
    db_file = tmp_path / "vacuum_test.db"
    init_db(db_file)

    conn = get_connection(db_file)
    with conn:
        for i in range(200):
            conn.execute(
                """
                INSERT INTO raw_provider_payloads (
                    ticker, provider, payload_json, retrieved_at
                ) VALUES (?, ?, ?, datetime('now'));
                """,
                (f"TICK_{i}", "mock_provider", "X" * 1000),
            )
    conn.close()

    # Delete payloads to create free pages
    conn = get_connection(db_file)
    with conn:
        conn.execute("DELETE FROM raw_provider_payloads;")
    conn.close()

    rep = vacuum_database(db_file)
    assert rep["status"] == "success"
    assert rep["integrity"] == "ok"
    assert rep["foreign_keys"] == "ok"
    assert rep["size_after_bytes"] <= rep["size_before_bytes"]
    assert rep["bytes_reclaimed"] >= 0


def test_vacuum_database_fk_violation_raises(tmp_path: Path):
    """Verify vacuum_database raises when foreign key violations exist."""
    db_file = tmp_path / "fk_fail.db"
    init_db(db_file)

    # Bypass foreign keys temporarily to inject orphan record
    conn = sqlite3.connect(str(db_file))
    conn.execute("PRAGMA foreign_keys = OFF;")
    conn.execute(
        """
        INSERT INTO market_observations (
            ticker, exchange, session_date, open, high, low, close, adj_close, volume, data_status, provider
        ) VALUES ('ORPHAN_TICKER', 'XNYS', '2026-01-01', 100, 101, 99, 100, 100, 1000, 'EOD', 'Mock');
        """
    )
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="foreign key check failed"):
        vacuum_database(db_file)


def test_cli_vacuum_and_risk_report_commands(monkeypatch, tmp_path: Path, sample_stress_df: pd.DataFrame):
    """Verify CLI vacuum-db and risk-report --stress-test commands."""
    db_file = tmp_path / "cli_stress.db"
    init_db(db_file)
    monkeypatch.setattr("src.data.db.DATABASE_PATH", db_file)

    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", db_path=db_file)
    save_observations(sample_stress_df, db_path=db_file)

    # Test vacuum-db
    vacuum_db_command(argparse.Namespace())

    # Test risk-report --stress-test
    risk_args = argparse.Namespace(ticker="JPM", stress_test=True)
    risk_report_command(risk_args)
