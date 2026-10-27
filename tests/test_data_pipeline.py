"""Comprehensive tests for NextOmni Stock data pipeline, SQLite storage, and provider adapters."""

from pathlib import Path
from typing import Any, Dict, Tuple
import pandas as pd
import pytest

from src.data.base_provider import BaseMarketDataProvider
from src.data.db import (
    get_connection,
    get_observations,
    init_db,
    save_observations,
    upsert_stock,
)
from src.data.pipeline import DataPipeline


class MockMarketDataProvider(BaseMarketDataProvider):
    """Deterministic mock provider for unit testing without external network calls."""

    @property
    def name(self) -> str:
        return "MockProvider"

    def fetch_historical_data(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # Return 5 synthetic sessions
        dates = ["2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11", "2024-01-12"]
        df = pd.DataFrame(
            {
                "ticker": ticker,
                "exchange": exchange,
                "session_date": dates,
                "open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "high": [102.0, 103.0, 104.0, 105.0, 106.0],
                "low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "close": [101.5, 102.5, 103.5, 104.5, 105.5],
                "adj_close": [101.0, 102.0, 103.0, 104.0, 105.0],
                "volume": [10000, 12000, 11000, 13000, 15000],
                "data_status": "END_OF_DAY",
                "provider": self.name,
            }
        )
        return df, {"mock_source": True, "ticker": ticker}

    def fetch_corporate_actions(
        self,
        ticker: str,
        exchange: str,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"action_date": "2024-01-10", "action_type": "DIVIDEND", "value": 0.50},
                {"action_date": "2024-01-12", "action_type": "SPLIT", "value": 2.0},
            ]
        )

    def fetch_latest_quote(
        self,
        ticker: str,
        exchange: str,
    ) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "exchange": exchange,
            "symbol": ticker,
            "last_price": 105.5,
            "previous_close": 104.5,
            "change": 1.0,
            "pct_change": 0.957,
            "currency": "USD" if exchange == "XNYS" else "MYR",
            "status": "DELAYED_15MIN",
            "provider": self.name,
        }


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create an isolated test database."""
    db_file = tmp_path / "test_stock.db"
    init_db(db_file)
    return db_file


def test_database_initialization(temp_db: Path):
    """Verify that all required tables and indexes are created."""
    conn = get_connection(temp_db)
    cursor = conn.cursor()

    tables = {
        row[0]
        for row in cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        ).fetchall()
    }

    assert "stocks" in tables
    assert "market_observations" in tables
    assert "raw_provider_payloads" in tables
    assert "corporate_actions" in tables
    assert "predictions" in tables
    assert "prediction_evaluations" in tables

    # Check interval columns in predictions table
    pred_cols = {
        col[1]
        for col in cursor.execute("PRAGMA table_info(predictions);").fetchall()
    }
    assert "lower_bound" in pred_cols
    assert "upper_bound" in pred_cols
    assert "confidence_level" in pred_cols
    conn.close()

    # Verify idempotency of init_db migrations
    init_db(temp_db)
    init_db(temp_db)


def test_upsert_stock(temp_db: Path):
    """Verify stock metadata insertion and update idempotency."""
    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", temp_db)

    conn = get_connection(temp_db)
    row = conn.execute("SELECT * FROM stocks WHERE ticker = 'JPM';").fetchone()
    assert row["ticker"] == "JPM"
    assert row["currency"] == "USD"

    # Update company name
    upsert_stock("JPM", "XNYS", "JPMorgan Chase & Co.", "USD", "America/New_York", temp_db)
    row = conn.execute("SELECT * FROM stocks WHERE ticker = 'JPM';").fetchone()
    assert row["company_name"] == "JPMorgan Chase & Co."
    conn.close()


def test_observations_storage_and_query(temp_db: Path):
    """Verify observations are saved, deduplicated on conflict, and queryable."""
    upsert_stock("JPM", "XNYS", "JPMorgan Chase", "USD", "America/New_York", temp_db)

    df = pd.DataFrame(
        {
            "ticker": ["JPM", "JPM"],
            "exchange": ["XNYS", "XNYS"],
            "session_date": ["2024-01-09", "2024-01-10"],
            "open": [170.0, 171.0],
            "high": [172.0, 173.0],
            "low": [169.0, 170.0],
            "close": [171.5, 172.5],
            "adj_close": [170.5, 171.5],
            "volume": [8000000, 9000000],
            "data_status": ["END_OF_DAY", "END_OF_DAY"],
            "provider": ["TestProvider", "TestProvider"],
        }
    )

    rows = save_observations(df, temp_db)
    assert rows == 2

    # Query all
    queried = get_observations("JPM", db_path=temp_db)
    assert len(queried) == 2
    assert list(queried["session_date"]) == ["2024-01-09", "2024-01-10"]

    # Re-insert with updated close price (idempotent upsert test)
    df_update = df.copy()
    df_update["adj_close"] = [170.9, 171.9]
    save_observations(df_update, temp_db)

    queried_after = get_observations("JPM", db_path=temp_db)
    assert len(queried_after) == 2  # No duplicate rows
    assert queried_after["adj_close"].iloc[0] == 170.9


def test_clean_and_validate_data():
    """Verify data deduplication, chronological sorting, and non-positive price filtering."""
    pipeline = DataPipeline(provider=MockMarketDataProvider())

    # Raw dataframe with duplicate date, reversed order, and negative price
    dirty_df = pd.DataFrame(
        {
            "session_date": ["2024-01-10", "2024-01-08", "2024-01-09", "2024-01-10", "2024-01-11"],
            "open": [100.0, 98.0, 99.0, 100.5, -5.0],  # Last row is invalid (negative)
            "high": [102.0, 99.0, 100.0, 102.0, 10.0],
            "low": [99.0, 97.0, 98.0, 99.0, 1.0],
            "close": [101.0, 98.5, 99.5, 101.5, 8.0],
            "adj_close": [101.0, 98.5, 99.5, 101.5, 8.0],
            "volume": [1000, 1000, 1000, 1200, 1000],
        }
    )

    cleaned, missing = pipeline.clean_and_validate_data(dirty_df, exchange="XNYS")

    # Negative price row removed, duplicate 2024-01-10 deduplicated (kept last: 101.5)
    assert len(cleaned) == 3
    assert list(cleaned["session_date"]) == ["2024-01-08", "2024-01-09", "2024-01-10"]
    assert cleaned.loc[cleaned["session_date"] == "2024-01-10", "close"].iloc[0] == 101.5


def test_pipeline_input_validation():
    """Verify input validation catches bad tickers, unsupported exchanges, and dates."""
    pipeline = DataPipeline(provider=MockMarketDataProvider())

    with pytest.raises(ValueError, match="Unsupported exchange"):
        pipeline.validate_inputs("AAPL", "LSE", "2024-01-01", "2024-01-10")

    with pytest.raises(ValueError, match="Ticker cannot be empty"):
        pipeline.validate_inputs("", "XNYS", "2024-01-01", "2024-01-10")

    with pytest.raises(ValueError, match="start_date .* cannot be after end_date"):
        pipeline.validate_inputs("JPM", "XNYS", "2024-01-20", "2024-01-10")

    with pytest.raises(ValueError, match="Invalid date format"):
        pipeline.validate_inputs("JPM", "XNYS", "2024/01/01", "2024-01-10")


def test_full_pipeline_sync_mock(temp_db: Path):
    """Verify end-to-end sync using Mock provider stores all data and payloads."""
    pipeline = DataPipeline(provider=MockMarketDataProvider(), db_path=temp_db)

    result = pipeline.sync_stock(
        ticker="JPM",
        exchange="XNYS",
        start_date="2024-01-08",
        end_date="2024-01-12",
        company_name="JPMorgan Chase",
    )

    assert result["status"] == "SUCCESS"
    assert result["sessions_synced"] == 5
    assert result["corporate_actions_synced"] == 2

    # Check observations persisted
    obs = get_observations("JPM", db_path=temp_db)
    assert len(obs) == 5

    # Check raw payloads stored
    conn = get_connection(temp_db)
    payloads = conn.execute("SELECT * FROM raw_provider_payloads;").fetchall()
    assert len(payloads) == 1
    assert payloads[0]["ticker"] == "JPM"

    # Check corporate actions stored
    actions = conn.execute("SELECT * FROM corporate_actions;").fetchall()
    assert len(actions) == 2
    conn.close()


def test_live_yahoo_finance_sync(temp_db: Path):
    """Smoke test live Yahoo Finance historical data synchronization for both NYSE and Bursa Malaysia."""
    pipeline = DataPipeline(db_path=temp_db)

    # 1. Test NYSE (JPMorgan Chase)
    nyse_res = pipeline.sync_stock(
        ticker="JPM",
        exchange="XNYS",
        start_date="2024-01-02",
        end_date="2024-01-20",
        company_name="JPMorgan Chase & Co.",
    )

    assert nyse_res["status"] == "SUCCESS"
    assert nyse_res["sessions_synced"] > 0
    nyse_obs = get_observations("JPM", db_path=temp_db)
    assert len(nyse_obs) == nyse_res["sessions_synced"]
    assert "adj_close" in nyse_obs.columns

    # 2. Test Bursa Malaysia (Maybank - Malayan Banking Berhad)
    bursa_res = pipeline.sync_stock(
        ticker="1155.KL",
        exchange="XKLS",
        start_date="2024-01-02",
        end_date="2024-01-20",
        company_name="Malayan Banking Berhad",
    )

    assert bursa_res["status"] == "SUCCESS"
    assert bursa_res["sessions_synced"] > 0
    bursa_obs = get_observations("1155.KL", db_path=temp_db)
    assert len(bursa_obs) == bursa_res["sessions_synced"]
    assert "adj_close" in bursa_obs.columns

