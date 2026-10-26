"""Historical data pipeline orchestrator for NextOmni Stock.

Coordinates validation, calendar cross-checks, corporate action ingestion,
and local SQLite persistence for NYSE and Bursa Malaysia stocks.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import exchange_calendars as xcals
import pandas as pd

from src.config import DATABASE_PATH, SUPPORTED_EXCHANGES
from src.data.base_provider import BaseMarketDataProvider
from src.data.db import (
    init_db,
    save_corporate_actions,
    save_observations,
    save_raw_payload,
    upsert_stock,
)
from src.data.yfinance_provider import YahooFinanceProvider


class DataPipeline:
    """End-to-end data pipeline for fetching, validating, and persisting market history."""

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        db_path: Optional[Path] = None,
    ):
        self.provider = provider or YahooFinanceProvider()
        self.db_path = db_path or DATABASE_PATH
        init_db(self.db_path)

    def validate_inputs(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> None:
        """Validate ticker, exchange, and date ranges."""
        if exchange not in SUPPORTED_EXCHANGES:
            raise ValueError(
                f"Unsupported exchange '{exchange}'. Supported exchanges: {list(SUPPORTED_EXCHANGES.keys())}"
            )

        if not ticker or not ticker.strip():
            raise ValueError("Ticker cannot be empty.")

        try:
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError as e:
            raise ValueError(f"Invalid date format. Expected YYYY-MM-DD: {e}")

        if start_dt > end_dt:
            raise ValueError(
                f"start_date ({start_date}) cannot be after end_date ({end_date})."
            )

    def clean_and_validate_data(
        self,
        df: pd.DataFrame,
        exchange: str,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Validate historical data integrity, sort chronologically, and check against calendar.

        Never silently fabricates or interpolates missing prices.
        """
        if df.empty:
            return df, []

        cleaned = df.copy()

        # 1. Deduplicate by session_date (keep last observation if duplicates exist)
        cleaned = cleaned.drop_duplicates(subset=["session_date"], keep="last")

        # 2. Chronological sort
        cleaned = cleaned.sort_values("session_date", ascending=True).reset_index(drop=True)

        # 3. Detect and filter invalid non-positive prices
        valid_prices = (
            (cleaned["open"] > 0)
            & (cleaned["high"] > 0)
            & (cleaned["low"] > 0)
            & (cleaned["close"] > 0)
            & (cleaned["adj_close"] > 0)
        )
        cleaned = cleaned[valid_prices].reset_index(drop=True)

        # 4. Exchange Calendar Cross-Reference
        # Identify if any observed dates fall on known non-trading days or holidays
        calendar = xcals.get_calendar(exchange)
        min_date = cleaned["session_date"].iloc[0]
        max_date = cleaned["session_date"].iloc[-1]

        # Get all official exchange sessions within the observed window
        official_sessions = [
            dt.strftime("%Y-%m-%d")
            for dt in calendar.sessions_in_range(min_date, max_date)
        ]

        observed_set = set(cleaned["session_date"])
        missing_sessions = [s for s in official_sessions if s not in observed_set]

        return cleaned, missing_sessions

    def sync_stock(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
        company_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch, validate, audit-log, and persist historical data and corporate actions."""
        self.validate_inputs(ticker, exchange, start_date, end_date)
        cfg = SUPPORTED_EXCHANGES[exchange]

        # 1. Register or update stock metadata
        upsert_stock(
            ticker=ticker,
            exchange=exchange,
            company_name=company_name or ticker,
            currency=cfg.currency,
            timezone=cfg.timezone,
            db_path=self.db_path,
        )

        # 2. Fetch raw historical OHLCV data
        raw_df, raw_payload = self.provider.fetch_historical_data(
            ticker=ticker,
            exchange=exchange,
            start_date=start_date,
            end_date=end_date,
        )

        # 3. Save raw payload for audit and reproducibility
        save_raw_payload(
            ticker=ticker,
            provider=self.provider.name,
            payload=raw_payload,
            period_start=start_date,
            period_end=end_date,
            db_path=self.db_path,
        )

        if raw_df.empty:
            return {
                "ticker": ticker,
                "exchange": exchange,
                "status": "NO_DATA",
                "sessions_synced": 0,
                "missing_calendar_sessions": [],
            }

        # 4. Clean, validate, and verify calendar alignment
        cleaned_df, missing_calendar_sessions = self.clean_and_validate_data(
            raw_df, exchange
        )

        # 5. Persist cleaned observations to SQLite
        saved_count = save_observations(cleaned_df, db_path=self.db_path)

        # 6. Fetch and store corporate actions (dividends and splits)
        corporate_actions_df = self.provider.fetch_corporate_actions(
            ticker=ticker, exchange=exchange
        )
        actions_saved = save_corporate_actions(
            ticker=ticker, actions_df=corporate_actions_df, db_path=self.db_path
        )

        return {
            "ticker": ticker,
            "exchange": exchange,
            "status": "SUCCESS",
            "sessions_synced": saved_count,
            "corporate_actions_synced": actions_saved,
            "first_session": cleaned_df["session_date"].iloc[0],
            "last_session": cleaned_df["session_date"].iloc[-1],
            "missing_calendar_sessions_count": len(missing_calendar_sessions),
            "missing_calendar_sessions": missing_calendar_sessions,
            "provider": self.provider.name,
        }



