"""Abstract base class interface for NextOmni Stock market data providers.

Ensures that providers (Yahoo Finance, Twelve Data, Mocks) adhere to a
consistent contract for historical series, quotes, and corporate actions.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import pandas as pd


class BaseMarketDataProvider(ABC):
    """Abstract interface for financial market data providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name of the provider."""
        pass

    @abstractmethod
    def fetch_historical_data(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch historical daily OHLCV observations for a given ticker and exchange.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]:
                - Cleaned observations DataFrame with standardized columns:
                  ['session_date', 'open', 'high', 'low', 'close', 'adj_close', 'volume']
                - Raw payload dict for audit logging and reproducibility.
        """
        pass

    @abstractmethod
    def fetch_corporate_actions(
        self,
        ticker: str,
        exchange: str,
    ) -> pd.DataFrame:
        """Fetch historical dividends and stock splits.

        Returns:
            pd.DataFrame: Columns ['action_date', 'action_type', 'value']
        """
        pass

    @abstractmethod
    def fetch_latest_quote(
        self,
        ticker: str,
        exchange: str,
    ) -> Dict[str, Any]:
        """Fetch the latest available market observation and metadata."""
        pass
