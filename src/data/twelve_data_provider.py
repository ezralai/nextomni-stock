"""Twelve Data market data provider implementation for NextOmni Stock.

Provides real-time quotes, rate-limiting guards (8 req/min for free tier),
and seamless fallback to Yahoo Finance when API keys are absent or quotas are reached.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd

from src.config import FEED_DELAYED, FEED_EOD, FEED_REALTIME, SUPPORTED_EXCHANGES
from src.data.base_provider import BaseMarketDataProvider
from src.data.yfinance_provider import YahooFinanceProvider


class TwelveDataProvider(BaseMarketDataProvider):
    """Market data provider using Twelve Data API with Yahoo Finance fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        max_requests_per_minute: int = 8,
        fallback_provider: Optional[BaseMarketDataProvider] = None,
    ):
        self.api_key = api_key or os.getenv("NEXTOMNI_STOCK_TWELVE_DATA_API_KEY", "")
        self.max_requests_per_minute = max_requests_per_minute
        self.fallback = fallback_provider or YahooFinanceProvider()
        self._request_timestamps: List[float] = []

    @property
    def name(self) -> str:
        return "TwelveData"

    def _resolve_symbol(self, ticker: str, exchange: str) -> str:
        """Map internal ticker to Twelve Data format."""
        clean = ticker.strip().upper()
        if exchange == "XKLS":
            # Twelve Data format for Bursa Malaysia is TICKER:KLSE or TICKER:MYX
            base = clean.replace(".KL", "")
            return f"{base}:KLSE"
        return clean

    def _check_rate_limit(self) -> bool:
        """Sliding window rate limit check (max N requests per 60 seconds)."""
        now = time.time()
        # Keep timestamps within the last 60 seconds
        self._request_timestamps = [t for t in self._request_timestamps if now - t < 60.0]
        if len(self._request_timestamps) >= self.max_requests_per_minute:
            return False
        return True

    def _record_request(self) -> None:
        """Record an API request timestamp."""
        self._request_timestamps.append(time.time())

    def fetch_latest_quote(
        self,
        ticker: str,
        exchange: str,
    ) -> Dict[str, Any]:
        """Fetch the latest quote using Twelve Data or fallback to Yahoo Finance."""
        if not self.api_key:
            # Fallback directly when no API key configured
            fb_quote = self.fallback.fetch_latest_quote(ticker, exchange)
            fb_quote["note"] = "Twelve Data API key not configured; using Yahoo Finance fallback."
            return fb_quote

        if not self._check_rate_limit():
            # Rate limit reached; fall back transparently
            fb_quote = self.fallback.fetch_latest_quote(ticker, exchange)
            fb_quote["note"] = "Twelve Data rate limit (8 req/min) active; using Yahoo Finance fallback."
            return fb_quote

        symbol = self._resolve_symbol(ticker, exchange)
        url = f"https://api.twelvedata.com/quote?symbol={symbol}&apikey={self.api_key}"

        try:
            self._record_request()
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                fb_quote = self.fallback.fetch_latest_quote(ticker, exchange)
                fb_quote["note"] = f"Twelve Data HTTP {resp.status_code}; fallback applied."
                return fb_quote

            data = resp.json()
            # Handle Twelve Data error responses
            if "code" in data and data.get("status") == "error":
                fb_quote = self.fallback.fetch_latest_quote(ticker, exchange)
                fb_quote["note"] = f"Twelve Data message: {data.get('message')}; fallback applied."
                return fb_quote

            # Successful real-time quote parsing
            close_price = float(data.get("close", data.get("previous_close", 0.0)))
            prev_close = float(data.get("previous_close", close_price))
            change = float(data.get("change", close_price - prev_close))
            pct_change = float(data.get("percent_change", (change / prev_close * 100) if prev_close > 0 else 0.0))

            return {
                "ticker": ticker,
                "exchange": exchange,
                "symbol": symbol,
                "last_price": close_price,
                "previous_close": prev_close,
                "change": change,
                "pct_change": pct_change,
                "open": float(data.get("open", 0.0)) if data.get("open") else None,
                "high": float(data.get("high", 0.0)) if data.get("high") else None,
                "low": float(data.get("low", 0.0)) if data.get("low") else None,
                "volume": int(data.get("volume", 0)) if data.get("volume") else None,
                "currency": data.get("currency", SUPPORTED_EXCHANGES.get(exchange, None).currency if exchange in SUPPORTED_EXCHANGES else "USD"),
                "status": FEED_REALTIME,
                "provider": self.name,
                "timestamp": data.get("timestamp"),
            }
        except Exception as e:
            fb_quote = self.fallback.fetch_latest_quote(ticker, exchange)
            fb_quote["note"] = f"Twelve Data exception ({str(e)}); fallback applied."
            return fb_quote

    def fetch_historical_data(
        self,
        ticker: str,
        exchange: str,
        start_date: str,
        end_date: str,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fetch historical daily OHLCV series (delegates to Yahoo Finance)."""
        return self.fallback.fetch_historical_data(ticker, exchange, start_date, end_date)

    def fetch_corporate_actions(
        self,
        ticker: str,
        exchange: str,
    ) -> pd.DataFrame:
        """Fetch corporate actions (dividends and splits via Yahoo Finance)."""
        return self.fallback.fetch_corporate_actions(ticker, exchange)
