"""Real-time market monitoring service for NextOmni Stock (nextomni-stock).

Orchestrates multi-provider quote polling, client-side TTL caching,
exchange session state tracking, and clear data latency labeling.
"""

from dataclasses import dataclass
from datetime import datetime
import time
from typing import Dict, List, Optional, Tuple

from src.config import INITIAL_TICKERS, SUPPORTED_EXCHANGES
from src.data.base_provider import BaseMarketDataProvider
from src.data.session_status import ExchangeSessionEngine, SessionStatusInfo
from src.data.twelve_data_provider import TwelveDataProvider


@dataclass(frozen=True)
class MarketQuote:
    """Enriched market quote with operational session state and latency badge."""
    ticker: str
    exchange: str
    exchange_name: str
    last_price: float
    previous_close: float
    change: float
    pct_change: float
    currency: str
    latency_badge: str
    session_status: str
    is_market_open: bool
    local_time: str
    next_event: str
    provider: str
    cached: bool
    note: str


class MarketMonitorService:
    """Service providing cached, enriched market quotes across supported exchanges."""

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        cache_ttl_seconds: int = 60,
    ):
        self.provider = provider or TwelveDataProvider()
        self.cache_ttl = cache_ttl_seconds
        self._cache: Dict[Tuple[str, str], Tuple[float, MarketQuote]] = {}

    def get_quote(
        self,
        ticker: str,
        exchange: str,
        force_refresh: bool = False,
    ) -> MarketQuote:
        """Fetch quote enriched with exchange operational status and latency labeling."""
        clean_ticker = ticker.strip().upper()
        clean_exchange = exchange.strip().upper()
        cache_key = (clean_ticker, clean_exchange)

        now = time.time()
        if not force_refresh and cache_key in self._cache:
            timestamp, cached_quote = self._cache[cache_key]
            if now - timestamp < self.cache_ttl:
                # Return cached quote marked with cached=True
                return MarketQuote(
                    ticker=cached_quote.ticker,
                    exchange=cached_quote.exchange,
                    exchange_name=cached_quote.exchange_name,
                    last_price=cached_quote.last_price,
                    previous_close=cached_quote.previous_close,
                    change=cached_quote.change,
                    pct_change=cached_quote.pct_change,
                    currency=cached_quote.currency,
                    latency_badge=cached_quote.latency_badge,
                    session_status=cached_quote.session_status,
                    is_market_open=cached_quote.is_market_open,
                    local_time=cached_quote.local_time,
                    next_event=cached_quote.next_event,
                    provider=cached_quote.provider,
                    cached=True,
                    note=cached_quote.note,
                )

        # 1. Fetch quote payload from provider
        quote_data = self.provider.fetch_latest_quote(clean_ticker, clean_exchange)

        # 2. Compute live session status from exchange calendar engine
        sess_info: SessionStatusInfo = ExchangeSessionEngine.get_status(clean_exchange)

        next_event = ""
        if sess_info.is_open and sess_info.next_close:
            next_event = f"Closes at {sess_info.next_close}"
        elif not sess_info.is_open and sess_info.next_open:
            next_event = f"Opens at {sess_info.next_open}"
        else:
            next_event = sess_info.details

        # Extract values with safe defaults
        last_price = float(quote_data.get("last_price", 0.0))
        prev_close = float(quote_data.get("previous_close", last_price))
        change = float(quote_data.get("change", last_price - prev_close))
        pct_change = float(quote_data.get("pct_change", 0.0))
        currency = quote_data.get("currency", SUPPORTED_EXCHANGES.get(clean_exchange, None).currency if clean_exchange in SUPPORTED_EXCHANGES else "USD")
        latency_badge = quote_data.get("status", "UNKNOWN")
        provider_name = quote_data.get("provider", self.provider.name)
        note = quote_data.get("note", "")

        quote = MarketQuote(
            ticker=clean_ticker,
            exchange=clean_exchange,
            exchange_name=SUPPORTED_EXCHANGES[clean_exchange].name if clean_exchange in SUPPORTED_EXCHANGES else clean_exchange,
            last_price=last_price,
            previous_close=prev_close,
            change=change,
            pct_change=pct_change,
            currency=currency,
            latency_badge=latency_badge,
            session_status=sess_info.status,
            is_market_open=sess_info.is_open,
            local_time=sess_info.local_time,
            next_event=next_event,
            provider=provider_name,
            cached=False,
            note=note,
        )

        self._cache[cache_key] = (now, quote)
        return quote

    def get_multi_quotes(
        self,
        tickers: Optional[List[str]] = None,
        exchange: Optional[str] = None,
        force_refresh: bool = False,
    ) -> List[MarketQuote]:
        """Fetch quotes for multiple tickers or all configured default tickers."""
        quotes = []
        if tickers and exchange:
            for t in tickers:
                quotes.append(self.get_quote(t, exchange, force_refresh=force_refresh))
        elif exchange:
            target_tickers = INITIAL_TICKERS.get(exchange, [])
            for t in target_tickers:
                quotes.append(self.get_quote(t, exchange, force_refresh=force_refresh))
        else:
            # Query all supported exchanges
            for exc, ticker_list in INITIAL_TICKERS.items():
                for t in ticker_list:
                    quotes.append(self.get_quote(t, exc, force_refresh=force_refresh))
        return quotes
