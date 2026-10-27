"""Unit and integration tests for Market Monitoring and Session Status Engine.

Tests operational session states (including Bursa Malaysia lunch break),
Twelve Data rate limiting & fallback, and MarketMonitorService caching.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo
import pytest

from src.config import FEED_DELAYED, FEED_REALTIME
from src.data.monitor import MarketMonitorService
from src.data.session_status import (
    ExchangeSessionEngine,
    STATUS_AFTER_HOURS,
    STATUS_CLOSED,
    STATUS_HOLIDAY,
    STATUS_LUNCH_BREAK,
    STATUS_OPEN,
    STATUS_PRE_MARKET,
    STATUS_WEEKEND,
)
from src.data.twelve_data_provider import TwelveDataProvider


def test_bursa_malaysia_session_schedule():
    """Verify Bursa Malaysia dual-session schedule and 12:30-14:30 lunch break."""
    # Morning session: 10:00 MYT on Tuesday 2024-10-01
    dt_am = datetime(2024, 10, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))
    status_am = ExchangeSessionEngine.get_status("XKLS", as_of=dt_am)
    assert status_am.status == STATUS_OPEN
    assert status_am.is_open is True
    assert "Morning trading session" in status_am.details

    # Midday Lunch Break: 13:00 MYT
    dt_lunch = datetime(2024, 10, 1, 13, 0, tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))
    status_lunch = ExchangeSessionEngine.get_status("XKLS", as_of=dt_lunch)
    assert status_lunch.status == STATUS_LUNCH_BREAK
    assert status_lunch.is_open is False
    assert status_lunch.next_open == "2024-10-01 14:30:00"
    assert "Midday lunch break" in status_lunch.details

    # Afternoon session: 15:00 MYT
    dt_pm = datetime(2024, 10, 1, 15, 0, tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))
    status_pm = ExchangeSessionEngine.get_status("XKLS", as_of=dt_pm)
    assert status_pm.status == STATUS_OPEN
    assert status_pm.is_open is True
    assert "Afternoon trading session" in status_pm.details

    # Market closed after 17:00 MYT
    dt_close = datetime(2024, 10, 1, 18, 0, tzinfo=ZoneInfo("Asia/Kuala_Lumpur"))
    status_close = ExchangeSessionEngine.get_status("XKLS", as_of=dt_close)
    assert status_close.status == STATUS_CLOSED
    assert status_close.is_open is False
    assert "Market closed for the day" in status_close.details


def test_nyse_session_schedule():
    """Verify NYSE pre-market, regular trading hours, and after-hours."""
    # Pre-market: 08:00 ET on Tuesday 2024-10-01
    dt_pre = datetime(2024, 10, 1, 8, 0, tzinfo=ZoneInfo("America/New_York"))
    status_pre = ExchangeSessionEngine.get_status("XNYS", as_of=dt_pre)
    assert status_pre.status == STATUS_PRE_MARKET
    assert status_pre.is_open is False
    assert status_pre.next_open == "2024-10-01 09:30:00"

    # Regular session: 11:00 ET
    dt_reg = datetime(2024, 10, 1, 11, 0, tzinfo=ZoneInfo("America/New_York"))
    status_reg = ExchangeSessionEngine.get_status("XNYS", as_of=dt_reg)
    assert status_reg.status == STATUS_OPEN
    assert status_reg.is_open is True
    assert status_reg.next_close == "2024-10-01 16:00:00"

    # After-hours: 17:30 ET
    dt_post = datetime(2024, 10, 1, 17, 30, tzinfo=ZoneInfo("America/New_York"))
    status_post = ExchangeSessionEngine.get_status("XNYS", as_of=dt_post)
    assert status_post.status == STATUS_AFTER_HOURS
    assert status_post.is_open is False

    # Closed overnight: 23:00 ET
    dt_night = datetime(2024, 10, 1, 23, 0, tzinfo=ZoneInfo("America/New_York"))
    status_night = ExchangeSessionEngine.get_status("XNYS", as_of=dt_night)
    assert status_night.status == STATUS_CLOSED
    assert status_night.is_open is False


def test_weekend_and_holiday_states():
    """Verify Saturday/Sunday and calendar holiday detection."""
    # Weekend: Saturday 2024-10-05
    dt_sat = datetime(2024, 10, 5, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    status_sat = ExchangeSessionEngine.get_status("XNYS", as_of=dt_sat)
    assert status_sat.status == STATUS_WEEKEND
    assert status_sat.is_open is False
    assert status_sat.next_open == "2024-10-07 09:30:00"

    # NYSE Holiday: New Year's Day 2024-01-01
    dt_hol = datetime(2024, 1, 1, 12, 0, tzinfo=ZoneInfo("America/New_York"))
    status_hol = ExchangeSessionEngine.get_status("XNYS", as_of=dt_hol)
    assert status_hol.status == STATUS_HOLIDAY
    assert status_hol.is_open is False
    assert status_hol.next_open == "2024-01-02 09:30:00"


def test_twelve_data_symbol_resolution():
    """Verify Twelve Data ticker symbol formatting for NYSE and Bursa Malaysia."""
    provider = TwelveDataProvider(api_key="mock_key")
    assert provider._resolve_symbol("JPM", "XNYS") == "JPM"
    assert provider._resolve_symbol("1155.KL", "XKLS") == "1155:KLSE"
    assert provider._resolve_symbol("1155", "XKLS") == "1155:KLSE"


def test_twelve_data_fallback_when_no_api_key():
    """Verify fallback to Yahoo Finance when API key is empty."""
    mock_fallback = MagicMock()
    mock_fallback.name = "MockYahoo"
    mock_fallback.fetch_latest_quote.return_value = {
        "ticker": "JPM",
        "exchange": "XNYS",
        "last_price": 220.0,
        "previous_close": 218.0,
        "change": 2.0,
        "pct_change": 0.917,
        "status": FEED_DELAYED,
        "provider": "MockYahoo",
    }

    provider = TwelveDataProvider(api_key="", fallback_provider=mock_fallback)
    quote = provider.fetch_latest_quote("JPM", "XNYS")

    assert quote["last_price"] == 220.0
    assert quote["status"] == FEED_DELAYED
    assert "using Yahoo Finance fallback" in quote["note"]
    mock_fallback.fetch_latest_quote.assert_called_once_with("JPM", "XNYS")


def test_twelve_data_live_quote_parsing():
    """Verify parsing of valid Twelve Data API JSON payload with REALTIME badge."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "symbol": "JPM",
        "name": "JPMorgan Chase & Co.",
        "exchange": "NYSE",
        "currency": "USD",
        "open": "221.00",
        "high": "223.50",
        "low": "220.10",
        "close": "222.80",
        "previous_close": "220.00",
        "change": "2.80",
        "percent_change": "1.27",
        "volume": "8540000",
        "timestamp": 1727798400,
    }

    provider = TwelveDataProvider(api_key="valid_key")
    with patch("requests.get", return_value=mock_response):
        quote = provider.fetch_latest_quote("JPM", "XNYS")

    assert quote["last_price"] == 222.80
    assert quote["previous_close"] == 220.00
    assert quote["change"] == 2.80
    assert quote["pct_change"] == 1.27
    assert quote["status"] == FEED_REALTIME
    assert quote["provider"] == "TwelveData"


def test_twelve_data_rate_limiting():
    """Verify sliding-window rate limit triggers fallback when quota is reached."""
    mock_fallback = MagicMock()
    mock_fallback.fetch_latest_quote.return_value = {
        "ticker": "JPM",
        "last_price": 200.0,
        "status": FEED_DELAYED,
        "provider": "MockYahoo",
    }

    # Limit to 2 requests per minute
    provider = TwelveDataProvider(api_key="valid_key", max_requests_per_minute=2, fallback_provider=mock_fallback)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "close": "205.0",
        "previous_close": "200.0",
        "change": "5.0",
        "percent_change": "2.5",
    }

    with patch("requests.get", return_value=mock_resp):
        # 1st request: OK (Twelve Data)
        q1 = provider.fetch_latest_quote("JPM", "XNYS")
        assert q1["status"] == FEED_REALTIME

        # 2nd request: OK (Twelve Data)
        q2 = provider.fetch_latest_quote("JPM", "XNYS")
        assert q2["status"] == FEED_REALTIME

        # 3rd request: Over limit -> Fallback invoked!
        q3 = provider.fetch_latest_quote("JPM", "XNYS")
        assert q3["status"] == FEED_DELAYED
        assert "rate limit" in q3["note"]


def test_market_monitor_service_caching():
    """Verify MarketMonitorService in-memory caching and TTL expiration."""
    mock_provider = MagicMock()
    mock_provider.name = "MockProvider"
    mock_provider.fetch_latest_quote.return_value = {
        "ticker": "JPM",
        "exchange": "XNYS",
        "last_price": 225.0,
        "previous_close": 220.0,
        "change": 5.0,
        "pct_change": 2.27,
        "status": FEED_REALTIME,
        "currency": "USD",
        "provider": "MockProvider",
    }

    service = MarketMonitorService(provider=mock_provider, cache_ttl_seconds=60)

    # Initial call -> Hits provider
    quote1 = service.get_quote("JPM", "XNYS")
    assert quote1.cached is False
    assert quote1.last_price == 225.0
    assert mock_provider.fetch_latest_quote.call_count == 1

    # Immediate second call -> Served from cache
    quote2 = service.get_quote("JPM", "XNYS")
    assert quote2.cached is True
    assert quote2.last_price == 225.0
    assert mock_provider.fetch_latest_quote.call_count == 1

    # Force refresh call -> Bypasses cache
    quote3 = service.get_quote("JPM", "XNYS")
    assert quote3.cached is True

    quote4 = service.get_quote("JPM", "XNYS", force_refresh=True)
    assert quote4.cached is False
    assert mock_provider.fetch_latest_quote.call_count == 2
