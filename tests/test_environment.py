"""Tests for environment dependencies, exchange calendars, and configuration integrity."""

import zoneinfo
import exchange_calendars as xcals
import pytest

from src.config import (
    HORIZON_FIVE_SESSIONS,
    HORIZON_NEXT_SESSION,
    HORIZON_TWENTY_ONE_SESSIONS,
    INITIAL_TICKERS,
    SUPPORTED_EXCHANGES,
    SUPPORTED_HORIZONS,
)


def test_core_dependencies_importable():
    """Verify that all core libraries can be imported without error."""
    import numpy as np
    import pandas as pd
    import plotly
    import sklearn
    import streamlit
    import yfinance

    assert np.__version__ is not None
    assert pd.__version__ is not None
    assert sklearn.__version__ is not None
    assert yfinance.__version__ is not None
    assert streamlit.__version__ is not None
    assert plotly.__version__ is not None


def test_exchange_calendars_availability():
    """Verify that exchange_calendars supports both NYSE (XNYS) and Bursa Malaysia (XKLS)."""
    nyse = xcals.get_calendar("XNYS")
    bursa = xcals.get_calendar("XKLS")

    assert nyse.name == "XNYS"
    assert bursa.name == "XKLS"

    # Test that a regular historical trading day (e.g. Wednesday 2024-01-10) was recognized
    assert nyse.is_session("2024-01-10") is True
    assert bursa.is_session("2024-01-10") is True

    # Test that a weekend (Saturday 2024-01-13) is NOT recognized as a session
    assert nyse.is_session("2024-01-13") is False
    assert bursa.is_session("2024-01-13") is False


def test_config_exchanges_and_timezones():
    """Verify that all supported exchanges have valid timezones and currencies."""
    assert "XNYS" in SUPPORTED_EXCHANGES
    assert "XKLS" in SUPPORTED_EXCHANGES

    for mic, cfg in SUPPORTED_EXCHANGES.items():
        assert cfg.mic == mic
        # Verify valid IANA timezone name
        tz = zoneinfo.ZoneInfo(cfg.timezone)
        assert tz is not None
        assert cfg.currency in {"USD", "MYR"}


def test_config_horizons():
    """Verify that forecasting horizons strictly match the project specifications."""
    assert HORIZON_NEXT_SESSION == 1
    assert HORIZON_FIVE_SESSIONS == 5
    assert HORIZON_TWENTY_ONE_SESSIONS == 21
    assert len(SUPPORTED_HORIZONS) == 3


def test_initial_tickers():
    """Verify that initial tickers exist for both supported exchanges."""
    assert "JPM" in INITIAL_TICKERS["XNYS"]
    assert "1155.KL" in INITIAL_TICKERS["XKLS"]
