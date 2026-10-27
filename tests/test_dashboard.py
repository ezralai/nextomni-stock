"""Unit and integration tests for NextOmni Stock Streamlit dashboard.

Ensures imports, caching routines, queries, and plotting data preparations work without errors.
"""

import py_compile
from pathlib import Path
import pandas as pd
import pytest

from src.config import PROJECT_ROOT
from src.dashboard.app import (
    load_market_quotes,
    load_predictions_data,
    load_stock_observations,
)


def test_dashboard_py_compile():
    """Verify src/dashboard/app.py compiles without syntax errors."""
    app_path = PROJECT_ROOT / "src" / "dashboard" / "app.py"
    assert app_path.exists(), "Dashboard app.py must exist."
    compiled_path = py_compile.compile(str(app_path), doraise=True)
    assert compiled_path is not None


def test_load_stock_observations():
    """Verify load_stock_observations retrieves a valid DataFrame."""
    df = load_stock_observations("JPM", "XNYS")
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "session_date" in df.columns
        assert "adj_close" in df.columns


def test_load_predictions_data():
    """Verify load_predictions_data retrieves predictions DataFrame."""
    df = load_predictions_data()
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "prediction_id" in df.columns
        assert "status" in df.columns


def test_load_market_quotes():
    """Verify load_market_quotes returns list of MarketQuote objects."""
    quotes = load_market_quotes()
    assert isinstance(quotes, list)
    assert len(quotes) > 0
    first_q = quotes[0]
    assert hasattr(first_q, "ticker")
    assert hasattr(first_q, "last_price")
    assert hasattr(first_q, "session_status")
    assert hasattr(first_q, "latency_badge")
