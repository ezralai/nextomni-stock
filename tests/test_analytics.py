"""Tests for NextOmni Stock exploratory data analysis, statistics, and visualization."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.eda import (
    compute_stock_analytics,
    export_eda_report,
    generate_eda_figures,
)
from src.data.db import init_db, save_observations, upsert_stock


@pytest.fixture
def sample_observations() -> pd.DataFrame:
    """Deterministic synthetic observations for analytics testing."""
    dates = [
        "2024-01-08",
        "2024-01-09",
        "2024-01-10",
        "2024-01-11",
        "2024-01-12",
        "2024-01-15",
    ]
    prices = [100.0, 120.0, 90.0, 105.0, 110.0, 115.0]
    volumes = [1000, 1500, 2000, 1200, 1300, 1400]

    return pd.DataFrame(
        {
            "ticker": ["TEST"] * 6,
            "exchange": ["XNYS"] * 6,
            "session_date": dates,
            "open": prices,
            "high": [p + 2.0 for p in prices],
            "low": [p - 2.0 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "data_status": ["END_OF_DAY"] * 6,
            "provider": ["Mock"] * 6,
        }
    )


def test_analytics_returns_and_drawdown(sample_observations: pd.DataFrame):
    """Verify exact mathematical calculation of returns and maximum drawdown."""
    enhanced_df, stats = compute_stock_analytics(sample_observations)

    # Day 0: No return (NaN)
    assert pd.isna(enhanced_df["daily_return"].iloc[0])
    # Day 1: 100 -> 120 (+20%)
    assert pytest.approx(enhanced_df["daily_return"].iloc[1], 0.0001) == 0.20
    # Day 2: 120 -> 90 (-25%)
    assert pytest.approx(enhanced_df["daily_return"].iloc[2], 0.0001) == -0.25

    # Peak at Day 1 was 120. Day 2 price was 90. Drawdown = (90 - 120) / 120 = -0.25 (-25%)
    assert pytest.approx(enhanced_df["drawdown"].iloc[2], 0.0001) == -0.25
    assert pytest.approx(stats.max_drawdown_pct, 0.01) == -25.0

    # Cumulative return: (115 - 100) / 100 = +15%
    assert pytest.approx(stats.cumulative_return_pct, 0.01) == 15.0
    assert stats.min_price == 90.0
    assert stats.max_price == 120.0
    assert stats.total_sessions == 6


def test_analytics_moving_averages(sample_observations: pd.DataFrame):
    """Verify moving average calculations."""
    enhanced_df, _ = compute_stock_analytics(sample_observations)

    # SMA 5 at index 4 (first 5 prices: 100, 120, 90, 105, 110. Sum = 525, Mean = 105)
    assert pytest.approx(enhanced_df["sma_5"].iloc[4], 0.0001) == 105.0


def test_analytics_insufficient_data():
    """Verify ValueError is raised when observations are insufficient."""
    empty_df = pd.DataFrame()
    with pytest.raises(ValueError, match="at least 2 observations"):
        compute_stock_analytics(empty_df)

    single_row = pd.DataFrame(
        {
            "ticker": ["TEST"],
            "exchange": ["XNYS"],
            "session_date": ["2024-01-08"],
            "adj_close": [100.0],
            "volume": [1000],
        }
    )
    with pytest.raises(ValueError, match="at least 2 observations"):
        compute_stock_analytics(single_row)


def test_generate_figures(sample_observations: pd.DataFrame):
    """Verify all 4 Plotly interactive figures are generated."""
    enhanced_df, _ = compute_stock_analytics(sample_observations)
    figures = generate_eda_figures(enhanced_df, ticker="TEST")

    assert "price_and_volume" in figures
    assert "return_distribution" in figures
    assert "rolling_volatility" in figures
    assert "drawdown" in figures


def test_export_eda_report(tmp_path: Path, sample_observations: pd.DataFrame):
    """Verify end-to-end HTML report generation."""
    test_db = tmp_path / "test.db"
    init_db(test_db)
    upsert_stock("TEST", "XNYS", "Test Corp", "USD", "America/New_York", test_db)
    save_observations(sample_observations, test_db)

    stats, report_path = export_eda_report(
        ticker="TEST", output_dir=tmp_path, db_path=test_db
    )

    assert stats.ticker == "TEST"
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "NextOmni Stock: Exploratory Data Analysis & Statistics" in content
    assert "TEST" in content
