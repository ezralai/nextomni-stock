"""Unit and integration tests for Conditional Drawdown-at-Risk (CDaR) and path-dependent risk analytics."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import (
    PathRiskReceipt,
    calculate_cdar,
    calculate_dar,
    calculate_pain_index,
    calculate_ulcer_index,
    calculate_underwater_durations,
    compute_drawdown_series,
    compute_path_risk_profile,
)
from src.cli import risk_report_command


@pytest.fixture
def sample_drawdown_prices_df() -> pd.DataFrame:
    """Generate price path with a known 25% peak-to-trough crash and partial recovery."""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="B").strftime("%Y-%m-%d").tolist()

    # Phase 1: 100 -> 120 (40 sessions)
    p1 = np.linspace(100.0, 120.0, 40)
    # Phase 2: 120 -> 90 (25% crash, 20 sessions)
    p2 = np.linspace(120.0, 90.0, 20)
    # Phase 3: 90 -> 105 (partial recovery, 40 sessions)
    p3 = np.linspace(90.0, 105.0, 40)

    prices = np.concatenate([p1, p2, p3])

    return pd.DataFrame(
        {
            "session_date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.uniform(1e6, 2e6, 100),
        }
    )


def test_compute_drawdown_series_monotonic_and_crash() -> None:
    """Verify drawdown calculation on monotonic increase, severe crash, and boundary properties."""
    # Monotonically increasing prices
    mon_prices = [10.0, 12.0, 15.0, 20.0, 25.0]
    dd_mon = compute_drawdown_series(mon_prices)
    assert np.allclose(dd_mon.values, 0.0)

    # 50% crash: 100 -> 50
    crash_prices = [100.0, 100.0, 50.0]
    dd_crash = compute_drawdown_series(crash_prices)
    assert np.isclose(dd_crash.iloc[-1], 0.50)

    # Empty inputs
    assert compute_drawdown_series([]).empty
    assert np.allclose(compute_drawdown_series([-10.0, 0.0]).values, 0.0)


def test_cdar_and_dar_mathematical_properties() -> None:
    """Verify CDaR is always greater than or equal to DaR (coherence property)."""
    # Sample drawdowns with fat tail
    drawdowns = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]

    dar_90 = calculate_dar(drawdowns, confidence_level=0.90)
    cdar_90 = calculate_cdar(drawdowns, confidence_level=0.90)

    assert dar_90 > 0.0
    assert cdar_90 >= dar_90, f"Expected CDaR ({cdar_90}) >= DaR ({dar_90})"

    # All-zero drawdowns
    zero_dd = [0.0] * 10
    assert calculate_dar(zero_dd) == 0.0
    assert calculate_cdar(zero_dd) == 0.0

    # Insufficient data guard
    assert calculate_dar([0.05, 0.10]) == 0.0
    assert calculate_cdar([0.05, 0.10]) == 0.0


def test_ulcer_index_and_pain_index() -> None:
    """Verify Ulcer Index and Pain Index calculations and positive monotonicity."""
    low_dd = [0.01, 0.02, 0.01, 0.03, 0.02]
    high_dd = [0.10, 0.20, 0.15, 0.25, 0.30]

    ui_low = calculate_ulcer_index(low_dd)
    ui_high = calculate_ulcer_index(high_dd)
    pain_low = calculate_pain_index(low_dd)
    pain_high = calculate_pain_index(high_dd)

    assert ui_high > ui_low > 0.0
    assert pain_high > pain_low > 0.0

    # Empty/insufficient guards
    assert calculate_ulcer_index([]) == 0.0
    assert calculate_pain_index([]) == 0.0


def test_calculate_underwater_durations() -> None:
    """Verify calculation of max and ongoing underwater sessions."""
    # Prices: 100, 90, 80, 100 (underwater for 2 sessions), 95, 90 (currently underwater for 2 sessions)
    prices = [100.0, 90.0, 80.0, 100.0, 95.0, 90.0]
    dd = compute_drawdown_series(prices)
    max_uw, curr_uw = calculate_underwater_durations(dd)

    assert max_uw == 2
    assert curr_uw == 2

    # Monotonically rising
    dd_rising = compute_drawdown_series([10.0, 20.0, 30.0, 40.0])
    max_uw_r, curr_uw_r = calculate_underwater_durations(dd_rising)
    assert max_uw_r == 0
    assert curr_uw_r == 0


def test_compute_path_risk_profile_comprehensive(sample_drawdown_prices_df: pd.DataFrame) -> None:
    """Verify compute_path_risk_profile returns complete PathRiskReceipt on full dataset."""
    receipt = compute_path_risk_profile(sample_drawdown_prices_df, ticker="TEST_CORP")

    assert isinstance(receipt, PathRiskReceipt)
    assert receipt.ticker == "TEST_CORP"
    assert receipt.sample_sessions == 100
    assert receipt.high_water_mark == 120.0
    assert receipt.current_price == 105.0
    # Current drawdown: (120 - 105) / 120 = 12.5%
    assert np.isclose(receipt.current_drawdown_pct, 12.5, atol=0.2)
    # Max drawdown: (120 - 90) / 120 = 25.0%
    assert np.isclose(receipt.max_drawdown_pct, 25.0, atol=0.2)
    assert receipt.cdar_95_pct >= receipt.dar_95_pct
    assert receipt.cdar_99_pct >= receipt.cdar_95_pct
    assert receipt.ulcer_index_pct > 0.0
    assert receipt.pain_index_pct > 0.0
    assert receipt.max_underwater_sessions > 0
    assert receipt.current_underwater_sessions > 0

    d = receipt.to_dict()
    assert d["ticker"] == "TEST_CORP"
    assert "summary" in d
    assert "cdar_95_pct" in d


def test_compute_path_risk_profile_insufficient_data() -> None:
    """Verify graceful receipt fallback on empty or insufficient data."""
    receipt = compute_path_risk_profile(pd.DataFrame(), ticker="EMPTY")
    assert receipt.sample_sessions == 0
    assert receipt.max_drawdown_pct == 0.0
    assert receipt.cdar_95_pct == 0.0
    assert "Insufficient" in receipt.summary


def test_cli_risk_report_command_json_and_stdout(sample_drawdown_prices_df: pd.DataFrame) -> None:
    """Verify CLI risk-report outputs path-risk metrics in stdout and JSON modes."""
    args_json = argparse.Namespace(ticker="JPM", stress_test=True, json=True)

    with patch("src.data.db.get_observations", return_value=sample_drawdown_prices_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        risk_report_command(args_json)
        output = mock_out.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["ticker"] == "JPM"
        assert "multi_horizon_var" in parsed
        assert "path_risk_profile" in parsed
        assert parsed["path_risk_profile"]["max_drawdown_pct"] > 0.0

    args_txt = argparse.Namespace(ticker="JPM", stress_test=False, json=False)
    with patch("src.data.db.get_observations", return_value=sample_drawdown_prices_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        risk_report_command(args_txt)
        output = mock_out.getvalue()
        assert "Multi-Horizon Quantitative Tail Risk Report" in output
        assert "Path-Dependent Cumulative Drawdown & CDaR Risk Profile" in output
        assert "Conditional DaR (95% CDaR)" in output
        assert "Peter Martin's Ulcer Index" in output
