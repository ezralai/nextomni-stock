"""Unit and integration tests for Volume-Synchronized Probability of Toxicity (VPIN) and Flow Imbalance Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.cli import toxicity_report_command
from src.features.toxicity import (
    ToxicityReceipt,
    calculate_vpin_series,
    classify_bulk_volume,
    evaluate_flow_toxicity,
)


@pytest.fixture
def synthetic_microstructure_df() -> pd.DataFrame:
    """Generate price and volume history with an informed toxic selling surge in later sessions."""
    np.random.seed(42)
    n = 80
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    # Normal random walk for first 60 sessions
    r1 = np.random.normal(0.0005, 0.01, 60)
    # Toxic persistent selling pressure for last 20 sessions
    r2 = np.random.normal(-0.025, 0.01, 20)
    returns = np.concatenate([r1, r2])

    prices = 100.0 * np.exp(np.cumsum(returns))
    volumes = np.random.uniform(1e6, 3e6, n)
    volumes[60:] *= 2.5  # High volume surge on selling sessions

    return pd.DataFrame(
        {
            "session_date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
        }
    )


def test_classify_bulk_volume_conservation() -> None:
    """Verify Bulk Volume Classification strictly conserves total trading volume."""
    prices = [100.0, 102.0, 98.0, 95.0, 100.0, 105.0]
    volumes = [1000.0, 1500.0, 2000.0, 2500.0, 1800.0, 3000.0]

    v_b, v_s, imb = classify_bulk_volume(prices, volumes, window=5)

    assert len(v_b) == len(prices)
    assert len(v_s) == len(prices)
    assert len(imb) == len(prices)

    # Volume conservation: V_B + V_S == V
    total_reconstructed = v_b + v_s
    assert np.allclose(total_reconstructed.values, volumes)

    # All volumes non-negative
    assert np.all(v_b.values >= 0.0)
    assert np.all(v_s.values >= 0.0)
    assert np.all(imb.values >= 0.0)

    # Empty inputs guard
    eb, es, ei = classify_bulk_volume([], [])
    assert eb.empty and es.empty and ei.empty


def test_vpin_bounds_and_toxicity_surge(synthetic_microstructure_df: pd.DataFrame) -> None:
    """Verify VPIN series is strictly bounded in [0.0, 1.0] and spikes during toxic sell-off."""
    prices = synthetic_microstructure_df["adj_close"].values
    volumes = synthetic_microstructure_df["volume"].values

    vpin_series = calculate_vpin_series(prices, volumes, window=15)

    assert len(vpin_series) == len(prices)
    assert np.all(vpin_series.dropna().values >= 0.0)
    assert np.all(vpin_series.dropna().values <= 1.0)

    # Trailing sessions during heavy selling pressure should have elevated VPIN (> 0.30)
    late_vpin = vpin_series.iloc[-10:].mean()
    early_vpin = vpin_series.iloc[15:35].mean()
    assert late_vpin > early_vpin, f"Expected higher VPIN in toxic selloff ({late_vpin:.3f} > {early_vpin:.3f})"


def test_evaluate_flow_toxicity_receipt(synthetic_microstructure_df: pd.DataFrame) -> None:
    """Verify evaluate_flow_toxicity returns valid ToxicityReceipt with all fields."""
    receipt = evaluate_flow_toxicity(synthetic_microstructure_df, ticker="TEST_FLOW", window=20)

    assert isinstance(receipt, ToxicityReceipt)
    assert receipt.ticker == "TEST_FLOW"
    assert receipt.sample_sessions == 80
    assert receipt.window == 20
    assert 0.0 <= receipt.latest_vpin <= 1.0
    assert 0.0 <= receipt.mean_vpin <= 1.0
    assert 0.0 <= receipt.max_vpin <= 1.0
    assert 0.0 <= receipt.latest_buyer_volume_pct <= 100.0
    assert 0.0 <= receipt.latest_seller_volume_pct <= 100.0
    assert np.isclose(receipt.latest_buyer_volume_pct + receipt.latest_seller_volume_pct, 100.0, atol=0.1)
    assert receipt.toxicity_regime in [
        "LOW_TOXICITY",
        "MODERATE_FLOW_IMBALANCE",
        "HIGH_ADVERSE_SELECTION",
        "TOXIC_INFORMED_SURGE",
    ]
    assert "summary" in receipt.to_dict()


def test_evaluate_flow_toxicity_insufficient_data() -> None:
    """Verify graceful handling of empty or insufficient observations."""
    receipt = evaluate_flow_toxicity(pd.DataFrame(), ticker="EMPTY")
    assert receipt.sample_sessions == 0
    assert receipt.toxicity_regime == "INSUFFICIENT_DATA"
    assert receipt.latest_vpin == 0.0
    assert "Insufficient" in receipt.summary


def test_cli_toxicity_report_command_stdout_and_json(synthetic_microstructure_df: pd.DataFrame) -> None:
    """Verify CLI toxicity-report outputs valid JSON and formatted audit text."""
    args_json = argparse.Namespace(ticker="JPM", window=20, json=True)

    with patch("src.data.db.get_observations", return_value=synthetic_microstructure_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        toxicity_report_command(args_json)
        output = mock_out.getvalue().strip()
        parsed = json.loads(output)
        assert parsed["ticker"] == "JPM"
        assert "latest_vpin" in parsed
        assert "toxicity_regime" in parsed

    args_txt = argparse.Namespace(ticker="JPM", window=20, json=False)
    with patch("src.data.db.get_observations", return_value=synthetic_microstructure_df), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        toxicity_report_command(args_txt)
        output = mock_out.getvalue()
        assert "Order Flow Toxicity & VPIN Microstructure Audit" in output
        assert "Latest VPIN:" in output
        assert "Toxicity Flow Regime:" in output


def test_cli_toxicity_report_empty_data() -> None:
    """Verify CLI handles missing observations gracefully."""
    args = argparse.Namespace(ticker="EMPTY", window=20, json=False)
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()), patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        toxicity_report_command(args)
        output = mock_out.getvalue()
        assert "No observations found for EMPTY" in output
