"""Unit and integration tests for multi-estimator realized volatility engine."""

import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.analytics.volatility import (
    VolatilityDiagnosticsReceipt,
    compute_close_to_close_volatility,
    compute_ewma_volatility,
    compute_garman_klass_volatility,
    compute_parkinson_volatility,
    compute_rogers_satchell_volatility,
    compute_yang_zhang_volatility,
    evaluate_volatility_profile,
)
from src.cli import volatility_report_command


@pytest.fixture
def sample_ohlc_df() -> pd.DataFrame:
    """Generate realistic synthetic OHLC series with known volatility dynamics."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")

    # Generate random walk with 15% annual volatility
    dt = 1.0 / 252.0
    sigma = 0.15
    returns = np.random.normal(0.0005, sigma * math.sqrt(dt), n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Realistic Open, High, Low relative to Close
    # Overnight gap
    open_p = np.roll(close, 1) * np.exp(np.random.normal(0, 0.002, n))
    open_p[0] = 100.0
    high_p = np.maximum(open_p, close) * np.exp(np.abs(np.random.normal(0, 0.005, n)))
    low_p = np.minimum(open_p, close) * np.exp(-np.abs(np.random.normal(0, 0.005, n)))

    return pd.DataFrame(
        {
            "session_date": dates,
            "open": open_p,
            "high": high_p,
            "low": low_p,
            "close": close,
            "volume": np.random.uniform(1e6, 5e6, n),
        }
    )


def test_compute_close_to_close_volatility_bounds(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify close-to-close volatility produces finite positive annualized values."""
    s = compute_close_to_close_volatility(sample_ohlc_df, window=20)
    assert isinstance(s, pd.Series)
    assert len(s) == len(sample_ohlc_df)
    valid = s.dropna()
    assert len(valid) > 0
    assert (valid > 0.0).all()
    # Should be around 15% annual vol (+- 10%)
    assert 0.05 < valid.iloc[-1] < 0.35


def test_compute_parkinson_volatility_bounds(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify Parkinson high-low estimator produces strictly non-negative volatility."""
    s = compute_parkinson_volatility(sample_ohlc_df, window=20)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()
    assert 0.05 < valid.iloc[-1] < 0.35


def test_compute_garman_klass_volatility_bounds(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify Garman-Klass continuous diffusion estimator bounds."""
    s = compute_garman_klass_volatility(sample_ohlc_df, window=20)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()
    assert 0.05 < valid.iloc[-1] < 0.35


def test_compute_rogers_satchell_volatility_bounds(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify Rogers-Satchell drift-independent estimator bounds."""
    s = compute_rogers_satchell_volatility(sample_ohlc_df, window=20)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()


def test_compute_yang_zhang_volatility_properties(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify Yang-Zhang minimum-variance unbiased estimator produces positive volatility."""
    s = compute_yang_zhang_volatility(sample_ohlc_df, window=20)
    assert isinstance(s, pd.Series)
    valid = s.dropna()
    assert len(valid) > 0
    assert (valid >= 0.0).all()
    # Yang-Zhang should capture overnight jumps and intraday moves
    assert 0.05 < valid.iloc[-1] < 0.40


def test_compute_ewma_volatility_properties(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify RiskMetrics EWMA volatility filter runs and responds smoothly."""
    s = compute_ewma_volatility(sample_ohlc_df, lambda_param=0.94)
    assert isinstance(s, pd.Series)
    assert len(s) == len(sample_ohlc_df)
    assert (s >= 0.0).all()
    assert 0.05 < s.iloc[-1] < 0.40


def test_degenerate_identical_high_low_handling() -> None:
    """Verify flat / zero-volatility sessions handle gracefully without divide-by-zero or NaNs."""
    n = 30
    flat_df = pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [100.0] * n,
            "low": [100.0] * n,
            "close": [100.0] * n,
        }
    )
    s_p = compute_parkinson_volatility(flat_df, window=10)
    s_gk = compute_garman_klass_volatility(flat_df, window=10)
    s_yz = compute_yang_zhang_volatility(flat_df, window=10)

    assert (s_p.fillna(0.0) == 0.0).all()
    assert (s_gk.fillna(0.0) == 0.0).all()
    assert (s_yz.fillna(0.0) == 0.0).all()


def test_evaluate_volatility_profile_comprehensive(sample_ohlc_df: pd.DataFrame) -> None:
    """Verify evaluate_volatility_profile produces complete, valid diagnostic receipt."""
    receipt = evaluate_volatility_profile(sample_ohlc_df, ticker="TEST", window=20)

    assert isinstance(receipt, VolatilityDiagnosticsReceipt)
    assert receipt.ticker == "TEST"
    assert receipt.sample_sessions == 100
    assert receipt.window == 20
    assert receipt.close_to_close_ann_pct > 0.0
    assert receipt.parkinson_ann_pct > 0.0
    assert receipt.garman_klass_ann_pct > 0.0
    assert receipt.yang_zhang_ann_pct > 0.0
    assert receipt.ewma_ann_pct > 0.0
    assert receipt.volatility_ratio_yz_cc > 0.0
    assert receipt.regime_signal in [
        "NORMAL_DIFFUSION",
        "OVERNIGHT_GAP_OR_INTRADAY_SURGE",
        "MEAN_REVERTING_INTRADAY",
        "HIGH_VOLATILITY_EXPANSION",
        "COMPRESSED_VOLATILITY",
    ]
    assert "20d" in receipt.term_structure_yang_zhang

    d = receipt.to_dict()
    assert d["ticker"] == "TEST"
    assert "yang_zhang_ann_pct" in d


def test_evaluate_volatility_profile_insufficient_data() -> None:
    """Verify safe fallback when sample size is insufficient."""
    short_df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
        }
    )
    receipt = evaluate_volatility_profile(short_df, ticker="SHORT", window=20)
    assert receipt.regime_signal == "INSUFFICIENT_DATA"
    assert receipt.yang_zhang_ann_pct == 0.0


def test_cli_volatility_report_command(sample_ohlc_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify CLI volatility-report execution in standard and JSON modes."""
    monkeypatch.setattr("src.data.db.get_observations", lambda ticker: sample_ohlc_df)

    # Human-readable mode
    args = argparse.Namespace(ticker="AAPL", window=20, json=False)
    volatility_report_command(args)
    captured = capsys.readouterr()
    assert "Multi-Estimator Realized Volatility Diagnostics" in captured.out
    assert "Yang-Zhang (2000 Optimal)" in captured.out
    assert "Volatility Ratio (YZ / CC)" in captured.out

    # JSON mode
    args_json = argparse.Namespace(ticker="AAPL", window=20, json=True)
    volatility_report_command(args_json)
    captured_json = capsys.readouterr()
    parsed = json.loads(captured_json.out)
    assert parsed["ticker"] == "AAPL"
    assert "yang_zhang_ann_pct" in parsed
    assert "volatility_ratio_yz_cc" in parsed
