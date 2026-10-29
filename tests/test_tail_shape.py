"""Test suite for Robust Quantile Tail Shape, Moors Kurtosis & Asymmetric Crash Risk Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.tail_shape import (
    GAUSSIAN_MOORS_KURTOSIS,
    TailShapeReceipt,
    compute_bowley_skewness,
    compute_hogg_tail_weight,
    compute_moors_kurtosis,
    evaluate_tail_shape_profile,
)
from src.cli import tail_shape_command


def test_moors_kurtosis_gaussian_vs_fat_tailed():
    """Verify Moors quantile kurtosis benchmark behavior across distributions."""
    np.random.seed(42)
    # Gaussian returns
    gaussian_ret = np.random.normal(0, 0.02, 5000)
    moors_gauss = compute_moors_kurtosis(gaussian_ret)
    assert abs(moors_gauss - GAUSSIAN_MOORS_KURTOSIS) < 0.15

    # Fat-tailed Student's t(3) returns
    t3_ret = np.random.standard_t(df=3, size=5000) * 0.02
    moors_fat = compute_moors_kurtosis(t3_ret)
    # Heavy tails must yield higher Moors kurtosis
    assert moors_fat > moors_gauss + 0.05


def test_bowley_skewness_directional_properties():
    """Verify Bowley quantile skewness detects directional tail asymmetry."""
    np.random.seed(42)
    # Symmetric
    sym = np.random.normal(0, 1, 3000)
    assert abs(compute_bowley_skewness(sym)) < 0.10

    # Right skewed (exponential)
    right = np.random.exponential(1.0, 3000)
    assert compute_bowley_skewness(right) > 0.20

    # Left skewed (negative exponential)
    left = -np.random.exponential(1.0, 3000)
    assert compute_bowley_skewness(left) < -0.20


def test_hogg_tail_weight_indicator():
    """Verify Hogg tail heaviness metric."""
    np.random.seed(42)
    gauss = np.random.normal(0, 1, 3000)
    hogg_gauss = compute_hogg_tail_weight(gauss)
    assert abs(hogg_gauss - 2.58) < 0.35

    fat = np.random.standard_t(df=2.5, size=3000)
    hogg_fat = compute_hogg_tail_weight(fat)
    assert hogg_fat > hogg_gauss


def test_evaluate_tail_shape_receipt_and_tamper():
    """Verify full tail shape receipt generation and HMAC tamper resistance."""
    np.random.seed(42)
    # Generate prices with severe left tail (crash susceptibility)
    n = 150
    ret = np.random.normal(0.001, 0.01, size=n)
    # Inject large negative jump crashes
    ret[20] = -0.09
    ret[65] = -0.08
    ret[110] = -0.07

    prices = 100.0 * np.exp(np.cumsum(ret))
    s_prices = pd.Series(prices)

    receipt = evaluate_tail_shape_profile("AAPL", s_prices, window=120)

    assert isinstance(receipt, TailShapeReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.sample_sessions == 119  # returns length
    assert receipt.tail_asymmetry_ratio > 1.0  # Left crash tail exceeds right tail
    assert receipt.verify_integrity() is True

    # Tamper with Moors kurtosis
    tampered_dict = receipt.to_dict()
    tampered_dict["moors_quantile_kurtosis"] += 1.0
    tampered_receipt = TailShapeReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False


def test_tail_shape_security_and_edge_cases():
    """Verify input validation, short series, and constant prices."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_tail_shape_profile("INVALID$$$", pd.Series([10.0] * 50))

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_tail_shape_profile("AAPL", pd.Series([10.0, 11.0, 12.0]))

    # Constant prices
    const_prices = pd.Series([100.0] * 40)
    ret_const = const_prices.pct_change().dropna().values
    assert compute_moors_kurtosis(ret_const) == 1.233


def test_cli_tail_shape_command_stdout_and_json(monkeypatch):
    """Verify CLI tail-shape command stdout formatting and JSON output."""
    np.random.seed(42)
    prices = pd.Series(100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, 80))))
    df_obs = pd.DataFrame({"close": prices})

    monkeypatch.setattr("src.data.db.get_observations", lambda ticker, db_path=None: df_obs)

    # 1. Standard stdout mode
    args_std = argparse.Namespace(ticker="AAPL", window=60, json=False)
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        tail_shape_command(args_std)
        output = fake_out.getvalue()
        assert "Robust Quantile Tail Shape & Moors Kurtosis Engine" in output
        assert "Security / Ticker:            AAPL" in output
        assert "Moors Quantile Kurtosis:" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(ticker="AAPL", window=60, json=True)
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        tail_shape_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert payload["ticker"] == "AAPL"
        assert "moors_quantile_kurtosis" in payload
        assert "tail_asymmetry_ratio" in payload
        assert "signature" in payload
