"""Unit and integration test suite for Joel Hasbrouck (1991) VAR Price Discovery & Information Share Engine."""

import argparse
import io
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.hasbrouck_var import (
    HasbrouckReceipt,
    classify_market_quality_regime,
    construct_signed_order_flow,
    estimate_hasbrouck_var,
    evaluate_hasbrouck_var_profile,
)
from src.cli import hasbrouck_var_command


def test_hasbrouck_var_known_synthetic_process() -> None:
    """Verify VAR estimation and impulse response convergence on a synthetic structural process."""
    np.random.seed(42)
    n = 250
    v1 = np.random.normal(0, 0.005, n)
    v2 = np.random.normal(0, 1.0, n)
    r = np.zeros(n)
    x = np.zeros(n)

    # True parameters: positive trade impact b0 = 0.004, autocorrelation d1 = 0.3
    for t in range(1, n):
        x[t] = 0.30 * x[t - 1] - 2.0 * r[t - 1] + v2[t]
        r[t] = -0.15 * r[t - 1] + 0.004 * x[t] - 0.001 * x[t - 1] + v1[t]

    (
        contemp_bps,
        perm_bps,
        transient_bps,
        perm_ratio_pct,
        rw2_pct,
        sig_v2,
        pub_bps,
        irf_curve,
        regime,
    ) = estimate_hasbrouck_var(
        returns=r,
        trade_flows=x,
        lags=2,
        irf_horizon=20,
    )

    # b_0 should be approx 0.004 * 10,000 = 40 bps
    assert contemp_bps == pytest.approx(40.0, abs=15.0)
    # Permanent impact should be positive
    assert perm_bps > 10.0
    # IRF curve should have 20 steps
    assert len(irf_curve) == 20
    # The IRF curve should converge (difference between last steps should be small)
    assert abs(irf_curve[-1] - irf_curve[-2]) < 0.5

    # Random-walk variance share R_w^2 should be bounded in [0, 100]
    assert 0.0 < rw2_pct < 100.0
    assert regime in ("HIGH_INFORMATIONAL_ASYMMETRY", "MODERATE_INFORMED_FLOW", "PUBLIC_NEWS_DOMINATED")


def test_hasbrouck_var_transient_decay_inventory_bounce() -> None:
    """Verify detection of transient inventory bounce where initial impact exceeds permanent revision."""
    np.random.seed(123)
    n = 300
    v1 = np.random.normal(0, 0.004, n)
    v2 = np.random.normal(0, 1.0, n)
    r = np.zeros(n)
    x = np.zeros(n)

    # Initial impact b0 = 0.008, but strong subsequent mean-reversion (b1 = -0.005)
    for t in range(1, n):
        x[t] = 0.20 * x[t - 1] + v2[t]
        r[t] = -0.30 * r[t - 1] + 0.008 * x[t] - 0.005 * x[t - 1] + v1[t]

    (
        contemp_bps,
        perm_bps,
        transient_bps,
        perm_ratio_pct,
        rw2_pct,
        _,
        _,
        _,
        _,
    ) = estimate_hasbrouck_var(
        returns=r,
        trade_flows=x,
        lags=2,
        irf_horizon=15,
    )

    # Initial impact should exceed permanent impact (transient friction bounce > 0)
    assert contemp_bps > perm_bps
    assert transient_bps > 0.0
    assert perm_ratio_pct < 100.0


def test_hasbrouck_var_pure_noise_market() -> None:
    """Verify that pure noise uncorrelated trade flow produces low trade information share."""
    np.random.seed(999)
    n = 250
    # Uncorrelated returns (large public news) and random trade flows
    r = np.random.normal(0, 0.02, n)
    x = np.random.normal(0, 1.0, n)

    (
        contemp_bps,
        perm_bps,
        transient_bps,
        perm_ratio_pct,
        rw2_pct,
        _,
        pub_bps,
        _,
        regime,
    ) = estimate_hasbrouck_var(
        returns=r,
        trade_flows=x,
        lags=2,
        irf_horizon=15,
    )

    # Trade information share should be small because trades carry no private info
    assert rw2_pct < 15.0
    assert regime == "PUBLIC_NEWS_DOMINATED"


def test_construct_signed_order_flow_lee_ready() -> None:
    """Verify Lee-Ready signed order flow proxy generation and standardization."""
    prices = [100.0, 101.0, 101.0, 99.0, 99.5, 102.0] * 5  # 30 observations
    volumes = [1000.0, 1500.0, 800.0, 2000.0, 1200.0, 1800.0] * 5

    rets, flows = construct_signed_order_flow(prices, volumes)

    assert len(rets) == len(prices) - 1
    assert len(flows) == len(volumes) - 1
    # Check standardization properties
    assert np.mean(flows) == pytest.approx(0.0, abs=1e-5)
    assert np.std(flows, ddof=1) == pytest.approx(1.0, abs=1e-5)


def test_market_quality_regime_classification() -> None:
    """Verify Hasbrouck information share tier classification."""
    assert classify_market_quality_regime(45.0) == "HIGH_INFORMATIONAL_ASYMMETRY"
    assert classify_market_quality_regime(25.0) == "MODERATE_INFORMED_FLOW"
    assert classify_market_quality_regime(10.0) == "PUBLIC_NEWS_DOMINATED"


def test_hasbrouck_receipt_tamper_integrity() -> None:
    """Verify HMAC-SHA256 signature verification and tamper detection."""
    np.random.seed(77)
    n = 100
    prices = 150.0 + np.cumsum(np.random.normal(0.05, 1.0, n))
    volumes = np.random.uniform(500000, 2000000, n)

    receipt = evaluate_hasbrouck_var_profile(
        ticker="AAPL",
        asset_prices=prices,
        asset_volumes=volumes,
        var_lags=3,
        irf_horizon=15,
        lookback_window=80,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.verify_integrity() is True
    assert len(receipt.signature) == 64

    # Tampering with permanent impact invalidates signature
    receipt.permanent_impact_bps += 5.0
    assert receipt.verify_integrity() is False


def test_hasbrouck_cli_command_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with formatted terminal report."""
    np.random.seed(88)
    n = 60
    prices = 200.0 + np.cumsum(np.random.normal(0.0, 1.5, n))
    volumes = np.random.uniform(1e6, 3e6, n)

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices, "volume": volumes})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="hasbrouck-var",
            ticker="MSFT",
            lags=3,
            horizon=15,
            lookback=50,
            json=False,
        )

        hasbrouck_var_command(args)
        captured = capsys.readouterr().out

        assert "Hasbrouck (1991) VAR Price Discovery & Information Share" in captured
        assert "Security / Ticker:            MSFT" in captured
        assert "Permanent Impact I(∞):" in captured
        assert "Trade Info Share (R_w²):" in captured
        assert "Audit Digest (HMAC):" in captured
        assert "Verified: True" in captured


def test_hasbrouck_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with JSON output."""
    np.random.seed(88)
    n = 60
    prices = 120.0 + np.cumsum(np.random.normal(0.0, 1.0, n))
    volumes = np.random.uniform(1e6, 2e6, n)

    with patch("src.data.db.get_observations") as mock_db:
        mock_df = pd.DataFrame({"adj_close": prices, "volume": volumes})
        mock_db.return_value = mock_df

        args = argparse.Namespace(
            command="hasbrouck-var",
            ticker="NVDA",
            lags=3,
            horizon=15,
            lookback=50,
            json=True,
        )

        hasbrouck_var_command(args)
        captured = capsys.readouterr().out
        data = json.loads(captured)

        assert data["ticker"] == "NVDA"
        assert "permanent_impact_bps" in data
        assert "information_share_rw2_pct" in data
        assert "signature" in data


def test_hasbrouck_input_validation() -> None:
    """Verify input validation guards against invalid tickers or parameters."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_hasbrouck_var_profile(ticker="BAD$$$", asset_prices=[100.0] * 30, asset_volumes=[1000.0] * 30)

    with pytest.raises(ValueError, match="Insufficient observations"):
        construct_signed_order_flow(prices=[100.0] * 10, volumes=[1000.0] * 10)
