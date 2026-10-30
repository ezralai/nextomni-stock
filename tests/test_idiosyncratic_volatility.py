"""Unit and integration test suite for Ang et al. (2006) Idiosyncratic Volatility Engine."""

import argparse
import io
import json
import math
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.idiosyncratic_volatility import (
    IdiosyncraticVolatilityReceipt,
    classify_ivol_regime_and_haircut,
    compute_idiosyncratic_higher_moments,
    evaluate_idiosyncratic_volatility_profile,
    extract_factor_residuals,
)
from src.cli import idiosyncratic_vol_command


@pytest.fixture
def sample_market_and_asset_returns():
    """Generate 100 sessions of synthetic benchmark and asset returns with known IVOL."""
    np.random.seed(42)
    n = 100
    rm = np.random.normal(0.0005, 0.012, n)
    beta = 1.2
    # Known idiosyncratic residuals with skewness
    eps = np.random.exponential(scale=0.008, size=n) - 0.008
    ri = 0.0002 + beta * rm + eps
    return pd.Series(ri), pd.Series(rm)


def test_compute_idiosyncratic_higher_moments_known_values() -> None:
    """Verify higher moment computations on symmetric and asymmetric known residual arrays."""
    # Perfectly symmetric zero-mean residuals
    resids = np.array([-0.02, -0.01, 0.0, 0.01, 0.02])
    dof = 4
    ivol, downside_ivol, ratio, iskew, ikurt = compute_idiosyncratic_higher_moments(resids, degrees_of_freedom=dof)

    # Hand calculation:
    # ss_resid = (-0.02)^2 + (-0.01)^2 + 0 + 0.01^2 + 0.02^2 = 0.0004 + 0.0001 + 0.0001 + 0.0004 = 0.0010
    # var_idio = 0.0010 / 4 = 0.00025
    # ivol_ann = sqrt(0.00025 * 252) * 100 = sqrt(0.063) * 100 =~ 25.0998%
    assert ivol == pytest.approx(25.0998, rel=1e-3)

    # Downside residuals: [-0.02, -0.01, 0.0, 0.0, 0.0]
    # ss_downside = 0.0005
    # var_downside = 0.0005 / 4 = 0.000125
    # downside_ivol = sqrt(0.000125 * 252) * 100 =~ 17.748%
    assert downside_ivol == pytest.approx(17.7482, rel=1e-3)
    assert ratio == pytest.approx(math.sqrt(0.5), rel=1e-3)

    # Symmetric array should have near-zero skewness
    assert abs(iskew) < 1e-6

    # Test positively skewed residuals (lottery ticket payoff)
    skewed_resids = np.array([-0.01, -0.01, -0.01, -0.01, 0.05])
    _, _, _, pos_skew, kurt = compute_idiosyncratic_higher_moments(skewed_resids, degrees_of_freedom=4)
    assert pos_skew > 0.5
    assert kurt > 2.0


def test_extract_factor_residuals_properties(sample_market_and_asset_returns: Tuple[pd.Series, pd.Series]) -> None:
    """Verify OLS factor residual extraction and variance decomposition."""
    r_asset, r_mkt = sample_market_and_asset_returns
    residuals, alpha_ann, total_vol_ann, r_sq, dof = extract_factor_residuals(
        asset_returns=r_asset,
        market_returns=r_mkt,
    )

    assert len(residuals) == len(r_asset)
    assert dof == len(r_asset) - 5
    assert 0.0 <= r_sq <= 1.0
    assert total_vol_ann > 0.0
    # Residual mean should be essentially zero
    assert abs(float(np.mean(residuals))) < 1e-4


def test_classify_ivol_regime_and_haircut() -> None:
    """Verify Ang et al. IVOL anomaly regime classification and haircut scaling."""
    # Low IVOL (defensive premium)
    regime, haircut = classify_ivol_regime_and_haircut(14.0)
    assert regime == "LOW_IVOL_ALPHA_PREMIUM"
    assert haircut == 0.0

    # Moderate IVOL
    regime, haircut = classify_ivol_regime_and_haircut(25.0)
    assert regime == "MODERATE_IVOL"
    assert haircut > 0.0

    # Elevated IVOL
    regime, haircut = classify_ivol_regime_and_haircut(42.0)
    assert regime == "ELEVATED_IVOL"
    assert haircut > 2.0

    # High IVOL drag (lottery stock)
    regime, haircut = classify_ivol_regime_and_haircut(65.0)
    assert regime == "HIGH_IVOL_ANOMALY_DRAG"
    assert haircut >= 6.0
    assert haircut <= 12.0


def test_evaluate_idiosyncratic_volatility_profile_integration() -> None:
    """Verify end-to-end IVOL evaluation workflow with price series."""
    np.random.seed(123)
    n = 80
    prices_asset = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))
    prices_bench = 400.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.010, n)))

    receipt = evaluate_idiosyncratic_volatility_profile(
        ticker="AAPL",
        asset_prices=prices_asset,
        benchmark_ticker="SPY",
        benchmark_prices=prices_bench,
        lookback_window=60,
    )

    assert isinstance(receipt, IdiosyncraticVolatilityReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.benchmark_ticker == "SPY"
    assert receipt.lookback_window == 60
    assert receipt.idiosyncratic_vol_ann_pct > 0.0
    assert receipt.downside_idiosyncratic_vol_ann_pct > 0.0
    assert 0.0 < receipt.downside_ivol_ratio <= 1.0
    assert receipt.idiosyncratic_variance_ratio_pct > 0.0
    assert receipt.verify_integrity() is True


def test_receipt_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    np.random.seed(999)
    prices_asset = 150.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.012, 60)))
    prices_bench = 450.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.009, 60)))

    receipt = evaluate_idiosyncratic_volatility_profile(
        ticker="NVDA",
        asset_prices=prices_asset,
        benchmark_ticker="SPY",
        benchmark_prices=prices_bench,
    )

    assert receipt.verify_integrity() is True

    # Tamper with IVOL
    receipt.idiosyncratic_vol_ann_pct += 5.0
    assert receipt.verify_integrity() is False

    # Reset and tamper with anomaly haircut
    receipt.idiosyncratic_vol_ann_pct -= 5.0
    receipt.ivol_expected_return_haircut_pct += 1.0
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify input validation against invalid tickers, non-finite inputs, and short series."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_idiosyncratic_volatility_profile(ticker="INVALID!TICKER", asset_prices=[100.0] * 30)

    # Insufficient observations
    with pytest.raises(ValueError, match="Insufficient asset price observations"):
        evaluate_idiosyncratic_volatility_profile(ticker="AAPL", asset_prices=[100.0, 101.0, 102.0])

    # Non-finite parameter
    with pytest.raises(ValueError, match="lookback_window must be finite"):
        evaluate_idiosyncratic_volatility_profile(
            ticker="AAPL",
            asset_prices=[100.0] * 30,
            benchmark_prices=[200.0] * 30,
            lookback_window=float("nan"),
        )


def test_cli_idiosyncratic_vol_command_stdout() -> None:
    """Verify CLI command standard formatted output."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    df_obs = pd.DataFrame({"adj_close": prices})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=False,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            idiosyncratic_vol_command(args)
            output = mock_stdout.getvalue()

            assert "Ang et al. (2006) Idiosyncratic Volatility" in output
            assert "AAPL" in output
            assert "SPY" in output
            assert "Idiosyncratic Vol (IVOL)" in output
            assert "Audit Digest (HMAC)" in output


def test_cli_idiosyncratic_vol_command_json() -> None:
    """Verify CLI command JSON output serialization."""
    np.random.seed(42)
    n = 70
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0004, 0.015, n)))
    df_obs = pd.DataFrame({"adj_close": prices})

    args = argparse.Namespace(
        ticker="AAPL",
        benchmark="SPY",
        lookback=60,
        json=True,
    )

    with patch("src.data.db.get_observations", return_value=df_obs):
        with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            idiosyncratic_vol_command(args)
            output = mock_stdout.getvalue()

            data = json.loads(output)
            assert data["ticker"] == "AAPL"
            assert data["benchmark_ticker"] == "SPY"
            assert "signature" in data
            assert "idiosyncratic_vol_ann_pct" in data
            assert "ivol_anomaly_regime" in data
