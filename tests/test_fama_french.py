"""Unit test suite for Fama-French (1993) & Carhart (1997) Multi-Factor Attribution Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.fama_french import (
    FamaFrenchReceipt,
    construct_synthetic_factor_proxies,
    evaluate_fama_french_carhart_attribution,
    evaluate_fama_french_profile,
)
from src.cli import fama_french_command


def test_construct_synthetic_factor_proxies() -> None:
    """Verify factor proxy generation structure, columns, and properties."""
    n = 100
    rm = np.random.normal(0.0005, 0.012, n)
    df_factors = construct_synthetic_factor_proxies(rm, seed=42)

    assert isinstance(df_factors, pd.DataFrame)
    assert len(df_factors) == n
    assert list(df_factors.columns) == ["MKT", "SMB", "HML", "UMD"]
    assert np.allclose(df_factors["MKT"].values, rm)
    # Check that SMB, HML, UMD are reasonably centered around zero
    assert abs(float(np.mean(df_factors["SMB"]))) < 1e-4
    assert abs(float(np.mean(df_factors["HML"]))) < 1e-4
    assert abs(float(np.mean(df_factors["UMD"]))) < 1e-4


def test_multivariate_ols_parameter_recovery() -> None:
    """Verify that OLS accurately recovers known factor betas and alpha."""
    np.random.seed(123)
    n = 200
    rm = np.random.normal(0.0003, 0.01, n)
    factors = construct_synthetic_factor_proxies(rm, seed=123)

    # Known parameters
    true_alpha = 0.0004  # ~10% annual alpha
    true_b_mkt = 1.25
    true_b_smb = 0.60
    true_b_hml = -0.40
    true_b_umd = 0.35

    noise = np.random.normal(0.0, 0.001, n)
    y = (
        true_alpha
        + true_b_mkt * rm
        + true_b_smb * factors["SMB"].values
        + true_b_hml * factors["HML"].values
        + true_b_umd * factors["UMD"].values
        + noise
    )

    (
        alpha_ann,
        alpha_bps,
        alpha_t,
        alpha_p,
        loadings,
        r2,
        r2_adj,
        total_vol,
        sys_vol,
        idio_vol,
    ) = evaluate_fama_french_carhart_attribution(
        asset_returns=y,
        market_returns=rm,
        factor_df=factors,
        risk_free_rate_ann=0.0,  # 0 for pure parameter recovery
    )

    # Betas recovered within 0.05
    assert loadings["Market (MKT-RF)"].beta == pytest.approx(true_b_mkt, abs=0.05)
    assert loadings["Size (SMB)"].beta == pytest.approx(true_b_smb, abs=0.05)
    assert loadings["Value (HML)"].beta == pytest.approx(true_b_hml, abs=0.05)
    assert loadings["Momentum (UMD)"].beta == pytest.approx(true_b_umd, abs=0.05)
    assert r2 > 0.90
    assert r2_adj > 0.90
    assert alpha_ann > 5.0  # Positive alpha detected
    assert total_vol > idio_vol


def test_white_hc0_robust_standard_errors() -> None:
    """Verify White (1980) heteroskedasticity-consistent standard errors under heteroskedastic noise."""
    np.random.seed(456)
    n = 150
    rm = np.random.normal(0.0, 0.015, n)
    # Heteroskedastic noise: error variance scales with market squared return
    hetero_noise = np.random.normal(0.0, 0.005 + 0.1 * np.abs(rm))
    ri = 1.1 * rm + hetero_noise

    (
        alpha_ann,
        _,
        alpha_t,
        alpha_p,
        loadings,
        r2,
        _,
        _,
        _,
        _,
    ) = evaluate_fama_french_carhart_attribution(
        asset_returns=ri,
        market_returns=rm,
    )

    for name, loading in loadings.items():
        assert loading.std_error > 0.0
        assert np.isfinite(loading.t_stat)
        assert 0.0 <= loading.p_value <= 1.0


def test_evaluate_fama_french_profile_synthetic_prices() -> None:
    """Test full profile generation workflow from price series."""
    np.random.seed(789)
    n = 80
    dates = pd.date_range("2026-01-01", periods=n, freq="B")
    ret_asset = np.random.normal(0.0008, 0.018, n)
    ret_bench = np.random.normal(0.0004, 0.010, n)

    prices_asset = pd.Series(150.0 * np.cumprod(1.0 + ret_asset), index=dates)
    prices_bench = pd.Series(450.0 * np.cumprod(1.0 + ret_bench), index=dates)

    receipt = evaluate_fama_french_profile(
        ticker="AAPL",
        asset_prices=prices_asset,
        benchmark_prices=prices_bench,
        benchmark_ticker="SPY",
        lookback_window=50,
    )

    assert isinstance(receipt, FamaFrenchReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.lookback_window == 50
    assert receipt.verify_integrity() is True
    assert len(receipt.factor_loadings) == 5
    assert 0.0 <= receipt.r_squared <= 1.0
    assert 0.0 <= receipt.idiosyncratic_variance_ratio_pct <= 100.0


def test_receipt_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature integrity and tamper detection."""
    receipt = FamaFrenchReceipt(
        ticker="NVDA",
        sample_sessions=100,
        lookback_window=60,
        annualized_alpha_pct=12.5,
        alpha_daily_bps=4.96,
        alpha_t_stat=2.35,
        alpha_p_value=0.019,
        is_alpha_significant=True,
        market_beta=1.45,
        size_beta=0.30,
        value_beta=-0.50,
        momentum_beta=0.65,
        r_squared=0.68,
        r_squared_adjusted=0.65,
        total_annualized_vol_pct=38.0,
        systematic_annualized_vol_pct=31.2,
        idiosyncratic_annualized_vol_pct=21.6,
        idiosyncratic_variance_ratio_pct=32.3,
        factor_loadings=[],
        summary="Test summary",
    )

    assert receipt.verify_integrity() is True

    # Tamper with market beta
    receipt.market_beta = 0.50
    assert receipt.verify_integrity() is False


def test_security_input_validation() -> None:
    """Verify ticker allowlist checks and window bounds."""
    prices = pd.Series([100.0 + i for i in range(50)])

    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_fama_french_profile("AAPL; DROP TABLE", prices)

    with pytest.raises(ValueError, match="Lookback window must be at least 20"):
        evaluate_fama_french_profile("AAPL", prices, lookback_window=10)


def test_cli_fama_french_command(capsys: pytest.CaptureFixture) -> None:
    """Verify CLI fama-french command in text and JSON output modes."""
    n = 80
    dates = [f"2026-01-{i+1:02d}" for i in range(n)]
    prices = [150.0 + (i * 0.4) for i in range(n)]
    df_obs = pd.DataFrame({"session_date": dates, "adj_close": prices, "close": prices})

    with patch("src.data.db.get_observations", return_value=df_obs):
        # 1. Text mode
        args = argparse.Namespace(ticker="MSFT", benchmark="SPY", lookback=40, json=False)
        fama_french_command(args)
        out = capsys.readouterr().out
        assert "Fama-French & Carhart (1997) Multi-Factor Attribution Engine" in out
        assert "Security / Ticker:            MSFT" in out
        assert "Market (MKT-RF)" in out
        assert "Size (SMB)" in out

        # 2. JSON mode
        args_json = argparse.Namespace(ticker="MSFT", benchmark="SPY", lookback=40, json=True)
        fama_french_command(args_json)
        out_json = capsys.readouterr().out
        parsed = json.loads(out_json)
        assert parsed["ticker"] == "MSFT"
        assert "annualized_alpha_pct" in parsed
        assert "market_beta" in parsed
        assert "signature" in parsed
