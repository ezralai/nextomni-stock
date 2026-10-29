"""Unit and integration test suite for Barndorff-Nielsen & Shephard Bipower Jump Engine."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analytics.jump_diffusion import (
    JumpDiffusionReceipt,
    compute_bipower_variation,
    compute_bns_jump_test_statistic,
    compute_realized_variance,
    compute_tripower_quarticity,
    evaluate_jump_diffusion_profile,
)
from src.cli import jump_test_command


@pytest.fixture
def pure_brownian_prices() -> pd.Series:
    """Generate 100 sessions of pure Brownian motion (no discrete jumps)."""
    np.random.seed(42)
    n = 100
    daily_returns = np.random.normal(0.0005, 0.01, size=n)
    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    return pd.Series(prices)


@pytest.fixture
def jump_diffusion_prices() -> pd.Series:
    """Generate 100 sessions of price path with 2 large discrete jumps (-12% and +9%)."""
    np.random.seed(42)
    n = 100
    daily_returns = np.random.normal(0.0002, 0.008, size=n)
    # Inject discrete jumps in the last 21 sessions
    daily_returns[-10] = -0.12  # -12% sudden gap down
    daily_returns[-5] = +0.09   # +9% sudden gap up
    prices = 100.0 * np.exp(np.cumsum(daily_returns))
    return pd.Series(prices)


def test_bipower_variation_continuous_path_properties(pure_brownian_prices):
    """Verify that on pure Brownian motion, BV closely tracks RV and jump ratio is near zero."""
    prices = pure_brownian_prices
    receipt = evaluate_jump_diffusion_profile("AAPL", prices, window=50)

    assert isinstance(receipt, JumpDiffusionReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.window == 50
    assert receipt.jump_contribution_ratio <= 0.15  # Negligible jump variation
    assert receipt.has_significant_jump is False
    assert receipt.jump_regime == "CONTINUOUS_DIFFUSION"
    assert receipt.p_value > 0.05
    assert receipt.verify_integrity() is True

    # Variance additivity check: continuous vol^2 + jump vol^2 approx total vol^2
    cont_var = (receipt.annualized_continuous_vol_pct / 100.0) ** 2
    jump_var = (receipt.annualized_jump_vol_pct / 100.0) ** 2
    total_var = (receipt.annualized_total_vol_pct / 100.0) ** 2
    assert pytest.approx(cont_var + jump_var, rel=1e-3) == total_var


def test_jump_detection_on_synthetic_jumps(jump_diffusion_prices):
    """Verify that injected large jumps trigger statistically significant BNS Z-score."""
    prices = jump_diffusion_prices
    receipt = evaluate_jump_diffusion_profile("NVDA", prices, window=21)

    assert receipt.ticker == "NVDA"
    assert receipt.jump_contribution_ratio > 0.30  # Substantial jump contribution
    assert receipt.bns_z_statistic > 2.0  # Significant jump test statistic
    assert receipt.p_value < 0.05
    assert receipt.has_significant_jump is True
    assert receipt.jump_regime in ("MODERATE_JUMP", "SEVERE_DISCONTINUITY")
    assert receipt.annualized_jump_vol_pct > 10.0


def test_tripower_quarticity_and_bns_statistics():
    """Verify mathematical properties of TQ and BNS Z-statistic under null and alternative."""
    np.random.seed(123)
    ret_null = np.random.normal(0.0, 0.01, size=30)
    rv_null = compute_realized_variance(ret_null)
    bv_null = compute_bipower_variation(ret_null)
    tq_null = compute_tripower_quarticity(ret_null)

    assert rv_null > 0.0
    assert bv_null > 0.0
    assert tq_null > 0.0

    z_null, p_null, sig_null = compute_bns_jump_test_statistic(
        rv=rv_null, bv=bv_null, tq=tq_null, sample_size=len(ret_null)
    )
    assert p_null > 0.01  # Should not falsely trigger at 99% level

    # Synthetic jump test
    ret_jump = np.random.normal(0.0, 0.01, size=40)
    ret_jump[10] = 0.15  # Massive +15% jump
    ret_jump[25] = -0.12  # Massive -12% jump
    rv_jump = compute_realized_variance(ret_jump)
    bv_jump = compute_bipower_variation(ret_jump)
    tq_jump = compute_tripower_quarticity(ret_jump)

    z_jump, p_jump, sig_jump = compute_bns_jump_test_statistic(
        rv=rv_jump, bv=bv_jump, tq=tq_jump, sample_size=len(ret_jump)
    )
    assert z_jump > z_null
    assert p_jump < 0.01
    assert sig_jump is True


def test_jump_receipt_cryptographic_tamper_detection(pure_brownian_prices):
    """Verify HMAC-SHA256 signature generation and tamper invalidation."""
    receipt = evaluate_jump_diffusion_profile("MSFT", pure_brownian_prices, window=25)
    assert receipt.verify_integrity() is True

    # Tamper with jump regime
    tampered_dict = receipt.to_dict()
    tampered_dict["jump_regime"] = "SEVERE_DISCONTINUITY"
    tampered_receipt = JumpDiffusionReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with jump ratio
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["jump_contribution_ratio"] = 0.9999
    tampered_receipt2 = JumpDiffusionReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_jump_diffusion_security_and_edge_cases():
    """Verify ticker allowlist validation, insufficient lookback, and short price history."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_jump_diffusion_profile("INVALID$$$", pd.Series([10.0] * 30), window=20)

    # Window too small (< 10)
    with pytest.raises(ValueError, match="Observation window must be at least 10"):
        evaluate_jump_diffusion_profile("AAPL", pd.Series([10.0] * 30), window=5)

    # Insufficient prices for window
    with pytest.raises(ValueError, match="Insufficient price history"):
        evaluate_jump_diffusion_profile("AAPL", pd.Series([10.0, 11.0, 12.0]), window=20)


def test_cli_jump_test_command_stdout_and_json(pure_brownian_prices, monkeypatch):
    """Verify CLI jump-test command execution across stdout and JSON modes."""
    df_obs = pd.DataFrame({"close": pure_brownian_prices})
    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda *args, **kwargs: df_obs,
    )

    # 1. Stdout execution
    args_stdout = argparse.Namespace(
        ticker="AAPL",
        window=25,
        json=False,
    )
    jump_test_command(args_stdout)

    # 2. JSON execution
    args_json = argparse.Namespace(
        ticker="AAPL",
        window=25,
        json=True,
    )
    jump_test_command(args_json)
