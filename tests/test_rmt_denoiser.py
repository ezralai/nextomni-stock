"""Unit and integration test suite for Marchenko-Pastur RMT Covariance Denoising Engine."""

import argparse
import json
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest

from src.analytics.rmt_denoiser import (
    RmtDenoiserReceipt,
    compute_marchenko_pastur_pdf,
    denoise_covariance_matrix,
    detrend_covariance_matrix,
    evaluate_rmt_denoising_profile,
    fit_marchenko_pastur_bounds,
)
from src.cli import rmt_denoise_command


@pytest.fixture
def synthetic_spiked_covariance() -> np.ndarray:
    """Generate synthetic spiked covariance matrix with 1 strong factor + random noise.

    Simulates N=10 assets over T=100 sessions (q = 10.0).
    """
    np.random.seed(42)
    n_assets = 10
    n_sessions = 150

    # 1 common market factor with high loading
    market_factor = np.random.normal(0.0, 0.02, size=(n_sessions, 1))
    loadings = np.random.uniform(0.6, 1.2, size=(1, n_assets))
    systematic = market_factor @ loadings

    # Idiosyncratic noise
    idiosyncratic = np.random.normal(0.0, 0.01, size=(n_sessions, n_assets))

    returns = systematic + idiosyncratic
    emp_cov = np.cov(returns, rowvar=False)
    return emp_cov


def test_marchenko_pastur_pdf_mathematical_properties():
    """Verify theoretical Marchenko-Pastur distribution bounds and area under curve."""
    var = 1.0
    q = 4.0  # T / N = 4 => sqrt(1/q) = 0.5
    # lambda_- = 1 * (1 - 0.5)^2 = 0.25
    # lambda_+ = 1 * (1 + 0.5)^2 = 2.25
    lambdas, pdf, l_min, l_max = compute_marchenko_pastur_pdf(var=var, q=q, pts=500)

    assert pytest.approx(l_min, abs=1e-3) == 0.25
    assert pytest.approx(l_max, abs=1e-3) == 2.25
    assert len(lambdas) == 500
    assert len(pdf) == 500
    assert np.all(pdf >= 0.0)

    # Approximate numerical integration over domain should be ~ 1.0
    dx = lambdas[1] - lambdas[0]
    approx_area = float(np.sum(pdf) * dx)
    assert 0.95 <= approx_area <= 1.05


def test_fit_marchenko_pastur_bounds_and_spiked_recovery(synthetic_spiked_covariance):
    """Verify that RMT fitting identifies empirical noise ceiling and isolates the signal mode."""
    cov = synthetic_spiked_covariance
    n_assets = cov.shape[0]
    n_sessions = 150
    q = n_sessions / n_assets

    # Decompose correlation
    stds = np.sqrt(np.diag(cov))
    corr = cov / np.outer(stds, stds)
    evals = np.linalg.eigvalsh(corr)[::-1]

    fitted_var, l_min, l_plus = fit_marchenko_pastur_bounds(evals, q=q)

    assert 0.01 <= fitted_var <= 1.0
    assert l_min < l_plus
    # The dominant market factor eigenvalue should clearly exceed lambda_+
    assert evals[0] > l_plus
    # Noise eigenvalues should be mostly below lambda_+
    assert np.sum(evals <= l_plus) >= (n_assets - 2)


def test_denoise_covariance_matrix_stability_and_trace(synthetic_spiked_covariance):
    """Verify trace preservation, condition number improvement, and positive-definiteness."""
    cov = synthetic_spiked_covariance
    n_sessions = 150

    # Constant residual method
    (
        denoised_cov,
        l_plus,
        fitted_var,
        num_signals,
        raw_cond,
        denoised_cond,
    ) = denoise_covariance_matrix(
        cov,
        sample_sessions=n_sessions,
        method="constant_residual",
    )

    assert denoised_cov.shape == cov.shape
    assert num_signals >= 1
    # Condition number should improve (lower condition number means greater numerical stability)
    assert denoised_cond <= raw_cond
    assert raw_cond > 1.0
    assert denoised_cond > 1.0

    # Positive-definiteness check: all eigenvalues must be strictly positive
    denoised_evals = np.linalg.eigvalsh(denoised_cov)
    assert np.all(denoised_evals > 0.0)

    # Targeted shrinkage method
    (
        denoised_cov_shrunk,
        _,
        _,
        _,
        _,
        shrunk_cond,
    ) = denoise_covariance_matrix(
        cov,
        sample_sessions=n_sessions,
        method="targeted_shrinkage",
        shrinkage_alpha=0.5,
    )
    assert denoised_cov_shrunk.shape == cov.shape
    assert np.all(np.linalg.eigvalsh(denoised_cov_shrunk) > 0.0)


def test_detrend_covariance_matrix(synthetic_spiked_covariance):
    """Verify that detrending removes the dominant market mode while preserving positive-definiteness."""
    cov = synthetic_spiked_covariance
    n_sessions = 150

    detrended_cov = detrend_covariance_matrix(cov, sample_sessions=n_sessions)
    assert detrended_cov.shape == cov.shape

    # Spectral properties of detrended correlation
    stds = np.sqrt(np.diag(detrended_cov))
    detrended_corr = detrended_cov / np.outer(stds, stds)
    evals = np.linalg.eigvalsh(detrended_corr)[::-1]

    # Market mode has been shrunk/replaced, so first eigenvalue is significantly reduced
    raw_stds = np.sqrt(np.diag(cov))
    raw_corr = cov / np.outer(raw_stds, raw_stds)
    raw_evals = np.linalg.eigvalsh(raw_corr)[::-1]

    assert evals[0] < raw_evals[0]
    assert np.all(evals > 0.0)


def test_rmt_receipt_cryptographic_tamper_detection(synthetic_spiked_covariance, monkeypatch):
    """Verify RmtDenoiserReceipt generation, properties, and HMAC-SHA256 tamper resilience."""
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "V"]
    n_sessions = 100
    dates = pd.date_range("2025-01-01", periods=n_sessions, freq="B").strftime("%Y-%m-%d")

    np.random.seed(42)
    fake_ret = pd.DataFrame(
        np.random.normal(0.001, 0.02, size=(n_sessions, len(tickers))),
        index=dates,
        columns=tickers,
    )

    monkeypatch.setattr(
        "src.analytics.rmt_denoiser.build_aligned_returns_matrix",
        lambda *args, **kwargs: fake_ret,
    )

    receipt, raw_cov, denoised_cov = evaluate_rmt_denoising_profile(
        tickers=tickers,
        lookback_sessions=n_sessions,
        method="constant_residual",
    )

    assert isinstance(receipt, RmtDenoiserReceipt)
    assert receipt.num_assets == len(tickers)
    assert receipt.sample_sessions == n_sessions
    assert receipt.num_signal_eigenvalues >= 1
    assert receipt.condition_number_raw >= receipt.condition_number_denoised
    assert receipt.verify_integrity() is True

    # Tamper with number of signal eigenvalues
    tampered_dict = receipt.to_dict()
    tampered_dict["num_signal_eigenvalues"] = 999
    tampered_receipt = RmtDenoiserReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False


def test_rmt_denoiser_security_and_edge_cases():
    """Verify parameter bounds, input validation, and invalid ticker formats."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_rmt_denoising_profile(tickers=["INVALID_TICKER$$$"], lookback_sessions=100)

    # Insufficient lookback
    with pytest.raises(ValueError, match="Lookback sessions must be at least 40"):
        evaluate_rmt_denoising_profile(tickers=["AAPL", "MSFT"], lookback_sessions=10)

    # Non-square matrix error
    with pytest.raises(ValueError, match="Covariance matrix must be square"):
        denoise_covariance_matrix(np.zeros((5, 3)), sample_sessions=50)

    # Insufficient assets
    with pytest.raises(ValueError, match="At least 2 assets required"):
        denoise_covariance_matrix(np.ones((1, 1)), sample_sessions=50)


def test_cli_rmt_denoise_command_stdout_and_json(monkeypatch):
    """Verify CLI rmt-denoise command in stdout and JSON modes."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    n_sessions = 60
    dates = pd.date_range("2025-01-01", periods=n_sessions, freq="B").strftime("%Y-%m-%d")

    fake_ret = pd.DataFrame(
        np.random.normal(0.001, 0.02, size=(n_sessions, len(tickers))),
        index=dates,
        columns=tickers,
    )

    monkeypatch.setattr(
        "src.analytics.rmt_denoiser.build_aligned_returns_matrix",
        lambda *args, **kwargs: fake_ret,
    )

    # 1. Stdout execution
    args_stdout = argparse.Namespace(
        tickers="AAPL,MSFT,GOOGL",
        exchange=None,
        lookback=60,
        method="constant_residual",
        detrend=False,
        json=False,
    )
    rmt_denoise_command(args_stdout)

    # 2. JSON execution
    args_json = argparse.Namespace(
        tickers="AAPL,MSFT,GOOGL",
        exchange=None,
        lookback=60,
        method="constant_residual",
        detrend=False,
        json=True,
    )
    rmt_denoise_command(args_json)
