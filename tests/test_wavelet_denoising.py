"""Unit and integration tests for Multiscale Wavelet Multiresolution Denoising Engine."""

import argparse
import json
import numpy as np
import pandas as pd
import pytest

from src.cli import wavelet_denoise_command
from src.features.wavelet import (
    WaveletReceipt,
    compute_modwt_decomposition,
    compute_wavelet_energy_scalogram,
    denoise_wavelet_series,
    evaluate_wavelet_profile,
)


def test_modwt_decomposition_additive_reconstruction() -> None:
    """Verify shift-invariant MODWT decomposition satisfies exact additive reconstruction."""
    np.random.seed(42)
    n = 64
    x = np.cumsum(np.random.normal(0.0005, 0.02, size=n)) + 100.0

    levels = 3
    details, approx = compute_modwt_decomposition(x, levels=levels)

    assert len(details) == levels
    assert approx.shape == (n,)

    # Exact additive identity: X_t = sum_j W_{j,t} + V_{J,t}
    reconstructed = approx.copy()
    for j in range(1, levels + 1):
        assert details[j].shape == (n,)
        reconstructed += details[j]

    np.testing.assert_allclose(reconstructed, x, atol=1e-10)

    # Empty and degenerate handling
    d_empty, a_empty = compute_modwt_decomposition([])
    assert d_empty == {}
    assert len(a_empty) == 0


def test_denoise_wavelet_series_noise_suppression_and_jump_preservation() -> None:
    """Verify VisuShrink soft thresholding suppresses high-frequency noise and preserves jumps."""
    np.random.seed(99)
    n = 100
    # True signal: step jump from 50 to 80 at t=50
    true_signal = np.zeros(n)
    true_signal[:50] = 50.0
    true_signal[50:] = 80.0

    # Additive Gaussian white noise
    noise_sigma = 2.0
    noise = np.random.normal(0.0, noise_sigma, size=n)
    noisy_signal = true_signal + noise

    denoised, sigma_hat, thresh = denoise_wavelet_series(noisy_signal, levels=3, soft=True)

    assert len(denoised) == n
    assert 1.0 <= sigma_hat <= 3.5
    assert thresh > 0.0

    # Noise variance on flat plateaus should be significantly suppressed
    flat_segment_raw_var = np.var(noisy_signal[10:40])
    flat_segment_denoised_var = np.var(denoised[10:40])
    assert flat_segment_denoised_var < flat_segment_raw_var

    # Step jump height should be preserved across the transition boundary
    jump_height = float(np.mean(denoised[60:90]) - np.mean(denoised[10:40]))
    assert abs(jump_height - 30.0) < 5.0


def test_wavelet_energy_scalogram_properties() -> None:
    """Verify multiscale energy scalogram isolates frequency octave concentrations."""
    n = 64
    # 1. High-frequency alternating series: +1, -1, +1, -1... (Nyquist frequency)
    hf_signal = np.tile([1.0, -1.0], n // 2)
    details_hf, approx_hf = compute_modwt_decomposition(hf_signal, levels=3)
    shares_hf = compute_wavelet_energy_scalogram(details_hf, approx_hf)

    # Scale 1 (2-4 day octave) should dominate almost all detail energy
    assert shares_hf["Scale_1_2to4d"] > 70.0
    assert np.isclose(sum(shares_hf.values()), 100.0, atol=1e-2)

    # 2. Smooth low-frequency trend: linear ramp
    lf_signal = np.linspace(10.0, 50.0, n)
    details_lf, approx_lf = compute_modwt_decomposition(lf_signal, levels=3)
    shares_lf = compute_wavelet_energy_scalogram(details_lf, approx_lf)
    assert np.isclose(sum(shares_lf.values()), 100.0, atol=1e-2)


def test_wavelet_receipt_signature_and_tamper() -> None:
    """Verify cryptographic HMAC-SHA256 signature and tamper detection."""
    receipt = WaveletReceipt(
        ticker="AAPL",
        sample_sessions=100,
        levels=3,
        noise_sigma_hat=1.234567,
        universal_threshold=3.765432,
        snr_db=25.4,
        energy_shares_pct={"Scale_1_2to4d": 45.0, "Scale_2_4to8d": 35.0, "Scale_3_8to16d": 20.0},
        dominant_scale_regime="BALANCED_MULTI_OCTAVE_SPECTRUM",
        latest_raw_price=175.50,
        latest_denoised_price=175.20,
        price_deviation_pct=0.17,
        summary="Test wavelet receipt",
    )

    assert receipt.signature != ""
    assert receipt.verify_integrity() is True

    # Tampering should break cryptographic verification
    receipt.latest_denoised_price = 199.99
    assert receipt.verify_integrity() is False


def test_evaluate_wavelet_profile_integration() -> None:
    """Verify end-to-end evaluation with pandas observation DataFrame."""
    np.random.seed(88)
    dates = pd.date_range("2025-01-01", periods=50, freq="B").strftime("%Y-%m-%d")
    prices = np.cumsum(np.random.normal(0.1, 1.5, size=50)) + 150.0

    df_obs = pd.DataFrame({
        "session_date": dates,
        "adj_close": prices,
        "close": prices,
    })

    receipt = evaluate_wavelet_profile(df_obs, ticker="MSFT", levels=3)

    assert receipt.ticker == "MSFT"
    assert receipt.sample_sessions == 50
    assert receipt.levels == 3
    assert receipt.noise_sigma_hat > 0.0
    assert receipt.snr_db > 0.0
    assert "Scale_1_2to4d" in receipt.energy_shares_pct
    assert receipt.verify_integrity() is True


def test_cli_wavelet_denoise_command(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """Verify CLI wavelet-denoise command stdout and JSON outputs."""
    fake_receipt = WaveletReceipt(
        ticker="NVDA",
        sample_sessions=80,
        levels=3,
        noise_sigma_hat=2.4500,
        universal_threshold=7.2500,
        snr_db=22.8,
        energy_shares_pct={"Scale_1_2to4d": 30.0, "Scale_2_4to8d": 40.0, "Scale_3_8to16d": 30.0},
        dominant_scale_regime="WEEKLY_CYCLICAL_DOMINATED",
        latest_raw_price=120.50,
        latest_denoised_price=120.10,
        price_deviation_pct=0.33,
        summary="NVDA wavelet analysis: Regime=WEEKLY_CYCLICAL_DOMINATED, SNR=22.8 dB.",
    )

    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda ticker, db_path=None: pd.DataFrame({"session_date": ["2025-01-01"], "adj_close": [100.0]}),
    )
    monkeypatch.setattr(
        "src.features.wavelet.evaluate_wavelet_profile",
        lambda df, ticker, levels: fake_receipt,
    )

    # 1. Standard tabular output
    args_std = argparse.Namespace(ticker="NVDA", levels=3, json=False)
    wavelet_denoise_command(args_std)
    out_std = capsys.readouterr().out
    assert "Multiscale Wavelet Denoising & Energy Scalogram Engine" in out_std
    assert "NVDA" in out_std
    assert "WEEKLY_CYCLICAL_DOMINATED" in out_std
    assert "22.80 dB" in out_std

    # 2. JSON structured output
    args_json = argparse.Namespace(ticker="NVDA", levels=3, json=True)
    wavelet_denoise_command(args_json)
    out_json = capsys.readouterr().out
    data = json.loads(out_json)
    assert data["ticker"] == "NVDA"
    assert data["dominant_scale_regime"] == "WEEKLY_CYCLICAL_DOMINATED"
    assert data["snr_db"] == 22.8
    assert "signature" in data
