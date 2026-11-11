"""Unit & CLI integration tests for Kim & Verrecchia (1991, 1994) Announcement Liquidity Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.kim_verrecchia_liquidity import (
    calculate_kim_verrecchia_liquidity,
    evaluate_kim_verrecchia_profile,
    persist_kim_verrecchia_record,
    verify_hmac_signature,
)


def test_homogeneous_precision_produces_zero_differential_volume():
    """KV (1991) Proposition: identical private precisions (sigma_p = 0) yield zero excess volume."""
    rec = calculate_kim_verrecchia_liquidity(
        precision_dispersion_sigma_p=0.0,
        announcement_price_change_abs=0.08,
        ticker="HOMO",
    )

    assert rec.expected_announcement_volume == pytest.approx(0.0, abs=1e-12)
    assert rec.volume_to_price_change_ratio == pytest.approx(0.0, abs=1e-12)
    assert rec.regime == "HOMOGENEOUS_PRECISION_NO_EXCESS_VOLUME"


def test_announcement_volume_scales_linearly_with_abs_price_change():
    """KV (1991) Prediction 1: expected volume rises linearly in |Delta P| with invariant slope."""
    small_dp = calculate_kim_verrecchia_liquidity(announcement_price_change_abs=0.02, ticker="DP2")
    large_dp = calculate_kim_verrecchia_liquidity(announcement_price_change_abs=0.06, ticker="DP6")

    assert large_dp.expected_announcement_volume == pytest.approx(3.0 * small_dp.expected_announcement_volume, rel=1e-9)
    assert large_dp.volume_to_price_change_ratio == pytest.approx(small_dp.volume_to_price_change_ratio, rel=1e-9)


def test_zero_price_change_safe_without_division_by_zero():
    """Verify exact behavior when announcement surprise |Delta P| == 0."""
    rec = calculate_kim_verrecchia_liquidity(announcement_price_change_abs=0.0, ticker="ZERO")

    assert rec.expected_announcement_volume == 0.0
    assert rec.volume_to_price_change_ratio >= 0.0
    assert rec.announcement_illiquidity_lambda >= rec.baseline_illiquidity_lambda


def test_differential_precision_dispersion_drives_both_volume_and_illiquidity():
    """KV (1991, 1994): greater private precision heterogeneity raises both volume and illiquidity."""
    low_disp = calculate_kim_verrecchia_liquidity(precision_dispersion_sigma_p=0.5, ticker="LOWD")
    high_disp = calculate_kim_verrecchia_liquidity(precision_dispersion_sigma_p=4.0, ticker="HIGHD")

    assert high_disp.expected_announcement_volume > low_disp.expected_announcement_volume
    assert high_disp.volume_to_price_change_ratio > low_disp.volume_to_price_change_ratio
    assert high_disp.announcement_illiquidity_lambda > low_disp.announcement_illiquidity_lambda
    assert high_disp.illiquidity_surge_bps > low_disp.illiquidity_surge_bps


def test_public_announcement_triggers_illiquidity_surge_over_baseline():
    """KV (1994): public signal precision s > 0 widens disclosure illiquidity above pre-event baseline."""
    no_event = calculate_kim_verrecchia_liquidity(public_signal_precision_s=0.0, ticker="NONE")
    event = calculate_kim_verrecchia_liquidity(public_signal_precision_s=2.5, ticker="EVNT")

    assert no_event.illiquidity_surge_bps == pytest.approx(0.0, abs=1e-12)
    assert event.announcement_illiquidity_lambda > event.baseline_illiquidity_lambda
    assert event.illiquidity_surge_bps > 0.0
    assert 0.0 < event.price_response_alpha < 1.0


def test_noise_liquidity_reduces_announcement_illiquidity():
    """Higher uninformed noise liquidity depth sigma_u strictly dampens price impact lambda_KV."""
    thin_book = calculate_kim_verrecchia_liquidity(noise_liquidity_sigma_u=0.25, ticker="THIN")
    deep_book = calculate_kim_verrecchia_liquidity(noise_liquidity_sigma_u=2.00, ticker="DEEP")

    assert deep_book.announcement_illiquidity_lambda < thin_book.announcement_illiquidity_lambda


def test_bayesian_price_response_bounds_and_prior_attenuation():
    """Higher prior precision h attenuates Bayesian public signal weight alpha_KV."""
    uncertain_prior = calculate_kim_verrecchia_liquidity(prior_precision_h=1.0, public_signal_precision_s=3.0)
    strong_prior = calculate_kim_verrecchia_liquidity(prior_precision_h=10.0, public_signal_precision_s=3.0)

    assert strong_prior.price_response_alpha < uncertain_prior.price_response_alpha


def test_regime_classification_boundaries():
    """Verify concentrated private processor and high-surprise belief dispersion regimes."""
    conc = calculate_kim_verrecchia_liquidity(
        prior_precision_h=2.0,
        public_signal_precision_s=2.5,
        mean_private_precision_p=3.0,
        precision_dispersion_sigma_p=3.0,
        noise_liquidity_sigma_u=0.40,
    )
    assert conc.regime == "CONCENTRATED_PRIVATE_PROCESSOR_ILLIQUIDITY"

    surge = calculate_kim_verrecchia_liquidity(
        prior_precision_h=3.0,
        public_signal_precision_s=1.0,
        mean_private_precision_p=1.0,
        precision_dispersion_sigma_p=1.2,
        announcement_price_change_abs=0.07,
        noise_liquidity_sigma_u=1.50,
    )
    assert surge.regime == "HIGH_SURPRISE_BELIEF_DISPERSION_SURGE"


def test_security_input_validation():
    """Verify strict ticker allowlist and positive precision/tolerance validation."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_kim_verrecchia_liquidity(ticker="BAD$$TICKER")

    with pytest.raises(ValueError, match="prior_precision_h must be strictly positive"):
        calculate_kim_verrecchia_liquidity(prior_precision_h=0.0)

    with pytest.raises(ValueError, match="must be non-negative"):
        calculate_kim_verrecchia_liquidity(precision_dispersion_sigma_p=-0.5)

    with pytest.raises(ValueError, match="must be strictly positive"):
        calculate_kim_verrecchia_liquidity(noise_liquidity_sigma_u=-1.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_kim_verrecchia_liquidity(ticker="AAPL")
    sig_payload = (
        f"{rec.ticker}:{rec.prior_precision_h:.4f}:{rec.public_signal_precision_s:.4f}:"
        f"{rec.mean_private_precision_p:.4f}:{rec.precision_dispersion_sigma_p:.4f}:"
        f"{rec.announcement_price_change_abs:.6f}:{rec.expected_announcement_volume:.6f}:"
        f"{rec.announcement_illiquidity_lambda:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_kim_verrecchia_profile_from_prices():
    """Verify empirical calibration from trailing price series."""
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.5) for i in range(40)]
    rec = evaluate_kim_verrecchia_profile(prices=prices, ticker="MSFT")

    assert rec.ticker == "MSFT"
    assert rec.expected_announcement_volume > 0.0
    assert rec.announcement_illiquidity_lambda > rec.baseline_illiquidity_lambda


def test_persist_kim_verrecchia_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_kim_verrecchia.db")
    rec = calculate_kim_verrecchia_liquidity(ticker="NVDA")

    assert persist_kim_verrecchia_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, expected_announcement_volume, announcement_illiquidity_lambda, regime FROM kim_verrecchia_audits WHERE ticker = ?",
            ("NVDA",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "NVDA"
        assert row[1] == pytest.approx(rec.expected_announcement_volume, rel=1e-4)
        assert row[2] == pytest.approx(rec.announcement_illiquidity_lambda, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_announcement_liquidity_table_and_json_execution():
    """Verify CLI execution for announcement-liquidity in both table and --json modes."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "announcement-liquidity",
        "--ticker",
        "KV91",
        "--dispersion",
        "2.0",
        "--abs-dp",
        "0.05",
    ]
    res_table = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Kim & Verrecchia (1991, 1994) Differential Precision & Announcement Liquidity Engine" in res_table.stdout
    assert "KV91" in res_table.stdout
    assert "Illiquidity Surge" in res_table.stdout

    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "announcement-liquidity",
        "--ticker",
        "TSLA",
        "--dispersion",
        "2.5",
        "--abs-dp",
        "0.08",
        "--json",
    ]
    res_json = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res_json.stdout)
    assert parsed["ticker"] == "TSLA"
    assert parsed["precision_dispersion_sigma_p"] == pytest.approx(2.5)
    assert parsed["expected_announcement_volume"] > 0.0
    assert "hmac_signature" in parsed

