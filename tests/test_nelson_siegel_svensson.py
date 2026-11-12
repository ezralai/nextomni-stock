"""Tests for Nelson-Siegel-Svensson (1987/1994) Yield Curve Factor Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.nelson_siegel_svensson import (
    _build_hmac_payload,
    build_synthetic_nss_curve,
    evaluate_nss_profile,
    evaluate_nss_yield,
    fit_nelson_siegel_svensson,
    persist_nss_record,
    verify_hmac_signature,
)


def test_nss_exact_parameter_recovery_and_zero_maturity_limit() -> None:
    """Nested OLS + bounded optimization recovers NSS curve within < 0.5 bps RMSE."""
    maturities, yields = build_synthetic_nss_curve(mode="normal")
    receipt = fit_nelson_siegel_svensson(maturities, yields, ticker="UST")
    assert receipt.n_maturities == 10
    assert receipt.rmse_bps < 0.5
    assert receipt.r_squared > 0.999
    assert receipt.spread_10y_2y_bps > 20.0
    assert receipt.regime == "NORMAL_STEEP"
    # Analytical zero-maturity limit matches lim_{m -> 0+} y(m) = beta0 + beta1
    y0 = evaluate_nss_yield(
        np.array([0.0]),
        receipt.beta0_level,
        receipt.beta1_slope,
        receipt.beta2_curvature,
        receipt.beta3_second_hump,
        receipt.tau1,
        receipt.tau2,
    )[0]
    assert y0 == pytest.approx(receipt.short_end_limit, abs=1e-9)


def test_nss_inverted_and_humped_regimes() -> None:
    """Verify INVERTED_BEAR and HUMPED_BULL regime classifications."""
    m_inv, y_inv = build_synthetic_nss_curve(mode="inverted")
    rec_inv = fit_nelson_siegel_svensson(m_inv, y_inv, ticker="UST2S10S")
    assert rec_inv.rmse_bps < 0.5
    assert rec_inv.spread_10y_2y_bps < -15.0
    assert rec_inv.regime == "INVERTED_BEAR"

    m_hump, y_hump = build_synthetic_nss_curve(mode="humped")
    rec_hump = fit_nelson_siegel_svensson(m_hump, y_hump, ticker="UST_HUMP")
    assert rec_hump.rmse_bps < 0.5
    assert rec_hump.regime == "HUMPED_BULL"


def test_nss_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite audit persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        fit_nelson_siegel_svensson([0.5, 1, 2, 5, 10], [0.04, 0.04, 0.04, 0.04, 0.04], ticker="BAD TICKER")

    with pytest.raises(ValueError, match="At least 5"):
        fit_nelson_siegel_svensson([1.0, 2.0, 5.0], [0.04, 0.042, 0.045], ticker="UST")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.001)))
    receipt = evaluate_nss_profile(prices, ticker="SPY")
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        n_maturities=receipt.n_maturities,
        beta0_level=receipt.beta0_level,
        beta1_slope=receipt.beta1_slope,
        beta2_curvature=receipt.beta2_curvature,
        beta3_second_hump=receipt.beta3_second_hump,
        tau1=receipt.tau1,
        tau2=receipt.tau2,
        rmse_bps=receipt.rmse_bps,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "nss_test.db"
    assert persist_nss_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, n_maturities, regime, hmac_signature FROM nss_curve_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 10
    assert rows[0][3] == receipt.hmac_signature


def test_cli_nss_curve_json_and_aliases() -> None:
    """Verify CLI nss-curve and aliases emit valid JSON receipt."""
    for alias in ("nss-curve", "nelson-siegel", "svensson", "nss-1994", "yield-curve-fit"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "UST",
                "--mode",
                "normal",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "UST"
        assert data["rmse_bps"] < 0.5
        assert data["regime"] == "NORMAL_STEEP"
        assert len(data["hmac_signature"]) == 64

