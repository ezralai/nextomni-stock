"""Tests for Asymmetric GJR-GARCH(1, 1) Volatility Forecasting Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.garch import (
    evaluate_garch_profile,
    fit_gjr_garch,
    forecast_garch_term_structure,
    GARCHReceipt,
)
from src.cli import garch_volatility_command


def test_fit_gjr_garch_parameter_constraints_and_convergence():
    """Verify QMLE parameter non-negativity and stationarity condition phi < 1.0."""
    np.random.seed(42)
    n = 500
    # Simulate stylized volatility-clustered returns
    sigma = np.zeros(n)
    r = np.zeros(n)
    sigma[0] = 0.015
    for t in range(1, n):
        i_prev = 1.0 if r[t - 1] < 0.0 else 0.0
        # GJR: omega=1e-5, alpha=0.06, gamma=0.08, beta=0.82 -> phi = 0.06 + 0.04 + 0.82 = 0.92
        sigma[t] = np.sqrt(1e-5 + (0.06 + 0.08 * i_prev) * (r[t - 1] ** 2) + 0.82 * (sigma[t - 1] ** 2))
        r[t] = np.random.normal(0, sigma[t])

    res = fit_gjr_garch(r, asymmetric=True)

    assert res["omega"] > 0.0
    assert res["alpha"] >= 0.0
    assert res["gamma"] >= 0.0
    assert res["beta"] >= 0.0
    assert res["persistence_phi"] < 1.0
    assert res["unconditional_variance"] > 0.0


def test_gjr_garch_asymmetric_vs_standard_specification():
    """Verify standard GARCH enforces gamma = 0 exactly."""
    np.random.seed(42)
    r = np.random.normal(0, 0.01, size=300)

    res_std = fit_gjr_garch(r, asymmetric=False)
    assert res_std["gamma"] == 0.0
    assert np.isclose(res_std["persistence_phi"], res_std["alpha"] + res_std["beta"])

    res_gjr = fit_gjr_garch(r, asymmetric=True)
    assert res_gjr["gamma"] >= 0.0


def test_forecast_garch_term_structure_mean_reversion():
    """Verify multi-step forward volatility term structure decays toward unconditional mean."""
    # Elevated initial volatility scenario: final_sigma2 > uncond_var
    uncond_var = 0.015**2
    fit_res = {
        "omega": 1e-5,
        "alpha": 0.05,
        "gamma": 0.06,
        "beta": 0.85,
        "persistence_phi": 0.93,
        "unconditional_variance": uncond_var,
        "final_sigma2": 0.040**2,  # Shocked high initial volatility
        "final_eps": -0.040,
    }

    term_forecasts = forecast_garch_term_structure(fit_res, horizons=[1, 5, 21])

    assert "1_session" in term_forecasts
    assert "5_sessions" in term_forecasts
    assert "21_sessions" in term_forecasts

    # Elevated volatility should mean-revert downward across longer horizons
    assert term_forecasts["1_session"] > term_forecasts["5_sessions"]
    assert term_forecasts["5_sessions"] > term_forecasts["21_sessions"]


def test_garch_receipt_signature_and_tamper_detection():
    """Verify HMAC-SHA256 signature integrity and tamper resilience."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1.2, size=150)))
    receipt = evaluate_garch_profile("JPM", prices, asymmetric=True)

    assert receipt.ticker == "JPM"
    assert len(receipt.signature) == 64
    assert receipt.verify_integrity() is True

    # Tampering with alpha must invalidate signature
    receipt.alpha = 0.999
    assert receipt.verify_integrity() is False


def test_evaluate_garch_profile_integration():
    """Verify end-to-end evaluation on realistic stock prices."""
    np.random.seed(42)
    n = 200
    prices = pd.Series(150.0 + np.cumsum(np.random.normal(0.05, 1.5, size=n)))

    receipt = evaluate_garch_profile("AAPL", prices, asymmetric=True)

    assert receipt.sample_sessions == 200
    assert receipt.model_type == "GJR-GARCH(1,1)"
    assert receipt.persistence_phi < 1.0
    assert receipt.current_conditional_volatility_ann_pct > 0.0
    assert receipt.unconditional_volatility_ann_pct > 0.0
    assert len(receipt.term_structure_forecast_ann_pct) == 3
    assert "AAPL" in receipt.summary


def test_cli_garch_volatility_command(capsys):
    """Verify CLI garch-volatility output formatting and JSON output."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    mock_df = pd.DataFrame({
        "session_date": dates,
        "adj_close": 180.0 + np.cumsum(np.random.normal(0.05, 1.8, size=n)),
        "close": 180.0 + np.cumsum(np.random.normal(0.05, 1.8, size=n)),
    })

    with patch("src.data.db.get_observations", return_value=mock_df):
        # 1. Text output
        args_text = argparse.Namespace(
            ticker="AAPL",
            model="gjr",
            json=False,
        )
        garch_volatility_command(args_text)
        out = capsys.readouterr().out
        assert "Asymmetric GJR-GARCH(1, 1) Volatility Forecasting Engine" in out
        assert "Security / Ticker:            AAPL" in out
        assert "Current Conditional Volatility:" in out
        assert "Forward Term Structure Projections:" in out

        # 2. JSON output
        args_json = argparse.Namespace(
            ticker="AAPL",
            model="standard",
            json=True,
        )
        garch_volatility_command(args_json)
        json_out = capsys.readouterr().out
        data = json.loads(json_out)
        assert data["ticker"] == "AAPL"
        assert data["model_type"] == "Standard GARCH(1,1)"
        assert "persistence_phi" in data
        assert "term_structure_forecast_ann_pct" in data
        assert "signature" in data


def test_garch_profile_insufficient_data(capsys):
    """Verify graceful handling when price series is too short or empty."""
    # 1. Short series (< 40)
    short_prices = pd.Series([100.0, 101.0, 99.5])
    receipt = evaluate_garch_profile("TINY", short_prices)
    assert receipt.sample_sessions == 3
    assert receipt.persistence_phi == 0.0

    # 2. CLI empty observation
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()):
        args_empty = argparse.Namespace(
            ticker="MSFT",
            model="gjr",
            json=False,
        )
        garch_volatility_command(args_empty)
        out = capsys.readouterr().out
        assert "No observation history found" in out
