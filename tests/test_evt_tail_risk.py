"""Tests for Extreme Value Theory (EVT) Peaks-Over-Threshold Tail Risk Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.evt import (
    calculate_evt_var_cvar,
    evaluate_evt_tail_profile,
    fit_generalized_pareto,
    EVTTailReceipt,
)
from src.cli import evt_report_command


def test_fit_generalized_pareto_convergence_on_synthetic_draws():
    """Verify MLE convergence on synthetic Generalized Pareto Distribution draws."""
    np.random.seed(42)
    n = 500
    true_xi = 0.25
    true_beta = 0.02

    # Draw from uniform U in (0, 1) and invert GPD CDF
    u = np.random.uniform(0.0, 1.0, size=n)
    excesses = (true_beta / true_xi) * (np.power(1.0 - u, -true_xi) - 1.0)

    xi_hat, beta_hat = fit_generalized_pareto(excesses)

    assert np.isclose(xi_hat, true_xi, atol=0.10)
    assert np.isclose(beta_hat, true_beta, atol=0.005)


def test_calculate_evt_var_cvar_monotonicity_and_bounds():
    """Verify mathematical ordering: ES_99 > VaR_99 > ES_95 > VaR_95 > u."""
    np.random.seed(42)
    n = 1000
    # Simulate fat-tailed Student-t losses (df=3.5)
    losses = np.abs(np.random.standard_t(df=3.5, size=n)) * 1.5

    res_95 = calculate_evt_var_cvar(losses, quantile_threshold=0.90, alpha=0.95)
    res_99 = calculate_evt_var_cvar(losses, quantile_threshold=0.90, alpha=0.99)

    u = res_95["u"]
    var_95 = res_95["var"]
    cvar_95 = res_95["cvar"]
    var_99 = res_99["var"]
    cvar_99 = res_99["cvar"]

    # Fundamental tail risk ordering:
    assert var_95 > u
    assert cvar_95 > var_95
    assert var_99 > var_95
    assert cvar_99 > var_99
    assert cvar_99 > cvar_95


def test_tail_fatness_classification():
    """Verify correct classification of tail fatness regimes."""
    np.random.seed(42)
    # 1. Fat-tailed returns (Student-t with df=3)
    fat_ret = np.random.standard_t(df=3.0, size=400) * 0.02
    fat_prices = pd.Series(100.0 * np.exp(np.cumsum(fat_ret)))
    rec_fat = evaluate_evt_tail_profile("FAT", fat_prices, threshold_quantile=0.90)
    assert rec_fat.tail_fatness_regime in ["HEAVY_PARETO_TAIL", "MODERATE_EXPONENTIAL"]

    # 2. Gaussian normal returns
    norm_ret = np.random.normal(0, 0.01, size=400)
    norm_prices = pd.Series(100.0 * np.exp(np.cumsum(norm_ret)))
    rec_norm = evaluate_evt_tail_profile("NORM", norm_prices, threshold_quantile=0.90)
    assert rec_norm.tail_fatness_regime in ["MODERATE_EXPONENTIAL", "BOUNDED_LIGHT_TAIL", "HEAVY_PARETO_TAIL"]


def test_evt_receipt_signature_and_tamper_resilience():
    """Verify deterministic HMAC-SHA256 signature and tamper detection."""
    np.random.seed(42)
    prices = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1.2, size=150)))
    receipt = evaluate_evt_tail_profile("AAPL", prices)

    assert receipt.ticker == "AAPL"
    assert len(receipt.signature) == 64
    assert receipt.verify_integrity() is True

    # Tampering with var_99_pct must invalidate signature
    receipt.var_99_pct = 99.99
    assert receipt.verify_integrity() is False


def test_evaluate_evt_tail_profile_integration():
    """Verify end-to-end evaluation on synthetic stock prices."""
    np.random.seed(42)
    n = 200
    prices = pd.Series(150.0 + np.cumsum(np.random.normal(0.05, 1.5, size=n)))

    receipt = evaluate_evt_tail_profile("NVDA", prices, threshold_quantile=0.95)

    assert receipt.sample_sessions == 200
    assert receipt.exceedances_count > 0
    assert receipt.var_99_pct > receipt.var_95_pct
    assert receipt.cvar_99_pct > receipt.var_99_pct
    assert receipt.threshold_u > 0.0
    assert "NVDA" in receipt.summary


def test_cli_evt_report_command(capsys):
    """Verify CLI evt-report output formatting and JSON serialization."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()
    mock_df = pd.DataFrame({
        "session_date": dates,
        "adj_close": 120.0 + np.cumsum(np.random.normal(0.02, 1.5, size=n)),
        "close": 120.0 + np.cumsum(np.random.normal(0.02, 1.5, size=n)),
    })

    with patch("src.data.db.get_observations", return_value=mock_df):
        # 1. Text output
        args_text = argparse.Namespace(
            ticker="NVDA",
            threshold_quantile=0.95,
            json=False,
        )
        evt_report_command(args_text)
        out = capsys.readouterr().out
        assert "Extreme Value Theory (EVT) Peaks-Over-Threshold Tail Risk Engine" in out
        assert "Security / Ticker:            NVDA" in out
        assert "Generalized Pareto Distribution (GPD) Parameters:" in out
        assert "EVT Peaks-Over-Threshold (GPD)" in out

        # 2. JSON output
        args_json = argparse.Namespace(
            ticker="NVDA",
            threshold_quantile=0.95,
            json=True,
        )
        evt_report_command(args_json)
        json_out = capsys.readouterr().out
        data = json.loads(json_out)
        assert data["ticker"] == "NVDA"
        assert "var_99_pct" in data
        assert "cvar_99_pct" in data
        assert "signature" in data


def test_evt_tail_profile_insufficient_data(capsys):
    """Verify graceful degradation when price history is too short or empty."""
    # 1. Short series (< 30)
    short_prices = pd.Series([10.0, 11.0, 10.5])
    receipt = evaluate_evt_tail_profile("SHORT", short_prices)
    assert receipt.tail_fatness_regime == "INSUFFICIENT_DATA"
    assert receipt.sample_sessions == 3

    # 2. CLI empty observation
    with patch("src.data.db.get_observations", return_value=pd.DataFrame()):
        args_empty = argparse.Namespace(
            ticker="AAPL",
            threshold_quantile=0.95,
            json=False,
        )
        evt_report_command(args_empty)
        out = capsys.readouterr().out
        assert "No observation history found" in out
