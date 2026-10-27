"""Unit tests for quantitative tail-risk analytics (VaR, CVaR) and daemon tamper alerting."""

from pathlib import Path
import sqlite3
import numpy as np
import pandas as pd
import pytest

from src.analytics.risk import (
    calculate_cvar_historical,
    calculate_var_cornish_fisher,
    calculate_var_gaussian,
    calculate_var_historical,
    compute_multi_horizon_risk_profile,
)
from src.data.db import init_db, save_observations, upsert_stock
from src.evaluation.daemon import EvaluationDaemon
from src.evaluation.tracker import PredictionTracker


@pytest.fixture
def return_sample() -> np.ndarray:
    """Generate 100 historical returns with slight negative skew and fat tails."""
    np.random.seed(42)
    # Mixture: normal + extreme negative jumps
    normal_rets = np.random.normal(0.0005, 0.015, size=95)
    jumps = np.random.uniform(-0.08, -0.04, size=5)
    return np.concatenate([normal_rets, jumps])


def test_var_gaussian_basic_and_monotonicity(return_sample: np.ndarray):
    """Verify parametric Gaussian VaR properties."""
    var_95 = calculate_var_gaussian(return_sample, confidence_level=0.95)
    var_99 = calculate_var_gaussian(return_sample, confidence_level=0.99)

    assert var_95 > 0.0
    assert var_99 > var_95
    # Small / empty sequence guards
    assert calculate_var_gaussian([]) == 0.0
    assert calculate_var_gaussian([0.01, 0.02]) == 0.0
    assert calculate_var_gaussian([0.05, 0.05, 0.05, 0.05, 0.05]) == 0.0


def test_var_cornish_fisher_skewness_adjustment(return_sample: np.ndarray):
    """Verify Cornish-Fisher VaR reflects fat-tailed negative jump risks."""
    var_95_g = calculate_var_gaussian(return_sample, confidence_level=0.95)
    var_95_cf = calculate_var_cornish_fisher(return_sample, confidence_level=0.95)
    var_99_cf = calculate_var_cornish_fisher(return_sample, confidence_level=0.99)

    # In negatively skewed fat-tailed distribution, Cornish-Fisher VaR is wider than Gaussian
    assert var_95_cf >= var_95_g
    assert var_99_cf > var_95_cf


def test_cvar_historical_exceeds_var(return_sample: np.ndarray):
    """Verify Expected Shortfall (CVaR) is greater than or equal to Historical VaR."""
    var_95 = calculate_var_historical(return_sample, confidence_level=0.95)
    cvar_95 = calculate_cvar_historical(return_sample, confidence_level=0.95)

    assert var_95 > 0.0
    assert cvar_95 >= var_95


def test_compute_multi_horizon_risk_profile():
    """Verify compute_multi_horizon_risk_profile creates 1, 5, 21 session metrics."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    prices = [100.0]
    for _ in range(59):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0005, 0.015)))

    df = pd.DataFrame(
        {
            "session_date": dates,
            "adj_close": prices,
        }
    )

    profile = compute_multi_horizon_risk_profile(df, horizons=[1, 5, 21])
    assert "1_SESSION" in profile
    assert "5_SESSIONS" in profile
    assert "21_SESSIONS" in profile

    h1 = profile["1_SESSION"]
    assert "var_95_gaussian_pct" in h1
    assert "var_95_cornish_fisher_pct" in h1
    assert "cvar_95_historical_pct" in h1
    assert h1["var_95_cornish_fisher_pct"] > 0.0


def test_daemon_tamper_detection_in_receipt(tmp_path: Path):
    """Verify EvaluationDaemon.run_once logs and flags tamper detection in execution receipt."""
    db_file = tmp_path / "daemon_test.db"
    init_db(db_path=db_file)
    upsert_stock("JPM", "XNYS", "JPMorgan Chase & Co.", "USD", "Financials", db_path=db_file)

    obs_df = pd.DataFrame([{
        "ticker": "JPM",
        "exchange": "XNYS",
        "session_date": "2026-10-20",
        "open": 200.0,
        "high": 205.0,
        "low": 199.0,
        "close": 204.0,
        "adj_close": 204.0,
        "volume": 1000000,
        "data_status": "FINAL",
        "provider": "Mock",
    }])
    save_observations(obs_df, db_path=db_file)

    tracker = PredictionTracker(db_path=db_file)
    tracker.record_prediction(
        ticker="JPM",
        exchange="XNYS",
        forecast_origin_session="2026-10-20",
        origin_price=204.0,
        horizon_sessions=1,
        target_session_date="2026-10-21",
        predicted_price=206.0,
        model_name="Linear_Direct_1",
        model_version="1.0.0",
    )

    daemon = EvaluationDaemon(db_path=db_file)
    receipt_intact = daemon.run_once()
    assert receipt_intact["integrity_report"]["is_intact"] is True

    # Tamper with the database
    conn = sqlite3.connect(db_file)
    conn.execute("UPDATE predictions SET predicted_price = 9999.0 WHERE ticker = 'JPM'")
    conn.commit()
    conn.close()

    receipt_tampered = daemon.run_once()
    assert receipt_tampered["integrity_report"]["is_intact"] is False
    assert receipt_tampered["integrity_report"]["predictions_valid"] == 0
