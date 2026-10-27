"""Tests for NextOmni Stock multi-horizon forecasting engine (1, 5, 21 sessions)."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.multi_horizon import MultiHorizonForecaster


@pytest.fixture
def multi_horizon_timeseries_df() -> pd.DataFrame:
    """Generate 70 sessions of synthetic data for multi-horizon testing."""
    dates = pd.date_range("2024-01-01", periods=70, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=70)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    volumes = [10000 + int(abs(r) * 100000) for r in returns]

    return pd.DataFrame(
        {
            "ticker": ["TEST"] * 70,
            "exchange": ["XNYS"] * 70,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "data_status": ["END_OF_DAY"] * 70,
            "provider": ["Mock"] * 70,
        }
    )


def test_multi_horizon_fit_and_predict(multi_horizon_timeseries_df: pd.DataFrame):
    """Verify multi-horizon forecaster trains dedicated direct models for 1, 5, and 21 sessions."""
    forecaster = MultiHorizonForecaster(alpha=1.0)
    assert forecaster.is_fitted is False

    forecaster.fit(multi_horizon_timeseries_df)
    assert forecaster.is_fitted is True

    predictions = forecaster.predict_all_horizons(
        multi_horizon_timeseries_df, ticker="TEST", exchange="XNYS"
    )

    assert "1_SESSION" in predictions
    assert "5_SESSIONS" in predictions
    assert "21_SESSIONS" in predictions

    for h_key in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        pred = predictions[h_key]
        assert pred["ticker"] == "TEST"
        assert pred["exchange"] == "XNYS"
        assert pred["predicted_price"] > 0
        assert pred["target_session_date"] > pred["forecast_origin_session"]

    # Target date ordering: 1 session < 5 sessions < 21 sessions
    d1 = predictions["1_SESSION"]["target_session_date"]
    d5 = predictions["5_SESSIONS"]["target_session_date"]
    d21 = predictions["21_SESSIONS"]["target_session_date"]
    assert d1 < d5 < d21


def test_multi_horizon_save_load(tmp_path: Path, multi_horizon_timeseries_df: pd.DataFrame):
    """Verify serialization and deserialization of all 3 horizon model artifacts."""
    forecaster = MultiHorizonForecaster(alpha=0.8)
    forecaster.fit(multi_horizon_timeseries_df)

    save_dir = forecaster.save_models(tmp_path, ticker="TEST")
    assert (save_dir / "1_SESSION.joblib").exists()
    assert (save_dir / "5_SESSIONS.joblib").exists()
    assert (save_dir / "21_SESSIONS.joblib").exists()

    loaded = MultiHorizonForecaster.load_models(tmp_path, ticker="TEST")
    assert loaded.is_fitted is True

    orig_preds = forecaster.predict_all_horizons(multi_horizon_timeseries_df, "TEST", "XNYS")
    loaded_preds = loaded.predict_all_horizons(multi_horizon_timeseries_df, "TEST", "XNYS")

    for h in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        assert orig_preds[h]["predicted_price"] == loaded_preds[h]["predicted_price"]


def test_multi_horizon_walk_forward(multi_horizon_timeseries_df: pd.DataFrame):
    """Verify walk-forward validation executes across all 3 horizons."""
    forecaster = MultiHorizonForecaster()
    # Min train 25 sessions
    res = forecaster.evaluate_walk_forward_all(multi_horizon_timeseries_df, min_train_sessions=25)

    assert "1_SESSION" in res
    assert "5_SESSIONS" in res
    assert "21_SESSIONS" in res

    for h in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        assert "model_metrics" in res[h]
        assert "baseline_metrics" in res[h]
        assert res[h]["walk_forward_steps"] > 0


def test_multi_horizon_gradient_boosting_fit_predict_save_load(
    tmp_path: Path, multi_horizon_timeseries_df: pd.DataFrame
):
    """Verify MultiHorizonForecaster with model_family='gradient_boosting'."""
    forecaster = MultiHorizonForecaster(
        model_family="gradient_boosting", max_iter=20, min_samples_leaf=5
    )
    assert forecaster.model_family == "gradient_boosting"
    assert forecaster.is_fitted is False

    forecaster.fit(multi_horizon_timeseries_df)
    assert forecaster.is_fitted is True

    predictions = forecaster.predict_all_horizons(
        multi_horizon_timeseries_df, ticker="TEST", exchange="XNYS"
    )
    for h_key in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        pred = predictions[h_key]
        assert "HistGradientBoosting" in pred["model_name"]
        assert pred["predicted_price"] > 0

    # Save and reload
    save_dir = forecaster.save_models(tmp_path, ticker="TEST")
    assert (save_dir / "metadata.joblib").exists()

    loaded = MultiHorizonForecaster.load_models(tmp_path, ticker="TEST")
    assert loaded.model_family == "gradient_boosting"
    assert loaded.is_fitted is True

    loaded_preds = loaded.predict_all_horizons(
        multi_horizon_timeseries_df, ticker="TEST", exchange="XNYS"
    )
    for h in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        assert predictions[h]["predicted_price"] == loaded_preds[h]["predicted_price"]


def test_multi_horizon_invalid_model_family():
    """Verify SecurityValidationError when invalid model family requested."""
    from src.security.validators import SecurityValidationError

    with pytest.raises(SecurityValidationError):
        MultiHorizonForecaster(model_family="unsupported_deep_net")

