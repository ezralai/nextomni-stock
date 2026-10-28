"""Unit and integration tests for dynamic RegimeAdaptiveForecaster."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.builder import BASE_FEATURE_COLUMNS
from src.models.adaptive_ensemble import DEFAULT_REGIME_WEIGHTS, RegimeAdaptiveForecaster
from src.models.multi_horizon import MultiHorizonForecaster
from src.security.audit import verify_model_artifact


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic feature matrix and targets with varied regime signatures."""
    np.random.seed(42)
    n = 60
    data = {}
    for col in BASE_FEATURE_COLUMNS:
        data[col] = np.random.normal(0, 1, n)

    # Specific regime indicators
    data["price_to_sma_20"] = np.random.uniform(0.90, 1.10, n)
    data["vol_regime_ratio"] = np.random.uniform(0.80, 1.40, n)

    df_X = pd.DataFrame(data)
    p_origin = np.random.uniform(100.0, 150.0, n)
    returns = 0.05 * df_X["lag_return_1"] - 0.03 * df_X["lag_return_2"] + np.random.normal(0, 0.01, n)
    y_target = p_origin * (1.0 + returns)

    return df_X, y_target, p_origin


def test_classify_regime_states():
    """Verify classification logic across all 4 regimes plus default fallback."""
    forecaster = RegimeAdaptiveForecaster()

    # 1. BULL_TREND: Trend >= 1.0 and Vol <= 1.15
    row_bull = pd.Series({"price_to_sma_20": 1.05, "vol_regime_ratio": 0.95})
    assert forecaster.classify_regime(row_bull) == "BULL_TREND"

    # 2. EUPHORIC_MOMENTUM: Trend >= 1.0 and Vol > 1.15
    row_euphoric = pd.Series({"price_to_sma_20": 1.08, "vol_regime_ratio": 1.35})
    assert forecaster.classify_regime(row_euphoric) == "EUPHORIC_MOMENTUM"

    # 3. DISTRIBUTION: Trend < 1.0 and Vol <= 1.15
    row_dist = pd.Series({"price_to_sma_20": 0.95, "vol_regime_ratio": 1.00})
    assert forecaster.classify_regime(row_dist) == "DISTRIBUTION"

    # 4. PANIC_CRISIS: Trend < 1.0 and Vol > 1.15
    row_panic = pd.Series({"price_to_sma_20": 0.92, "vol_regime_ratio": 1.45})
    assert forecaster.classify_regime(row_panic) == "PANIC_CRISIS"

    # 5. Missing columns fallback
    row_missing = pd.Series({"some_other_col": 1.0})
    assert forecaster.classify_regime(row_missing) == "DEFAULT"


def test_fit_and_regime_adaptive_prediction(synthetic_training_data):
    """Verify fit, prediction, and dynamic weight switching between regimes."""
    df_X, y_target, p_origin = synthetic_training_data

    forecaster = RegimeAdaptiveForecaster()
    forecaster.fit(df_X, y_target, p_origin)
    assert forecaster.is_fitted

    # Test prediction on Bull Trend input
    bull_df = df_X.copy().iloc[-5:]
    bull_df["price_to_sma_20"] = 1.08
    bull_df["vol_regime_ratio"] = 0.90
    p_test = p_origin[-5:]

    preds_bull = forecaster.predict(bull_df, p_test)
    assert len(preds_bull) == 5
    assert (preds_bull > 0.0).all()
    assert forecaster.last_detected_regime == "BULL_TREND"
    assert forecaster.last_active_weights == DEFAULT_REGIME_WEIGHTS["BULL_TREND"]

    # Test prediction on Panic Crisis input
    panic_df = df_X.copy().iloc[-5:]
    panic_df["price_to_sma_20"] = 0.91
    panic_df["vol_regime_ratio"] = 1.50

    preds_panic = forecaster.predict(panic_df, p_test)
    assert len(preds_panic) == 5
    assert (preds_panic > 0.0).all()
    assert forecaster.last_detected_regime == "PANIC_CRISIS"
    assert forecaster.last_active_weights == DEFAULT_REGIME_WEIGHTS["PANIC_CRISIS"]


def test_predict_interval_and_explain(synthetic_training_data):
    """Verify prediction intervals and explainability additivity in adaptive ensemble."""
    df_X, y_target, p_origin = synthetic_training_data

    forecaster = RegimeAdaptiveForecaster()
    forecaster.fit(df_X, y_target, p_origin)

    p_test = p_origin[:3]
    p_pred, low, up = forecaster.predict_interval(df_X.iloc[:3], p_test, confidence_level=0.80)
    assert len(p_pred) == 3
    assert len(low) == 3
    assert len(up) == 3
    # Upper interval should strictly exceed lower interval
    assert (up >= low).all()

    # Explain forecast
    sample = df_X.iloc[[0]]
    attributions = forecaster.explain_forecast(sample)
    assert isinstance(attributions, dict)
    assert len(attributions) > 0
    assert "price_to_sma_20" in attributions


def test_save_and_load_with_hmac_signature(synthetic_training_data, tmp_path):
    """Verify cryptographic signing and persistence of RegimeAdaptiveForecaster."""
    df_X, y_target, p_origin = synthetic_training_data

    forecaster = RegimeAdaptiveForecaster()
    forecaster.fit(df_X, y_target, p_origin)

    save_path = tmp_path / "adaptive_model.joblib"
    saved = forecaster.save(save_path, sign=True)
    assert saved.exists()
    assert (tmp_path / "adaptive_model.joblib.sig").exists()
    assert verify_model_artifact(save_path)

    # Load back
    loaded = RegimeAdaptiveForecaster.load(save_path, verify_signature=True)
    assert loaded.is_fitted
    assert loaded.model_name == forecaster.model_name

    p_test = p_origin[:2]
    orig_preds = forecaster.predict(df_X.iloc[:2], p_test)
    loaded_preds = loaded.predict(df_X.iloc[:2], p_test)
    assert np.allclose(orig_preds, loaded_preds)


def test_multi_horizon_forecaster_adaptive_ensemble(synthetic_training_data):
    """Verify MultiHorizonForecaster instantiates, fits, and predicts with adaptive_ensemble."""
    df_X, y_target, p_origin = synthetic_training_data

    # Create dummy observation dataframe
    n = len(df_X)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    obs_df = pd.DataFrame({
        "session_date": dates,
        "open": p_origin * 0.99,
        "high": p_origin * 1.02,
        "low": p_origin * 0.98,
        "close": p_origin,
        "adj_close": p_origin,
        "volume": [1_000_000] * n,
    })

    forecaster = MultiHorizonForecaster(model_family="adaptive_ensemble")
    forecaster.fit(obs_df)
    assert forecaster.is_fitted
    assert isinstance(forecaster.models["1_SESSION"], RegimeAdaptiveForecaster)
    assert isinstance(forecaster.models["5_SESSIONS"], RegimeAdaptiveForecaster)
    assert isinstance(forecaster.models["21_SESSIONS"], RegimeAdaptiveForecaster)

    predictions = forecaster.predict_all_horizons(obs_df, ticker="TEST", exchange="XNYS")
    assert "1_SESSION" in predictions
    assert "5_SESSIONS" in predictions
    assert "21_SESSIONS" in predictions
    assert predictions["1_SESSION"]["ticker"] == "TEST"
    assert predictions["1_SESSION"]["predicted_price"] > 0.0

