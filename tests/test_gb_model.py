"""Unit tests for GradientBoostingForecaster."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.features.builder import build_feature_matrix
from src.models.gb_model import GradientBoostingForecaster


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate 60 sessions of deterministic synthetic price data."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.015, size=60)
    prices = [100.0]
    for r in returns[1:]:
        prices.append(prices[-1] * (1.0 + r))

    volumes = [10000 + int(abs(r) * 100000) for r in returns]

    return pd.DataFrame(
        {
            "ticker": ["TEST"] * 60,
            "exchange": ["XNYS"] * 60,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": volumes,
            "data_status": ["END_OF_DAY"] * 60,
            "provider": ["Mock"] * 60,
        }
    )


def test_gradient_boosting_forecaster_fit_predict_save_load(sample_timeseries_df, tmp_path):
    """Verify GradientBoostingForecaster fits, predicts price, and serializes cleanly."""
    feature_df, feat_cols = build_feature_matrix(
        sample_timeseries_df, include_target=True, horizon_sessions=1
    )
    valid_df = feature_df.dropna(subset=["target_price"]).copy().reset_index(drop=True)

    X_train = valid_df[feat_cols]
    y_train = valid_df["target_price"]
    p_orig = valid_df["adj_close"]

    forecaster = GradientBoostingForecaster(max_iter=10, min_samples_leaf=5)
    assert not forecaster.is_fitted

    forecaster.fit(X_train, y_train, p_orig)
    assert forecaster.is_fitted
    assert forecaster.model_name == "HistGradientBoosting"

    # Prediction
    test_X = X_train.iloc[-1:]
    test_p_orig = float(p_orig.iloc[-1])
    preds = forecaster.predict(test_X, test_p_orig)
    assert len(preds) == 1
    assert isinstance(preds[0], float)
    assert preds[0] > 0

    # Prediction Interval
    pred_p, low_p, up_p = forecaster.predict_interval(test_X, test_p_orig, confidence_level=0.80)
    assert len(low_p) == 1
    assert len(up_p) == 1
    assert low_p[0] <= pred_p[0] <= up_p[0]

    # Feature importances
    importances = forecaster.get_feature_importances()
    assert isinstance(importances, dict)
    assert len(importances) == len(feat_cols)

    # Save and Load
    artifact_path = tmp_path / "gb_test.joblib"
    saved_path = forecaster.save(artifact_path)
    assert saved_path.exists()

    loaded_forecaster = GradientBoostingForecaster.load(artifact_path)
    assert loaded_forecaster.is_fitted
    loaded_preds = loaded_forecaster.predict(test_X, test_p_orig)
    np.testing.assert_allclose(preds, loaded_preds, rtol=1e-5)

    l_pred, l_low, l_up = loaded_forecaster.predict_interval(test_X, test_p_orig, confidence_level=0.80)
    np.testing.assert_allclose(low_p, l_low, rtol=1e-5)
    np.testing.assert_allclose(up_p, l_up, rtol=1e-5)
    assert getattr(loaded_forecaster, "_quantile_fitted", False) is True

    # Test arbitrary confidence level fallback
    f_pred, f_low, f_up = loaded_forecaster.predict_interval(test_X, test_p_orig, confidence_level=0.95)
    assert f_low[0] < l_low[0]
    assert f_up[0] > l_up[0]

