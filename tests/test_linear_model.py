"""Tests for NextOmni Stock feature engineering, Linear Regression, and Walk-Forward Validation."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.evaluation.validation import WalkForwardValidator
from src.features.builder import FEATURE_COLUMNS, build_feature_matrix
from src.models.linear_model import LinearRegressionForecaster


@pytest.fixture
def sample_timeseries_df() -> pd.DataFrame:
    """Generate 60 sessions of deterministic synthetic price data."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    np.random.seed(42)
    # Simulated geometric random walk
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


def test_build_feature_matrix_shapes_and_columns(sample_timeseries_df: pd.DataFrame):
    """Verify feature matrix contains all required features and aligns target price."""
    feature_df, feat_cols = build_feature_matrix(
        sample_timeseries_df, include_target=True, horizon_sessions=1
    )

    # 60 total rows - 21 initial warm-up rows (1 for pct_change + 20 rolling returns) = 39 rows
    assert len(feature_df) == 39

    for col in FEATURE_COLUMNS:
        assert col in feature_df.columns
        assert not feature_df[col].isna().any()

    # Target price present
    assert "target_price" in feature_df.columns
    # The last row must have NaN for target_price because horizon=1 is unobserved
    assert pd.isna(feature_df["target_price"].iloc[-1])


def test_linear_forecaster_fit_predict_save_load(tmp_path: Path, sample_timeseries_df: pd.DataFrame):
    """Verify Linear Regression fits, predicts, and can be saved/loaded via joblib."""
    feature_df, feat_cols = build_feature_matrix(sample_timeseries_df, include_target=True)
    trainable = feature_df.dropna(subset=["target_price"]).copy()

    X = trainable[feat_cols]
    y = trainable["target_price"]
    p_orig = trainable["adj_close"]

    forecaster = LinearRegressionForecaster(alpha=0.5)
    assert forecaster.is_fitted is False

    forecaster.fit(X, y, p_orig)
    assert forecaster.is_fitted is True

    # Check predictions
    preds = forecaster.predict(X, p_orig)
    assert len(preds) == len(trainable)
    assert np.all(preds > 0)

    # Check calibrated prediction intervals
    pred_p, low_p, up_p = forecaster.predict_interval(X, p_orig, confidence_level=0.80)
    assert len(low_p) == len(trainable)
    assert len(up_p) == len(trainable)
    assert np.all(low_p <= pred_p)
    assert np.all(pred_p <= up_p)

    # Check coefficients
    coefs = forecaster.get_coefficients()
    assert len(coefs) == len(feat_cols) + 1  # Features + intercept
    assert "intercept" in coefs
    for f in feat_cols:
        assert f in coefs

    # Check persistence
    artifact_path = tmp_path / "test_model.joblib"
    forecaster.save(artifact_path)
    assert artifact_path.exists()

    loaded = LinearRegressionForecaster.load(artifact_path)
    assert loaded.is_fitted is True
    loaded_preds = loaded.predict(X, p_orig)
    np.testing.assert_allclose(preds, loaded_preds)

    l_pred, l_low, l_up = loaded.predict_interval(X, p_orig, confidence_level=0.80)
    np.testing.assert_allclose(low_p, l_low)
    np.testing.assert_allclose(up_p, l_up)


def test_walk_forward_validator(sample_timeseries_df: pd.DataFrame):
    """Verify walk-forward validation runs chronologically and computes comparative metrics."""
    # Min train 25 sessions, step 1
    validator = WalkForwardValidator(min_train_sessions=25, step_size=1)
    results = validator.evaluate(sample_timeseries_df, horizon_sessions=1, alpha=1.0)

    assert "model_metrics" in results
    assert "baseline_metrics" in results
    assert "mae_improvement_pct" in results
    assert "linear_model_coefficients" in results
    assert results["walk_forward_steps"] > 0

    # Ensure model metrics contain standard keys
    for key in ["mae", "rmse", "mape_pct", "directional_accuracy_pct"]:
        assert key in results["model_metrics"]
        assert key in results["baseline_metrics"]
