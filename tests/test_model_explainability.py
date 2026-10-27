"""Unit and mathematical verification tests for Explainable AI forecast attribution.

Verifies:
1. Exact linear additivity of feature attributions for LinearRegressionForecaster.
2. Additive marginal attribution consistency for GradientBoostingForecaster.
3. Convex combination blending of attributions for ConvexEnsembleForecaster.
4. Proper integration in MultiHorizonForecaster and error handling for unfitted models.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import FEATURE_COLUMNS, build_feature_matrix
from src.models.ensemble import ConvexEnsembleForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster
from src.models.multi_horizon import MultiHorizonForecaster


@pytest.fixture
def explain_sample_df() -> pd.DataFrame:
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


def test_linear_explain_forecast_additivity(explain_sample_df: pd.DataFrame):
    """Verify that linear feature attributions sum exactly to the predicted return."""
    feat_df, cols = build_feature_matrix(explain_sample_df, include_target=True, horizon_sessions=1)
    train_df = feat_df.dropna(subset=["target_price"])
    X = train_df[cols]
    y = train_df["target_price"]
    p_orig = train_df["adj_close"]

    model = LinearRegressionForecaster(alpha=1.0)
    model.fit(X, y, p_orig)

    test_row = X.iloc[-1:]
    origin_price = float(p_orig.iloc[-1])
    pred_price = float(model.predict(test_row, origin_price)[0])
    expected_return = (pred_price - origin_price) / origin_price

    attribution = model.explain_forecast(test_row)
    assert "intercept" in attribution
    for col in cols:
        assert col in attribution

    total_explained_return = sum(attribution.values())
    assert abs(total_explained_return - expected_return) < 1e-4


def test_gb_explain_forecast_additivity(explain_sample_df: pd.DataFrame):
    """Verify that GBDT feature attributions sum to the predicted return."""
    feat_df, cols = build_feature_matrix(explain_sample_df, include_target=True, horizon_sessions=1)
    train_df = feat_df.dropna(subset=["target_price"])
    X = train_df[cols]
    y = train_df["target_price"]
    p_orig = train_df["adj_close"]

    model = GradientBoostingForecaster(max_depth=2, max_iter=30, random_state=42)
    model.fit(X, y, p_orig, fit_quantiles=False)

    test_row = X.iloc[-1:]
    origin_price = float(p_orig.iloc[-1])
    pred_price = float(model.predict(test_row, origin_price)[0])
    expected_return = (pred_price - origin_price) / origin_price

    attribution = model.explain_forecast(test_row)
    assert "intercept" in attribution
    for col in cols:
        assert col in attribution

    total_explained_return = sum(attribution.values())
    assert abs(total_explained_return - expected_return) < 1e-4


def test_ensemble_explain_forecast_blending(explain_sample_df: pd.DataFrame):
    """Verify that Convex Ensemble blends submodel attributions additively."""
    feat_df, cols = build_feature_matrix(explain_sample_df, include_target=True, horizon_sessions=1)
    train_df = feat_df.dropna(subset=["target_price"])
    X = train_df[cols]
    y = train_df["target_price"]
    p_orig = train_df["adj_close"]

    model = ConvexEnsembleForecaster()
    model.fit(X, y, p_orig, fit_quantiles=False)

    test_row = X.iloc[-1:]
    origin_price = float(p_orig.iloc[-1])
    pred_price = float(model.predict(test_row, origin_price)[0])
    expected_return = (pred_price - origin_price) / origin_price

    attribution = model.explain_forecast(test_row)
    assert "intercept" in attribution
    for col in cols:
        assert col in attribution

    total_explained_return = sum(attribution.values())
    assert abs(total_explained_return - expected_return) < 1e-3


def test_multi_horizon_forecast_includes_attributions(explain_sample_df: pd.DataFrame):
    """Verify that MultiHorizonForecaster populates signal_attributions for each horizon."""
    forecaster = MultiHorizonForecaster(model_family="ensemble")
    forecaster.fit(explain_sample_df)

    forecasts = forecaster.predict_all_horizons(explain_sample_df, ticker="TEST", exchange="XNYS")
    for h_key in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        fc = forecasts[h_key]
        assert "signal_attributions" in fc
        attrs = fc["signal_attributions"]
        assert "intercept" in attrs
        assert len(attrs) >= len(FEATURE_COLUMNS)


def test_unfitted_model_explain_raises():
    """Verify that calling explain_forecast on unfitted models raises RuntimeError."""
    lin = LinearRegressionForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        lin.explain_forecast(pd.DataFrame([[1.0] * 12], columns=FEATURE_COLUMNS))

    gb = GradientBoostingForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        gb.explain_forecast(pd.DataFrame([[1.0] * 12], columns=FEATURE_COLUMNS))

    ens = ConvexEnsembleForecaster()
    with pytest.raises(RuntimeError, match="must be fitted"):
        ens.explain_forecast(pd.DataFrame([[1.0] * 12], columns=FEATURE_COLUMNS))
