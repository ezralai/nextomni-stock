"""Unit and integration tests for Out-of-Fold Stacking Meta-Learner (Super Learner) forecaster."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.models.multi_horizon import MultiHorizonForecaster
from src.models.stacking import StackingForecaster
from src.security.validators import SecurityValidationError


@pytest.fixture
def sample_features_and_targets() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Generate realistic synthetic feature matrix and target prices for 80 sessions."""
    np.random.seed(42)
    n = 80
    feat_dict = {
        "return_lag_1": np.random.normal(0.001, 0.015, n),
        "return_lag_5": np.random.normal(0.005, 0.03, n),
        "volatility_20": np.random.uniform(0.10, 0.30, n),
        "price_to_sma_20": np.random.normal(1.01, 0.04, n),
        "vol_regime_ratio": np.random.uniform(0.8, 1.4, n),
        "rsi_14": np.random.uniform(30.0, 70.0, n),
    }
    X = pd.DataFrame(feat_dict)
    p_origin = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, n)))
    # Future target price has mild momentum correlation with return_lag_1
    future_returns = 0.4 * X["return_lag_1"].values + np.random.normal(0, 0.005, n)
    y_target_price = p_origin * (1.0 + future_returns)

    return X, y_target_price, p_origin


def test_stacking_meta_optimization_constraints() -> None:
    """Verify meta-weights optimization strictly enforces non-negativity and unit sum."""
    model = StackingForecaster()

    # Case 1: Collinear predictions with different residual errors
    np.random.seed(42)
    n = 50
    y_true = np.random.normal(0.01, 0.02, n)
    z1 = y_true + np.random.normal(0, 0.005, n)  # Accurate model
    z2 = y_true + np.random.normal(0, 0.05, n)   # Noisy model
    Z = np.column_stack([z1, z2])

    w1, w2 = model._optimize_meta_weights(Z, y_true)

    assert w1 >= 0.0
    assert w2 >= 0.0
    assert np.isclose(w1 + w2, 1.0, atol=1e-3)
    assert w1 > w2, f"Expected accurate model to receive higher weight ({w1} > {w2})"


def test_stacking_fit_and_predict(sample_features_and_targets) -> None:
    """Verify full out-of-fold stacking training, weight convergence, and positive predictions."""
    X, y, p = sample_features_and_targets
    forecaster = StackingForecaster(n_splits=3, min_fold_size=25)

    forecaster.fit(X, y, p, fit_quantiles=True)

    assert forecaster._fitted is True
    w1, w2 = forecaster.weights
    assert w1 >= 0.0 and w2 >= 0.0
    assert np.isclose(w1 + w2, 1.0, atol=1e-3)

    preds = forecaster.predict(X, p)
    assert len(preds) == len(X)
    assert np.all(preds > 0.0)
    assert np.all(np.isfinite(preds))


def test_stacking_predict_interval_and_explain(sample_features_and_targets) -> None:
    """Verify calibrated prediction intervals and additive signal attribution blending."""
    X, y, p = sample_features_and_targets
    forecaster = StackingForecaster().fit(X, y, p)

    pred, low, up = forecaster.predict_interval(X, p, confidence_level=0.80)

    assert len(pred) == len(X)
    assert len(low) == len(X)
    assert len(up) == len(X)
    assert np.all(low <= up)

    exp = forecaster.explain_forecast(X.iloc[-1:])
    assert isinstance(exp, dict)
    assert "return_lag_1" in exp
    assert np.all([np.isfinite(v) for v in exp.values()])


def test_stacking_save_and_load_hmac(sample_features_and_targets, tmp_path: Path) -> None:
    """Verify cryptographic HMAC-SHA256 signature verification during save and load."""
    X, y, p = sample_features_and_targets
    forecaster = StackingForecaster().fit(X, y, p)

    model_path = tmp_path / "models" / "stacking_test.joblib"
    saved = forecaster.save(model_path)
    assert saved.exists()
    assert (tmp_path / "models" / "stacking_test.joblib.sig").exists()

    loaded = StackingForecaster.load(model_path)
    assert loaded._fitted is True
    assert np.allclose(loaded.weights, forecaster.weights)

    orig_pred = forecaster.predict(X, p)
    load_pred = loaded.predict(X, p)
    assert np.allclose(orig_pred, load_pred)

    # Tampering test
    with open(model_path, "ab") as f:
        f.write(b"\x00CORRUPT_BYTES")

    with pytest.raises(SecurityValidationError, match="HMAC integrity verification failed"):
        StackingForecaster.load(model_path)


def test_stacking_small_sample_fallback() -> None:
    """Verify fallback to equal weights when training sample size is under 60."""
    np.random.seed(123)
    n = 25
    X = pd.DataFrame({"feat1": np.random.randn(n), "feat2": np.random.randn(n)})
    p = np.full(n, 50.0)
    y = p * (1.0 + np.random.normal(0, 0.01, n))

    model = StackingForecaster()
    model.fit(X, y, p)

    assert model._fitted is True
    assert model.weights == (0.50, 0.50)


def test_multi_horizon_stacking_integration() -> None:
    """Verify MultiHorizonForecaster orchestrates StackingForecaster across 1, 5, and 21 horizons."""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    prices = 100.0 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n)))

    df = pd.DataFrame(
        {
            "session_date": dates,
            "open": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "adj_close": prices,
            "volume": np.random.uniform(1e6, 2e6, n),
        }
    )

    mh = MultiHorizonForecaster(model_family="stacking")
    mh.fit(df)

    assert mh.is_fitted is True
    assert isinstance(mh.models["1_SESSION"], StackingForecaster)
    assert isinstance(mh.models["5_SESSIONS"], StackingForecaster)
    assert isinstance(mh.models["21_SESSIONS"], StackingForecaster)

    forecasts = mh.predict_all_horizons(df, ticker="TEST_STACK", exchange="XNYS", confidence_level=0.80)
    assert len(forecasts) == 3
    for h_key, f in forecasts.items():
        assert f["predicted_price"] > 0.0
        assert f["lower_bound"] <= f["predicted_price"] <= f["upper_bound"] or f["lower_bound"] <= f["upper_bound"]
