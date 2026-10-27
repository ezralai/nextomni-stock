"""Unit tests for Convex Ensemble forecasting and cryptographic artifact signing."""

from pathlib import Path
from typing import Tuple
import numpy as np
import pandas as pd
import pytest

from src.models.ensemble import ConvexEnsembleForecaster
from src.models.multi_horizon import MultiHorizonForecaster
from src.security.audit import sign_model_artifact, verify_model_artifact
from src.security.validators import SecurityValidationError


@pytest.fixture
def synthetic_data() -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate synthetic feature matrix, price target, and origin price."""
    np.random.seed(42)
    n = 60
    X = pd.DataFrame(
        {
            "ret_1": np.random.normal(0.001, 0.01, size=n),
            "ret_5": np.random.normal(0.005, 0.02, size=n),
            "vol_5": np.random.uniform(0.01, 0.03, size=n),
            "rsi_14": np.random.uniform(30, 70, size=n),
        }
    )
    p_orig = pd.Series(100.0 + np.cumsum(np.random.normal(0, 1.0, size=n)))
    y_target = p_orig * (1.0 + X["ret_1"] * 0.8 + np.random.normal(0, 0.002, size=n))
    return X, y_target, p_orig


def test_convex_ensemble_fit_and_predict(synthetic_data):
    """Verify ConvexEnsembleForecaster fits, solves convex weights, and predicts accurately."""
    X, y, p_orig = synthetic_data
    ensemble = ConvexEnsembleForecaster()
    assert ensemble.is_fitted is False

    ensemble.fit(X, y, p_orig)
    assert ensemble.is_fitted is True

    w_lin, w_gb = ensemble.weights
    assert w_lin >= 0.0
    assert w_gb >= 0.0
    assert pytest.approx(w_lin + w_gb, abs=1e-3) == 1.0

    preds = ensemble.predict(X.iloc[:5], p_orig.iloc[:5])
    assert len(preds) == 5
    assert all(p > 0 for p in preds)


def test_convex_ensemble_predict_interval(synthetic_data):
    """Verify calibrated prediction interval blending."""
    X, y, p_orig = synthetic_data
    ensemble = ConvexEnsembleForecaster()
    ensemble.fit(X, y, p_orig)

    pred, lower, upper = ensemble.predict_interval(X.iloc[:3], p_orig.iloc[:3], confidence_level=0.80)
    for p, l, u in zip(pred, lower, upper):
        assert l <= p <= u


def test_convex_ensemble_feature_importances(synthetic_data):
    """Verify combined quantitative feature importances."""
    X, y, p_orig = synthetic_data
    ensemble = ConvexEnsembleForecaster()
    ensemble.fit(X, y, p_orig)

    importances = ensemble.get_feature_importances()
    assert isinstance(importances, dict)
    assert len(importances) >= 4
    for feat in ["ret_1", "ret_5", "vol_5", "rsi_14"]:
        assert feat in importances
    assert all(v >= 0.0 for v in importances.values())


def test_convex_ensemble_save_load_and_signature_verification(tmp_path: Path, synthetic_data):
    """Verify serialization generates HMAC signature and loads successfully."""
    X, y, p_orig = synthetic_data
    ensemble = ConvexEnsembleForecaster()
    ensemble.fit(X, y, p_orig)

    save_path = tmp_path / "ensemble_test.joblib"
    saved = ensemble.save(save_path)
    assert saved.exists()

    sig_path = tmp_path / "ensemble_test.joblib.sig"
    assert sig_path.exists()
    assert verify_model_artifact(saved) is True

    # Deserialization with signature validation
    loaded = ConvexEnsembleForecaster.load(saved, verify_signature=True)
    assert loaded.is_fitted is True
    assert loaded.weights == ensemble.weights

    orig_preds = ensemble.predict(X.iloc[:2], p_orig.iloc[:2])
    loaded_preds = loaded.predict(X.iloc[:2], p_orig.iloc[:2])
    np.testing.assert_allclose(orig_preds, loaded_preds, rtol=1e-5)


def test_convex_ensemble_tamper_detection(tmp_path: Path, synthetic_data):
    """Verify SecurityValidationError when model artifact bytes are tampered."""
    X, y, p_orig = synthetic_data
    ensemble = ConvexEnsembleForecaster()
    ensemble.fit(X, y, p_orig)

    save_path = tmp_path / "tamper_test.joblib"
    ensemble.save(save_path)

    # Tamper with file
    with open(save_path, "ab") as f:
        f.write(b"\x00\xffCORRUPT")

    with pytest.raises(SecurityValidationError, match="Cryptographic signature verification failed"):
        ConvexEnsembleForecaster.load(save_path, verify_signature=True)


def test_multi_horizon_forecaster_with_ensemble(tmp_path: Path):
    """Verify MultiHorizonForecaster end-to-end integration with model_family='ensemble'."""
    dates = pd.date_range("2024-01-01", periods=60, freq="B").strftime("%Y-%m-%d")
    prices = [100.0]
    for _ in range(59):
        prices.append(prices[-1] * (1.0 + np.random.normal(0.0005, 0.015)))

    df = pd.DataFrame(
        {
            "ticker": ["TEST"] * 60,
            "exchange": ["XNYS"] * 60,
            "session_date": dates,
            "open": prices,
            "high": [p * 1.01 for p in prices],
            "low": [p * 0.99 for p in prices],
            "close": prices,
            "adj_close": prices,
            "volume": [100000] * 60,
            "data_status": ["FINAL"] * 60,
            "provider": ["Mock"] * 60,
        }
    )

    forecaster = MultiHorizonForecaster(model_family="ensemble")
    forecaster.fit(df)
    assert forecaster.is_fitted is True

    predictions = forecaster.predict_all_horizons(df, ticker="TEST", exchange="XNYS")
    for h in ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]:
        assert h in predictions
        assert predictions[h]["predicted_price"] > 0
        assert predictions[h]["lower_bound"] <= predictions[h]["predicted_price"] <= predictions[h]["upper_bound"]

    # Save and load with signatures
    models_dir = tmp_path / "models"
    forecaster.save_models(models_dir, ticker="TEST")

    loaded = MultiHorizonForecaster.load_models(models_dir, ticker="TEST")
    assert loaded.model_family == "ensemble"
    assert loaded.is_fitted is True
