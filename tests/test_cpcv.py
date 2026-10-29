"""Unit and integration test suite for Marcos López de Prado (2018) CPCV & PBO Engine."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.cli import cpcv_validate_command
from src.evaluation.cpcv import (
    CpcvReceipt,
    compute_probability_of_backtest_overfitting,
    evaluate_cpcv_profile,
    generate_combinatorial_splits,
    get_train_test_purged_embargoed_indices,
)


@pytest.fixture
def synthetic_ohlc_df() -> pd.DataFrame:
    """Generate 180 sessions of synthetic OHLC data for CPCV testing."""
    np.random.seed(42)
    n = 180
    dates = pd.date_range("2025-01-01", periods=n, freq="B").strftime("%Y-%m-%d")
    returns = np.random.normal(0.0005, 0.015, size=n)
    close = 100.0 * np.exp(np.cumsum(returns))
    high = close * (1.0 + np.random.uniform(0.002, 0.01, size=n))
    low = close * (1.0 - np.random.uniform(0.002, 0.01, size=n))
    open_p = close * (1.0 + np.random.normal(0.0, 0.003, size=n))
    volume = np.random.uniform(500_000, 2_000_000, size=n)

    df = pd.DataFrame({
        "session_date": dates,
        "open": open_p,
        "high": high,
        "low": low,
        "close": close,
        "adj_close": close,
        "volume": volume,
    })
    return df


def test_combinatorial_splits_and_purging_embargoing():
    """Verify that N=6, k=2 produces exact C(6, 2) = 15 folds with zero train/test leakage."""
    n_samples = 120
    num_groups = 6
    test_groups = 2
    horizon = 5
    embargo_pct = 0.02  # 2% of 120 = 2 sessions

    folds = generate_combinatorial_splits(
        n_samples=n_samples,
        num_groups=num_groups,
        test_groups=test_groups,
        horizon_sessions=horizon,
        embargo_pct=embargo_pct,
    )

    assert len(folds) == 15  # C(6, 2) = 15

    for fold in folds:
        train_idx = fold["train_indices"]
        test_idx = fold["test_indices"]
        # Absolute zero overlap
        overlap = np.intersect1d(train_idx, test_idx)
        assert len(overlap) == 0

        # Purging count should be positive because horizon=5 overlaps boundaries
        assert fold["purged_count"] > 0
        assert fold["embargo_sessions"] >= 2


def test_purged_embargoed_boundary_isolation():
    """Verify exact boundary purging before test blocks and post-test embargo buffers."""
    n_samples = 100
    test_ranges = [(40, 60)]  # Test block from 40 to 59
    horizon = 3  # Label finishes at t + 3
    embargo = 5  # Post-test quarantine

    train_idx, test_idx, purged = get_train_test_purged_embargoed_indices(
        n_samples=n_samples,
        test_block_ranges=test_ranges,
        horizon_sessions=horizon,
        embargo_sessions=embargo,
    )

    # Test indices must be exactly range(40, 60)
    assert np.array_equal(test_idx, np.arange(40, 60))

    # Pre-test purge: samples 38, 39 finish at 41, 42 (inside test). Must not be in train!
    assert 38 not in train_idx
    assert 39 not in train_idx

    # Post-test embargo: samples 60, 61, 62, 63, 64 must be embargoed!
    for emb_idx in range(60, 65):
        assert emb_idx not in train_idx

    # Sample 65 should be in train
    assert 65 in train_idx
    # Sample 37 finishes at 40 (start of test), should be preserved or purged depending on boundary
    assert 35 in train_idx


def test_probability_of_backtest_overfitting_calculation():
    """Verify PBO calculation across varying path metric distributions."""
    # Symmetrically distributed Sharpe ratios
    scores_balanced = [1.5, 1.2, 0.8, -0.2, -0.6]
    pbo_balanced = compute_probability_of_backtest_overfitting(scores_balanced)
    assert 0.0 <= pbo_balanced <= 100.0

    # Superior consistent alpha across all paths
    scores_superior = [2.5, 2.3, 2.1, 1.9, 1.8]
    pbo_superior = compute_probability_of_backtest_overfitting(scores_superior)
    assert 0.0 <= pbo_superior <= 100.0


def test_evaluate_cpcv_profile_and_tamper(synthetic_ohlc_df):
    """Verify end-to-end CPCV evaluation, metrics, and HMAC-SHA256 tamper protection."""
    df = synthetic_ohlc_df
    receipt = evaluate_cpcv_profile(
        ticker="AAPL",
        df=df,
        horizon_sessions=1,
        num_groups=5,
        test_groups=2,
        embargo_pct=0.01,
    )

    assert isinstance(receipt, CpcvReceipt)
    assert receipt.ticker == "AAPL"
    assert receipt.num_groups == 5
    assert receipt.test_groups == 2
    assert receipt.num_folds == 10  # C(5, 2) = 10
    assert receipt.num_paths == 4  # (2 / 5) * 10 = 4
    assert receipt.mean_model_mae > 0.0
    assert 0.0 <= receipt.directional_accuracy_pct <= 100.0
    assert 0.0 <= receipt.probability_of_backtest_overfitting_pct <= 100.0
    assert receipt.cpcv_status in ("ROBUST_LOW_OVERFITTING", "MODERATE_SELECTION_BIAS", "SEVERELY_OVERFIT")
    assert receipt.verify_integrity() is True

    # Tamper with PBO percentage
    tampered_dict = receipt.to_dict()
    tampered_dict["probability_of_backtest_overfitting_pct"] = 99.99
    tampered_receipt = CpcvReceipt(**tampered_dict)
    assert tampered_receipt.verify_integrity() is False

    # Tamper with status
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["cpcv_status"] = "SEVERELY_OVERFIT" if receipt.cpcv_status != "SEVERELY_OVERFIT" else "ROBUST_LOW_OVERFITTING"
    tampered_receipt2 = CpcvReceipt(**tampered_dict2)
    assert tampered_receipt2.verify_integrity() is False


def test_cpcv_security_and_edge_cases(synthetic_ohlc_df):
    """Verify ticker allowlist validation, parameter constraints, and short history exceptions."""
    # Invalid ticker format
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_cpcv_profile("INVALID$$$", synthetic_ohlc_df)

    # Test groups >= num_groups
    with pytest.raises(ValueError, match="strictly less than"):
        generate_combinatorial_splits(100, num_groups=4, test_groups=4)

    # Insufficient dataset length
    short_df = synthetic_ohlc_df.head(40)
    with pytest.raises(ValueError, match="Insufficient observations"):
        evaluate_cpcv_profile("AAPL", short_df)


def test_cli_cpcv_validate_command_stdout_and_json(synthetic_ohlc_df, monkeypatch):
    """Verify CLI cpcv-validate command execution across stdout and JSON output modes."""
    monkeypatch.setattr(
        "src.data.db.get_observations",
        lambda *args, **kwargs: synthetic_ohlc_df,
    )

    # 1. Stdout execution
    args_stdout = argparse.Namespace(
        ticker="AAPL",
        horizon=1,
        groups=5,
        test_groups=2,
        embargo=0.01,
        json=False,
    )
    cpcv_validate_command(args_stdout)

    # 2. JSON execution
    args_json = argparse.Namespace(
        ticker="AAPL",
        horizon=1,
        groups=5,
        test_groups=2,
        embargo=0.01,
        json=True,
    )
    cpcv_validate_command(args_json)
