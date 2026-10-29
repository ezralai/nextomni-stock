"""Test suite for Minimum Redundancy Maximum Relevance (mRMR) Feature Selection Engine."""

import argparse
import io
import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.cli import mrmr_select_command
from src.features.mrmr import (
    MRMRReceipt,
    MRMRStep,
    evaluate_mrmr_profile,
    select_mrmr_features,
)


@pytest.fixture
def synthetic_feature_data():
    """Create reproducible synthetic dataset with collinear features and known signal."""
    np.random.seed(42)
    n = 120
    # Core predictive signals
    signal1 = np.random.normal(0, 1, n)
    signal2 = np.random.normal(0, 1, n)
    y = 0.7 * signal1 + 0.7 * signal2 + np.random.normal(0, 0.1, n)

    # Feature 1: Highly correlated with signal1
    f1 = signal1 + np.random.normal(0, 0.05, n)
    # Feature 2: Highly collinear duplicate of f1 (redundant)
    f2 = f1 + np.random.normal(0, 0.05, n)
    # Feature 3: Independent predictive signal2
    f3 = signal2 + np.random.normal(0, 0.05, n)
    # Feature 4: Noise
    f4 = np.random.normal(0, 1, n)
    # Feature 5: Noise
    f5 = np.random.normal(0, 1, n)

    df_X = pd.DataFrame({
        "feat_signal_1": f1,
        "feat_dup_redundant": f2,
        "feat_indep_signal": f3,
        "feat_noise_1": f4,
        "feat_noise_2": f5,
    })
    return df_X, y


def test_select_mrmr_features_mid_and_miq(synthetic_feature_data):
    """Verify greedy selection, relevance ranking, and redundancy penalties."""
    df_X, y = synthetic_feature_data

    # Test MID
    receipt_mid = select_mrmr_features(
        X=df_X,
        y=y,
        n_features_to_select=3,
        criterion="MID",
        ticker="AAPL",
        horizon_sessions=1,
    )

    assert isinstance(receipt_mid, MRMRReceipt)
    assert receipt_mid.num_selected == 3
    assert receipt_mid.num_candidates == 5
    assert len(receipt_mid.selected_features) == 3
    assert receipt_mid.criterion == "MID"
    # First selected must be a true predictive signal
    assert receipt_mid.selected_features[0] in ("feat_signal_1", "feat_dup_redundant", "feat_indep_signal")
    # Both independent signals are selected into top 2, suppressing redundant duplicate
    assert "feat_indep_signal" in receipt_mid.selected_features[:2]
    assert ("feat_signal_1" in receipt_mid.selected_features[:2] or "feat_dup_redundant" in receipt_mid.selected_features[:2])
    assert receipt_mid.redundancy_reduction_pct >= 0.0

    # Test MIQ
    receipt_miq = select_mrmr_features(
        X=df_X,
        y=y,
        n_features_to_select=3,
        criterion="MIQ",
        ticker="AAPL",
        horizon_sessions=1,
    )
    assert receipt_miq.criterion == "MIQ"
    assert receipt_miq.num_selected == 3


def test_mrmr_receipt_tamper_detection(synthetic_feature_data):
    """Verify cryptographic HMAC-SHA256 signature integrity and tamper detection."""
    df_X, y = synthetic_feature_data
    receipt = select_mrmr_features(
        X=df_X,
        y=y,
        n_features_to_select=3,
        criterion="MID",
        ticker="NVDA",
    )

    assert receipt.verify_integrity() is True

    # Tamper with correlation
    tampered_receipt = MRMRReceipt(
        ticker=receipt.ticker,
        horizon_sessions=receipt.horizon_sessions,
        sample_sessions=receipt.sample_sessions,
        criterion=receipt.criterion,
        num_candidates=receipt.num_candidates,
        num_selected=receipt.num_selected,
        selected_features=receipt.selected_features,
        relevance_scores=receipt.relevance_scores,
        redundancy_scores=receipt.redundancy_scores,
        mrmr_scores=receipt.mrmr_scores,
        selection_history=receipt.selection_history,
        mean_candidate_correlation=receipt.mean_candidate_correlation,
        mean_selected_correlation=receipt.mean_selected_correlation + 0.5,
        redundancy_reduction_pct=receipt.redundancy_reduction_pct,
        summary=receipt.summary,
        signature=receipt.signature,
    )
    assert tampered_receipt.verify_integrity() is False


def test_mrmr_input_validation_and_edge_cases(synthetic_feature_data):
    """Test validation constraints on criteria, sample sizes, and invalid tickers."""
    df_X, y = synthetic_feature_data

    # Invalid criterion
    with pytest.raises(ValueError, match="Invalid mRMR criterion"):
        select_mrmr_features(df_X, y, criterion="INVALID_CRIT")

    # Empty DataFrame
    with pytest.raises(ValueError, match="Insufficient feature rows"):
        select_mrmr_features(pd.DataFrame(), y)

    # Invalid ticker in evaluate_mrmr_profile
    with pytest.raises(ValueError, match="Invalid ticker format"):
        evaluate_mrmr_profile("BAD_TICKER$$$")

    # Invalid horizon
    with pytest.raises(ValueError, match="Invalid horizon_sessions"):
        evaluate_mrmr_profile("AAPL", horizon_sessions=10)


def test_cli_mrmr_select_command_stdout_and_json(synthetic_feature_data, monkeypatch):
    """Test CLI mrmr-select command in both stdout and JSON formats."""
    df_X, y = synthetic_feature_data
    mock_receipt = select_mrmr_features(df_X, y, n_features_to_select=3, ticker="AAPL")

    monkeypatch.setattr(
        "src.features.mrmr.evaluate_mrmr_profile",
        lambda ticker, horizon_sessions=1, n_features=5, criterion="MID", db_path=None: mock_receipt,
    )

    # 1. Standard stdout format
    args_std = argparse.Namespace(
        ticker="AAPL",
        horizon=1,
        n_features=3,
        criterion="MID",
        json=False,
    )
    with patch("sys.stdout", new=io.StringIO()) as fake_out:
        mrmr_select_command(args_std)
        output = fake_out.getvalue()
        assert "Minimum Redundancy Maximum Relevance (mRMR) Feature Selection" in output
        assert "Security / Ticker:            AAPL" in output
        assert "Greedy Forward Selection History:" in output
        assert "Audit Digest (HMAC):" in output

    # 2. JSON mode
    args_json = argparse.Namespace(
        ticker="AAPL",
        horizon=1,
        n_features=3,
        criterion="MID",
        json=True,
    )
    with patch("sys.stdout", new=io.StringIO()) as fake_json_out:
        mrmr_select_command(args_json)
        payload = json.loads(fake_json_out.getvalue())
        assert payload["ticker"] == "AAPL"
        assert "selected_features" in payload
        assert "signature" in payload
        assert payload["criterion"] == "MID"
