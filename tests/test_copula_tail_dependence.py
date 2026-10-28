"""Unit and integration tests for Asymmetric Lower-Tail Dependence and Joint Crash Copula Engine."""

import argparse
import io
import json
from unittest.mock import patch
import numpy as np
import pandas as pd
import pytest

from src.analytics.copula import (
    TailDependenceReceipt,
    calculate_bivariate_lower_tail_dependence,
    compute_empirical_pit,
    compute_lower_tail_dependence_matrix,
    evaluate_universe_tail_dependence,
)
from src.cli import tail_dependence_command


@pytest.fixture
def synthetic_cocrash_returns_df() -> pd.DataFrame:
    """Generate 3-asset returns matrix where AssetA and AssetB exhibit high joint crash dependence."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2024-01-01", periods=n, freq="B").strftime("%Y-%m-%d").tolist()

    # Asset A & B: Strong lower tail co-crash dependence (common market factor with negative skewness)
    common_factor = np.random.normal(0.0005, 0.015, n)
    # Severe systemic crash events on 6 sessions
    crash_indices = [15, 30, 45, 60, 75, 90]
    common_factor[crash_indices] = -0.06

    r_a = common_factor + np.random.normal(0, 0.005, n)
    r_b = 0.9 * common_factor + np.random.normal(0, 0.005, n)
    # Asset C: Uncorrelated defensive asset (e.g. gold or utilities)
    r_c = np.random.normal(0.0002, 0.01, n)

    df = pd.DataFrame(
        {
            "ASSET_A": r_a,
            "ASSET_B": r_b,
            "ASSET_C": r_c,
        },
        index=dates,
    )
    return df


def test_compute_empirical_pit_bounds_and_ranking() -> None:
    """Verify empirical PIT produces uniform margins bounded strictly in (0, 1) preserving order."""
    returns = [-0.05, -0.02, 0.0, 0.01, 0.04]
    u = compute_empirical_pit(returns)

    assert len(u) == 5
    assert np.all(u > 0.0)
    assert np.all(u < 1.0)
    # Preserves order
    assert np.all(np.diff(u) > 0)

    # Empty inputs guard
    assert len(compute_empirical_pit([])) == 0


def test_bivariate_lower_tail_dependence_properties() -> None:
    """Verify identity, symmetry, and tail sensitivity of bivariate lower-tail dependence."""
    np.random.seed(42)
    n = 100
    u_base = np.random.uniform(0.01, 0.99, n)

    # 1. Identity property: lambda(x, x) == 1.0
    hyb_id, emp_id, cla_id = calculate_bivariate_lower_tail_dependence(u_base, u_base)
    assert hyb_id == 1.0
    assert emp_id == 1.0
    assert cla_id == 1.0

    # 2. Symmetry property: lambda(u1, u2) == lambda(u2, u1)
    u_other = np.random.uniform(0.01, 0.99, n)
    hyb_12, _, _ = calculate_bivariate_lower_tail_dependence(u_base, u_other)
    hyb_21, _, _ = calculate_bivariate_lower_tail_dependence(u_other, u_base)
    assert np.isclose(hyb_12, hyb_21)

    # 3. High co-crash dependence on shared crash indices
    u_crash1 = np.random.uniform(0.2, 0.9, n)
    u_crash2 = np.random.uniform(0.2, 0.9, n)
    # Force simultaneous crash at bottom 5% tail for first 5 sessions
    u_crash1[:5] = 0.02
    u_crash2[:5] = 0.03
    hyb_cr, emp_cr, _ = calculate_bivariate_lower_tail_dependence(u_crash1, u_crash2, threshold=0.05)
    assert emp_cr > 0.40, f"Expected high empirical co-crash tail dependence, got {emp_cr}"


def test_compute_lower_tail_dependence_matrix_properties(synthetic_cocrash_returns_df: pd.DataFrame) -> None:
    """Verify tail dependence matrix is symmetric, positive, and has unit diagonal."""
    mat = compute_lower_tail_dependence_matrix(synthetic_cocrash_returns_df, threshold=0.05)

    assert mat.shape == (3, 3)
    # Unit diagonal
    assert np.allclose(np.diag(mat.values), 1.0)
    # Symmetric
    assert np.allclose(mat.values, mat.values.T)
    # Elements in [0, 1]
    assert np.all(mat.values >= 0.0)
    assert np.all(mat.values <= 1.0)

    # Asset A & B should have higher tail dependence than Asset A & C
    lambda_ab = mat.loc["ASSET_A", "ASSET_B"]
    lambda_ac = mat.loc["ASSET_A", "ASSET_C"]
    assert lambda_ab > lambda_ac, f"Expected lambda_AB ({lambda_ab}) > lambda_AC ({lambda_ac})"


def test_evaluate_universe_tail_dependence_receipt(synthetic_cocrash_returns_df: pd.DataFrame) -> None:
    """Verify evaluate_universe_tail_dependence generates complete TailDependenceReceipt."""
    with patch("src.analytics.copula.build_aligned_returns_matrix", return_value=synthetic_cocrash_returns_df):
        receipt = evaluate_universe_tail_dependence(tickers=["ASSET_A", "ASSET_B", "ASSET_C"], threshold=0.05)

    assert isinstance(receipt, TailDependenceReceipt)
    assert receipt.sample_sessions == 120
    assert len(receipt.tickers) == 3
    assert receipt.tail_threshold == 0.05
    assert len(receipt.systemic_fragility_scores) == 3
    assert receipt.highest_risk_pair[0] in ["ASSET_A", "ASSET_B"]
    assert receipt.highest_risk_pair[1] in ["ASSET_A", "ASSET_B"]
    assert receipt.market_joint_crash_regime in [
        "LOW_CO_CRASH_RISK",
        "MODERATE_TAIL_DEPENDENCE",
        "HIGH_CONTAGION_RISK",
    ]

    d = receipt.to_dict()
    assert "tail_dependence_matrix" in d
    assert "highest_risk_pair" in d
    assert "summary" in d


def test_evaluate_universe_tail_dependence_insufficient_data() -> None:
    """Verify fallback handling when single asset or empty dataframe is provided."""
    receipt = evaluate_universe_tail_dependence(tickers=["SINGLE_ASSET"])
    assert receipt.market_joint_crash_regime == "INSUFFICIENT_ASSETS"
    assert receipt.sample_sessions == 0
    assert "Insufficient" in receipt.summary


def test_cli_tail_dependence_command_stdout_and_json(synthetic_cocrash_returns_df: pd.DataFrame) -> None:
    """Verify CLI tail-dependence command outputs formatted table and valid JSON."""
    args_json = argparse.Namespace(exchange=None, threshold=0.05, json=True)

    with patch("src.analytics.copula.build_aligned_returns_matrix", return_value=synthetic_cocrash_returns_df), \
         patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        tail_dependence_command(args_json)
        output = mock_out.getvalue().strip()
        parsed = json.loads(output)
        assert "tickers" in parsed
        assert "tail_dependence_matrix" in parsed
        assert "systemic_fragility_scores" in parsed

    args_txt = argparse.Namespace(exchange=None, threshold=0.05, json=False)
    with patch("src.analytics.copula.build_aligned_returns_matrix", return_value=synthetic_cocrash_returns_df), \
         patch("sys.stdout", new_callable=io.StringIO) as mock_out:
        tail_dependence_command(args_txt)
        output = mock_out.getvalue()
        assert "Asymmetric Lower-Tail Dependence & Joint Crash Fragility" in output
        assert "Market Joint Crash Regime:" in output
        assert "Lower-Tail Dependence Matrix" in output
