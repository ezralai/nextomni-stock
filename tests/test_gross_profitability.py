"""Unit and integration test suite for Novy-Marx (2013) Gross Profitability & QMJ Engine."""

import argparse
import io
import json
import math
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.analytics.gross_profitability import (
    GrossProfitabilityReceipt,
    calculate_gross_profitability,
    classify_quality_regime,
    evaluate_gross_profitability_profile,
)
from src.cli import profitability_command


def test_gross_profitability_prime_compounder() -> None:
    """Verify high-productivity quality compounder profile with robust profitability and modest leverage."""
    # Firm: Rev = $380B, COGS = $210B (GP = $170B), Assets = $350B, Debt = $100B
    rev = 3.8e11
    cogs = 2.1e11
    assets = 3.5e11
    debt = 1.0e11

    (
        gp,
        gp_a_pct,
        gm_pct,
        ato,
        debt_to_assets,
        quality_score,
        is_value_trap,
        expected_alpha,
        regime,
    ) = calculate_gross_profitability(
        revenue=rev,
        cogs=cogs,
        total_assets=assets,
        total_debt=debt,
    )

    # Gross Profit = 170B
    assert gp == pytest.approx(1.7e11, rel=1e-5)
    # GP/A = 170 / 350 = 48.57%
    assert gp_a_pct == pytest.approx(48.57, abs=0.1)
    # Gross Margin = 170 / 380 = 44.74%
    assert gm_pct == pytest.approx(44.74, abs=0.1)
    # Asset turnover = 380 / 350 = 1.09x
    assert ato == pytest.approx(1.09, abs=0.05)
    # Leverage = 100 / 350 = 28.57%
    assert debt_to_assets == pytest.approx(28.57, abs=0.1)

    # Quality assessments
    assert quality_score >= 70.0
    assert is_value_trap is False
    assert expected_alpha > 0.0
    assert regime in ("PRIME_QUALITY_COMPOUNDER", "PROFITABLE_GROWTH")


def test_gross_profitability_value_trap() -> None:
    """Verify detection of unprofitable value trap with low GP/A and high debt burden."""
    # Impaired firm: Rev = $100M, COGS = $95M (GP = $5M), Assets = $100M, Debt = $80M
    rev = 1.0e8
    cogs = 9.5e7
    assets = 1.0e8
    debt = 8.0e7

    (
        gp,
        gp_a_pct,
        gm_pct,
        ato,
        debt_to_assets,
        quality_score,
        is_value_trap,
        expected_alpha,
        regime,
    ) = calculate_gross_profitability(
        revenue=rev,
        cogs=cogs,
        total_assets=assets,
        total_debt=debt,
    )

    assert gp == 5.0e6
    assert gp_a_pct == 5.0
    assert debt_to_assets == 80.0
    assert is_value_trap is True
    assert quality_score < 40.0
    assert expected_alpha < 0.0
    assert regime == "UNPROFITABLE_VALUE_TRAP"


def test_gross_profitability_distressed_junk_negative_profit() -> None:
    """Verify handling of cash-burning company with negative gross profit."""
    rev = 5.0e7
    cogs = 7.0e7  # COGS > Revenue -> Negative Gross Profit
    assets = 1.0e8
    debt = 2.0e7

    (
        gp,
        gp_a_pct,
        _,
        _,
        _,
        quality_score,
        is_value_trap,
        expected_alpha,
        regime,
    ) = calculate_gross_profitability(
        revenue=rev,
        cogs=cogs,
        total_assets=assets,
        total_debt=debt,
    )

    assert gp < 0.0
    assert gp_a_pct < 0.0
    assert is_value_trap is True
    assert quality_score < 30.0
    assert expected_alpha < -2.0
    assert regime in ("UNPROFITABLE_VALUE_TRAP", "DISTRESSED_JUNK")


def test_quality_score_monotonicity() -> None:
    """Verify that increasing gross profit holding assets constant monotonically improves quality score."""
    assets = 1.0e8
    debt = 2.0e7
    cogs = 5.0e7

    revenues = [6.0e7, 8.0e7, 1.2e8, 1.8e8]
    scores: list[float] = []
    gpas: list[float] = []
    alphas: list[float] = []

    for rev in revenues:
        _, gp_a, _, _, _, q_score, _, exp_a, _ = calculate_gross_profitability(
            revenue=rev,
            cogs=cogs,
            total_assets=assets,
            total_debt=debt,
        )
        scores.append(q_score)
        gpas.append(gp_a)
        alphas.append(exp_a)

    for i in range(len(revenues) - 1):
        assert gpas[i] < gpas[i + 1]
        assert scores[i] < scores[i + 1]
        assert alphas[i] <= alphas[i + 1]


def test_gross_profitability_regime_classification() -> None:
    """Verify regime tier boundaries."""
    assert classify_quality_regime(80.0, 35.0, False) == "PRIME_QUALITY_COMPOUNDER"
    assert classify_quality_regime(65.0, 25.0, False) == "PROFITABLE_GROWTH"
    assert classify_quality_regime(45.0, 15.0, False) == "MEDIOCRE_NEUTRAL"
    assert classify_quality_regime(30.0, 5.0, True) == "UNPROFITABLE_VALUE_TRAP"
    assert classify_quality_regime(30.0, 5.0, False) == "DISTRESSED_JUNK"


def test_gross_profitability_receipt_tamper_integrity() -> None:
    """Verify HMAC-SHA256 signature verification and tamper rejection."""
    receipt = evaluate_gross_profitability_profile(
        ticker="AAPL",
        revenue=3.8e11,
        cogs=2.1e11,
        total_assets=3.5e11,
        total_debt=1.0e11,
    )

    assert receipt.ticker == "AAPL"
    assert receipt.verify_integrity() is True
    assert len(receipt.signature) == 64

    # Tampering with quality score invalidates signature
    receipt.quality_score += 5.0
    assert receipt.verify_integrity() is False


def test_profitability_cli_command_plain(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with formatted terminal report."""
    args = argparse.Namespace(
        command="profitability",
        ticker="MSFT",
        revenue=2.2e11,
        cogs=6.8e10,
        assets=4.1e11,
        debt=8.0e10,
        json=False,
    )

    profitability_command(args)
    captured = capsys.readouterr().out

    assert "Novy-Marx (2013) Gross Profitability & QMJ Engine" in captured
    assert "Security / Ticker:            MSFT" in captured
    assert "Gross Profitability (GP/A):" in captured
    assert "Quality Score (Q):" in captured
    assert "Audit Digest (HMAC):" in captured
    assert "Verified: True" in captured


def test_profitability_cli_command_json(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI command invocation with JSON output."""
    args = argparse.Namespace(
        command="profitability",
        ticker="GOOGL",
        revenue=3.0e11,
        cogs=1.3e11,
        assets=3.9e11,
        debt=3.0e10,
        json=True,
    )

    profitability_command(args)
    captured = capsys.readouterr().out
    data = json.loads(captured)

    assert data["ticker"] == "GOOGL"
    assert "gross_profitability_pct" in data
    assert "quality_score" in data
    assert "is_value_trap" in data
    assert "signature" in data


def test_profitability_input_validation() -> None:
    """Verify input validation guards against invalid parameters."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        evaluate_gross_profitability_profile(ticker="BAD$$$")

    with pytest.raises(ValueError, match="total_assets must be strictly positive"):
        calculate_gross_profitability(revenue=1e8, cogs=5e7, total_assets=-1e6)

    with pytest.raises(ValueError, match="revenue cannot be negative"):
        calculate_gross_profitability(revenue=-1e6, cogs=5e7, total_assets=1e8)
