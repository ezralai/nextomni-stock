"""Unit & CLI integration tests for Black & Cox (1976) First-Passage Covenant Credit Engine."""

from __future__ import annotations

import json
import math
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.black_cox_covenants import (
    calculate_black_cox_covenants,
    evaluate_black_cox_profile,
    persist_black_cox_record,
    verify_hmac_signature,
)


def test_zero_covenant_recovers_exact_merton_benchmark():
    """As safety covenant K -> 0, Black-Cox equity, debt, spread, and PD match Merton (1974)."""
    rec = calculate_black_cox_covenants(
        asset_value_v0=140.0,
        debt_face_f=100.0,
        covenant_level_k=0.0,
        maturity_years=5.0,
        asset_volatility=0.30,
        ticker="MERT",
    )

    assert rec.black_cox_equity_value == pytest.approx(rec.merton_equity_value, rel=1e-9)
    assert rec.black_cox_debt_value == pytest.approx(rec.merton_debt_value, rel=1e-9)
    assert rec.black_cox_credit_spread_bps == pytest.approx(rec.merton_credit_spread_bps, rel=1e-9)
    assert rec.first_passage_barrier_pd == pytest.approx(0.0, abs=1e-12)
    assert rec.total_default_prob == pytest.approx(rec.merton_terminal_pd, rel=1e-9)


def test_safety_covenant_transfers_value_from_equity_to_debt():
    """Tighter safety covenants protect bondholders: D_BC >= D_Merton and s_BC <= s_Merton."""
    loose = calculate_black_cox_covenants(
        asset_value_v0=130.0,
        debt_face_f=100.0,
        covenant_level_k=50.0,
        covenant_growth_gamma=0.02,
        maturity_years=5.0,
        asset_volatility=0.32,
        ticker="LOOSE",
    )
    tight = calculate_black_cox_covenants(
        asset_value_v0=130.0,
        debt_face_f=100.0,
        covenant_level_k=95.0,
        covenant_growth_gamma=0.02,
        maturity_years=5.0,
        asset_volatility=0.32,
        ticker="TIGHT",
    )

    assert tight.black_cox_debt_value > loose.black_cox_debt_value > tight.merton_debt_value
    assert tight.black_cox_credit_spread_bps < loose.black_cox_credit_spread_bps < tight.merton_credit_spread_bps
    assert tight.covenant_equity_transfer > loose.covenant_equity_transfer > 0.0
    assert tight.covenant_spread_savings_bps > 0.0


def test_first_passage_total_pd_dominates_terminal_merton_pd():
    """Total default probability under first-passage monitoring must always >= terminal Merton PD."""
    rec = calculate_black_cox_covenants(
        asset_value_v0=120.0,
        debt_face_f=100.0,
        covenant_level_k=90.0,
        covenant_growth_gamma=0.01,
        maturity_years=5.0,
        asset_volatility=0.35,
        ticker="FPPD",
    )

    assert rec.total_default_prob >= rec.merton_terminal_pd
    assert rec.total_default_prob >= rec.first_passage_barrier_pd
    assert rec.pre_maturity_default_share > 0.0
    assert 0.0 < rec.first_passage_barrier_pd < 1.0


def test_down_and_out_boundary_continuity_at_barrier():
    """Verify equity smoothly approaches 0 near the barrier and triggers technical breach at V0 <= H0."""
    near_barrier = calculate_black_cox_covenants(
        asset_value_v0=86.1,
        debt_face_f=100.0,
        covenant_level_k=95.0,
        covenant_growth_gamma=0.02,
        maturity_years=5.0,
        asset_volatility=0.28,
        ticker="NEAR",
    )
    # H0 = 95 * exp(-0.10) = 85.9595...
    assert near_barrier.covenant_breached_at_inception is False
    assert near_barrier.black_cox_equity_value > 0.0
    assert near_barrier.black_cox_equity_value < 2.0

    breached = calculate_black_cox_covenants(
        asset_value_v0=85.0,
        debt_face_f=100.0,
        covenant_level_k=95.0,
        covenant_growth_gamma=0.02,
        maturity_years=5.0,
        asset_volatility=0.28,
        ticker="BRCH",
    )
    assert breached.covenant_breached_at_inception is True
    assert breached.black_cox_equity_value == 0.0
    assert breached.total_default_prob == 1.0
    assert breached.regime == "TECHNICAL_COVENANT_BREACH_DEFAULT"


def test_log_space_reflection_no_overflow_at_extreme_low_vol():
    """Verify no NaN/Overflow when asset/barrier ratio is high and volatility is tiny (0*inf hazard)."""
    rec = calculate_black_cox_covenants(
        asset_value_v0=500.0,
        debt_face_f=100.0,
        covenant_level_k=80.0,
        covenant_growth_gamma=0.02,
        maturity_years=5.0,
        asset_volatility=0.05,
        ticker="LOWV",
    )

    assert math.isfinite(rec.black_cox_equity_value)
    assert math.isfinite(rec.first_passage_barrier_pd)
    assert math.isfinite(rec.total_default_prob)
    assert rec.total_default_prob == pytest.approx(0.0, abs=1e-9)
    assert rec.regime == "INVESTMENT_GRADE_COVENANT_PROTECTED"


def test_security_input_validation():
    """Verify strict ticker allowlist and boundary checks reject invalid/injection payloads."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_black_cox_covenants(ticker="CORP;DROP")

    with pytest.raises(ValueError, match="cannot exceed debt_face_f"):
        calculate_black_cox_covenants(debt_face_f=100.0, covenant_level_k=110.0)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_black_cox_covenants(asset_value_v0=-50.0)

    with pytest.raises(ValueError, match="strictly positive"):
        calculate_black_cox_covenants(asset_volatility=0.0)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_black_cox_covenants(ticker="F")
    sig_payload = (
        f"{rec.ticker}:{rec.asset_value_v0:.4f}:{rec.debt_face_f:.4f}:"
        f"{rec.covenant_level_k:.4f}:{rec.covenant_growth_gamma:.4f}:"
        f"{rec.maturity_years:.4f}:{rec.asset_volatility:.4f}:"
        f"{rec.black_cox_equity_value:.6f}:{rec.black_cox_credit_spread_bps:.2f}:"
        f"{rec.total_default_prob:.6f}:{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_black_cox_profile_from_prices():
    """Verify empirical de-levered volatility calibration from price series."""
    prices = [120.0 + 0.4 * i + 2.5 * np.sin(i / 2.5) for i in range(45)]
    rec = evaluate_black_cox_profile(
        prices=prices,
        ticker="AAPL",
        leverage_ratio=0.60,
        covenant_coverage=0.80,
    )

    assert rec.ticker == "AAPL"
    assert rec.asset_value_v0 == pytest.approx(prices[-1], rel=1e-6)
    assert rec.black_cox_debt_value > 0.0
    assert 0.0 <= rec.total_default_prob <= 1.0


def test_persist_black_cox_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized placeholders and FK pragma."""
    db_file = str(tmp_path / "test_black_cox.db")
    rec = calculate_black_cox_covenants(ticker="IBM")

    assert persist_black_cox_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, black_cox_credit_spread_bps, total_default_prob, regime FROM black_cox_covenant_audits WHERE ticker = ?",
            ("IBM",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "IBM"
        assert row[1] == pytest.approx(rec.black_cox_credit_spread_bps, rel=1e-4)
        assert row[2] == pytest.approx(rec.total_default_prob, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_black_cox_table_execution():
    """Verify CLI execution for black-cox command in table output mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "black-cox",
        "--ticker",
        "CORP",
        "--v0",
        "140",
        "--debt",
        "100",
        "--covenant-k",
        "85",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Black & Cox (1976) First-Passage Structural Credit & Safety Covenant Engine" in res.stdout
    assert "CORP" in res.stdout
    assert "Covenant Spread Savings" in res.stdout


def test_cli_black_cox_json_execution():
    """Verify CLI execution for black-cox command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "black-cox",
        "--ticker",
        "BA",
        "--v0",
        "130",
        "--debt",
        "100",
        "--covenant-k",
        "90",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "BA"
    assert parsed["covenant_level_k"] == pytest.approx(90.0)
    assert parsed["total_default_prob"] >= parsed["merton_terminal_pd"]
    assert "hmac_signature" in parsed


def test_asset_volatility_monotonically_increases_first_passage_pd():
    """Higher asset volatility increases first-passage barrier hitting probability and widens spreads."""
    low_vol = calculate_black_cox_covenants(asset_volatility=0.18, asset_value_v0=130.0, debt_face_f=100.0, covenant_level_k=80.0)
    high_vol = calculate_black_cox_covenants(asset_volatility=0.42, asset_value_v0=130.0, debt_face_f=100.0, covenant_level_k=80.0)

    assert high_vol.first_passage_barrier_pd > low_vol.first_passage_barrier_pd
    assert high_vol.total_default_prob > low_vol.total_default_prob
    assert high_vol.black_cox_credit_spread_bps > low_vol.black_cox_credit_spread_bps

