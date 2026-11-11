"""Unit & CLI integration tests for Jeremy C. Stein (1997) Internal Capital Markets Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess

import numpy as np
import pytest

from src.analytics.stein_internal_capital import (
    calculate_stein_internal_capital,
    evaluate_stein_internal_capital_profile,
    persist_stein_internal_capital_record,
    verify_hmac_signature,
)


def test_winner_picking_equalizes_marginal_products_and_conserves_capital():
    """Optimal winner-picking equalizes divisional marginal returns and conserves K_1 + K_2 == C."""
    rec = calculate_stein_internal_capital(
        div1_productivity_theta1=8.0,
        div2_productivity_theta2=4.0,
        div1_standalone_budget_c1=4.0,
        div2_standalone_budget_c2=4.0,
        agency_subsidy_lambda=0.0,
        ticker="WP97",
    )

    assert rec.marginal_product_div1 == pytest.approx(rec.marginal_product_div2, rel=1e-9)
    assert rec.winner_pick_k1 + rec.winner_pick_k2 == pytest.approx(rec.total_available_capital, rel=1e-9)
    assert rec.winner_pick_k1 > rec.div1_standalone_budget_c1
    assert rec.winner_pick_k2 < rec.div2_standalone_budget_c2
    assert rec.winner_pick_total_value > rec.standalone_total_value
    assert rec.gross_winner_picking_surplus > 0.0
    assert rec.regime == "ACTIVE_WINNER_PICKING_CONGLOMERATE_PREMIUM"


def test_widening_productivity_dispersion_raises_winner_picking_premium():
    """Greater spread between divisions amplifies conglomerate reallocation value."""
    narrow = calculate_stein_internal_capital(
        div1_productivity_theta1=6.5,
        div2_productivity_theta2=5.5,
        div1_standalone_budget_c1=5.0,
        div2_standalone_budget_c2=5.0,
        agency_subsidy_lambda=0.0,
    )
    wide = calculate_stein_internal_capital(
        div1_productivity_theta1=9.0,
        div2_productivity_theta2=3.0,
        div1_standalone_budget_c1=5.0,
        div2_standalone_budget_c2=5.0,
        agency_subsidy_lambda=0.0,
    )

    assert wide.gross_winner_picking_surplus > narrow.gross_winner_picking_surplus


def test_corporate_socialism_creates_conglomerate_discount():
    """Excessive internal lobbying/cross-subsidization destroys value vs. standalone focus."""
    rec = calculate_stein_internal_capital(
        div1_productivity_theta1=8.0,
        div2_productivity_theta2=4.0,
        div1_standalone_budget_c1=5.0,
        div2_standalone_budget_c2=5.0,
        agency_subsidy_lambda=0.85,
        ticker="SOC",
    )

    assert rec.agency_socialism_loss > rec.gross_winner_picking_surplus
    assert rec.net_conglomerate_premium < 0.0
    assert rec.conglomerate_premium_pct <= -1.0
    assert rec.regime == "CORPORATE_SOCIALISM_CONGLOMERATE_DISCOUNT"


def test_unconstrained_capital_recovers_standalone_parity():
    """When standalone budgets exceed first-best capacity (theta^2/4), HQ adds zero surplus."""
    rec = calculate_stein_internal_capital(
        div1_productivity_theta1=4.0,   # FB = 4
        div2_productivity_theta2=4.0,   # FB = 4
        div1_standalone_budget_c1=10.0,
        div2_standalone_budget_c2=10.0,
        agency_subsidy_lambda=0.30,
    )

    assert rec.budget_constrained is False
    assert rec.gross_winner_picking_surplus == pytest.approx(0.0, abs=1e-9)
    assert rec.actual_conglomerate_value == pytest.approx(rec.standalone_total_value, rel=1e-9)
    assert rec.regime == "UNCONSTRAINED_FIRST_BEST_PARITY"


def test_symmetric_divisions_yield_fifty_fifty_zero_lobbying_loss():
    """Equal division productivities yield 50/50 split with zero reallocation or socialism wedge."""
    rec = calculate_stein_internal_capital(
        div1_productivity_theta1=6.0,
        div2_productivity_theta2=6.0,
        div1_standalone_budget_c1=4.0,
        div2_standalone_budget_c2=4.0,
        agency_subsidy_lambda=0.75,
    )
    assert rec.winner_pick_k1 == pytest.approx(rec.winner_pick_k2, rel=1e-9)
    assert rec.agency_socialism_loss == pytest.approx(0.0, abs=1e-9)


def test_cross_subsidization_drag_vs_balanced_regimes():
    """Verify moderate cross-subsidization drag classification."""
    rec = calculate_stein_internal_capital(
        div1_productivity_theta1=8.0,
        div2_productivity_theta2=4.0,
        div1_standalone_budget_c1=5.0,
        div2_standalone_budget_c2=5.0,
        agency_subsidy_lambda=0.52,
    )
    assert rec.agency_socialism_loss > rec.gross_winner_picking_surplus
    assert rec.regime in ("CROSS_SUBSIDIZATION_VALUE_DRAG", "CORPORATE_SOCIALISM_CONGLOMERATE_DISCOUNT")


def test_security_input_validation():
    """Verify strict ticker allowlist, positive productivity, and unit agency bounds."""
    with pytest.raises(ValueError, match="Invalid ticker symbol"):
        calculate_stein_internal_capital(ticker="ST97;DROP")

    with pytest.raises(ValueError, match="must be positive"):
        calculate_stein_internal_capital(div1_productivity_theta1=0.0)

    with pytest.raises(ValueError, match="total capital positive"):
        calculate_stein_internal_capital(div1_standalone_budget_c1=0.0, div2_standalone_budget_c2=0.0)

    with pytest.raises(ValueError, match="agency_subsidy_lambda must satisfy"):
        calculate_stein_internal_capital(agency_subsidy_lambda=1.5)


def test_hmac_cryptographic_sealing():
    """Verify HMAC-SHA256 generation and tampering rejection."""
    rec = calculate_stein_internal_capital(ticker="BRK")
    sig_payload = (
        f"{rec.ticker}:{rec.div1_productivity_theta1:.4f}:{rec.div2_productivity_theta2:.4f}:"
        f"{rec.div1_standalone_budget_c1:.4f}:{rec.div2_standalone_budget_c2:.4f}:"
        f"{rec.agency_subsidy_lambda:.4f}:{rec.winner_pick_total_value:.6f}:"
        f"{rec.actual_conglomerate_value:.6f}:{rec.conglomerate_premium_pct:.4f}:"
        f"{rec.regime}:{rec.timestamp}"
    )

    assert verify_hmac_signature(sig_payload, rec.hmac_signature) is True
    assert verify_hmac_signature(sig_payload + "_tampered", rec.hmac_signature) is False


def test_evaluate_stein_internal_capital_profile_from_prices():
    """Verify trailing volatility-based divisional dispersion calibration."""
    prices = [100.0 + 0.3 * i + 2.0 * np.sin(i / 2.2) for i in range(40)]
    rec = evaluate_stein_internal_capital_profile(prices=prices, ticker="GE")

    assert rec.ticker == "GE"
    assert rec.winner_pick_k1 > rec.winner_pick_k2


def test_persist_stein_internal_capital_record_sqlite(tmp_path):
    """Verify SQLite persistence using parameterized queries and FK enforcement."""
    db_file = str(tmp_path / "test_stein_internal.db")
    rec = calculate_stein_internal_capital(ticker="HON")

    assert persist_stein_internal_capital_record(rec, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, winner_pick_k1, net_conglomerate_premium, regime FROM stein_internal_capital_audits WHERE ticker = ?",
            ("HON",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "HON"
        assert row[1] == pytest.approx(rec.winner_pick_k1, rel=1e-4)
        assert row[2] == pytest.approx(rec.net_conglomerate_premium, rel=1e-4)
        assert row[3] == rec.regime


def test_cli_internal_capital_table_execution():
    """Verify CLI execution for internal-capital command in formatted table mode."""
    cmd_table = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "internal-capital",
        "--ticker",
        "ST97",
        "--theta1",
        "8.5",
        "--theta2",
        "4.0",
    ]
    res = subprocess.run(cmd_table, capture_output=True, text=True, check=True)
    assert "Stein (1997) Internal Capital Markets & Winner-Picking Engine" in res.stdout
    assert "ST97" in res.stdout
    assert "Conglomerate Premium" in res.stdout


def test_cli_internal_capital_json_execution():
    """Verify CLI execution for internal-capital command with --json flag."""
    cmd_json = [
        ".venv/bin/python3",
        "-m",
        "src.cli",
        "internal-capital",
        "--ticker",
        "DHR",
        "--theta1",
        "9.0",
        "--theta2",
        "3.5",
        "--agency-lambda",
        "0.10",
        "--json",
    ]
    res = subprocess.run(cmd_json, capture_output=True, text=True, check=True)
    parsed = json.loads(res.stdout)
    assert parsed["ticker"] == "DHR"
    assert parsed["net_conglomerate_premium"] > 0.0
    assert "hmac_signature" in parsed

