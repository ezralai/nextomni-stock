"""Tests for Ho (1992) Key-Rate Duration & Fisher-Weil (1971) Immunization."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.key_rate_immunization import (
    _build_hmac_payload,
    evaluate_krd_profile,
    key_rate_tent_weights,
    persist_krd_record,
    solve_key_rate_immunization,
    verify_hmac_signature,
)


def test_krd_partition_of_unity_and_barbell_convexity_surplus() -> None:
    """Ho (1992) tents sum to 1.0 everywhere and barbell immunization yields convexity surplus."""
    t_grid = np.linspace(0.25, 35.0, 50)
    tents = key_rate_tent_weights(t_grid)
    assert np.allclose(np.sum(tents, axis=0), 1.0, atol=1e-12)

    rec = solve_key_rate_immunization(
        liability_maturity=10.0,
        liability_amount=1000.0,
        mode="barbell",
        ticker="PENSION",
    )
    assert rec.asset_duration == pytest.approx(rec.liability_duration, abs=1e-4)
    assert rec.partition_error < 1e-3
    assert rec.convexity_surplus > 0.0


def test_ladder_minimizes_twist_gap_for_immunization() -> None:
    """Ladder achieves smaller KRD twist L2 gap than pure 2Y/30Y barbell."""
    barbell = solve_key_rate_immunization(liability_maturity=10.0, mode="barbell", ticker="ALM")
    ladder = solve_key_rate_immunization(liability_maturity=10.0, mode="ladder", ticker="ALM")
    assert ladder.krd_twist_l2_gap < barbell.krd_twist_l2_gap
    assert ladder.asset_duration == pytest.approx(ladder.liability_duration, abs=1e-4)


def test_krd_validation_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Validate input guards, HMAC-SHA256 verification, and SQLite audit persistence."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        solve_key_rate_immunization(ticker="BAD TICKER")

    with pytest.raises(ValueError, match="liability_maturity"):
        solve_key_rate_immunization(liability_maturity=0.5, ticker="ALM")

    prices = 100.0 * np.exp(np.cumsum(np.full(60, 0.001)))
    receipt = evaluate_krd_profile(prices, ticker="SPY", liability_maturity=10.0)
    payload = _build_hmac_payload(
        ticker=receipt.ticker,
        liability_maturity=receipt.liability_maturity,
        liability_pv=receipt.liability_pv,
        liability_duration=receipt.liability_duration,
        asset_convexity=receipt.asset_convexity,
        convexity_surplus=receipt.convexity_surplus,
        krd_twist_l2_gap=receipt.krd_twist_l2_gap,
        regime=receipt.regime,
        timestamp=receipt.timestamp,
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True
    assert verify_hmac_signature(payload + "|tampered", receipt.hmac_signature) is False

    db_file = tmp_path / "krd_test.db"
    assert persist_krd_record(receipt, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, liability_maturity, regime, hmac_signature FROM key_rate_immunization_records"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "SPY"
    assert rows[0][1] == 10.0
    assert rows[0][3] == receipt.hmac_signature


def test_cli_key_rate_duration_json_and_aliases() -> None:
    """Verify CLI key-rate-duration and aliases emit valid JSON receipt."""
    for alias in ("key-rate-duration", "krd-1992", "fisher-weil", "immunization"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "ALM",
                "--liab-maturity",
                "10",
                "--mode",
                "ladder",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "ALM"
        assert data["partition_error"] < 1e-3
        assert len(data["hmac_signature"]) == 64

