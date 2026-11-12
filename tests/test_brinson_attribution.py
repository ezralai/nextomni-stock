"""Tests for Brinson, Hood & Beebower (1986) & Brinson & Fachler (1985) Return Attribution."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import pytest

from src.analytics.brinson_attribution import (
    _build_hmac_payload,
    calculate_brinson_attribution,
    evaluate_brinson_profile,
    persist_brinson_record,
    verify_hmac_signature,
)


def test_exact_attribution_additivity_and_selection_alpha() -> None:
    """Allocation + Selection + Interaction must sum exactly to R_p - R_b."""
    wp = [0.50, 0.30, 0.20]
    wb = [0.40, 0.40, 0.20]
    rp = [0.14, 0.08, 0.05]
    rb = [0.10, 0.06, 0.05]
    rec = calculate_brinson_attribution(
        wp,
        wb,
        rp,
        rb,
        sector_names=["Tech", "Industrials", "Utilities"],
        ticker="SPY",
    )
    reconciled = rec.allocation_effect + rec.selection_effect + rec.interaction_effect
    assert reconciled == pytest.approx(rec.active_return, abs=1e-12)
    assert rec.portfolio_return == pytest.approx(0.104, abs=1e-9)
    assert rec.benchmark_return == pytest.approx(0.074, abs=1e-9)
    assert rec.active_return == pytest.approx(0.030, abs=1e-9)
    assert rec.top_driver_sector == "Tech"
    assert rec.regime == "SELECTION_DRIVEN_ALPHA"


def test_pure_allocation_and_bear_market_fachler_sign() -> None:
    """Matched sector returns yield pure allocation; bear market credits relative outperformer."""
    wb = [0.50, 0.50]
    wp = [0.70, 0.30]
    rb = [-0.02, -0.14]  # Sector 1 fell less than benchmark average (-0.08) -> positive BF alloc!
    rp = [-0.02, -0.14]
    rec = calculate_brinson_attribution(wp, wb, rp, rb, sector_names=["Defensive", "Cyclical"])
    assert rec.selection_effect == pytest.approx(0.0, abs=1e-12)
    assert rec.interaction_effect == pytest.approx(0.0, abs=1e-12)
    assert rec.sector_allocation_effects[0] > 0.0
    assert rec.sector_allocation_effects[1] > 0.0
    assert rec.allocation_effect == pytest.approx(rec.active_return, abs=1e-12)
    assert rec.regime == "ALLOCATION_DRIVEN_ALPHA"


def test_underperformance_matched_tracking_and_empirical_profile() -> None:
    """Verify UNDERPERFORMANCE, MATCHED_TRACKING regimes and price-profile helper."""
    rec_match = calculate_brinson_attribution([0.5, 0.5], [0.5, 0.5], [0.06, 0.04], [0.06, 0.04])
    assert rec_match.regime == "MATCHED_BENCHMARK_TRACKING"

    rec_drag = calculate_brinson_attribution([0.5, 0.5], [0.5, 0.5], [0.02, 0.03], [0.06, 0.05])
    assert rec_drag.active_return < 0.0
    assert rec_drag.regime == "ACTIVE_UNDERPERFORMANCE_DRAG"

    emp = evaluate_brinson_profile([100.0 + i * 0.5 for i in range(40)], ticker="QQQ")
    assert emp.n_sectors == 4
    assert emp.regime in ("SELECTION_DRIVEN_ALPHA", "ALLOCATION_DRIVEN_ALPHA", "ACTIVE_UNDERPERFORMANCE_DRAG", "MATCHED_BENCHMARK_TRACKING")


def test_validation_guards_hmac_and_sqlite_persistence(tmp_path) -> None:
    """Guard bad weights/tickers/lengths, verify HMAC seal, and persist to SQLite."""
    with pytest.raises(ValueError, match="Invalid ticker"):
        calculate_brinson_attribution([0.5, 0.5], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1], ticker="../BAD")
    with pytest.raises(ValueError, match="sum to 1.0"):
        calculate_brinson_attribution([0.6, 0.6], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1])
    with pytest.raises(ValueError, match="non-negative"):
        calculate_brinson_attribution([1.2, -0.2], [0.5, 0.5], [0.1, 0.1], [0.1, 0.1])

    rec = calculate_brinson_attribution([0.6, 0.4], [0.5, 0.5], [0.12, 0.06], [0.10, 0.05], ticker="AAPL")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_sectors=rec.n_sectors,
        portfolio_return=rec.portfolio_return,
        benchmark_return=rec.benchmark_return,
        active_return=rec.active_return,
        allocation_effect=rec.allocation_effect,
        selection_effect=rec.selection_effect,
        interaction_effect=rec.interaction_effect,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("AAPL", "MSFT"), rec.hmac_signature) is False

    db_file = str(tmp_path / "brinson_test.db")
    assert persist_brinson_record(rec, db_path=db_file) is True
    with sqlite3.connect(db_file) as conn:
        rows = conn.execute("SELECT ticker, n_sectors, regime FROM brinson_attribution_audits").fetchall()
    assert rows == [("AAPL", 2, rec.regime)]


def test_cli_brinson_attribution_json_and_aliases() -> None:
    """Verify CLI brinson-attribution and aliases emit valid JSON receipt."""
    for alias in ("brinson-attribution", "bhb-1986", "brinson-fachler"):
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "src.cli",
                alias,
                "--ticker",
                "SPY",
                "--active-alpha",
                "0.03",
                "--json",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        assert data["ticker"] == "SPY"
        assert data["n_sectors"] == 4
        reconciled = data["allocation_effect"] + data["selection_effect"] + data["interaction_effect"]
        assert reconciled == pytest.approx(data["active_return"], abs=1e-9)

