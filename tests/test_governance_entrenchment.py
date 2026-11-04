"""Unit and integration tests for Paul A. Gompers, Joy L. Ishii & Andrew Metrick (2003) /
Lucian Bebchuk, Alma Cohen & Allen Ferrell (2009) Corporate Governance & Managerial Entrenchment Engine (Iteration 91).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.governance_entrenchment import (
    GovernanceEntrenchmentReceipt,
    calculate_governance_entrenchment,
    classify_governance_regime,
    evaluate_governance_entrenchment_profile,
)
from src.cli import governance_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_shareholder_democracy_compounder_regime() -> None:
    """Verify low E-Index (<= 1), low G-Index (<= 6), and high board independence classify as SHAREHOLDER_DEMOCRACY_COMPOUNDER."""
    rec = calculate_governance_entrenchment(
        ticker="AAPL",
        latest_price=190.0,
        bebchuk_e_index=1,
        gompers_g_index=5,
        dual_class_voting_wedge=0.0,
        board_independence_ratio=0.88,
        unreturned_fcf_ratio=0.15,
    )
    assert rec.other_irrc_provisions == 4
    assert rec.composite_entrenchment_score <= 0.22
    assert rec.regime == "SHAREHOLDER_DEMOCRACY_COMPOUNDER"
    assert rec.is_shareholder_democracy is True
    assert rec.is_managerial_dictatorship_trap is False
    assert rec.expected_governance_alpha_pct > 2.0
    assert rec.verify_integrity() is True


def test_managerial_dictatorship_trap_regime() -> None:
    """Verify high E-Index (>= 5), high G-Index (>= 14), and dual-class voting wedge trigger MANAGERIAL_DICTATORSHIP_TRAP."""
    rec = calculate_governance_entrenchment(
        ticker="META",
        latest_price=550.0,
        bebchuk_e_index=5,
        gompers_g_index=16,
        dual_class_voting_wedge=0.45,
        board_independence_ratio=0.55,
        unreturned_fcf_ratio=0.75,
    )
    assert rec.other_irrc_provisions == 11
    assert rec.composite_entrenchment_score >= 0.72
    assert rec.regime == "MANAGERIAL_DICTATORSHIP_TRAP"
    assert rec.is_managerial_dictatorship_trap is True
    assert rec.is_shareholder_democracy is False
    assert rec.expected_governance_alpha_pct < -5.0
    assert rec.verify_integrity() is True


def test_entrenched_board_agency_watch_regime() -> None:
    """Verify E-Index == 4 with moderate G-Index classifies as ENTRENCHED_BOARD_AGENCY_WATCH."""
    rec = calculate_governance_entrenchment(
        ticker="DIS",
        latest_price=110.0,
        bebchuk_e_index=4,
        gompers_g_index=10,
        dual_class_voting_wedge=0.05,
        board_independence_ratio=0.75,
        unreturned_fcf_ratio=0.35,
    )
    assert rec.composite_entrenchment_score < 0.72
    assert rec.regime == "ENTRENCHED_BOARD_AGENCY_WATCH"
    assert rec.is_managerial_dictatorship_trap is False
    assert rec.is_shareholder_democracy is False


def test_balanced_governance_structure_regime() -> None:
    """Verify intermediate E-Index (2 or 3) and G-Index classify as BALANCED_GOVERNANCE_STRUCTURE."""
    rec = calculate_governance_entrenchment(
        ticker="MSFT",
        latest_price=420.0,
        bebchuk_e_index=2,
        gompers_g_index=8,
        dual_class_voting_wedge=0.0,
        board_independence_ratio=0.82,
        unreturned_fcf_ratio=0.25,
    )
    assert 0.22 < rec.composite_entrenchment_score < 0.55
    assert rec.regime == "BALANCED_GOVERNANCE_STRUCTURE"
    assert rec.is_shareholder_democracy is False
    assert rec.is_managerial_dictatorship_trap is False


def test_jensen_fcf_agency_amplification() -> None:
    """Verify Jensen (1986) unreturned FCF amplifies agency_cost_wedge_index and lowers expected governance alpha."""
    rec_high_payout = calculate_governance_entrenchment(
        ticker="GOOG",
        latest_price=175.0,
        bebchuk_e_index=3,
        gompers_g_index=9,
        dual_class_voting_wedge=0.35,
        unreturned_fcf_ratio=0.10,
    )
    rec_hoarded_fcf = calculate_governance_entrenchment(
        ticker="GOOG",
        latest_price=175.0,
        bebchuk_e_index=3,
        gompers_g_index=9,
        dual_class_voting_wedge=0.35,
        unreturned_fcf_ratio=0.90,
    )
    assert rec_hoarded_fcf.agency_cost_wedge_index > rec_high_payout.agency_cost_wedge_index
    assert rec_hoarded_fcf.expected_governance_alpha_pct < rec_high_payout.expected_governance_alpha_pct


def test_classify_governance_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_governance_regime."""
    assert classify_governance_regime(1, 5, 0.14) == "SHAREHOLDER_DEMOCRACY_COMPOUNDER"
    assert classify_governance_regime(2, 8, 0.32) == "BALANCED_GOVERNANCE_STRUCTURE"
    assert classify_governance_regime(4, 10, 0.58) == "ENTRENCHED_BOARD_AGENCY_WATCH"
    assert classify_governance_regime(5, 15, 0.78) == "MANAGERIAL_DICTATORSHIP_TRAP"


def test_constitutional_g_index_subset_consistency() -> None:
    """Verify g_index < e_index or (g_index - e_index) > 18 raises ValueError."""
    with pytest.raises(ValueError, match="gompers_g_index must be greater than or equal to bebchuk_e_index"):
        calculate_governance_entrenchment("AAPL", 180.0, bebchuk_e_index=4, gompers_g_index=2)
    with pytest.raises(ValueError, match="other_irrc_provisions"):
        calculate_governance_entrenchment("AAPL", 180.0, bebchuk_e_index=1, gompers_g_index=22)


def test_evaluate_governance_entrenchment_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_governance_entrenchment_profile reads closing prices from SQLite."""
    db_file = tmp_path / "governance_test.db"
    db.init_db(db_file)
    db.upsert_stock(
        ticker="AAPL",
        exchange="XNAS",
        company_name="Apple Inc.",
        currency="USD",
        timezone="America/New_York",
        db_path=db_file,
    )
    records = []
    for i in range(25):
        p = 180.0 + i * 0.5
        records.append(
            {
                "ticker": "AAPL",
                "exchange": "XNAS",
                "session_date": f"2026-01-{i+1:02d}",
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "adj_close": p,
                "volume": 50_000_000,
                "data_status": "END_OF_DAY",
                "provider": "MOCK",
            }
        )
    db.save_observations(pd.DataFrame(records), db_path=db_file)

    rec = evaluate_governance_entrenchment_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert abs(rec.latest_price - 192.0) < 1e-2
    assert rec.bebchuk_e_index == 1
    assert rec.gompers_g_index == 5
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, e_index, g_index, voting wedge, board independence, and FCF retention raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_governance_entrenchment("AAPL", 0.0)
    with pytest.raises(ValueError, match="bebchuk_e_index must be an integer in"):
        calculate_governance_entrenchment("AAPL", 150.0, bebchuk_e_index=7, gompers_g_index=10)
    with pytest.raises(ValueError, match="gompers_g_index must be an integer in"):
        calculate_governance_entrenchment("AAPL", 150.0, bebchuk_e_index=5, gompers_g_index=25)
    with pytest.raises(ValueError, match="dual_class_voting_wedge must be in"):
        calculate_governance_entrenchment("AAPL", 150.0, dual_class_voting_wedge=0.99)
    with pytest.raises(ValueError, match="board_independence_ratio must be in"):
        calculate_governance_entrenchment("AAPL", 150.0, board_independence_ratio=1.2)
    with pytest.raises(ValueError, match="unreturned_fcf_ratio must be in"):
        calculate_governance_entrenchment("AAPL", 150.0, unreturned_fcf_ratio=-0.1)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_governance_entrenchment(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_governance_entrenchment(
        ticker="AAPL",
        latest_price=185.0,
        bebchuk_e_index=1,
        gompers_g_index=5,
    )
    assert rec.verify_integrity() is True
    tampered = GovernanceEntrenchmentReceipt(
        **{**rec.to_dict(), "expected_governance_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_governance_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI governance_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="AAPL",
        e_index=1,
        g_index=5,
        dual_class_wedge=0.0,
        board_independence=0.88,
        fcf_retention=0.15,
        json=False,
    )
    governance_command(args_text)
    captured = capsys.readouterr()
    assert "Gompers et al. (2003) & Bebchuk et al. (2009) Governance Engine" in captured.out
    assert "AAPL" in captured.out
    assert "SHAREHOLDER_DEMOCRACY_COMPOUNDER" in captured.out

    args_json = argparse.Namespace(
        ticker="AAPL",
        e_index=1,
        g_index=5,
        dual_class_wedge=0.0,
        board_independence=0.88,
        fcf_retention=0.15,
        json=True,
    )
    governance_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "AAPL"
    assert payload["regime"] == "SHAREHOLDER_DEMOCRACY_COMPOUNDER"
    assert payload["is_shareholder_democracy"] is True
    assert len(payload["signature"]) == 64

