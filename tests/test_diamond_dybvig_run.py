"""Unit, boundary, security, and integration tests for Douglas W. Diamond & Philip H. Dybvig (1983)

Bank Runs, Liquidity Transformation & Sequential Service Engine.
"""

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from src.analytics.diamond_dybvig_run import (
    DiamondDybvigReceipt,
    calculate_diamond_dybvig_run,
    classify_diamond_dybvig_regime,
    evaluate_diamond_dybvig_profile,
    persist_diamond_dybvig_record,
    verify_diamond_dybvig_audit_digest,
)
from src.cli import diamond_dybvig_command
from src.security.validators import SecurityValidationError


def test_calculate_diamond_dybvig_normal_stability() -> None:
    """Verify low early withdrawals within cash reserves maintain normal stability."""
    receipt = calculate_diamond_dybvig_run(
        p_t=100.0,
        adv=1_000_000.0,
        liquid_reserve_ratio=0.20,
        early_withdrawal_ratio=0.10,
        r1_payout=1.05,
        R2_yield=1.35,
        salvage_L=0.65,
        ticker="STABLE",
    )

    assert receipt.ticker == "STABLE"
    assert receipt.regime == "NORMAL_STABILITY"
    assert receipt.is_normal_stability is True
    assert receipt.early_liquidation_required == 0.0
    assert receipt.fire_sale_loss_fraction == 0.0
    assert receipt.run_hazard_probability == 0.0
    assert receipt.patient_t2_expected_payoff > receipt.r1_payout
    assert verify_diamond_dybvig_audit_digest(receipt) is True


def test_calculate_diamond_dybvig_elevated_run_hazard() -> None:
    """Verify withdrawals exceeding liquid reserves force premature salvage sales."""
    receipt = calculate_diamond_dybvig_run(
        p_t=120.0,
        adv=500_000.0,
        liquid_reserve_ratio=0.15,
        early_withdrawal_ratio=0.30,
        r1_payout=1.05,
        R2_yield=1.35,
        salvage_L=0.65,
        ticker="ELEVAT",
    )

    assert receipt.regime == "ELEVATED_RUN_HAZARD"
    assert receipt.is_elevated_run_hazard is True
    assert receipt.early_liquidation_required > 0.0
    assert receipt.fire_sale_loss_fraction > 0.0
    assert receipt.distressed_asset_price < receipt.latest_price
    assert verify_diamond_dybvig_audit_digest(receipt) is True


def test_calculate_diamond_dybvig_systemic_coordination_run() -> None:
    """Verify severe withdrawal demand exceeding exhaustion threshold triggers systemic coordination run."""
    receipt = calculate_diamond_dybvig_run(
        p_t=50.0,
        adv=2_000_000.0,
        liquid_reserve_ratio=0.15,
        early_withdrawal_ratio=0.75,
        r1_payout=1.05,
        R2_yield=1.35,
        salvage_L=0.65,
        ticker="PANIC",
    )

    assert receipt.regime == "SYSTEMIC_COORDINATION_RUN"
    assert receipt.is_systemic_coordination_run is True
    assert receipt.run_hazard_probability == 1.0
    assert receipt.patient_t2_expected_payoff == 0.0
    assert receipt.early_withdrawal_ratio > receipt.exhaustion_threshold_fbar
    assert receipt.distressed_asset_price < receipt.latest_price * 0.50
    assert verify_diamond_dybvig_audit_digest(receipt) is True


def test_calculate_diamond_dybvig_insured_safety() -> None:
    """Verify government deposit insurance eliminates run coordination failure."""
    receipt = calculate_diamond_dybvig_run(
        p_t=100.0,
        adv=1_000_000.0,
        liquid_reserve_ratio=0.15,
        early_withdrawal_ratio=0.80,
        deposit_insurance=True,
        ticker="INSUR",
    )

    assert receipt.regime == "INSURED_SAFETY"
    assert receipt.is_insured_safety is True
    assert receipt.deposit_insurance_active is True
    assert receipt.run_hazard_probability == 0.0
    assert verify_diamond_dybvig_audit_digest(receipt) is True


def test_mathematical_monotonicity_properties() -> None:
    """Verify fundamental monotonicity: in liquidation zone, higher withdrawals reduce patient payoff,

    while fire-sale losses and run hazard increase monotonically across all withdrawal fractions.
    """
    withdrawals = [0.15, 0.25, 0.40, 0.55, 0.70]
    payoffs = []
    losses = []
    run_hazards = []

    for f1 in withdrawals:
        rec = calculate_diamond_dybvig_run(
            p_t=100.0,
            adv=1_000_000.0,
            liquid_reserve_ratio=0.15,
            early_withdrawal_ratio=f1,
            r1_payout=1.05,
            R2_yield=1.35,
            salvage_L=0.65,
        )
        payoffs.append(rec.patient_t2_expected_payoff)
        losses.append(rec.fire_sale_loss_fraction)
        run_hazards.append(rec.run_hazard_probability)

    # In liquidation zone (f1 >= c/r1), patient expected payoff strictly decreases or hits 0
    for i in range(len(payoffs) - 1):
        assert payoffs[i] >= payoffs[i + 1]

    # Fire-sale loss must weakly increase
    for i in range(len(losses) - 1):
        assert losses[i] <= losses[i + 1]

    # Run hazard must weakly increase
    for i in range(len(run_hazards) - 1):
        assert run_hazards[i] <= run_hazards[i + 1]


def test_boundary_conditions() -> None:
    """Verify extreme boundaries f_1 = 0.0 and f_1 = 1.0 behave cleanly."""
    rec_zero = calculate_diamond_dybvig_run(
        p_t=100.0,
        adv=1_000_000.0,
        liquid_reserve_ratio=0.20,
        early_withdrawal_ratio=0.0,
        r1_payout=1.05,
        R2_yield=1.35,
    )
    assert rec_zero.run_hazard_probability == 0.0
    assert rec_zero.patient_t2_expected_payoff > 1.25

    rec_full = calculate_diamond_dybvig_run(
        p_t=100.0,
        adv=1_000_000.0,
        liquid_reserve_ratio=0.20,
        early_withdrawal_ratio=1.0,
        r1_payout=1.05,
        R2_yield=1.35,
    )
    assert rec_full.run_hazard_probability == 1.0
    assert rec_full.patient_t2_expected_payoff == 0.0
    assert rec_full.regime == "SYSTEMIC_COORDINATION_RUN"


def test_audit_digest_hmac_tamper_detection() -> None:
    """Verify cryptographic HMAC-SHA256 signature detects tampering."""
    receipt = calculate_diamond_dybvig_run(
        p_t=150.0,
        adv=1_000_000.0,
        liquid_reserve_ratio=0.20,
        early_withdrawal_ratio=0.25,
        ticker="AUTHEN",
    )
    assert verify_diamond_dybvig_audit_digest(receipt) is True

    # Tamper with run hazard probability
    tampered_dict = receipt.to_dict()
    tampered_dict["run_hazard_probability"] = 0.9999
    tampered_receipt = DiamondDybvigReceipt(**tampered_dict)
    assert verify_diamond_dybvig_audit_digest(tampered_receipt) is False

    # Tamper with price
    tampered_dict2 = receipt.to_dict()
    tampered_dict2["latest_price"] = 999.99
    tampered_receipt2 = DiamondDybvigReceipt(**tampered_dict2)
    assert verify_diamond_dybvig_audit_digest(tampered_receipt2) is False


def test_invalid_inputs_raise_errors() -> None:
    """Verify invalid numerical inputs and ticker formats raise appropriate errors."""
    with pytest.raises(ValueError):
        calculate_diamond_dybvig_run(p_t=-10.0, adv=1_000_000.0)

    with pytest.raises(ValueError):
        calculate_diamond_dybvig_run(p_t=100.0, adv=-500.0)

    with pytest.raises(ValueError):
        calculate_diamond_dybvig_run(p_t=float("nan"), adv=1_000_000.0)

    with pytest.raises(SecurityValidationError):
        evaluate_diamond_dybvig_profile(ticker="INVALID$$$TICKER")


def test_evaluate_diamond_dybvig_profile_with_prices() -> None:
    """Verify evaluate_diamond_dybvig_profile works with explicit price series."""
    prices = [100.0 + i * 0.5 for i in range(30)]
    volumes = [500_000.0 for _ in range(30)]

    receipt = evaluate_diamond_dybvig_profile(
        ticker="AAPL",
        liquid_reserve_ratio=0.18,
        early_withdrawal_ratio=0.22,
        prices=prices,
        volumes=volumes,
    )
    assert receipt.ticker == "AAPL"
    assert receipt.latest_price == float(prices[-1])
    assert verify_diamond_dybvig_audit_digest(receipt) is True


def test_persist_diamond_dybvig_record(tmp_path: Path) -> None:
    """Verify persisting Diamond-Dybvig record to SQLite database."""
    db_file = tmp_path / "test_diamond_dybvig.db"

    receipt = calculate_diamond_dybvig_run(
        p_t=110.0,
        adv=750_000.0,
        liquid_reserve_ratio=0.15,
        early_withdrawal_ratio=0.20,
        ticker="TESTDB",
    )

    success = persist_diamond_dybvig_record(receipt, db_path=db_file)
    assert success is True

    # Verify directly from SQLite
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("SELECT ticker, latest_price, regime, audit_digest FROM diamond_dybvig_run_records")
    row = cursor.fetchone()
    conn.close()

    assert row is not None
    assert row[0] == "TESTDB"
    assert row[1] == 110.0
    assert row[2] == receipt.regime
    assert row[3] == receipt.audit_digest


def test_cli_diamond_dybvig_command(capsys: pytest.CaptureFixture) -> None:
    """Verify diamond-dybvig CLI command runs cleanly in human-readable mode."""
    args = argparse.Namespace(
        ticker="AAPL",
        reserve=0.15,
        withdrawals=0.20,
        r1=1.05,
        r2=1.35,
        salvage=0.65,
        insured=False,
        json=False,
    )

    diamond_dybvig_command(args)
    captured = capsys.readouterr().out

    assert "NextOmni Stock: Diamond & Dybvig (1983)" in captured
    assert "AAPL" in captured
    assert "Liquid Reserve Buffer" in captured
    assert "Exhaustion Cutoff" in captured
    assert "Audit Digest (HMAC)" in captured


def test_cli_diamond_dybvig_command_json(capsys: pytest.CaptureFixture) -> None:
    """Verify diamond-dybvig CLI command outputs valid JSON with complete fields."""
    args = argparse.Namespace(
        ticker="NVDA",
        reserve=0.20,
        withdrawals=0.25,
        r1=1.05,
        r2=1.35,
        salvage=0.65,
        insured=False,
        json=True,
    )

    diamond_dybvig_command(args)
    captured = capsys.readouterr().out
    data = json.loads(captured)

    assert data["ticker"] == "NVDA"
    assert "run_hazard_probability" in data
    assert "exhaustion_threshold_fbar" in data
    assert "regime" in data
    assert "audit_digest" in data
