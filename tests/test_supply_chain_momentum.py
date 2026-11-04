"""Unit and integration tests for Lauren Cohen & Andrea Frazzini (2008) / Kenneth R. Ahern (2013)
Economic Links, Supply-Chain Customer Momentum & Network Centrality Engine (Iteration 90).
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest

from src.analytics.supply_chain_momentum import (
    SupplyChainMomentumReceipt,
    calculate_supply_chain_momentum,
    classify_supply_chain_regime,
    evaluate_supply_chain_momentum_profile,
)
from src.cli import supply_chain_command
import src.data.db as db
from src.security.validators import SecurityValidationError


def test_positive_customer_spillover_alpha_regime() -> None:
    """Verify strong positive customer return not yet priced into supplier triggers POSITIVE_CUSTOMER_SPILLOVER_ALPHA."""
    rec = calculate_supply_chain_momentum(
        ticker="TSM",
        latest_price=185.0,
        customer_portfolio_return=0.14,
        supplier_own_return=0.02,
        customer_revenue_share=0.45,
        shared_coverage_ratio=0.30,
        network_centrality_score=0.72,
    )
    assert rec.customer_supplier_return_wedge > 0.10
    assert rec.limited_attention_spillover_impulse >= 0.035
    assert rec.regime == "POSITIVE_CUSTOMER_SPILLOVER_ALPHA"
    assert rec.is_positive_spillover_leader is True
    assert rec.is_upstream_demand_collapse_trap is False
    assert rec.expected_supply_chain_alpha_pct > 5.0
    assert rec.verify_integrity() is True


def test_upstream_demand_collapse_trap_regime() -> None:
    """Verify sharp negative customer shock unpriced by supplier triggers UPSTREAM_DEMAND_COLLAPSE_TRAP."""
    rec = calculate_supply_chain_momentum(
        ticker="QCOM",
        latest_price=160.0,
        customer_portfolio_return=-0.16,
        supplier_own_return=-0.01,
        customer_revenue_share=0.50,
        shared_coverage_ratio=0.25,
        network_centrality_score=0.50,
    )
    assert rec.customer_supplier_return_wedge < -0.10
    assert rec.limited_attention_spillover_impulse <= -0.035
    assert rec.regime == "UPSTREAM_DEMAND_COLLAPSE_TRAP"
    assert rec.is_upstream_demand_collapse_trap is True
    assert rec.is_positive_spillover_leader is False
    assert rec.expected_supply_chain_alpha_pct < -5.0
    assert rec.verify_integrity() is True


def test_concentrated_customer_dependency_watch_regime() -> None:
    """Verify high customer revenue share (>= 0.55) with aligned returns classifies as dependency watch."""
    rec = calculate_supply_chain_momentum(
        ticker="SWKS",
        latest_price=95.0,
        customer_portfolio_return=0.02,
        supplier_own_return=0.02,
        customer_revenue_share=0.65,
        shared_coverage_ratio=0.60,
        network_centrality_score=0.45,
    )
    assert abs(rec.limited_attention_spillover_impulse) < 0.035
    assert rec.regime == "CONCENTRATED_CUSTOMER_DEPENDENCY_WATCH"
    assert rec.is_positive_spillover_leader is False
    assert rec.is_upstream_demand_collapse_trap is False


def test_synchronized_supply_chain_equilibrium_regime() -> None:
    """Verify aligned customer and supplier returns with moderate dependency classify as equilibrium."""
    rec = calculate_supply_chain_momentum(
        ticker="APH",
        latest_price=70.0,
        customer_portfolio_return=0.03,
        supplier_own_return=0.04,
        customer_revenue_share=0.25,
        shared_coverage_ratio=0.70,
        network_centrality_score=0.55,
    )
    assert abs(rec.limited_attention_spillover_impulse) < 0.035
    assert rec.regime == "SYNCHRONIZED_SUPPLY_CHAIN_EQUILIBRIUM"
    assert rec.is_positive_spillover_leader is False
    assert rec.is_upstream_demand_collapse_trap is False


def test_contemporaneous_catchup_guard() -> None:
    """Verify supplier that has already rallied past positive customer return has non-positive unpriced wedge."""
    rec = calculate_supply_chain_momentum(
        ticker="NVDA",
        latest_price=140.0,
        customer_portfolio_return=0.08,
        supplier_own_return=0.18,
        customer_revenue_share=0.40,
        shared_coverage_ratio=0.50,
    )
    assert rec.customer_supplier_return_wedge <= 0.0
    assert rec.limited_attention_spillover_impulse <= 0.0


def test_inattention_and_centrality_monotonicity() -> None:
    """Verify lower shared coverage (higher inattention) and higher Ahern centrality increase expected alpha."""
    rec_high_cov = calculate_supply_chain_momentum(
        ticker="TSM",
        latest_price=180.0,
        customer_portfolio_return=0.10,
        supplier_own_return=0.01,
        customer_revenue_share=0.40,
        shared_coverage_ratio=0.90,
        network_centrality_score=0.25,
    )
    rec_low_cov = calculate_supply_chain_momentum(
        ticker="TSM",
        latest_price=180.0,
        customer_portfolio_return=0.10,
        supplier_own_return=0.01,
        customer_revenue_share=0.40,
        shared_coverage_ratio=0.15,
        network_centrality_score=0.85,
    )
    assert rec_low_cov.inattention_diffusion_multiplier > rec_high_cov.inattention_diffusion_multiplier
    assert rec_low_cov.limited_attention_spillover_impulse > rec_high_cov.limited_attention_spillover_impulse
    assert rec_low_cov.expected_supply_chain_alpha_pct > rec_high_cov.expected_supply_chain_alpha_pct


def test_classify_supply_chain_regime_boundaries() -> None:
    """Verify exact boundary transitions in classify_supply_chain_regime."""
    assert classify_supply_chain_regime(0.050, 0.30) == "POSITIVE_CUSTOMER_SPILLOVER_ALPHA"
    assert classify_supply_chain_regime(-0.045, 0.30) == "UPSTREAM_DEMAND_COLLAPSE_TRAP"
    assert classify_supply_chain_regime(0.010, 0.60) == "CONCENTRATED_CUSTOMER_DEPENDENCY_WATCH"
    assert classify_supply_chain_regime(0.010, 0.30) == "SYNCHRONIZED_SUPPLY_CHAIN_EQUILIBRIUM"


def test_evaluate_supply_chain_momentum_profile_with_db(tmp_path: Path) -> None:
    """Verify evaluate_supply_chain_momentum_profile reads closing prices from SQLite and computes spillover."""
    db_file = tmp_path / "supply_chain_test.db"
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

    rec = evaluate_supply_chain_momentum_profile(ticker="AAPL", db_path=db_file)
    assert rec.ticker == "AAPL"
    assert abs(rec.latest_price - 192.0) < 1e-2
    assert rec.customer_revenue_share == 0.36
    assert rec.verify_integrity() is True


def test_invalid_inputs_rejected() -> None:
    """Verify invalid price, returns, revenue share, shared coverage, and centrality raise ValueError."""
    with pytest.raises(ValueError, match="latest_price must be strictly positive"):
        calculate_supply_chain_momentum("AAPL", 0.0)
    with pytest.raises(ValueError, match="customer_portfolio_return must be in"):
        calculate_supply_chain_momentum("AAPL", 150.0, customer_portfolio_return=-1.2)
    with pytest.raises(ValueError, match="supplier_own_return must be in"):
        calculate_supply_chain_momentum("AAPL", 150.0, supplier_own_return=6.0)
    with pytest.raises(ValueError, match="customer_revenue_share must be in"):
        calculate_supply_chain_momentum("AAPL", 150.0, customer_revenue_share=1.2)
    with pytest.raises(ValueError, match="shared_coverage_ratio must be in"):
        calculate_supply_chain_momentum("AAPL", 150.0, shared_coverage_ratio=-0.1)
    with pytest.raises(ValueError, match="network_centrality_score must be in"):
        calculate_supply_chain_momentum("AAPL", 150.0, network_centrality_score=1.5)


def test_ticker_security_allowlist_enforcement() -> None:
    """Verify path traversal and SQL injection payloads in ticker are rejected with SecurityValidationError."""
    for malicious in ["../etc/passwd", "AAPL; DROP TABLE stocks;--", "bad$ticker"]:
        with pytest.raises(SecurityValidationError):
            calculate_supply_chain_momentum(malicious, 150.0)


def test_hmac_signature_tamper_detection() -> None:
    """Verify HMAC-SHA256 integrity check fails if any metric on the receipt is tampered with."""
    rec = calculate_supply_chain_momentum(
        ticker="GOOG",
        latest_price=175.0,
        customer_portfolio_return=0.09,
        supplier_own_return=0.02,
    )
    assert rec.verify_integrity() is True
    tampered = SupplyChainMomentumReceipt(
        **{**rec.to_dict(), "expected_supply_chain_alpha_pct": 99.99}
    )
    assert tampered.verify_integrity() is False


def test_cli_supply_chain_command(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify CLI supply_chain_command text and JSON output modes."""
    args_text = argparse.Namespace(
        ticker="TSM",
        customer_return=0.14,
        supplier_return=0.02,
        revenue_share=0.45,
        shared_coverage=0.30,
        network_centrality=0.72,
        json=False,
    )
    supply_chain_command(args_text)
    captured = capsys.readouterr()
    assert "Cohen & Frazzini (2008) & Ahern (2013) Supply-Chain Engine" in captured.out
    assert "TSM" in captured.out
    assert "POSITIVE_CUSTOMER_SPILLOVER_ALPHA" in captured.out

    args_json = argparse.Namespace(
        ticker="TSM",
        customer_return=0.14,
        supplier_return=0.02,
        revenue_share=0.45,
        shared_coverage=0.30,
        network_centrality=0.72,
        json=True,
    )
    supply_chain_command(args_json)
    captured_json = capsys.readouterr()
    payload = json.loads(captured_json.out)
    assert payload["ticker"] == "TSM"
    assert payload["regime"] == "POSITIVE_CUSTOMER_SPILLOVER_ALPHA"
    assert payload["is_positive_spillover_leader"] is True
    assert len(payload["signature"]) == 64

