"""Tests for William F. Sharpe (1992) Returns-Based Style Analysis Engine."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys

import numpy as np
import pytest

from src.analytics.sharpe_style_analysis import (
    DEFAULT_STYLE_NAMES,
    _build_hmac_payload,
    build_synthetic_style_returns,
    calculate_sharpe_style_analysis,
    evaluate_sharpe_style_profile,
    persist_sharpe_style_record,
    verify_hmac_signature,
)


def test_sharpe_rbsa_exact_zero_noise_replication() -> None:
    """Under pure style replication (no noise/alpha), R^2 == 1.0 and weights recover cleanly."""
    true_weights = [0.40, 0.35, 0.15, 0.10]
    rp, factors, names = build_synthetic_style_returns(
        true_weights, n_obs=250, alpha_ann=0.0, selection_vol_ann=0.0, seed=179
    )
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="VTI")

    assert rec.style_r2 == pytest.approx(1.0, abs=1e-6)
    assert rec.selection_var_share == pytest.approx(0.0, abs=1e-6)
    assert rec.dominant_style == "Value_Equity"
    for name, w_true in zip(names, true_weights):
        assert rec.style_weights[name] == pytest.approx(w_true, abs=1e-4)
    assert rec.regime == "PURE_STYLE_REPLICATOR"


def test_sharpe_rbsa_simplex_constraints_and_variance_partition() -> None:
    """Weights must obey non-negativity, unit sum, and Style R^2 + Selection == 1.0."""
    rp, factors, names = build_synthetic_style_returns(
        [0.10, 0.65, 0.25, 0.00], n_obs=150, alpha_ann=0.04, selection_vol_ann=0.06, seed=42
    )
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="ARKK")

    weights = list(rec.style_weights.values())
    assert sum(weights) == pytest.approx(1.0, rel=1e-12)
    assert all(w >= 0.0 for w in weights)
    assert rec.style_r2 + rec.selection_var_share == pytest.approx(1.0, rel=1e-12)
    assert rec.dominant_style == "Growth_Equity"


def test_sharpe_rbsa_alpha_stock_picker_and_drift_trap_regimes() -> None:
    """Distinguish skilled stock picker from negative-alpha style-drift trap."""
    rp_alpha, factors, _ = build_synthetic_style_returns(
        [0.25, 0.25, 0.25, 0.25], n_obs=160, alpha_ann=0.12, selection_vol_ann=0.10, seed=179
    )
    rng = np.random.default_rng(999)
    trap_noise = rng.normal(0.0, 0.025, size=160)
    rp_trap = (trap_noise - float(np.mean(trap_noise))) - (0.08 / 252.0)

    rec_alpha = calculate_sharpe_style_analysis(rp_alpha, factors, ticker="ALPHA")
    rec_trap = calculate_sharpe_style_analysis(rp_trap, factors, ticker="TRAP")
    assert rec_alpha.regime == "ALPHA_DRIVEN_STOCK_PICKER"
    assert rec_trap.regime == "UNEXPLAINED_STYLE_DRIFT_TRAP"


def test_sharpe_rbsa_style_tilted_core_regime() -> None:
    """Moderate active variance with neutral selection alpha triggers STYLE_TILTED_CORE_MANAGER."""
    rp, factors, names = build_synthetic_style_returns(
        [0.50, 0.30, 0.10, 0.10], n_obs=200, alpha_ann=0.0, selection_vol_ann=0.05, seed=7
    )
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="CORE")
    assert 0.65 <= rec.style_r2 < 0.90
    assert rec.regime == "STYLE_TILTED_CORE_MANAGER"


def test_sharpe_rbsa_hhi_concentration_bounds() -> None:
    """Verify weight Herfindahl concentration satisfies 1/K <= HHI <= 1.0."""
    rp, factors, names = build_synthetic_style_returns(
        [0.0, 1.0, 0.0, 0.0], n_obs=100, alpha_ann=0.0, selection_vol_ann=0.0
    )
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="UGRO")
    assert rec.weight_hhi_concentration == pytest.approx(1.0, abs=1e-4)


def test_sharpe_rbsa_input_validation_guards() -> None:
    """Enforce ticker regex, sample length, factor count, and non-zero variance checks."""
    rp, factors, _ = build_synthetic_style_returns([0.25, 0.25, 0.25, 0.25], n_obs=40)
    with pytest.raises(ValueError, match="Invalid ticker format"):
        calculate_sharpe_style_analysis(rp, factors, ticker="SPY; DROP")
    with pytest.raises(ValueError, match="At least 14"):
        calculate_sharpe_style_analysis(rp[:10], factors[:10])
    with pytest.raises(ValueError, match="insufficient variance"):
        calculate_sharpe_style_analysis(np.zeros(40), factors)
    with pytest.raises(ValueError, match="Weights must be a 4-element"):
        build_synthetic_style_returns([0.5, 0.6, 0.0, 0.0])


def test_sharpe_rbsa_hmac_integrity_and_tamper_detection() -> None:
    """Validate HMAC-SHA256 seal integrity and tamper sensitivity."""
    rp, factors, names = build_synthetic_style_returns([0.4, 0.3, 0.2, 0.1], n_obs=60)
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="IVW")
    payload = _build_hmac_payload(
        ticker=rec.ticker,
        n_obs=rec.n_obs,
        n_factors=rec.n_factors,
        dominant_style=rec.dominant_style,
        dominant_weight=rec.dominant_weight,
        style_r2=rec.style_r2,
        selection_alpha_ann=rec.selection_alpha_ann,
        selection_te_ann=rec.selection_te_ann,
        regime=rec.regime,
        timestamp=rec.timestamp,
    )
    assert verify_hmac_signature(payload, rec.hmac_signature) is True
    assert verify_hmac_signature(payload.replace("IVW", "IVE"), rec.hmac_signature) is False


def test_sharpe_rbsa_empirical_profile_calibration() -> None:
    """Calibrate multi-style factor attribution from trailing asset close prices."""
    rng = np.random.default_rng(179)
    prices = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.012, size=80)))
    rec = evaluate_sharpe_style_profile(prices, ticker="SPY")
    assert rec.ticker == "SPY"
    assert sum(rec.style_weights.values()) == pytest.approx(1.0, rel=1e-12)


def test_sharpe_rbsa_sqlite_persistence(tmp_path) -> None:
    """Verify parameterized SQLite audit persistence into sharpe_style_analysis_audits."""
    db_file = tmp_path / "rbsa_audit.db"
    rp, factors, names = build_synthetic_style_returns([0.3, 0.4, 0.2, 0.1], n_obs=60)
    rec = calculate_sharpe_style_analysis(rp, factors, factor_names=names, ticker="MTUM")
    assert persist_sharpe_style_record(rec, db_path=str(db_file)) is True
    with sqlite3.connect(str(db_file)) as conn:
        rows = conn.execute(
            "SELECT ticker, dominant_style, style_r2, selection_alpha_ann, regime "
            "FROM sharpe_style_analysis_audits"
        ).fetchall()
    assert len(rows) == 1
    assert rows[0][0] == "MTUM"
    assert rows[0][2] == pytest.approx(rec.style_r2, rel=1e-9)


def test_sharpe_rbsa_cli_integration_json() -> None:
    """Execute CLI subcommand `style-analysis --json` end-to-end."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.cli",
            "style-analysis",
            "--ticker",
            "FUND",
            "--w-value",
            "0.40",
            "--w-growth",
            "0.35",
            "--w-small",
            "0.15",
            "--w-bond",
            "0.10",
            "--n-obs",
            "120",
            "--json",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["ticker"] == "FUND"
    assert sum(payload["style_weights"].values()) == pytest.approx(1.0, rel=1e-9)
    assert payload["style_r2"] > 0.70
