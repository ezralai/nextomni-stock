"""Unit tests for Kingsley Y. L. Fong, Craig W. Holden & Charles A. Trzcinka (2017) FHT Spread Engine."""

import sqlite3
import numpy as np
import pytest
from scipy import stats

from src.analytics.fht_closed_form_spread import (
    FHTReceipt,
    calculate_fht_spread,
    evaluate_fht_from_prices,
    persist_fht_record,
    verify_hmac_signature,
)


def test_fht_closed_form_exact_identity():
    """Verify FHT spread matches theoretical closed-form 2 * sigma * Phi^{-1}((1+z)/2)."""
    # Create controlled return series with exact zero-return proportion z = 0.25
    T = 100
    n_zeros = 25
    r = np.zeros(T)
    # 25 zeros, 75 non-zeros with known sample volatility
    non_zeros = np.array([0.01 if i % 2 == 0 else -0.01 for i in range(75)])
    r[25:] = non_zeros

    sigma = float(np.std(r, ddof=1))
    z = 0.25
    arg_p = (1.0 + z) / 2.0  # 0.625
    expected_probit = float(stats.norm.ppf(arg_p))
    expected_spread_pct = float(2.0 * sigma * expected_probit)
    expected_spread_bps = float(expected_spread_pct * 10000.0)

    receipt = calculate_fht_spread(r, ticker="FHT_ID", zero_tolerance=1e-5)

    assert isinstance(receipt, FHTReceipt)
    assert receipt.ticker == "FHT_ID"
    assert receipt.sample_size == T
    assert receipt.zero_return_count == n_zeros
    assert abs(receipt.zero_return_frequency - z) < 1e-6
    assert abs(receipt.effective_spread_pct - expected_spread_pct) < 1e-6
    assert abs(receipt.effective_spread_bps - expected_spread_bps) < 1e-4
    assert abs(receipt.probit_quantile - expected_probit) < 1e-4

    # Cryptographic HMAC verification
    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.zero_return_count}|{receipt.zero_return_frequency:.4f}|"
        f"{receipt.return_volatility_pct / 100.0:.6f}|{receipt.effective_spread_pct:.6f}|"
        f"{receipt.effective_spread_bps:.2f}|{receipt.half_spread_bps:.2f}|{receipt.probit_quantile:.4f}|"
        f"{receipt.roll_spread_bps:.2f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True


def test_fht_zero_return_monotonicity():
    """Verify that higher zero-return frequency z monotonically inflates FHT spread."""
    rng = np.random.default_rng(42)
    T = 300
    base_r = rng.normal(0.0, 0.01, size=T)

    # 10% zeros
    r_low_z = np.copy(base_r)
    r_low_z[:30] = 0.0

    # 40% zeros
    r_high_z = np.copy(base_r)
    r_high_z[:120] = 0.0

    rec_low = calculate_fht_spread(r_low_z, ticker="LOW_Z")
    rec_high = calculate_fht_spread(r_high_z, ticker="HIGH_Z")

    assert rec_high.zero_return_frequency > rec_low.zero_return_frequency
    assert rec_high.probit_quantile > rec_low.probit_quantile
    assert rec_high.effective_spread_bps > rec_low.effective_spread_bps


def test_fht_hmac_tamper_detection():
    """Verify cryptographic HMAC-SHA256 signature detects modifications."""
    prices = [100.0 * (1.001 ** i) if i % 3 != 0 else 100.0 * (1.001 ** (i - 1)) for i in range(50)]
    receipt = evaluate_fht_from_prices(prices, ticker="AUTHENTIC_FHT")

    payload = (
        f"{receipt.ticker}|{receipt.sample_size}|{receipt.zero_return_count}|{receipt.zero_return_frequency:.4f}|"
        f"{receipt.return_volatility_pct / 100.0:.6f}|{receipt.effective_spread_pct:.6f}|"
        f"{receipt.effective_spread_bps:.2f}|{receipt.half_spread_bps:.2f}|{receipt.probit_quantile:.4f}|"
        f"{receipt.roll_spread_bps:.2f}|{receipt.regime}|{receipt.derivation_mode}|{receipt.timestamp}"
    )
    assert verify_hmac_signature(payload, receipt.hmac_signature) is True

    # Tampered payload fails
    tampered = payload.replace("AUTHENTIC_FHT", "FORGED_FHT")
    assert verify_hmac_signature(tampered, receipt.hmac_signature) is False


def test_fht_sqlite_persistence(tmp_path):
    """Verify SQLite database persistence and schema creation."""
    db_file = str(tmp_path / "test_fht.db")
    prices = [100.0, 100.0, 100.2, 100.2, 100.5, 99.8, 99.8, 100.1, 100.1, 100.3] * 4

    receipt = evaluate_fht_from_prices(prices, ticker="PERSIST_FHT")
    assert persist_fht_record(receipt, db_path=db_file) is True

    with sqlite3.connect(db_file) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT ticker, effective_spread_bps, zero_return_count, regime, hmac_signature FROM fht_spread_receipts WHERE ticker = ?",
            ("PERSIST_FHT",),
        )
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == "PERSIST_FHT"
        assert abs(row[1] - receipt.effective_spread_bps) < 1e-4
        assert row[2] == receipt.zero_return_count
        assert row[3] == receipt.regime
        assert row[4] == receipt.hmac_signature

