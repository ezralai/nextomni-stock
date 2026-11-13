"""Thomas J. George, Gautam Kaul & M. Nimalendran (1991) GKN Bid-Ask Spread Decomposition Engine.

Implements the econometric spread decomposition framework formulated in:
George, Thomas J., Gautam Kaul, and M. Nimalendran (1991).
"Estimation of the Bid-Ask Spread and Its Components: A New Approach."
The Review of Financial Studies, 4(4), 623-656.

GKN proves that Roll's (1984) serial covariance spread estimator is biased downward
due to time-varying expected returns. By taking the difference between transaction
returns (R_t) and quote midpoint returns (R_t^Q):
    RD_t = R_t - R_t^Q
time-varying expected returns and public news shocks cancel out symmetrically,
allowing exact isolation of the order-processing component (s_0), the adverse selection
component (s_i), and the hidden variance of expected returns:
    Cov(RD_t, RD_{t-1}) = - (s_0^2) / 4
    s_0 = 2 * sqrt(max(0, -Cov(RD_t, RD_{t-1})))
    s_i = max(0, S_eff - s_0)
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,16}$")
SECRET_SALT = os.environ.get("NEXTOMNI_AUDIT_SALT", "nextomni_gkn_audit_salt").encode("utf-8")


@dataclass(frozen=True)
class GKNReceipt:
    """Immutable receipt containing George, Kaul & Nimalendran (1991) spread decomposition."""

    ticker: str
    sample_size: int
    total_effective_spread_bps: float
    order_processing_spread_bps: float
    adverse_selection_spread_bps: float
    order_processing_ratio: float
    adverse_selection_ratio: float
    roll_spread_bps: float
    expected_return_autocovariance: float
    roll_bias_bps: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_hmac(payload: str) -> str:
    return hmac.new(SECRET_SALT, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify cryptographic HMAC-SHA256 signature against payload."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


def calculate_gkn_spread(
    transaction_prices: Sequence[float],
    quote_midpoints: Optional[Sequence[float]] = None,
    ticker: str = "SPY",
) -> GKNReceipt:
    """Calculate George, Kaul & Nimalendran (1991) spread decomposition.

    Parameters
    ----------
    transaction_prices : Sequence[float]
        Series of trade or closing transaction prices.
    quote_midpoints : Optional[Sequence[float]]
        Series of bid-ask quote midpoints (M_t = (A_t + B_t) / 2).
        If None, a synthesized smoothed midquote proxy is derived.
    ticker : str
        Stock ticker symbol.

    Returns
    -------
    GKNReceipt
        Complete decomposition receipt.
    """
    clean_ticker = ticker.strip().upper() if TICKER_REGEX.match(ticker.strip().upper()) else "UNKNOWN"
    now_iso = datetime.now(timezone.utc).isoformat()

    p_arr = np.asarray(transaction_prices, dtype=np.float64)
    valid_p = np.isfinite(p_arr) & (p_arr > 0.0)
    p_clean = p_arr[valid_p]
    n_pts = len(p_clean)

    if n_pts < 12:
        empty_payload = f"{clean_ticker}|0|0.0|0.0|0.0|{now_iso}"
        sig = _compute_hmac(empty_payload)
        return GKNReceipt(
            ticker=clean_ticker,
            sample_size=n_pts,
            total_effective_spread_bps=0.0,
            order_processing_spread_bps=0.0,
            adverse_selection_spread_bps=0.0,
            order_processing_ratio=0.0,
            adverse_selection_ratio=0.0,
            roll_spread_bps=0.0,
            expected_return_autocovariance=0.0,
            roll_bias_bps=0.0,
            regime="INSUFFICIENT_DATA",
            derivation_mode="NONE",
            hmac_signature=sig,
            timestamp=now_iso,
        )

    # Resolve quote midpoints
    if quote_midpoints is not None:
        m_arr = np.asarray(quote_midpoints, dtype=np.float64)
        if len(m_arr) == len(p_arr):
            m_clean = m_arr[valid_p]
            derivation_mode = "OBSERVED_QUOTE_MIDPOINTS"
        else:
            # Fallback to synthesized proxy if length mismatch
            m_clean = _synthesize_quote_midpoint(p_clean)
            derivation_mode = "SYNTHESIZED_PROXY_MISMATCH"
    else:
        m_clean = _synthesize_quote_midpoint(p_clean)
        derivation_mode = "SYNTHESIZED_MIDPOINT_PROXY"

    # Compute transaction returns and midpoint returns
    r_t = np.diff(p_clean) / p_clean[:-1]
    rq_t = np.diff(m_clean) / m_clean[:-1]

    # Return difference: RD_t = R_t - R_t^Q
    rd_t = r_t - rq_t
    n_ret = len(rd_t)

    # 1. Autocovariance of return differences: Cov(RD_t, RD_{t-1})
    cov_rd = _autocovariance(rd_t, lag=1)

    # Under GKN: Cov(RD_t, RD_{t-1}) = - (s_0^2) / 4
    if cov_rd < 0.0:
        s0_prop = 2.0 * math.sqrt(-cov_rd)
    else:
        # Transitory bounce minimal or non-existent
        s0_prop = 0.0

    # 2. Total empirical effective spread S_eff
    # Proportional effective spread: 2 * |P_t - M_t| / M_t
    abs_deviations = 2.0 * np.abs(p_clean - m_clean) / m_clean
    s_eff_prop = float(np.mean(abs_deviations))

    # If s0 exceeds s_eff due to finite sample variance, bound s_eff
    s_eff_prop = max(s_eff_prop, s0_prop, 1e-4)

    # 3. Adverse selection component: s_i = S_eff - s_0
    si_prop = max(0.0, s_eff_prop - s0_prop)

    # Component ratios
    s0_ratio = s0_prop / s_eff_prop if s_eff_prop > 1e-6 else 0.0
    si_ratio = si_prop / s_eff_prop if s_eff_prop > 1e-6 else 0.0

    # 4. Classic Roll autocovariance and comparison
    cov_r = _autocovariance(r_t, lag=1)
    if cov_r < 0.0:
        roll_prop = 2.0 * math.sqrt(-cov_r)
    else:
        roll_prop = 0.0

    # GKN: Cov(R_t, R_{t-1}) = - (s_0^2) / 4 + Cov(mu_t, mu_{t-1})
    # Therefore, time-varying expected return autocovariance:
    expected_return_autocov = float(cov_r + 0.25 * (s0_prop**2))

    # Convert to basis points
    s_eff_bps = float(s_eff_prop * 10000.0)
    s0_bps = float(s0_prop * 10000.0)
    si_bps = float(si_prop * 10000.0)
    roll_bps = float(roll_prop * 10000.0)
    roll_bias_bps = float(s0_bps - roll_bps)

    # Regime classification
    if si_ratio >= 0.55:
        regime = "HIGH_ADVERSE_SELECTION"
    elif s0_ratio >= 0.70:
        regime = "ORDER_PROCESSING_DOMINATED"
    elif s_eff_bps < 15.0:
        regime = "TIGHT_EFFICIENCY"
    else:
        regime = "BALANCED_SPREAD_EQUILIBRIUM"

    payload = (
        f"{clean_ticker}|{n_pts}|{s_eff_bps:.2f}|{s0_bps:.2f}|{si_bps:.2f}|"
        f"{s0_ratio:.4f}|{si_ratio:.4f}|{roll_bps:.2f}|{expected_return_autocov:.8f}|"
        f"{roll_bias_bps:.2f}|{regime}|{derivation_mode}|{now_iso}"
    )
    sig = _compute_hmac(payload)

    return GKNReceipt(
        ticker=clean_ticker,
        sample_size=n_pts,
        total_effective_spread_bps=s_eff_bps,
        order_processing_spread_bps=s0_bps,
        adverse_selection_spread_bps=si_bps,
        order_processing_ratio=s0_ratio,
        adverse_selection_ratio=si_ratio,
        roll_spread_bps=roll_bps,
        expected_return_autocovariance=expected_return_autocov,
        roll_bias_bps=roll_bias_bps,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig,
        timestamp=now_iso,
    )


def _autocovariance(series: np.ndarray, lag: int = 1) -> float:
    """Calculate sample autocovariance at specified lag."""
    if len(series) <= lag:
        return 0.0
    s_curr = series[lag:]
    s_prev = series[:-lag]
    m_curr = float(np.mean(s_curr))
    m_prev = float(np.mean(s_prev))
    return float(np.mean((s_curr - m_curr) * (s_prev - m_prev)))


def _synthesize_quote_midpoint(prices: np.ndarray) -> np.ndarray:
    """Synthesize smooth fundamental quote midpoint proxy from transaction prices.

    Uses a fast 3-period rolling average with boundary preservation to isolate
    the permanent underlying quote component from transitory bid-ask bounces.
    """
    n = len(prices)
    if n < 3:
        return np.copy(prices)
    mid = np.copy(prices)
    # Moving average filter: M_t = 0.25 * P_{t-1} + 0.50 * P_t + 0.25 * P_{t+1}
    mid[1:-1] = 0.25 * prices[:-2] + 0.50 * prices[1:-1] + 0.25 * prices[2:]
    mid[0] = 0.67 * prices[0] + 0.33 * prices[1]
    mid[-1] = 0.67 * prices[-1] + 0.33 * prices[-2]
    return mid


def persist_gkn_record(
    receipt: GKNReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist GKN spread decomposition receipt in SQLite with foreign keys & HMAC."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gkn_spread_receipts (
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                total_effective_spread_bps REAL NOT NULL,
                order_processing_spread_bps REAL NOT NULL,
                adverse_selection_spread_bps REAL NOT NULL,
                order_processing_ratio REAL NOT NULL,
                adverse_selection_ratio REAL NOT NULL,
                roll_spread_bps REAL NOT NULL,
                expected_return_autocovariance REAL NOT NULL,
                roll_bias_bps REAL NOT NULL,
                regime TEXT NOT NULL,
                derivation_mode TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (ticker, timestamp)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO gkn_spread_receipts (
                ticker, sample_size, total_effective_spread_bps,
                order_processing_spread_bps, adverse_selection_spread_bps,
                order_processing_ratio, adverse_selection_ratio,
                roll_spread_bps, expected_return_autocovariance,
                roll_bias_bps, regime, derivation_mode, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.total_effective_spread_bps,
                receipt.order_processing_spread_bps,
                receipt.adverse_selection_spread_bps,
                receipt.order_processing_ratio,
                receipt.adverse_selection_ratio,
                receipt.roll_spread_bps,
                receipt.expected_return_autocovariance,
                receipt.roll_bias_bps,
                receipt.regime,
                receipt.derivation_mode,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

