r"""Ananth Madhavan, Matthew Richardson & Mark Roomans (1997) MRR Microstructure Engine.

Implements Ananth Madhavan, Matthew Richardson, and Mark Roomans (1997), "Why Do Security Prices
Change? A Transaction-Level Analysis of NYSE Stocks", The Review of Financial Studies, 10(4), pp. 1035-1064:

- Structural market microstructure specification incorporating serial trade autocorrelation:
  1. Fundamental value updates exclusively from unexpected order flow surprise:
         mu_t = mu_{t-1} + theta * (x_t - E[x_t | x_{t-1}]) + epsilon_t
     where x_t in {-1, +1} is trade direction indicator, order flow follows a Markov chain with
     continuation probability rho = P(x_t = x_{t-1}) and autocorrelation phi = 2*rho - 1,
     giving expected order flow E[x_t | x_{t-1}] = phi * x_{t-1}.
     theta >= 0 measures the permanent information impact of order flow surprise.
  2. Transaction price incorporates order processing costs and transitory noise:
         p_t = mu_t + phi0 * x_t + xi_t
     where phi0 >= 0 is the order processing cost per share, and xi_t is rounding/discreteness noise.
  3. Price change regression specification:
         Delta p_t = (theta + phi0) * x_t - (phi * theta + phi0) * x_{t-1} + u_t
                   = beta_1 * x_t + beta_2 * x_{t-1} + u_t
- Econometric parameter recovery:
      hat{theta} = (beta_1 + beta_2) / (1 - phi)
      hat{phi0} = beta_1 - hat{theta}
      Effective Spread S = 2 * (hat{theta} + hat{phi0}) = 2 * beta_1
      Adverse selection ratio = hat{theta} / (hat{theta} + hat{phi0})
      Order processing ratio = hat{phi0} / (hat{theta} + hat{phi0})
- Strict ticker input validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-mrr-1997-audit-key"


@dataclass(frozen=True)
class MRRReceipt:
    """Immutable receipt for Madhavan, Richardson & Roomans (1997) MRR Microstructure evaluation."""

    ticker: str
    sample_size: int
    theta_dollar: float
    phi0_dollar: float
    effective_spread_bps: float
    adverse_selection_ratio: float
    order_processing_ratio: float
    order_autocorrelation_phi: float
    continuation_prob_rho: float
    r_squared: float
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(
            f"Invalid ticker: '{ticker}'. Must be alphanumeric with optional ._- and max 20 chars."
        )
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    sample_size: int,
    theta_dollar: float,
    phi0_dollar: float,
    effective_spread_bps: float,
    adverse_selection_ratio: float,
    order_processing_ratio: float,
    order_autocorrelation_phi: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{sample_size}|{theta_dollar:.6f}|{phi0_dollar:.6f}|"
        f"{effective_spread_bps:.4f}|{adverse_selection_ratio:.4f}|"
        f"{order_processing_ratio:.4f}|{order_autocorrelation_phi:.4f}|"
        f"{regime}|{timestamp}"
    )


def sign_mrr_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_mrr_receipt(payload)
    return hmac.compare_digest(expected, signature)


def calculate_mrr_microstructure(
    price_changes: Sequence[float] | np.ndarray,
    trade_indicators: Sequence[int | float] | np.ndarray,
    *,
    ticker: str = "SPY",
    reference_price: float = 100.0,
) -> MRRReceipt:
    """Estimate MRR (1997) structural parameters via multi-variate OLS regression on trade signs."""
    clean_ticker = _validate_ticker(ticker)
    dp = np.asarray(price_changes, dtype=float)
    x = np.asarray(trade_indicators, dtype=float)

    if len(dp) != len(x):
        raise ValueError(
            f"Length mismatch: price_changes ({len(dp)}) must match trade_indicators ({len(x)})"
        )
    if len(dp) < 30:
        raise ValueError(f"At least 30 observations required for MRR estimation, got {len(dp)}")
    if reference_price <= 0.0:
        raise ValueError(f"Reference price must be positive, got {reference_price}")

    # Clean binary trade indicators x_t in {-1, +1}
    x_clean = np.where(x >= 0, 1.0, -1.0)
    n = len(x_clean)

    # 1. Estimate order flow continuation probability rho and autocorrelation phi
    continuations = np.sum(x_clean[1:] == x_clean[:-1])
    rho = float(continuations / (n - 1))
    phi = float(np.clip(2.0 * rho - 1.0, -0.90, 0.90))

    # 2. Setup regression Delta p_t = beta_1 * x_t + beta_2 * x_{t-1} + u_t
    y = dp[1:]
    x0 = x_clean[1:]
    x1 = x_clean[:-1]
    X = np.column_stack([x0, x1])

    try:
        beta, residuals, rank, s_vals = np.linalg.lstsq(X, y, rcond=None)
        b1, b2 = float(beta[0]), float(beta[1])
    except np.linalg.LinAlgError:
        b1, b2 = 0.02, -0.01

    # Guard against non-positive total spread coefficient
    b1 = max(1e-5, b1)

    # 3. Structural parameter recovery:
    # theta = (b1 + b2) / (1 - phi)
    # phi0 = b1 - theta
    one_minus_phi = max(0.05, 1.0 - phi)
    raw_theta = (b1 + b2) / one_minus_phi
    raw_phi0 = b1 - raw_theta

    # Enforce non-negativity constraints
    theta = float(max(0.0, raw_theta))
    phi0 = float(max(0.0, raw_phi0))

    total_cost = theta + phi0
    if total_cost > 1e-8:
        adverse_ratio = float(theta / total_cost)
        order_proc_ratio = float(phi0 / total_cost)
    else:
        adverse_ratio = 0.50
        order_proc_ratio = 0.50
        total_cost = 1e-4

    # Implied effective spread: S = 2 * (theta + phi0)
    effective_spread_dollar = 2.0 * total_cost
    effective_spread_bps = float((effective_spread_dollar / reference_price) * 10000.0)

    # R-squared calculation
    y_pred = X @ np.array([b1, b2])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = float(max(0.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 1e-12 else 0.0

    if adverse_ratio >= 0.60:
        regime = "INFORMATION_ASYMMETRY_DOMINANT"
    elif order_proc_ratio >= 0.65:
        regime = "ORDER_PROCESSING_FRICTION_DOMINANT"
    elif abs(phi) >= 0.40:
        regime = "MOMENTUM_ORDER_PERSISTENCE"
    else:
        regime = "BALANCED_MRR_MICROSTRUCTURE"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        sample_size=len(dp),
        theta_dollar=theta,
        phi0_dollar=phi0,
        effective_spread_bps=effective_spread_bps,
        adverse_selection_ratio=adverse_ratio,
        order_processing_ratio=order_proc_ratio,
        order_autocorrelation_phi=phi,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_mrr_receipt(payload)

    return MRRReceipt(
        ticker=clean_ticker,
        sample_size=len(dp),
        theta_dollar=theta,
        phi0_dollar=phi0,
        effective_spread_bps=effective_spread_bps,
        adverse_selection_ratio=adverse_ratio,
        order_processing_ratio=order_proc_ratio,
        order_autocorrelation_phi=phi,
        continuation_prob_rho=rho,
        r_squared=r_squared,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_mrr_from_prices(
    prices: Sequence[float] | np.ndarray,
    *,
    ticker: str = "SPY",
) -> MRRReceipt:
    """Infer MRR structural parameters directly from historical price sequence."""
    clean_ticker = _validate_ticker(ticker)
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 32:
        raise ValueError(f"At least 32 price observations required, got {len(p_arr)}")

    dp = np.diff(p_arr)
    # Trade sign inferred from price direction
    x = np.where(dp >= 0, 1.0, -1.0)
    ref_price = float(p_arr[-1])

    return calculate_mrr_microstructure(
        dp,
        x,
        ticker=clean_ticker,
        reference_price=ref_price,
    )


def persist_mrr_record(receipt: MRRReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist MRR microstructure receipt into SQLite database under parameterized query."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mrr_microstructure_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                theta_dollar REAL NOT NULL,
                phi0_dollar REAL NOT NULL,
                effective_spread_bps REAL NOT NULL,
                adverse_selection_ratio REAL NOT NULL,
                order_processing_ratio REAL NOT NULL,
                order_autocorrelation_phi REAL NOT NULL,
                continuation_prob_rho REAL NOT NULL,
                r_squared REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO mrr_microstructure_records (
                ticker, sample_size, theta_dollar, phi0_dollar,
                effective_spread_bps, adverse_selection_ratio, order_processing_ratio,
                order_autocorrelation_phi, continuation_prob_rho, r_squared,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.theta_dollar,
                receipt.phi0_dollar,
                receipt.effective_spread_bps,
                receipt.adverse_selection_ratio,
                receipt.order_processing_ratio,
                receipt.order_autocorrelation_phi,
                receipt.continuation_prob_rho,
                receipt.r_squared,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

