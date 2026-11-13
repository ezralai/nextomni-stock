r"""Roger D. Huang & Hans R. Stoll (1997) Three-Way Bid-Ask Spread Decomposition Engine.

Implements Roger D. Huang and Hans R. Stoll (1997), "The Components of the Bid-Ask Spread:
A General Approach", The Review of Financial Studies, 10(4), pp. 995-1034:

- Decomposes the effective bid-ask spread S into three distinct structural economic components:
  1. Adverse Selection component (alpha): Compensates market makers for private information risk.
  2. Inventory Holding component (beta): Compensates market makers for carrying inventory risk.
  3. Order Processing component (gamma = 1 - alpha - beta): Operational costs and normal quote profits.
- Structural transaction price change specification:
      Delta P_t = (S / 2) * q_t - (1 - alpha - beta) * (S / 2) * q_{t-1} - beta * (S / 2) * q_{t-2} + e_t
  where q_t in {-1, +1} is the trade direction indicator (buyer-initiated vs seller-initiated).
- Regressing Delta P_t on (q_t, q_{t-1}, q_{t-2}) yields structural coefficients:
      c1 = S / 2
      c2 = -(1 - alpha - beta) * (S / 2) = -gamma * (S / 2)
      c3 = -beta * (S / 2)
  from which:
      S = 2 * c1
      beta = -c3 / c1
      gamma = -c2 / c1
      alpha = 1 - beta - gamma
- Trade continuation probability:
      pi = P(q_t = q_{t-1})
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
DEFAULT_HMAC_KEY = "nextomni-stock-huang-stoll-1997-audit-key"


@dataclass(frozen=True)
class HuangStollReceipt:
    """Immutable receipt for Huang & Stoll (1997) Three-Way Spread Decomposition."""

    ticker: str
    sample_size: int
    effective_spread_bps: float
    adverse_selection_pct: float
    inventory_holding_pct: float
    order_processing_pct: float
    adverse_spread_bps: float
    inventory_spread_bps: float
    processing_spread_bps: float
    trade_continuation_prob: float
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
    effective_spread_bps: float,
    adverse_selection_pct: float,
    inventory_holding_pct: float,
    order_processing_pct: float,
    trade_continuation_prob: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{sample_size}|{effective_spread_bps:.4f}|"
        f"{adverse_selection_pct:.4f}|{inventory_holding_pct:.4f}|"
        f"{order_processing_pct:.4f}|{trade_continuation_prob:.4f}|"
        f"{regime}|{timestamp}"
    )


def sign_huang_stoll_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_huang_stoll_receipt(payload)
    return hmac.compare_digest(expected, signature)


def calculate_huang_stoll_decomposition(
    price_changes: Sequence[float] | np.ndarray,
    trade_indicators: Sequence[int | float] | np.ndarray,
    *,
    ticker: str = "SPY",
    reference_price: float = 100.0,
) -> HuangStollReceipt:
    """Estimate Huang & Stoll (1997) structural spread components via multi-variate OLS regression."""
    clean_ticker = _validate_ticker(ticker)
    dp = np.asarray(price_changes, dtype=float)
    q = np.asarray(trade_indicators, dtype=float)

    if len(dp) != len(q):
        raise ValueError(
            f"Length mismatch: price_changes ({len(dp)}) must match trade_indicators ({len(q)})"
        )
    if len(dp) < 30:
        raise ValueError(f"At least 30 observations required for structural regression, got {len(dp)}")
    if reference_price <= 0.0:
        raise ValueError(f"Reference price must be positive, got {reference_price}")

    # Binary sign alignment q_t in {-1, +1}
    q_clean = np.where(q >= 0, 1.0, -1.0)

    # Calculate trade continuation probability pi = P(q_t == q_{t-1})
    continuations = np.sum(q_clean[1:] == q_clean[:-1])
    trade_continuation_prob = float(continuations / (len(q_clean) - 1))

    # Construct structural lag matrices: t ranges from 2 to N-1
    y = dp[2:]
    x_q0 = q_clean[2:]
    x_q1 = q_clean[1:-1]
    x_q2 = q_clean[:-2]

    # Design matrix X = [q_t, q_{t-1}, q_{t-2}]
    X = np.column_stack([x_q0, x_q1, x_q2])

    # OLS estimation: c = (X'X)^(-1) X'y
    try:
        c, residuals, rank, s_vals = np.linalg.lstsq(X, y, rcond=None)
        c1, c2, c3 = float(c[0]), float(c[1]), float(c[2])
    except np.linalg.LinAlgError:
        c1, c2, c3 = 0.01, -0.005, -0.002

    # If c1 (half-spread) is non-positive due to sampling noise, enforce minimal floor
    c1 = max(1e-5, c1)
    s_dollar = 2.0 * c1

    # Raw proportions:
    # beta = -c3 / c1
    # gamma = -c2 / c1
    # alpha = 1 - beta - gamma
    beta_raw = max(0.0, -c3 / c1)
    gamma_raw = max(0.0, -c2 / c1)
    alpha_raw = max(0.0, 1.0 - beta_raw - gamma_raw)

    total_proportions = alpha_raw + beta_raw + gamma_raw
    if total_proportions > 1e-12:
        alpha = float(alpha_raw / total_proportions)
        beta = float(beta_raw / total_proportions)
        gamma = float(gamma_raw / total_proportions)
    else:
        alpha, beta, gamma = 0.3333, 0.3333, 0.3334

    # Calculate R-squared
    y_pred = X @ np.array([c1, c2, c3])
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    ss_res = float(np.sum((y - y_pred) ** 2))
    r_squared = float(max(0.0, 1.0 - (ss_res / ss_tot))) if ss_tot > 1e-12 else 0.0

    # Convert to basis points relative to reference price
    effective_spread_bps = float((s_dollar / reference_price) * 10000.0)
    adverse_spread_bps = float(alpha * effective_spread_bps)
    inventory_spread_bps = float(beta * effective_spread_bps)
    processing_spread_bps = float(gamma * effective_spread_bps)

    if alpha >= 0.50:
        regime = "ADVERSE_SELECTION_DOMINANT"
    elif beta >= 0.40:
        regime = "INVENTORY_HOLDING_RISK_DOMINANT"
    elif gamma >= 0.60:
        regime = "ORDER_PROCESSING_PROFIT_DOMINANT"
    else:
        regime = "BALANCED_SPREAD_EQUILIBRIUM"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        sample_size=len(dp),
        effective_spread_bps=effective_spread_bps,
        adverse_selection_pct=alpha,
        inventory_holding_pct=beta,
        order_processing_pct=gamma,
        trade_continuation_prob=trade_continuation_prob,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_huang_stoll_receipt(payload)

    return HuangStollReceipt(
        ticker=clean_ticker,
        sample_size=len(dp),
        effective_spread_bps=effective_spread_bps,
        adverse_selection_pct=alpha,
        inventory_holding_pct=beta,
        order_processing_pct=gamma,
        adverse_spread_bps=adverse_spread_bps,
        inventory_spread_bps=inventory_spread_bps,
        processing_spread_bps=processing_spread_bps,
        trade_continuation_prob=trade_continuation_prob,
        r_squared=r_squared,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_huang_stoll_from_prices(
    prices: Sequence[float] | np.ndarray,
    *,
    ticker: str = "SPY",
) -> HuangStollReceipt:
    """Infer Huang & Stoll (1997) spread decomposition from price history using tick-rule trade signs."""
    clean_ticker = _validate_ticker(ticker)
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 32:
        raise ValueError(f"At least 32 price observations required, got {len(arr)}")

    dp = np.diff(arr)
    # Trade direction indicator via Lee-Ready tick test proxy
    q = np.where(dp >= 0, 1.0, -1.0)
    ref_price = float(arr[-1])

    return calculate_huang_stoll_decomposition(
        dp,
        q,
        ticker=clean_ticker,
        reference_price=ref_price,
    )


def persist_huang_stoll_record(receipt: HuangStollReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist Huang & Stoll spread decomposition receipt into SQLite store under parameterized query."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS huang_stoll_spread_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                effective_spread_bps REAL NOT NULL,
                adverse_selection_pct REAL NOT NULL,
                inventory_holding_pct REAL NOT NULL,
                order_processing_pct REAL NOT NULL,
                adverse_spread_bps REAL NOT NULL,
                inventory_spread_bps REAL NOT NULL,
                processing_spread_bps REAL NOT NULL,
                trade_continuation_prob REAL NOT NULL,
                r_squared REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO huang_stoll_spread_records (
                ticker, sample_size, effective_spread_bps,
                adverse_selection_pct, inventory_holding_pct, order_processing_pct,
                adverse_spread_bps, inventory_spread_bps, processing_spread_bps,
                trade_continuation_prob, r_squared, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.effective_spread_bps,
                receipt.adverse_selection_pct,
                receipt.inventory_holding_pct,
                receipt.order_processing_pct,
                receipt.adverse_spread_bps,
                receipt.inventory_spread_bps,
                receipt.processing_spread_bps,
                receipt.trade_continuation_prob,
                receipt.r_squared,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

