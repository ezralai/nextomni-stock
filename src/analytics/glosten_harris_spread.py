"""Lawrence R. Glosten & Lawrence E. Harris (1988) Trade-Size Dependent Spread Decomposition Engine.

Implements the structural market microstructure econometric model formulated in:
Glosten, Lawrence R., & Harris, Lawrence E. (1988).
"Estimating the Components of the Bid/Ask Spread."
Journal of Financial Economics, 21(1), 123-142.

Unlike models that assume homogeneous or binary trade size, Glosten & Harris (1988)
decomposes the effective bid-ask spread into fixed and variable (trade-size dependent)
components for both transitory order processing/inventory holding costs and
permanent adverse selection / price impact:
    Delta P_t = c_0 * Delta q_t + c_1 * Delta (q_t * V_t) + z_0 * q_t + z_1 * (q_t * V_t) + e_t
where:
    c_0: Fixed transitory order processing friction ($/share)
    c_1: Variable transitory inventory holding friction ($/share per unit volume)
    z_0: Fixed adverse selection / asymmetric information friction ($/share)
    z_1: Variable adverse selection / trade-size price impact ($/share per unit volume)
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
SECRET_SALT = os.environ.get("NEXTOMNI_AUDIT_SALT", "nextomni_glosten_harris_audit_salt").encode("utf-8")


@dataclass(frozen=True)
class GlostenHarrisReceipt:
    """Immutable receipt containing Glosten & Harris (1988) spread decomposition."""

    ticker: str
    sample_size: int
    benchmark_volume: float
    c0_fixed_transitory: float
    c1_variable_transitory: float
    z0_fixed_adverse: float
    z1_variable_adverse: float
    effective_spread_dollar: float
    effective_spread_bps: float
    transitory_spread_dollar: float
    adverse_spread_dollar: float
    adverse_selection_ratio: float
    order_processing_ratio: float
    r_squared: float
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


def calculate_glosten_harris_spread(
    price_changes: Sequence[float],
    trade_indicators: Sequence[float],
    trade_volumes: Sequence[float],
    ticker: str = "SPY",
    reference_price: float = 100.0,
    benchmark_volume: Optional[float] = None,
    derivation_mode: str = "DIRECT_TRADE_FLOW",
) -> GlostenHarrisReceipt:
    """Estimate Glosten & Harris (1988) trade-size dependent spread decomposition via OLS.

    Parameters
    ----------
    price_changes : Sequence[float]
        Delta P_t = P_t - P_{t-1} transaction price changes (length N).
    trade_indicators : Sequence[float]
        Trade direction indicators q_t in {-1, +1} (length N + 1 or N).
    trade_volumes : Sequence[float]
        Trade volumes V_t (length N + 1 or N).
    ticker : str
        Stock ticker symbol.
    reference_price : float
        Benchmark underlying price to convert dollar spreads to basis points.
    benchmark_volume : Optional[float]
        Benchmark trade volume for evaluating effective spread (defaults to median volume).
    derivation_mode : str
        Source mode ("DIRECT_TRADE_FLOW" or "SYNTHESIZED_PROXY").

    Returns
    -------
    GlostenHarrisReceipt
        Detailed econometric spread decomposition receipt.
    """
    clean_ticker = ticker.strip().upper() if TICKER_REGEX.match(ticker.strip().upper()) else "UNKNOWN"
    now_iso = datetime.now(timezone.utc).isoformat()

    dp = np.asarray(price_changes, dtype=np.float64)
    q = np.asarray(trade_indicators, dtype=np.float64)
    v = np.asarray(trade_volumes, dtype=np.float64)

    # Ensure consistent lengths
    # If q and v are of length len(dp) + 1, use current and lagged
    if len(q) == len(dp) + 1 and len(v) == len(dp) + 1:
        q_curr = q[1:]
        q_prev = q[:-1]
        v_curr = v[1:]
        v_prev = v[:-1]
    elif len(q) == len(dp) and len(v) == len(dp):
        q_curr = q[1:]
        q_prev = q[:-1]
        v_curr = v[1:]
        v_prev = v[:-1]
        dp = dp[1:]
    else:
        min_len = min(len(dp), len(q), len(v))
        if min_len < 15:
            return _empty_receipt(clean_ticker, now_iso)
        dp = dp[:min_len][1:]
        q_curr = q[:min_len][1:]
        q_prev = q[:min_len][:-1]
        v_curr = v[:min_len][1:]
        v_prev = v[:min_len][:-1]

    n_obs = len(dp)
    if n_obs < 15:
        return _empty_receipt(clean_ticker, now_iso)

    # Clean non-finite or degenerate values
    valid = np.isfinite(dp) & np.isfinite(q_curr) & np.isfinite(q_prev) & np.isfinite(v_curr) & np.isfinite(v_prev)
    dp = dp[valid]
    q_curr = q_curr[valid]
    q_prev = q_prev[valid]
    v_curr = np.maximum(v_curr[valid], 0.0)
    v_prev = np.maximum(v_prev[valid], 0.0)
    n_clean = len(dp)

    if n_clean < 15:
        return _empty_receipt(clean_ticker, now_iso)

    # Regressors:
    # X1 = Delta q_t = q_t - q_{t-1}
    # X2 = Delta (q_t * V_t) = q_t * V_t - q_{t-1} * V_{t-1}
    # X3 = q_t
    # X4 = q_t * V_t
    x1 = q_curr - q_prev
    x2 = (q_curr * v_curr) - (q_prev * v_prev)
    x3 = q_curr
    x4 = q_curr * v_curr

    X = np.column_stack([x1, x2, x3, x4])
    y = dp

    # Regularized OLS regression to ensure numerical stability
    xtx = X.T @ X
    ridge_lambda = 1e-6 * (float(np.trace(xtx)) / 4.0 if np.trace(xtx) > 0 else 1.0)
    xtx_reg = xtx + ridge_lambda * np.eye(4)
    xty = X.T @ y

    try:
        beta = np.linalg.solve(xtx_reg, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(xtx_reg) @ xty

    # Coefficients: [c0, c1, z0, z1]
    # Bound non-negative to adhere to financial viability
    c0 = max(0.0, float(beta[0]))
    c1 = max(0.0, float(beta[1]))
    z0 = max(0.0, float(beta[2]))
    z1 = max(0.0, float(beta[3]))

    # Model R-squared
    y_pred = X @ np.array([c0, c1, z0, z1], dtype=np.float64)
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_sq = max(0.0, float(1.0 - (ss_res / ss_tot))) if ss_tot > 1e-10 else 0.0

    # Benchmark volume evaluation
    if benchmark_volume is None or benchmark_volume <= 0.0:
        v_bench = float(np.median(v_curr)) if len(v_curr) > 0 else 1.0
    else:
        v_bench = float(benchmark_volume)

    # Spread components at benchmark volume
    # Half-spread: C(V) = c0 + c1 * V, Z(V) = z0 + z1 * V
    # Full spread: S(V) = 2 * (C(V) + Z(V))
    transitory_spread = 2.0 * (c0 + c1 * v_bench)
    adverse_spread = 2.0 * (z0 + z1 * v_bench)
    total_effective_spread = transitory_spread + adverse_spread

    # Relative basis points
    ref_p = max(reference_price, 1.0)
    eff_spread_bps = float((total_effective_spread / ref_p) * 10000.0)

    # Component ratios
    if total_effective_spread > 1e-6:
        adverse_ratio = float(adverse_spread / total_effective_spread)
        order_processing_ratio = float(transitory_spread / total_effective_spread)
    else:
        adverse_ratio = 0.50
        order_processing_ratio = 0.50

    # Microstructure regime
    if adverse_ratio >= 0.55:
        regime = "HIGH_ADVERSE_SELECTION"
    elif order_processing_ratio >= 0.70:
        regime = "ORDER_PROCESSING_DOMINATED"
    elif z1 > 0.001 or c1 > 0.001:
        regime = "STRONG_VOLUME_SENSITIVITY"
    else:
        regime = "BALANCED_VOLUME_IMPACT"

    payload = (
        f"{clean_ticker}|{n_clean}|{v_bench:.2f}|{c0:.6f}|{c1:.6f}|{z0:.6f}|{z1:.6f}|"
        f"{total_effective_spread:.4f}|{eff_spread_bps:.2f}|{adverse_ratio:.4f}|"
        f"{order_processing_ratio:.4f}|{r_sq:.4f}|{regime}|{derivation_mode}|{now_iso}"
    )
    sig = _compute_hmac(payload)

    return GlostenHarrisReceipt(
        ticker=clean_ticker,
        sample_size=n_clean,
        benchmark_volume=v_bench,
        c0_fixed_transitory=c0,
        c1_variable_transitory=c1,
        z0_fixed_adverse=z0,
        z1_variable_adverse=z1,
        effective_spread_dollar=total_effective_spread,
        effective_spread_bps=eff_spread_bps,
        transitory_spread_dollar=transitory_spread,
        adverse_spread_dollar=adverse_spread,
        adverse_selection_ratio=adverse_ratio,
        order_processing_ratio=order_processing_ratio,
        r_squared=r_sq,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig,
        timestamp=now_iso,
    )


def evaluate_glosten_harris_from_prices(
    prices: Sequence[float],
    volumes: Optional[Sequence[float]] = None,
    ticker: str = "SPY",
    benchmark_volume: float = 1.0,
) -> GlostenHarrisReceipt:
    """Evaluate Glosten & Harris (1988) spread decomposition from price & volume history."""
    p_arr = np.asarray(prices, dtype=np.float64)
    valid_p = np.isfinite(p_arr) & (p_arr > 0.0)
    p_clean = p_arr[valid_p]
    n_pts = len(p_clean)

    if n_pts < 16:
        return _empty_receipt(ticker, datetime.now(timezone.utc).isoformat())

    dp = np.diff(p_clean)

    # Trade indicator proxy (Lee-Ready sign heuristic)
    q = np.ones(n_pts, dtype=np.float64)
    for i in range(1, n_pts):
        if p_clean[i] > p_clean[i - 1]:
            q[i] = 1.0
        elif p_clean[i] < p_clean[i - 1]:
            q[i] = -1.0
        else:
            q[i] = q[i - 1]

    # Volume proxy (normalized by median)
    if volumes is not None and len(volumes) == len(p_arr):
        v_arr = np.asarray(volumes, dtype=np.float64)[valid_p]
        med_v = float(np.median(v_arr)) if float(np.median(v_arr)) > 0 else 1.0
        v_norm = v_arr / med_v
        derivation_mode = "OBSERVED_VOLUME_PROXY"
    else:
        v_norm = np.ones(n_pts, dtype=np.float64)
        derivation_mode = "SYNTHESIZED_UNIT_VOLUME"

    ref_price = float(np.mean(p_clean))
    return calculate_glosten_harris_spread(
        price_changes=dp,
        trade_indicators=q,
        trade_volumes=v_norm,
        ticker=ticker,
        reference_price=ref_price,
        benchmark_volume=benchmark_volume,
        derivation_mode=derivation_mode,
    )


def _empty_receipt(ticker: str, timestamp_iso: str) -> GlostenHarrisReceipt:
    empty_payload = f"{ticker}|0|0.0|0.0|0.0|0.0|0.0|{timestamp_iso}"
    sig = _compute_hmac(empty_payload)
    return GlostenHarrisReceipt(
        ticker=ticker,
        sample_size=0,
        benchmark_volume=1.0,
        c0_fixed_transitory=0.0,
        c1_variable_transitory=0.0,
        z0_fixed_adverse=0.0,
        z1_variable_adverse=0.0,
        effective_spread_dollar=0.0,
        effective_spread_bps=0.0,
        transitory_spread_dollar=0.0,
        adverse_spread_dollar=0.0,
        adverse_selection_ratio=0.0,
        order_processing_ratio=0.0,
        r_squared=0.0,
        regime="INSUFFICIENT_DATA",
        derivation_mode="NONE",
        hmac_signature=sig,
        timestamp=timestamp_iso,
    )


def persist_glosten_harris_record(
    receipt: GlostenHarrisReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Store Glosten & Harris spread receipt in SQLite database with audit integrity."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS glosten_harris_receipts (
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                benchmark_volume REAL NOT NULL,
                c0_fixed_transitory REAL NOT NULL,
                c1_variable_transitory REAL NOT NULL,
                z0_fixed_adverse REAL NOT NULL,
                z1_variable_adverse REAL NOT NULL,
                effective_spread_dollar REAL NOT NULL,
                effective_spread_bps REAL NOT NULL,
                transitory_spread_dollar REAL NOT NULL,
                adverse_spread_dollar REAL NOT NULL,
                adverse_selection_ratio REAL NOT NULL,
                order_processing_ratio REAL NOT NULL,
                r_squared REAL NOT NULL,
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
            INSERT OR REPLACE INTO glosten_harris_receipts (
                ticker, sample_size, benchmark_volume,
                c0_fixed_transitory, c1_variable_transitory,
                z0_fixed_adverse, z1_variable_adverse,
                effective_spread_dollar, effective_spread_bps,
                transitory_spread_dollar, adverse_spread_dollar,
                adverse_selection_ratio, order_processing_ratio,
                r_squared, regime, derivation_mode, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.benchmark_volume,
                receipt.c0_fixed_transitory,
                receipt.c1_variable_transitory,
                receipt.z0_fixed_adverse,
                receipt.z1_variable_adverse,
                receipt.effective_spread_dollar,
                receipt.effective_spread_bps,
                receipt.transitory_spread_dollar,
                receipt.adverse_spread_dollar,
                receipt.adverse_selection_ratio,
                receipt.order_processing_ratio,
                receipt.r_squared,
                receipt.regime,
                receipt.derivation_mode,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

