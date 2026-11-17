"""
Robert Almgren, Chee Thum, Emmanuel Hauptmann & Hong Li (2005)
Empirical Nonlinear Metaorder Market Impact & Implementation Shortfall Engine.

References:
    Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005).
    Direct Estimation of Equity Market Impact.
    Risk, 18(7), 58–62.

    Almgren, R., & Chriss, N. (2000).
    Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-40.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,10}$")
_DEFAULT_HMAC_SECRET = b"nextomni-almgren-thum-metaorder-2005-secret"

# Calibrated empirical constants from Citigroup execution data (Almgren et al. 2005)
DEFAULT_GAMMA_PERM = 0.314   # Permanent price impact parameter
DEFAULT_ETA_TEMP = 0.142     # Temporary price impact parameter
DEFAULT_ALPHA_EXP = 0.60     # Sublinear velocity exponent (alpha ~= 0.6)
DEFAULT_RISK_AVERSION = 1e-6 # Default trader CARA parameter lambda ($^-1)


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: AlmgrenThumReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.price:.2f}:{receipt.order_shares:.1f}:"
        f"{receipt.adv_shares:.1f}:{receipt.daily_volatility:.6f}:{receipt.horizon_days:.2f}:"
        f"{receipt.expected_shortfall_bps:.2f}:{receipt.optimal_horizon_days:.2f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class AlmgrenThumReceipt:
    ticker: str
    price: float
    order_shares: float
    adv_shares: float
    daily_volatility: float
    risk_aversion: float
    horizon_days: float
    permanent_impact_bps: float
    temporary_impact_bps: float
    expected_shortfall_bps: float
    expected_shortfall_dollar: float
    variance_shortfall_dollar: float
    optimal_horizon_days: float
    optimal_participation_pct: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> AlmgrenThumReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0.00:0.0:0.0:0.000000:0.00:0.00:0.00:ERROR_{error_msg}:{ts}"
    return AlmgrenThumReceipt(
        ticker=ticker,
        price=0.0,
        order_shares=0.0,
        adv_shares=0.0,
        daily_volatility=0.0,
        risk_aversion=0.0,
        horizon_days=0.0,
        permanent_impact_bps=0.0,
        temporary_impact_bps=0.0,
        expected_shortfall_bps=0.0,
        expected_shortfall_dollar=0.0,
        variance_shortfall_dollar=0.0,
        optimal_horizon_days=1.0,
        optimal_participation_pct=0.0,
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_almgren_thum_impact(
    price: float,
    order_shares: float,
    adv_shares: float,
    daily_volatility: float,
    horizon_days: float = 1.0,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
    gamma_perm: float = DEFAULT_GAMMA_PERM,
    eta_temp: float = DEFAULT_ETA_TEMP,
    alpha_exp: float = DEFAULT_ALPHA_EXP,
    ticker: str = "SPY",
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> AlmgrenThumReceipt:
    """
    Calculate Almgren, Thum, Hauptmann & Li (2005) nonlinear metaorder market impact,
    expected implementation shortfall, variance risk, and optimal trading horizon T*.

    Args:
        price: Current underlying stock price ($)
        order_shares: Target metaorder size in shares (Q > 0)
        adv_shares: 20-session Average Daily Volume in shares (V > 0)
        daily_volatility: Daily returns volatility (sigma > 0)
        horizon_days: Planned execution duration in trading days (T > 0)
        risk_aversion: Trader CARA parameter lambda (>= 0)
        gamma_perm: Empirical permanent impact coefficient (approx 0.314)
        eta_temp: Empirical temporary impact coefficient (approx 0.142)
        alpha_exp: Sublinear velocity exponent (approx 0.60)
        ticker: Symbol string
        derivation_mode: Calibration tag

    Returns:
        AlmgrenThumReceipt with shortfall breakdown, optimal horizon, and HMAC signature.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    p = float(price)
    q = float(order_shares)
    v = float(adv_shares)
    sig = float(daily_volatility)
    t_days = max(0.01, float(horizon_days))
    lam = max(1e-9, float(risk_aversion))

    if p <= 0.0 or q <= 0.0 or v <= 10.0 or sig <= 0.0001:
        return _empty_receipt(clean_ticker, "INVALID_INPUT")

    # Normalized participation fraction f = Q / V
    f = q / v

    # Trading velocity rate = Q / (T * V) = f / T
    v_rate = f / t_days

    # 1. Permanent Price Impact: I_perm = gamma * sigma * (Q / V)
    # in bps and in dollars/share
    i_perm_bps = gamma_perm * sig * f * 10000.0
    i_perm_dollar_per_share = p * (i_perm_bps / 10000.0)

    # 2. Temporary Price Impact: I_temp = eta * sigma * (v_rate)^alpha
    # in bps and in dollars/share
    i_temp_bps = eta_temp * sig * (v_rate ** alpha_exp) * 10000.0
    i_temp_dollar_per_share = p * (i_temp_bps / 10000.0)

    # 3. Expected Implementation Shortfall (IS)
    # Under TWAP, trader pays 0.5 * permanent impact + full temporary impact
    is_bps = 0.5 * i_perm_bps + i_temp_bps
    is_dollar_total = q * p * (is_bps / 10000.0)

    # 4. Variance of Implementation Shortfall (Timing Risk)
    # Var(IS) = (1/3) * (P * sigma)^2 * Q^2 * T
    var_shortfall = (1.0 / 3.0) * ((p * sig) ** 2) * (q ** 2) * t_days

    # 5. Optimal Trading Horizon T* (Almgren et al. 2005)
    # Minimizes U(T) = E[IS] + lambda * Var(IS)
    # T* = [ (3 * alpha * eta) / (lambda * P * sigma * Q^(1 - alpha) * V^alpha) ] ^ (1 / (1 + alpha))
    denom = lam * p * sig * (q ** (1.0 - alpha_exp)) * (v ** alpha_exp)
    if denom > 1e-15:
        base_ratio = (3.0 * alpha_exp * eta_temp) / denom
        t_star = base_ratio ** (1.0 / (1.0 + alpha_exp))
    else:
        t_star = 1.0

    # Clamp optimal horizon to realistic market bounds: 0.05 days (20 mins) to 10 days
    t_star = float(np.clip(t_star, 0.05, 10.0))
    optimal_pov_pct = float(round((f / t_star) * 100.0, 2))

    # 6. Execution Regime Classification
    if t_star < 0.25:
        regime = "RAPID_RISK_DOMINATED_EXECUTION"
    elif t_star > 2.0:
        regime = "PATIENT_IMPACT_MINIMIZING_EXECUTION"
    else:
        regime = "BALANCED_TWAP_EXECUTION"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_p = float(round(p, 2))
    r_q = float(round(q, 1))
    r_v = float(round(v, 1))
    r_sig = float(round(sig, 6))
    r_t = float(round(t_days, 2))
    r_i_perm = float(round(i_perm_bps, 2))
    r_i_temp = float(round(i_temp_bps, 2))
    r_is_bps = float(round(is_bps, 2))
    r_is_dlr = float(round(is_dollar_total, 2))
    r_var_dlr = float(round(var_shortfall, 2))
    r_t_star = float(round(t_star, 2))

    raw_payload = (
        f"{clean_ticker}:{r_p:.2f}:{r_q:.1f}:{r_v:.1f}:{r_sig:.6f}:{r_t:.2f}:"
        f"{r_is_bps:.2f}:{r_t_star:.2f}:{regime}:{ts}"
    )
    sig_hmac = _compute_hmac(raw_payload)

    return AlmgrenThumReceipt(
        ticker=clean_ticker,
        price=r_p,
        order_shares=r_q,
        adv_shares=r_v,
        daily_volatility=r_sig,
        risk_aversion=float(lam),
        horizon_days=r_t,
        permanent_impact_bps=r_i_perm,
        temporary_impact_bps=r_i_temp,
        expected_shortfall_bps=r_is_bps,
        expected_shortfall_dollar=r_is_dlr,
        variance_shortfall_dollar=r_var_dlr,
        optimal_horizon_days=r_t_star,
        optimal_participation_pct=optimal_pov_pct,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig_hmac,
        timestamp=ts,
    )


def evaluate_almgren_thum_from_order(
    prices: Sequence[float],
    volumes: Sequence[float],
    order_shares: float,
    ticker: str = "SPY",
    horizon_days: float = 1.0,
    risk_aversion: float = DEFAULT_RISK_AVERSION,
) -> AlmgrenThumReceipt:
    """
    Evaluate Almgren-Thum metaorder impact from historical price and volume time series.
    """
    p_arr = np.asarray(prices, dtype=np.float64)
    v_arr = np.asarray(volumes, dtype=np.float64)

    if len(p_arr) < 20 or len(v_arr) < 20:
        return _empty_receipt(ticker, "INSUFFICIENT_HISTORY")

    curr_price = float(p_arr[-1])
    trailing_adv = float(np.mean(v_arr[-20:]))

    log_returns = np.diff(np.log(p_arr[-60:])) if len(p_arr) >= 60 else np.diff(np.log(p_arr))
    daily_vol = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.015

    return calculate_almgren_thum_impact(
        price=curr_price,
        order_shares=order_shares,
        adv_shares=trailing_adv,
        daily_volatility=daily_vol,
        horizon_days=horizon_days,
        risk_aversion=risk_aversion,
        ticker=ticker,
        derivation_mode="EMPIRICAL_ORDER_EVALUATION",
    )


def persist_as_record(
    receipt: AlmgrenThumReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Almgren-Thum metaorder receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS almgren_thum_metaorder_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    price REAL NOT NULL,
                    order_shares REAL NOT NULL,
                    adv_shares REAL NOT NULL,
                    daily_volatility REAL NOT NULL,
                    risk_aversion REAL NOT NULL,
                    horizon_days REAL NOT NULL,
                    permanent_impact_bps REAL NOT NULL,
                    temporary_impact_bps REAL NOT NULL,
                    expected_shortfall_bps REAL NOT NULL,
                    expected_shortfall_dollar REAL NOT NULL,
                    variance_shortfall_dollar REAL NOT NULL,
                    optimal_horizon_days REAL NOT NULL,
                    optimal_participation_pct REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_almgren_thum_ticker_created
                ON almgren_thum_metaorder_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO almgren_thum_metaorder_records (
                    ticker, price, order_shares, adv_shares, daily_volatility,
                    risk_aversion, horizon_days, permanent_impact_bps,
                    temporary_impact_bps, expected_shortfall_bps,
                    expected_shortfall_dollar, variance_shortfall_dollar,
                    optimal_horizon_days, optimal_participation_pct,
                    regime, derivation_mode, hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.price,
                    receipt.order_shares,
                    receipt.adv_shares,
                    receipt.daily_volatility,
                    receipt.risk_aversion,
                    receipt.horizon_days,
                    receipt.permanent_impact_bps,
                    receipt.temporary_impact_bps,
                    receipt.expected_shortfall_bps,
                    receipt.expected_shortfall_dollar,
                    receipt.variance_shortfall_dollar,
                    receipt.optimal_horizon_days,
                    receipt.optimal_participation_pct,
                    receipt.regime,
                    receipt.derivation_mode,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()

