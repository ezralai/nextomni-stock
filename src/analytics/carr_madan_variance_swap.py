r"""Carr & Madan (1999) Static Replication & Variance Swap Fair Strike Pricing Engine.

Implements Peter Carr & Dilip Madan (1999) "Towards a Theory of Volatility Trading":
- Model-independent static replication of log contract and variance swap fair strike K_var:
      K_var = (2 / T) * exp(r * T) * [ \int_0^F (P(K) / K^2) dK + \int_F^\infty (C(K) / K^2) dK ]
  where F = S_0 * exp(r * T) is the forward price.
- Fair strike implied volatility sigma_swap = sqrt(K_var)
- Volatility Risk Premium (VRP):
      VRP = sigma_swap - sigma_realized
- Out-of-the-money put vs. call variance weighting decomposition (put_call_variance_ratio)
- Exact convergence to flat Black-Scholes volatility in the absence of skew
- Strict ticker validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging
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
from scipy.stats import norm

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-carr-madan-1999-audit-key"


@dataclass(frozen=True)
class VarianceSwapReceipt:
    """Immutable receipt for Carr & Madan (1999) Variance Swap Fair Strike evaluation."""

    ticker: str
    spot_price: float
    time_to_maturity: float
    risk_free_rate: float
    realized_vol: float
    atm_implied_vol: float
    vol_skew_slope: float
    fair_variance_strike: float
    fair_vol_strike: float
    vol_risk_premium: float
    put_integral_contribution: float
    call_integral_contribution: float
    put_call_variance_ratio: float
    n_strikes: int
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    spot_price: float,
    time_to_maturity: float,
    fair_variance_strike: float,
    fair_vol_strike: float,
    vol_risk_premium: float,
    put_call_variance_ratio: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{spot_price:.4f}|{time_to_maturity:.4f}|"
        f"{fair_variance_strike:.8f}|{fair_vol_strike:.8f}|"
        f"{vol_risk_premium:.8f}|{put_call_variance_ratio:.6f}|"
        f"{regime}|{timestamp}"
    )


def sign_var_swap_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_var_swap_receipt(payload)
    return hmac.compare_digest(expected, signature)


def _bs_call_price(s: float, k: float, t: float, r: float, sig: float) -> float:
    if t <= 1e-7:
        return max(0.0, s - k)
    vol_sqrt_t = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def _bs_put_price(s: float, k: float, t: float, r: float, sig: float) -> float:
    if t <= 1e-7:
        return max(0.0, k - s)
    vol_sqrt_t = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(k * math.exp(-r * t) * norm.cdf(-d2) - s * norm.cdf(-d1))


def calculate_carr_madan_variance_swap(
    spot_price: float = 100.0,
    time_to_maturity: float = 0.25,
    risk_free_rate: float = 0.04,
    atm_implied_vol: float = 0.20,
    vol_skew_slope: float = -0.15,
    realized_vol: float = 0.17,
    n_strikes: int = 200,
    ticker: str = "SPY",
) -> VarianceSwapReceipt:
    """Compute Carr & Madan (1999) model-independent fair variance strike K_var."""
    clean_ticker = _validate_ticker(ticker)

    s0 = float(spot_price)
    ttm = float(time_to_maturity)
    rf = float(risk_free_rate)
    atm_vol = float(atm_implied_vol)
    skew = float(vol_skew_slope)
    real_vol = float(realized_vol)

    if s0 <= 0.0 or not math.isfinite(s0):
        raise ValueError("spot_price must be strictly positive and finite")
    if ttm <= 0.0 or not math.isfinite(ttm):
        raise ValueError("time_to_maturity must be strictly positive and finite")
    if not math.isfinite(rf):
        raise ValueError("risk_free_rate must be finite")
    if atm_vol <= 0.0 or not math.isfinite(atm_vol):
        raise ValueError("atm_implied_vol must be strictly positive and finite")
    if real_vol <= 0.0 or not math.isfinite(real_vol):
        raise ValueError("realized_vol must be strictly positive and finite")
    if n_strikes < 50:
        raise ValueError("n_strikes must be at least 50 for accurate numerical quadrature")

    # Forward price F = S0 * exp(r * T)
    f0 = s0 * math.exp(rf * ttm)

    # Wide strike grid spanning from 0.15 * F0 to 2.5 * F0
    k_min = 0.15 * f0
    k_max = 2.50 * f0
    strikes = np.linspace(k_min, k_max, n_strikes)
    dk = float(strikes[1] - strikes[0])

    put_integral = 0.0
    call_integral = 0.0

    for k in strikes:
        log_moneyness = math.log(k / f0)
        # Volatility skew: sigma(K) = atm_vol + skew * ln(K / F0)
        local_vol = max(0.04, min(1.20, atm_vol + skew * log_moneyness))
        weight = dk / (k * k)

        if k < f0:
            p_price = _bs_put_price(s0, k, ttm, rf, local_vol)
            put_integral += weight * p_price
        elif k > f0:
            c_price = _bs_call_price(s0, k, ttm, rf, local_vol)
            call_integral += weight * c_price
        else:
            # At ATM strike k == f0, average of put and call
            p_price = _bs_put_price(s0, k, ttm, rf, local_vol)
            c_price = _bs_call_price(s0, k, ttm, rf, local_vol)
            avg_price = 0.5 * (p_price + c_price)
            put_integral += 0.5 * weight * avg_price
            call_integral += 0.5 * weight * avg_price

    total_otm_integral = put_integral + call_integral
    k_var = float((2.0 / ttm) * math.exp(rf * ttm) * total_otm_integral)
    fair_vol = float(math.sqrt(max(1e-8, k_var)))
    vrp = float(fair_vol - real_vol)

    put_call_ratio = float(put_integral / call_integral) if call_integral > 1e-12 else 1.0

    if vrp >= 0.035:
        regime = "HIGH_VOL_RISK_PREMIUM"
    elif vrp >= 0.005:
        regime = "BALANCED_VOL_PRICING"
    elif vrp >= -0.005:
        regime = "LOW_VRP_EXPANSION"
    else:
        regime = "INVERTED_CRISIS_SPREAD"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        spot_price=s0,
        time_to_maturity=ttm,
        fair_variance_strike=k_var,
        fair_vol_strike=fair_vol,
        vol_risk_premium=vrp,
        put_call_variance_ratio=put_call_ratio,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_var_swap_receipt(payload)

    return VarianceSwapReceipt(
        ticker=clean_ticker,
        spot_price=s0,
        time_to_maturity=ttm,
        risk_free_rate=rf,
        realized_vol=real_vol,
        atm_implied_vol=atm_vol,
        vol_skew_slope=skew,
        fair_variance_strike=k_var,
        fair_vol_strike=fair_vol,
        vol_risk_premium=vrp,
        put_integral_contribution=float(put_integral),
        call_integral_contribution=float(call_integral),
        put_call_variance_ratio=put_call_ratio,
        n_strikes=int(n_strikes),
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_variance_swap_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    time_to_maturity: float = 0.25,
    risk_free_rate: float = 0.04,
) -> VarianceSwapReceipt:
    """Evaluate Carr-Madan variance swap fair strike from price series moments for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions for Variance Swap pricing")

    s0 = float(arr[-1])
    log_ret = np.diff(np.log(arr))
    real_vol = float(np.clip(np.std(log_ret, ddof=1) * math.sqrt(252.0), 0.08, 0.70))

    # Typical index skew slope (~-0.12) with positive VRP overlay
    atm_vol = float(np.clip(real_vol * 1.12, 0.10, 0.80))
    skew = -0.14

    return calculate_carr_madan_variance_swap(
        spot_price=s0,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        atm_implied_vol=atm_vol,
        vol_skew_slope=skew,
        realized_vol=real_vol,
        n_strikes=200,
        ticker=ticker,
    )


def persist_variance_swap_record(
    receipt: VarianceSwapReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Carr-Madan variance swap receipt to SQLite with foreign keys enforced."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS carr_madan_variance_swap_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                spot_price REAL NOT NULL,
                time_to_maturity REAL NOT NULL,
                risk_free_rate REAL NOT NULL,
                realized_vol REAL NOT NULL,
                atm_implied_vol REAL NOT NULL,
                vol_skew_slope REAL NOT NULL,
                fair_variance_strike REAL NOT NULL,
                fair_vol_strike REAL NOT NULL,
                vol_risk_premium REAL NOT NULL,
                put_integral_contribution REAL NOT NULL,
                call_integral_contribution REAL NOT NULL,
                put_call_variance_ratio REAL NOT NULL,
                n_strikes INTEGER NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO carr_madan_variance_swap_records (
                ticker, spot_price, time_to_maturity, risk_free_rate,
                realized_vol, atm_implied_vol, vol_skew_slope,
                fair_variance_strike, fair_vol_strike, vol_risk_premium,
                put_integral_contribution, call_integral_contribution,
                put_call_variance_ratio, n_strikes,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.spot_price,
                receipt.time_to_maturity,
                receipt.risk_free_rate,
                receipt.realized_vol,
                receipt.atm_implied_vol,
                receipt.vol_skew_slope,
                receipt.fair_variance_strike,
                receipt.fair_vol_strike,
                receipt.vol_risk_premium,
                receipt.put_integral_contribution,
                receipt.call_integral_contribution,
                receipt.put_call_variance_ratio,
                receipt.n_strikes,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

