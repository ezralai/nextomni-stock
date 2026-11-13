r"""Patrick S. Hagan et al. (2002) SABR Stochastic Volatility Smile Calibration Engine.

Implements Patrick S. Hagan, Deep Kumar, Andrew S. Lesniewski, and Diana E. Woodward (2002),
"Managing Smile Risk", Wilmott Magazine, September issue, pp. 84-108.

- Forward price and stochastic volatility dynamics:
      dF_t = alpha_t * F_t^beta * dW_t^{(1)}
      dalpha_t = nu * alpha_t * dW_t^{(2)}
      d<W^{(1)}, W^{(2)}>_t = rho * dt
  subject to alpha > 0, beta in [0, 1], rho in (-1, 1), and nu >= 0.
- Closed-form asymptotic Black-76 implied volatility expansion:
      sigma_B(K, F) = (alpha / ((F * K)^((1 - beta)/2) * [1 + ((1 - beta)^2 / 24) * ln^2(F/K) + ((1 - beta)^4 / 1920) * ln^4(F/K)]))
                    * (z / chi(z))
                    * [1 + (((1 - beta)^2 / 24) * alpha^2 / (F * K)^(1 - beta) + (1/4) * (rho * beta * nu * alpha) / (F * K)^((1 - beta)/2) + ((2 - 3 * rho^2)/24) * nu^2) * T]
  where:
      z = (nu / alpha) * (F * K)^((1 - beta)/2) * ln(F/K)
      chi(z) = ln((sqrt(1 - 2 * rho * z + z^2) + z - rho) / (1 - rho))
- At-the-money (ATM) limit (K -> F):
      sigma_ATM = (alpha / F^(1 - beta)) * [1 + (((1 - beta)^2 / 24) * alpha^2 / F^(2 * (1 - beta)) + (1/4) * (rho * beta * nu * alpha) / F^(1 - beta) + ((2 - 3 * rho^2)/24) * nu^2) * T]
- Smooth asymptotic series expansion for z -> 0 (|z| < 1e-5):
      z / chi(z) ~ 1 - 0.5 * rho * z + ((2 - 3 * rho^2) / 24) * z^2
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
from scipy.optimize import minimize

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-sabr-2002-audit-key"


@dataclass(frozen=True)
class SABRReceipt:
    """Immutable receipt for Hagan et al. (2002) SABR Stochastic Volatility smile evaluation."""

    ticker: str
    forward_price: float
    time_to_maturity: float
    alpha: float
    beta: float
    rho: float
    nu: float
    atm_vol: float
    otm_put_vol: float
    otm_call_vol: float
    skew_slope: float
    smile_convexity: float
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
    forward_price: float,
    time_to_maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    atm_vol: float,
    skew_slope: float,
    smile_convexity: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{forward_price:.4f}|{time_to_maturity:.4f}|{alpha:.6f}|"
        f"{beta:.4f}|{rho:.4f}|{nu:.4f}|{atm_vol:.6f}|{skew_slope:.6f}|"
        f"{smile_convexity:.6f}|{regime}|{timestamp}"
    )


def sign_sabr_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_sabr_receipt(payload)
    return hmac.compare_digest(expected, signature)


def hagan_sabr_atm_volatility(
    forward: float,
    time_to_maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute the analytical at-the-money (K = F) Black-76 implied volatility."""
    f = float(forward)
    t = float(time_to_maturity)
    a = float(alpha)
    b = float(beta)
    r = float(rho)
    v = float(nu)

    if f <= 0.0:
        raise ValueError(f"Forward price must be positive, got {f}")
    if t <= 0.0:
        raise ValueError(f"Time to maturity must be positive, got {t}")
    if a <= 0.0:
        raise ValueError(f"Alpha (initial volatility) must be positive, got {a}")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"Beta must be in [0, 1], got {b}")
    if not (-1.0 < r < 1.0):
        raise ValueError(f"Rho must be strictly in (-1, 1), got {r}")
    if v < 0.0:
        raise ValueError(f"Nu (vol-of-vol) must be non-negative, got {v}")

    one_minus_b = 1.0 - b
    f_pow = f ** one_minus_b
    prefactor = a / f_pow

    term1 = (one_minus_b ** 2 / 24.0) * (a ** 2) / (f ** (2.0 * one_minus_b))
    term2 = 0.25 * (r * b * v * a) / f_pow
    term3 = ((2.0 - 3.0 * (r ** 2)) / 24.0) * (v ** 2)
    correction = 1.0 + (term1 + term2 + term3) * t

    return float(prefactor * correction)


def hagan_sabr_implied_volatility(
    forward: float,
    strike: float,
    time_to_maturity: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
) -> float:
    """Compute Hagan et al. (2002) asymptotic Black-76 implied volatility."""
    f = float(forward)
    k = float(strike)
    t = float(time_to_maturity)
    a = float(alpha)
    b = float(beta)
    r = float(rho)
    v = float(nu)

    if f <= 0.0 or k <= 0.0:
        raise ValueError(f"Forward and strike prices must be positive, got F={f}, K={k}")
    if t <= 0.0:
        raise ValueError(f"Time to maturity must be positive, got {t}")
    if a <= 0.0:
        raise ValueError(f"Alpha must be positive, got {a}")
    if not (0.0 <= b <= 1.0):
        raise ValueError(f"Beta must be in [0, 1], got {b}")
    if not (-1.0 < r < 1.0):
        raise ValueError(f"Rho must be strictly in (-1, 1), got {r}")
    if v < 0.0:
        raise ValueError(f"Nu must be non-negative, got {v}")

    # Exact ATM branch
    if math.isclose(f, k, rel_tol=1e-8, abs_tol=1e-8):
        return hagan_sabr_atm_volatility(f, t, a, b, r, v)

    one_minus_b = 1.0 - b
    fk = f * k
    log_fk = math.log(f / k)
    log_fk_sq = log_fk ** 2

    # Denominator factor 1: (F * K)^((1 - beta)/2) * [1 + ((1 - beta)^2 / 24) * ln^2(F/K) + ((1 - beta)^4 / 1920) * ln^4(F/K)]
    denom1 = (fk ** (one_minus_b / 2.0)) * (
        1.0
        + (one_minus_b ** 2 / 24.0) * log_fk_sq
        + (one_minus_b ** 4 / 1920.0) * (log_fk_sq ** 2)
    )

    # z argument
    z = (v / a) * (fk ** (one_minus_b / 2.0)) * log_fk

    # z / chi(z) with small-z Taylor series expansion for stability
    if abs(z) < 1e-5:
        z_over_chi = 1.0 - 0.5 * r * z + ((2.0 - 3.0 * (r ** 2)) / 24.0) * (z ** 2)
    else:
        arg = (math.sqrt(1.0 - 2.0 * r * z + z ** 2) + z - r) / (1.0 - r)
        if arg <= 0.0:
            z_over_chi = 1.0
        else:
            chi_z = math.log(arg)
            z_over_chi = z / chi_z

    # Higher-order time correction term
    term1 = (one_minus_b ** 2 / 24.0) * (a ** 2) / (fk ** one_minus_b)
    term2 = 0.25 * (r * b * v * a) / (fk ** (one_minus_b / 2.0))
    term3 = ((2.0 - 3.0 * (r ** 2)) / 24.0) * (v ** 2)
    correction = 1.0 + (term1 + term2 + term3) * t

    implied_vol = (a / denom1) * z_over_chi * correction
    return float(max(1e-4, implied_vol))


def calculate_sabr_profile(
    *,
    forward_price: float = 100.0,
    time_to_maturity: float = 0.50,
    alpha: float = 0.20,
    beta: float = 0.50,
    rho: float = -0.30,
    nu: float = 0.40,
    ticker: str = "SPY",
) -> SABRReceipt:
    """Calculate SABR volatility smile metrics and return an immutable audit receipt."""
    clean_ticker = _validate_ticker(ticker)
    f = float(forward_price)
    t = float(time_to_maturity)
    a = float(alpha)
    b = float(beta)
    r = float(rho)
    v = float(nu)

    atm_vol = hagan_sabr_atm_volatility(f, t, a, b, r, v)
    put_strike = 0.90 * f
    call_strike = 1.10 * f

    otm_put_vol = hagan_sabr_implied_volatility(f, put_strike, t, a, b, r, v)
    otm_call_vol = hagan_sabr_implied_volatility(f, call_strike, t, a, b, r, v)

    # Skew slope: dVol / d(ln(K/F)) proxy via 10% OTM wings
    skew_slope = float((otm_call_vol - otm_put_vol) / (math.log(call_strike / f) - math.log(put_strike / f)))
    # Smile convexity (butterfly proxy): (Vol_put + Vol_call - 2 * Vol_atm)
    smile_convexity = float(otm_put_vol + otm_call_vol - 2.0 * atm_vol)

    if r <= -0.40 and v >= 0.40:
        regime = "STEEP_NEGATIVE_SKEW_CRASH_FEAR"
    elif r >= 0.20:
        regime = "POSITIVE_SKEW_BUBBLE_REGIME"
    elif smile_convexity > 0.04:
        regime = "HIGH_CONVEXITY_SMILE"
    elif abs(skew_slope) < 0.05 and smile_convexity < 0.01:
        regime = "FLAT_SABR_SURFACE"
    else:
        regime = "MODERATE_EQUITY_SKEW"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        forward_price=f,
        time_to_maturity=t,
        alpha=a,
        beta=b,
        rho=r,
        nu=v,
        atm_vol=atm_vol,
        skew_slope=skew_slope,
        smile_convexity=smile_convexity,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_sabr_receipt(payload)

    return SABRReceipt(
        ticker=clean_ticker,
        forward_price=f,
        time_to_maturity=t,
        alpha=a,
        beta=b,
        rho=r,
        nu=v,
        atm_vol=atm_vol,
        otm_put_vol=otm_put_vol,
        otm_call_vol=otm_call_vol,
        skew_slope=skew_slope,
        smile_convexity=smile_convexity,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_sabr_from_history(
    prices: Sequence[float] | np.ndarray,
    *,
    ticker: str = "SPY",
    time_to_maturity: float = 0.50,
    beta: float = 0.50,
) -> SABRReceipt:
    """Infer SABR parameters from historical asset prices and return an audit receipt."""
    clean_ticker = _validate_ticker(ticker)
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 25:
        raise ValueError(f"At least 25 price observations required, got {len(arr)}")

    log_returns = np.diff(np.log(arr))
    realized_vol = float(np.std(log_returns, ddof=1) * math.sqrt(252))

    # Return skewness proxy for correlation rho
    m3 = float(np.mean((log_returns - np.mean(log_returns)) ** 3))
    s3 = float(np.std(log_returns, ddof=1) ** 3)
    sample_skew = (m3 / s3) if s3 > 1e-12 else 0.0
    rho_est = float(np.clip(sample_skew * 0.45, -0.85, 0.85))

    # Volatility of volatility proxy from absolute returns
    abs_returns = np.abs(log_returns)
    vol_of_vol_est = float(np.clip(np.std(abs_returns, ddof=1) * math.sqrt(252) * 1.5, 0.10, 1.20))

    forward = float(arr[-1])
    # Alpha estimate from realized vol: alpha ~ sigma_realized * F^(1 - beta)
    alpha_est = float(max(0.01, realized_vol * (forward ** (1.0 - beta))))

    return calculate_sabr_profile(
        forward_price=forward,
        time_to_maturity=time_to_maturity,
        alpha=alpha_est,
        beta=beta,
        rho=rho_est,
        nu=vol_of_vol_est,
        ticker=clean_ticker,
    )


def persist_sabr_record(receipt: SABRReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist SABR volatility receipt into SQLite store under parameterized query."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sabr_volatility_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                forward_price REAL NOT NULL,
                time_to_maturity REAL NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                rho REAL NOT NULL,
                nu REAL NOT NULL,
                atm_vol REAL NOT NULL,
                otm_put_vol REAL NOT NULL,
                otm_call_vol REAL NOT NULL,
                skew_slope REAL NOT NULL,
                smile_convexity REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO sabr_volatility_records (
                ticker, forward_price, time_to_maturity, alpha, beta, rho, nu,
                atm_vol, otm_put_vol, otm_call_vol, skew_slope, smile_convexity,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.forward_price,
                receipt.time_to_maturity,
                receipt.alpha,
                receipt.beta,
                receipt.rho,
                receipt.nu,
                receipt.atm_vol,
                receipt.otm_put_vol,
                receipt.otm_call_vol,
                receipt.skew_slope,
                receipt.smile_convexity,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

