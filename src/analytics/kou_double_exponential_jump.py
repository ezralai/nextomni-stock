r"""Kou (2002) Double Exponential Jump-Diffusion Option Pricing Engine.

Implements Steven G. Kou (2002) "A Jump-Diffusion Model for Option Pricing", Management Science:
- Asset price dynamics:
      dS_t / S_{t-} = (r - lambda * zeta) dt + sigma dW_t + (J - 1) dN_t
  where N_t is Poisson with intensity lambda, and Y = ln(J) has asymmetric double exponential density:
      f_Y(y) = p * eta1 * exp(-eta1 * y) * 1_{y >= 0} + (1 - p) * eta2 * exp(eta2 * y) * 1_{y < 0}
  subject to eta1 > 1 (finite mean jump) and eta2 > 0.
- Mean percentage jump:
      zeta = E[J - 1] = p * (eta1 / (eta1 - 1)) + (1 - p) * (eta2 / (eta2 + 1)) - 1
- Characteristic function phi_T(u) and Fourier inversion via Gauss-Legendre quadrature
- Benchmark comparison against Black-Scholes (lambda -> 0 limit) and jump risk premium
- Strict ticker validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging
"""

from __future__ import annotations

import cmath
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
DEFAULT_HMAC_KEY = "nextomni-stock-kou-2002-audit-key"


@dataclass(frozen=True)
class KouJumpReceipt:
    """Immutable receipt for Kou (2002) Double Exponential Jump-Diffusion option pricing."""

    ticker: str
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    diffusion_vol: float
    jump_intensity: float
    prob_up_jump: float
    eta1_up: float
    eta2_down: float
    compensated_drift: float
    kou_call_price: float
    kou_put_price: float
    bs_benchmark_call: float
    jump_premium: float
    atm_skew: float
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
    strike_price: float,
    time_to_maturity: float,
    kou_call_price: float,
    kou_put_price: float,
    bs_benchmark_call: float,
    jump_premium: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{spot_price:.4f}|{strike_price:.4f}|{time_to_maturity:.4f}|"
        f"{kou_call_price:.8f}|{kou_put_price:.8f}|{bs_benchmark_call:.8f}|"
        f"{jump_premium:.8f}|{regime}|{timestamp}"
    )


def sign_kou_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_kou_receipt(payload)
    return hmac.compare_digest(expected, signature)


def _kou_characteristic_function(
    u: complex,
    t: float,
    r: float,
    sigma: float,
    lam: float,
    p: float,
    eta1: float,
    eta2: float,
    zeta: float,
) -> complex:
    """Characteristic function E[exp(i * u * ln(S_T / S_0))] under Kou (2002)."""
    # Jump characteristic function phi_Y(u) = p * eta1 / (eta1 - i*u) + (1-p) * eta2 / (eta2 + i*u)
    iu = 1j * u
    phi_y = p * eta1 / (eta1 - iu) + (1.0 - p) * eta2 / (eta2 + iu)
    drift_term = iu * (r - 0.5 * sigma * sigma - lam * zeta)
    diff_term = -0.5 * (sigma ** 2) * (u ** 2)
    jump_term = lam * (phi_y - 1.0)
    psi = drift_term + diff_term + jump_term
    return cmath.exp(t * psi)


def _bs_call(s: float, k: float, t: float, r: float, sig: float) -> float:
    if t <= 1e-7:
        return max(0.0, s - k)
    v_sqrt_t = sig * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * sig * sig) * t) / v_sqrt_t
    d2 = d1 - v_sqrt_t
    return float(s * norm.cdf(d1) - k * math.exp(-r * t) * norm.cdf(d2))


def calculate_kou_option_price(
    spot_price: float = 100.0,
    strike_price: float = 100.0,
    time_to_maturity: float = 0.50,
    risk_free_rate: float = 0.04,
    diffusion_vol: float = 0.16,
    jump_intensity: float = 0.75,
    prob_up_jump: float = 0.35,
    eta1_up: float = 10.0,
    eta2_down: float = 5.0,
    ticker: str = "SPY",
) -> KouJumpReceipt:
    """Price European call & put options under Kou (2002) Double Exponential Jump-Diffusion."""
    clean_ticker = _validate_ticker(ticker)

    s0 = float(spot_price)
    k_strike = float(strike_price)
    ttm = float(time_to_maturity)
    rf = float(risk_free_rate)
    sig = float(diffusion_vol)
    lam = float(jump_intensity)
    p_up = float(prob_up_jump)
    e1 = float(eta1_up)
    e2 = float(eta2_down)

    if s0 <= 0.0 or not math.isfinite(s0):
        raise ValueError("spot_price must be strictly positive and finite")
    if k_strike <= 0.0 or not math.isfinite(k_strike):
        raise ValueError("strike_price must be strictly positive and finite")
    if ttm <= 0.0 or not math.isfinite(ttm):
        raise ValueError("time_to_maturity must be strictly positive and finite")
    if not math.isfinite(rf):
        raise ValueError("risk_free_rate must be finite")
    if sig <= 0.0 or not math.isfinite(sig):
        raise ValueError("diffusion_vol must be strictly positive and finite")
    if lam < 0.0 or not math.isfinite(lam):
        raise ValueError("jump_intensity must be non-negative and finite")
    if not (0.0 <= p_up <= 1.0):
        raise ValueError("prob_up_jump must be in [0.0, 1.0]")
    if e1 <= 1.0 or not math.isfinite(e1):
        raise ValueError("eta1_up must be strictly greater than 1.0 for finite mean jump")
    if e2 <= 0.0 or not math.isfinite(e2):
        raise ValueError("eta2_down must be strictly positive and finite")

    # Compensated jump drift zeta = E[e^Y - 1]
    zeta = p_up * (e1 / (e1 - 1.0)) + (1.0 - p_up) * (e2 / (e2 + 1.0)) - 1.0

    # Gauss-Legendre quadrature for Gil-Pelaez inversion over [0, U_max]
    u_max = 180.0
    n_quad = 128
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    # Map [-1, 1] to [0, u_max]
    u_vals = 0.5 * (nodes + 1.0) * u_max
    w_vals = 0.5 * weights * u_max

    ln_k_s0 = math.log(k_strike / s0)
    exp_minus_rt = math.exp(-rf * ttm)

    int1 = 0.0
    int2 = 0.0

    for u, w in zip(u_vals, w_vals):
        if u < 1e-8:
            continue
        # Characteristic functions
        phi2 = _kou_characteristic_function(complex(u, 0.0), ttm, rf, sig, lam, p_up, e1, e2, zeta)
        phi1 = _kou_characteristic_function(complex(u, -1.0), ttm, rf, sig, lam, p_up, e1, e2, zeta)

        # Integrand for Pi2: Re(phi(u) * exp(-i * u * ln(K/S0)) / (i * u))
        phase = cmath.exp(-1j * u * ln_k_s0)
        integrand2 = ((phi2 * phase) / (1j * u)).real
        int2 += w * integrand2

        # Integrand for Pi1: Re(exp(-r*T) * phi(u - i) * exp(-i * u * ln(K/S0)) / (i * u))
        integrand1 = ((exp_minus_rt * phi1 * phase) / (1j * u)).real
        int1 += w * integrand1

    pi2 = float(np.clip(0.5 + (1.0 / math.pi) * int2, 0.0, 1.0))
    pi1 = float(np.clip(0.5 + (1.0 / math.pi) * int1, 0.0, 1.0))

    call_price = float(max(0.0, s0 * pi1 - k_strike * exp_minus_rt * pi2))
    # Put price via Put-Call Parity: P = C - S0 + K * exp(-r * T)
    put_price = float(max(0.0, call_price - s0 + k_strike * exp_minus_rt))

    # Black-Scholes benchmark with pure diffusion volatility sigma
    bs_call_px = _bs_call(s0, k_strike, ttm, rf, sig)
    jump_prem = float(call_price - bs_call_px)

    # ATM Skew proxy: (P(0.95*S) - C(1.05*S)) / S0
    c_otm = float(max(0.0, s0 * pi1 - (1.05 * s0) * exp_minus_rt * pi2))
    p_otm = float(max(0.0, _bs_call(s0, 0.95 * s0, ttm, rf, sig) - s0 + 0.95 * s0 * exp_minus_rt))
    atm_skew = float((p_otm - c_otm) / s0)

    if lam >= 0.50 and p_up <= 0.40 and e2 <= 8.0:
        regime = "HIGH_ASYMMETRIC_CRASH_RISK"
    elif e1 <= 3.0 or e2 <= 3.0:
        regime = "EXTREME_TAIL_CONVEXITY"
    elif lam > 0.15:
        regime = "BALANCED_JUMP_DIFFUSION"
    else:
        regime = "DIFFUSION_DOMINANT"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        spot_price=s0,
        strike_price=k_strike,
        time_to_maturity=ttm,
        kou_call_price=call_price,
        kou_put_price=put_price,
        bs_benchmark_call=bs_call_px,
        jump_premium=jump_prem,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_kou_receipt(payload)

    return KouJumpReceipt(
        ticker=clean_ticker,
        spot_price=s0,
        strike_price=k_strike,
        time_to_maturity=ttm,
        risk_free_rate=rf,
        diffusion_vol=sig,
        jump_intensity=lam,
        prob_up_jump=p_up,
        eta1_up=e1,
        eta2_down=e2,
        compensated_drift=float(zeta),
        kou_call_price=call_price,
        kou_put_price=put_price,
        bs_benchmark_call=float(bs_call_px),
        jump_premium=jump_prem,
        atm_skew=atm_skew,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_kou_jump_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    time_to_maturity: float = 0.50,
    risk_free_rate: float = 0.04,
) -> KouJumpReceipt:
    """Calibrate Kou jump-diffusion profile from empirical price series for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions for Kou jump option pricing")

    s0 = float(arr[-1])
    log_ret = np.diff(np.log(arr))
    ann_vol = float(np.clip(np.std(log_ret, ddof=1) * math.sqrt(252.0), 0.08, 0.60))

    # Realized skewness proxy for asymmetric jump intensity
    skew_sample = float(np.mean((log_ret - np.mean(log_ret)) ** 3) / max(1e-12, np.std(log_ret, ddof=1) ** 3))
    p_up = float(np.clip(0.5 + 0.15 * skew_sample, 0.20, 0.60))

    return calculate_kou_option_price(
        spot_price=s0,
        strike_price=s0,
        time_to_maturity=time_to_maturity,
        risk_free_rate=risk_free_rate,
        diffusion_vol=ann_vol * 0.85,
        jump_intensity=0.80,
        prob_up_jump=p_up,
        eta1_up=12.0,
        eta2_down=6.0,
        ticker=ticker,
    )


def persist_kou_record(
    receipt: KouJumpReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Kou jump-diffusion option pricing receipt to SQLite with foreign keys enforced."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS kou_jump_diffusion_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                spot_price REAL NOT NULL,
                strike_price REAL NOT NULL,
                time_to_maturity REAL NOT NULL,
                risk_free_rate REAL NOT NULL,
                diffusion_vol REAL NOT NULL,
                jump_intensity REAL NOT NULL,
                prob_up_jump REAL NOT NULL,
                eta1_up REAL NOT NULL,
                eta2_down REAL NOT NULL,
                compensated_drift REAL NOT NULL,
                kou_call_price REAL NOT NULL,
                kou_put_price REAL NOT NULL,
                bs_benchmark_call REAL NOT NULL,
                jump_premium REAL NOT NULL,
                atm_skew REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO kou_jump_diffusion_records (
                ticker, spot_price, strike_price, time_to_maturity, risk_free_rate,
                diffusion_vol, jump_intensity, prob_up_jump, eta1_up, eta2_down,
                compensated_drift, kou_call_price, kou_put_price, bs_benchmark_call,
                jump_premium, atm_skew, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.spot_price,
                receipt.strike_price,
                receipt.time_to_maturity,
                receipt.risk_free_rate,
                receipt.diffusion_vol,
                receipt.jump_intensity,
                receipt.prob_up_jump,
                receipt.eta1_up,
                receipt.eta2_down,
                receipt.compensated_drift,
                receipt.kou_call_price,
                receipt.kou_put_price,
                receipt.bs_benchmark_call,
                receipt.jump_premium,
                receipt.atm_skew,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True
