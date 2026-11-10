"""Steven L. Heston (1993) Stochastic Volatility Option Pricing & Characteristic Function Engine.

Theoretical Foundation:
    1. Heston, Steven L. (1993). "A Closed-Form Solution for Options with Stochastic Volatility
       with Applications to Bond and Currency Options". The Review of Financial Studies, 6(2),
       pp. 327-343.
       The canonical stochastic volatility model in quantitative derivatives pricing, generating
       volatility smiles and equity crash skews from mean-reverting stochastic variance.

    2. Albrecher, Hansjorg, Mayer, Philipp, Schoutens, Wim, & Tistaert, Jurgen (2007).
       "The Little Heston Trap". Wilmott Magazine, pp. 83-92.
       Eliminates complex-logarithm branch-cut discontinuities via decaying exponential formulation.

Key Concepts:
    - Joint Spot-Variance Dynamics under Risk-Neutral Measure Q:
          dS_t = r S_t dt + sqrt(v_t) S_t dW_t^S
          dv_t = kappa * (theta - v_t) dt + xi * sqrt(v_t) dW_t^v,  d<W^S, W^v>_t = rho dt
    - Feller Stability Condition:
          2 * kappa * theta > xi^2 ensures variance remains strictly positive (zero absorption-free).
    - Volatility Smile & Leverage Skew:
          Spot-vol correlation rho < 0 produces negative return skewness and higher OTM put implied
          volatility relative to OTM calls, while vol-of-vol xi amplifies kurtosis and wing premia.
"""

from __future__ import annotations

import cmath
import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"heston-stochastic-vol-v139-secret"

# Precomputed 64-point Gauss-Legendre quadrature nodes and weights on [-1, 1]
_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(64)


@dataclass(frozen=True)
class HestonPricingReceipt:
    """Immutable quantitative audit receipt for Heston (1993) stochastic volatility pricing."""

    ticker: str
    spot_price: float
    strike_price: float
    risk_free_rate: float
    time_to_expiry: float
    initial_variance: float           # v_0
    long_run_variance: float          # theta
    mean_reversion_kappa: float       # kappa
    vol_of_vol_xi: float              # xi
    spot_vol_corr_rho: float          # rho
    feller_ratio: float               # (2 * kappa * theta) / xi^2
    feller_condition_satisfied: bool
    heston_call_price: float
    heston_put_price: float
    bs_benchmark_call_price: float
    atm_implied_vol: float
    otm_put_implied_vol: float        # K = 0.85 * S_0
    otm_call_implied_vol: float       # K = 1.15 * S_0
    volatility_skew_spread: float     # IV_put(85%) - IV_call(115%)
    butterfly_wing_spread: float      # 0.5*(IV_put + IV_call) - IV_atm
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        return asdict(self)


def generate_hmac_signature(payload_str: str) -> str:
    """Generate SHA256 HMAC for cryptographic receipt sealing."""
    return hmac.new(
        HMAC_SECRET,
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac_signature(payload_str: str, signature: str) -> bool:
    """Verify cryptographic receipt signature."""
    expected = generate_hmac_signature(payload_str)
    return hmac.compare_digest(expected, signature)


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_call(spot: float, strike: float, rate: float, t: float, sigma: float) -> float:
    """European Black-Scholes call pricing benchmark."""
    if t <= 1e-8:
        return max(0.0, spot - strike)
    if sigma <= 1e-8:
        return max(0.0, spot - strike * math.exp(-rate * t))
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return max(0.0, spot * _norm_cdf(d1) - strike * math.exp(-rate * t) * _norm_cdf(d2))


def implied_vol_bisection(
    target_call: float,
    spot: float,
    strike: float,
    rate: float,
    t: float,
) -> float:
    """Solve Black-Scholes implied volatility via safe 40-step bisection."""
    intrinsic = max(0.0, spot - strike * math.exp(-rate * t))
    if target_call <= intrinsic + 1e-9:
        return 0.01
    low, high = 0.01, 2.50
    for _ in range(45):
        mid = 0.5 * (low + high)
        price = black_scholes_call(spot, strike, rate, t, mid)
        if price < target_call:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def _heston_char_func(
    u: complex,
    spot: float,
    rate: float,
    t: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
) -> complex:
    """Branch-cut-free 'Little Heston Trap' characteristic function phi(u) = E^Q[exp(i*u*ln(S_T))]."""
    if xi <= 1e-6:
        # Deterministic Black-Scholes / Heston zero-vol-of-vol limit
        if abs(kappa * t) < 1e-6:
            int_v = v0 * t
        else:
            int_v = theta * t + (v0 - theta) * (1.0 - math.exp(-kappa * t)) / kappa
        return cmath.exp(1j * u * (math.log(spot) + rate * t - 0.5 * int_v) - 0.5 * (u ** 2) * int_v)

    alpha_cf = kappa - rho * xi * 1j * u
    d = cmath.sqrt(alpha_cf ** 2 + (xi ** 2) * (1j * u + u ** 2))
    denom = alpha_cf + d
    if abs(denom) < 1e-14:
        denom = 1e-14 + 0j
    g = (alpha_cf - d) / denom
    exp_neg_dt = cmath.exp(-d * t)

    one_minus_g_exp = 1.0 - g * exp_neg_dt
    one_minus_g = 1.0 - g
    if abs(one_minus_g_exp) < 1e-14:
        one_minus_g_exp = 1e-14 + 0j
    if abs(one_minus_g) < 1e-14:
        one_minus_g = 1e-14 + 0j

    c_term = rate * 1j * u * t + (kappa * theta / (xi ** 2)) * (
        (alpha_cf - d) * t - 2.0 * cmath.log(one_minus_g_exp / one_minus_g)
    )
    d_term = ((alpha_cf - d) / (xi ** 2)) * ((1.0 - exp_neg_dt) / one_minus_g_exp)
    return cmath.exp(c_term + d_term * v0 + 1j * u * math.log(spot))


def price_heston_call(
    spot: float,
    strike: float,
    rate: float,
    t: float,
    v0: float,
    theta: float,
    kappa: float,
    xi: float,
    rho: float,
    u_max: float = 120.0,
) -> float:
    """Price European call option under Heston (1993) via 64-point Gauss-Legendre Fourier integration."""
    if t <= 1e-8:
        return max(0.0, spot - strike)

    ln_k = math.log(strike)
    forward = spot * math.exp(rate * t)
    # Transform u in [0.0001, u_max] using Legendre nodes on [-1, 1]
    u_min = 1e-5
    half_len = 0.5 * (u_max - u_min)
    mid_pt = 0.5 * (u_max + u_min)

    integral_p1 = 0.0
    integral_p2 = 0.0

    for node, weight in zip(_GL_NODES, _GL_WEIGHTS):
        u = mid_pt + half_len * float(node)
        phi_u = _heston_char_func(complex(u, 0.0), spot, rate, t, v0, theta, kappa, xi, rho)
        phi_u_minus_i = _heston_char_func(complex(u, -1.0), spot, rate, t, v0, theta, kappa, xi, rho)

        exp_ik = cmath.exp(-1j * u * ln_k)
        integrand_p2 = (exp_ik * phi_u / (1j * u)).real
        integrand_p1 = (exp_ik * phi_u_minus_i / (1j * u * forward)).real

        integral_p1 += float(weight) * integrand_p1
        integral_p2 += float(weight) * integrand_p2

    p1 = 0.5 + (half_len / math.pi) * integral_p1
    p2 = 0.5 + (half_len / math.pi) * integral_p2
    call_val = spot * p1 - strike * math.exp(-rate * t) * p2
    intrinsic = max(0.0, spot - strike * math.exp(-rate * t))
    return float(max(intrinsic, min(spot, call_val)))


def classify_heston_regime(
    feller_satisfied: bool,
    volatility_skew_spread: float,
    butterfly_wing_spread: float,
) -> str:
    """Classify the Heston stochastic volatility smile and variance stability regime."""
    if not feller_satisfied and butterfly_wing_spread >= 0.02:
        return "FELLER_VIOLATED_EXTREME_VOL_OF_VOL_WINGS"
    if volatility_skew_spread >= 0.04:
        return "STRONG_CRASH_LEVERAGE_PUT_SKEW"
    if volatility_skew_spread <= -0.04:
        return "UPSIDE_GAMMA_RIGHT_TAIL_SKEW"
    if abs(volatility_skew_spread) < 0.01 and abs(butterfly_wing_spread) < 0.01:
        return "NEAR_CONSTANT_VOLATILITY_BLACK_SCHOLES"
    return "MEAN_REVERTING_STOCHASTIC_VOL_SMILE"


def calculate_heston_pricing(
    spot_price: float = 100.0,
    strike_price: float = 100.0,
    risk_free_rate: float = 0.04,
    time_to_expiry: float = 0.50,
    initial_variance: float = 0.04,
    long_run_variance: float = 0.04,
    mean_reversion_kappa: float = 2.0,
    vol_of_vol_xi: float = 0.35,
    spot_vol_corr_rho: float = -0.65,
    ticker: str = "VOL",
) -> HestonPricingReceipt:
    """Compute Heston (1993) call/put prices, Feller stability, and volatility surface metrics.

    Args:
        spot_price: Current underlying price S_0 (> 0).
        strike_price: Target option strike K (> 0).
        risk_free_rate: Continuously compounded risk-free rate r.
        time_to_expiry: Option maturity T in years (> 0).
        initial_variance: Current instantaneous variance v_0 (> 0).
        long_run_variance: Long-run variance attractor theta (> 0).
        mean_reversion_kappa: Variance mean-reversion speed kappa (> 0).
        vol_of_vol_xi: Volatility of variance xi (>= 0).
        spot_vol_corr_rho: Brownian correlation rho between spot and variance in [-0.99, 0.99].
        ticker: Security ticker symbol.

    Returns:
        HestonPricingReceipt with semi-closed-form prices, implied vol smile, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "VOL"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    s0 = float(spot_price)
    k = float(strike_price)
    t = float(time_to_expiry)
    r = float(risk_free_rate)
    if not math.isfinite(s0) or s0 <= 0.0:
        raise ValueError(f"spot_price must be finite and positive, got: {spot_price}")
    if not math.isfinite(k) or k <= 0.0:
        raise ValueError(f"strike_price must be finite and positive, got: {strike_price}")
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"time_to_expiry must be finite and positive, got: {time_to_expiry}")

    v0 = max(1e-4, float(initial_variance))
    theta = max(1e-4, float(long_run_variance))
    kappa = max(0.05, float(mean_reversion_kappa))
    xi = max(0.0, float(vol_of_vol_xi))
    rho = float(np.clip(spot_vol_corr_rho, -0.99, 0.99))

    feller_ratio = (2.0 * kappa * theta) / max(1e-8, xi ** 2) if xi > 1e-4 else 100.0
    feller_ok = (2.0 * kappa * theta) > (xi ** 2)

    call_price = price_heston_call(s0, k, r, t, v0, theta, kappa, xi, rho)
    # Put-Call Parity: P = C - S_0 + K * e^{-rT}
    put_price = max(0.0, call_price - s0 + k * math.exp(-r * t))

    avg_var = theta + (v0 - theta) * (1.0 - math.exp(-kappa * t)) / (kappa * t)
    bs_call = black_scholes_call(s0, k, r, t, math.sqrt(max(1e-6, avg_var)))

    # Evaluate implied vol smile across 85% / 100% / 115% spot strikes
    call_atm = price_heston_call(s0, s0, r, t, v0, theta, kappa, xi, rho)
    call_put_wing = price_heston_call(s0, 0.85 * s0, r, t, v0, theta, kappa, xi, rho)
    call_call_wing = price_heston_call(s0, 1.15 * s0, r, t, v0, theta, kappa, xi, rho)

    iv_atm = implied_vol_bisection(call_atm, s0, s0, r, t)
    iv_put_wing = implied_vol_bisection(call_put_wing, s0, 0.85 * s0, r, t)
    iv_call_wing = implied_vol_bisection(call_call_wing, s0, 1.15 * s0, r, t)

    skew_spread = iv_put_wing - iv_call_wing
    wing_spread = 0.5 * (iv_put_wing + iv_call_wing) - iv_atm

    regime = classify_heston_regime(
        feller_satisfied=feller_ok,
        volatility_skew_spread=skew_spread,
        butterfly_wing_spread=wing_spread,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{s0:.4f}:{k:.4f}:{t:.4f}:{v0:.4f}:{theta:.4f}:"
        f"{kappa:.4f}:{xi:.4f}:{rho:.4f}:{call_price:.6f}:{put_price:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return HestonPricingReceipt(
        ticker=cleaned_ticker,
        spot_price=s0,
        strike_price=k,
        risk_free_rate=r,
        time_to_expiry=t,
        initial_variance=v0,
        long_run_variance=theta,
        mean_reversion_kappa=kappa,
        vol_of_vol_xi=xi,
        spot_vol_corr_rho=rho,
        feller_ratio=feller_ratio,
        feller_condition_satisfied=feller_ok,
        heston_call_price=call_price,
        heston_put_price=put_price,
        bs_benchmark_call_price=bs_call,
        atm_implied_vol=iv_atm,
        otm_put_implied_vol=iv_put_wing,
        otm_call_implied_vol=iv_call_wing,
        volatility_skew_spread=skew_spread,
        butterfly_wing_spread=wing_spread,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_heston_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "VOL",
    vol_of_vol_xi: float = 0.35,
    spot_vol_corr_rho: float = -0.65,
    time_to_expiry: float = 0.50,
    risk_free_rate: float = 0.04,
) -> HestonPricingReceipt:
    """Calibrate initial/long-run variance from historical price returns and run Heston pricing."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        s0 = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr)))
        long_sigma = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.10, 0.80))
        recent_sigma = float(np.clip(np.std(log_rets[-10:]) * math.sqrt(252.0), 0.10, 0.90))
        v0 = recent_sigma ** 2
        theta = long_sigma ** 2
    else:
        s0 = 100.0
        v0 = 0.04
        theta = 0.04

    return calculate_heston_pricing(
        spot_price=s0,
        strike_price=s0,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        initial_variance=v0,
        long_run_variance=theta,
        mean_reversion_kappa=2.0,
        vol_of_vol_xi=vol_of_vol_xi,
        spot_vol_corr_rho=spot_vol_corr_rho,
        ticker=ticker,
    )


def persist_heston_record(
    receipt: HestonPricingReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Heston stochastic volatility pricing record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS heston_vol_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        spot_price REAL NOT NULL,
        strike_price REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        time_to_expiry REAL NOT NULL,
        initial_variance REAL NOT NULL,
        long_run_variance REAL NOT NULL,
        mean_reversion_kappa REAL NOT NULL,
        vol_of_vol_xi REAL NOT NULL,
        spot_vol_corr_rho REAL NOT NULL,
        feller_ratio REAL NOT NULL,
        feller_condition_satisfied INTEGER NOT NULL,
        heston_call_price REAL NOT NULL,
        heston_put_price REAL NOT NULL,
        bs_benchmark_call_price REAL NOT NULL,
        atm_implied_vol REAL NOT NULL,
        otm_put_implied_vol REAL NOT NULL,
        otm_call_implied_vol REAL NOT NULL,
        volatility_skew_spread REAL NOT NULL,
        butterfly_wing_spread REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_heston_ticker_time ON heston_vol_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO heston_vol_audits (
        ticker, spot_price, strike_price, risk_free_rate, time_to_expiry,
        initial_variance, long_run_variance, mean_reversion_kappa, vol_of_vol_xi,
        spot_vol_corr_rho, feller_ratio, feller_condition_satisfied,
        heston_call_price, heston_put_price, bs_benchmark_call_price,
        atm_implied_vol, otm_put_implied_vol, otm_call_implied_vol,
        volatility_skew_spread, butterfly_wing_spread, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.spot_price,
                    receipt.strike_price,
                    receipt.risk_free_rate,
                    receipt.time_to_expiry,
                    receipt.initial_variance,
                    receipt.long_run_variance,
                    receipt.mean_reversion_kappa,
                    receipt.vol_of_vol_xi,
                    receipt.spot_vol_corr_rho,
                    receipt.feller_ratio,
                    1 if receipt.feller_condition_satisfied else 0,
                    receipt.heston_call_price,
                    receipt.heston_put_price,
                    receipt.bs_benchmark_call_price,
                    receipt.atm_implied_vol,
                    receipt.otm_put_implied_vol,
                    receipt.otm_call_implied_vol,
                    receipt.volatility_skew_spread,
                    receipt.butterfly_wing_spread,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

