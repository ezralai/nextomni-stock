"""Douglas T. Breeden & Robert H. Litzenberger (1978) Arrow-Debreu State Price & Risk-Neutral Density Engine.

Theoretical Foundation:
    Breeden, Douglas T., & Litzenberger, Robert H. (1978). "Prices of State-Contingent Claims
    Implicit in Option Prices". The Journal of Business, 51(4), pp. 621-651.
    The foundational theorem connecting option markets to Arrow-Debreu general equilibrium
    state pricing and model-free Risk-Neutral Density (RND) recovery.

Key Concepts:
    1. Arrow-Debreu State Price Theorem:
       The current price p(K) of a pure Arrow-Debreu state-contingent claim paying $1 if the
       underlying terminal price S_T equals K is given by the discounted second partial
       derivative of the European call option price with respect to strike K:
           p(K) = e^{-rT} f^Q(K) = d^2 C(K, T) / dK^2

    2. Model-Free Risk-Neutral Density (RND):
       f^Q(S_T = K) = e^{rT} * (d^2 C(K, T) / dK^2)
       Recovered non-parametrically from observed call prices across strikes without assuming
       lognormality or constant volatility.

    3. Butterfly Spread Replication & Arbitrage Bounds:
       Discrete butterfly spread approximates the second derivative:
           [C(K - dK) - 2*C(K) + C(K + dK)] / (dK)^2 ~= d^2 C / dK^2 >= 0
       Negative second derivatives imply static butterfly arbitrage (convexity violation).

    4. Model-Free Risk-Neutral Moments & Crash Risk:
       Integrates across the recovered RND to compute E^Q[S_T], standard deviation, skewness
       (left-tail crash fear / put skew), and kurtosis (jump / fat-tail intensity).
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"breeden-litzenberger-rnd-v134-secret"


@dataclass(frozen=True)
class BreedenLitzenbergerReceipt:
    """Immutable quantitative audit receipt for Breeden-Litzenberger (1978) RND recovery."""

    ticker: str
    spot_price: float
    risk_free_rate: float
    time_to_expiry: float
    num_strikes: int
    theoretical_forward: float
    rnd_mean: float
    rnd_std: float
    rnd_implied_vol: float
    rnd_skewness: float
    rnd_kurtosis: float
    left_tail_prob_10pct: float
    right_tail_prob_10pct: float
    raw_probability_mass: float
    normalized_integral: float
    peak_state_price: float
    peak_state_strike: float
    butterfly_arbitrage_detected: bool
    butterfly_violation_count: int
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


def black_scholes_call(
    spot: float,
    strike: float,
    rate: float,
    time_to_expiry: float,
    sigma: float,
) -> float:
    """Compute exact European Black-Scholes call option price."""
    if time_to_expiry <= 0.0:
        return max(0.0, spot - strike)
    if sigma <= 1e-8:
        return max(0.0, spot - strike * math.exp(-rate * time_to_expiry))
    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * time_to_expiry) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return spot * _norm_cdf(d1) - strike * math.exp(-rate * time_to_expiry) * _norm_cdf(d2)


def generate_skewed_call_surface(
    spot_price: float = 100.0,
    atm_vol: float = 0.25,
    vol_skew: float = 0.20,
    vol_smile: float = 0.05,
    risk_free_rate: float = 0.04,
    time_to_expiry: float = 0.25,
    num_strikes: int = 49,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a smooth European call price curve with implied volatility skew and smile."""
    s0 = max(1.0, float(spot_price))
    t = max(1e-4, float(time_to_expiry))
    r = float(risk_free_rate)
    sigma_0 = max(0.05, float(atm_vol))
    n = max(7, int(num_strikes))

    strikes = np.linspace(0.40 * s0, 1.60 * s0, n)
    call_prices = np.empty(n, dtype=float)
    for idx, k in enumerate(strikes):
        moneyness = (s0 - k) / s0
        local_vol = float(np.clip(sigma_0 + vol_skew * moneyness + vol_smile * (moneyness ** 2), 0.05, 1.50))
        call_prices[idx] = black_scholes_call(s0, float(k), r, t, local_vol)

    return strikes, call_prices


def classify_rnd_regime(
    rnd_skewness: float,
    rnd_kurtosis: float,
    butterfly_arbitrage_detected: bool,
) -> str:
    """Classify the Breeden-Litzenberger state-price & tail-risk regime."""
    if butterfly_arbitrage_detected:
        return "BUTTERFLY_ARBITRAGE_CONVEXITY_VIOLATION"
    if rnd_skewness <= -0.35 or rnd_kurtosis >= 4.20:
        return "CRASH_FEAR_LEFT_TAIL_SKEW"
    if rnd_skewness >= 0.35:
        return "RIGHT_TAIL_SPECULATIVE_UPSIDE"
    if abs(rnd_skewness) <= 0.15 and rnd_kurtosis < 3.40:
        return "NEAR_LOGNORMAL_RISK_NEUTRAL"
    return "MODERATE_SKEW_EQUILIBRIUM"


def calculate_breeden_litzenberger_rnd(
    spot_price: float,
    strikes: Sequence[float] | np.ndarray,
    call_prices: Sequence[float] | np.ndarray,
    risk_free_rate: float = 0.04,
    time_to_expiry: float = 0.25,
    ticker: str = "RND",
) -> BreedenLitzenbergerReceipt:
    """Recover Arrow-Debreu state prices and Risk-Neutral Density via Breeden-Litzenberger (1978).

    Args:
        spot_price: Current underlying spot price S_0 (> 0).
        strikes: Strictly increasing sequence of option strikes K_0 < ... < K_{N-1} (len >= 3).
        call_prices: Corresponding European call prices C(K_i, T).
        risk_free_rate: Continuously compounded risk-free rate r.
        time_to_expiry: Time to option expiration T in years (> 0).
        ticker: Security ticker symbol.

    Returns:
        BreedenLitzenbergerReceipt containing state prices, normalized RND moments, and audit metrics.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "RND"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    s0 = float(spot_price)
    if not math.isfinite(s0) or s0 <= 0.0:
        raise ValueError(f"Spot price must be finite and positive, got: {spot_price}")

    t = float(time_to_expiry)
    if not math.isfinite(t) or t <= 0.0:
        raise ValueError(f"Time to expiry must be positive, got: {time_to_expiry}")

    r = float(risk_free_rate)
    if not math.isfinite(r):
        raise ValueError(f"Risk-free rate must be finite, got: {risk_free_rate}")

    k_arr = np.asarray(strikes, dtype=float)
    c_arr = np.asarray(call_prices, dtype=float)

    if k_arr.ndim != 1 or c_arr.ndim != 1 or len(k_arr) != len(c_arr):
        raise ValueError("Strikes and call_prices must be 1D sequences of equal length.")
    if len(k_arr) < 3:
        raise ValueError("At least 3 strikes are required to evaluate butterfly second derivatives.")
    if not np.all(np.isfinite(k_arr)) or not np.all(np.isfinite(c_arr)):
        raise ValueError("Strikes and call_prices must contain only finite numbers.")
    if np.any(np.diff(k_arr) <= 0.0):
        raise ValueError("Strikes must be strictly increasing.")

    n = len(k_arr)
    interior_k = k_arr[1:-1]
    state_prices_raw = np.empty(n - 2, dtype=float)
    widths = np.empty(n - 2, dtype=float)

    for j in range(1, n - 1):
        dk_left = k_arr[j] - k_arr[j - 1]
        dk_right = k_arr[j + 1] - k_arr[j]
        widths[j - 1] = 0.5 * (dk_left + dk_right)
        # General second derivative d^2C/dK^2 reducing to (C_{j-1} - 2C_j + C_{j+1}) / dk^2 when uniform
        d2c = (2.0 / (dk_left + dk_right)) * (
            (c_arr[j + 1] - c_arr[j]) / dk_right - (c_arr[j] - c_arr[j - 1]) / dk_left
        )
        state_prices_raw[j - 1] = d2c

    # Check static butterfly arbitrage (negative convexity d^2C/dK^2 < 0)
    violations = state_prices_raw < -1e-6
    violation_count = int(np.sum(violations))
    arbitrage_detected = violation_count > 0

    discount_inv = math.exp(r * t)
    nonneg_state_prices = np.maximum(0.0, state_prices_raw)
    raw_density = discount_inv * nonneg_state_prices
    raw_weights = raw_density * widths
    raw_mass = float(np.sum(raw_weights))

    if raw_mass <= 1e-12:
        # Degenerate flat or linear call curve fallback to uniform interior weights
        probs = np.full(len(interior_k), 1.0 / len(interior_k), dtype=float)
    else:
        probs = raw_weights / raw_mass

    normalized_density = probs / widths
    normalized_integral = float(np.sum(normalized_density * widths))

    peak_idx = int(np.argmax(nonneg_state_prices)) if len(nonneg_state_prices) > 0 else 0
    peak_state_price = float(nonneg_state_prices[peak_idx])
    peak_state_strike = float(interior_k[peak_idx])

    # Model-free risk-neutral moments:
    # Price-level mean and standard deviation
    rnd_mean = float(np.sum(probs * interior_k))
    theoretical_forward = s0 * math.exp(r * t)
    diff = interior_k - rnd_mean
    variance = float(np.sum(probs * (diff ** 2)))
    rnd_std = math.sqrt(max(1e-12, variance))
    rnd_implied_vol = rnd_std / (max(1e-8, rnd_mean) * math.sqrt(t))

    # Standardized log-return moments (Bakshi-Kapadia-Madan / B-L convention so flat BS has skew=0, kurt=3)
    log_rets = np.log(interior_k / s0)
    log_mean = float(np.sum(probs * log_rets))
    log_diff = log_rets - log_mean
    log_var = float(np.sum(probs * (log_diff ** 2)))
    log_std = math.sqrt(max(1e-12, log_var))

    if log_std > 1e-8:
        z_log = log_diff / log_std
        rnd_skewness = float(np.sum(probs * (z_log ** 3)))
        rnd_kurtosis = float(np.sum(probs * (z_log ** 4)))
    else:
        rnd_skewness = 0.0
        rnd_kurtosis = 3.0

    left_tail_prob = float(np.sum(probs[interior_k < 0.90 * s0]))
    right_tail_prob = float(np.sum(probs[interior_k > 1.10 * s0]))

    regime = classify_rnd_regime(
        rnd_skewness=rnd_skewness,
        rnd_kurtosis=rnd_kurtosis,
        butterfly_arbitrage_detected=arbitrage_detected,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{s0:.4f}:{r:.4f}:{t:.4f}:{rnd_mean:.4f}:"
        f"{rnd_std:.4f}:{rnd_skewness:.4f}:{rnd_kurtosis:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BreedenLitzenbergerReceipt(
        ticker=cleaned_ticker,
        spot_price=s0,
        risk_free_rate=r,
        time_to_expiry=t,
        num_strikes=n,
        theoretical_forward=theoretical_forward,
        rnd_mean=rnd_mean,
        rnd_std=rnd_std,
        rnd_implied_vol=rnd_implied_vol,
        rnd_skewness=rnd_skewness,
        rnd_kurtosis=rnd_kurtosis,
        left_tail_prob_10pct=left_tail_prob,
        right_tail_prob_10pct=right_tail_prob,
        raw_probability_mass=raw_mass,
        normalized_integral=normalized_integral,
        peak_state_price=peak_state_price,
        peak_state_strike=peak_state_strike,
        butterfly_arbitrage_detected=arbitrage_detected,
        butterfly_violation_count=violation_count,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_breeden_litzenberger_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "RND",
    vol_skew: float = 0.22,
    vol_smile: float = 0.04,
    risk_free_rate: float = 0.04,
    time_to_expiry: float = 0.25,
    num_strikes: int = 49,
) -> BreedenLitzenbergerReceipt:
    """Empirically calibrate an option call curve from equity prices and recover its RND."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        s0 = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        atm_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.90))
    else:
        s0 = 100.0
        atm_vol = 0.25

    strikes, calls = generate_skewed_call_surface(
        spot_price=s0,
        atm_vol=atm_vol,
        vol_skew=vol_skew,
        vol_smile=vol_smile,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        num_strikes=num_strikes,
    )
    return calculate_breeden_litzenberger_rnd(
        spot_price=s0,
        strikes=strikes,
        call_prices=calls,
        risk_free_rate=risk_free_rate,
        time_to_expiry=time_to_expiry,
        ticker=ticker,
    )


def persist_breeden_litzenberger_record(
    receipt: BreedenLitzenbergerReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Breeden-Litzenberger RND record to SQLite database using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS breeden_litzenberger_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        spot_price REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        time_to_expiry REAL NOT NULL,
        num_strikes INTEGER NOT NULL,
        theoretical_forward REAL NOT NULL,
        rnd_mean REAL NOT NULL,
        rnd_std REAL NOT NULL,
        rnd_implied_vol REAL NOT NULL,
        rnd_skewness REAL NOT NULL,
        rnd_kurtosis REAL NOT NULL,
        left_tail_prob_10pct REAL NOT NULL,
        right_tail_prob_10pct REAL NOT NULL,
        raw_probability_mass REAL NOT NULL,
        normalized_integral REAL NOT NULL,
        peak_state_price REAL NOT NULL,
        peak_state_strike REAL NOT NULL,
        butterfly_arbitrage_detected INTEGER NOT NULL,
        butterfly_violation_count INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_bl_rnd_ticker_time ON breeden_litzenberger_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO breeden_litzenberger_audits (
        ticker, spot_price, risk_free_rate, time_to_expiry, num_strikes,
        theoretical_forward, rnd_mean, rnd_std, rnd_implied_vol,
        rnd_skewness, rnd_kurtosis, left_tail_prob_10pct, right_tail_prob_10pct,
        raw_probability_mass, normalized_integral, peak_state_price,
        peak_state_strike, butterfly_arbitrage_detected, butterfly_violation_count,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    receipt.risk_free_rate,
                    receipt.time_to_expiry,
                    receipt.num_strikes,
                    receipt.theoretical_forward,
                    receipt.rnd_mean,
                    receipt.rnd_std,
                    receipt.rnd_implied_vol,
                    receipt.rnd_skewness,
                    receipt.rnd_kurtosis,
                    receipt.left_tail_prob_10pct,
                    receipt.right_tail_prob_10pct,
                    receipt.raw_probability_mass,
                    receipt.normalized_integral,
                    receipt.peak_state_price,
                    receipt.peak_state_strike,
                    1 if receipt.butterfly_arbitrage_detected else 0,
                    receipt.butterfly_violation_count,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
