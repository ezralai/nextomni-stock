"""Thomas E. Copeland & Dan Galai (1983) Option-Theoretic Bid-Ask Spread Engine.

Theoretical Foundation:
    Copeland, Thomas E., and Galai, Dan (1983).
    "Information Effects on the Bid-Ask Spread".
    The Journal of Finance, 38(5), pp. 1457-1469.

Key Concepts:
    - Competitive Dealer Quote as a Free Straddle:
      A market maker posting Ask K_A = P_0 + delta and Bid K_B = P_0 - delta grants
      informed traders (fraction pi_I) a free call option at strike K_A and a free
      put option at strike K_B over quote-life horizon tau, while collecting
      half-spread delta from uninformed liquidity arrivals (fraction 1 - pi_I).
    - Zero-Profit Competitive Spread Equilibrium:
      (1 - pi_I) * delta* = pi_I * [C(P_0, P_0 + delta*, sigma, tau)
                                  + P(P_0, P_0 - delta*, sigma, tau)],
      yielding unique bid-ask spread S* = 2 * delta* increasing monotonically in
      informed trader probability pi_I and fundamental asset volatility sigma.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Sequence

import numpy as np
from scipy.optimize import brentq

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"copeland-galai-spread-v188-secret"
INV_SQRT_2 = 1.0 / math.sqrt(2.0)


@dataclass(frozen=True)
class CopelandGalaiReceipt:
    """Immutable quantitative audit receipt for Copeland & Galai (1983) spread equilibrium."""

    ticker: str
    spot_price: float
    volatility_annual: float
    informed_prob: float
    quote_life_years: float
    bid_price: float
    ask_price: float
    half_spread: float
    equilibrium_spread: float
    spread_bps: float
    informed_call_value: float
    informed_put_value: float
    atm_straddle_upper_bound: float
    equilibrium_residual: float
    spread_sensitivity_per_1pct_pi: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def _validate_ticker(ticker: str) -> str:
    cleaned = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned):
        raise ValueError(
            f"Invalid ticker format '{ticker}'. Expected 1-6 alphanumeric chars plus optional '.XX' suffix."
        )
    return cleaned


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-float(x) * INV_SQRT_2)


def _bs_zero_carry_call(spot: float, strike: float, vol_sqrt_t: float) -> float:
    if strike <= 0.0:
        return spot
    if vol_sqrt_t <= 1e-12:
        return max(0.0, spot - strike)
    d1 = math.log(spot / strike) / vol_sqrt_t + 0.5 * vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(spot * _norm_cdf(d1) - strike * _norm_cdf(d2))


def _bs_zero_carry_put(spot: float, strike: float, vol_sqrt_t: float) -> float:
    if strike <= 0.0:
        return 0.0
    if vol_sqrt_t <= 1e-12:
        return max(0.0, strike - spot)
    d1 = math.log(spot / strike) / vol_sqrt_t + 0.5 * vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(strike * _norm_cdf(-d2) - spot * _norm_cdf(-d1))


def _solve_equilibrium_half_spread(spot: float, vol_sqrt_t: float, pi_i: float) -> float:
    if pi_i <= 0.0:
        return 0.0

    max_delta = spot * (1.0 - 1e-6)

    def dealer_profit_net(delta: float) -> float:
        call_val = _bs_zero_carry_call(spot, spot + delta, vol_sqrt_t)
        put_val = _bs_zero_carry_put(spot, spot - delta, vol_sqrt_t)
        return (1.0 - pi_i) * delta - pi_i * (call_val + put_val)

    f_low = dealer_profit_net(0.0)
    if f_low >= 0.0:
        return 0.0
    f_high = dealer_profit_net(max_delta)
    if f_high < 0.0:
        raise ValueError(
            "No finite zero-profit spread equilibrium exists (informed share triggers market breakdown)."
        )

    return float(brentq(dealer_profit_net, 0.0, max_delta, xtol=1e-12, rtol=1e-12))


def _build_hmac_payload(
    *,
    ticker: str,
    spot_price: float,
    volatility_annual: float,
    informed_prob: float,
    equilibrium_spread: float,
    spread_bps: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{spot_price:.6f}|{volatility_annual:.6f}|{informed_prob:.6f}|"
        f"{equilibrium_spread:.6f}|{spread_bps:.6f}|{regime}|{timestamp}"
    )


def sign_copeland_galai_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Copeland-Galai audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Copeland-Galai audit HMAC signature."""
    expected = sign_copeland_galai_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def calculate_copeland_galai_spread(
    spot_price: float,
    volatility_annual: float = 0.25,
    informed_prob: float = 0.15,
    quote_life_years: float = 1.0 / 252.0,
    ticker: str = "SPY",
) -> CopelandGalaiReceipt:
    """Compute competitive zero-profit Copeland & Galai (1983) option-theoretic spread."""
    clean_ticker = _validate_ticker(ticker)

    p0 = float(spot_price)
    sigma = float(volatility_annual)
    pi_i = float(informed_prob)
    tau = float(quote_life_years)

    if not math.isfinite(p0) or p0 <= 0.0:
        raise ValueError(f"spot_price must be strictly positive, got {spot_price}.")
    if not math.isfinite(sigma) or sigma <= 0.0 or sigma > 3.0:
        raise ValueError(f"volatility_annual must be finite in (0, 3.0], got {volatility_annual}.")
    if not math.isfinite(pi_i) or pi_i < 0.0 or pi_i >= 0.85:
        raise ValueError(f"informed_prob must be finite in [0.0, 0.85), got {informed_prob}.")
    if not math.isfinite(tau) or tau <= 0.0 or tau > 1.0:
        raise ValueError(f"quote_life_years must be finite in (0.0, 1.0], got {quote_life_years}.")

    vol_sqrt_t = sigma * math.sqrt(tau)
    delta_star = _solve_equilibrium_half_spread(p0, vol_sqrt_t, pi_i)
    bid = float(p0 - delta_star)
    ask = float(p0 + delta_star)
    spread = float(2.0 * delta_star)
    spread_bps = float((spread / p0) * 10000.0)

    call_eq = _bs_zero_carry_call(p0, ask, vol_sqrt_t)
    put_eq = _bs_zero_carry_put(p0, bid, vol_sqrt_t)
    atm_straddle = _bs_zero_carry_call(p0, p0, vol_sqrt_t) + _bs_zero_carry_put(p0, p0, vol_sqrt_t)
    eq_residual = float(abs((1.0 - pi_i) * delta_star - pi_i * (call_eq + put_eq)))

    # Sensitivity: spread bps increase per +1% (0.01) informed arrival share
    pi_up = min(0.84, pi_i + 0.01)
    delta_up = _solve_equilibrium_half_spread(p0, vol_sqrt_t, pi_up)
    step_pi = max(pi_up - pi_i, 1e-6)
    sens_1pct = float(((2.0 * (delta_up - delta_star)) / p0) * 10000.0 * (0.01 / step_pi))

    if spread_bps < 1.0:
        regime = "ZERO_ADVERSE_SELECTION_FRICTIONLESS"
    elif spread_bps < 35.0:
        regime = "TIGHT_LIQUID_DEALER_EQUILIBRIUM"
    elif spread_bps < 120.0:
        regime = "ELEVATED_INFORMATION_SPREAD"
    else:
        regime = "SEVERE_ADVERSE_SELECTION_ILLIQUIDITY"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        spot_price=p0,
        volatility_annual=sigma,
        informed_prob=pi_i,
        equilibrium_spread=spread,
        spread_bps=spread_bps,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_copeland_galai_receipt(payload)

    return CopelandGalaiReceipt(
        ticker=clean_ticker,
        spot_price=p0,
        volatility_annual=sigma,
        informed_prob=pi_i,
        quote_life_years=tau,
        bid_price=bid,
        ask_price=ask,
        half_spread=delta_star,
        equilibrium_spread=spread,
        spread_bps=spread_bps,
        informed_call_value=call_eq,
        informed_put_value=put_eq,
        atm_straddle_upper_bound=atm_straddle,
        equilibrium_residual=eq_residual,
        spread_sensitivity_per_1pct_pi=sens_1pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_copeland_galai_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    informed_prob: float = 0.15,
) -> CopelandGalaiReceipt:
    """Calibrate spot price and annualized volatility from trailing closing prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 20:
        spot = max(1e-4, float(p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 180) :])))
        vol_ann = max(0.05, min(2.99, float(np.std(log_rets, ddof=1) * math.sqrt(252.0))))
    else:
        spot = 100.0
        vol_ann = 0.25

    return calculate_copeland_galai_spread(
        spot_price=spot,
        volatility_annual=vol_ann,
        informed_prob=float(informed_prob),
        quote_life_years=1.0 / 252.0,
        ticker=ticker,
    )


def persist_copeland_galai_record(
    receipt: CopelandGalaiReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Copeland & Galai (1983) bid-ask spread audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS copeland_galai_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        spot_price REAL NOT NULL,
        volatility_annual REAL NOT NULL,
        informed_prob REAL NOT NULL,
        bid_price REAL NOT NULL,
        ask_price REAL NOT NULL,
        equilibrium_spread REAL NOT NULL,
        spread_bps REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO copeland_galai_audits (
        ticker, spot_price, volatility_annual, informed_prob, bid_price,
        ask_price, equilibrium_spread, spread_bps, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.spot_price,
                receipt.volatility_annual,
                receipt.informed_prob,
                receipt.bid_price,
                receipt.ask_price,
                receipt.equilibrium_spread,
                receipt.spread_bps,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

