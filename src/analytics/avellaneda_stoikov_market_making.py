"""Marco Avellaneda & Sasha Stoikov (2008) Optimal Market Making & Inventory Risk Engine.

Implements the optimal high-frequency market making and inventory control framework formulated in:
Avellaneda, Marco, & Stoikov, Sasha (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224.

By solving the Hamilton-Jacobi-Bellman (HJB) equation for an inventory-risk-averse
liquidity provider with CARA utility, the Avellaneda-Stoikov model derives:
1. Reservation (Indifference) Price:
       r(s, q, tau) = s - q * gamma * sigma^2 * tau
2. Optimal Asymmetric Bid/Ask Limit Quotes:
       r_a = r(s, q, tau) + (1 / gamma) * ln(1 + gamma / kappa)
       r_b = r(s, q, tau) - (1 / gamma) * ln(1 + gamma / kappa)
3. Fill Arrival Intensities & Probabilities:
       lambda(delta) = A * exp(-kappa * delta)
       P(fill) = 1 - exp(-lambda * tau)
where:
    s: Mid-price
    q: Current inventory position (shares/contracts)
    gamma: Inventory risk aversion parameter
    sigma: Asset volatility
    tau: Remaining trading horizon (T - t)
    kappa: Order book liquidity decay parameter
    A: Base order arrival intensity
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
SECRET_SALT = os.environ.get("NEXTOMNI_AUDIT_SALT", "nextomni_avellaneda_stoikov_salt").encode("utf-8")


@dataclass(frozen=True)
class AvellanedaStoikovReceipt:
    """Immutable receipt containing Avellaneda & Stoikov (2008) optimal quotes and inventory risk."""

    ticker: str
    mid_price: float
    inventory_q: int
    volatility_sigma: float
    risk_aversion_gamma: float
    order_intensity_kappa: float
    horizon_tau: float
    reservation_price: float
    inventory_skew_dollar: float
    optimal_bid_price: float
    optimal_ask_price: float
    optimal_spread_dollar: float
    optimal_spread_bps: float
    bid_depth_delta: float
    ask_depth_delta: float
    bid_fill_probability: float
    ask_fill_probability: float
    inventory_risk_penalty: float
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


def calculate_avellaneda_stoikov(
    mid_price: float,
    inventory: int = 0,
    volatility: float = 0.02,
    risk_aversion: float = 0.10,
    kappa: float = 1.50,
    base_intensity_a: float = 140.0,
    horizon_tau: float = 1.0,
    ticker: str = "SPY",
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> AvellanedaStoikovReceipt:
    """Calculate optimal Avellaneda-Stoikov (2008) reservation price and asymmetric quotes.

    Parameters
    ----------
    mid_price : float
        Current asset mid-price s > 0.
    inventory : int
        Current inventory position q (positive for long, negative for short).
    volatility : float
        Asset volatility sigma > 0 (dollar or daily return standard deviation).
    risk_aversion : float
        Inventory risk aversion coefficient gamma > 0.
    kappa : float
        Order book liquidity/depth decay intensity parameter kappa > 0.
    base_intensity_a : float
        Base order arrival rate parameter A > 0.
    horizon_tau : float
        Remaining trading horizon tau = T - t > 0.
    ticker : str
        Stock ticker symbol.
    derivation_mode : str
        Source mode label.

    Returns
    -------
    AvellanedaStoikovReceipt
        Complete optimal quoting receipt.
    """
    clean_ticker = ticker.strip().upper() if TICKER_REGEX.match(ticker.strip().upper()) else "UNKNOWN"
    now_iso = datetime.now(timezone.utc).isoformat()

    s = max(float(mid_price), 0.01)
    q = int(inventory)
    sigma = max(float(volatility), 1e-4)
    gamma = max(float(risk_aversion), 1e-4)
    k = max(float(kappa), 1e-4)
    tau = max(float(horizon_tau), 1e-4)
    a_rate = max(float(base_intensity_a), 1.0)

    # 1. Reservation Price: r(s, q, tau) = s - q * gamma * sigma^2 * tau
    # Dollar variance: if volatility is fractional, convert to dollar variance s^2 * sigma^2
    # Standard Avellaneda-Stoikov specifies sigma in absolute dollar units
    # If sigma < 1.0 (e.g. 0.02 = 2% return vol), scale dollar sigma = s * sigma
    sigma_dollar = s * sigma if sigma < 1.0 else sigma
    variance_dollar = sigma_dollar**2

    inventory_skew = float(-q * gamma * variance_dollar * tau)
    res_price = float(s + inventory_skew)

    # 2. Optimal Half-Spread & Quotes:
    # delta_0 = (1 / gamma) * ln(1 + gamma / kappa)
    ratio_gk = gamma / k
    # Use log1p for numerical precision
    delta_0 = float((1.0 / gamma) * math.log1p(ratio_gk))
    # Bound delta_0 to a sensible minimum
    delta_0 = max(delta_0, 0.005)

    # Optimal limit prices:
    # r_a = r + delta_0, r_b = r - delta_0
    ask_price = float(res_price + delta_0)
    bid_price = float(res_price - delta_0)
    opt_spread = float(ask_price - bid_price)  # 2 * delta_0

    # Distances from mid-price s:
    delta_a = float(ask_price - s)
    delta_b = float(s - bid_price)

    # Effective spread in basis points
    eff_spread_bps = float((opt_spread / s) * 10000.0)

    # 3. Fill arrival intensities and probabilities over horizon tau
    # lambda_a = A * exp(-kappa * delta_a), lambda_b = A * exp(-kappa * delta_b)
    # Clip exponent to avoid overflow/underflow
    exp_a = float(np.clip(-k * delta_a, -25.0, 5.0))
    exp_b = float(np.clip(-k * delta_b, -25.0, 5.0))

    lambda_a = a_rate * math.exp(exp_a)
    lambda_b = a_rate * math.exp(exp_b)

    # Fill probability: P(fill) = 1 - exp(-lambda * tau)
    prob_fill_a = float(1.0 - math.exp(-min(lambda_a * tau, 25.0)))
    prob_fill_b = float(1.0 - math.exp(-min(lambda_b * tau, 25.0)))

    # 4. Inventory risk penalty: (1/2) * gamma * sigma^2 * q^2 * tau
    risk_penalty = float(0.5 * gamma * variance_dollar * (q**2) * tau)

    # 5. Regime classification
    skew_ratio = abs(inventory_skew) / opt_spread if opt_spread > 1e-6 else 0.0
    if skew_ratio > 0.50:
        regime = "EXTREME_INVENTORY_RISK"
    elif q > 0 and inventory_skew < -0.10 * opt_spread:
        regime = "LONG_INVENTORY_SKEW_DOWN"
    elif q < 0 and inventory_skew > 0.10 * opt_spread:
        regime = "SHORT_INVENTORY_SKEW_UP"
    else:
        regime = "BALANCED_MARKET_MAKING"

    payload = (
        f"{clean_ticker}|{s:.4f}|{q}|{sigma:.6f}|{gamma:.4f}|{k:.4f}|{tau:.4f}|"
        f"{res_price:.4f}|{inventory_skew:.4f}|{bid_price:.4f}|{ask_price:.4f}|"
        f"{opt_spread:.4f}|{eff_spread_bps:.2f}|{prob_fill_b:.4f}|{prob_fill_a:.4f}|"
        f"{risk_penalty:.4f}|{regime}|{derivation_mode}|{now_iso}"
    )
    sig = _compute_hmac(payload)

    return AvellanedaStoikovReceipt(
        ticker=clean_ticker,
        mid_price=s,
        inventory_q=q,
        volatility_sigma=sigma,
        risk_aversion_gamma=gamma,
        order_intensity_kappa=k,
        horizon_tau=tau,
        reservation_price=res_price,
        inventory_skew_dollar=inventory_skew,
        optimal_bid_price=bid_price,
        optimal_ask_price=ask_price,
        optimal_spread_dollar=opt_spread,
        optimal_spread_bps=eff_spread_bps,
        bid_depth_delta=delta_b,
        ask_depth_delta=delta_a,
        bid_fill_probability=prob_fill_b,
        ask_fill_probability=prob_fill_a,
        inventory_risk_penalty=risk_penalty,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig,
        timestamp=now_iso,
    )


def evaluate_avellaneda_stoikov_from_prices(
    prices: Sequence[float],
    inventory: int = 0,
    risk_aversion: float = 0.10,
    kappa: float = 1.50,
    ticker: str = "SPY",
) -> AvellanedaStoikovReceipt:
    """Calibrate Avellaneda-Stoikov parameters from historical price series."""
    p_arr = np.asarray(prices, dtype=np.float64)
    valid_p = np.isfinite(p_arr) & (p_arr > 0.0)
    p_clean = p_arr[valid_p]

    if len(p_clean) < 15:
        return _empty_receipt(ticker, datetime.now(timezone.utc).isoformat())

    latest_s = float(p_clean[-1])
    ret = np.diff(p_clean) / p_clean[:-1]
    hist_vol = float(np.std(ret, ddof=1)) if len(ret) > 1 else 0.015

    return calculate_avellaneda_stoikov(
        mid_price=latest_s,
        inventory=inventory,
        volatility=hist_vol,
        risk_aversion=risk_aversion,
        kappa=kappa,
        horizon_tau=1.0,
        ticker=ticker,
        derivation_mode="HISTORICAL_PRICE_CALIBRATION",
    )


def _empty_receipt(ticker: str, timestamp_iso: str) -> AvellanedaStoikovReceipt:
    empty_payload = f"{ticker}|0.0|0|0.0|0.0|0.0|0.0|{timestamp_iso}"
    sig = _compute_hmac(empty_payload)
    return AvellanedaStoikovReceipt(
        ticker=ticker,
        mid_price=0.0,
        inventory_q=0,
        volatility_sigma=0.0,
        risk_aversion_gamma=0.0,
        order_intensity_kappa=0.0,
        horizon_tau=0.0,
        reservation_price=0.0,
        inventory_skew_dollar=0.0,
        optimal_bid_price=0.0,
        optimal_ask_price=0.0,
        optimal_spread_dollar=0.0,
        optimal_spread_bps=0.0,
        bid_depth_delta=0.0,
        ask_depth_delta=0.0,
        bid_fill_probability=0.0,
        ask_fill_probability=0.0,
        inventory_risk_penalty=0.0,
        regime="INSUFFICIENT_DATA",
        derivation_mode="NONE",
        hmac_signature=sig,
        timestamp=timestamp_iso,
    )


def persist_as_record(
    receipt: AvellanedaStoikovReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Store Avellaneda-Stoikov receipt in SQLite database with audit verification."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS avellaneda_stoikov_receipts (
                ticker TEXT NOT NULL,
                mid_price REAL NOT NULL,
                inventory_q INTEGER NOT NULL,
                volatility_sigma REAL NOT NULL,
                risk_aversion_gamma REAL NOT NULL,
                order_intensity_kappa REAL NOT NULL,
                horizon_tau REAL NOT NULL,
                reservation_price REAL NOT NULL,
                inventory_skew_dollar REAL NOT NULL,
                optimal_bid_price REAL NOT NULL,
                optimal_ask_price REAL NOT NULL,
                optimal_spread_dollar REAL NOT NULL,
                optimal_spread_bps REAL NOT NULL,
                bid_depth_delta REAL NOT NULL,
                ask_depth_delta REAL NOT NULL,
                bid_fill_probability REAL NOT NULL,
                ask_fill_probability REAL NOT NULL,
                inventory_risk_penalty REAL NOT NULL,
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
            INSERT OR REPLACE INTO avellaneda_stoikov_receipts (
                ticker, mid_price, inventory_q, volatility_sigma,
                risk_aversion_gamma, order_intensity_kappa, horizon_tau,
                reservation_price, inventory_skew_dollar,
                optimal_bid_price, optimal_ask_price,
                optimal_spread_dollar, optimal_spread_bps,
                bid_depth_delta, ask_depth_delta,
                bid_fill_probability, ask_fill_probability,
                inventory_risk_penalty, regime, derivation_mode,
                hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.mid_price,
                receipt.inventory_q,
                receipt.volatility_sigma,
                receipt.risk_aversion_gamma,
                receipt.order_intensity_kappa,
                receipt.horizon_tau,
                receipt.reservation_price,
                receipt.inventory_skew_dollar,
                receipt.optimal_bid_price,
                receipt.optimal_ask_price,
                receipt.optimal_spread_dollar,
                receipt.optimal_spread_bps,
                receipt.bid_depth_delta,
                receipt.ask_depth_delta,
                receipt.bid_fill_probability,
                receipt.ask_fill_probability,
                receipt.inventory_risk_penalty,
                receipt.regime,
                receipt.derivation_mode,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

