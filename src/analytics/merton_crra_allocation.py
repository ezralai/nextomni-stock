"""Samuelson (1969) & Merton (1969/1971) CRRA Lifetime Portfolio & Consumption Engine.

Implements:
- Continuous-time intertemporal portfolio choice and consumption with CRRA utility:
      U(c) = c^(1 - gamma) / (1 - gamma)   (or ln(c) when gamma = 1)
- Myopic Merton Share:
      w_myopic = (mu - r_f) / (gamma * sigma^2)
- Merton-Campbell-Viceira Intertemporal Hedging Demand:
      w_hedge = (1 - 1 / gamma) * (-rho_state) * (sigma_state / sigma) * horizon_factor
      w_total = w_myopic + w_hedge
- Optimal continuous consumption-to-wealth rate c / W:
      c / W = rho / gamma + (1 - 1 / gamma) * (r_f + (mu - r_f)^2 / (2 * gamma * sigma^2))
      (smooth analytical limit c / W -> rho as gamma -> 1)
- Certainty-Equivalent Return (r_ce), Expected Growth Rate (g_port), and
  allocation regime classification
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

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-merton-1969-audit-key"


@dataclass(frozen=True)
class MertonCRRAReceipt:
    """Immutable receipt for Samuelson (1969) & Merton (1969/1971) CRRA allocation."""

    ticker: str
    gamma_risk_aversion: float
    time_discount_rate: float
    expected_return: float
    volatility: float
    risk_free_rate: float
    state_correlation: float
    merton_share_myopic: float
    hedging_demand: float
    total_risky_weight: float
    consumption_wealth_ratio: float
    certainty_equivalent_return: float
    expected_growth_rate: float
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
    gamma_risk_aversion: float,
    time_discount_rate: float,
    expected_return: float,
    volatility: float,
    risk_free_rate: float,
    total_risky_weight: float,
    consumption_wealth_ratio: float,
    certainty_equivalent_return: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{gamma_risk_aversion:.6f}|{time_discount_rate:.6f}|"
        f"{expected_return:.6f}|{volatility:.6f}|{risk_free_rate:.6f}|"
        f"{total_risky_weight:.8f}|{consumption_wealth_ratio:.8f}|"
        f"{certainty_equivalent_return:.8f}|{regime}|{timestamp}"
    )


def sign_merton_crra_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_merton_crra_receipt(payload)
    return hmac.compare_digest(expected, signature)


def calculate_merton_crra_allocation(
    expected_return: float = 0.09,
    volatility: float = 0.16,
    risk_free_rate: float = 0.04,
    gamma_risk_aversion: float = 3.0,
    time_discount_rate: float = 0.03,
    state_correlation: float = -0.40,
    state_volatility: float = 0.08,
    horizon_factor: float = 1.5,
    ticker: str = "SPY",
) -> MertonCRRAReceipt:
    """Compute Samuelson (1969) & Merton (1969/1971) optimal CRRA portfolio & consumption."""
    clean_ticker = _validate_ticker(ticker)

    mu = float(expected_return)
    sigma = float(volatility)
    rf = float(risk_free_rate)
    gamma = float(gamma_risk_aversion)
    rho = float(time_discount_rate)
    rho_state = float(state_correlation)
    sig_state = float(state_volatility)
    h_factor = float(horizon_factor)

    if not math.isfinite(mu) or not math.isfinite(rf):
        raise ValueError("expected_return and risk_free_rate must be finite")
    if sigma <= 0.0 or not math.isfinite(sigma):
        raise ValueError("volatility must be strictly positive and finite")
    if gamma <= 0.0 or not math.isfinite(gamma):
        raise ValueError("gamma_risk_aversion must be strictly positive and finite")
    if rho <= 0.0 or not math.isfinite(rho):
        raise ValueError("time_discount_rate must be strictly positive and finite")
    if not (-1.0 <= rho_state <= 1.0):
        raise ValueError("state_correlation must be in [-1.0, 1.0]")
    if sig_state < 0.0 or not math.isfinite(sig_state):
        raise ValueError("state_volatility must be non-negative and finite")
    if h_factor < 0.0 or not math.isfinite(h_factor):
        raise ValueError("horizon_factor must be non-negative and finite")

    # 1. Myopic Merton Share
    excess = mu - rf
    var_stock = sigma * sigma
    w_myopic = excess / (gamma * var_stock)

    # 2. Intertemporal Hedging Demand (Campbell-Viceira / Merton 1971)
    # When rho_state < 0 (mean reversion), investors with gamma > 1 demand positive hedge
    w_hedge = (1.0 - (1.0 / gamma)) * (-rho_state) * (sig_state / sigma) * h_factor
    w_total = w_myopic + w_hedge

    # 3. Certainty-Equivalent Return
    r_ce = rf + (excess * excess) / (2.0 * gamma * var_stock)

    # 4. Optimal Consumption-to-Wealth Ratio c / W
    # Limit as gamma -> 1 is exactly rho (Log utility / Kelly)
    if abs(gamma - 1.0) < 1e-7:
        c_w_ratio = rho
    else:
        c_w_ratio = (rho / gamma) + (1.0 - (1.0 / gamma)) * r_ce
    c_w_ratio = max(c_w_ratio, 1e-4)

    # 5. Expected Portfolio Growth Rate: g = rf + w * excess - 0.5 * w^2 * var_stock
    growth_rate = rf + w_total * excess - 0.5 * (w_total ** 2) * var_stock

    if w_total > 1.0:
        regime = "LEVERED_GROWTH_KELLY"
    elif w_total >= 0.50:
        regime = "BALANCED_CRRA_INTERTEMPORAL"
    elif w_total >= 0.15:
        regime = "CONSERVATIVE_INCOME_PRESERVATION"
    else:
        regime = "DEFENSIVE_CASH_TILT"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        gamma_risk_aversion=gamma,
        time_discount_rate=rho,
        expected_return=mu,
        volatility=sigma,
        risk_free_rate=rf,
        total_risky_weight=w_total,
        consumption_wealth_ratio=c_w_ratio,
        certainty_equivalent_return=r_ce,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_merton_crra_receipt(payload)

    return MertonCRRAReceipt(
        ticker=clean_ticker,
        gamma_risk_aversion=gamma,
        time_discount_rate=rho,
        expected_return=mu,
        volatility=sigma,
        risk_free_rate=rf,
        state_correlation=rho_state,
        merton_share_myopic=w_myopic,
        hedging_demand=w_hedge,
        total_risky_weight=w_total,
        consumption_wealth_ratio=c_w_ratio,
        certainty_equivalent_return=r_ce,
        expected_growth_rate=growth_rate,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_merton_crra_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    gamma_risk_aversion: float = 3.0,
    risk_free_rate: float = 0.04,
) -> MertonCRRAReceipt:
    """Calibrate annualized moments from empirical price series for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions for Merton CRRA allocation")

    log_ret = np.diff(np.log(arr))
    ann_mu = float(np.clip(np.mean(log_ret) * 252.0, -0.10, 0.40))
    ann_vol = float(np.clip(np.std(log_ret, ddof=1) * math.sqrt(252.0), 0.06, 0.60))

    # Realized momentum vs volatility as proxy state variable
    return calculate_merton_crra_allocation(
        expected_return=ann_mu,
        volatility=ann_vol,
        risk_free_rate=risk_free_rate,
        gamma_risk_aversion=gamma_risk_aversion,
        time_discount_rate=0.03,
        state_correlation=-0.35,
        state_volatility=ann_vol * 0.5,
        horizon_factor=1.2,
        ticker=ticker,
    )


def persist_merton_crra_record(
    receipt: MertonCRRAReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Merton CRRA allocation receipt to SQLite with foreign keys enforced."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS merton_crra_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                gamma_risk_aversion REAL NOT NULL,
                time_discount_rate REAL NOT NULL,
                expected_return REAL NOT NULL,
                volatility REAL NOT NULL,
                risk_free_rate REAL NOT NULL,
                state_correlation REAL NOT NULL,
                merton_share_myopic REAL NOT NULL,
                hedging_demand REAL NOT NULL,
                total_risky_weight REAL NOT NULL,
                consumption_wealth_ratio REAL NOT NULL,
                certainty_equivalent_return REAL NOT NULL,
                expected_growth_rate REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO merton_crra_records (
                ticker, gamma_risk_aversion, time_discount_rate,
                expected_return, volatility, risk_free_rate, state_correlation,
                merton_share_myopic, hedging_demand, total_risky_weight,
                consumption_wealth_ratio, certainty_equivalent_return,
                expected_growth_rate, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.gamma_risk_aversion,
                receipt.time_discount_rate,
                receipt.expected_return,
                receipt.volatility,
                receipt.risk_free_rate,
                receipt.state_correlation,
                receipt.merton_share_myopic,
                receipt.hedging_demand,
                receipt.total_risky_weight,
                receipt.consumption_wealth_ratio,
                receipt.certainty_equivalent_return,
                receipt.expected_growth_rate,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

