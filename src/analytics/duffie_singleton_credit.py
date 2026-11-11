"""Darrell Duffie & Kenneth J. Singleton (1999) Reduced-Form Credit Spread & Hazard Engine.

Theoretical Foundation:
    Duffie, Darrell, and Singleton, Kenneth J. (1999).
    "Modeling Term Structures of Defaultable Bonds".
    Review of Financial Studies, 12(4), pp. 687-720.

Key Concepts:
    - Recovery of Market Value (RMV) Default-Adjusted Short Rate:
      Given risk-free rate r, Poisson default hazard intensity h > 0, and fractional
      loss-given-default L = 1 - omega (where omega in [0, 0.99] is RMV recovery rate),
      a defaultable zero-coupon bond prices cleanly as default-free using adjusted rate:
      R = r + h * L = r + s,   v_RMV(T) = Notional * exp(-R * T).
    - Credit Triangle & Recovery-of-Treasury Convexity Bound:
      Par instantaneous credit spread obeys s == h * (1 - omega), while Recovery of
      Treasury bond valuation v_RT(T) = Notional * exp(-r * T) * [omega + (1 - omega)*exp(-h*T)]
      strictly satisfies v_RMV(T) <= v_RT(T) by compounding convexity (equality at omega = 0).
"""

from __future__ import annotations

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
HMAC_SECRET = b"duffie-singleton-credit-v173-secret"


@dataclass(frozen=True)
class DuffieSingletonReceipt:
    """Immutable quantitative audit receipt for Duffie & Singleton (1999) Credit."""

    ticker: str
    hazard_rate_h: float
    recovery_rate_omega: float
    loss_given_default_lgd: float
    risk_free_rate_r: float
    maturity_years: float
    notional: float
    credit_spread_s: float
    credit_spread_bps: float
    adjusted_short_rate: float
    rmv_bond_price: float
    rt_bond_price: float
    risk_free_bond_price: float
    cva_discount: float
    survival_probability: float
    default_probability: float
    rmv_rt_convexity_gap: float
    implied_hazard_check: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def generate_hmac_signature(payload: str, secret: bytes = HMAC_SECRET) -> str:
    """Generate SHA-256 HMAC signature for cryptographic receipt integrity."""
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(
    payload: str,
    signature: str,
    secret: bytes = HMAC_SECRET,
) -> bool:
    """Constant-time verification of HMAC-SHA256 receipt signature."""
    expected = generate_hmac_signature(payload, secret=secret)
    return hmac.compare_digest(expected, signature)


def classify_duffie_singleton_regime(credit_spread_bps: float) -> str:
    """Classify corporate credit risk regime by credit-triangle spread in bps."""
    if credit_spread_bps >= 500.0:
        return "DISTRESSED_JUMP_TO_DEFAULT"
    if credit_spread_bps >= 250.0:
        return "HIGH_YIELD_INTENSITY_BUILDUP"
    if credit_spread_bps >= 100.0:
        return "CROSSOVER_SPREAD_WATCH"
    return "INVESTMENT_GRADE_LOW_HAZARD"


def _build_hmac_payload(
    ticker: str,
    hazard_h: float,
    recovery_omega: float,
    rate_r: float,
    maturity: float,
    spread_s: float,
    rmv_price: float,
    survival_prob: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{hazard_h:.6f}:{recovery_omega:.4f}:{rate_r:.6f}:"
        f"{maturity:.4f}:{spread_s:.6f}:{rmv_price:.6f}:{survival_prob:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_duffie_singleton_credit(
    hazard_rate_h: float,
    recovery_rate_omega: float = 0.40,
    risk_free_rate_r: float = 0.045,
    maturity_years: float = 5.0,
    notional: float = 100.0,
    ticker: str = "HYG",
) -> DuffieSingletonReceipt:
    """Price defaultable zero-coupon debt under Duffie & Singleton (1999) RMV & RT."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    h_rate = float(hazard_rate_h)
    omega = float(recovery_rate_omega)
    rate_r = float(risk_free_rate_r)
    maturity = float(maturity_years)
    face = float(notional)

    if not (0.0 < h_rate <= 2.0):
        raise ValueError("Hazard intensity h must be in (0, 2.0].")
    if not (0.0 <= omega <= 0.99):
        raise ValueError("Recovery rate omega must be in [0.0, 0.99].")
    if not (-0.05 <= rate_r <= 0.40):
        raise ValueError("Risk-free rate r must be in [-0.05, 0.40].")
    if not (0.0 < maturity <= 50.0):
        raise ValueError("Maturity T must be in (0, 50.0] years.")
    if not (0.0 < face <= 1e12):
        raise ValueError("Notional must be in (0, 1e12].")

    lgd = 1.0 - omega
    spread_s = h_rate * lgd
    spread_bps = spread_s * 10000.0
    adj_rate = rate_r + spread_s

    rf_price = face * math.exp(-rate_r * maturity)
    rmv_price = face * math.exp(-adj_rate * maturity)
    surv_prob = math.exp(-h_rate * maturity)
    def_prob = 1.0 - surv_prob
    rt_price = rf_price * (omega + lgd * surv_prob)

    cva_disc = max(0.0, rf_price - rmv_price)
    convexity_gap = max(0.0, rt_price - rmv_price)
    implied_h = spread_s / lgd

    regime = classify_duffie_singleton_regime(spread_bps)
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        hazard_h=h_rate,
        recovery_omega=omega,
        rate_r=rate_r,
        maturity=maturity,
        spread_s=spread_s,
        rmv_price=rmv_price,
        survival_prob=surv_prob,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return DuffieSingletonReceipt(
        ticker=cleaned_ticker,
        hazard_rate_h=h_rate,
        recovery_rate_omega=omega,
        loss_given_default_lgd=lgd,
        risk_free_rate_r=rate_r,
        maturity_years=maturity,
        notional=face,
        credit_spread_s=spread_s,
        credit_spread_bps=spread_bps,
        adjusted_short_rate=adj_rate,
        rmv_bond_price=rmv_price,
        rt_bond_price=rt_price,
        risk_free_bond_price=rf_price,
        cva_discount=cva_disc,
        survival_probability=surv_prob,
        default_probability=def_prob,
        rmv_rt_convexity_gap=convexity_gap,
        implied_hazard_check=implied_h,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_duffie_singleton_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "HYG",
    recovery_rate_omega: float = 0.40,
    risk_free_rate_r: float = 0.045,
    maturity_years: float = 5.0,
) -> DuffieSingletonReceipt:
    """Calibrate implied default hazard intensity from trailing asset volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.05, 1.50))
        h_est = float(np.clip(0.005 + 0.12 * (ann_vol ** 1.5), 0.002, 0.50))
    else:
        h_est = 0.025

    return calculate_duffie_singleton_credit(
        hazard_rate_h=h_est,
        recovery_rate_omega=float(recovery_rate_omega),
        risk_free_rate_r=float(risk_free_rate_r),
        maturity_years=float(maturity_years),
        notional=100.0,
        ticker=ticker,
    )


def persist_duffie_singleton_record(
    receipt: DuffieSingletonReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Duffie & Singleton (1999) credit audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS duffie_singleton_credit_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        hazard_rate_h REAL NOT NULL,
        recovery_rate_omega REAL NOT NULL,
        maturity_years REAL NOT NULL,
        credit_spread_bps REAL NOT NULL,
        adjusted_short_rate REAL NOT NULL,
        rmv_bond_price REAL NOT NULL,
        rt_bond_price REAL NOT NULL,
        cva_discount REAL NOT NULL,
        survival_probability REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO duffie_singleton_credit_audits (
        ticker, hazard_rate_h, recovery_rate_omega, maturity_years,
        credit_spread_bps, adjusted_short_rate, rmv_bond_price, rt_bond_price,
        cva_discount, survival_probability, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.hazard_rate_h,
                receipt.recovery_rate_omega,
                receipt.maturity_years,
                receipt.credit_spread_bps,
                receipt.adjusted_short_rate,
                receipt.rmv_bond_price,
                receipt.rt_bond_price,
                receipt.cva_discount,
                receipt.survival_probability,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

