"""Hayne E. Leland & David H. Pyle (1977) Insider Equity Retention Signaling Engine.

Theoretical Foundation:
    Leland, Hayne E., & Pyle, David H. (1977). "Informational Asymmetries, Financial Structure,
    and Financial Intermediation". The Journal of Finance, 32(2), pp. 371-387.
    Seminal paper demonstrating that risk-averse entrepreneurs signal private project quality
    credibly to uninformed capital markets by retaining costly undiversified insider equity
    stakes (alpha), while simultaneously explaining why financial intermediaries form coalitions
    to economize on deadweight diversification costs.

Key Concepts:
    1. Separating Rational Expectations Signaling Schedule:
       With entrepreneur risk-aversion scale K_lp = gamma * sigma^2 * mu_min, market valuation
       is strictly increasing and strictly convex in insider equity retention alpha in [0, 1):
           V(alpha) = mu_min - K_lp * [ alpha + ln(1 - alpha) ]
       Satisfying the Incentive Compatibility First-Order Condition identically:
           (1 - alpha) * dV/dalpha = K_lp * alpha

    2. Deadweight Undiversified Risk-Bearing Cost:
       Retaining alpha of risky equity instead of fully diversifying imposes cost:
           C_risk(alpha) = 0.5 * K_lp * alpha^2
       Financial intermediation adds net surplus whenever verification cost < C_risk(alpha).

    3. Quality-Implied Retention & Insider Selling Discount:
       Higher intrinsic project quality mu requires higher credible retention alpha*(mu).
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
HMAC_SECRET = b"leland-pyle-signaling-v144-secret"


@dataclass(frozen=True)
class LelandPyleReceipt:
    """Immutable quantitative audit receipt for Leland & Pyle (1977) ownership signaling."""

    ticker: str
    floor_value_mu_min: float            # mu_min (lemon floor)
    true_project_value: float            # mu_true
    annual_volatility: float             # sigma
    relative_risk_aversion: float        # gamma
    insider_retention_alpha: float       # alpha in [0, 0.98]
    signaling_scale_k: float             # gamma * sigma^2 * mu_min
    signaled_market_value: float         # V(alpha)
    signaling_premium_dollar: float      # V(alpha) - mu_min
    signaling_premium_pct: float
    marginal_value_sensitivity: float    # dV/dalpha = K * alpha / (1 - alpha)
    diversification_risk_cost: float     # 0.5 * K * alpha^2
    quality_implied_alpha: float         # alpha*(mu_true)
    market_mispricing_wedge: float       # V(alpha) - mu_true
    intermediary_savings_threshold: float# Max monitoring cost C_risk(alpha)
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


def _solve_implied_alpha(target_lift: float, k_scale: float) -> float:
    """Invert V(alpha) - mu_min = -K * [alpha + ln(1 - alpha)] via 45-step bisection."""
    if target_lift <= 1e-12 or k_scale <= 1e-12:
        return 0.0
    target_g = target_lift / k_scale
    low, high = 0.0, 0.98
    if -(high + math.log(1.0 - high)) <= target_g:
        return high
    for _ in range(45):
        mid = 0.5 * (low + high)
        g_mid = -(mid + math.log(1.0 - mid))
        if g_mid < target_g:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def classify_signaling_regime(
    alpha: float,
    implied_alpha: float,
    signaling_premium_pct: float,
    risk_cost: float,
) -> str:
    """Classify the Leland-Pyle insider equity signaling and intermediation regime."""
    if alpha <= 1e-6:
        return "UNSIGNALED_LEMON_FLOOR_VALUATION"
    if alpha < implied_alpha - 0.10:
        return "INSUFFICIENT_SKIN_IN_GAME_DISCOUNTED"
    if risk_cost >= 5.0 and alpha >= 0.45:
        return "HIGH_SIGNALING_COST_INTERMEDIARY_JUSTIFIED"
    if signaling_premium_pct >= 0.15:
        return "HIGH_QUALITY_CREDIBLE_SEPARATING_SIGNAL"
    return "EQUILIBRIUM_SIGNALING_DIVERSIFICATION_TRADEOFF"


def calculate_leland_pyle_equilibrium(
    floor_value_mu_min: float = 100.0,
    true_project_value: float = 115.0,
    annual_volatility: float = 0.40,
    relative_risk_aversion: float = 3.0,
    insider_retention_alpha: float = 0.55,
    ticker: str = "SIGNAL",
) -> LelandPyleReceipt:
    """Solve Leland & Pyle (1977) separating signaling schedule and risk-bearing costs.

    Args:
        floor_value_mu_min: Lowest-quality baseline project value mu_min (> 0).
        true_project_value: Entrepreneur's true project expected value mu_true (>= mu_min).
        annual_volatility: Idiosyncratic/project asset return volatility sigma (> 0).
        relative_risk_aversion: Entrepreneur relative risk aversion gamma (>= 0).
        insider_retention_alpha: Equity stake alpha in [0.0, 0.98] retained by insiders.
        ticker: Security ticker symbol.

    Returns:
        LelandPyleReceipt detailing signaled value, marginal sensitivity, risk cost, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "SIGNAL"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    mu_min = float(floor_value_mu_min)
    if not math.isfinite(mu_min) or mu_min <= 0.0:
        raise ValueError(f"floor_value_mu_min must be finite and positive, got: {floor_value_mu_min}")

    mu_true = max(mu_min, float(true_project_value))
    sigma = max(0.01, float(annual_volatility))
    gamma = max(0.0, float(relative_risk_aversion))
    alpha = float(np.clip(insider_retention_alpha, 0.0, 0.98))

    # Leland-Pyle risk coefficient scaled to enterprise value units: K = gamma * sigma^2 * mu_min
    k_lp = gamma * (sigma ** 2) * mu_min

    # V(alpha) = mu_min - K * [alpha + ln(1 - alpha)]
    signaling_factor = -(alpha + math.log(max(1e-12, 1.0 - alpha)))
    prem_dollar = k_lp * signaling_factor
    v_market = mu_min + prem_dollar
    prem_pct = prem_dollar / mu_min

    # dV/dalpha = K * alpha / (1 - alpha)
    marginal_sens = k_lp * alpha / max(1e-8, 1.0 - alpha)
    # Diversification / signaling cost: C_risk(alpha) = 0.5 * K * alpha^2
    risk_cost = 0.5 * k_lp * (alpha ** 2)

    implied_alpha = _solve_implied_alpha(mu_true - mu_min, k_lp)
    mispricing_wedge = v_market - mu_true

    regime = classify_signaling_regime(
        alpha=alpha,
        implied_alpha=implied_alpha,
        signaling_premium_pct=prem_pct,
        risk_cost=risk_cost,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{mu_min:.4f}:{mu_true:.4f}:{sigma:.4f}:{gamma:.4f}:"
        f"{alpha:.4f}:{v_market:.4f}:{risk_cost:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return LelandPyleReceipt(
        ticker=cleaned_ticker,
        floor_value_mu_min=mu_min,
        true_project_value=mu_true,
        annual_volatility=sigma,
        relative_risk_aversion=gamma,
        insider_retention_alpha=alpha,
        signaling_scale_k=k_lp,
        signaled_market_value=v_market,
        signaling_premium_dollar=prem_dollar,
        signaling_premium_pct=prem_pct,
        marginal_value_sensitivity=marginal_sens,
        diversification_risk_cost=risk_cost,
        quality_implied_alpha=implied_alpha,
        market_mispricing_wedge=mispricing_wedge,
        intermediary_savings_threshold=risk_cost,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_leland_pyle_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SIGNAL",
    insider_retention_alpha: float = 0.55,
    relative_risk_aversion: float = 3.0,
) -> LelandPyleReceipt:
    """Empirically calibrate Leland-Pyle ownership signaling schedule from price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        sigma = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.15, 0.80))
        mu_min = spot * 0.90
        mu_true = max(mu_min, spot * 1.05)
    else:
        mu_min = 100.0
        mu_true = 115.0
        sigma = 0.40

    return calculate_leland_pyle_equilibrium(
        floor_value_mu_min=mu_min,
        true_project_value=mu_true,
        annual_volatility=sigma,
        relative_risk_aversion=relative_risk_aversion,
        insider_retention_alpha=insider_retention_alpha,
        ticker=ticker,
    )


def persist_leland_pyle_record(
    receipt: LelandPyleReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Leland-Pyle signaling record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS leland_pyle_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        floor_value_mu_min REAL NOT NULL,
        true_project_value REAL NOT NULL,
        annual_volatility REAL NOT NULL,
        relative_risk_aversion REAL NOT NULL,
        insider_retention_alpha REAL NOT NULL,
        signaling_scale_k REAL NOT NULL,
        signaled_market_value REAL NOT NULL,
        signaling_premium_dollar REAL NOT NULL,
        signaling_premium_pct REAL NOT NULL,
        marginal_value_sensitivity REAL NOT NULL,
        diversification_risk_cost REAL NOT NULL,
        quality_implied_alpha REAL NOT NULL,
        market_mispricing_wedge REAL NOT NULL,
        intermediary_savings_threshold REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_lp_ticker_time ON leland_pyle_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO leland_pyle_audits (
        ticker, floor_value_mu_min, true_project_value, annual_volatility,
        relative_risk_aversion, insider_retention_alpha, signaling_scale_k,
        signaled_market_value, signaling_premium_dollar, signaling_premium_pct,
        marginal_value_sensitivity, diversification_risk_cost, quality_implied_alpha,
        market_mispricing_wedge, intermediary_savings_threshold, regime,
        hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.floor_value_mu_min,
                    receipt.true_project_value,
                    receipt.annual_volatility,
                    receipt.relative_risk_aversion,
                    receipt.insider_retention_alpha,
                    receipt.signaling_scale_k,
                    receipt.signaled_market_value,
                    receipt.signaling_premium_dollar,
                    receipt.signaling_premium_pct,
                    receipt.marginal_value_sensitivity,
                    receipt.diversification_risk_cost,
                    receipt.quality_implied_alpha,
                    receipt.market_mispricing_wedge,
                    receipt.intermediary_savings_threshold,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

