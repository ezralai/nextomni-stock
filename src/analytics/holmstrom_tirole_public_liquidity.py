"""Bengt Holmström & Jean Tirole (1998) Private and Public Supply of Liquidity Engine.

Theoretical Foundation:
    Holmström, Bengt, & Tirole, Jean (1998).
    "Private and Public Supply of Liquidity".
    Journal of Political Economy, 106(1), pp. 1-40.

Key Concepts:
    - Idiosyncratic Pooling vs. Aggregate Correlated Liquidity Shortage:
      Firms facing interim reinvestment shocks rho_shock > rho_0 experience a pledgeable
      liquidity gap = max(0, rho_shock - rho_0). Purely idiosyncratic shocks (lambda_agg = 0)
      can be fully pooled by private intermediaries without wasteful cash hoarding (q_L = 1.0).
      When aggregate correlated risk lambda_agg > 0 strikes simultaneously, private claims
      lose liquidity value, driving Treasury liquidity premiums q_L > 1.0 and shrinking
      feasible investment scale below contingent public-liquidity benchmarks.
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
HMAC_SECRET = b"holmstrom-tirole-public-liquidity-v160-secret"


@dataclass(frozen=True)
class PublicLiquidityReceipt:
    """Immutable quantitative audit receipt for Holmström & Tirole (1998) liquidity supply."""

    ticker: str
    entrepreneur_equity_a: float
    project_cost_i: float
    pledgeable_return_rho0: float
    total_return_rho1: float
    interim_shock_rho: float
    shock_prob_theta: float
    aggregate_corr_lambda: float
    pledgeable_liquidity_gap: float
    continuation_efficient: bool
    hoarding_investment_scale: float
    pooled_investment_scale: float
    aggregate_investment_scale: float
    public_liquidity_scale_gain: float
    treasury_liquidity_price_ql: float
    liquidity_shortage_premium_bps: float
    optimal_reserve_ratio_pct: float
    welfare_gain_public_supply: float
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


def classify_public_liquidity_regime(
    gap: float,
    continuation_ok: bool,
    lambda_agg: float,
    prem_bps: float,
) -> str:
    """Classify corporate liquidity supply adequacy and public intervention need."""
    if not continuation_ok:
        return "EX_POST_LIQUIDATION_EFFICIENT_SHOCK_EXCEEDS_RHO1"
    if gap <= 1e-6:
        return "SELF_LIQUID_PLEDGEABLE_INCOME_SUFFICIENT"
    if lambda_agg <= 1e-6:
        return "PRIVATE_INTERMEDIARY_POOLING_SECOND_BEST"
    if prem_bps >= 450.0 or lambda_agg >= 0.70:
        return "SEVERE_AGGREGATE_LIQUIDITY_SHORTAGE_PUBLIC_TREASURY_NEEDED"
    return "MODERATE_AGGREGATE_LIQUIDITY_PREMIUM"


def _build_hmac_payload(
    ticker: str,
    equity_a: float,
    rho0: float,
    rho1: float,
    shock_rho: float,
    lambda_agg: float,
    agg_scale: float,
    q_l: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{equity_a:.4f}:{rho0:.4f}:{rho1:.4f}:{shock_rho:.4f}:"
        f"{lambda_agg:.4f}:{agg_scale:.4f}:{q_l:.6f}:{regime}:{timestamp}"
    )


def calculate_public_liquidity_supply(
    entrepreneur_equity_a: float = 20.0,
    project_cost_i: float = 1.0,
    pledgeable_return_rho0: float = 0.70,
    total_return_rho1: float = 1.45,
    interim_shock_rho: float = 0.90,
    shock_prob_theta: float = 0.35,
    aggregate_corr_lambda: float = 0.60,
    ticker: str = "HT98",
) -> PublicLiquidityReceipt:
    """Evaluate Holmström & Tirole (1998) private pooling vs. public liquidity supply equilibrium."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    eq_a = float(entrepreneur_equity_a)
    cost_i = float(project_cost_i)
    rho0 = float(pledgeable_return_rho0)
    rho1 = float(total_return_rho1)
    shock = float(interim_shock_rho)
    theta = float(shock_prob_theta)
    lam = float(aggregate_corr_lambda)

    if eq_a <= 0.0 or cost_i <= 0.0 or rho0 <= 0.0:
        raise ValueError("entrepreneur_equity_a, project_cost_i, and pledgeable_return_rho0 must be positive.")
    if rho1 <= rho0:
        raise ValueError("total_return_rho1 must strictly exceed pledgeable_return_rho0.")
    if shock < 0.0:
        raise ValueError("interim_shock_rho must be non-negative.")
    if not (0.0 < theta < 1.0):
        raise ValueError("shock_prob_theta must satisfy 0.0 < theta < 1.0.")
    if not (0.0 <= lam <= 1.0):
        raise ValueError("aggregate_corr_lambda must satisfy 0.0 <= lambda <= 1.0.")

    cont_ok = shock <= rho1
    gap = max(0.0, shock - rho0) if cont_ok else 0.0

    # Effective per-unit net funding costs across institutional liquidity mechanisms
    c_pooled = max(0.05, cost_i + theta * gap - rho0)
    c_hoard = max(0.05, cost_i + gap - rho0)
    c_agg = (1.0 - lam) * c_pooled + lam * c_hoard

    i_pooled = eq_a / c_pooled
    i_hoard = eq_a / c_hoard
    i_agg = eq_a / c_agg

    pub_gain = max(0.0, i_pooled - i_agg) if cont_ok else 0.0

    if not cont_ok or gap <= 1e-9 or lam <= 1e-9:
        q_l = 1.0
    else:
        q_l = 1.0 + lam * theta * (gap / max(1e-6, rho1 - rho0))

    prem_bps = (q_l - 1.0) * 10000.0
    res_ratio_pct = ((lam * gap) / cost_i) * 100.0

    # Social NPV per unit investment under continuation
    unit_social_surplus = max(0.0, rho1 - cost_i - theta * shock) if cont_ok else max(0.0, (1.0 - theta) * (rho1 - cost_i))
    welfare_gain = pub_gain * unit_social_surplus

    regime = classify_public_liquidity_regime(
        gap=gap,
        continuation_ok=cont_ok,
        lambda_agg=lam,
        prem_bps=prem_bps,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        equity_a=eq_a,
        rho0=rho0,
        rho1=rho1,
        shock_rho=shock,
        lambda_agg=lam,
        agg_scale=i_agg,
        q_l=q_l,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return PublicLiquidityReceipt(
        ticker=cleaned_ticker,
        entrepreneur_equity_a=eq_a,
        project_cost_i=cost_i,
        pledgeable_return_rho0=rho0,
        total_return_rho1=rho1,
        interim_shock_rho=shock,
        shock_prob_theta=theta,
        aggregate_corr_lambda=lam,
        pledgeable_liquidity_gap=gap,
        continuation_efficient=cont_ok,
        hoarding_investment_scale=i_hoard,
        pooled_investment_scale=i_pooled,
        aggregate_investment_scale=i_agg,
        public_liquidity_scale_gain=pub_gain,
        treasury_liquidity_price_ql=q_l,
        liquidity_shortage_premium_bps=prem_bps,
        optimal_reserve_ratio_pct=res_ratio_pct,
        welfare_gain_public_supply=welfare_gain,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_public_liquidity_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "HT98",
    aggregate_corr_lambda: float = 0.60,
    shock_prob_theta: float = 0.35,
) -> PublicLiquidityReceipt:
    """Calibrate interim liquidity shock severity from trailing asset volatility and drawdowns."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        running_max = np.maximum.accumulate(window)
        max_dd = float(np.max((running_max - window) / np.maximum(1e-6, running_max)))
        interim_shock = float(np.clip(0.75 + 0.8 * max_dd + 0.4 * ann_vol, 0.65, 1.35))
    else:
        interim_shock = 0.90

    return calculate_public_liquidity_supply(
        interim_shock_rho=interim_shock,
        shock_prob_theta=float(shock_prob_theta),
        aggregate_corr_lambda=float(aggregate_corr_lambda),
        ticker=ticker,
    )


def persist_public_liquidity_record(
    receipt: PublicLiquidityReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Holmström & Tirole (1998) liquidity supply evaluation to SQLite via ? queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ht1998_public_liquidity_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        entrepreneur_equity_a REAL NOT NULL,
        interim_shock_rho REAL NOT NULL,
        aggregate_corr_lambda REAL NOT NULL,
        pledgeable_liquidity_gap REAL NOT NULL,
        pooled_investment_scale REAL NOT NULL,
        aggregate_investment_scale REAL NOT NULL,
        public_liquidity_scale_gain REAL NOT NULL,
        treasury_liquidity_price_ql REAL NOT NULL,
        liquidity_shortage_premium_bps REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO ht1998_public_liquidity_audits (
        ticker, entrepreneur_equity_a, interim_shock_rho, aggregate_corr_lambda,
        pledgeable_liquidity_gap, pooled_investment_scale, aggregate_investment_scale,
        public_liquidity_scale_gain, treasury_liquidity_price_ql, liquidity_shortage_premium_bps,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.entrepreneur_equity_a,
                receipt.interim_shock_rho,
                receipt.aggregate_corr_lambda,
                receipt.pledgeable_liquidity_gap,
                receipt.pooled_investment_scale,
                receipt.aggregate_investment_scale,
                receipt.public_liquidity_scale_gain,
                receipt.treasury_liquidity_price_ql,
                receipt.liquidity_shortage_premium_bps,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

