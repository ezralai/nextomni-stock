"""Lawrence R. Glosten & Paul R. Milgrom (1985) Sequential Trade Adverse Selection Spread Engine.

Theoretical Foundation:
    Glosten, Lawrence R., & Milgrom, Paul R. (1985). "Bid, Ask and Transaction Prices in a
    Specialist Market with Heterogeneously Informed Traders". Journal of Financial Economics,
    14(1), pp. 71-100.
    The Nobel-Prize foundational sequential-trade market microstructure model demonstrating
    that bid-ask spreads arise purely from asymmetric information adverse selection, even
    under zero inventory costs and zero transaction overhead.

Key Concepts:
    1. Regret-Free Competitive Specialist Quotes:
       Asset fundamental value V in {V_L, V_H} with prior P(V = V_H) = theta.
       Informed traders (fraction pi) trade directionally on private news while noise
       traders (1 - pi) buy or sell symmetrically. Market maker sets Bayes-updated quotes:
           Ask = E[V | Buy]  = V_L + theta_Buy * (V_H - V_L)
           Bid = E[V | Sell] = V_L + theta_Sell * (V_H - V_L)

    2. Adverse Selection Spread Identity:
       At symmetric prior theta = 0.50, spread simplifies exactly to S = pi * (V_H - V_L).
       Zero informed flow (pi = 0) collapses spread to zero; pi = 1 triggers illiquid market breakdown.

    3. Sequential Bayesian Price Discovery:
       Each consecutive buy or sell order updates belief theta_t toward the true state (0 or 1),
       systematically narrowing posterior spreads as private information impounds into prices.
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
HMAC_SECRET = b"glosten-milgrom-spread-v143-secret"


@dataclass(frozen=True)
class GlostenMilgromReceipt:
    """Immutable quantitative audit receipt for Glosten & Milgrom (1985) sequential quote setting."""

    ticker: str
    value_low: float                  # V_L
    value_high: float                 # V_H
    value_uncertainty_band: float     # V_H - V_L
    informed_fraction_pi: float       # pi in [0, 1]
    prior_prob_high: float            # theta_0
    initial_bid: float
    initial_ask: float
    initial_midprice: float
    initial_spread: float
    initial_spread_bps: float
    order_flow_steps: int
    posterior_prob_high: float        # theta_T after order flow
    final_bid: float
    final_ask: float
    final_midprice: float
    final_spread: float
    spread_narrowing_pct: float       # (initial_spread - final_spread) / initial_spread
    price_discovery_gain: float       # Reduction in posterior uncertainty std
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


def _bayes_quotes(
    theta: float,
    pi: float,
    v_low: float,
    v_high: float,
) -> tuple[float, float, float, float, float]:
    """Compute Bayes-updated posterior probabilities and regret-free Bid/Ask at belief theta."""
    dv = v_high - v_low
    # Prob(Buy | V_H) = (1 + pi)/2, Prob(Buy | V_L) = (1 - pi)/2
    p_buy_h = 0.5 * (1.0 + pi)
    p_buy_l = 0.5 * (1.0 - pi)
    denom_buy = theta * p_buy_h + (1.0 - theta) * p_buy_l
    theta_buy = (theta * p_buy_h) / denom_buy if denom_buy > 1e-14 else theta

    # Prob(Sell | V_H) = (1 - pi)/2, Prob(Sell | V_L) = (1 + pi)/2
    p_sell_h = 0.5 * (1.0 - pi)
    p_sell_l = 0.5 * (1.0 + pi)
    denom_sell = theta * p_sell_h + (1.0 - theta) * p_sell_l
    theta_sell = (theta * p_sell_h) / denom_sell if denom_sell > 1e-14 else theta

    ask = v_low + theta_buy * dv
    bid = v_low + theta_sell * dv
    mid = v_low + theta * dv
    return bid, ask, mid, theta_buy, theta_sell


def classify_glosten_milgrom_regime(
    informed_pi: float,
    final_spread_bps: float,
    spread_narrowing_pct: float,
) -> str:
    """Classify the Glosten-Milgrom adverse selection & sequential price discovery regime."""
    if informed_pi >= 0.99:
        return "EXTREME_ADVERSE_SELECTION_MARKET_BREAKDOWN"
    if informed_pi <= 1e-6:
        return "ZERO_INFO_ASYMMETRY_ZERO_SPREAD"
    if spread_narrowing_pct >= 0.25:
        return "BAYESIAN_PRICE_DISCOVERY_CONVERGENCE"
    if final_spread_bps >= 500.0:
        return "HIGH_TOXICITY_WIDE_SPECIALIST_SPREAD"
    return "COMPETITIVE_SEQUENTIAL_LIQUIDITY_EQUILIBRIUM"


def calculate_glosten_milgrom_spread(
    value_low: float = 90.0,
    value_high: float = 110.0,
    informed_fraction_pi: float = 0.25,
    prior_prob_high: float = 0.50,
    order_flow: Optional[Sequence[int]] = None,
    ticker: str = "BOOK",
) -> GlostenMilgromReceipt:
    """Solve Glosten & Milgrom (1985) regret-free specialist Bid/Ask spreads & sequential learning.

    Args:
        value_low: Low asset terminal liquidation state V_L (> 0).
        value_high: High asset terminal liquidation state V_H (> V_L).
        informed_fraction_pi: Fraction of informed traders pi in [0.0, 1.0].
        prior_prob_high: Market maker's prior belief P(V = V_H) in (0.001, 0.999).
        order_flow: Optional sequence of signed order indicators (+1 buy, -1 sell).
        ticker: Security ticker symbol.

    Returns:
        GlostenMilgromReceipt detailing initial/posterior spreads, Bayesian learning, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "BOOK"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    vl = float(value_low)
    vh = float(value_high)
    if not math.isfinite(vl) or not math.isfinite(vh) or vl <= 0.0 or vh <= vl:
        raise ValueError("Require 0 < value_low < value_high with finite values.")

    pi = float(np.clip(informed_fraction_pi, 0.0, 1.0))
    theta_0 = float(np.clip(prior_prob_high, 1e-4, 1.0 - 1e-4))
    dv = vh - vl

    bid_0, ask_0, mid_0, _, _ = _bayes_quotes(theta_0, pi, vl, vh)
    spread_0 = max(0.0, ask_0 - bid_0)
    spread_0_bps = (spread_0 / max(1e-8, mid_0)) * 10000.0

    # Sequential Bayesian updating along order_flow
    theta_t = theta_0
    steps = 0
    if order_flow is not None:
        for order in order_flow:
            if order > 0:
                _, _, _, theta_buy, _ = _bayes_quotes(theta_t, pi, vl, vh)
                theta_t = theta_buy
                steps += 1
            elif order < 0:
                _, _, _, _, theta_sell = _bayes_quotes(theta_t, pi, vl, vh)
                theta_t = theta_sell
                steps += 1

    bid_t, ask_t, mid_t, _, _ = _bayes_quotes(theta_t, pi, vl, vh)
    spread_t = max(0.0, ask_t - bid_t)
    spread_t_bps = (spread_t / max(1e-8, mid_t)) * 10000.0

    narrowing = (spread_0 - spread_t) / spread_0 if spread_0 > 1e-12 else 0.0

    # Fundamental std dev sigma(V) = dv * sqrt(theta * (1 - theta))
    std_0 = dv * math.sqrt(theta_0 * (1.0 - theta_0))
    std_t = dv * math.sqrt(theta_t * (1.0 - theta_t))
    discovery_gain = max(0.0, std_0 - std_t)

    regime = classify_glosten_milgrom_regime(
        informed_pi=pi,
        final_spread_bps=spread_t_bps,
        spread_narrowing_pct=narrowing,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{vl:.4f}:{vh:.4f}:{pi:.4f}:{theta_0:.4f}:"
        f"{steps}:{theta_t:.6f}:{spread_0:.4f}:{spread_t:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return GlostenMilgromReceipt(
        ticker=cleaned_ticker,
        value_low=vl,
        value_high=vh,
        value_uncertainty_band=dv,
        informed_fraction_pi=pi,
        prior_prob_high=theta_0,
        initial_bid=bid_0,
        initial_ask=ask_0,
        initial_midprice=mid_0,
        initial_spread=spread_0,
        initial_spread_bps=spread_0_bps,
        order_flow_steps=steps,
        posterior_prob_high=theta_t,
        final_bid=bid_t,
        final_ask=ask_t,
        final_midprice=mid_t,
        final_spread=spread_t,
        spread_narrowing_pct=narrowing,
        price_discovery_gain=discovery_gain,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_glosten_milgrom_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BOOK",
    informed_fraction_pi: float = 0.25,
    uncertainty_band_pct: float = 0.20,
) -> GlostenMilgromReceipt:
    """Calibrate Glosten-Milgrom Bayesian specialist spread profile from price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        # Infer signed order flow proxy from trailing 5 daily return signs
        recent_rets = np.diff(p_arr[-min(len(p_arr), 6):])
        flow = [1 if r > 0 else (-1 if r < 0 else 0) for r in recent_rets]
    else:
        spot = 100.0
        flow = [1, 1, -1, 1]

    half_band = 0.5 * spot * max(0.02, float(uncertainty_band_pct))
    vl = max(1.0, spot - half_band)
    vh = spot + half_band

    return calculate_glosten_milgrom_spread(
        value_low=vl,
        value_high=vh,
        informed_fraction_pi=informed_fraction_pi,
        prior_prob_high=0.50,
        order_flow=flow,
        ticker=ticker,
    )


def persist_glosten_milgrom_record(
    receipt: GlostenMilgromReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Glosten-Milgrom sequential spread record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS glosten_milgrom_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        value_low REAL NOT NULL,
        value_high REAL NOT NULL,
        value_uncertainty_band REAL NOT NULL,
        informed_fraction_pi REAL NOT NULL,
        prior_prob_high REAL NOT NULL,
        initial_bid REAL NOT NULL,
        initial_ask REAL NOT NULL,
        initial_midprice REAL NOT NULL,
        initial_spread REAL NOT NULL,
        initial_spread_bps REAL NOT NULL,
        order_flow_steps INTEGER NOT NULL,
        posterior_prob_high REAL NOT NULL,
        final_bid REAL NOT NULL,
        final_ask REAL NOT NULL,
        final_midprice REAL NOT NULL,
        final_spread REAL NOT NULL,
        spread_narrowing_pct REAL NOT NULL,
        price_discovery_gain REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_gm_ticker_time ON glosten_milgrom_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO glosten_milgrom_audits (
        ticker, value_low, value_high, value_uncertainty_band, informed_fraction_pi,
        prior_prob_high, initial_bid, initial_ask, initial_midprice, initial_spread,
        initial_spread_bps, order_flow_steps, posterior_prob_high, final_bid,
        final_ask, final_midprice, final_spread, spread_narrowing_pct,
        price_discovery_gain, regime, hmac_signature, timestamp
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
                    receipt.value_low,
                    receipt.value_high,
                    receipt.value_uncertainty_band,
                    receipt.informed_fraction_pi,
                    receipt.prior_prob_high,
                    receipt.initial_bid,
                    receipt.initial_ask,
                    receipt.initial_midprice,
                    receipt.initial_spread,
                    receipt.initial_spread_bps,
                    receipt.order_flow_steps,
                    receipt.posterior_prob_high,
                    receipt.final_bid,
                    receipt.final_ask,
                    receipt.final_midprice,
                    receipt.final_spread,
                    receipt.spread_narrowing_pct,
                    receipt.price_discovery_gain,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

