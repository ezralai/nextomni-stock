"""Sanford J. Grossman & Oliver D. Hart (1980) Takeover Bids & Free-Rider Engine.

Theoretical Foundation:
    1. Grossman, Sanford J., & Hart, Oliver D. (1980). "Takeover Bids, The Free-Rider Problem,
       and the Theory of the Corporation". The Bell Journal of Economics, 11(1), pp. 42-64.
       Nobel-Prize corporate control analysis proving that atomistic shareholders refuse to
       tender below full post-takeover value v, causing value-creating M&A bids to fail
       unless corporate charters allow minority freeze-out dilution phi.

    2. Shleifer, Andrei, & Vishny, Robert W. (1986). "Large Shareholders and Corporate Control".
       Journal of Political Economy, 94(3), pp. 461-488.
       Demonstrates that an activist bidder's pre-bid toehold stake alpha accumulated at status-quo
       price q earns capital gains alpha * (v - q), overcoming free-riding without minority dilution.

Key Concepts:
    - Free-Rider Equilibrium Tender Price:
      p* = max(q, v - phi), where v = q + Delta_v is improved post-takeover value.
    - Raider Takeover Profit:
      pi_bidder = alpha * (v - q) + (1 - alpha) * (v - p*) - c
    - Optimal Charter Dilution Floor:
      phi* = max(0, (c - alpha * Delta_v) / (1 - alpha))
    - Social Surplus Lost to Free-Riding:
      When Delta_v > c yet pi_bidder <= 0, free-riding blocks efficient restructuring.
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
HMAC_SECRET = b"grossman-hart-takeover-v142-secret"


@dataclass(frozen=True)
class GrossmanHartTakeoverReceipt:
    """Immutable quantitative audit receipt for Grossman & Hart (1980) M&A takeover equilibrium."""

    ticker: str
    status_quo_value: float             # q
    restructuring_synergy: float        # Delta_v
    post_takeover_value: float          # v = q + Delta_v
    bidding_cost: float                 # c
    dilution_cap_phi: float             # phi
    pre_bid_toehold_alpha: float        # alpha in [0, 0.49]
    equilibrium_tender_price: float     # p* = max(q, v - phi)
    bidder_net_profit: float            # pi_bidder
    takeover_succeeds: bool
    target_bid_premium: float           # max(0, p* - q) if succeeds else 0
    target_bid_premium_pct: float
    optimal_min_dilution: float         # phi* = max(0, (c - alpha * dv)/(1 - alpha))
    breakeven_toehold_zero_phi: float   # alpha* = min(1, c / dv)
    missed_synergy_loss: float          # max(0, dv - c) if fails else 0
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


def classify_takeover_regime(
    synergy: float,
    bidding_cost: float,
    takeover_succeeds: bool,
    dilution_phi: float,
    toehold_alpha: float,
) -> str:
    """Classify the Grossman-Hart M&A tender offer and corporate control regime."""
    if synergy <= bidding_cost:
        return "UNECONOMIC_BID_INSUFFICIENT_SYNERGY"
    if not takeover_succeeds:
        return "FREE_RIDER_PARADOX_PARALYSIS"
    if dilution_phi <= 1e-8 and toehold_alpha > 0.0:
        return "SHLEIFER_VISHNY_TOEHOLD_CONTROL"
    if dilution_phi > synergy:
        return "EXCESSIVE_DILUTION_COERCED_TENDER"
    return "DILUTION_ENABLED_EFFICIENT_TAKEOVER"


def calculate_grossman_hart_takeover(
    status_quo_value: float = 100.0,
    restructuring_synergy: float = 30.0,
    bidding_cost: float = 10.0,
    dilution_cap_phi: float = 14.0,
    pre_bid_toehold_alpha: float = 0.05,
    ticker: str = "TARGET",
) -> GrossmanHartTakeoverReceipt:
    """Compute Grossman & Hart (1980) tender offer equilibrium, bidder profit, and free-rider loss.

    Args:
        status_quo_value: Target firm status-quo share/enterprise value q (> 0).
        restructuring_synergy: Value improvement created by raider Delta_v (>= 0).
        bidding_cost: Fixed takeover research & execution cost c (>= 0).
        dilution_cap_phi: Maximum minority freeze-out dilution allowed by charter phi (>= 0).
        pre_bid_toehold_alpha: Bidder's pre-announcement equity stake alpha in [0, 0.49].
        ticker: Security ticker symbol.

    Returns:
        GrossmanHartTakeoverReceipt with equilibrium tender bid, bidder profit, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "TARGET"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    q = float(status_quo_value)
    if not math.isfinite(q) or q <= 0.0:
        raise ValueError(f"status_quo_value must be finite and positive, got: {status_quo_value}")

    dv = float(restructuring_synergy)
    c = max(0.0, float(bidding_cost))
    phi = max(0.0, float(dilution_cap_phi))
    alpha = float(np.clip(pre_bid_toehold_alpha, 0.0, 0.49))

    v = q + dv
    # Atomistic shareholders hold out unless bid >= v - phi (and at least q if voluntary)
    p_star = max(q, v - phi)

    # Bidder gain = toehold capital gain alpha*(v - q) + gain on tendered shares (1-alpha)*(v - p*) - c
    bidder_profit = alpha * dv + (1.0 - alpha) * max(0.0, v - p_star) - c
    succeeds = (dv > 0.0) and (bidder_profit > 1e-8)

    if succeeds:
        prem = max(0.0, p_star - q)
        prem_pct = prem / q
        missed = 0.0
    else:
        prem = 0.0
        prem_pct = 0.0
        missed = max(0.0, dv - c)

    if dv > 1e-8:
        opt_phi = max(0.0, (c - alpha * dv) / (1.0 - alpha))
        be_toehold = float(np.clip(c / dv, 0.0, 1.0))
    else:
        opt_phi = c
        be_toehold = 1.0

    regime = classify_takeover_regime(
        synergy=dv,
        bidding_cost=c,
        takeover_succeeds=succeeds,
        dilution_phi=phi,
        toehold_alpha=alpha,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{q:.4f}:{dv:.4f}:{c:.4f}:{phi:.4f}:{alpha:.4f}:"
        f"{p_star:.4f}:{bidder_profit:.4f}:{succeeds}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return GrossmanHartTakeoverReceipt(
        ticker=cleaned_ticker,
        status_quo_value=q,
        restructuring_synergy=dv,
        post_takeover_value=v,
        bidding_cost=c,
        dilution_cap_phi=phi,
        pre_bid_toehold_alpha=alpha,
        equilibrium_tender_price=p_star,
        bidder_net_profit=bidder_profit,
        takeover_succeeds=succeeds,
        target_bid_premium=prem,
        target_bid_premium_pct=prem_pct,
        optimal_min_dilution=opt_phi,
        breakeven_toehold_zero_phi=be_toehold,
        missed_synergy_loss=missed,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_grossman_hart_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "TARGET",
    dilution_pct: float = 0.14,
    toehold_alpha: float = 0.05,
    synergy_pct: float = 0.30,
) -> GrossmanHartTakeoverReceipt:
    """Calibrate Grossman-Hart takeover feasibility from equity price history."""
    p_arr = np.asarray(prices, dtype=float)
    q = float(p_arr[-1]) if len(p_arr) >= 5 else 100.0
    dv = q * max(0.0, float(synergy_pct))
    phi = q * max(0.0, float(dilution_pct))
    cost = q * 0.10

    return calculate_grossman_hart_takeover(
        status_quo_value=q,
        restructuring_synergy=dv,
        bidding_cost=cost,
        dilution_cap_phi=phi,
        pre_bid_toehold_alpha=toehold_alpha,
        ticker=ticker,
    )


def persist_grossman_hart_record(
    receipt: GrossmanHartTakeoverReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Grossman-Hart takeover equilibrium record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS grossman_hart_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        status_quo_value REAL NOT NULL,
        restructuring_synergy REAL NOT NULL,
        post_takeover_value REAL NOT NULL,
        bidding_cost REAL NOT NULL,
        dilution_cap_phi REAL NOT NULL,
        pre_bid_toehold_alpha REAL NOT NULL,
        equilibrium_tender_price REAL NOT NULL,
        bidder_net_profit REAL NOT NULL,
        takeover_succeeds INTEGER NOT NULL,
        target_bid_premium REAL NOT NULL,
        target_bid_premium_pct REAL NOT NULL,
        optimal_min_dilution REAL NOT NULL,
        breakeven_toehold_zero_phi REAL NOT NULL,
        missed_synergy_loss REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_gh_takeover_ticker_time ON grossman_hart_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO grossman_hart_audits (
        ticker, status_quo_value, restructuring_synergy, post_takeover_value,
        bidding_cost, dilution_cap_phi, pre_bid_toehold_alpha, equilibrium_tender_price,
        bidder_net_profit, takeover_succeeds, target_bid_premium, target_bid_premium_pct,
        optimal_min_dilution, breakeven_toehold_zero_phi, missed_synergy_loss,
        regime, hmac_signature, timestamp
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
                    receipt.status_quo_value,
                    receipt.restructuring_synergy,
                    receipt.post_takeover_value,
                    receipt.bidding_cost,
                    receipt.dilution_cap_phi,
                    receipt.pre_bid_toehold_alpha,
                    receipt.equilibrium_tender_price,
                    receipt.bidder_net_profit,
                    1 if receipt.takeover_succeeds else 0,
                    receipt.target_bid_premium,
                    receipt.target_bid_premium_pct,
                    receipt.optimal_min_dilution,
                    receipt.breakeven_toehold_zero_phi,
                    receipt.missed_synergy_loss,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

