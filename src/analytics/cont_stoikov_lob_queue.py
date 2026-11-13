"""
Rama Cont, Sasha Stoikov, and Rishi Talreja (2010) Markovian Limit Order Book
Queue Dynamics & First-Hitting-Time Engine.

References:
    Cont, R., Stoikov, S., & Talreja, R. (2010).
    A Stochastic Model for Order Book Dynamics.
    Operations Research, 58(3), 549-563.

    Cont, R., & de Larrard, A. (2013).
    Price Dynamics in a Markovian Limit Order Book.
    SIAM Journal on Financial Mathematics, 4(1), 1-25.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import math
import os
import re
import sqlite3
from typing import Any, Dict, Optional, Tuple

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,10}$")
_DEFAULT_HMAC_SECRET = b"nextomni-cont-stoikov-lob-2010-secret"


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: ContStoikovLOBReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.best_bid:.4f}:{receipt.best_ask:.4f}:"
        f"{receipt.bid_depth:.2f}:{receipt.ask_depth:.2f}:"
        f"{receipt.prob_up_tick:.4f}:{receipt.prob_down_tick:.4f}:"
        f"{receipt.expected_duration_sec:.3f}:{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class ContStoikovLOBReceipt:
    ticker: str
    best_bid: float
    best_ask: float
    mid_price: float
    bid_depth: float
    ask_depth: float
    lambda_limit: float
    mu_market: float
    theta_cancel: float
    prob_up_tick: float
    prob_down_tick: float
    expected_duration_sec: float
    bid_fill_prob: float
    ask_fill_prob: float
    order_priority_k: int
    order_imbalance_ratio: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> ContStoikovLOBReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0.0000:0.0000:0.00:0.00:0.5000:0.5000:0.000:ERROR_{error_msg}:{ts}"
    return ContStoikovLOBReceipt(
        ticker=ticker,
        best_bid=0.0,
        best_ask=0.0,
        mid_price=0.0,
        bid_depth=0.0,
        ask_depth=0.0,
        lambda_limit=0.0,
        mu_market=0.0,
        theta_cancel=0.0,
        prob_up_tick=0.5,
        prob_down_tick=0.5,
        expected_duration_sec=0.0,
        bid_fill_prob=0.0,
        ask_fill_prob=0.0,
        order_priority_k=1,
        order_imbalance_ratio=0.0,
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_cont_stoikov_lob(
    best_bid: float,
    best_ask: float,
    bid_depth: float,
    ask_depth: float,
    lambda_limit: float = 5.0,
    mu_market: float = 6.0,
    theta_cancel: float = 3.0,
    order_priority_k: int = 10,
    ticker: str = "SPY",
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> ContStoikovLOBReceipt:
    """
    Calculate Markovian LOB queue dynamics, first hitting time probabilities,
    and queue priority execution probabilities under Cont, Stoikov & Talreja (2010).

    Args:
        best_bid: Current best bid quote ($)
        best_ask: Current best ask quote ($)
        bid_depth: Best bid queue volume/shares (x >= 0)
        ask_depth: Best ask queue volume/shares (y >= 0)
        lambda_limit: Limit order Poisson arrival intensity (> 0)
        mu_market: Market order Poisson execution intensity (> 0)
        theta_cancel: Cancellation intensity (> 0)
        order_priority_k: Queue position for limit order fill probability (>= 1)
        ticker: Symbol string
        derivation_mode: Calibration tag

    Returns:
        ContStoikovLOBReceipt with mathematical results and HMAC signature.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    if best_bid <= 0.0 or best_ask <= 0.0 or best_bid > best_ask:
        return _empty_receipt(clean_ticker, "INVALID_QUOTES")

    x = max(0.0, float(bid_depth))
    y = max(0.0, float(ask_depth))
    lam = max(0.001, float(lambda_limit))
    mu = max(0.001, float(mu_market))
    theta = max(0.001, float(theta_cancel))
    k = max(1, int(order_priority_k))

    mid = 0.5 * (best_bid + best_ask)

    # Order Imbalance Ratio: (x - y) / (x + y)
    total_depth = x + y
    if total_depth <= 1e-8:
        oir = 0.0
        prob_up = 0.5
        prob_down = 0.5
        expected_duration = 0.0
    elif y <= 1e-8:
        # Ask queue is empty -> immediate upward move
        oir = 1.0
        prob_up = 1.0
        prob_down = 0.0
        expected_duration = 0.0
    elif x <= 1e-8:
        # Bid queue is empty -> immediate downward move
        oir = -1.0
        prob_up = 0.0
        prob_down = 1.0
        expected_duration = 0.0
    else:
        oir = (x - y) / total_depth

        # First hitting time probability: 2D queue diffusion limit / harmonic measure
        # Cont & de Larrard (2013) Proposition 1 / Cont, Stoikov & Talreja (2010):
        # Probability that ask queue hits 0 before bid queue hits 0:
        # P(tau_a < tau_b) = (2 / pi) * arctan(x / y)
        angle = math.atan2(x, y)  # atan2(x, y) = arctan(x / y) in [0, pi/2]
        prob_up = (2.0 / math.pi) * angle

        prob_up = min(1.0, max(0.0, prob_up))
        prob_down = 1.0 - prob_up

        # Expected duration until first quote depletion E[tau | x, y]
        # Approximation via bottleneck queue depletion
        effective_depletion_rate = max(0.01, mu + 0.5 * theta)
        expected_duration = (x * y) / (total_depth * effective_depletion_rate)

    # Queue priority fill probability: (mu / (mu + theta))^k
    # Probability of being executed before entire order cancellation
    survival_ratio = mu / (mu + theta)
    bid_fill_prob = min(1.0, max(0.0, float(survival_ratio ** min(k, max(1, int(x))))))
    ask_fill_prob = min(1.0, max(0.0, float(survival_ratio ** min(k, max(1, int(y))))))

    # Market regime classification
    if oir > 0.25:
        regime = "STRONG_BUY_PRESSURE"
    elif oir < -0.25:
        regime = "STRONG_SELL_PRESSURE"
    else:
        regime = "BALANCED_ORDER_BOOK"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_bid = float(round(best_bid, 4))
    r_ask = float(round(best_ask, 4))
    r_mid = float(round(mid, 4))
    r_x = float(round(x, 2))
    r_y = float(round(y, 2))
    r_pup = float(round(prob_up, 4))
    r_pdown = float(round(prob_down, 4))
    r_dur = float(round(expected_duration, 3))

    raw_payload = (
        f"{clean_ticker}:{r_bid:.4f}:{r_ask:.4f}:{r_x:.2f}:{r_y:.2f}:"
        f"{r_pup:.4f}:{r_pdown:.4f}:{r_dur:.3f}:{regime}:{ts}"
    )
    sig = _compute_hmac(raw_payload)

    return ContStoikovLOBReceipt(
        ticker=clean_ticker,
        best_bid=r_bid,
        best_ask=r_ask,
        mid_price=r_mid,
        bid_depth=r_x,
        ask_depth=r_y,
        lambda_limit=float(round(lam, 4)),
        mu_market=float(round(mu, 4)),
        theta_cancel=float(round(theta, 4)),
        prob_up_tick=r_pup,
        prob_down_tick=r_pdown,
        expected_duration_sec=r_dur,
        bid_fill_prob=float(round(bid_fill_prob, 4)),
        ask_fill_prob=float(round(ask_fill_prob, 4)),
        order_priority_k=k,
        order_imbalance_ratio=float(round(oir, 4)),
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig,
        timestamp=ts,
    )


def evaluate_cont_stoikov_from_quotes(
    best_bid: float,
    best_ask: float,
    bid_depth: float,
    ask_depth: float,
    ticker: str = "SPY",
) -> ContStoikovLOBReceipt:
    """Empirical calibration wrapper using default US equity high-frequency rates."""
    return calculate_cont_stoikov_lob(
        best_bid=best_bid,
        best_ask=best_ask,
        bid_depth=bid_depth,
        ask_depth=ask_depth,
        lambda_limit=5.0,
        mu_market=6.0,
        theta_cancel=3.0,
        order_priority_k=10,
        ticker=ticker,
        derivation_mode="EMPIRICAL_QUOTE_EVALUATION",
    )


def persist_as_record(
    receipt: ContStoikovLOBReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Cont-Stoikov LOB receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cont_stoikov_lob_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    best_bid REAL NOT NULL,
                    best_ask REAL NOT NULL,
                    bid_depth REAL NOT NULL,
                    ask_depth REAL NOT NULL,
                    prob_up_tick REAL NOT NULL,
                    prob_down_tick REAL NOT NULL,
                    expected_duration_sec REAL NOT NULL,
                    bid_fill_prob REAL NOT NULL,
                    ask_fill_prob REAL NOT NULL,
                    order_imbalance_ratio REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_cont_stoikov_ticker_created
                ON cont_stoikov_lob_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO cont_stoikov_lob_records (
                    ticker, best_bid, best_ask, bid_depth, ask_depth,
                    prob_up_tick, prob_down_tick, expected_duration_sec,
                    bid_fill_prob, ask_fill_prob, order_imbalance_ratio,
                    regime, derivation_mode, hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.best_bid,
                    receipt.best_ask,
                    receipt.bid_depth,
                    receipt.ask_depth,
                    receipt.prob_up_tick,
                    receipt.prob_down_tick,
                    receipt.expected_duration_sec,
                    receipt.bid_fill_prob,
                    receipt.ask_fill_prob,
                    receipt.order_imbalance_ratio,
                    receipt.regime,
                    receipt.derivation_mode,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()
