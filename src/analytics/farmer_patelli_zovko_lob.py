"""
J. Doyne Farmer, Paolo Patelli & Ilija I. Zovko (2005)
Zero-Intelligence Order Book & Power-Law Liquidity Profile Engine.

References:
    Farmer, J. D., Patelli, P., & Zovko, I. I. (2005).
    The Predictive Power of Zero Intelligence in Financial Markets.
    Proceedings of the National Academy of Sciences (PNAS), 102(6), 2254–2259.

    Mike, S., & Farmer, J. D. (2008).
    An empirical behavioral model of liquidity and volatility.
    Journal of Economic Dynamics and Control, 32(1), 200–234.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,10}$")
_DEFAULT_HMAC_SECRET = b"nextomni-farmer-patelli-zovko-2005-secret"

# Default empirical level-depth ladder (multipliers of best-level depth)
DEFAULT_DEPTH_MULTIPLIERS = [1.0, 1.45, 1.85, 2.20, 2.50]


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: FarmerPatelliZovkoReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.mid_price:.2f}:{receipt.levels_analyzed}:"
        f"{receipt.depth_exponent_delta:.4f}:{receipt.placement_exponent_mu:.4f}:"
        f"{receipt.zi_equilibrium_spread_bps:.2f}:{receipt.zi_spread_ratio:.4f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class FarmerPatelliZovkoReceipt:
    ticker: str
    mid_price: float
    levels_analyzed: int
    depth_exponent_delta: float
    placement_exponent_mu: float
    cumulative_depth_total: float
    touch_depth_ratio: float
    zi_equilibrium_spread_bps: float
    actual_spread_bps: float
    zi_spread_ratio: float
    zi_implied_volatility: float
    book_elasticity: float
    price_distances: List[float]
    cumulative_depths: List[float]
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> FarmerPatelliZovkoReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0.00:0:0.0000:0.0000:0.00:1.0000:ERROR_{error_msg}:{ts}"
    return FarmerPatelliZovkoReceipt(
        ticker=ticker,
        mid_price=0.0,
        levels_analyzed=0,
        depth_exponent_delta=0.60,
        placement_exponent_mu=1.40,
        cumulative_depth_total=0.0,
        touch_depth_ratio=1.0,
        zi_equilibrium_spread_bps=0.0,
        actual_spread_bps=0.0,
        zi_spread_ratio=1.0,
        zi_implied_volatility=0.0,
        book_elasticity=1.0,
        price_distances=[],
        cumulative_depths=[],
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_fpz_liquidity_profile(
    mid_price: float,
    bid_depths: Sequence[float],
    ask_depths: Sequence[float],
    tick_size: float = 0.01,
    actual_spread: Optional[float] = None,
    limit_arrival_rate_alpha: float = 250.0,
    market_order_rate_mu: float = 120.0,
    ticker: str = "SPY",
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> FarmerPatelliZovkoReceipt:
    """
    Calculate Farmer, Patelli & Zovko (2005) power-law depth profile exponent delta,
    placement density exponent mu, and zero-intelligence market clearing spread.

    Args:
        mid_price: Current mid-quote price ($)
        bid_depths: Sequence of volume sizes across bid levels 1..K
        ask_depths: Sequence of volume sizes across ask levels 1..K
        tick_size: Minimum price variation (e.g. 0.01)
        actual_spread: Observed bid-ask spread ($)
        limit_arrival_rate_alpha: ZI limit order intensity (shares/$/sec)
        market_order_rate_mu: ZI market order execution intensity (shares/sec)
        ticker: Symbol string
        derivation_mode: Analytical calibration tag

    Returns:
        FarmerPatelliZovkoReceipt with power-law exponents, ZI spread, and HMAC signature.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    p_mid = float(mid_price)
    b_arr = np.asarray(bid_depths, dtype=np.float64)
    a_arr = np.asarray(ask_depths, dtype=np.float64)

    k = min(len(b_arr), len(a_arr))
    if p_mid <= 0.0 or k < 2:
        return _empty_receipt(clean_ticker, "INSUFFICIENT_LEVELS")

    # Symmetric level volumes: average of bid and ask queue at each level
    q_levels = 0.5 * (b_arr[:k] + a_arr[:k])
    q_levels = np.maximum(1.0, q_levels)

    # Distances from touch: p_j = (j - 0.5) * tick_size
    tick = max(0.001, float(tick_size))
    p_distances = np.array([(j + 0.5) * tick for j in range(k)], dtype=np.float64)

    # Cumulative depth profile: N(p_k) = sum_{j=1}^k Q_j
    cum_depth = np.cumsum(q_levels)
    total_depth = float(cum_depth[-1])
    touch_ratio = float(round(q_levels[0] / total_depth, 4))

    # 1. Fit power-law cumulative depth profile: N(p) = A * p^delta
    # log(N(p)) = log(A) + delta * log(p)
    log_p = np.log(p_distances)
    log_n = np.log(cum_depth)

    # Linear regression on log-log
    poly_fit = np.polyfit(log_p, log_n, deg=1)
    delta = float(np.clip(poly_fit[0], 0.10, 1.80))

    # 2. Placement density exponent mu: f(Delta) ~ Delta^(-mu)
    # Since dN/dp ~ p^(delta - 1), and placement density mirrors liquidity replenishment,
    # mu ~= 2 - delta or bounded in [0.8, 2.2]
    mu = float(np.clip(2.0 - delta, 0.80, 2.20))

    # 3. Zero-Intelligence Equilibrium Spread (Farmer, Patelli & Zovko 2005)
    # S_bar = (mu_mo / alpha)^(1 / (1 + delta))
    alpha = max(1.0, float(limit_arrival_rate_alpha))
    mu_mo = max(1.0, float(market_order_rate_mu))
    flow_ratio = mu_mo / alpha
    s_zi_dollar = float(flow_ratio ** (1.0 / (1.0 + delta)))
    # Express in basis points
    s_zi_bps = float(round((s_zi_dollar / p_mid) * 10000.0, 2))
    s_zi_bps = max(0.10, s_zi_bps)

    # Actual observed spread
    if actual_spread is not None and actual_spread > 0:
        act_s_dlr = float(actual_spread)
        act_s_bps = float(round((act_s_dlr / p_mid) * 10000.0, 2))
    else:
        act_s_bps = s_zi_bps

    zi_ratio = float(round(act_s_bps / s_zi_bps, 4))

    # 4. Zero-Intelligence Implied Volatility: sigma_zi = sqrt(mu_mo) / (alpha * P)
    zi_vol = float(round(math.sqrt(mu_mo) / (alpha * p_mid), 6))

    # 5. Book Elasticity: slope of depth response to price distance
    book_elasticity = float(round(delta, 4))

    # 6. Regime Classification
    if delta < 0.40:
        regime = "TOUCH_HEAVY_CONCENTRATED_BOOK"
    elif delta > 0.85:
        regime = "DEEP_ACCUMULATED_SUPPORT_BOOK"
    elif zi_ratio > 1.35:
        regime = "HIGH_SPREAD_INDUCED_FRICTION"
    else:
        regime = "BALANCED_POWER_LAW_BOOK"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_mid = float(round(p_mid, 2))
    r_delta = float(round(delta, 4))
    r_mu = float(round(mu, 4))
    r_tot_depth = float(round(total_depth, 1))

    raw_payload = (
        f"{clean_ticker}:{r_mid:.2f}:{k}:{r_delta:.4f}:{r_mu:.4f}:"
        f"{s_zi_bps:.2f}:{zi_ratio:.4f}:{regime}:{ts}"
    )
    sig_hmac = _compute_hmac(raw_payload)

    dist_list = [float(round(p_distances[i], 4)) for i in range(k)]
    cum_list = [float(round(cum_depth[i], 1)) for i in range(k)]

    return FarmerPatelliZovkoReceipt(
        ticker=clean_ticker,
        mid_price=r_mid,
        levels_analyzed=k,
        depth_exponent_delta=r_delta,
        placement_exponent_mu=r_mu,
        cumulative_depth_total=r_tot_depth,
        touch_depth_ratio=touch_ratio,
        zi_equilibrium_spread_bps=s_zi_bps,
        actual_spread_bps=act_s_bps,
        zi_spread_ratio=zi_ratio,
        zi_implied_volatility=zi_vol,
        book_elasticity=book_elasticity,
        price_distances=dist_list,
        cumulative_depths=cum_list,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig_hmac,
        timestamp=ts,
    )


def evaluate_fpz_from_order_book(
    mid_price: float,
    bid_depth_touch: float = 250.0,
    ask_depth_touch: float = 200.0,
    actual_spread: Optional[float] = None,
    ticker: str = "SPY",
    num_levels: int = 5,
) -> FarmerPatelliZovkoReceipt:
    """
    Evaluate Farmer-Patelli-Zovko LOB profile using inside quotes and empirical depth scaling.
    """
    n_levels = max(2, min(10, int(num_levels)))
    # Generate calibrated multi-level depth ladder
    mults = DEFAULT_DEPTH_MULTIPLIERS[:n_levels]
    if len(mults) < n_levels:
        mults += [mults[-1] * (1.15 ** i) for i in range(1, n_levels - len(mults) + 1)]

    bids = [float(bid_depth_touch * m) for m in mults]
    asks = [float(ask_depth_touch * m) for m in mults]

    return calculate_fpz_liquidity_profile(
        mid_price=mid_price,
        bid_depths=bids,
        ask_depths=asks,
        actual_spread=actual_spread,
        ticker=ticker,
        derivation_mode="CALIBRATED_LOB_LEVEL_EVALUATION",
    )


def persist_as_record(
    receipt: FarmerPatelliZovkoReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Farmer-Patelli-Zovko LOB receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS farmer_patelli_zovko_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    mid_price REAL NOT NULL,
                    levels_analyzed INTEGER NOT NULL,
                    depth_exponent_delta REAL NOT NULL,
                    placement_exponent_mu REAL NOT NULL,
                    cumulative_depth_total REAL NOT NULL,
                    touch_depth_ratio REAL NOT NULL,
                    zi_equilibrium_spread_bps REAL NOT NULL,
                    actual_spread_bps REAL NOT NULL,
                    zi_spread_ratio REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_fpz_ticker_created
                ON farmer_patelli_zovko_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO farmer_patelli_zovko_records (
                    ticker, mid_price, levels_analyzed, depth_exponent_delta,
                    placement_exponent_mu, cumulative_depth_total, touch_depth_ratio,
                    zi_equilibrium_spread_bps, actual_spread_bps, zi_spread_ratio,
                    regime, derivation_mode, hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.mid_price,
                    receipt.levels_analyzed,
                    receipt.depth_exponent_delta,
                    receipt.placement_exponent_mu,
                    receipt.cumulative_depth_total,
                    receipt.touch_depth_ratio,
                    receipt.zi_equilibrium_spread_bps,
                    receipt.actual_spread_bps,
                    receipt.zi_spread_ratio,
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

