"""
Albert S. Kyle & Anna A. Obizhaeva (2016)
Market Microstructure Invariance & Invariant Scaling Laws Engine.

References:
    Kyle, A. S., & Obizhaeva, A. A. (2016).
    Market Microstructure Invariance: Theory and Empirical Tests.
    Econometrica, 84(4), 1345–1404.

    Kyle, A. S., & Obizhaeva, A. A. (2018).
    Dimensional Analysis and Market Microstructure Invariance.
    Financial Analysts Journal, 74(1), 58–70.
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
_DEFAULT_HMAC_SECRET = b"nextomni-kyle-obizhaeva-invariance-2016-secret"

# Calibrated invariance constants (Kyle & Obizhaeva 2016)
C_N_DEFAULT = 2.50    # Trade frequency constant: N* = C_N * (V * sigma^2)^(2/3)
C_S_DEFAULT = 1.50    # Spread constant: S* = C_S * (sigma / V)^(1/3) * 10,000 bps
C_D_DEFAULT = 100.0   # Market depth constant: D* = C_D * V^(2/3) / sigma^(4/3)


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: KyleObizhaevaReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.price:.2f}:{receipt.daily_volume_dollar:.2f}:"
        f"{receipt.daily_volatility:.6f}:{receipt.business_risk_w:.2f}:"
        f"{receipt.invariant_trades_n:.1f}:{receipt.invariant_spread_bps:.2f}:"
        f"{receipt.friction_ratio:.4f}:{receipt.slicing_intensity:.4f}:"
        f"{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class KyleObizhaevaReceipt:
    ticker: str
    price: float
    daily_volume_dollar: float
    daily_volatility: float
    business_risk_w: float
    invariant_trades_n: float
    invariant_trade_size_dollar: float
    invariant_spread_bps: float
    invariant_depth_dollar: float
    actual_spread_bps: float
    friction_ratio: float
    actual_trades: int
    slicing_intensity: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> KyleObizhaevaReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0.00:0.00:0.000000:0.00:0.0:0.00:1.0000:1.0000:ERROR_{error_msg}:{ts}"
    return KyleObizhaevaReceipt(
        ticker=ticker,
        price=0.0,
        daily_volume_dollar=0.0,
        daily_volatility=0.0,
        business_risk_w=0.0,
        invariant_trades_n=0.0,
        invariant_trade_size_dollar=0.0,
        invariant_spread_bps=0.0,
        invariant_depth_dollar=0.0,
        actual_spread_bps=0.0,
        friction_ratio=1.0,
        actual_trades=0,
        slicing_intensity=1.0,
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_kyle_obizhaeva_invariance(
    price: float,
    daily_volume_dollar: float,
    daily_volatility: float,
    actual_spread_bps: Optional[float] = None,
    actual_trades: Optional[int] = None,
    c_n: float = C_N_DEFAULT,
    c_s: float = C_S_DEFAULT,
    c_d: float = C_D_DEFAULT,
    ticker: str = "SPY",
    derivation_mode: str = "PARAMETRIC_CALIBRATION",
) -> KyleObizhaevaReceipt:
    """
    Calculate Kyle & Obizhaeva (2016) Market Microstructure Invariance scaling laws,
    business risk velocity, invariant spread benchmark, and friction ratio.

    Args:
        price: Current underlying security price ($)
        daily_volume_dollar: Daily dollar trading volume ($) = price * shares
        daily_volatility: Daily returns volatility (sigma, e.g. 0.015 for 1.5%)
        actual_spread_bps: Optional observed bid-ask spread in bps
        actual_trades: Optional observed daily trade executions
        c_n: Frequency scaling constant
        c_s: Spread scaling constant
        c_d: Market depth scaling constant
        ticker: Symbol string
        derivation_mode: Analytical calibration mode

    Returns:
        KyleObizhaevaReceipt with invariant predictions, friction ratio, and HMAC.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    p = float(price)
    v = float(daily_volume_dollar)
    sig = float(daily_volatility)

    if p <= 0.0 or v <= 100.0 or sig <= 0.0001:
        return _empty_receipt(clean_ticker, "INVALID_MARKET_DATA")

    # Safe bounds
    v = max(1000.0, v)
    sig = max(0.0005, min(0.50, sig))

    # 1. Business risk velocity W = V * sigma^2 (daily variance flow)
    w = v * (sig ** 2)

    # 2. Invariant trade frequency N* = C_N * W^(2/3)
    n_star = c_n * (w ** (2.0 / 3.0))
    n_star = max(1.0, n_star)
    r_n = float(round(n_star, 1))

    # 3. Invariant trade size Q* = V / N*
    q_star = v / r_n
    r_q = float(round(q_star, 2))

    # 4. Invariant bid-ask spread S* = C_S * (sigma / V)^(1/3) * 10,000 (in bps)
    spread_ratio = sig / v
    s_star_bps = c_s * (spread_ratio ** (1.0 / 3.0)) * 10000.0
    s_star_bps = max(0.01, s_star_bps)

    # 5. Invariant market depth D* = C_D * V^(2/3) / sigma^(4/3)
    d_star = c_d * (v ** (2.0 / 3.0)) / (sig ** (4.0 / 3.0))

    # 6. Empirical comparison: Friction ratio S_actual / S*
    if actual_spread_bps is not None and actual_spread_bps > 0:
        act_s = float(actual_spread_bps)
        friction = act_s / s_star_bps
    else:
        act_s = s_star_bps
        friction = 1.0

    # 7. Slicing intensity: N_actual / N*
    if actual_trades is not None and actual_trades > 0:
        act_n = int(actual_trades)
        slicing = float(act_n) / r_n
    else:
        act_n = int(round(r_n))
        slicing = 1.0

    # 8. Regime classification
    if friction > 1.40:
        regime = "HIGH_FRICTION_SPREAD_WIDENING"
    elif slicing > 2.50:
        regime = "INTENSE_ALGORITHMIC_SLICING"
    elif friction < 0.75:
        regime = "COMPETITIVE_TIGHT_LIQUIDITY"
    else:
        regime = "INVARIANT_EQUILIBRIUM"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_p = float(round(p, 2))
    r_v = float(round(v, 2))
    r_sig = float(round(sig, 6))
    r_w = float(round(w, 2))
    r_s = float(round(s_star_bps, 2))
    r_d = float(round(d_star, 2))
    r_acts = float(round(act_s, 2))
    r_fric = float(round(friction, 4))
    r_slice = float(round(slicing, 4))

    raw_payload = (
        f"{clean_ticker}:{r_p:.2f}:{r_v:.2f}:{r_sig:.6f}:{r_w:.2f}:"
        f"{r_n:.1f}:{r_s:.2f}:{r_fric:.4f}:{r_slice:.4f}:{regime}:{ts}"
    )
    sig_hmac = _compute_hmac(raw_payload)

    return KyleObizhaevaReceipt(
        ticker=clean_ticker,
        price=r_p,
        daily_volume_dollar=r_v,
        daily_volatility=r_sig,
        business_risk_w=r_w,
        invariant_trades_n=r_n,
        invariant_trade_size_dollar=r_q,
        invariant_spread_bps=r_s,
        invariant_depth_dollar=r_d,
        actual_spread_bps=r_acts,
        friction_ratio=r_fric,
        actual_trades=act_n,
        slicing_intensity=r_slice,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig_hmac,
        timestamp=ts,
    )


def evaluate_kyle_obizhaeva_from_market_data(
    prices: Sequence[float],
    volumes: Sequence[float],
    ticker: str = "SPY",
    actual_spread_bps: Optional[float] = None,
) -> KyleObizhaevaReceipt:
    """
    Evaluate Kyle-Obizhaeva invariance from price and volume historical series.
    """
    p_arr = np.asarray(prices, dtype=np.float64)
    v_arr = np.asarray(volumes, dtype=np.float64)

    if len(p_arr) < 20 or len(v_arr) < 20:
        return _empty_receipt(ticker, "INSUFFICIENT_HISTORY")

    curr_price = float(p_arr[-1])
    # Average daily dollar volume over trailing 20 sessions
    trailing_adv_shares = float(np.mean(v_arr[-20:]))
    daily_volume_dollar = curr_price * trailing_adv_shares

    # Daily percentage returns volatility over trailing sessions
    log_returns = np.diff(np.log(p_arr[-60:])) if len(p_arr) >= 60 else np.diff(np.log(p_arr))
    daily_vol = float(np.std(log_returns, ddof=1)) if len(log_returns) > 1 else 0.015

    return calculate_kyle_obizhaeva_invariance(
        price=curr_price,
        daily_volume_dollar=daily_volume_dollar,
        daily_volatility=daily_vol,
        actual_spread_bps=actual_spread_bps,
        ticker=ticker,
        derivation_mode="EMPIRICAL_MARKET_CALIBRATION",
    )


def persist_as_record(
    receipt: KyleObizhaevaReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Kyle-Obizhaeva Invariance receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS kyle_obizhaeva_invariance_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    price REAL NOT NULL,
                    daily_volume_dollar REAL NOT NULL,
                    daily_volatility REAL NOT NULL,
                    business_risk_w REAL NOT NULL,
                    invariant_trades_n REAL NOT NULL,
                    invariant_trade_size_dollar REAL NOT NULL,
                    invariant_spread_bps REAL NOT NULL,
                    invariant_depth_dollar REAL NOT NULL,
                    friction_ratio REAL NOT NULL,
                    slicing_intensity REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_kyle_obizhaeva_ticker_created
                ON kyle_obizhaeva_invariance_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO kyle_obizhaeva_invariance_records (
                    ticker, price, daily_volume_dollar, daily_volatility,
                    business_risk_w, invariant_trades_n, invariant_trade_size_dollar,
                    invariant_spread_bps, invariant_depth_dollar, friction_ratio,
                    slicing_intensity, regime, derivation_mode, hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.price,
                    receipt.daily_volume_dollar,
                    receipt.daily_volatility,
                    receipt.business_risk_w,
                    receipt.invariant_trades_n,
                    receipt.invariant_trade_size_dollar,
                    receipt.invariant_spread_bps,
                    receipt.invariant_depth_dollar,
                    receipt.friction_ratio,
                    receipt.slicing_intensity,
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
