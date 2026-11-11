"""Eugene F. Fama & Robert R. Bliss (1987) Forward-Spot Term Spread & Bond Excess Return Engine.

Theoretical Foundation:
    Fama, Eugene F., and Bliss, Robert R. (1987).
    "The Information in Long-Maturity Forward Rates".
    American Economic Review, 77(4), pp. 680-692.

Key Concepts:
    - Forward-Spot Yield Decomposition:
      Given log zero-coupon bond prices p_t^(k) = -k * y_t^(k), the implied 1-year
      forward rate n-1 years ahead is f_t^(n) = n * y_t^(n) - (n - 1) * y_t^(n-1),
      defining the forward-spot spread s_t^(n) = f_t^(n) - y_t^(1).
    - Exact Accounting & OLS Complementarity Identity:
      The 1-year excess log return rx_{t+1}^(n) = p_{t+1}^(n-1) - p_t^(n) - y_t^(1)
      and scaled future yield change (n - 1) * (y_{t+1}^(n-1) - y_t^(n-1)) satisfy:
      rx_{t+1}^(n) + (n - 1) * (y_{t+1}^(n-1) - y_t^(n-1)) == f_t^(n) - y_t^(1).
      Consequently, OLS slopes on s_t^(n) sum identically to unity (beta + c == 1.0),
      testing Pure Expectations Hypothesis (beta = 0, c = 1) vs time-varying term premia.
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
HMAC_SECRET = b"fama-bliss-forward-spread-v172-secret"


@dataclass(frozen=True)
class FamaBlissReceipt:
    """Immutable quantitative audit receipt for Fama & Bliss (1987) Forward-Spot Spread."""

    ticker: str
    maturity_n: int
    spot_y1: float
    prev_maturity_y_nm1: float
    long_yn: float
    next_y_nm1: float
    log_price_p1: float
    log_price_p_nm1: float
    log_price_pn: float
    implied_forward_fn: float
    forward_spot_spread: float
    forward_spread_bps: float
    holding_return_hn: float
    excess_bond_return_rx: float
    scaled_yield_change: float
    identity_residual: float
    eh_breakeven_yield_shift: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class FamaBlissRegressionResult:
    """OLS Fama-Bliss complementarity test estimates (beta + c == 1.0)."""

    maturity_n: int
    n_obs: int
    alpha_rx: float
    beta_rx: float
    r2_rx: float
    alpha_yield_chg: float
    c_yield_chg: float
    r2_yield_chg: float
    slope_sum_check: float
    intercept_sum_check: float


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


def _interpolate_prev_yield(spot_y1: float, long_yn: float, maturity_n: int) -> float:
    """Linearly interpolate (n-1)-year spot yield when omitted."""
    if maturity_n == 2:
        return spot_y1
    weight = (maturity_n - 2) / float(maturity_n - 1)
    return spot_y1 + weight * (long_yn - spot_y1)


def classify_fama_bliss_regime(forward_spot_spread: float) -> str:
    """Classify forward-spot term premium & carry regime."""
    if forward_spot_spread <= -0.0025:
        return "INVERTED_CARRY_COMPRESSION"
    if forward_spot_spread < 0.0040:
        return "EXPECTATIONS_NEUTRAL_CURVE"
    if forward_spot_spread < 0.0150:
        return "BALANCED_FORWARD_SLOPE"
    return "EXTENDED_TERM_PREMIUM_CARRY"


def _build_hmac_payload(
    ticker: str,
    maturity_n: int,
    spot_y1: float,
    long_yn: float,
    implied_forward_fn: float,
    forward_spot_spread: float,
    excess_bond_return_rx: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{maturity_n}:{spot_y1:.6f}:{long_yn:.6f}:"
        f"{implied_forward_fn:.6f}:{forward_spot_spread:.6f}:"
        f"{excess_bond_return_rx:.6f}:{regime}:{timestamp}"
    )


def calculate_fama_bliss_forward_spread(
    spot_y1: float,
    long_yn: float,
    maturity_n: int = 2,
    prev_maturity_y_nm1: float | None = None,
    next_y_nm1: float | None = None,
    ticker: str = "TLT",
) -> FamaBlissReceipt:
    """Compute Fama & Bliss (1987) forward rate, spread, carry return, and accounting identity."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    n_int = int(maturity_n)
    if not (2 <= n_int <= 30):
        raise ValueError("Maturity n must be an integer in [2, 30] years.")

    y1 = float(spot_y1)
    yn = float(long_yn)
    if not (-0.05 <= y1 <= 0.40) or not (-0.05 <= yn <= 0.40):
        raise ValueError("Spot yields must be in [-0.05, 0.40].")

    if n_int == 2 and prev_maturity_y_nm1 is None:
        y_nm1 = y1
    elif prev_maturity_y_nm1 is None:
        y_nm1 = _interpolate_prev_yield(y1, yn, n_int)
    else:
        y_nm1 = float(prev_maturity_y_nm1)
        if not (-0.05 <= y_nm1 <= 0.40):
            raise ValueError("Prior maturity yield must be in [-0.05, 0.40].")
        if n_int == 2 and abs(y_nm1 - y1) > 1e-12:
            raise ValueError("When maturity n == 2, y_(n-1) must equal spot y1.")

    y_next = y_nm1 if next_y_nm1 is None else float(next_y_nm1)
    if not (-0.05 <= y_next <= 0.40):
        raise ValueError("Next-period spot yield must be in [-0.05, 0.40].")

    p1 = -1.0 * y1
    p_nm1 = -(n_int - 1) * y_nm1
    pn = -float(n_int) * yn

    fwd_fn = p_nm1 - pn
    fwd_spread = fwd_fn - y1
    spread_bps = fwd_spread * 10000.0

    p_next_nm1 = -(n_int - 1) * y_next
    hold_ret = p_next_nm1 - pn
    excess_rx = hold_ret - y1
    scaled_dy = (n_int - 1) * (y_next - y_nm1)
    residual = (excess_rx + scaled_dy) - fwd_spread
    eh_breakeven_dy = fwd_spread / float(n_int - 1)

    regime = classify_fama_bliss_regime(fwd_spread)
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        maturity_n=n_int,
        spot_y1=y1,
        long_yn=yn,
        implied_forward_fn=fwd_fn,
        forward_spot_spread=fwd_spread,
        excess_bond_return_rx=excess_rx,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return FamaBlissReceipt(
        ticker=cleaned_ticker,
        maturity_n=n_int,
        spot_y1=y1,
        prev_maturity_y_nm1=y_nm1,
        long_yn=yn,
        next_y_nm1=y_next,
        log_price_p1=p1,
        log_price_p_nm1=p_nm1,
        log_price_pn=pn,
        implied_forward_fn=fwd_fn,
        forward_spot_spread=fwd_spread,
        forward_spread_bps=spread_bps,
        holding_return_hn=hold_ret,
        excess_bond_return_rx=excess_rx,
        scaled_yield_change=scaled_dy,
        identity_residual=residual,
        eh_breakeven_yield_shift=eh_breakeven_dy,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def estimate_fama_bliss_ols(
    spot_y1_series: List[float] | np.ndarray,
    long_yn_series: List[float] | np.ndarray,
    next_y_nm1_series: List[float] | np.ndarray,
    prev_y_nm1_series: List[float] | np.ndarray | None = None,
    maturity_n: int = 2,
) -> FamaBlissRegressionResult:
    """Run paired OLS Fama-Bliss regressions verifying beta + c == 1.0."""
    n_int = int(maturity_n)
    if not (2 <= n_int <= 30):
        raise ValueError("Maturity n must be in [2, 30].")

    y1 = np.asarray(spot_y1_series, dtype=float)
    yn = np.asarray(long_yn_series, dtype=float)
    y_next = np.asarray(next_y_nm1_series, dtype=float)
    if n_int == 2 and prev_y_nm1_series is None:
        y_nm1 = y1.copy()
    elif prev_y_nm1_series is None:
        w = (n_int - 2) / float(n_int - 1)
        y_nm1 = y1 + w * (yn - y1)
    else:
        y_nm1 = np.asarray(prev_y_nm1_series, dtype=float)

    n_obs = len(y1)
    if n_obs < 5 or not (len(yn) == len(y_next) == len(y_nm1) == n_obs):
        raise ValueError("Yield series must align cleanly with at least 5 observations.")

    fwd = float(n_int) * yn - float(n_int - 1) * y_nm1
    spread = fwd - y1
    rx = -float(n_int - 1) * y_next + float(n_int) * yn - y1
    scaled_dy = float(n_int - 1) * (y_next - y_nm1)

    var_s = float(np.var(spread))
    if var_s <= 1e-14:
        raise ValueError("Forward-spot spread series has insufficient variance for OLS.")

    s_mean = float(np.mean(spread))
    s_dev = spread - s_mean
    denom = float(np.sum(s_dev * s_dev))

    beta_rx = float(np.sum(s_dev * (rx - float(np.mean(rx)))) / denom)
    alpha_rx = float(np.mean(rx) - beta_rx * s_mean)

    c_dy = float(np.sum(s_dev * (scaled_dy - float(np.mean(scaled_dy)))) / denom)
    alpha_dy = float(np.mean(scaled_dy) - c_dy * s_mean)

    tss_rx = float(np.sum((rx - float(np.mean(rx))) ** 2))
    tss_dy = float(np.sum((scaled_dy - float(np.mean(scaled_dy))) ** 2))
    rss_rx = float(np.sum((rx - (alpha_rx + beta_rx * spread)) ** 2))
    rss_dy = float(np.sum((scaled_dy - (alpha_dy + c_dy * spread)) ** 2))

    r2_rx = 1.0 - (rss_rx / tss_rx) if tss_rx > 1e-14 else 0.0
    r2_dy = 1.0 - (rss_dy / tss_dy) if tss_dy > 1e-14 else 0.0

    return FamaBlissRegressionResult(
        maturity_n=n_int,
        n_obs=n_obs,
        alpha_rx=alpha_rx,
        beta_rx=beta_rx,
        r2_rx=r2_rx,
        alpha_yield_chg=alpha_dy,
        c_yield_chg=c_dy,
        r2_yield_chg=r2_dy,
        slope_sum_check=beta_rx + c_dy,
        intercept_sum_check=alpha_rx + alpha_dy,
    )


def evaluate_fama_bliss_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "TLT",
    spot_y1: float = 0.042,
    maturity_n: int = 5,
) -> FamaBlissReceipt:
    """Calibrate forward term slope proxy from trailing return drift and volatility."""
    p_arr = np.asarray(prices, dtype=float)
    y1 = float(np.clip(spot_y1, -0.02, 0.25))
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.05, 0.80))
        term_slope = 0.004 + 0.035 * ann_vol
        yn = float(np.clip(y1 + term_slope, -0.02, 0.25))
    else:
        yn = float(np.clip(y1 + 0.010, -0.02, 0.25))

    return calculate_fama_bliss_forward_spread(
        spot_y1=y1,
        long_yn=yn,
        maturity_n=int(maturity_n),
        ticker=ticker,
    )


def persist_fama_bliss_record(
    receipt: FamaBlissReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Fama & Bliss (1987) forward-spot audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS fama_bliss_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        maturity_n INTEGER NOT NULL,
        spot_y1 REAL NOT NULL,
        long_yn REAL NOT NULL,
        implied_forward_fn REAL NOT NULL,
        forward_spot_spread REAL NOT NULL,
        excess_bond_return_rx REAL NOT NULL,
        scaled_yield_change REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO fama_bliss_audits (
        ticker, maturity_n, spot_y1, long_yn, implied_forward_fn,
        forward_spot_spread, excess_bond_return_rx, scaled_yield_change,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.maturity_n,
                receipt.spot_y1,
                receipt.long_yn,
                receipt.implied_forward_fn,
                receipt.forward_spot_spread,
                receipt.excess_bond_return_rx,
                receipt.scaled_yield_change,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

