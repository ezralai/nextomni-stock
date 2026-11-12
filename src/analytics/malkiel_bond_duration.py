"""Frederick R. Macaulay (1938) & Burton G. Malkiel (1962) Bond Duration & Convexity Engine.

Theoretical Foundation:
    Macaulay, Frederick R. (1938).
    "Some Theoretical Problems Suggested by the Movements of Interest Rates, Bond Yields
    and Stock Prices in the United States Since 1856". NBER.

    Malkiel, Burton G. (1962).
    "Expectations, Bond Prices, and the Term Structure of Interest Rates".
    Quarterly Journal of Economics, 76(2), pp. 197-218.

Key Concepts:
    - Exact Coupon Bond Valuation, Macaulay & Modified Duration:
      For face par F, coupon rate c, compounding frequency m, maturity T (N = m*T periods),
      and yield y: P(y) = sum_{k=1}^N PV_k, Macaulay D_Mac = sum((k/m)*PV_k)/P,
      Modified D_Mod = D_Mac / (1 + y/m), and DV01 = D_Mod * P / 10000.
      Zero-coupon bonds satisfy D_Mac == T identically; par bonds (c == y) satisfy P == F.
    - Analytical Convexity & Malkiel's Price-Yield Asymmetry Theorem:
      Positive discrete convexity C > 0 guarantees that equal symmetric yield shocks
      produce larger capital gains on yield declines than capital losses on yield hikes:
      [P(y - dy) - P(y)] - [P(y) - P(y + dy)] > 0.
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
ALLOWED_FREQUENCIES = (1, 2, 4, 12)
HMAC_SECRET = b"malkiel-bond-duration-v177-secret"


@dataclass(frozen=True)
class MalkielDurationReceipt:
    """Immutable quantitative audit receipt for Macaulay & Malkiel Bond Duration."""

    ticker: str
    par_value: float
    coupon_rate: float
    yield_to_maturity: float
    maturity_years: float
    compounding_freq: int
    n_periods: int
    shock_bps: float
    bond_price: float
    par_status: str
    macaulay_duration: float
    modified_duration: float
    dv01: float
    convexity: float
    repriced_on_yield_cut: float
    repriced_on_yield_hike: float
    taylor_on_yield_cut: float
    taylor_on_yield_hike: float
    exact_gain_on_cut: float
    exact_loss_on_hike: float
    malkiel_convexity_asymmetry: float
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


def _bond_price_and_moments(
    par: float,
    coupon_rate: float,
    ytm: float,
    n_periods: int,
    freq: int,
) -> tuple[float, float, float, float]:
    """Compute exact bond price, Macaulay duration, Modified duration, and convexity."""
    period_rate = ytm / float(freq)
    if 1.0 + period_rate <= 0.05:
        raise ValueError("Period discount factor (1 + ytm/freq) must exceed 0.05.")

    k = np.arange(1, n_periods + 1, dtype=float)
    dfs = (1.0 + period_rate) ** k
    cfs = np.full(n_periods, par * coupon_rate / float(freq), dtype=float)
    cfs[-1] += par
    pvs = cfs / dfs

    price = float(np.sum(pvs))
    t_k = k / float(freq)
    mac_dur = float(np.sum(t_k * pvs) / price)
    mod_dur = mac_dur / (1.0 + period_rate)
    conv = float(
        np.sum((k * (k + 1.0) / (float(freq) ** 2)) * pvs)
        / (price * ((1.0 + period_rate) ** 2))
    )
    return price, mac_dur, mod_dur, conv


def classify_malkiel_regime(
    mac_dur: float,
    mod_dur: float,
    coupon_rate: float,
) -> str:
    """Classify fixed-income duration and convexity exposure regime."""
    if mod_dur >= 10.0:
        return "EXTENDED_DURATION_CONVEX_BOND"
    if coupon_rate >= 0.06 and mac_dur < 5.0:
        return "HIGH_COUPON_SHORT_CARRY"
    if mac_dur < 3.5:
        return "SHORT_DURATION_DEFENSIVE"
    return "CORE_INVESTMENT_DURATION"


def _build_hmac_payload(
    ticker: str,
    par: float,
    coupon: float,
    ytm: float,
    maturity: float,
    freq: int,
    price: float,
    mac_dur: float,
    mod_dur: float,
    convexity: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{par:.2f}:{coupon:.6f}:{ytm:.6f}:{maturity:.4f}:{freq}:"
        f"{price:.6f}:{mac_dur:.6f}:{mod_dur:.6f}:{convexity:.6f}:{regime}:{timestamp}"
    )


def calculate_malkiel_bond_duration(
    coupon_rate: float,
    yield_to_maturity: float,
    maturity_years: float = 10.0,
    compounding_freq: int = 2,
    shock_bps: float = 100.0,
    par_value: float = 100.0,
    ticker: str = "TLT",
) -> MalkielDurationReceipt:
    """Evaluate coupon bond valuation, duration, convexity, and Malkiel shock asymmetry."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    par = float(par_value)
    coupon = float(coupon_rate)
    ytm = float(yield_to_maturity)
    maturity = float(maturity_years)
    freq = int(compounding_freq)
    dy_bps = float(shock_bps)

    if not (0.0 < par <= 1e12):
        raise ValueError("Par value must be in (0.0, 1e12].")
    if not (0.0 <= coupon <= 0.30):
        raise ValueError("Coupon rate must be in [0.0, 0.30].")
    if not (-0.04 <= ytm <= 0.50):
        raise ValueError("Yield to maturity must be in [-0.04, 0.50].")
    if freq not in ALLOWED_FREQUENCIES:
        raise ValueError(f"Compounding frequency must be one of {ALLOWED_FREQUENCIES}.")
    if not (0.0 < maturity <= 50.0):
        raise ValueError("Maturity must be in (0.0, 50.0] years.")

    raw_periods = maturity * float(freq)
    n_periods = int(round(raw_periods))
    if n_periods < 1 or abs(raw_periods - n_periods) > 1e-6:
        raise ValueError("Maturity * compounding_freq must yield an integer number of periods.")
    if not (1.0 <= dy_bps <= 1000.0):
        raise ValueError("Parallel yield shock must be in [1.0, 1000.0] bps.")

    dy = dy_bps / 10000.0
    price, mac_dur, mod_dur, conv = _bond_price_and_moments(
        par=par,
        coupon_rate=coupon,
        ytm=ytm,
        n_periods=n_periods,
        freq=freq,
    )
    p_cut, _, _, _ = _bond_price_and_moments(
        par=par,
        coupon_rate=coupon,
        ytm=ytm - dy,
        n_periods=n_periods,
        freq=freq,
    )
    p_hike, _, _, _ = _bond_price_and_moments(
        par=par,
        coupon_rate=coupon,
        ytm=ytm + dy,
        n_periods=n_periods,
        freq=freq,
    )

    gain_cut = p_cut - price
    loss_hike = price - p_hike
    asymmetry = gain_cut - loss_hike

    taylor_cut = price * (1.0 + mod_dur * dy + 0.5 * conv * (dy * dy))
    taylor_hike = price * (1.0 - mod_dur * dy + 0.5 * conv * (dy * dy))
    dv01_val = (mod_dur * price) / 10000.0

    if abs(price - par) <= 1e-6 * par:
        par_status = "AT_PAR"
    elif price > par:
        par_status = "PREMIUM"
    else:
        par_status = "DISCOUNT"

    regime = classify_malkiel_regime(
        mac_dur=mac_dur,
        mod_dur=mod_dur,
        coupon_rate=coupon,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        par=par,
        coupon=coupon,
        ytm=ytm,
        maturity=maturity,
        freq=freq,
        price=price,
        mac_dur=mac_dur,
        mod_dur=mod_dur,
        convexity=conv,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MalkielDurationReceipt(
        ticker=cleaned_ticker,
        par_value=par,
        coupon_rate=coupon,
        yield_to_maturity=ytm,
        maturity_years=maturity,
        compounding_freq=freq,
        n_periods=n_periods,
        shock_bps=dy_bps,
        bond_price=price,
        par_status=par_status,
        macaulay_duration=mac_dur,
        modified_duration=mod_dur,
        dv01=dv01_val,
        convexity=conv,
        repriced_on_yield_cut=p_cut,
        repriced_on_yield_hike=p_hike,
        taylor_on_yield_cut=taylor_cut,
        taylor_on_yield_hike=taylor_hike,
        exact_gain_on_cut=gain_cut,
        exact_loss_on_hike=loss_hike,
        malkiel_convexity_asymmetry=asymmetry,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_malkiel_duration_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "TLT",
    coupon_rate: float = 0.045,
    maturity_years: float = 10.0,
) -> MalkielDurationReceipt:
    """Calibrate implied yield-shock stress from trailing close price volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.04, 0.80))
        ytm_est = float(np.clip(0.035 + 0.06 * ann_vol, 0.02, 0.14))
        shock_bps = float(np.clip(ann_vol * 400.0, 25.0, 250.0))
    else:
        ytm_est = 0.045
        shock_bps = 100.0

    return calculate_malkiel_bond_duration(
        coupon_rate=float(coupon_rate),
        yield_to_maturity=ytm_est,
        maturity_years=float(maturity_years),
        compounding_freq=2,
        shock_bps=shock_bps,
        par_value=100.0,
        ticker=ticker,
    )


def persist_malkiel_duration_record(
    receipt: MalkielDurationReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Macaulay & Malkiel (1962) bond duration audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS malkiel_bond_duration_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        coupon_rate REAL NOT NULL,
        yield_to_maturity REAL NOT NULL,
        maturity_years REAL NOT NULL,
        bond_price REAL NOT NULL,
        macaulay_duration REAL NOT NULL,
        modified_duration REAL NOT NULL,
        dv01 REAL NOT NULL,
        convexity REAL NOT NULL,
        malkiel_convexity_asymmetry REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO malkiel_bond_duration_audits (
        ticker, coupon_rate, yield_to_maturity, maturity_years,
        bond_price, macaulay_duration, modified_duration, dv01,
        convexity, malkiel_convexity_asymmetry, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.coupon_rate,
                receipt.yield_to_maturity,
                receipt.maturity_years,
                receipt.bond_price,
                receipt.macaulay_duration,
                receipt.modified_duration,
                receipt.dv01,
                receipt.convexity,
                receipt.malkiel_convexity_asymmetry,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

