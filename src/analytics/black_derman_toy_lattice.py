"""Black, Derman & Toy (1990) Lognormal Arbitrage-Free Interest Rate Binomial Lattice.

Implements:
- Arrow-Debreu forward-induction calibration of median short rates U_i across N
  binomial steps so tree zero-coupon bond prices match an observed spot yield curve:
      r_{i,j} = U_i * exp(2 * sigma_i * sqrt(dt) * (j - i / 2))
- Zero-rate positivity guarantee (lognormal short-rate tree)
- Backward-induction valuation of straight vs. Bermudan callable coupon bonds
  to quantify embedded call option value and arbitrage-free calibration error
- Synthetic yield curve generator & empirical curve evaluator for Dashboard Tab 2
- Strict ticker validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-bdt-1990-audit-key"


@dataclass(frozen=True)
class BDTLatticeReceipt:
    """Immutable receipt for Black-Derman-Toy (1990) binomial lattice calibration."""

    ticker: str
    n_steps: int
    dt: float
    short_rate_initial: float
    terminal_rate_low: float
    terminal_rate_median: float
    terminal_rate_high: float
    curve_slope_bps: float
    volatility: float
    max_calibration_error: float
    straight_bond_price: float
    callable_bond_price: float
    embedded_call_value: float
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    n_steps: int,
    dt: float,
    short_rate_initial: float,
    terminal_rate_median: float,
    max_calibration_error: float,
    straight_bond_price: float,
    callable_bond_price: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_steps}|{dt:.6f}|{short_rate_initial:.8f}|"
        f"{terminal_rate_median:.8f}|{max_calibration_error:.10e}|"
        f"{straight_bond_price:.6f}|{callable_bond_price:.6f}|{regime}|{timestamp}"
    )


def sign_bdt_lattice_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_bdt_lattice_receipt(payload)
    return hmac.compare_digest(expected, signature)


def build_synthetic_yield_curve(
    n_steps: int = 8,
    base_rate: float = 0.04,
    slope_bps: float = 120.0,
) -> np.ndarray:
    """Generate a smooth N-step annualized zero-coupon spot yield curve."""
    if n_steps < 2:
        raise ValueError("n_steps must be >= 2")
    if base_rate <= 0.0:
        raise ValueError("base_rate must be strictly positive")
    t_grid = np.linspace(0.0, 1.0, n_steps)
    slope = slope_bps / 10000.0
    curve = base_rate + slope * np.sqrt(t_grid)
    if np.any(curve <= 1e-5):
        raise ValueError("Yield curve must remain strictly positive across all maturities")
    return curve.astype(float)


def calibrate_bdt_lattice(
    spot_rates: Sequence[float] | np.ndarray,
    volatilities: float | Sequence[float] | np.ndarray = 0.15,
    dt: float = 1.0,
    coupon_rate: float = 0.05,
    call_price: float = 100.0,
    face_value: float = 100.0,
    ticker: str = "UST",
) -> BDTLatticeReceipt:
    """Calibrate a lognormal Black-Derman-Toy (1990) binomial short-rate lattice.

    Parameters
    ----------
    spot_rates:
        Sequence of annualized spot zero rates y_1, ..., y_N for maturities
        dt, 2*dt, ..., N*dt. All spot rates must be strictly positive.
    volatilities:
        Scalar or N-length sequence of annualized lognormal short-rate volatilities.
    dt:
        Time step in years (must be > 0).
    coupon_rate:
        Annualized par coupon rate paid per step (coupon = face_value * coupon_rate * dt).
    call_price:
        Bermudan call strike price for callable bond valuation (exercisable at steps 1..N-1).
    face_value:
        Bond principal face value (default 100.0).
    ticker:
        Identifier for the curve/instrument.
    """
    clean_ticker = _validate_ticker(ticker)
    y = np.asarray(spot_rates, dtype=float)
    if y.ndim != 1 or len(y) < 2:
        raise ValueError("spot_rates must be a 1D sequence with at least 2 maturities")
    if not np.all(np.isfinite(y)) or np.any(y <= 0.0):
        raise ValueError("All spot_rates must be finite and strictly positive")
    if dt <= 0.0 or not math.isfinite(dt):
        raise ValueError("dt must be finite and > 0")
    if face_value <= 0.0 or call_price <= 0.0 or coupon_rate < 0.0:
        raise ValueError("Invalid bond parameters: face_value/call_price must be > 0 and coupon_rate >= 0")

    n_steps = len(y)
    if np.ndim(volatilities) == 0:
        sig = np.full(n_steps, float(volatilities), dtype=float)
    else:
        sig = np.asarray(volatilities, dtype=float)
        if sig.shape != (n_steps,):
            raise ValueError(f"volatilities must have length {n_steps}")
    if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
        raise ValueError("All volatilities must be finite and strictly positive")

    # Market zero-coupon discount factors P_mkt(0, (i+1)*dt)
    maturities = np.arange(1, n_steps + 1, dtype=float) * dt
    p_mkt = np.exp(-y * maturities)

    # Forward induction using Arrow-Debreu state prices Q_{i,j}
    rates = np.zeros((n_steps, n_steps), dtype=float)
    q_curr = np.array([1.0], dtype=float)
    max_calib_err = 0.0

    for i in range(n_steps):
        j_idx = np.arange(i + 1, dtype=float)
        spread_exp = np.exp(2.0 * sig[i] * math.sqrt(dt) * (j_idx - 0.5 * i))
        target_p = float(p_mkt[i])

        def _price_diff(u_val: float) -> float:
            r_row = u_val * spread_exp
            disc = np.exp(-r_row * dt)
            return float(np.dot(q_curr, disc)) - target_p

        u_star = float(brentq(_price_diff, 1e-7, 5.0, xtol=1e-12, rtol=1e-12, maxiter=200))
        r_row = u_star * spread_exp
        rates[i, : i + 1] = r_row
        disc_row = np.exp(-r_row * dt)
        tree_p = float(np.dot(q_curr, disc_row))
        max_calib_err = max(max_calib_err, abs(tree_p - target_p))

        if i < n_steps - 1:
            q_next = np.zeros(i + 2, dtype=float)
            contrib = 0.5 * q_curr * disc_row
            q_next[:-1] += contrib
            q_next[1:] += contrib
            q_curr = q_next

    # Backward induction for straight and Bermudan callable coupon bond
    coupon_pmt = face_value * coupon_rate * dt
    v_straight = np.full(n_steps + 1, face_value + coupon_pmt, dtype=float)
    v_callable = np.full(n_steps + 1, face_value + coupon_pmt, dtype=float)

    for i in range(n_steps - 1, -1, -1):
        disc_row = np.exp(-rates[i, : i + 1] * dt)
        v_straight = disc_row * 0.5 * (v_straight[:-1] + v_straight[1:]) + (coupon_pmt if i > 0 else 0.0)
        cont_call = disc_row * 0.5 * (v_callable[:-1] + v_callable[1:]) + (coupon_pmt if i > 0 else 0.0)
        if i >= 1:
            # Issuer calls when continuation ex-coupon exceeds call_price (+ accrued step coupon)
            v_callable = np.minimum(cont_call, call_price + coupon_pmt)
        else:
            v_callable = cont_call

    straight_px = float(v_straight[0])
    callable_px = float(v_callable[0])
    embedded_call = max(0.0, straight_px - callable_px)

    r_init = float(rates[0, 0])
    last_row = rates[n_steps - 1, :n_steps]
    r_low = float(last_row[0])
    r_high = float(last_row[-1])
    r_med = float(np.median(last_row))
    slope_bps = float((y[-1] - y[0]) * 10000.0)
    mean_vol = float(np.mean(sig))

    if mean_vol >= 0.30:
        regime = "HIGH_VOL_CURVATURE"
    elif slope_bps < -15.0:
        regime = "INVERTED_TIGHTENING"
    elif slope_bps > 25.0:
        regime = "NORMAL_STEEP"
    else:
        regime = "FLAT_NEUTRAL"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_steps=n_steps,
        dt=dt,
        short_rate_initial=r_init,
        terminal_rate_median=r_med,
        max_calibration_error=max_calib_err,
        straight_bond_price=straight_px,
        callable_bond_price=callable_px,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_bdt_lattice_receipt(payload)

    return BDTLatticeReceipt(
        ticker=clean_ticker,
        n_steps=n_steps,
        dt=float(dt),
        short_rate_initial=r_init,
        terminal_rate_low=r_low,
        terminal_rate_median=r_med,
        terminal_rate_high=r_high,
        curve_slope_bps=slope_bps,
        volatility=mean_vol,
        max_calibration_error=max_calib_err,
        straight_bond_price=straight_px,
        callable_bond_price=callable_px,
        embedded_call_value=embedded_call,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_bdt_lattice_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    n_steps: int = 8,
    base_rate: float = 0.042,
) -> BDTLatticeReceipt:
    """Derive a stylized equity-implied rate vol & slope curve for Tab 2 inspection."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price observations to calibrate BDT lattice profile")

    log_ret = np.diff(np.log(arr))
    realized_vol = float(np.clip(np.std(log_ret, ddof=1) * math.sqrt(252.0) * 0.75, 0.08, 0.45))
    momentum = float((arr[-1] / arr[0]) - 1.0)
    slope_bps = float(np.clip(momentum * 250.0, -150.0, 220.0))
    curve = build_synthetic_yield_curve(n_steps=n_steps, base_rate=base_rate, slope_bps=slope_bps)

    return calibrate_bdt_lattice(
        spot_rates=curve,
        volatilities=realized_vol,
        dt=1.0,
        coupon_rate=base_rate + 0.005,
        call_price=100.0,
        ticker=ticker,
    )


def persist_bdt_lattice_record(
    receipt: BDTLatticeReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Black-Derman-Toy lattice calibration receipt to SQLite with foreign keys enabled."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bdt_lattice_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_steps INTEGER NOT NULL,
                dt REAL NOT NULL,
                short_rate_initial REAL NOT NULL,
                terminal_rate_low REAL NOT NULL,
                terminal_rate_median REAL NOT NULL,
                terminal_rate_high REAL NOT NULL,
                curve_slope_bps REAL NOT NULL,
                volatility REAL NOT NULL,
                max_calibration_error REAL NOT NULL,
                straight_bond_price REAL NOT NULL,
                callable_bond_price REAL NOT NULL,
                embedded_call_value REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO bdt_lattice_records (
                ticker, n_steps, dt, short_rate_initial,
                terminal_rate_low, terminal_rate_median, terminal_rate_high,
                curve_slope_bps, volatility, max_calibration_error,
                straight_bond_price, callable_bond_price, embedded_call_value,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_steps,
                receipt.dt,
                receipt.short_rate_initial,
                receipt.terminal_rate_low,
                receipt.terminal_rate_median,
                receipt.terminal_rate_high,
                receipt.curve_slope_bps,
                receipt.volatility,
                receipt.max_calibration_error,
                receipt.straight_bond_price,
                receipt.callable_bond_price,
                receipt.embedded_call_value,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

