"""Nelson, Siegel (1987) & Svensson (1994) Parametric Yield Curve Factor Engine.

Implements:
- Exact 6-parameter Nelson-Siegel-Svensson (NSS) spot rate and instantaneous
  forward rate factor decomposition:
      y(m) = beta0
           + beta1 * ((1 - exp(-m / tau1)) / (m / tau1))
           + beta2 * (((1 - exp(-m / tau1)) / (m / tau1)) - exp(-m / tau1))
           + beta3 * (((1 - exp(-m / tau2)) / (m / tau2)) - exp(-m / tau2))
- Nested OLS estimation of factor loadings (beta0, beta1, beta2, beta3) conditional
  on decay parameters (tau1, tau2), optimized via multi-start bounded L-BFGS-B
- Analytical zero-maturity limit lim_{m -> 0+} y(m) = beta0 + beta1
- Synthetic yield curve generator (normal steep, humped bull, inverted bear, flat)
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
from scipy.optimize import minimize

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-nss-1994-audit-key"


@dataclass(frozen=True)
class NSSCurveReceipt:
    """Immutable receipt for Nelson-Siegel-Svensson yield curve factor fit."""

    ticker: str
    n_maturities: int
    beta0_level: float
    beta1_slope: float
    beta2_curvature: float
    beta3_second_hump: float
    tau1: float
    tau2: float
    short_end_limit: float
    spread_10y_2y_bps: float
    rmse_bps: float
    r_squared: float
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
    n_maturities: int,
    beta0_level: float,
    beta1_slope: float,
    beta2_curvature: float,
    beta3_second_hump: float,
    tau1: float,
    tau2: float,
    rmse_bps: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_maturities}|{beta0_level:.8f}|{beta1_slope:.8f}|"
        f"{beta2_curvature:.8f}|{beta3_second_hump:.8f}|{tau1:.6f}|{tau2:.6f}|"
        f"{rmse_bps:.6f}|{regime}|{timestamp}"
    )


def sign_nss_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_nss_receipt(payload)
    return hmac.compare_digest(expected, signature)


def nss_design_matrix(
    maturities: Sequence[float] | np.ndarray,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Compute the N x 4 Nelson-Siegel-Svensson factor loading matrix."""
    m = np.asarray(maturities, dtype=float)
    if tau1 <= 0.0 or tau2 <= 0.0:
        raise ValueError("Decay parameters tau1 and tau2 must be strictly positive")

    x1 = m / tau1
    x2 = m / tau2

    # Exact Taylor limit (1 - exp(-x))/x -> 1 - x/2 as x -> 0+
    f1 = np.where(x1 < 1e-7, 1.0 - 0.5 * x1, (1.0 - np.exp(-x1)) / np.maximum(x1, 1e-12))
    f2 = f1 - np.exp(-x1)

    g1 = np.where(x2 < 1e-7, 1.0 - 0.5 * x2, (1.0 - np.exp(-x2)) / np.maximum(x2, 1e-12))
    f3 = g1 - np.exp(-x2)

    ones = np.ones_like(m, dtype=float)
    return np.column_stack([ones, f1, f2, f3])


def evaluate_nss_yield(
    maturities: Sequence[float] | np.ndarray,
    beta0: float,
    beta1: float,
    beta2: float,
    beta3: float,
    tau1: float,
    tau2: float,
) -> np.ndarray:
    """Evaluate NSS spot yield curve at given maturities (in years)."""
    x = nss_design_matrix(maturities, tau1=tau1, tau2=tau2)
    betas = np.array([beta0, beta1, beta2, beta3], dtype=float)
    return x @ betas


def build_synthetic_nss_curve(
    mode: str = "normal",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate benchmark Treasury maturities (years) and NSS-consistent yields."""
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)
    clean_mode = str(mode).strip().lower()
    if clean_mode == "normal":
        y = evaluate_nss_yield(maturities, 0.048, -0.015, -0.008, 0.012, 1.8, 6.5)
    elif clean_mode == "humped":
        y = evaluate_nss_yield(maturities, 0.042, -0.012, 0.028, 0.005, 1.5, 5.5)
    elif clean_mode == "inverted":
        y = evaluate_nss_yield(maturities, 0.036, 0.018, -0.014, 0.006, 1.6, 5.0)
    elif clean_mode == "flat":
        y = evaluate_nss_yield(maturities, 0.041, -0.001, 0.002, -0.001, 2.0, 6.0)
    else:
        raise ValueError(f"Unsupported synthetic NSS mode: {mode!r}")
    return maturities, y


def fit_nelson_siegel_svensson(
    maturities: Sequence[float] | np.ndarray,
    yields: Sequence[float] | np.ndarray,
    ticker: str = "UST",
) -> NSSCurveReceipt:
    """Fit Nelson-Siegel-Svensson (1994) 6-parameter yield curve model via nested OLS."""
    clean_ticker = _validate_ticker(ticker)
    m = np.asarray(maturities, dtype=float)
    y = np.asarray(yields, dtype=float)

    if m.ndim != 1 or y.ndim != 1 or len(m) != len(y):
        raise ValueError("maturities and yields must be 1D arrays of equal length")
    if len(m) < 5:
        raise ValueError("At least 5 distinct maturities are required for NSS estimation")
    if not np.all(np.isfinite(m)) or not np.all(np.isfinite(y)):
        raise ValueError("maturities and yields must be finite")
    if np.any(m < 0.0):
        raise ValueError("maturities must be non-negative")
    if len(np.unique(np.round(m, 6))) < 5:
        raise ValueError("At least 5 distinct maturities are required for NSS estimation")

    def _solve_betas(tau1: float, tau2: float) -> tuple[np.ndarray, float]:
        x = nss_design_matrix(m, tau1=tau1, tau2=tau2)
        # Small Tikhonov ridge only if near-collinear tau1 ~= tau2
        if abs(tau1 - tau2) < 0.05:
            xtx = x.T @ x + 1e-8 * np.eye(4)
            betas = np.linalg.solve(xtx, x.T @ y)
        else:
            betas, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - (x @ betas)
        sse = float(np.dot(resid, resid))
        return betas, sse

    def _objective(params: np.ndarray) -> float:
        t1, t2 = float(params[0]), float(params[1])
        _, sse = _solve_betas(t1, t2)
        return sse

    best_res = None
    starts = [(1.2, 4.5), (2.0, 7.0), (0.8, 3.0), (3.0, 10.0)]
    bounds = [(0.15, 10.0), (0.5, 25.0)]
    for s in starts:
        res = minimize(
            _objective,
            x0=np.array(s, dtype=float),
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-14, "gtol": 1e-10, "maxiter": 200},
        )
        if best_res is None or float(res.fun) < float(best_res.fun):
            best_res = res

    tau1_opt = float(best_res.x[0])
    tau2_opt = float(best_res.x[1])
    betas_opt, sse_opt = _solve_betas(tau1_opt, tau2_opt)

    beta0, beta1, beta2, beta3 = [float(v) for v in betas_opt]
    y_hat = evaluate_nss_yield(m, beta0, beta1, beta2, beta3, tau1_opt, tau2_opt)
    rmse_bps = float(math.sqrt(sse_opt / len(y)) * 10000.0)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse_opt / ss_tot) if ss_tot > 1e-15 else 1.0
    r2 = float(np.clip(r2, -1.0, 1.0))

    short_limit = beta0 + beta1
    y_2y = float(evaluate_nss_yield(np.array([2.0]), beta0, beta1, beta2, beta3, tau1_opt, tau2_opt)[0])
    y_10y = float(evaluate_nss_yield(np.array([10.0]), beta0, beta1, beta2, beta3, tau1_opt, tau2_opt)[0])
    spread_10y_2y_bps = float((y_10y - y_2y) * 10000.0)

    if max(abs(beta2), abs(beta3)) > 0.015 and abs(spread_10y_2y_bps) <= 45.0:
        regime = "HUMPED_BULL"
    elif spread_10y_2y_bps < -15.0:
        regime = "INVERTED_BEAR"
    elif spread_10y_2y_bps > 20.0:
        regime = "NORMAL_STEEP"
    else:
        regime = "FLAT_ANCHORED"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_maturities=len(m),
        beta0_level=beta0,
        beta1_slope=beta1,
        beta2_curvature=beta2,
        beta3_second_hump=beta3,
        tau1=tau1_opt,
        tau2=tau2_opt,
        rmse_bps=rmse_bps,
        regime=regime,
        timestamp=ts,
    )
    sig = sign_nss_receipt(payload)

    return NSSCurveReceipt(
        ticker=clean_ticker,
        n_maturities=int(len(m)),
        beta0_level=beta0,
        beta1_slope=beta1,
        beta2_curvature=beta2,
        beta3_second_hump=beta3,
        tau1=tau1_opt,
        tau2=tau2_opt,
        short_end_limit=short_limit,
        spread_10y_2y_bps=spread_10y_2y_bps,
        rmse_bps=rmse_bps,
        r_squared=r2,
        regime=regime,
        hmac_signature=sig,
        timestamp=ts,
    )


def evaluate_nss_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
) -> NSSCurveReceipt:
    """Derive a price-implied macro term structure and fit NSS factors for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions to evaluate NSS yield profile")

    log_ret = np.diff(np.log(arr))
    trend = float((arr[-1] / arr[0]) - 1.0)
    vol = float(np.std(log_ret, ddof=1) * math.sqrt(252.0))
    maturities = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 10.0, 20.0, 30.0], dtype=float)
    b0 = float(np.clip(0.045 + 0.01 * trend, 0.025, 0.075))
    b1 = float(np.clip(-0.02 * trend, -0.025, 0.025))
    b2 = float(np.clip(0.03 * (vol - 0.18), -0.025, 0.030))
    b3 = 0.008
    yields = evaluate_nss_yield(maturities, b0, b1, b2, b3, 1.6, 5.8)
    return fit_nelson_siegel_svensson(maturities, yields, ticker=ticker)


def persist_nss_record(
    receipt: NSSCurveReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Nelson-Siegel-Svensson yield curve fit receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nss_curve_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_maturities INTEGER NOT NULL,
                beta0_level REAL NOT NULL,
                beta1_slope REAL NOT NULL,
                beta2_curvature REAL NOT NULL,
                beta3_second_hump REAL NOT NULL,
                tau1 REAL NOT NULL,
                tau2 REAL NOT NULL,
                short_end_limit REAL NOT NULL,
                spread_10y_2y_bps REAL NOT NULL,
                rmse_bps REAL NOT NULL,
                r_squared REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO nss_curve_records (
                ticker, n_maturities, beta0_level, beta1_slope,
                beta2_curvature, beta3_second_hump, tau1, tau2,
                short_end_limit, spread_10y_2y_bps, rmse_bps, r_squared,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_maturities,
                receipt.beta0_level,
                receipt.beta1_slope,
                receipt.beta2_curvature,
                receipt.beta3_second_hump,
                receipt.tau1,
                receipt.tau2,
                receipt.short_end_limit,
                receipt.spread_10y_2y_bps,
                receipt.rmse_bps,
                receipt.r_squared,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True
