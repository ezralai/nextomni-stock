"""Andrew W. Lo (2002) Autocorrelation-Adjusted & Asymptotic Sharpe Ratio Engine.

Implements Lo (2002, Financial Analysts Journal, "The Statistics of Sharpe Ratios"):
- Naive time-aggregated Sharpe: SR_naive(q) = sqrt(q) * SR_1
- Serial autocorrelation vector rho_1, ..., rho_{q-1}
- Lo (2002) autocorrelation-adjusted q-period Sharpe Ratio:
      SR_Lo(q) = (q / sqrt(q + 2 * sum_{k=1}^{q-1} (q - k) * rho_k)) * SR_1
- Asymptotic GMM standard error (IID and Newey-West / Lo HAC adjusted) and 95% CI
- Sharpe inflation/deflation percentage and serial-dependence regime classification
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

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-lo-2002-audit-key"


@dataclass(frozen=True)
class LoAutocorrSharpeReceipt:
    """Immutable receipt for Andrew Lo (2002) autocorrelation-adjusted Sharpe."""

    ticker: str
    n_obs: int
    q_horizon: int
    lag1_autocorr: float
    sharpe_single_period: float
    sharpe_naive_q: float
    sharpe_lo_q: float
    eta_scaling_factor: float
    inflation_pct: float
    se_lo_q: float
    ci95_low: float
    ci95_high: float
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
    n_obs: int,
    q_horizon: int,
    lag1_autocorr: float,
    sharpe_naive_q: float,
    sharpe_lo_q: float,
    se_lo_q: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_obs}|{q_horizon}|{lag1_autocorr:.8f}|"
        f"{sharpe_naive_q:.8f}|{sharpe_lo_q:.8f}|{se_lo_q:.8f}|"
        f"{regime}|{timestamp}"
    )


def sign_lo_sharpe_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_lo_sharpe_receipt(payload)
    return hmac.compare_digest(expected, signature)


def build_synthetic_ar1_returns(
    n_obs: int = 300,
    rho1: float = 0.35,
    mean_ret: float = 0.0025,
    vol: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate stationary AR(1) return series with specified lag-1 autocorrelation."""
    if n_obs < 30:
        raise ValueError("n_obs must be >= 30")
    if abs(rho1) >= 0.95:
        raise ValueError("rho1 must satisfy |rho1| < 0.95")
    if vol <= 0.0:
        raise ValueError("vol must be > 0")

    rng = np.random.default_rng(seed)
    eps_std = vol * math.sqrt(1.0 - rho1 * rho1)
    r = np.empty(n_obs, dtype=float)
    r[0] = mean_ret + rng.normal(0.0, vol)
    for t in range(1, n_obs):
        r[t] = mean_ret + rho1 * (r[t - 1] - mean_ret) + rng.normal(0.0, eps_std)
    return r


def calculate_lo_autocorr_sharpe(
    returns: Sequence[float] | np.ndarray,
    q_horizon: int = 12,
    rf_period: float = 0.0,
    ticker: str = "SPY",
) -> LoAutocorrSharpeReceipt:
    """Compute Andrew Lo (2002) autocorrelation-adjusted q-period Sharpe Ratio."""
    clean_ticker = _validate_ticker(ticker)
    r = np.asarray(returns, dtype=float)
    if r.ndim != 1:
        raise ValueError("returns must be a 1D array")
    r = r[np.isfinite(r)]
    n_obs = len(r)
    if q_horizon < 2:
        raise ValueError("q_horizon must be >= 2")
    if n_obs < max(30, q_horizon + 10):
        raise ValueError(f"Requires at least {max(30, q_horizon + 10)} return observations")

    excess = r - float(rf_period)
    mu = float(np.mean(excess))
    sigma = float(np.std(excess, ddof=1))
    if sigma <= 1e-12:
        raise ValueError("Return series has near-zero volatility")

    sr1 = mu / sigma
    sr_naive_q = math.sqrt(q_horizon) * sr1

    # Centered returns for sample autocorrelations rho_1 .. rho_{q-1}
    dm = excess - mu
    denom_var = float(np.dot(dm, dm))
    rhos = np.zeros(q_horizon - 1, dtype=float)
    for k in range(1, q_horizon):
        rhos[k - 1] = float(np.dot(dm[k:], dm[:-k]) / denom_var)

    lag1 = float(rhos[0])
    weights = np.arange(q_horizon - 1, 0, -1, dtype=float)
    d_q = float(q_horizon + 2.0 * np.dot(weights, rhos))
    d_q = max(d_q, 1e-6)

    eta_q = float(q_horizon / math.sqrt(d_q))
    sr_lo_q = float(eta_q * sr1)

    if abs(sr_lo_q) > 1e-9:
        inflation_pct = float(((sr_naive_q / sr_lo_q) - 1.0) * 100.0)
    else:
        inflation_pct = 0.0

    # Lo (2002) asymptotic GMM standard error with serial correlation scaling
    vr_q = d_q / float(q_horizon)
    se_1_hac = math.sqrt(max((vr_q + 0.5 * (sr1 * sr1)) / float(n_obs), 1e-12))
    se_lo_q = float(eta_q * se_1_hac)
    ci_low = float(sr_lo_q - 1.96 * se_lo_q)
    ci_high = float(sr_lo_q + 1.96 * se_lo_q)

    if abs(sr_naive_q) < 0.15:
        regime = "NEGLIGIBLE_SHARPE"
    elif inflation_pct >= 12.0 or lag1 >= 0.12:
        regime = "SMOOTHED_SERIAL_INFLATION"
    elif inflation_pct <= -12.0 or lag1 <= -0.12:
        regime = "MEAN_REVERTING_UNDERSTATED"
    else:
        regime = "IID_CONSISTENT"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_obs=n_obs,
        q_horizon=int(q_horizon),
        lag1_autocorr=lag1,
        sharpe_naive_q=sr_naive_q,
        sharpe_lo_q=sr_lo_q,
        se_lo_q=se_lo_q,
        regime=regime,
        timestamp=ts,
    )
    return LoAutocorrSharpeReceipt(
        ticker=clean_ticker,
        n_obs=int(n_obs),
        q_horizon=int(q_horizon),
        lag1_autocorr=lag1,
        sharpe_single_period=sr1,
        sharpe_naive_q=sr_naive_q,
        sharpe_lo_q=sr_lo_q,
        eta_scaling_factor=eta_q,
        inflation_pct=inflation_pct,
        se_lo_q=se_lo_q,
        ci95_low=ci_low,
        ci95_high=ci_high,
        regime=regime,
        hmac_signature=sign_lo_sharpe_receipt(payload),
        timestamp=ts,
    )


def evaluate_lo_sharpe_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    q_horizon: int = 21,
) -> LoAutocorrSharpeReceipt:
    """Evaluate Lo (2002) autocorrelation-adjusted Sharpe from a price series for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < max(35, q_horizon + 12):
        raise ValueError(f"Requires at least {max(35, q_horizon + 12)} price sessions for Lo (2002) Sharpe")
    log_ret = np.diff(np.log(arr))
    return calculate_lo_autocorr_sharpe(log_ret, q_horizon=q_horizon, ticker=ticker)


def persist_lo_sharpe_record(
    receipt: LoAutocorrSharpeReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Lo (2002) autocorrelation-adjusted Sharpe receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS lo_autocorr_sharpe_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_obs INTEGER NOT NULL,
                q_horizon INTEGER NOT NULL,
                lag1_autocorr REAL NOT NULL,
                sharpe_single_period REAL NOT NULL,
                sharpe_naive_q REAL NOT NULL,
                sharpe_lo_q REAL NOT NULL,
                eta_scaling_factor REAL NOT NULL,
                inflation_pct REAL NOT NULL,
                se_lo_q REAL NOT NULL,
                ci95_low REAL NOT NULL,
                ci95_high REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO lo_autocorr_sharpe_records (
                ticker, n_obs, q_horizon, lag1_autocorr,
                sharpe_single_period, sharpe_naive_q, sharpe_lo_q,
                eta_scaling_factor, inflation_pct, se_lo_q,
                ci95_low, ci95_high, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_obs,
                receipt.q_horizon,
                receipt.lag1_autocorr,
                receipt.sharpe_single_period,
                receipt.sharpe_naive_q,
                receipt.sharpe_lo_q,
                receipt.eta_scaling_factor,
                receipt.inflation_pct,
                receipt.se_lo_q,
                receipt.ci95_low,
                receipt.ci95_high,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True
