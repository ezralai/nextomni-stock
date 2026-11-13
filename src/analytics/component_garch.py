r"""Robert F. Engle & Gary G.J. Lee (1999) Component GARCH (C-GARCH) Two-Scale Volatility Engine.

Implements Robert F. Engle & Gary G. J. Lee (1999), "A Permanent and Transitory Component Model
of Stock Return Volatility", in Cointegration, Causality, and Forecasting: A Festschrift in Honour
of Clive W.J. Granger, Oxford University Press, pp. 475-497:

- Decomposes total conditional variance h_t into two distinct physical time scales:
  1. Permanent (long-term trend) component q_t:
         q_t = omega + rho * (q_{t-1} - omega) + phi * (epsilon_{t-1}^2 - h_{t-1})
     where rho in (0.80, 0.999) captures persistent macroeconomic baseline volatility shifts,
     and phi measures how return shocks update long-term trend beliefs.
  2. Transitory (short-term mean-reverting) component (h_t - q_t):
         (h_t - q_t) = alpha * (epsilon_{t-1}^2 - q_{t-1}) + beta * (h_{t-1} - q_{t-1})
     where (alpha + beta) captures rapid reversion toward the time-varying trend q_t.
  3. Total conditional variance:
         h_t = q_t + alpha * (epsilon_{t-1}^2 - q_{t-1}) + beta * (h_{t-1} - q_{t-1})
- Two-scale persistence hierarchy:
      0 < alpha + beta < rho < 1.0
  Ensuring transitory shocks decay substantially faster than structural trend innovations:
      t_{1/2}^{trans} = ln(0.5) / ln(alpha + beta) < t_{1/2}^{perm} = ln(0.5) / ln(rho)
- Log-likelihood evaluation and Akaike Information Criterion (AIC).
- Strict ticker input validation, HMAC-SHA256 audit sealing, and parameterized SQLite logging.
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
DEFAULT_HMAC_KEY = "nextomni-stock-cgarch-1999-audit-key"


@dataclass(frozen=True)
class CGarchReceipt:
    """Immutable receipt for Robert F. Engle & Gary G.J. Lee (1999) Component GARCH evaluation."""

    ticker: str
    sample_size: int
    omega: float
    rho: float
    phi: float
    alpha: float
    beta: float
    latest_permanent_vol: float
    latest_transitory_vol: float
    latest_total_vol: float
    permanent_variance_ratio: float
    transitory_half_life: float
    permanent_half_life: float
    log_likelihood: float
    aic: float
    regime: str
    hmac_signature: str
    timestamp: str


def _validate_ticker(ticker: str) -> str:
    clean = str(ticker).strip().upper()
    if not clean or not TICKER_REGEX.match(clean):
        raise ValueError(
            f"Invalid ticker: '{ticker}'. Must be alphanumeric with optional ._- and max 20 chars."
        )
    return clean


def _build_hmac_payload(
    *,
    ticker: str,
    sample_size: int,
    omega: float,
    rho: float,
    alpha: float,
    beta: float,
    latest_total_vol: float,
    latest_permanent_vol: float,
    permanent_variance_ratio: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{sample_size}|{omega:.8f}|{rho:.6f}|{alpha:.6f}|{beta:.6f}|"
        f"{latest_total_vol:.6f}|{latest_permanent_vol:.6f}|"
        f"{permanent_variance_ratio:.6f}|{regime}|{timestamp}"
    )


def sign_cgarch_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_cgarch_receipt(payload)
    return hmac.compare_digest(expected, signature)


def calculate_component_garch_trajectory(
    returns: Sequence[float] | np.ndarray,
    *,
    omega: float = 0.0001,
    rho: float = 0.98,
    phi: float = 0.04,
    alpha: float = 0.06,
    beta: float = 0.70,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Execute the recursive Component GARCH(1,1) two-scale variance filter."""
    rets = np.asarray(returns, dtype=float)
    n = len(rets)
    if n < 20:
        raise ValueError(f"At least 20 return observations required, got {n}")

    w = float(omega)
    r = float(rho)
    p = float(phi)
    a = float(alpha)
    b = float(beta)

    if w <= 0.0:
        raise ValueError(f"Omega must be positive, got {w}")
    if not (0.0 < r < 1.0):
        raise ValueError(f"Rho (permanent persistence) must be in (0, 1), got {r}")
    if p < 0.0:
        raise ValueError(f"Phi must be non-negative, got {p}")
    if a < 0.0 or b < 0.0:
        raise ValueError(f"Alpha and Beta must be non-negative, got alpha={a}, beta={b}")
    if (a + b) >= r:
        raise ValueError(
            f"Hierarchy violation: Transitory persistence (alpha + beta = {a + b:.4f}) "
            f"must be strictly less than permanent persistence (rho = {r:.4f})."
        )

    # Initialize variance states with sample variance
    unconditional_var = float(max(1e-6, np.var(rets, ddof=1)))
    q_series = np.zeros(n, dtype=float)
    h_series = np.zeros(n, dtype=float)

    # State at t=0
    q_prev = max(1e-6, w) if abs(w - unconditional_var) < 1.0 else unconditional_var
    h_prev = unconditional_var
    eps_sq_prev = unconditional_var

    log_lik = 0.0

    for t in range(n):
        eps_sq = rets[t] ** 2

        # 1. Update permanent component q_t
        q_t = w + r * (q_prev - w) + p * (eps_sq_prev - h_prev)
        q_t = max(1e-6, q_t)

        # 2. Update total conditional variance h_t = q_t + alpha*(eps^2 - q) + beta*(h - q)
        transitory_comp = a * (eps_sq_prev - q_prev) + b * (h_prev - q_prev)
        h_t = max(1e-6, q_t + transitory_comp)

        q_series[t] = q_t
        h_series[t] = h_t

        # Gaussian log-likelihood increment: -0.5 * (ln(2*pi) + ln(h_t) + eps^2 / h_t)
        log_lik += -0.5 * (math.log(2.0 * math.pi) + math.log(h_t) + (eps_sq / h_t))

        # Advance state
        q_prev = q_t
        h_prev = h_t
        eps_sq_prev = eps_sq

    return q_series, h_series, float(log_lik)


def calculate_component_garch(
    returns: Sequence[float] | np.ndarray,
    *,
    omega: float = 0.0001,
    rho: float = 0.98,
    phi: float = 0.04,
    alpha: float = 0.06,
    beta: float = 0.70,
    ticker: str = "SPY",
) -> CGarchReceipt:
    """Evaluate Robert F. Engle & Gary G.J. Lee (1999) Component GARCH metrics and return an immutable audit receipt."""
    clean_ticker = _validate_ticker(ticker)
    rets = np.asarray(returns, dtype=float)
    n = len(rets)

    q_series, h_series, log_lik = calculate_component_garch_trajectory(
        rets, omega=omega, rho=rho, phi=phi, alpha=alpha, beta=beta
    )

    # 4 parameters estimated: omega, rho, phi, (alpha + beta)
    k_params = 5
    aic = float(2.0 * k_params - 2.0 * log_lik)

    # Half-lives in trading sessions
    trans_persistence = alpha + beta
    trans_half_life = float(math.log(0.5) / math.log(trans_persistence)) if trans_persistence < 0.9999 else 999.0
    perm_half_life = float(math.log(0.5) / math.log(rho)) if rho < 0.9999 else 999.0

    # Annualized volatility states (252 sessions/year)
    latest_q = q_series[-1]
    latest_h = h_series[-1]
    latest_trans_var = max(0.0, latest_h - latest_q)

    latest_perm_vol = float(math.sqrt(latest_q * 252.0))
    latest_trans_vol = float(math.sqrt(latest_trans_var * 252.0))
    latest_total_vol = float(math.sqrt(latest_h * 252.0))

    perm_var_ratio = float(latest_q / latest_h)

    if latest_h > 1.35 * latest_q:
        regime = "ACUTE_TRANSITORY_SHOCK_SPIKE"
    elif latest_h < 0.80 * latest_q:
        regime = "COMPRESSED_TRANSITORY_VARIANCE"
    elif perm_half_life >= 100.0:
        regime = "HIGH_PERSISTENCE_STRUCTURAL_SHIFT"
    else:
        regime = "STABLE_EQUILIBRIUM_TWO_SCALE"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        sample_size=n,
        omega=omega,
        rho=rho,
        alpha=alpha,
        beta=beta,
        latest_total_vol=latest_total_vol,
        latest_permanent_vol=latest_perm_vol,
        permanent_variance_ratio=perm_var_ratio,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_cgarch_receipt(payload)

    return CGarchReceipt(
        ticker=clean_ticker,
        sample_size=n,
        omega=omega,
        rho=rho,
        phi=phi,
        alpha=alpha,
        beta=beta,
        latest_permanent_vol=latest_perm_vol,
        latest_transitory_vol=latest_trans_vol,
        latest_total_vol=latest_total_vol,
        permanent_variance_ratio=perm_var_ratio,
        transitory_half_life=trans_half_life,
        permanent_half_life=perm_half_life,
        log_likelihood=log_lik,
        aic=aic,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def evaluate_component_garch_profile(
    prices: Sequence[float] | np.ndarray,
    *,
    ticker: str = "SPY",
    omega: float | None = None,
    rho: float = 0.98,
    phi: float = 0.04,
    alpha: float = 0.06,
    beta: float = 0.70,
) -> CGarchReceipt:
    """Infer Component GARCH profile from raw price observations."""
    clean_ticker = _validate_ticker(ticker)
    arr = np.asarray(prices, dtype=float)
    if len(arr) < 25:
        raise ValueError(f"At least 25 price observations required, got {len(arr)}")

    log_returns = np.diff(np.log(arr))
    sample_var = float(np.var(log_returns, ddof=1))

    # Baseline daily omega calibrated from sample variance
    omega_calib = float(sample_var * (1.0 - rho)) if omega is None else float(omega)

    return calculate_component_garch(
        log_returns,
        omega=max(1e-7, omega_calib),
        rho=rho,
        phi=phi,
        alpha=alpha,
        beta=beta,
        ticker=clean_ticker,
    )


def persist_component_garch_record(receipt: CGarchReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist Component GARCH receipt into SQLite database under parameterized query."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS component_garch_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                omega REAL NOT NULL,
                rho REAL NOT NULL,
                phi REAL NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                latest_permanent_vol REAL NOT NULL,
                latest_transitory_vol REAL NOT NULL,
                latest_total_vol REAL NOT NULL,
                permanent_variance_ratio REAL NOT NULL,
                transitory_half_life REAL NOT NULL,
                permanent_half_life REAL NOT NULL,
                log_likelihood REAL NOT NULL,
                aic REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO component_garch_records (
                ticker, sample_size, omega, rho, phi, alpha, beta,
                latest_permanent_vol, latest_transitory_vol, latest_total_vol,
                permanent_variance_ratio, transitory_half_life, permanent_half_life,
                log_likelihood, aic, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.omega,
                receipt.rho,
                receipt.phi,
                receipt.alpha,
                receipt.beta,
                receipt.latest_permanent_vol,
                receipt.latest_transitory_vol,
                receipt.latest_total_vol,
                receipt.permanent_variance_ratio,
                receipt.transitory_half_life,
                receipt.permanent_half_life,
                receipt.log_likelihood,
                receipt.aic,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

