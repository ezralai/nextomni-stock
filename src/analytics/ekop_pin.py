r"""David Easley, Nicholas M. Kiefer, Maureen O'Hara & Joseph B. Paperman (1996) EKOP Probability of Informed Trading (PIN) Engine.

Implements David Easley, Nicholas M. Kiefer, Maureen O'Hara, and Joseph B. Paperman (1996),
"Liquidity, Information, and Infrequently Traded Stocks", The Journal of Finance, 51(4), pp. 1405-1436:

- Structural market microstructure tree model with asymmetric private information:
  1. Information event arrives at the start of each session with probability alpha in [0, 1].
  2. Conditional on an event, it is bad news with probability delta in [0, 1] (or good news with 1 - delta).
  3. Uninformed buyer and seller orders each arrive as independent Poisson processes with intensity epsilon > 0.
  4. Informed traders arrive only if an information event occurred, with Poisson intensity mu > 0
     (buying on good news, selling on bad news).
- Structural Probability of Informed Trading (PIN):
      PIN = (alpha * mu) / (alpha * mu + 2 * epsilon)
- Numerically stable log-sum-exp log-likelihood formulation over daily Buy (B_t) and Sell (S_t) order counts:
      ln L_t = -2*epsilon + (B_t + S_t)*ln(epsilon) - ln(B_t! * S_t!) + ln[ (1 - alpha) + alpha*(1 - delta)*e^(-mu)*(1 + mu/epsilon)^B_t + alpha*delta*e^(-mu)*(1 + mu/epsilon)^S_t ]
  preventing catastrophic floating-point factorial overflow and underflow via scipy.special.logsumexp and gammaln.
- Bounded maximum likelihood parameter estimation via Nelder-Mead / L-BFGS-B optimization.
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
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-ekop-1996-audit-key"


@dataclass(frozen=True)
class EKOPReceipt:
    """Immutable receipt for Easley et al. (1996) EKOP Probability of Informed Trading."""

    ticker: str
    sample_size: int
    alpha: float
    delta: float
    mu: float
    epsilon: float
    pin: float
    informed_order_share: float
    uninformed_order_share: float
    daily_order_turnover: float
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
    alpha: float,
    delta: float,
    mu: float,
    epsilon: float,
    pin: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{sample_size}|{alpha:.6f}|{delta:.6f}|{mu:.4f}|"
        f"{epsilon:.4f}|{pin:.6f}|{regime}|{timestamp}"
    )


def sign_ekop_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_ekop_receipt(payload)
    return hmac.compare_digest(expected, signature)


def calculate_ekop_pin(
    *,
    alpha: float = 0.30,
    delta: float = 0.40,
    mu: float = 120.0,
    epsilon: float = 80.0,
    sample_size: int = 60,
    ticker: str = "SPY",
    log_likelihood: float = -500.0,
) -> EKOPReceipt:
    """Calculate closed-form EKOP (1996) Probability of Informed Trading and return audit receipt."""
    clean_ticker = _validate_ticker(ticker)
    a = float(alpha)
    d = float(delta)
    m = float(mu)
    e = float(epsilon)

    if not (0.0 <= a <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {a}")
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"Delta must be in [0, 1], got {d}")
    if m < 0.0:
        raise ValueError(f"Mu (informed intensity) must be non-negative, got {m}")
    if e <= 0.0:
        raise ValueError(f"Epsilon (uninformed intensity) must be positive, got {e}")

    denom = a * m + 2.0 * e
    pin = float((a * m) / denom) if denom > 1e-12 else 0.0

    informed_share = pin
    uninformed_share = float(1.0 - pin)
    daily_turnover = float(a * m + 2.0 * e)

    # 4 structural parameters estimated: alpha, delta, mu, epsilon
    aic = float(2.0 * 4 - 2.0 * log_likelihood)

    if pin >= 0.40:
        regime = "HIGH_INFORMED_TRADING_TOXICITY"
    elif pin >= 0.25:
        regime = "MODERATE_INFORMED_ASYMMETRY"
    elif a <= 0.10 or pin <= 0.08:
        regime = "NOISE_TRADER_DOMINATED_LIQUIDITY"
    else:
        regime = "BALANCED_INFORMATION_EQUILIBRIUM"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        sample_size=sample_size,
        alpha=a,
        delta=d,
        mu=m,
        epsilon=e,
        pin=pin,
        regime=regime,
        timestamp=ts,
    )
    sig_hex = sign_ekop_receipt(payload)

    return EKOPReceipt(
        ticker=clean_ticker,
        sample_size=sample_size,
        alpha=a,
        delta=d,
        mu=m,
        epsilon=e,
        pin=pin,
        informed_order_share=informed_share,
        uninformed_order_share=uninformed_share,
        daily_order_turnover=daily_turnover,
        log_likelihood=float(log_likelihood),
        aic=aic,
        regime=regime,
        hmac_signature=sig_hex,
        timestamp=ts,
    )


def _ekop_daily_log_likelihood(
    buys: np.ndarray,
    sells: np.ndarray,
    alpha: float,
    delta: float,
    mu: float,
    epsilon: float,
) -> float:
    """Evaluate factored, numerically stable EKOP log-likelihood using logsumexp and gammaln."""
    a = float(np.clip(alpha, 1e-6, 1.0 - 1e-6))
    d = float(np.clip(delta, 1e-6, 1.0 - 1e-6))
    m = float(max(1e-4, mu))
    e = float(max(1e-4, epsilon))

    b = np.asarray(buys, dtype=float)
    s = np.asarray(sells, dtype=float)

    # Common terms: -2*epsilon + (B + S)*ln(epsilon) - ln(B!) - ln(S!)
    base = -2.0 * e + (b + s) * math.log(e) - gammaln(b + 1.0) - gammaln(s + 1.0)

    # 3-event mixture via log-sum-exp:
    # 1. No event: ln(1 - a)
    e1 = np.full_like(b, math.log(1.0 - a))
    # 2. Good news: ln(a) + ln(1 - d) - m + B * ln(1 + m/e)
    e2 = math.log(a) + math.log(1.0 - d) - m + b * math.log(1.0 + m / e)
    # 3. Bad news: ln(a) + ln(d) - m + S * ln(1 + m/e)
    e3 = math.log(a) + math.log(d) - m + s * math.log(1.0 + m / e)

    mixture_stack = np.vstack([e1, e2, e3])
    log_mixture = logsumexp(mixture_stack, axis=0)

    total_ll = np.sum(base + log_mixture)
    return float(total_ll) if math.isfinite(total_ll) else -1e10


def calibrate_ekop_pin(
    buys: Sequence[int | float] | np.ndarray,
    sells: Sequence[int | float] | np.ndarray,
    *,
    ticker: str = "SPY",
) -> EKOPReceipt:
    """Calibrate EKOP (1996) structural parameters via maximum likelihood optimization."""
    clean_ticker = _validate_ticker(ticker)
    b_arr = np.asarray(buys, dtype=float)
    s_arr = np.asarray(sells, dtype=float)

    if len(b_arr) != len(s_arr):
        raise ValueError(f"Buys ({len(b_arr)}) and sells ({len(s_arr)}) must have equal length")
    if len(b_arr) < 15:
        raise ValueError(f"At least 15 trading sessions required for EKOP calibration, got {len(b_arr)}")

    mean_b = float(np.mean(b_arr))
    mean_s = float(np.mean(s_arr))
    init_eps = max(1.0, min(mean_b, mean_s) * 0.8)
    init_mu = max(1.0, abs(mean_b - mean_s) * 2.0 + 10.0)

    # Parameters: [alpha, delta, mu, epsilon]
    x0 = np.array([0.30, 0.50, init_mu, init_eps])
    bounds = [(0.01, 0.99), (0.01, 0.99), (0.1, 100000.0), (0.1, 100000.0)]

    def neg_ll(params: np.ndarray) -> float:
        a_p, d_p, m_p, e_p = params[0], params[1], params[2], params[3]
        return -_ekop_daily_log_likelihood(b_arr, s_arr, a_p, d_p, m_p, e_p)

    res = minimize(
        neg_ll,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-7},
    )

    if res.success or math.isfinite(res.fun):
        opt_alpha = float(res.x[0])
        opt_delta = float(res.x[1])
        opt_mu = float(res.x[2])
        opt_epsilon = float(res.x[3])
        final_ll = float(-res.fun)
    else:
        opt_alpha, opt_delta, opt_mu, opt_epsilon = 0.30, 0.50, init_mu, init_eps
        final_ll = _ekop_daily_log_likelihood(b_arr, s_arr, opt_alpha, opt_delta, opt_mu, opt_epsilon)

    return calculate_ekop_pin(
        alpha=opt_alpha,
        delta=opt_delta,
        mu=opt_mu,
        epsilon=opt_epsilon,
        sample_size=len(b_arr),
        ticker=clean_ticker,
        log_likelihood=final_ll,
    )


def evaluate_ekop_pin_from_trades(
    prices: Sequence[float] | np.ndarray,
    volumes: Sequence[float] | np.ndarray | None = None,
    *,
    ticker: str = "SPY",
    n_days: int = 30,
) -> EKOPReceipt:
    """Infer daily buy and sell trade counts from price changes and volume, then calibrate EKOP PIN."""
    clean_ticker = _validate_ticker(ticker)
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 20:
        raise ValueError(f"At least 20 observations required, got {len(p_arr)}")

    returns = np.diff(np.log(p_arr))
    n = len(returns)

    # In absence of tick level order book, decompose volume or session volatility into buy/sell counts
    if volumes is not None and len(volumes) >= len(p_arr):
        v_arr = np.asarray(volumes[1:], dtype=float)
    else:
        v_arr = np.full(n, 1000.0)

    # Signed flow allocation via return sign
    vol_scaled = np.clip(v_arr / (np.mean(v_arr) + 1e-12) * 50.0, 5.0, 500.0)
    buys = np.round(np.where(returns >= 0, vol_scaled * 1.25, vol_scaled * 0.75)).astype(int)
    sells = np.round(np.where(returns < 0, vol_scaled * 1.25, vol_scaled * 0.75)).astype(int)

    return calibrate_ekop_pin(buys, sells, ticker=clean_ticker)


def persist_ekop_record(receipt: EKOPReceipt, db_path: str = "data/nextomni_stock.db") -> bool:
    """Persist EKOP Probability of Informed Trading receipt into SQLite store."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(str(path)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS ekop_pin_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                alpha REAL NOT NULL,
                delta REAL NOT NULL,
                mu REAL NOT NULL,
                epsilon REAL NOT NULL,
                pin REAL NOT NULL,
                informed_order_share REAL NOT NULL,
                uninformed_order_share REAL NOT NULL,
                daily_order_turnover REAL NOT NULL,
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
            INSERT INTO ekop_pin_records (
                ticker, sample_size, alpha, delta, mu, epsilon, pin,
                informed_order_share, uninformed_order_share, daily_order_turnover,
                log_likelihood, aic, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.alpha,
                receipt.delta,
                receipt.mu,
                receipt.epsilon,
                receipt.pin,
                receipt.informed_order_share,
                receipt.uninformed_order_share,
                receipt.daily_order_turnover,
                receipt.log_likelihood,
                receipt.aic,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

