"""Robert F. Engle & Jeffrey R. Russell (1998) Autoregressive Conditional Duration (ACD) Engine.

Implements the point-process econometric model formulated in:
Engle, Robert F., & Russell, Jeffrey R. (1998).
"Autoregressive Conditional Duration: A New Model for Irregularly Spaced Transaction Data."
Econometrica, 66(5), 1127-1162.

ACD models the stochastic arrival times between market events (trades, price revisions,
or quote updates) as an autoregressive point process:
    x_i = t_i - t_{i-1} > 0  (inter-event duration)
    psi_i = E[x_i | F_{i-1}] = omega + alpha * x_{i-1} + beta * psi_{i-1}
    x_i = psi_i * eps_i,  eps_i ~ i.i.d. Exponential(1)
where:
    omega > 0: baseline expected duration
    alpha >= 0: arrival shock sensitivity
    beta >= 0: expected duration persistence
    alpha + beta < 1: covariance stationarity

Provides real-time modeling of trade clustering, liquidity droughts, and order arrival acceleration.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

import numpy as np
from scipy import optimize

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,16}$")
SECRET_SALT = os.environ.get("NEXTOMNI_AUDIT_SALT", "nextomni_acd_audit_salt").encode("utf-8")


@dataclass(frozen=True)
class ACDReceipt:
    """Immutable receipt containing Engle & Russell (1998) ACD duration estimation."""

    ticker: str
    sample_size: int
    mean_duration: float
    omega: float
    alpha: float
    beta: float
    persistence: float
    unconditional_expected_duration: float
    current_conditional_duration: float
    current_hazard_rate: float
    trading_speed_ratio: float
    half_life_shocks: float
    log_likelihood: float
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> dict:
        return asdict(self)


def _compute_hmac(payload: str) -> str:
    return hmac.new(SECRET_SALT, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify cryptographic HMAC-SHA256 signature against payload."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


def calculate_acd_model(
    durations: Sequence[float],
    ticker: str = "SPY",
    derivation_mode: str = "DIRECT_DURATIONS",
) -> ACDReceipt:
    """Estimate Engle & Russell (1998) EACD(1, 1) model via Maximum Likelihood.

    Parameters
    ----------
    durations : Sequence[float]
        Series of strictly positive inter-event durations x_i.
    ticker : str
        Stock ticker symbol.
    derivation_mode : str
        Source mode label.

    Returns
    -------
    ACDReceipt
        Complete duration model estimation receipt.
    """
    clean_ticker = ticker.strip().upper() if TICKER_REGEX.match(ticker.strip().upper()) else "UNKNOWN"
    now_iso = datetime.now(timezone.utc).isoformat()

    x_arr = np.asarray(durations, dtype=np.float64)
    valid_mask = np.isfinite(x_arr) & (x_arr > 0.0)
    x = x_arr[valid_mask]
    n_pts = len(x)

    if n_pts < 20:
        return _empty_receipt(clean_ticker, now_iso)

    mean_x = float(np.mean(x))
    if mean_x < 1e-6:
        return _empty_receipt(clean_ticker, now_iso)

    # Initial parameter guesses
    omega_init = 0.10 * mean_x
    alpha_init = 0.15
    beta_init = 0.75

    # Parameter constraints: omega > 0, alpha >= 0, beta >= 0, alpha + beta <= 0.98
    bounds = [
        (0.005 * mean_x, 2.0 * mean_x),  # omega
        (0.01, 0.45),                    # alpha
        (0.01, 0.95),                    # beta
    ]

    def neg_log_likelihood(params: np.ndarray) -> float:
        om, al, be = params
        if om <= 0.0 or al < 0.0 or be < 0.0 or (al + be) >= 0.99:
            return 1e9

        psi = np.empty(n_pts, dtype=np.float64)
        psi[0] = mean_x
        for i in range(1, n_pts):
            psi[i] = om + al * x[i - 1] + be * psi[i - 1]

        # Guard against zero or negative psi
        psi = np.maximum(psi, 1e-6)

        # Log-likelihood: ln L = - sum(ln(psi_i) + x_i / psi_i)
        ll = -np.sum(np.log(psi) + (x / psi))
        return -float(ll)

    init_params = np.array([omega_init, alpha_init, beta_init], dtype=np.float64)

    # Optimization via SLSQP with inequality constraint: 0.985 - (alpha + beta) >= 0
    constraints = [
        {"type": "ineq", "fun": lambda p: 0.985 - (p[1] + p[2])}
    ]

    try:
        res = optimize.minimize(
            neg_log_likelihood,
            init_params,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 200, "ftol": 1e-7},
        )
        if res.success:
            om_opt, al_opt, be_opt = res.x
            ll_opt = -float(res.fun)
        else:
            # Fallback to Nelder-Mead with penalty
            res_nm = optimize.minimize(
                neg_log_likelihood,
                init_params,
                method="Nelder-Mead",
                options={"maxiter": 300},
            )
            om_opt = float(np.clip(res_nm.x[0], bounds[0][0], bounds[0][1]))
            al_opt = float(np.clip(res_nm.x[1], bounds[1][0], bounds[1][1]))
            be_opt = float(np.clip(res_nm.x[2], bounds[2][0], bounds[2][1]))
            if al_opt + be_opt >= 0.985:
                scale = 0.980 / (al_opt + be_opt)
                al_opt *= scale
                be_opt *= scale
            ll_opt = -neg_log_likelihood(np.array([om_opt, al_opt, be_opt]))
    except Exception:
        om_opt, al_opt, be_opt = omega_init, alpha_init, beta_init
        ll_opt = -neg_log_likelihood(init_params)

    persistence = float(al_opt + be_opt)
    uncond_psi = float(om_opt / (1.0 - persistence)) if persistence < 0.99 else mean_x

    # Calculate final conditional duration
    psi_final = mean_x
    for i in range(1, n_pts):
        psi_final = om_opt + al_opt * x[i - 1] + be_opt * psi_final

    curr_hazard = float(1.0 / psi_final) if psi_final > 1e-6 else 0.0
    # Relative trading speed: ratio of sample mean duration to current conditional duration
    speed_ratio = float(mean_x / psi_final) if psi_final > 1e-6 else 1.0

    # Half-life of duration shocks
    if 0.0 < persistence < 0.999:
        half_life = float(math.log(0.5) / math.log(persistence))
    else:
        half_life = 0.0

    # Regime categorization
    if speed_ratio >= 1.40:
        regime = "ACCELERATED_TRADING_CLUSTERING"
    elif speed_ratio <= 0.65:
        regime = "DECELERATED_LIQUIDITY_DROUGHT"
    elif persistence >= 0.85:
        regime = "HIGH_DURATION_PERSISTENCE"
    else:
        regime = "BALANCED_ARRIVAL_DYNAMICS"

    payload = (
        f"{clean_ticker}|{n_pts}|{mean_x:.4f}|{om_opt:.6f}|{al_opt:.6f}|{be_opt:.6f}|"
        f"{persistence:.4f}|{uncond_psi:.4f}|{psi_final:.4f}|{curr_hazard:.4f}|"
        f"{speed_ratio:.4f}|{half_life:.2f}|{ll_opt:.2f}|{regime}|{derivation_mode}|{now_iso}"
    )
    sig = _compute_hmac(payload)

    return ACDReceipt(
        ticker=clean_ticker,
        sample_size=n_pts,
        mean_duration=mean_x,
        omega=float(om_opt),
        alpha=float(al_opt),
        beta=float(be_opt),
        persistence=persistence,
        unconditional_expected_duration=uncond_psi,
        current_conditional_duration=float(psi_final),
        current_hazard_rate=curr_hazard,
        trading_speed_ratio=speed_ratio,
        half_life_shocks=half_life,
        log_likelihood=ll_opt,
        regime=regime,
        derivation_mode=derivation_mode,
        hmac_signature=sig,
        timestamp=now_iso,
    )


def evaluate_acd_from_prices(
    prices: Sequence[float],
    volumes: Optional[Sequence[float]] = None,
    ticker: str = "SPY",
) -> ACDReceipt:
    """Evaluate ACD point process model from daily price and volume pacing."""
    p_arr = np.asarray(prices, dtype=np.float64)
    valid_p = np.isfinite(p_arr) & (p_arr > 0.0)
    p_clean = p_arr[valid_p]

    if len(p_clean) < 25:
        return _empty_receipt(ticker, datetime.now(timezone.utc).isoformat())

    # If volume is available, compute volume pacing durations (inverse turnover pacing)
    if volumes is not None and len(volumes) == len(p_arr):
        v_arr = np.asarray(volumes, dtype=np.float64)[valid_p]
        med_v = float(np.median(v_arr)) if float(np.median(v_arr)) > 0 else 1.0
        v_clean = np.maximum(v_arr, 1.0)
        durations = med_v / v_clean
        derivation_mode = "VOLUME_PACING_DURATIONS"
    else:
        # Price-revision duration proxy: inverse absolute return
        ret = np.abs(np.diff(p_clean) / p_clean[:-1])
        ret_med = float(np.median(ret)) if float(np.median(ret)) > 1e-6 else 0.01
        durations = ret_med / np.maximum(ret, 1e-4)
        derivation_mode = "PRICE_REVISION_DURATIONS"

    return calculate_acd_model(
        durations=durations,
        ticker=ticker,
        derivation_mode=derivation_mode,
    )


def _empty_receipt(ticker: str, timestamp_iso: str) -> ACDReceipt:
    empty_payload = f"{ticker}|0|0.0|0.0|0.0|0.0|0.0|{timestamp_iso}"
    sig = _compute_hmac(empty_payload)
    return ACDReceipt(
        ticker=ticker,
        sample_size=0,
        mean_duration=0.0,
        omega=0.0,
        alpha=0.0,
        beta=0.0,
        persistence=0.0,
        unconditional_expected_duration=0.0,
        current_conditional_duration=0.0,
        current_hazard_rate=0.0,
        trading_speed_ratio=1.0,
        half_life_shocks=0.0,
        log_likelihood=0.0,
        regime="INSUFFICIENT_DATA",
        derivation_mode="NONE",
        hmac_signature=sig,
        timestamp=timestamp_iso,
    )


def persist_acd_record(
    receipt: ACDReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Store ACD duration receipt in SQLite database with audit verification."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS acd_duration_receipts (
                ticker TEXT NOT NULL,
                sample_size INTEGER NOT NULL,
                mean_duration REAL NOT NULL,
                omega REAL NOT NULL,
                alpha REAL NOT NULL,
                beta REAL NOT NULL,
                persistence REAL NOT NULL,
                unconditional_expected_duration REAL NOT NULL,
                current_conditional_duration REAL NOT NULL,
                current_hazard_rate REAL NOT NULL,
                trading_speed_ratio REAL NOT NULL,
                half_life_shocks REAL NOT NULL,
                log_likelihood REAL NOT NULL,
                regime TEXT NOT NULL,
                derivation_mode TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                PRIMARY KEY (ticker, timestamp)
            )
            """
        )
        conn.execute(
            """
            INSERT OR REPLACE INTO acd_duration_receipts (
                ticker, sample_size, mean_duration, omega, alpha, beta,
                persistence, unconditional_expected_duration,
                current_conditional_duration, current_hazard_rate,
                trading_speed_ratio, half_life_shocks, log_likelihood,
                regime, derivation_mode, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.sample_size,
                receipt.mean_duration,
                receipt.omega,
                receipt.alpha,
                receipt.beta,
                receipt.persistence,
                receipt.unconditional_expected_duration,
                receipt.current_conditional_duration,
                receipt.current_hazard_rate,
                receipt.trading_speed_ratio,
                receipt.half_life_shocks,
                receipt.log_likelihood,
                receipt.regime,
                receipt.derivation_mode,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True
