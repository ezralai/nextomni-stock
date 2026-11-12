"""Roy (1952), Kataoka (1963) & Telser (1955) Safety-First Portfolio Selection Engine.

Implements all three classical shortfall-constrained portfolio criteria over long-only
asset weights (sum(w) == 1, w >= 0):
1. Roy (1952) Safety-First Ratio (SFR):
       max_w (mu_p - R_L) / sigma_p
   minimizing shortfall probability P(R_p <= R_L) and Chebyshev upper bound
2. Kataoka (1963) Maximal Safety Floor:
       max_w (mu_p - z_{1-alpha} * sigma_p)
   maximizing worst-case return threshold R_L at failure probability alpha
3. Telser (1955) Shortfall-Constrained Mean Maximization:
       max_w mu_p   s.t.   mu_p - z_{1-alpha} * sigma_p >= R_L
   detecting feasibility cleanly when no portfolio meets the shortfall floor
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
from scipy.stats import norm

TICKER_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,20}$")
DEFAULT_HMAC_KEY = "nextomni-stock-roy-1952-audit-key"


@dataclass(frozen=True)
class SafetyFirstReceipt:
    """Immutable receipt for Roy / Kataoka / Telser Safety-First optimization."""

    ticker: str
    n_assets: int
    threshold_return: float
    alpha_shortfall: float
    roy_sfr: float
    roy_mean: float
    roy_vol: float
    roy_shortfall_prob: float
    chebyshev_bound: float
    kataoka_floor: float
    kataoka_mean: float
    kataoka_vol: float
    telser_feasible: bool
    telser_mean: float
    telser_vol: float
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
    n_assets: int,
    threshold_return: float,
    alpha_shortfall: float,
    roy_sfr: float,
    kataoka_floor: float,
    telser_feasible: bool,
    telser_mean: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_assets}|{threshold_return:.6f}|{alpha_shortfall:.6f}|"
        f"{roy_sfr:.8f}|{kataoka_floor:.8f}|{int(telser_feasible)}|"
        f"{telser_mean:.8f}|{regime}|{timestamp}"
    )


def sign_safety_first_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_safety_first_receipt(payload)
    return hmac.compare_digest(expected, signature)


def build_synthetic_safety_first_inputs(
    n_assets: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate benchmark annualized expected returns and covariance matrix."""
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2")
    mu = np.linspace(0.05, 0.14, n_assets)
    vols = np.linspace(0.08, 0.22, n_assets)
    corr = np.full((n_assets, n_assets), 0.20, dtype=float)
    np.fill_diagonal(corr, 1.0)
    sigma = np.outer(vols, vols) * corr
    return mu, sigma


def solve_safety_first_portfolio(
    expected_returns: Sequence[float] | np.ndarray,
    cov_matrix: np.ndarray,
    threshold_return: float = 0.02,
    alpha_shortfall: float = 0.05,
    ticker: str = "PORTFOLIO",
) -> SafetyFirstReceipt:
    """Solve Roy (1952), Kataoka (1963), and Telser (1955) long-only portfolios."""
    clean_ticker = _validate_ticker(ticker)
    mu = np.asarray(expected_returns, dtype=float)
    sig = np.asarray(cov_matrix, dtype=float)

    if mu.ndim != 1 or len(mu) < 2:
        raise ValueError("expected_returns must be a 1D vector with at least 2 assets")
    n_assets = len(mu)
    if sig.shape != (n_assets, n_assets):
        raise ValueError(f"cov_matrix must have shape ({n_assets}, {n_assets})")
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(sig)):
        raise ValueError("expected_returns and cov_matrix must be finite")
    if not math.isfinite(threshold_return):
        raise ValueError("threshold_return must be finite")
    if not (0.0 < alpha_shortfall < 0.5):
        raise ValueError("alpha_shortfall must be strictly in (0, 0.5)")

    sig_sym = 0.5 * (sig + sig.T)
    if float(np.min(np.linalg.eigvalsh(sig_sym))) <= 0.0:
        raise ValueError("cov_matrix must be strictly positive definite")

    z_alpha = float(norm.ppf(1.0 - alpha_shortfall))
    w0 = np.full(n_assets, 1.0 / n_assets, dtype=float)
    bounds = [(0.0, 1.0) for _ in range(n_assets)]
    sum_con = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}

    def _port_stats(w: np.ndarray) -> tuple[float, float]:
        m = float(np.dot(w, mu))
        v = math.sqrt(max(float(w @ sig_sym @ w), 1e-16))
        return m, v

    # 1. Roy (1952): maximize (mu_p - R_L) / sigma_p
    def _roy_neg_sfr(w: np.ndarray) -> float:
        m, v = _port_stats(w)
        return -((m - threshold_return) / v)

    res_roy = minimize(
        _roy_neg_sfr,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_con],
        options={"ftol": 1e-11, "maxiter": 250},
    )
    w_roy = np.asarray(res_roy.x, dtype=float)
    roy_mean, roy_vol = _port_stats(w_roy)
    roy_sfr = float((roy_mean - threshold_return) / roy_vol)
    roy_shortfall_prob = float(norm.cdf((threshold_return - roy_mean) / roy_vol))
    chebyshev = float(min(1.0, (roy_vol * roy_vol) / max((roy_mean - threshold_return) ** 2, 1e-12))) if roy_mean > threshold_return else 1.0

    # 2. Kataoka (1963): maximize mu_p - z_{1-alpha} * sigma_p
    def _kataoka_neg(w: np.ndarray) -> float:
        m, v = _port_stats(w)
        return -(m - z_alpha * v)

    res_kat = minimize(
        _kataoka_neg,
        w0,
        method="SLSQP",
        bounds=bounds,
        constraints=[sum_con],
        options={"ftol": 1e-11, "maxiter": 250},
    )
    w_kat = np.asarray(res_kat.x, dtype=float)
    kat_mean, kat_vol = _port_stats(w_kat)
    kat_floor = float(kat_mean - z_alpha * kat_vol)

    # 3. Telser (1955): max mu_p s.t. mu_p - z_{1-alpha} * sigma_p >= R_L
    # Feasible iff Kataoka's maximal floor >= threshold_return - tol
    if kat_floor >= threshold_return - 1e-9:
        telser_cons = [
            sum_con,
            {
                "type": "ineq",
                "fun": lambda w: float(
                    np.dot(w, mu) - z_alpha * math.sqrt(max(float(w @ sig_sym @ w), 1e-16)) - threshold_return
                ),
            },
        ]
        res_tel = minimize(
            lambda w: -float(np.dot(w, mu)),
            w_kat,
            method="SLSQP",
            bounds=bounds,
            constraints=telser_cons,
            options={"ftol": 1e-11, "maxiter": 250},
        )
        w_tel = np.asarray(res_tel.x, dtype=float)
        tel_mean, tel_vol = _port_stats(w_tel)
        telser_feasible = bool(tel_mean - z_alpha * tel_vol >= threshold_return - 1e-5)
        if not telser_feasible:
            tel_mean, tel_vol = kat_mean, kat_vol
            telser_feasible = True
    else:
        telser_feasible = False
        tel_mean = kat_mean
        tel_vol = kat_vol

    if not telser_feasible:
        regime = "INFEASIBLE_THRESHOLD_BREACH"
    elif roy_sfr >= 0.75:
        regime = "TELSER_FEASIBLE_SAFE"
    else:
        regime = "ROY_DOMINANT_TANGENCY"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_assets=n_assets,
        threshold_return=float(threshold_return),
        alpha_shortfall=float(alpha_shortfall),
        roy_sfr=roy_sfr,
        kataoka_floor=kat_floor,
        telser_feasible=telser_feasible,
        telser_mean=tel_mean,
        regime=regime,
        timestamp=ts,
    )
    return SafetyFirstReceipt(
        ticker=clean_ticker,
        n_assets=int(n_assets),
        threshold_return=float(threshold_return),
        alpha_shortfall=float(alpha_shortfall),
        roy_sfr=roy_sfr,
        roy_mean=roy_mean,
        roy_vol=roy_vol,
        roy_shortfall_prob=roy_shortfall_prob,
        chebyshev_bound=chebyshev,
        kataoka_floor=kat_floor,
        kataoka_mean=kat_mean,
        kataoka_vol=kat_vol,
        telser_feasible=telser_feasible,
        telser_mean=tel_mean,
        telser_vol=tel_vol,
        regime=regime,
        hmac_signature=sign_safety_first_receipt(payload),
        timestamp=ts,
    )


def evaluate_safety_first_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    threshold_return: float = -0.10,
    alpha_shortfall: float = 0.05,
) -> SafetyFirstReceipt:
    """Derive a 2-asset (asset vs defensive sleeve) safety-first profile for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions for Safety-First optimization")

    log_ret = np.diff(np.log(arr))
    ann_mu = float(np.clip(np.mean(log_ret) * 252.0, -0.15, 0.35))
    ann_vol = float(np.clip(np.std(log_ret, ddof=1) * math.sqrt(252.0), 0.06, 0.60))
    mu = np.array([ann_mu, 0.04], dtype=float)
    cov = np.array(
        [
            [ann_vol ** 2, 0.1 * ann_vol * 0.05],
            [0.1 * ann_vol * 0.05, 0.05 ** 2],
        ],
        dtype=float,
    )
    return solve_safety_first_portfolio(
        expected_returns=mu,
        cov_matrix=cov,
        threshold_return=threshold_return,
        alpha_shortfall=alpha_shortfall,
        ticker=ticker,
    )


def persist_safety_first_record(
    receipt: SafetyFirstReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Roy/Kataoka/Telser Safety-First receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS roy_safety_first_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_assets INTEGER NOT NULL,
                threshold_return REAL NOT NULL,
                alpha_shortfall REAL NOT NULL,
                roy_sfr REAL NOT NULL,
                roy_mean REAL NOT NULL,
                roy_vol REAL NOT NULL,
                roy_shortfall_prob REAL NOT NULL,
                chebyshev_bound REAL NOT NULL,
                kataoka_floor REAL NOT NULL,
                kataoka_mean REAL NOT NULL,
                kataoka_vol REAL NOT NULL,
                telser_feasible INTEGER NOT NULL,
                telser_mean REAL NOT NULL,
                telser_vol REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO roy_safety_first_records (
                ticker, n_assets, threshold_return, alpha_shortfall,
                roy_sfr, roy_mean, roy_vol, roy_shortfall_prob, chebyshev_bound,
                kataoka_floor, kataoka_mean, kataoka_vol,
                telser_feasible, telser_mean, telser_vol,
                regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_assets,
                receipt.threshold_return,
                receipt.alpha_shortfall,
                receipt.roy_sfr,
                receipt.roy_mean,
                receipt.roy_vol,
                receipt.roy_shortfall_prob,
                receipt.chebyshev_bound,
                receipt.kataoka_floor,
                receipt.kataoka_mean,
                receipt.kataoka_vol,
                int(receipt.telser_feasible),
                receipt.telser_mean,
                receipt.telser_vol,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

