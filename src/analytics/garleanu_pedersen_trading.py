"""Gârleanu & Pedersen (2013) Dynamic Trading with Predictable Returns and Transaction Costs.

Implements Theorem 1 & 2 of Gârleanu & Pedersen (Journal of Finance, 2013):
- Discrete-time LQ Bellman / Riccati fixed-point solution for optimal trading speed:
      A_xx = Lambda - Lambda @ inv(gamma * Sigma + Lambda + disc * A_xx) @ Lambda
      M_rate = inv(gamma * Sigma + Lambda + disc * A_xx) @ (gamma * Sigma + disc * A_xx)
- Forward-looking Aim Portfolio weighting multi-speed decaying alpha signals:
      A_xf = Lambda @ inv(gamma * Sigma + Lambda + disc * A_xx) @ (B + disc * A_xf @ (I - Phi))
      aim_t = inv(gamma * Sigma + disc * A_xx) @ (B + disc * A_xf @ (I - Phi)) @ f_t
- Optimal dynamic execution rule:
      Delta x_t = M_rate @ (aim_t - x_{t-1})
- Comparison against the myopic Markowitz portfolio (Lambda -> 0 limit),
  turnover reduction ratio, fast-vs-slow alpha attenuation, and regime classification
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
DEFAULT_HMAC_KEY = "nextomni-stock-gp-2013-audit-key"


@dataclass(frozen=True)
class GarleanuPedersenReceipt:
    """Immutable receipt for Gârleanu & Pedersen (2013) dynamic trading policy."""

    ticker: str
    n_assets: int
    n_factors: int
    risk_aversion: float
    cost_bps: float
    mean_trade_rate: float
    myopic_norm: float
    aim_norm: float
    optimal_trade_norm: float
    turnover_reduction_pct: float
    fast_alpha_weight_ratio: float
    slow_alpha_weight_ratio: float
    riccati_residual: float
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
    n_factors: int,
    risk_aversion: float,
    cost_bps: float,
    mean_trade_rate: float,
    aim_norm: float,
    optimal_trade_norm: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_assets}|{n_factors}|{risk_aversion:.6f}|{cost_bps:.4f}|"
        f"{mean_trade_rate:.8f}|{aim_norm:.8f}|{optimal_trade_norm:.8f}|{regime}|{timestamp}"
    )


def sign_gp_receipt(payload: str) -> str:
    key = os.environ.get("NEXTOMNI_STOCK_HMAC_KEY", DEFAULT_HMAC_KEY).encode("utf-8")
    return hmac.new(key, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    expected = sign_gp_receipt(payload)
    return hmac.compare_digest(expected, signature)


def build_synthetic_gp_inputs(
    n_assets: int = 3,
    fast_rho: float = 0.35,
    slow_rho: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build well-conditioned covariance, loadings, factor decay, and current state."""
    if n_assets < 2:
        raise ValueError("n_assets must be >= 2")
    if not (0.0 < slow_rho < fast_rho < 1.0):
        raise ValueError("Decay speeds must satisfy 0 < slow_rho < fast_rho < 1")

    vols = np.linspace(0.012, 0.020, n_assets)
    corr = np.full((n_assets, n_assets), 0.25, dtype=float)
    np.fill_diagonal(corr, 1.0)
    sigma = np.outer(vols, vols) * corr

    # 2 factors: factor 0 = fast alpha, factor 1 = slow alpha
    b_loadings = np.zeros((n_assets, 2), dtype=float)
    b_loadings[:, 0] = np.linspace(0.0012, 0.0008, n_assets)
    b_loadings[:, 1] = np.linspace(0.0008, 0.0012, n_assets)

    phi = np.array([fast_rho, slow_rho], dtype=float)
    f_t = np.array([1.0, 1.0], dtype=float)
    x_prev = np.zeros(n_assets, dtype=float)
    return sigma, b_loadings, phi, f_t, x_prev


def solve_garleanu_pedersen(
    sigma: np.ndarray,
    b_loadings: np.ndarray,
    phi_decay: Sequence[float] | np.ndarray,
    f_t: Sequence[float] | np.ndarray,
    x_prev: Sequence[float] | np.ndarray,
    risk_aversion: float = 2.5,
    cost_bps: float = 15.0,
    discount: float = 0.999,
    ticker: str = "SPY",
) -> GarleanuPedersenReceipt:
    """Solve Gârleanu & Pedersen (2013) discrete-time dynamic portfolio policy."""
    clean_ticker = _validate_ticker(ticker)
    sig_mat = np.asarray(sigma, dtype=float)
    b_mat = np.asarray(b_loadings, dtype=float)
    phi = np.asarray(phi_decay, dtype=float)
    f_vec = np.asarray(f_t, dtype=float)
    x_0 = np.asarray(x_prev, dtype=float)

    if sig_mat.ndim != 2 or sig_mat.shape[0] != sig_mat.shape[1] or sig_mat.shape[0] < 2:
        raise ValueError("sigma must be a square N x N matrix with N >= 2")
    n_assets = sig_mat.shape[0]
    if b_mat.ndim != 2 or b_mat.shape[0] != n_assets or b_mat.shape[1] < 1:
        raise ValueError("b_loadings must have shape (N, K) with K >= 1")
    n_factors = b_mat.shape[1]
    if phi.shape != (n_factors,) or f_vec.shape != (n_factors,):
        raise ValueError(f"phi_decay and f_t must have length {n_factors}")
    if x_0.shape != (n_assets,):
        raise ValueError(f"x_prev must have length {n_assets}")
    if risk_aversion <= 0.0 or not math.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite and > 0")
    if cost_bps < 0.0 or not math.isfinite(cost_bps):
        raise ValueError("cost_bps must be finite and >= 0")
    if not (0.0 < discount < 1.0):
        raise ValueError("discount factor must be strictly in (0, 1)")
    if np.any(phi <= 0.0) or np.any(phi >= 1.0):
        raise ValueError("All phi_decay speeds must be strictly in (0, 1)")

    eigvals = np.linalg.eigvalsh(0.5 * (sig_mat + sig_mat.T))
    if np.min(eigvals) <= 0.0:
        raise ValueError("sigma must be strictly positive definite")

    # Transaction cost matrix Lambda proportional to diagonal risk + cost bps
    cost_scale = max(cost_bps, 0.0) / 10000.0
    if cost_scale == 0.0:
        # Exact zero-cost myopic Markowitz limit
        myopic = np.linalg.solve(risk_aversion * sig_mat, b_mat @ f_vec)
        delta_x = myopic - x_0
        myopic_norm = float(np.linalg.norm(myopic))
        trade_norm = float(np.linalg.norm(delta_x))
        ts = datetime.now(timezone.utc).isoformat()
        regime = "FAST_ALPHA_HIGH_TURNOVER"
        payload = _build_hmac_payload(
            ticker=clean_ticker,
            n_assets=n_assets,
            n_factors=n_factors,
            risk_aversion=risk_aversion,
            cost_bps=cost_bps,
            mean_trade_rate=1.0,
            aim_norm=myopic_norm,
            optimal_trade_norm=trade_norm,
            regime=regime,
            timestamp=ts,
        )
        return GarleanuPedersenReceipt(
            ticker=clean_ticker,
            n_assets=n_assets,
            n_factors=n_factors,
            risk_aversion=float(risk_aversion),
            cost_bps=0.0,
            mean_trade_rate=1.0,
            myopic_norm=myopic_norm,
            aim_norm=myopic_norm,
            optimal_trade_norm=trade_norm,
            turnover_reduction_pct=0.0,
            fast_alpha_weight_ratio=1.0,
            slow_alpha_weight_ratio=1.0,
            riccati_residual=0.0,
            regime=regime,
            hmac_signature=sign_gp_receipt(payload),
            timestamp=ts,
        )

    lam = np.diag(np.diag(sig_mat) * (cost_bps * 0.25) + cost_scale * 1e-3)
    gamma_sig = risk_aversion * sig_mat

    # Solve A_xx = Lambda - Lambda @ inv(gamma * Sigma + Lambda + discount * A_xx) @ Lambda
    a_xx = np.zeros_like(lam)
    riccati_res = 0.0
    for _ in range(400):
        denom = gamma_sig + lam + discount * a_xx
        inv_denom_lam = np.linalg.solve(denom, lam)
        a_next = lam - lam @ inv_denom_lam
        a_next = 0.5 * (a_next + a_next.T)
        riccati_res = float(np.max(np.abs(a_next - a_xx)))
        a_xx = a_next
        if riccati_res < 1e-13:
            break

    denom_final = gamma_sig + lam + discount * a_xx
    m_rate = np.linalg.solve(denom_final, gamma_sig + discount * a_xx)
    mean_trade_rate = float(np.mean(np.diag(m_rate)))

    # Solve A_xf = Lambda @ inv(denom_final) @ (B + discount * A_xf @ (I - Phi))
    i_minus_phi = np.diag(1.0 - phi)
    gain_xf = lam @ np.linalg.inv(denom_final)
    a_xf = np.zeros((n_assets, n_factors), dtype=float)
    for _ in range(400):
        a_xf_next = gain_xf @ (b_mat + discount * (a_xf @ i_minus_phi))
        if float(np.max(np.abs(a_xf_next - a_xf))) < 1e-13:
            a_xf = a_xf_next
            break
        a_xf = a_xf_next

    aim_gain = np.linalg.solve(gamma_sig + discount * a_xx, b_mat + discount * (a_xf @ i_minus_phi))
    aim_t = aim_gain @ f_vec
    myopic_gain = np.linalg.solve(gamma_sig, b_mat)
    myopic_t = myopic_gain @ f_vec

    delta_x = m_rate @ (aim_t - x_0)
    naive_trade = myopic_t - x_0
    myopic_norm = float(np.linalg.norm(myopic_t))
    aim_norm = float(np.linalg.norm(aim_t))
    optimal_trade_norm = float(np.linalg.norm(delta_x))
    naive_trade_norm = float(np.linalg.norm(naive_trade))
    turnover_red = (
        float((1.0 - optimal_trade_norm / naive_trade_norm) * 100.0)
        if naive_trade_norm > 1e-12
        else 0.0
    )

    # Attenuation ratios of aim loading vs myopic loading for fastest & slowest factors
    fast_idx = int(np.argmax(phi))
    slow_idx = int(np.argmin(phi))
    fast_myopic_n = float(np.linalg.norm(myopic_gain[:, fast_idx]))
    slow_myopic_n = float(np.linalg.norm(myopic_gain[:, slow_idx]))
    fast_ratio = (
        float(np.linalg.norm(aim_gain[:, fast_idx]) / fast_myopic_n)
        if fast_myopic_n > 1e-12
        else 1.0
    )
    slow_ratio = (
        float(np.linalg.norm(aim_gain[:, slow_idx]) / slow_myopic_n)
        if slow_myopic_n > 1e-12
        else 1.0
    )

    if mean_trade_rate >= 0.70:
        regime = "FAST_ALPHA_HIGH_TURNOVER"
    elif mean_trade_rate >= 0.30:
        regime = "BALANCED_AIM_TRACKING"
    elif mean_trade_rate >= 0.08:
        regime = "COST_CONSTRAINED_SMOOTH"
    else:
        regime = "STATIC_HOLD_REGIME"

    ts = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_assets=n_assets,
        n_factors=n_factors,
        risk_aversion=risk_aversion,
        cost_bps=cost_bps,
        mean_trade_rate=mean_trade_rate,
        aim_norm=aim_norm,
        optimal_trade_norm=optimal_trade_norm,
        regime=regime,
        timestamp=ts,
    )
    return GarleanuPedersenReceipt(
        ticker=clean_ticker,
        n_assets=n_assets,
        n_factors=n_factors,
        risk_aversion=float(risk_aversion),
        cost_bps=float(cost_bps),
        mean_trade_rate=mean_trade_rate,
        myopic_norm=myopic_norm,
        aim_norm=aim_norm,
        optimal_trade_norm=optimal_trade_norm,
        turnover_reduction_pct=turnover_red,
        fast_alpha_weight_ratio=fast_ratio,
        slow_alpha_weight_ratio=slow_ratio,
        riccati_residual=riccati_res,
        regime=regime,
        hmac_signature=sign_gp_receipt(payload),
        timestamp=ts,
    )


def evaluate_gp_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
    cost_bps: float = 15.0,
) -> GarleanuPedersenReceipt:
    """Calibrate a 2-asset dynamic trading profile from realized price volatility for Tab 2."""
    arr = np.asarray(prices, dtype=float)
    arr = arr[np.isfinite(arr) & (arr > 0.0)]
    if len(arr) < 25:
        raise ValueError("Requires at least 25 price sessions to calibrate Gârleanu-Pedersen policy")

    log_ret = np.diff(np.log(arr))
    daily_vol = float(np.clip(np.std(log_ret, ddof=1), 0.005, 0.05))
    sigma = np.array(
        [
            [daily_vol ** 2, 0.3 * (daily_vol ** 2)],
            [0.3 * (daily_vol ** 2), (1.15 * daily_vol) ** 2],
        ],
        dtype=float,
    )
    momentum = float(np.clip((arr[-1] / arr[-10]) - 1.0, -0.15, 0.15)) if len(arr) >= 10 else 0.01
    b_mat = np.array([[0.0012, 0.0008], [0.0007, 0.0011]], dtype=float)
    phi = np.array([0.35, 0.04], dtype=float)
    f_t = np.array([1.0 + 2.0 * momentum, 1.0 - momentum], dtype=float)
    x_prev = np.zeros(2, dtype=float)
    return solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        risk_aversion=2.5,
        cost_bps=cost_bps,
        ticker=ticker,
    )


def persist_gp_record(
    receipt: GarleanuPedersenReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Gârleanu-Pedersen dynamic trading policy receipt to SQLite."""
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(db_file)) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gp_dynamic_trading_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                n_assets INTEGER NOT NULL,
                n_factors INTEGER NOT NULL,
                risk_aversion REAL NOT NULL,
                cost_bps REAL NOT NULL,
                mean_trade_rate REAL NOT NULL,
                myopic_norm REAL NOT NULL,
                aim_norm REAL NOT NULL,
                optimal_trade_norm REAL NOT NULL,
                turnover_reduction_pct REAL NOT NULL,
                fast_alpha_weight_ratio REAL NOT NULL,
                slow_alpha_weight_ratio REAL NOT NULL,
                riccati_residual REAL NOT NULL,
                regime TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO gp_dynamic_trading_records (
                ticker, n_assets, n_factors, risk_aversion, cost_bps,
                mean_trade_rate, myopic_norm, aim_norm, optimal_trade_norm,
                turnover_reduction_pct, fast_alpha_weight_ratio, slow_alpha_weight_ratio,
                riccati_residual, regime, hmac_signature, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.n_assets,
                receipt.n_factors,
                receipt.risk_aversion,
                receipt.cost_bps,
                receipt.mean_trade_rate,
                receipt.myopic_norm,
                receipt.aim_norm,
                receipt.optimal_trade_norm,
                receipt.turnover_reduction_pct,
                receipt.fast_alpha_weight_ratio,
                receipt.slow_alpha_weight_ratio,
                receipt.riccati_residual,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

