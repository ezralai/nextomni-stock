"""Yves Choueifaty & Yves Coignard (2008) Maximum Diversification Portfolio Engine.

Theoretical Foundation:
    Choueifaty, Yves, and Coignard, Yves (2008).
    "Toward Maximum Diversification".
    The Journal of Portfolio Management, 35(1), pp. 40-51.

Key Concepts:
    - Diversification Ratio (DR):
      DR(w) = (w' sigma) / sqrt(w' Sigma w) >= 1, measuring the ratio of weighted
      average standalone asset volatilities to total portfolio volatility without
      requiring fragile expected-return forecasts.
    - Exact Convex Minimum-Correlation Reduction:
      Under long-only constraints (w >= 0, 1'w = 1), maximizing DR(w) transforms via
      y = (D w) / (1' D w) into minimizing 0.5 * y' C y on the probability simplex,
      where C = D^{-1} Sigma D^{-1} is the correlation matrix.
    - Equal-Correlation Optimality & Effective Independent Bets:
      Every asset held with strictly positive weight (w_i > 0) in the Maximum
      Diversification Portfolio (MDP) shares the exact same correlation 1 / DR(w*)
      with the MDP, while DR(w*)^2 quantifies effective uncorrelated risk bets.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"choueifaty-diversification-v186-secret"


@dataclass(frozen=True)
class DiversificationReceipt:
    """Immutable quantitative audit receipt for Choueifaty & Coignard (2008) MDP."""

    ticker: str
    n_assets: int
    n_active_holdings: int
    dr_mdp: float
    dr_equal_weight: float
    dr_min_variance: float
    effective_independent_bets: float
    diversification_benefit_pct: float
    mdp_volatility: float
    equal_weight_volatility: float
    active_correlation_spread: float
    max_weight: float
    weights: List[float]
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def _validate_ticker(ticker: str) -> str:
    cleaned = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned):
        raise ValueError(
            f"Invalid ticker format '{ticker}'. Expected 1-6 alphanumeric chars plus optional '.XX' suffix."
        )
    return cleaned


def _build_hmac_payload(
    *,
    ticker: str,
    n_assets: int,
    dr_mdp: float,
    dr_equal_weight: float,
    effective_independent_bets: float,
    mdp_volatility: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_assets}|{dr_mdp:.6f}|{dr_equal_weight:.6f}|"
        f"{effective_independent_bets:.6f}|{mdp_volatility:.6f}|{regime}|{timestamp}"
    )


def sign_diversification_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for Choueifaty MDP audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of Choueifaty MDP audit HMAC signature."""
    expected = sign_diversification_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def _project_probability_simplex(v: np.ndarray) -> np.ndarray:
    """Exact Euclidean projection onto the unit probability simplex {y >= 0, sum(y) = 1}."""
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, n + 1, dtype=float)
    cond = u - cssv / ind > 0.0
    rho = int(np.nonzero(cond)[0][-1]) + 1
    theta = cssv[rho - 1] / float(rho)
    w = np.maximum(v - theta, 0.0)
    return w / float(np.sum(w))


def _solve_simplex_qp(quad_mat: np.ndarray, max_iter: int = 2500, tol: float = 1e-12) -> np.ndarray:
    """Minimize 0.5 * y' Q y over unit simplex using FISTA accelerated projected gradient."""
    n = quad_mat.shape[0]
    eigvals = np.linalg.eigvalsh(quad_mat)
    lipschitz = max(float(eigvals[-1]), 1e-8)
    step = 1.0 / lipschitz

    y = np.full(n, 1.0 / float(n), dtype=float)
    z = y.copy()
    t = 1.0

    for _ in range(max_iter):
        grad = quad_mat @ z
        y_next = _project_probability_simplex(z - step * grad)
        if float(np.linalg.norm(y_next - y)) <= tol:
            y = y_next
            break
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        z = y_next + ((t - 1.0) / t_next) * (y_next - y)
        y = y_next
        t = t_next

    return y


def _diversification_ratio(weights: np.ndarray, vols: np.ndarray, cov: np.ndarray) -> float:
    port_vol = math.sqrt(max(1e-18, float(weights @ cov @ weights)))
    return float(float(weights @ vols) / port_vol)


def build_synthetic_correlation_cov(
    n_assets: int = 5,
    base_corr: float = 0.30,
    vols: Sequence[float] | None = None,
) -> np.ndarray:
    """Construct symmetric PD covariance matrix with controlled pairwise correlation."""
    n = int(n_assets)
    if n < 2 or n > 50:
        raise ValueError("n_assets must be within [2, 50].")
    rho = float(base_corr)
    if rho <= -1.0 / float(n - 1) or rho >= 0.99:
        raise ValueError(f"base_corr={base_corr} outside valid PD bounds for n={n}.")

    if vols is None:
        vol_arr = np.linspace(0.14, 0.26, n)
    else:
        vol_arr = np.asarray(vols, dtype=float)
        if len(vol_arr) != n or np.any(vol_arr <= 0.0):
            raise ValueError("vols must match n_assets with strictly positive entries.")

    corr = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    cov = np.outer(vol_arr, vol_arr) * corr
    return cov


def calculate_maximum_diversification(
    covariance_matrix: np.ndarray | Sequence[Sequence[float]],
    ticker: str = "SPY",
) -> DiversificationReceipt:
    """Compute Choueifaty & Coignard (2008) long-only Maximum Diversification Portfolio."""
    clean_ticker = _validate_ticker(ticker)
    cov = np.asarray(covariance_matrix, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("covariance_matrix must be a square (N, N) array with N >= 2.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance_matrix must contain only finite floats.")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("covariance_matrix must be symmetric.")

    cov_sym = 0.5 * (cov + cov.T)
    min_eig = float(np.min(np.linalg.eigvalsh(cov_sym)))
    if min_eig <= 1e-12:
        raise ValueError("covariance_matrix must be strictly positive definite (min eigenvalue > 1e-12).")

    n_assets = cov_sym.shape[0]
    vols = np.sqrt(np.diag(cov_sym))
    inv_d = 1.0 / vols
    corr = cov_sym * np.outer(inv_d, inv_d)
    corr = 0.5 * (corr + corr.T)

    # Solve exact minimum-correlation QP on unit simplex -> recover MDP weights
    y_opt = _solve_simplex_qp(corr)
    raw_w = inv_d * y_opt
    w_mdp = raw_w / float(np.sum(raw_w))

    # Global Minimum Variance (long-only) and Equal Weight baselines
    w_gmv = _solve_simplex_qp(cov_sym)
    w_ew = np.full(n_assets, 1.0 / float(n_assets), dtype=float)

    dr_mdp = _diversification_ratio(w_mdp, vols, cov_sym)
    dr_ew = _diversification_ratio(w_ew, vols, cov_sym)
    dr_gmv = _diversification_ratio(w_gmv, vols, cov_sym)

    mdp_vol = float(math.sqrt(max(1e-18, float(w_mdp @ cov_sym @ w_mdp))))
    ew_vol = float(math.sqrt(max(1e-18, float(w_ew @ cov_sym @ w_ew))))

    # Choueifaty-Coignard equal-correlation theorem across held positions
    asset_mdp_corrs = (cov_sym @ w_mdp) / (vols * mdp_vol)
    active_mask = w_mdp > 1e-4
    n_active = int(np.sum(active_mask))
    if n_active >= 2:
        active_corrs = asset_mdp_corrs[active_mask]
        corr_spread = float(np.max(active_corrs) - np.min(active_corrs))
    else:
        corr_spread = 0.0

    eff_bets = float(dr_mdp * dr_mdp)
    div_benefit = float(max(0.0, (dr_mdp - 1.0) / max(dr_mdp, 1e-12)))
    max_w = float(np.max(w_mdp))

    if dr_mdp >= 1.50:
        regime = "HIGH_DIVERSIFICATION_LEVERAGE"
    elif dr_mdp >= 1.20:
        regime = "MODERATE_MULTI_BET_DISPERSION"
    else:
        regime = "CONCENTRATED_SINGLE_FACTOR_RISK"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_assets=n_assets,
        dr_mdp=dr_mdp,
        dr_equal_weight=dr_ew,
        effective_independent_bets=eff_bets,
        mdp_volatility=mdp_vol,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_diversification_receipt(payload)

    return DiversificationReceipt(
        ticker=clean_ticker,
        n_assets=n_assets,
        n_active_holdings=n_active,
        dr_mdp=dr_mdp,
        dr_equal_weight=dr_ew,
        dr_min_variance=dr_gmv,
        effective_independent_bets=eff_bets,
        diversification_benefit_pct=div_benefit,
        mdp_volatility=mdp_vol,
        equal_weight_volatility=ew_vol,
        active_correlation_spread=corr_spread,
        max_weight=max_w,
        weights=[float(x) for x in w_mdp],
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_diversification_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
) -> DiversificationReceipt:
    """Calibrate multi-sleeve proxy covariance matrix from trailing asset volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 30:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 180) :])))
        base_vol = max(0.05, float(np.std(log_rets, ddof=1) * math.sqrt(252.0)))
    else:
        base_vol = 0.20

    vols = [base_vol * m for m in (0.75, 0.90, 1.00, 1.15, 1.30)]
    cov = build_synthetic_correlation_cov(n_assets=5, base_corr=0.35, vols=vols)
    return calculate_maximum_diversification(cov, ticker=ticker)


def persist_diversification_record(
    receipt: DiversificationReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Choueifaty & Coignard (2008) Maximum Diversification audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS choueifaty_dr_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_assets INTEGER NOT NULL,
        n_active_holdings INTEGER NOT NULL,
        dr_mdp REAL NOT NULL,
        dr_equal_weight REAL NOT NULL,
        effective_independent_bets REAL NOT NULL,
        mdp_volatility REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO choueifaty_dr_audits (
        ticker, n_assets, n_active_holdings, dr_mdp, dr_equal_weight,
        effective_independent_bets, mdp_volatility, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_assets,
                receipt.n_active_holdings,
                receipt.dr_mdp,
                receipt.dr_equal_weight,
                receipt.effective_independent_bets,
                receipt.mdp_volatility,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

