"""Maillard, Roncalli & Teiletche (2010) & Spinu (2013) Equal Risk Contribution Engine.

Theoretical Foundation:
    Maillard, Sebastien, Roncalli, Thierry, and Teiletche, Jerome (2010).
    "The Properties of Equally Weighted Risk Contribution Portfolios".
    The Journal of Portfolio Management, 36(4), pp. 60-70.

    Spinu, Florin (2013).
    "An Algorithm for Computing Risk Parity Weights".
    SSRN Working Paper 2297383.

Key Concepts:
    - Euler Risk-Contribution Decomposition:
      Total volatility sigma_p(w) = sqrt(w' Sigma w) decomposes additively into
      asset risk contributions RC_i = w_i * (Sigma w)_i / sigma_p(w).
      General risk budgeting enforces RC_i / sigma_p(w) = b_i for target budgets b > 0
      (with Equal Risk Contribution / Risk Parity setting b_i = 1/N).
    - Spinu (2013) Strictly Convex Log-Barrier Reformulation:
      Minimizing f(x) = 0.5 * x' Sigma x - sum_i b_i * ln(x_i) over x > 0 via
      damped Newton-Raphson yields unique global optimum w* = x* / sum(x*) with
      quadratic convergence and zero solver clipping warnings.
    - Volatility Ordering Theorem:
      In positive-definite asset universes, sigma_GMV <= sigma_ERC <= sigma_EW.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"maillard-erc-portfolio-v189-secret"


@dataclass(frozen=True)
class RiskBudgetReceipt:
    """Immutable quantitative audit receipt for MRT (2010) / Spinu (2013) Risk Parity."""

    ticker: str
    n_assets: int
    erc_volatility: float
    equal_weight_volatility: float
    inverse_vol_volatility: float
    max_risk_budget_error: float
    diversification_ratio: float
    effective_risk_bets: float
    weight_herfindahl: float
    weights: List[float]
    risk_shares: List[float]
    target_budgets: List[float]
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
    erc_volatility: float,
    equal_weight_volatility: float,
    max_risk_budget_error: float,
    effective_risk_bets: float,
    regime: str,
    timestamp: str,
) -> str:
    return (
        f"{ticker}|{n_assets}|{erc_volatility:.6f}|{equal_weight_volatility:.6f}|"
        f"{max_risk_budget_error:.8f}|{effective_risk_bets:.6f}|{regime}|{timestamp}"
    )


def sign_erc_receipt(payload: str) -> str:
    """Compute HMAC-SHA256 hexadecimal signature for MRT ERC audit payload."""
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Constant-time verification of MRT ERC audit HMAC signature."""
    expected = sign_erc_receipt(payload)
    return hmac.compare_digest(expected, str(signature))


def _solve_spinu_newton(cov: np.ndarray, budgets: np.ndarray, max_iter: int = 80) -> np.ndarray:
    """Solve 0.5 * x' Sigma x - sum b_i ln(x_i) over x > 0 via damped Newton-Raphson."""
    vols = np.sqrt(np.diag(cov))
    x = np.sqrt(budgets) / np.maximum(vols, 1e-8)

    def obj(vec: np.ndarray) -> float:
        return float(0.5 * float(vec @ cov @ vec) - float(np.sum(budgets * np.log(vec))))

    for _ in range(max_iter):
        grad = cov @ x - budgets / x
        if float(np.max(np.abs(grad))) <= 1e-12:
            break
        hess = cov + np.diag(budgets / (x * x))
        step_dir = -np.linalg.solve(hess, grad)

        # Preserve strict x > 0 domain
        neg_mask = step_dir < 0.0
        if np.any(neg_mask):
            alpha_max = float(np.min(-0.95 * x[neg_mask] / step_dir[neg_mask]))
            alpha = min(1.0, alpha_max)
        else:
            alpha = 1.0

        f0 = obj(x)
        slope = float(grad @ step_dir)
        for _ in range(40):
            x_cand = x + alpha * step_dir
            if np.all(x_cand > 0.0) and obj(x_cand) <= f0 + 1e-4 * alpha * slope:
                x = x_cand
                break
            alpha *= 0.5
        else:
            break

    w = x / float(np.sum(x))
    return w


def build_synthetic_erc_cov(
    vols: Sequence[float] = (0.10, 0.16, 0.24, 0.32),
    base_corr: float = 0.25,
) -> np.ndarray:
    """Build heterogeneous-volatility symmetric PD covariance matrix for ERC testing."""
    vol_arr = np.asarray(vols, dtype=float)
    n = len(vol_arr)
    if n < 2 or np.any(vol_arr <= 0.0):
        raise ValueError("vols must contain at least 2 strictly positive entries.")
    rho = float(base_corr)
    if rho <= -1.0 / float(n - 1) or rho >= 0.99:
        raise ValueError(f"base_corr={base_corr} outside valid PD bounds for n={n}.")

    corr = np.full((n, n), rho, dtype=float)
    np.fill_diagonal(corr, 1.0)
    return np.outer(vol_arr, vol_arr) * corr


def calculate_erc_portfolio(
    covariance_matrix: np.ndarray | Sequence[Sequence[float]],
    risk_budgets: Optional[Sequence[float]] = None,
    ticker: str = "SPY",
) -> RiskBudgetReceipt:
    """Compute Maillard-Roncalli-Teiletche (2010) Equal Risk / Budgeted portfolio."""
    clean_ticker = _validate_ticker(ticker)
    cov = np.asarray(covariance_matrix, dtype=float)

    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] < 2:
        raise ValueError("covariance_matrix must be a square (N, N) matrix with N >= 2.")
    if not np.all(np.isfinite(cov)):
        raise ValueError("covariance_matrix must contain only finite floats.")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("covariance_matrix must be symmetric.")

    cov_sym = 0.5 * (cov + cov.T)
    if float(np.min(np.linalg.eigvalsh(cov_sym))) <= 1e-12:
        raise ValueError("covariance_matrix must be strictly positive definite.")

    n_assets = cov_sym.shape[0]
    if risk_budgets is None:
        budgets = np.full(n_assets, 1.0 / float(n_assets), dtype=float)
        is_equal_budget = True
    else:
        b_arr = np.asarray(risk_budgets, dtype=float)
        if b_arr.shape != (n_assets,) or not np.all(np.isfinite(b_arr)) or np.any(b_arr <= 0.0):
            raise ValueError("risk_budgets must be strictly positive floats matching N assets.")
        budgets = b_arr / float(np.sum(b_arr))
        is_equal_budget = bool(np.allclose(budgets, 1.0 / float(n_assets), atol=1e-4))

    w_erc = _solve_spinu_newton(cov_sym, budgets)
    erc_var = float(w_erc @ cov_sym @ w_erc)
    erc_vol = float(math.sqrt(max(1e-18, erc_var)))

    # Euler risk shares
    rc = w_erc * (cov_sym @ w_erc) / max(erc_vol, 1e-12)
    risk_shares = rc / float(np.sum(rc))
    max_err = float(np.max(np.abs(risk_shares - budgets)))

    # Baselines: Equal-Weight (1/N) and Naive Inverse-Volatility
    w_ew = np.full(n_assets, 1.0 / float(n_assets), dtype=float)
    ew_vol = float(math.sqrt(max(1e-18, float(w_ew @ cov_sym @ w_ew))))

    vols = np.sqrt(np.diag(cov_sym))
    inv_v = 1.0 / vols
    w_iv = inv_v / float(np.sum(inv_v))
    iv_vol = float(math.sqrt(max(1e-18, float(w_iv @ cov_sym @ w_iv))))

    div_ratio = float(float(w_erc @ vols) / max(erc_vol, 1e-12))
    eff_risk_bets = float(1.0 / max(float(np.sum(risk_shares * risk_shares)), 1e-12))
    weight_hhi = float(np.sum(w_erc * w_erc))

    if max_err > 1e-4:
        regime = "RISK_CONCENTRATION_WARNING"
    elif is_equal_budget:
        regime = "EXACT_EQUAL_RISK_PARITY"
    else:
        regime = "CUSTOM_RISK_BUDGET_CONVERGED"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = _build_hmac_payload(
        ticker=clean_ticker,
        n_assets=n_assets,
        erc_volatility=erc_vol,
        equal_weight_volatility=ew_vol,
        max_risk_budget_error=max_err,
        effective_risk_bets=eff_risk_bets,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = sign_erc_receipt(payload)

    return RiskBudgetReceipt(
        ticker=clean_ticker,
        n_assets=n_assets,
        erc_volatility=erc_vol,
        equal_weight_volatility=ew_vol,
        inverse_vol_volatility=iv_vol,
        max_risk_budget_error=max_err,
        diversification_ratio=div_ratio,
        effective_risk_bets=eff_risk_bets,
        weight_herfindahl=weight_hhi,
        weights=[float(x) for x in w_erc],
        risk_shares=[float(x) for x in risk_shares],
        target_budgets=[float(x) for x in budgets],
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_erc_profile(
    prices: Sequence[float] | np.ndarray,
    ticker: str = "SPY",
) -> RiskBudgetReceipt:
    """Calibrate multi-sleeve ERC proxy covariance from trailing asset volatility."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 25:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 180) :])))
        base_vol = max(0.05, float(np.std(log_rets, ddof=1) * math.sqrt(252.0)))
    else:
        base_vol = 0.20

    vols = [base_vol * m for m in (0.60, 0.85, 1.15, 1.50)]
    cov = build_synthetic_erc_cov(vols=vols, base_corr=0.25)
    return calculate_erc_portfolio(cov, ticker=ticker)


def persist_erc_record(
    receipt: RiskBudgetReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Maillard-Roncalli-Teiletche (2010) ERC audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS maillard_erc_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_assets INTEGER NOT NULL,
        erc_volatility REAL NOT NULL,
        equal_weight_volatility REAL NOT NULL,
        max_risk_budget_error REAL NOT NULL,
        effective_risk_bets REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO maillard_erc_audits (
        ticker, n_assets, erc_volatility, equal_weight_volatility,
        max_risk_budget_error, effective_risk_bets, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_assets,
                receipt.erc_volatility,
                receipt.equal_weight_volatility,
                receipt.max_risk_budget_error,
                receipt.effective_risk_bets,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

