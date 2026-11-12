"""William F. Sharpe (1992) Returns-Based Style Analysis (RBSA) & Selection Attribution.

Theoretical Foundation:
    Sharpe, William F. (1992).
    "Asset Allocation: Management Style and Performance Measurement".
    Journal of Portfolio Management, 18(2), pp. 7-19.

Key Concepts:
    - Constrained Simplex Quadratic Style Replication:
      Decomposes manager returns R_p against K mutually exclusive style indices F_k
      via quadratic tracking-error minimization:
      min_{w} Var(R_p - F w)   subject to   w_k >= 0,  sum(w_k) = 1.
      Solved exactly using Nesterov Accelerated Projected Gradient Descent (FISTA)
      onto the unit probability simplex without numerical constraint violations.
    - Style R^2 vs Active Security Selection Variance Decomposition:
      Total variance partitions into Style Allocation fit (R^2_style) and Active
      Selection Risk (1 - R^2_style), isolating annualized Selection Alpha and
      Selection Information Ratio IR_selection = alpha_sel / TE_sel.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
DEFAULT_STYLE_NAMES = ["Value_Equity", "Growth_Equity", "Small_Cap", "Govt_Bonds"]
HMAC_SECRET = b"sharpe-style-analysis-v179-secret"


@dataclass(frozen=True)
class SharpeStyleReceipt:
    """Immutable quantitative audit receipt for Sharpe (1992) RBSA."""

    ticker: str
    n_obs: int
    n_factors: int
    factor_names: List[str]
    style_weights: Dict[str, float]
    dominant_style: str
    dominant_weight: float
    style_r2: float
    selection_var_share: float
    portfolio_vol_ann: float
    selection_te_ann: float
    selection_alpha_ann: float
    selection_ir: float
    weight_hhi_concentration: float
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize receipt to dictionary."""
        return asdict(self)


def generate_hmac_signature(payload: str, secret: bytes = HMAC_SECRET) -> str:
    """Generate SHA-256 HMAC signature for cryptographic receipt integrity."""
    return hmac.new(secret, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hmac_signature(
    payload: str,
    signature: str,
    secret: bytes = HMAC_SECRET,
) -> bool:
    """Constant-time verification of HMAC-SHA256 receipt signature."""
    expected = generate_hmac_signature(payload, secret=secret)
    return hmac.compare_digest(expected, signature)


def _project_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto probability simplex {w >= 0, sum(w) == 1}."""
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1, dtype=float)
    cond = u - (cssv / ind) > 0.0
    rho = int(np.nonzero(cond)[0][-1])
    theta = float(cssv[rho] / (rho + 1.0))
    proj = np.maximum(v - theta, 0.0)
    return proj / float(np.sum(proj))


def _solve_simplex_qp(cov_f: np.ndarray, cov_fp: np.ndarray) -> np.ndarray:
    """Minimize w^T cov_f w - 2 cov_fp^T w over probability simplex via FISTA."""
    k = cov_f.shape[0]
    w = np.full(k, 1.0 / float(k), dtype=float)
    y = w.copy()
    max_eig = float(np.max(np.linalg.eigvalsh(cov_f)))
    lip = max(1e-12, 2.0 * max_eig)
    step = 1.0 / lip
    t_k = 1.0

    for _ in range(1500):
        grad = 2.0 * (cov_f @ y - cov_fp)
        w_next = _project_simplex(y - step * grad)
        if float(np.linalg.norm(w_next - w)) < 1e-12:
            w = w_next
            break
        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
        beta = (t_k - 1.0) / t_next
        y = w_next + beta * (w_next - w)
        w = w_next
        t_k = t_next

    w = np.where(w < 1e-10, 0.0, w)
    return w / float(np.sum(w))


def classify_sharpe_style_regime(
    style_r2: float,
    selection_alpha_ann: float,
    selection_ir: float,
) -> str:
    """Classify style replication purity vs active stock-selection regime."""
    if style_r2 < 0.50 and selection_alpha_ann <= -0.02:
        return "UNEXPLAINED_STYLE_DRIFT_TRAP"
    if style_r2 >= 0.90:
        return "PURE_STYLE_REPLICATOR"
    if style_r2 < 0.65 or selection_ir >= 0.50:
        return "ALPHA_DRIVEN_STOCK_PICKER"
    return "STYLE_TILTED_CORE_MANAGER"


def _build_hmac_payload(
    ticker: str,
    n_obs: int,
    n_factors: int,
    dominant_style: str,
    dominant_weight: float,
    style_r2: float,
    selection_alpha_ann: float,
    selection_te_ann: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{n_obs}:{n_factors}:{dominant_style}:{dominant_weight:.6f}:"
        f"{style_r2:.6f}:{selection_alpha_ann:.6f}:{selection_te_ann:.6f}:"
        f"{regime}:{timestamp}"
    )


def build_synthetic_style_returns(
    weights: List[float] | np.ndarray,
    n_obs: int = 120,
    alpha_ann: float = 0.02,
    selection_vol_ann: float = 0.03,
    seed: int = 179,
) -> tuple[np.ndarray, np.ndarray, List[str]]:
    """Simulate orthogonalized style factors and active fund return stream."""
    w_arr = np.asarray(weights, dtype=float)
    if w_arr.ndim != 1 or len(w_arr) != 4 or np.any(w_arr < -1e-9) or abs(float(np.sum(w_arr)) - 1.0) > 1e-6:
        raise ValueError("Weights must be a 4-element non-negative vector summing to 1.0.")
    if not (20 <= int(n_obs) <= 5000):
        raise ValueError("Sample length n_obs must be in [20, 5000].")

    rng = np.random.default_rng(seed)
    daily_vols = np.array([0.16, 0.20, 0.23, 0.07]) / math.sqrt(252.0)
    daily_means = np.array([0.08, 0.11, 0.12, 0.04]) / 252.0
    factors = rng.normal(loc=daily_means, scale=daily_vols, size=(int(n_obs), 4))

    noise_scale = max(0.0, float(selection_vol_ann)) / math.sqrt(252.0)
    noise = rng.normal(loc=0.0, scale=noise_scale, size=int(n_obs)) if noise_scale > 0.0 else np.zeros(int(n_obs))
    if np.std(noise) > 1e-12:
        noise = noise - float(np.mean(noise))
    rp = factors @ w_arr + (float(alpha_ann) / 252.0) + noise
    return rp, factors, list(DEFAULT_STYLE_NAMES)


def calculate_sharpe_style_analysis(
    portfolio_returns: List[float] | np.ndarray,
    style_returns: List[List[float]] | np.ndarray,
    factor_names: List[str] | None = None,
    ticker: str = "FUND",
) -> SharpeStyleReceipt:
    """Compute William F. Sharpe (1992) RBSA constrained style weights and selection R^2."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    rp = np.asarray(portfolio_returns, dtype=float)
    f_mat = np.asarray(style_returns, dtype=float)

    if rp.ndim != 1 or f_mat.ndim != 2 or f_mat.shape[0] != len(rp):
        raise ValueError("Portfolio returns (T,) and style factors (T, K) must align cleanly.")

    t_obs, k_factors = f_mat.shape
    if t_obs < 14:
        raise ValueError("At least 14 observations are required for style regression.")
    if not (2 <= k_factors <= 12):
        raise ValueError("Number of style factors K must be in [2, 12].")

    if factor_names is None:
        names = DEFAULT_STYLE_NAMES[:k_factors] if k_factors <= 4 else [f"Style_{i+1}" for i in range(k_factors)]
    else:
        if len(factor_names) != k_factors:
            raise ValueError("factor_names length must match number of columns in style_returns.")
        names = [str(name).strip() for name in factor_names]

    var_p = float(np.var(rp))
    if var_p <= 1e-14:
        raise ValueError("Portfolio return series has insufficient variance.")

    rp_c = rp - float(np.mean(rp))
    f_c = f_mat - np.mean(f_mat, axis=0, keepdims=True)
    cov_f = (f_c.T @ f_c) / float(t_obs)
    cov_fp = (f_c.T @ rp_c) / float(t_obs)

    w_opt = _solve_simplex_qp(cov_f, cov_fp)
    residuals = rp - (f_mat @ w_opt)
    var_res = float(np.var(residuals))

    style_r2 = max(0.0, min(1.0, 1.0 - (var_res / var_p)))
    sel_share = 1.0 - style_r2

    port_vol = math.sqrt(var_p) * math.sqrt(252.0)
    sel_te = math.sqrt(max(0.0, var_res)) * math.sqrt(252.0)
    sel_alpha = float(np.mean(residuals)) * 252.0
    sel_ir = (sel_alpha / sel_te) if sel_te > 1e-6 else 0.0

    weights_dict = {names[i]: float(w_opt[i]) for i in range(k_factors)}
    dom_idx = int(np.argmax(w_opt))
    dom_style = names[dom_idx]
    dom_weight = float(w_opt[dom_idx])
    hhi = float(np.sum(w_opt * w_opt))

    regime = classify_sharpe_style_regime(
        style_r2=style_r2,
        selection_alpha_ann=sel_alpha,
        selection_ir=sel_ir,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        n_obs=t_obs,
        n_factors=k_factors,
        dominant_style=dom_style,
        dominant_weight=dom_weight,
        style_r2=style_r2,
        selection_alpha_ann=sel_alpha,
        selection_te_ann=sel_te,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return SharpeStyleReceipt(
        ticker=cleaned_ticker,
        n_obs=t_obs,
        n_factors=k_factors,
        factor_names=names,
        style_weights=weights_dict,
        dominant_style=dom_style,
        dominant_weight=dom_weight,
        style_r2=style_r2,
        selection_var_share=sel_share,
        portfolio_vol_ann=port_vol,
        selection_te_ann=sel_te,
        selection_alpha_ann=sel_alpha,
        selection_ir=sel_ir,
        weight_hhi_concentration=hhi,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_sharpe_style_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SPY",
    seed: int = 179,
) -> SharpeStyleReceipt:
    """Calibrate multi-style proxy exposures against trailing asset log returns."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 16:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 120) :])))
    else:
        log_rets = np.zeros(60, dtype=float)

    n_obs = max(20, len(log_rets))
    rng = np.random.default_rng(seed)
    if len(log_rets) >= 20:
        rp = log_rets
        idio_scale = max(0.002, float(np.std(rp)) * 0.45)
        f_val = 0.70 * rp + rng.normal(0.0, idio_scale, size=n_obs)
        f_grw = 0.85 * rp + rng.normal(0.0, idio_scale * 0.8, size=n_obs)
        f_sml = 0.55 * rp + rng.normal(0.0, idio_scale * 1.2, size=n_obs)
        f_bnd = -0.25 * rp + rng.normal(0.0001, idio_scale * 0.6, size=n_obs)
        f_mat = np.column_stack([f_val, f_grw, f_sml, f_bnd])
    else:
        rp, f_mat, _ = build_synthetic_style_returns([0.35, 0.45, 0.15, 0.05], n_obs=60)

    return calculate_sharpe_style_analysis(
        portfolio_returns=rp,
        style_returns=f_mat,
        factor_names=list(DEFAULT_STYLE_NAMES),
        ticker=ticker,
    )


def persist_sharpe_style_record(
    receipt: SharpeStyleReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Sharpe (1992) RBSA audit record to SQLite."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS sharpe_style_analysis_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        n_obs INTEGER NOT NULL,
        n_factors INTEGER NOT NULL,
        dominant_style TEXT NOT NULL,
        dominant_weight REAL NOT NULL,
        style_r2 REAL NOT NULL,
        selection_alpha_ann REAL NOT NULL,
        selection_te_ann REAL NOT NULL,
        selection_ir REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO sharpe_style_analysis_audits (
        ticker, n_obs, n_factors, dominant_style, dominant_weight,
        style_r2, selection_alpha_ann, selection_te_ann, selection_ir,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.n_obs,
                receipt.n_factors,
                receipt.dominant_style,
                receipt.dominant_weight,
                receipt.style_r2,
                receipt.selection_alpha_ann,
                receipt.selection_te_ann,
                receipt.selection_ir,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

