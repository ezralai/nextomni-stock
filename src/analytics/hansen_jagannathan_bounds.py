"""Lars Peter Hansen & Ravi Jagannathan (1991, 1997) SDF Volatility Bounds & Pricing Kernel Distance Engine.

Theoretical Foundation:
    1. Hansen, Lars Peter, & Jagannathan, Ravi (1991). "Implications of Security Market Data
       for Models of Dynamic Economies". Journal of Political Economy, 99(2), pp. 225-262.
       Nobel-Prize winning duality establishing the non-parametric lower bound on the
       volatility of any admissible Stochastic Discount Factor (SDF).

    2. Hansen, Lars Peter, & Jagannathan, Ravi (1997). "Assessing Specification Errors in
       Stochastic Discount Factor Models". The Journal of Finance, 52(2), pp. 557-590.
       Introduces the HJ-Distance (delta_HJ) measuring the minimum L^2 distance between a
       candidate pricing kernel and the set of arbitrage-free pricing kernels.

Key Concepts:
    - Hansen-Jagannathan Sharpe Ratio / Volatility Bound:
      For any valid pricing kernel m pricing test assets R with excess return mean mu_e and
      covariance Sigma:
          sigma(m) / E[m] >= sqrt(mu_e^T Sigma^{-1} mu_e) = SR_max
      Violating this inequality proves the candidate SDF cannot explain observed equity premia.

    - Minimum-Variance Admissible SDF Projection (m*):
      m* = R * (E[R R^T]^{-1} 1), attaining exact pricing E[m* R] = 1 with minimal variance.

    - Hansen-Jagannathan Specification Distance (delta_HJ):
      delta_HJ = sqrt(g^T E[R R^T]^{-1} g), where g = E[m R] - 1 are pricing errors.
      Equals the maximum annualized pricing error per unit of portfolio payoff norm.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"hansen-jagannathan-sdf-bounds-v137-secret"


@dataclass(frozen=True)
class HansenJagannathanReceipt:
    """Immutable quantitative audit receipt for Hansen-Jagannathan (1991, 1997) SDF diagnostics."""

    ticker: str
    num_observations: int
    num_assets: int
    risk_free_gross: float
    max_sharpe_period: float
    max_sharpe_annualized: float
    candidate_sdf_mean: float
    candidate_sdf_std: float
    hj_volatility_bound: float
    hj_dispersion_ratio: float      # sigma(m) / E[m]
    hj_bound_satisfied: bool
    hj_distance: float              # delta_HJ = sqrt(g^T G_2^{-1} g)
    mean_abs_pricing_error: float   # mean |E[m R] - 1|
    implied_min_risk_aversion: float# SR_max / sigma_consumption
    regime: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        return asdict(self)


def generate_hmac_signature(payload_str: str) -> str:
    """Generate SHA256 HMAC for cryptographic receipt sealing."""
    return hmac.new(
        HMAC_SECRET,
        payload_str.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def verify_hmac_signature(payload_str: str, signature: str) -> bool:
    """Verify cryptographic receipt signature."""
    expected = generate_hmac_signature(payload_str)
    return hmac.compare_digest(expected, signature)


def _safe_invert(mat: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    """Invert symmetric positive semi-definite matrix with ridge fallback."""
    n = mat.shape[0]
    try:
        cond = float(np.linalg.cond(mat))
        if math.isfinite(cond) and cond < 1e7:
            return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        pass
    return np.linalg.pinv(mat + ridge * np.eye(n))


def classify_hj_regime(
    hj_bound_satisfied: bool,
    hj_distance: float,
    max_sharpe_annualized: float,
) -> str:
    """Classify the Hansen-Jagannathan pricing kernel admissibility regime."""
    if not hj_bound_satisfied:
        return "EQUITY_PREMIUM_PUZZLE_HJ_BOUND_VIOLATION"
    if hj_distance <= 1e-4:
        return "EXACT_NO_ARBITRAGE_PRICING_KERNEL"
    if hj_distance <= 0.05:
        return "ADMISSIBLE_LOW_SPECIFICATION_ERROR"
    if max_sharpe_annualized >= 2.0:
        return "EXTREME_SHARPE_FRONTIER_NEAR_ARBITRAGE"
    return "HIGH_MISSPECIFICATION_SDF_REJECTED"


def calculate_hansen_jagannathan_bounds(
    gross_returns: np.ndarray | List[List[float]],
    candidate_sdf: Optional[np.ndarray | List[float]] = None,
    risk_free_gross: float = 1.00015,
    annualization_factor: float = 252.0,
    consumption_vol_period: float = 0.002,
    ticker: str = "SDF",
) -> HansenJagannathanReceipt:
    """Compute Hansen-Jagannathan (1991) volatility bounds and (1997) specification distance.

    Args:
        gross_returns: Matrix R of shape (T, N) with gross asset returns (1 + r).
        candidate_sdf: Optional candidate pricing kernel m of length T. If omitted,
                       computes the exact minimum-variance admissible projection SDF m*.
        risk_free_gross: Per-period gross risk-free rate R_f (> 0).
        annualization_factor: Periods per year (e.g. 252.0 daily, 12.0 monthly, 1.0 annual).
        consumption_vol_period: Per-period consumption growth std for CRRA gamma diagnostic.
        ticker: Security or portfolio ticker identifier.

    Returns:
        HansenJagannathanReceipt with volatility bounds, HJ-distance, and HMAC signature.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "SDF"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    r_mat = np.asarray(gross_returns, dtype=float)
    if r_mat.ndim == 1:
        r_mat = r_mat.reshape(-1, 1)
    if r_mat.ndim != 2:
        raise ValueError("gross_returns must be a 2D array of shape (T, N).")

    t_obs, n_assets = r_mat.shape
    if t_obs < 5 or n_assets < 1:
        raise ValueError("Require at least 5 time observations and 1 basis asset.")
    if not np.all(np.isfinite(r_mat)):
        raise ValueError("gross_returns must contain only finite values.")

    rf = float(risk_free_gross)
    if not math.isfinite(rf) or rf <= 0.0:
        raise ValueError(f"risk_free_gross must be finite and positive, got: {risk_free_gross}")

    # Excess returns and sample covariance (ddof=0 so sample moments match G_2 = E[R^T R])
    r_excess = r_mat - rf
    mu_e = np.mean(r_excess, axis=0)
    sigma_mat = np.cov(r_mat, rowvar=False, ddof=0)
    if n_assets == 1:
        sigma_mat = np.array([[float(sigma_mat)]])

    sigma_inv = _safe_invert(sigma_mat)
    sr_sq = max(0.0, float(mu_e.T @ sigma_inv @ mu_e))
    max_sharpe_period = math.sqrt(sr_sq)
    max_sharpe_ann = max_sharpe_period * math.sqrt(max(1.0, float(annualization_factor)))

    # Second-moment matrix G_2 = (1/T) R^T R for HJ-Distance (Hansen & Jagannathan 1997)
    g2_mat = (r_mat.T @ r_mat) / float(t_obs)
    g2_inv = _safe_invert(g2_mat)

    if candidate_sdf is None:
        # Exact minimum-variance admissible projection SDF m* = R * (G_2^{-1} 1)
        ones = np.ones(n_assets, dtype=float)
        theta_star = g2_inv @ ones
        m_arr = r_mat @ theta_star
    else:
        m_arr = np.asarray(candidate_sdf, dtype=float).ravel()
        if len(m_arr) != t_obs:
            raise ValueError(f"candidate_sdf length ({len(m_arr)}) must match gross_returns rows ({t_obs}).")
        if not np.all(np.isfinite(m_arr)):
            raise ValueError("candidate_sdf must contain only finite values.")

    sdf_mean = float(np.mean(m_arr))
    sdf_std = float(np.std(m_arr, ddof=0))
    hj_vol_bound = abs(sdf_mean) * max_sharpe_period
    hj_disp_ratio = sdf_std / max(1e-12, abs(sdf_mean))
    hj_bound_satisfied = sdf_std >= (hj_vol_bound - 1e-7)

    # Pricing error vector g = E[m * R] - 1
    e_mr = (r_mat.T @ m_arr) / float(t_obs)
    pricing_errors = e_mr - 1.0
    hj_dist_sq = max(0.0, float(pricing_errors.T @ g2_inv @ pricing_errors))
    hj_distance = math.sqrt(hj_dist_sq)
    mean_abs_error = float(np.mean(np.abs(pricing_errors)))

    c_vol = max(1e-6, float(consumption_vol_period))
    implied_min_gamma = max_sharpe_period / c_vol

    regime = classify_hj_regime(
        hj_bound_satisfied=hj_bound_satisfied,
        hj_distance=hj_distance,
        max_sharpe_annualized=max_sharpe_ann,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{t_obs}:{n_assets}:{max_sharpe_period:.6f}:{sdf_mean:.6f}:"
        f"{sdf_std:.6f}:{hj_vol_bound:.6f}:{hj_distance:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return HansenJagannathanReceipt(
        ticker=cleaned_ticker,
        num_observations=t_obs,
        num_assets=n_assets,
        risk_free_gross=rf,
        max_sharpe_period=max_sharpe_period,
        max_sharpe_annualized=max_sharpe_ann,
        candidate_sdf_mean=sdf_mean,
        candidate_sdf_std=sdf_std,
        hj_volatility_bound=hj_vol_bound,
        hj_dispersion_ratio=hj_disp_ratio,
        hj_bound_satisfied=hj_bound_satisfied,
        hj_distance=hj_distance,
        mean_abs_pricing_error=mean_abs_error,
        implied_min_risk_aversion=implied_min_gamma,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_hansen_jagannathan_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "SDF",
    risk_aversion: float = 5.0,
    discount_beta: float = 0.99,
    risk_free_annual: float = 0.04,
) -> HansenJagannathanReceipt:
    """Calibrate basis factor portfolios from historical price series and evaluate CRRA SDF against HJ bounds."""
    p_arr = np.asarray(prices, dtype=float)
    rf_daily = 1.0 + (float(risk_free_annual) / 252.0)

    if len(p_arr) >= 20:
        rets = np.diff(p_arr) / np.maximum(1e-6, p_arr[:-1])
        # Build 3 basis factor test portfolios: Base asset, Momentum-tilted, Low-Vol hedged
        t = len(rets)
        mom_signal = np.roll(rets, 1)
        mom_signal[0] = 0.0
        r_asset = 1.0 + rets
        r_mom = 1.0 + 1.15 * rets + 0.10 * np.sign(mom_signal) * np.abs(rets)
        r_hedged = 1.0 + 0.60 * rets + 0.40 * (rf_daily - 1.0)
        r_mat = np.column_stack([r_asset, r_mom, r_hedged])
        # Consumption proxy correlated with equity wealth shocks
        cons_growth = 0.0001 + 0.15 * rets
    else:
        # Synthetic well-behaved daily baseline series if no price history provided
        rng = np.random.default_rng(42)
        t = 60
        base = rng.normal(0.0005, 0.012, size=t)
        r_mat = np.column_stack([
            1.0 + base,
            1.0 + 1.2 * base + rng.normal(0.0002, 0.005, size=t),
            1.0 + 0.5 * base + rng.normal(0.0003, 0.004, size=t),
        ])
        cons_growth = 0.0001 + 0.15 * base

    beta_daily = float(np.clip(discount_beta, 0.80, 0.9999)) ** (1.0 / 252.0)
    gamma = max(0.1, float(risk_aversion))
    c_ratio = np.maximum(0.5, 1.0 + cons_growth)
    crra_sdf = beta_daily * (c_ratio ** (-gamma))

    return calculate_hansen_jagannathan_bounds(
        gross_returns=r_mat,
        candidate_sdf=crra_sdf,
        risk_free_gross=rf_daily,
        annualization_factor=252.0,
        consumption_vol_period=float(np.std(cons_growth, ddof=0)) if np.std(cons_growth) > 1e-6 else 0.002,
        ticker=ticker,
    )


def persist_hansen_jagannathan_record(
    receipt: HansenJagannathanReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Hansen-Jagannathan diagnostics receipt to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS hansen_jagannathan_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        num_observations INTEGER NOT NULL,
        num_assets INTEGER NOT NULL,
        risk_free_gross REAL NOT NULL,
        max_sharpe_period REAL NOT NULL,
        max_sharpe_annualized REAL NOT NULL,
        candidate_sdf_mean REAL NOT NULL,
        candidate_sdf_std REAL NOT NULL,
        hj_volatility_bound REAL NOT NULL,
        hj_dispersion_ratio REAL NOT NULL,
        hj_bound_satisfied INTEGER NOT NULL,
        hj_distance REAL NOT NULL,
        mean_abs_pricing_error REAL NOT NULL,
        implied_min_risk_aversion REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_hj_ticker_time ON hansen_jagannathan_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO hansen_jagannathan_audits (
        ticker, num_observations, num_assets, risk_free_gross,
        max_sharpe_period, max_sharpe_annualized, candidate_sdf_mean,
        candidate_sdf_std, hj_volatility_bound, hj_dispersion_ratio,
        hj_bound_satisfied, hj_distance, mean_abs_pricing_error,
        implied_min_risk_aversion, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.num_observations,
                    receipt.num_assets,
                    receipt.risk_free_gross,
                    receipt.max_sharpe_period,
                    receipt.max_sharpe_annualized,
                    receipt.candidate_sdf_mean,
                    receipt.candidate_sdf_std,
                    receipt.hj_volatility_bound,
                    receipt.hj_dispersion_ratio,
                    1 if receipt.hj_bound_satisfied else 0,
                    receipt.hj_distance,
                    receipt.mean_abs_pricing_error,
                    receipt.implied_min_risk_aversion,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

