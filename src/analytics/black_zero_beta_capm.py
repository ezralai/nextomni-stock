"""Fischer Black (1972) Zero-Beta Restricted-Borrowing CAPM & SML Flattening Engine.

Theoretical Foundation:
    Black, Fischer (1972).
    "Capital Market Equilibrium with Restricted Borrowing".
    The Journal of Business, 45(3), pp. 444-455.

Key Concepts:
    - Zero-Beta Security Market Line (SML):
      Without unlimited riskless borrowing at r_f, market equilibrium prices any asset i
      relative to the minimum-variance Zero-Beta portfolio (Cov(R_z, R_m) = 0):
      E[R_i] = E[R_z] + beta_i * (E[R_m] - E[R_z]).
    - SML Flattening Alpha & Risky Frontier Geometry:
      Restricted borrowing flattens the SML slope when R_z > r_f, imparting systematic
      positive Black alpha = (R_z - r_f) * (1 - beta_i) to defensive low-beta assets
      relative to the standard Sharpe-Lintner CAPM. On any multi-asset Markowitz risky
      frontier, w_z = (sigma_m^2 * Sigma^-1 * 1 - w_m) / (C * sigma_m^2 - 1) recovers the
      exact minimum-variance zero-covariance companion portfolio.
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
HMAC_SECRET = b"black-zero-beta-capm-v166-secret"


@dataclass(frozen=True)
class BlackZeroBetaReceipt:
    """Immutable quantitative audit receipt for Fischer Black (1972) Zero-Beta CAPM."""

    ticker: str
    asset_beta: float
    risk_free_rate_rf: float
    zero_beta_rate_rz: float
    market_return_rm: float
    zero_beta_spread: float
    black_expected_return: float
    sharpe_lintner_return: float
    sml_alpha_wedge_pct: float
    black_sml_slope: float
    sharpe_lintner_slope: float
    sml_flattening_pct: float
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


def classify_black_zero_beta_regime(
    beta: float,
    rz: float,
    rf: float,
    alpha_wedge_pct: float,
) -> str:
    """Classify asset positioning along flattened Zero-Beta Security Market Line."""
    if abs(rz - rf) <= 1e-6:
        return "UNCONSTRAINED_SHARPE_LINTNER_PARITY"
    if beta < 0.85 and alpha_wedge_pct > 0.25:
        return "LOW_BETA_DEFENSIVE_POSITIVE_BLACK_ALPHA"
    if beta > 1.15 and alpha_wedge_pct < -0.25:
        return "HIGH_BETA_BORROWING_CONSTRAINED_OVERPRICED"
    return "NEUTRAL_BETA_FLATTENED_SML_EQUILIBRIUM"


def _build_hmac_payload(
    ticker: str,
    beta: float,
    rf: float,
    rz: float,
    rm: float,
    black_ret: float,
    sl_ret: float,
    alpha_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{beta:.4f}:{rf:.6f}:{rz:.6f}:{rm:.6f}:"
        f"{black_ret:.6f}:{sl_ret:.6f}:{alpha_pct:.4f}:{regime}:{timestamp}"
    )


def calculate_black_zero_beta_capm(
    asset_beta: float = 0.70,
    risk_free_rate_rf: float = 0.035,
    zero_beta_rate_rz: float = 0.055,
    market_return_rm: float = 0.095,
    ticker: str = "BZ72",
) -> BlackZeroBetaReceipt:
    """Evaluate Black (1972) Zero-Beta expected return and SML flattening vs. standard CAPM."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    beta = float(asset_beta)
    rf = float(risk_free_rate_rf)
    rz = float(zero_beta_rate_rz)
    rm = float(market_return_rm)

    if not math.isfinite(beta) or not (-2.0 <= beta <= 5.0):
        raise ValueError("asset_beta must be finite and within [-2.0, 5.0].")
    if rz < rf - 1e-9:
        raise ValueError("Restricted borrowing requires zero_beta_rate_rz >= risk_free_rate_rf.")
    if rm <= rz:
        raise ValueError("Positive equity risk premium requires market_return_rm > zero_beta_rate_rz.")

    zb_spread = rz - rf
    black_slope = rm - rz
    sl_slope = rm - rf

    black_ret = rz + beta * black_slope
    sl_ret = rf + beta * sl_slope
    alpha_wedge_pct = (black_ret - sl_ret) * 100.0
    flat_pct = ((sl_slope - black_slope) / sl_slope) * 100.0

    regime = classify_black_zero_beta_regime(
        beta=beta,
        rz=rz,
        rf=rf,
        alpha_wedge_pct=alpha_wedge_pct,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        beta=beta,
        rf=rf,
        rz=rz,
        rm=rm,
        black_ret=black_ret,
        sl_ret=sl_ret,
        alpha_pct=alpha_wedge_pct,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BlackZeroBetaReceipt(
        ticker=cleaned_ticker,
        asset_beta=beta,
        risk_free_rate_rf=rf,
        zero_beta_rate_rz=rz,
        market_return_rm=rm,
        zero_beta_spread=zb_spread,
        black_expected_return=black_ret,
        sharpe_lintner_return=sl_ret,
        sml_alpha_wedge_pct=alpha_wedge_pct,
        black_sml_slope=black_slope,
        sharpe_lintner_slope=sl_slope,
        sml_flattening_pct=flat_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def solve_frontier_zero_beta_portfolio(
    expected_returns: List[float] | np.ndarray,
    cov_matrix: np.ndarray,
    market_weights: List[float] | np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute exact minimum-variance orthogonal Zero-Beta companion on Markowitz frontier."""
    mu = np.asarray(expected_returns, dtype=float)
    cov = np.asarray(cov_matrix, dtype=float)
    n_assets = len(mu)

    if n_assets < 2 or cov.shape != (n_assets, n_assets):
        raise ValueError("Requires >= 2 assets and matching square cov_matrix.")
    if not np.allclose(cov, cov.T, atol=1e-8):
        raise ValueError("cov_matrix must be symmetric.")
    eigvals = np.linalg.eigvalsh(cov)
    if np.min(eigvals) <= 1e-10:
        raise ValueError("cov_matrix must be strictly positive definite.")

    ones = np.ones(n_assets, dtype=float)
    inv_cov = np.linalg.inv(cov)
    inv_ones = inv_cov @ ones
    inv_mu = inv_cov @ mu

    a_coef = float(ones @ inv_mu)
    b_coef = float(mu @ inv_mu)
    c_coef = float(ones @ inv_ones)
    disc = b_coef * c_coef - a_coef * a_coef
    if disc <= 1e-10:
        raise ValueError("Expected returns cannot be identical across all risky assets (degenerate frontier).")

    mu_gmv = a_coef / c_coef
    var_gmv = 1.0 / c_coef

    if market_weights is None:
        target_mu_m = mu_gmv + math.sqrt(disc) / c_coef
        g_vec = (b_coef * inv_ones - a_coef * inv_mu) / disc
        h_vec = (c_coef * inv_mu - a_coef * inv_ones) / disc
        w_m = g_vec + h_vec * target_mu_m
    else:
        w_m = np.asarray(market_weights, dtype=float)
        if w_m.shape != (n_assets,) or not math.isclose(float(np.sum(w_m)), 1.0, abs_tol=1e-5):
            raise ValueError("market_weights must match asset count and sum to 1.0.")

    mu_m = float(mu @ w_m)
    var_m = float(w_m @ cov @ w_m)
    if var_m <= var_gmv + 1e-10:
        raise ValueError("Market portfolio variance must strictly exceed Global Minimum Variance to define finite zero-beta companion.")

    denom = c_coef * var_m - 1.0
    w_z = (var_m * inv_ones - w_m) / denom
    mu_z = float(mu @ w_z)
    var_z = float(w_z @ cov @ w_z)
    cov_zm = float(w_z @ cov @ w_m)

    return {
        "market_weights": w_m.tolist(),
        "zero_beta_weights": w_z.tolist(),
        "market_expected_return": mu_m,
        "zero_beta_expected_return": mu_z,
        "gmv_expected_return": mu_gmv,
        "market_volatility": math.sqrt(var_m),
        "zero_beta_volatility": math.sqrt(max(0.0, var_z)),
        "zero_beta_covariance": cov_zm,
    }


def evaluate_black_zero_beta_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BZ72",
    asset_beta: float | None = None,
    risk_free_rate_rf: float = 0.035,
    zero_beta_rate_rz: float = 0.055,
) -> BlackZeroBetaReceipt:
    """Calibrate asset beta proxy and market return benchmark from trailing market series."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        log_rets = np.diff(np.log(np.maximum(1e-6, window)))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        eff_beta = float(asset_beta) if asset_beta is not None else float(np.clip(ann_vol / 0.18, 0.35, 2.20))
        eff_rm = max(float(zero_beta_rate_rz) + 0.025, 0.065 + 0.15 * ann_vol)
    else:
        eff_beta = float(asset_beta) if asset_beta is not None else 0.70
        eff_rm = 0.095

    return calculate_black_zero_beta_capm(
        asset_beta=eff_beta,
        risk_free_rate_rf=float(risk_free_rate_rf),
        zero_beta_rate_rz=float(zero_beta_rate_rz),
        market_return_rm=eff_rm,
        ticker=ticker,
    )


def persist_black_zero_beta_record(
    receipt: BlackZeroBetaReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Black (1972) Zero-Beta CAPM record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS black_zero_beta_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        asset_beta REAL NOT NULL,
        risk_free_rate_rf REAL NOT NULL,
        zero_beta_rate_rz REAL NOT NULL,
        market_return_rm REAL NOT NULL,
        black_expected_return REAL NOT NULL,
        sharpe_lintner_return REAL NOT NULL,
        sml_alpha_wedge_pct REAL NOT NULL,
        sml_flattening_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO black_zero_beta_audits (
        ticker, asset_beta, risk_free_rate_rf, zero_beta_rate_rz, market_return_rm,
        black_expected_return, sharpe_lintner_return, sml_alpha_wedge_pct,
        sml_flattening_pct, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.asset_beta,
                receipt.risk_free_rate_rf,
                receipt.zero_beta_rate_rz,
                receipt.market_return_rm,
                receipt.black_expected_return,
                receipt.sharpe_lintner_return,
                receipt.sml_alpha_wedge_pct,
                receipt.sml_flattening_pct,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

