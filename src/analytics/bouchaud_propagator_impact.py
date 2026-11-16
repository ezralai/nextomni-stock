"""
Jean-Philippe Bouchaud, Yuval Gefen, Marc Potters & Matthieu Wyart (2004)
Transient Market Impact & Power-Law Propagator Engine.

References:
    Bouchaud, J.-P., Gefen, Y., Potters, M., & Wyart, M. (2004).
    Fluctuations and response in financial markets: the subtle nature of 'random' price changes.
    Quantitative Finance, 4(2), 176–190.

    Bouchaud, J.-P., Farmer, J. D., & Lillo, F. (2009).
    How markets slowly digest changes in supply and demand.
    Handbook of Financial Markets: Dynamics and Evolution, Elsevier, pp. 57–160.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import hmac
import math
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9\.\_\-]{1,10}$")
_DEFAULT_HMAC_SECRET = b"nextomni-bouchaud-propagator-2004-secret"


def _get_hmac_secret() -> bytes:
    env_secret = os.getenv("NEXTOMNI_HMAC_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")
    return _DEFAULT_HMAC_SECRET


def _compute_hmac(payload: str) -> str:
    return hmac.new(_get_hmac_secret(), payload.encode("utf-8"), hashlib.sha256).hexdigest()


def compute_receipt_payload(receipt: BouchaudPropagatorReceipt) -> str:
    """Canonical string representation for cryptographic HMAC signing."""
    return (
        f"{receipt.ticker}:{receipt.sample_trades}:{receipt.gamma_order_flow:.4f}:"
        f"{receipt.beta_propagator_decay:.4f}:{receipt.gamma_0_instantaneous:.6f}:"
        f"{receipt.efficiency_gap:.4f}:{receipt.resilience_half_life:.2f}:{receipt.regime}:{receipt.timestamp}"
    )


def verify_hmac_signature(payload: str, signature: str) -> bool:
    """Verify HMAC-SHA256 signature against system secret."""
    expected = _compute_hmac(payload)
    return hmac.compare_digest(expected, signature)


@dataclasses.dataclass(frozen=True)
class BouchaudPropagatorReceipt:
    ticker: str
    sample_trades: int
    gamma_order_flow: float
    beta_propagator_decay: float
    gamma_0_instantaneous: float
    theoretical_beta_balance: float
    efficiency_gap: float
    resilience_half_life: float
    permanent_to_transient_ratio: float
    peak_response_dollar: float
    response_lags: List[int]
    response_values: List[float]
    propagator_values: List[float]
    regime: str
    derivation_mode: str
    hmac_signature: str
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


def _empty_receipt(ticker: str, error_msg: str = "INVALID_INPUT") -> BouchaudPropagatorReceipt:
    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    raw = f"{ticker}:0:0.0000:0.0000:0.000000:0.0000:0.00:ERROR_{error_msg}:{ts}"
    return BouchaudPropagatorReceipt(
        ticker=ticker,
        sample_trades=0,
        gamma_order_flow=0.50,
        beta_propagator_decay=0.25,
        gamma_0_instantaneous=0.0,
        theoretical_beta_balance=0.25,
        efficiency_gap=0.0,
        resilience_half_life=0.0,
        permanent_to_transient_ratio=0.0,
        peak_response_dollar=0.0,
        response_lags=[],
        response_values=[],
        propagator_values=[],
        regime=f"ERROR_{error_msg}",
        derivation_mode="FAILED",
        hmac_signature=_compute_hmac(raw),
        timestamp=ts,
    )


def calculate_bouchaud_propagator(
    prices: Sequence[float],
    signed_trades: Sequence[int],
    ticker: str = "SPY",
    max_lags: int = 20,
    regularization_lambda: float = 1e-4,
) -> BouchaudPropagatorReceipt:
    """
    Calculate Bouchaud et al. (2004) transient price impact kernel,
    order flow memory exponent gamma, and efficiency balance condition.

    Args:
        prices: Sequence of transaction or mid-quote prices ($)
        signed_trades: Sequence of trade directions (+1 buy, -1 sell)
        ticker: Symbol string
        max_lags: Maximum lag horizon for propagator estimation (e.g. 20)
        regularization_lambda: Tikhonov damping factor for Toeplitz inversion

    Returns:
        BouchaudPropagatorReceipt with fitted exponents, half-life, and HMAC signature.
    """
    clean_ticker = (ticker or "SPY").strip().upper()
    if not TICKER_REGEX.match(clean_ticker):
        clean_ticker = "SPY"

    p_arr = np.asarray(prices, dtype=np.float64)
    s_arr = np.asarray(signed_trades, dtype=np.float64)

    n = min(len(p_arr), len(s_arr))
    if n < 30:
        return _empty_receipt(clean_ticker, "INSUFFICIENT_TRADES")

    p_arr = p_arr[:n]
    s_arr = s_arr[:n]

    # Clean signed trades: ensure {-1, +1}
    s_arr = np.where(s_arr >= 0, 1.0, -1.0)

    lags_cap = max(3, min(int(max_lags), n // 4))

    # 1. Compute order flow autocorrelation C(l) = <s_t * s_{t+l}>
    c_lags = np.zeros(lags_cap + 1, dtype=np.float64)
    c_lags[0] = 1.0
    for ell in range(1, lags_cap + 1):
        c_lags[ell] = np.mean(s_arr[:-ell] * s_arr[ell:])

    # Fit order flow decay exponent gamma: C(l) ~ c0 * l^(-gamma)
    # Use positive correlations for log-log fit
    fit_lags = []
    fit_c = []
    for ell in range(1, lags_cap + 1):
        if c_lags[ell] > 1e-4:
            fit_lags.append(ell)
            fit_c.append(c_lags[ell])

    if len(fit_lags) >= 2:
        log_l = np.log(np.array(fit_lags, dtype=np.float64))
        log_c = np.log(np.array(fit_c, dtype=np.float64))
        # Slope = -gamma
        slope, _ = np.polyfit(log_l, log_c, deg=1)
        gamma = float(np.clip(-slope, 0.05, 0.95))
    else:
        gamma = 0.50

    # 2. Compute price response function R(l) = <(P_{t+l} - P_t) * s_t>
    r_lags = np.zeros(lags_cap, dtype=np.float64)
    for ell in range(1, lags_cap + 1):
        dp = p_arr[ell:] - p_arr[:-ell]
        r_lags[ell - 1] = np.mean(dp * s_arr[:-ell])

    # 3. Invert Toeplitz convolution R = C_toeplitz * G
    # Construct Toeplitz matrix C_mat[i, j] = C(|i - j|)
    c_mat = np.zeros((lags_cap, lags_cap), dtype=np.float64)
    for i in range(lags_cap):
        for j in range(lags_cap):
            dist = abs(i - j)
            if dist <= lags_cap:
                c_mat[i, j] = c_lags[dist]

    # Regularized inversion
    reg = max(1e-6, float(regularization_lambda))
    inv_mat = np.linalg.pinv(c_mat + reg * np.eye(lags_cap))
    g_raw = inv_mat @ r_lags

    # Ensure non-negative bare propagator values for power-law decay
    # G(0) is instantaneous impact
    gamma_0 = float(max(1e-6, g_raw[0] if g_raw[0] > 0 else np.mean(np.abs(r_lags[:2]))))
    g_clean = np.maximum(1e-6, g_raw)

    # 4. Fit power-law propagator G(l) = Gamma_0 / (1 + l)^beta
    # log(G(l)) = log(Gamma_0) - beta * log(1 + l)
    ell_indices = np.arange(lags_cap, dtype=np.float64)
    log_1_plus_l = np.log(1.0 + ell_indices)
    log_g = np.log(g_clean)

    # Weighted or linear fit
    poly_fit = np.polyfit(log_1_plus_l, log_g, deg=1)
    beta = float(np.clip(-poly_fit[0], 0.02, 1.50))
    gamma_0_fitted = float(np.exp(poly_fit[1]))

    # If fit gamma_0 is sensible, use it, else raw G(0)
    gamma_0_final = gamma_0 if (0.01 * gamma_0 < gamma_0_fitted < 100.0 * gamma_0) else gamma_0

    # 5. Theoretical balance condition (Bouchaud et al. 2004): beta_balance = (1 - gamma) / 2
    beta_balance = float(round((1.0 - gamma) / 2.0, 4))
    eff_gap = float(round(abs(beta - beta_balance), 4))

    # 6. Resilience half-life: G(tau) = 0.5 * Gamma_0 => (1 + tau)^beta = 2 => tau = 2^(1/beta) - 1
    if beta > 0.01:
        resilience_half_life = float(round(min(500.0, max(0.1, (2.0 ** (1.0 / beta)) - 1.0)), 2))
    else:
        resilience_half_life = 100.0

    # Permanent to transient ratio: G(max_lags) / G(0) = (1 + max_lags)^(-beta)
    perm_ratio = float(round((1.0 + lags_cap) ** (-beta), 4))
    peak_response = float(round(float(np.max(np.abs(r_lags))), 5))

    # Market regime classification
    if eff_gap <= 0.12:
        regime = "EFFICIENT_DIFFUSIVE_EQUILIBRIUM"
    elif beta > beta_balance + 0.12:
        regime = "SUB_DIFFUSIVE_OVER_MEAN_REVERSION"
    else:
        regime = "SUPER_DIFFUSIVE_MOMENTUM_RUNS"

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    r_gamma = float(round(gamma, 4))
    r_beta = float(round(beta, 4))
    r_g0 = float(round(gamma_0_final, 6))

    raw_payload = (
        f"{clean_ticker}:{n}:{r_gamma:.4f}:{r_beta:.4f}:{r_g0:.6f}:"
        f"{eff_gap:.4f}:{resilience_half_life:.2f}:{regime}:{ts}"
    )
    sig = _compute_hmac(raw_payload)

    lags_list = [int(i + 1) for i in range(lags_cap)]
    r_list = [float(round(r_lags[i], 6)) for i in range(lags_cap)]
    g_list = [float(round(g_clean[i], 6)) for i in range(lags_cap)]

    return BouchaudPropagatorReceipt(
        ticker=clean_ticker,
        sample_trades=n,
        gamma_order_flow=r_gamma,
        beta_propagator_decay=r_beta,
        gamma_0_instantaneous=r_g0,
        theoretical_beta_balance=beta_balance,
        efficiency_gap=eff_gap,
        resilience_half_life=resilience_half_life,
        permanent_to_transient_ratio=perm_ratio,
        peak_response_dollar=peak_response,
        response_lags=lags_list,
        response_values=r_list,
        propagator_values=g_list,
        regime=regime,
        derivation_mode="EMPIRICAL_TRADE_CONVOLUTION",
        hmac_signature=sig,
        timestamp=ts,
    )


def evaluate_bouchaud_propagator_from_prices(
    prices: Sequence[float],
    ticker: str = "SPY",
    max_lags: int = 15,
) -> BouchaudPropagatorReceipt:
    """
    Evaluate Bouchaud propagator using tick price changes to infer trade sign (tick test).
    """
    p_arr = np.asarray(prices, dtype=np.float64)
    if len(p_arr) < 30:
        return _empty_receipt(ticker, "INSUFFICIENT_PRICES")

    # Infer signed order flow via tick rule: sign(P_t - P_{t-1})
    diffs = np.diff(p_arr)
    signs = np.zeros(len(diffs), dtype=np.float64)
    curr_sign = 1.0
    for i, d in enumerate(diffs):
        if d > 0:
            curr_sign = 1.0
        elif d < 0:
            curr_sign = -1.0
        signs[i] = curr_sign

    receipt = calculate_bouchaud_propagator(
        prices=p_arr[1:],
        signed_trades=signs,
        ticker=ticker,
        max_lags=max_lags,
    )
    return dataclasses.replace(receipt, derivation_mode="TICK_TEST_INFERRED_TRADES")


def persist_as_record(
    receipt: BouchaudPropagatorReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Bouchaud Propagator receipt into SQLite audit ledger with foreign key protection."""
    if receipt.derivation_mode == "FAILED":
        return False

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=10.0)
    try:
        with conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS bouchaud_propagator_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticker TEXT NOT NULL,
                    sample_trades INTEGER NOT NULL,
                    gamma_order_flow REAL NOT NULL,
                    beta_propagator_decay REAL NOT NULL,
                    gamma_0_instantaneous REAL NOT NULL,
                    theoretical_beta_balance REAL NOT NULL,
                    efficiency_gap REAL NOT NULL,
                    resilience_half_life REAL NOT NULL,
                    permanent_to_transient_ratio REAL NOT NULL,
                    peak_response_dollar REAL NOT NULL,
                    regime TEXT NOT NULL,
                    derivation_mode TEXT NOT NULL,
                    hmac_signature TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_bouchaud_ticker_created
                ON bouchaud_propagator_records (ticker, created_at);
                """
            )
            conn.execute(
                """
                INSERT INTO bouchaud_propagator_records (
                    ticker, sample_trades, gamma_order_flow, beta_propagator_decay,
                    gamma_0_instantaneous, theoretical_beta_balance, efficiency_gap,
                    resilience_half_life, permanent_to_transient_ratio, peak_response_dollar,
                    regime, derivation_mode, hmac_signature, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    receipt.ticker,
                    receipt.sample_trades,
                    receipt.gamma_order_flow,
                    receipt.beta_propagator_decay,
                    receipt.gamma_0_instantaneous,
                    receipt.theoretical_beta_balance,
                    receipt.efficiency_gap,
                    receipt.resilience_half_life,
                    receipt.permanent_to_transient_ratio,
                    receipt.peak_response_dollar,
                    receipt.regime,
                    receipt.derivation_mode,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
        return True
    except Exception:
        return False
    finally:
        conn.close()

