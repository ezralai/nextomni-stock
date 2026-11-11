"""Fischer Black & John C. Cox (1976) First-Passage Structural Credit & Safety Covenant Engine.

Theoretical Foundation:
    Black, Fischer, & Cox, John C. (1976).
    "Valuing Corporate Securities: Some Effects of Bond Indenture Provisions".
    The Journal of Finance, 31(2), pp. 351-367.

Key Concepts:
    - Exponential Safety Covenant Barrier:
      Debt holders enforce a continuous reorganization barrier C(t) = K * exp(-gamma * (T - t)),
      shifting default from terminal-only checking (Merton 1974) to finite-horizon first passage.
    - Down-and-Out Equity & Covenant Credit Spreads:
      Equity is priced as a finite-horizon down-and-out call option on firm assets, reducing to
      standard Merton (1974) valuation as K -> 0. Risky debt D_BC = V_0 - E_BC gains safety
      protection relative to Merton debt, lowering credit spreads by (s_Merton - s_BC).
    - Log-Space Reflection Principle Evaluation:
      Prevents exponent overflow at low volatilities or extreme asset-to-barrier ratios.
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
from scipy.special import log_ndtr

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"black-cox-covenants-v152-secret"


@dataclass(frozen=True)
class BlackCoxCovenantReceipt:
    """Immutable quantitative audit receipt for Black & Cox (1976) structural credit."""

    ticker: str
    asset_value_v0: float
    debt_face_f: float
    covenant_level_k: float
    covenant_growth_gamma: float
    initial_barrier_h0: float
    maturity_years: float
    asset_volatility: float
    risk_free_rate: float
    payout_yield_q: float
    asset_drift_mu: float
    covenant_breached_at_inception: bool
    black_cox_equity_value: float
    merton_equity_value: float
    covenant_equity_transfer: float
    black_cox_debt_value: float
    merton_debt_value: float
    black_cox_credit_spread_bps: float
    merton_credit_spread_bps: float
    covenant_spread_savings_bps: float
    first_passage_barrier_pd: float
    total_default_prob: float
    merton_terminal_pd: float
    pre_maturity_default_share: float
    distance_to_covenant_barrier: float
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


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _safe_power_times_norm_cdf(log_ratio: float, power: float, z: float) -> float:
    """Evaluate exp(power * log_ratio) * N(z) safely in log space without 0*inf NaNs."""
    log_nd = float(log_ndtr(z))
    log_val = power * log_ratio + log_nd
    if log_val < -700.0:
        return 0.0
    if log_val > 700.0:
        return math.exp(700.0)
    return float(math.exp(log_val))


def _survived_terminal_prob(
    x0: float,
    k_thresh: float,
    drift_x: float,
    vol: float,
    maturity: float,
) -> float:
    """Compute P(X_T > k_thresh, min_{0<=t<=T} X_t > 0) for Brownian motion starting at x0 > 0."""
    vol_rt = vol * math.sqrt(maturity)
    if vol_rt <= 1e-12:
        if x0 <= 0.0:
            return 0.0
        # Deterministic path check: stays > 0 at t=0 and T, and finishes > k_thresh
        end_x = x0 + drift_x * maturity
        return 1.0 if (x0 > 0.0 and end_x > max(0.0, k_thresh)) else 0.0

    z1 = (x0 - k_thresh + drift_x * maturity) / vol_rt
    z2 = (-x0 - k_thresh + drift_x * maturity) / vol_rt
    direct = _norm_cdf(z1)
    # log(H0/V0) == -x0; exponent == 2 * drift_x / vol^2
    power = (2.0 * drift_x) / (vol * vol)
    reflected = _safe_power_times_norm_cdf(log_ratio=-x0, power=power, z=z2)
    return float(np.clip(direct - reflected, 0.0, 1.0))


def classify_black_cox_regime(
    breached: bool,
    total_pd: float,
    pre_maturity_share: float,
    spread_bps: float,
) -> str:
    """Classify structural credit and covenant protection regime."""
    if breached:
        return "TECHNICAL_COVENANT_BREACH_DEFAULT"
    if total_pd >= 0.25 or spread_bps >= 350.0:
        return "DISTRESSED_FIRST_PASSAGE_RISK"
    if pre_maturity_share >= 0.40:
        return "COVENANT_DOMINATED_PREMATURITY_RISK"
    if total_pd >= 0.05 or spread_bps >= 75.0:
        return "SUB_INVESTMENT_LEVERAGED_CREDIT"
    return "INVESTMENT_GRADE_COVENANT_PROTECTED"


def _build_hmac_payload(
    ticker: str,
    v0: float,
    f_debt: float,
    k_cov: float,
    gamma: float,
    maturity: float,
    vol: float,
    bc_equity: float,
    bc_spread: float,
    total_pd: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{v0:.4f}:{f_debt:.4f}:{k_cov:.4f}:{gamma:.4f}:"
        f"{maturity:.4f}:{vol:.4f}:{bc_equity:.6f}:{bc_spread:.2f}:"
        f"{total_pd:.6f}:{regime}:{timestamp}"
    )


def calculate_black_cox_covenants(
    asset_value_v0: float = 150.0,
    debt_face_f: float = 100.0,
    covenant_level_k: float = 80.0,
    covenant_growth_gamma: float = 0.03,
    maturity_years: float = 5.0,
    asset_volatility: float = 0.28,
    risk_free_rate: float = 0.045,
    payout_yield_q: float = 0.015,
    asset_drift_mu: float = 0.08,
    ticker: str = "CORP",
) -> BlackCoxCovenantReceipt:
    """Evaluate Black & Cox (1976) down-and-out equity, covenant debt value, and first-passage PD."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    v0 = float(asset_value_v0)
    f_debt = float(debt_face_f)
    k_cov = float(covenant_level_k)
    gamma = float(covenant_growth_gamma)
    t_mat = float(maturity_years)
    vol = float(asset_volatility)
    r_rate = float(risk_free_rate)
    q_yield = float(payout_yield_q)
    mu_phys = float(asset_drift_mu)

    if v0 <= 0.0 or f_debt <= 0.0:
        raise ValueError("asset_value_v0 and debt_face_f must be strictly positive.")
    if k_cov < 0.0:
        raise ValueError("covenant_level_k must be non-negative.")
    if k_cov > f_debt:
        raise ValueError("covenant_level_k at maturity cannot exceed debt_face_f.")
    if t_mat <= 0.0:
        raise ValueError("maturity_years must be strictly positive.")
    if vol <= 0.0:
        raise ValueError("asset_volatility must be strictly positive.")

    h0 = k_cov * math.exp(-gamma * t_mat) if k_cov > 0.0 else 0.0
    disc_r = math.exp(-r_rate * t_mat)
    disc_q = math.exp(-q_yield * t_mat)
    vol_rt = vol * math.sqrt(t_mat)

    # Merton (1974) terminal-only benchmark
    d1_merton = (math.log(v0 / f_debt) + (r_rate - q_yield + 0.5 * vol * vol) * t_mat) / vol_rt
    d2_merton = d1_merton - vol_rt
    merton_eq = float(max(0.0, v0 * disc_q * _norm_cdf(d1_merton) - f_debt * disc_r * _norm_cdf(d2_merton)))
    merton_debt = float(max(1e-8, v0 - merton_eq))
    merton_yield = -math.log(merton_debt / f_debt) / t_mat
    merton_spread_bps = max(0.0, (merton_yield - r_rate) * 10000.0)

    # Physical terminal-only Merton default probability
    d2_phys = (math.log(v0 / f_debt) + (mu_phys - q_yield - 0.5 * vol * vol) * t_mat) / vol_rt
    merton_pd = float(np.clip(_norm_cdf(-d2_phys), 0.0, 1.0))

    breached = (k_cov > 0.0) and (v0 <= h0)

    if breached:
        bc_eq = 0.0
        bc_debt = min(v0, f_debt * disc_r)
        first_passage_pd = 1.0
        total_pd = 1.0
        dist_barrier = 0.0
    elif k_cov == 0.0:
        bc_eq = merton_eq
        bc_debt = merton_debt
        first_passage_pd = 0.0
        total_pd = merton_pd
        dist_barrier = 10.0
    else:
        x0 = math.log(v0 / h0)
        k_thresh = math.log(f_debt / k_cov)
        dist_barrier = x0 / vol_rt

        # Risk-neutral stock-numeraire drift and money-market drift for X_t = ln(V_t / H(t))
        drift_s = r_rate - q_yield - gamma + 0.5 * vol * vol
        drift_q = r_rate - q_yield - gamma - 0.5 * vol * vol

        prob_stock = _survived_terminal_prob(x0, k_thresh, drift_s, vol, t_mat)
        prob_q = _survived_terminal_prob(x0, k_thresh, drift_q, vol, t_mat)
        bc_eq = float(max(0.0, min(merton_eq, v0 * disc_q * prob_stock - f_debt * disc_r * prob_q)))
        bc_debt = float(max(1e-8, v0 - bc_eq))

        # Physical first-passage probability of hitting H(t) during (0, T]
        mu_hat = mu_phys - q_yield - gamma - 0.5 * vol * vol
        z_fp1 = (-x0 - mu_hat * t_mat) / vol_rt
        z_fp2 = (-x0 + mu_hat * t_mat) / vol_rt
        fp_power = -(2.0 * mu_hat) / (vol * vol)
        first_passage_pd = float(
            np.clip(
                _norm_cdf(z_fp1) + _safe_power_times_norm_cdf(log_ratio=x0, power=fp_power, z=z_fp2),
                0.0,
                1.0,
            )
        )

        # Total physical default probability: 1 - P(survived barrier in [0, T] AND V_T > F)
        surv_total = _survived_terminal_prob(x0, k_thresh, mu_hat, vol, t_mat)
        total_pd = float(np.clip(max(first_passage_pd, merton_pd, 1.0 - surv_total), 0.0, 1.0))

    bc_yield = -math.log(max(1e-8, bc_debt) / f_debt) / t_mat
    bc_spread_bps = max(0.0, (bc_yield - r_rate) * 10000.0)
    cov_transfer = max(0.0, merton_eq - bc_eq)
    spread_savings = max(0.0, merton_spread_bps - bc_spread_bps)
    pre_mat_share = float(np.clip((total_pd - merton_pd) / max(1e-9, total_pd), 0.0, 1.0))

    regime = classify_black_cox_regime(
        breached=breached,
        total_pd=total_pd,
        pre_maturity_share=pre_mat_share,
        spread_bps=bc_spread_bps,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        v0=v0,
        f_debt=f_debt,
        k_cov=k_cov,
        gamma=gamma,
        maturity=t_mat,
        vol=vol,
        bc_equity=bc_eq,
        bc_spread=bc_spread_bps,
        total_pd=total_pd,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return BlackCoxCovenantReceipt(
        ticker=cleaned_ticker,
        asset_value_v0=v0,
        debt_face_f=f_debt,
        covenant_level_k=k_cov,
        covenant_growth_gamma=gamma,
        initial_barrier_h0=h0,
        maturity_years=t_mat,
        asset_volatility=vol,
        risk_free_rate=r_rate,
        payout_yield_q=q_yield,
        asset_drift_mu=mu_phys,
        covenant_breached_at_inception=breached,
        black_cox_equity_value=bc_eq,
        merton_equity_value=merton_eq,
        covenant_equity_transfer=cov_transfer,
        black_cox_debt_value=bc_debt,
        merton_debt_value=merton_debt,
        black_cox_credit_spread_bps=bc_spread_bps,
        merton_credit_spread_bps=merton_spread_bps,
        covenant_spread_savings_bps=spread_savings,
        first_passage_barrier_pd=first_passage_pd,
        total_default_prob=total_pd,
        merton_terminal_pd=merton_pd,
        pre_maturity_default_share=pre_mat_share,
        distance_to_covenant_barrier=dist_barrier,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_black_cox_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "CORP",
    leverage_ratio: float = 0.55,
    covenant_coverage: float = 0.80,
    maturity_years: float = 5.0,
) -> BlackCoxCovenantReceipt:
    """Calibrate asset volatility proxy from trailing equity closes and evaluate Black-Cox credit."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        v0 = float(max(10.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        eq_vol = float(np.std(log_rets) * math.sqrt(252.0))
        # De-lever equity volatility roughly to asset volatility
        eff_vol = float(np.clip(eq_vol * (1.0 - 0.5 * leverage_ratio), 0.10, 0.90))
    else:
        v0 = 150.0
        eff_vol = 0.28

    f_debt = v0 * float(np.clip(leverage_ratio, 0.10, 0.95))
    k_cov = f_debt * float(np.clip(covenant_coverage, 0.0, 0.98))
    return calculate_black_cox_covenants(
        asset_value_v0=v0,
        debt_face_f=f_debt,
        covenant_level_k=k_cov,
        maturity_years=maturity_years,
        asset_volatility=eff_vol,
        ticker=ticker,
    )


def persist_black_cox_record(
    receipt: BlackCoxCovenantReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Black & Cox (1976) covenant credit audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS black_cox_covenant_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        asset_value_v0 REAL NOT NULL,
        debt_face_f REAL NOT NULL,
        covenant_level_k REAL NOT NULL,
        initial_barrier_h0 REAL NOT NULL,
        maturity_years REAL NOT NULL,
        asset_volatility REAL NOT NULL,
        black_cox_equity_value REAL NOT NULL,
        black_cox_debt_value REAL NOT NULL,
        black_cox_credit_spread_bps REAL NOT NULL,
        first_passage_barrier_pd REAL NOT NULL,
        total_default_prob REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO black_cox_covenant_audits (
        ticker, asset_value_v0, debt_face_f, covenant_level_k, initial_barrier_h0,
        maturity_years, asset_volatility, black_cox_equity_value, black_cox_debt_value,
        black_cox_credit_spread_bps, first_passage_barrier_pd, total_default_prob,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.asset_value_v0,
                receipt.debt_face_f,
                receipt.covenant_level_k,
                receipt.initial_barrier_h0,
                receipt.maturity_years,
                receipt.asset_volatility,
                receipt.black_cox_equity_value,
                receipt.black_cox_debt_value,
                receipt.black_cox_credit_spread_bps,
                receipt.first_passage_barrier_pd,
                receipt.total_default_prob,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

