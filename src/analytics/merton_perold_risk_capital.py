"""Robert C. Merton & Andre F. Perold (1993) Risk Capital & Institutional Allocation Engine.

Theoretical Foundation:
    Merton, Robert C., and Perold, Andre F. (1993).
    "Theory of Risk Capital in Financial Firms".
    Journal of Applied Corporate Finance, 6(3), pp. 16-32.

Key Concepts:
    - Option-Theoretic Definition of Risk Capital (RC):
      Risk capital is the smallest sum invested today that guarantees the value of
      net assets at horizon T against shortfall below risk-free accumulation F = V*exp(r*T).
      Equivalently, it equals a Black-Scholes asset-guarantee put struck at F:
      RC(V, sigma, T) = V * [2 * N(0.5 * sigma * sqrt(T)) - 1].
    - Sub-Additivity, Diversification Surplus & Incremental Under-Allocation Theorem:
      For combined desks (V1, sigma1) and (V2, sigma2) with asset correlation rho < 1,
      firm risk capital satisfies RC_P < RC_1 + RC_2. Because incremental marginal
      capital dRC_i = RC_P - RC_{-i} satisfies dRC_1 + dRC_2 < RC_P, naive incremental
      allocation leaves an unallocated capital gap exactly equal to Diversification Benefit.
    - Euler Homogeneous Allocation & RAROC:
      Allocating total RC_P via Euler partial derivatives V_i * d(RC_P)/d(V_i) sums
      exactly to 100% of firm risk capital (RC_1_Euler + RC_2_Euler == RC_P) while
      reflecting diversification credits in desk-level Risk-Adjusted Return on Risk Capital.
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
HMAC_SECRET = b"merton-perold-risk-capital-v171-secret"


@dataclass(frozen=True)
class MertonPeroldReceipt:
    """Immutable quantitative audit receipt for Merton & Perold (1993) Risk Capital."""

    ticker: str
    unit1_asset_v1: float
    unit1_volatility: float
    unit2_asset_v2: float
    unit2_volatility: float
    cross_unit_correlation: float
    horizon_years: float
    risk_free_rate_r: float
    expected_excess_return: float
    standalone_rc_1: float
    standalone_rc_2: float
    sum_standalone_rc: float
    portfolio_volatility: float
    portfolio_rc: float
    marginal_rc_1: float
    marginal_rc_2: float
    diversification_benefit: float
    unallocated_capital_gap: float
    euler_rc_1: float
    euler_rc_2: float
    capital_intensity_ratio: float
    diversification_benefit_pct: float
    expected_excess_profit: float
    standalone_raroc: float
    euler_raroc: float
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


def _norm_cdf(x: float) -> float:
    """Standard normal cumulative distribution function N(x)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    """Standard normal probability density function n(x)."""
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _standalone_risk_capital(asset_value: float, volatility: float, horizon: float) -> float:
    """Merton-Perold guarantee option value V * [2*N(0.5*sigma*sqrt(T)) - 1]."""
    z = 0.5 * volatility * math.sqrt(horizon)
    return asset_value * (2.0 * _norm_cdf(z) - 1.0)


def _euler_unit_allocation(
    v_unit: float,
    sig_unit: float,
    v_other: float,
    sig_other: float,
    correlation: float,
    sqrt_t: float,
    s_p: float,
    v_p: float,
    base_frac: float,
    pdf_z: float,
) -> float:
    """Exact closed-form Euler partial contribution V_i * d(RC_P)/d(V_i)."""
    cov_term = (v_unit * v_unit * sig_unit * sig_unit) + (
        correlation * v_unit * v_other * sig_unit * sig_other
    )
    ds_part = (cov_term / s_p) - (v_unit * s_p / v_p)
    return v_unit * base_frac + pdf_z * sqrt_t * ds_part


def classify_merton_perold_regime(
    capital_intensity_ratio: float,
    diversification_benefit_pct: float,
    cross_unit_correlation: float,
    euler_raroc: float,
) -> str:
    """Classify institutional risk capital regime."""
    if capital_intensity_ratio >= 0.40:
        return "INSOLVENCY_TAIL_CAPITAL_TRAP"
    if cross_unit_correlation >= 0.85 or diversification_benefit_pct < 0.05:
        return "CONCENTRATED_CORRELATION_DRAG"
    if diversification_benefit_pct >= 0.20 and euler_raroc > 0.15:
        return "DIVERSIFICATION_SURPLUS_ACCUMULATOR"
    return "BALANCED_RISK_CAPITAL_BOOK"


def _build_hmac_payload(
    ticker: str,
    v1: float,
    vol1: float,
    v2: float,
    vol2: float,
    corr: float,
    horizon: float,
    standalone_rc_1: float,
    portfolio_rc: float,
    euler_rc_1: float,
    euler_raroc: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic colon-delimited payload for HMAC-SHA256 signing."""
    return (
        f"{ticker}:{v1:.4f}:{vol1:.4f}:{v2:.4f}:{vol2:.4f}:{corr:.4f}:{horizon:.4f}:"
        f"{standalone_rc_1:.6f}:{portfolio_rc:.6f}:{euler_rc_1:.6f}:{euler_raroc:.6f}:"
        f"{regime}:{timestamp}"
    )


def calculate_merton_perold_risk_capital(
    unit1_asset_v1: float,
    unit1_volatility: float,
    unit2_asset_v2: float = 100.0,
    unit2_volatility: float = 0.20,
    cross_unit_correlation: float = 0.25,
    horizon_years: float = 1.0,
    risk_free_rate_r: float = 0.05,
    expected_excess_return: float = 0.06,
    ticker: str = "JPM",
) -> MertonPeroldReceipt:
    """Compute Merton & Perold (1993) standalone, incremental, and Euler risk capital."""
    cleaned_ticker = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker format: {ticker}")

    v1 = float(unit1_asset_v1)
    v2 = float(unit2_asset_v2)
    sig1 = float(unit1_volatility)
    sig2 = float(unit2_volatility)
    corr = float(cross_unit_correlation)
    horizon = float(horizon_years)
    rate_r = float(risk_free_rate_r)
    excess_ret = float(expected_excess_return)

    if not (0.0 < v1 <= 1e12) or not (0.0 < v2 <= 1e12):
        raise ValueError("Unit asset values V1 and V2 must be in (0, 1e12].")
    if not (0.0 < sig1 <= 5.0) or not (0.0 < sig2 <= 5.0):
        raise ValueError("Volatilities must be in (0, 5.0].")
    if not (-0.9999 <= corr <= 1.0):
        raise ValueError("Cross-unit correlation must be in [-0.9999, 1.0].")
    if not (0.0 < horizon <= 30.0):
        raise ValueError("Horizon T must be in (0, 30.0] years.")
    if not (-0.05 <= rate_r <= 0.50):
        raise ValueError("Risk-free rate r must be in [-0.05, 0.50].")
    if not (-0.50 <= excess_ret <= 1.00):
        raise ValueError("Expected excess return must be in [-0.50, 1.00].")

    rc_1 = _standalone_risk_capital(v1, sig1, horizon)
    rc_2 = _standalone_risk_capital(v2, sig2, horizon)
    sum_standalone = rc_1 + rc_2

    v_p = v1 + v2
    var_dollar = (v1 * sig1) ** 2 + (v2 * sig2) ** 2 + 2.0 * corr * v1 * v2 * sig1 * sig2
    s_p = math.sqrt(max(1e-18, var_dollar))
    vol_p = s_p / v_p
    rc_p = _standalone_risk_capital(v_p, vol_p, horizon)

    marg_rc_1 = rc_p - rc_2
    marg_rc_2 = rc_p - rc_1
    div_benefit = max(0.0, sum_standalone - rc_p)
    unalloc_gap = rc_p - (marg_rc_1 + marg_rc_2)

    sqrt_t = math.sqrt(horizon)
    z_p = 0.5 * vol_p * sqrt_t
    base_frac = 2.0 * _norm_cdf(z_p) - 1.0
    pdf_z = _norm_pdf(z_p)

    euler_1 = _euler_unit_allocation(
        v_unit=v1,
        sig_unit=sig1,
        v_other=v2,
        sig_other=sig2,
        correlation=corr,
        sqrt_t=sqrt_t,
        s_p=s_p,
        v_p=v_p,
        base_frac=base_frac,
        pdf_z=pdf_z,
    )
    euler_2 = _euler_unit_allocation(
        v_unit=v2,
        sig_unit=sig2,
        v_other=v1,
        sig_other=sig1,
        correlation=corr,
        sqrt_t=sqrt_t,
        s_p=s_p,
        v_p=v_p,
        base_frac=base_frac,
        pdf_z=pdf_z,
    )

    cap_intensity = rc_p / v_p
    div_pct = div_benefit / max(1e-12, sum_standalone)

    exp_profit = v1 * excess_ret * horizon
    standalone_raroc = exp_profit / max(1e-9, rc_1)
    euler_raroc = (exp_profit / euler_1) if euler_1 > 1e-6 else standalone_raroc

    regime = classify_merton_perold_regime(
        capital_intensity_ratio=cap_intensity,
        diversification_benefit_pct=div_pct,
        cross_unit_correlation=corr,
        euler_raroc=euler_raroc,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        v1=v1,
        vol1=sig1,
        v2=v2,
        vol2=sig2,
        corr=corr,
        horizon=horizon,
        standalone_rc_1=rc_1,
        portfolio_rc=rc_p,
        euler_rc_1=euler_1,
        euler_raroc=euler_raroc,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return MertonPeroldReceipt(
        ticker=cleaned_ticker,
        unit1_asset_v1=v1,
        unit1_volatility=sig1,
        unit2_asset_v2=v2,
        unit2_volatility=sig2,
        cross_unit_correlation=corr,
        horizon_years=horizon,
        risk_free_rate_r=rate_r,
        expected_excess_return=excess_ret,
        standalone_rc_1=rc_1,
        standalone_rc_2=rc_2,
        sum_standalone_rc=sum_standalone,
        portfolio_volatility=vol_p,
        portfolio_rc=rc_p,
        marginal_rc_1=marg_rc_1,
        marginal_rc_2=marg_rc_2,
        diversification_benefit=div_benefit,
        unallocated_capital_gap=unalloc_gap,
        euler_rc_1=euler_1,
        euler_rc_2=euler_2,
        capital_intensity_ratio=cap_intensity,
        diversification_benefit_pct=div_pct,
        expected_excess_profit=exp_profit,
        standalone_raroc=standalone_raroc,
        euler_raroc=euler_raroc,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_merton_perold_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "JPM",
    horizon_years: float = 1.0,
    secondary_volatility: float = 0.20,
    cross_unit_correlation: float = 0.25,
    expected_excess_return: float = 0.06,
    risk_free_rate_r: float = 0.05,
) -> MertonPeroldReceipt:
    """Calibrate primary desk asset exposure and trailing volatility from close prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        v1 = float(max(1.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60) :])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.08, 1.50))
    else:
        v1 = 100.0
        ann_vol = 0.25

    return calculate_merton_perold_risk_capital(
        unit1_asset_v1=v1,
        unit1_volatility=ann_vol,
        unit2_asset_v2=v1,
        unit2_volatility=float(secondary_volatility),
        cross_unit_correlation=float(cross_unit_correlation),
        horizon_years=float(horizon_years),
        risk_free_rate_r=float(risk_free_rate_r),
        expected_excess_return=float(expected_excess_return),
        ticker=ticker,
    )


def persist_merton_perold_record(
    receipt: MertonPeroldReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Merton & Perold (1993) audit record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS merton_perold_risk_capital_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        unit1_asset_v1 REAL NOT NULL,
        unit1_volatility REAL NOT NULL,
        cross_unit_correlation REAL NOT NULL,
        horizon_years REAL NOT NULL,
        standalone_rc_1 REAL NOT NULL,
        portfolio_rc REAL NOT NULL,
        marginal_rc_1 REAL NOT NULL,
        diversification_benefit REAL NOT NULL,
        euler_rc_1 REAL NOT NULL,
        euler_raroc REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO merton_perold_risk_capital_audits (
        ticker, unit1_asset_v1, unit1_volatility, cross_unit_correlation,
        horizon_years, standalone_rc_1, portfolio_rc, marginal_rc_1,
        diversification_benefit, euler_rc_1, euler_raroc, regime,
        hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.unit1_asset_v1,
                receipt.unit1_volatility,
                receipt.cross_unit_correlation,
                receipt.horizon_years,
                receipt.standalone_rc_1,
                receipt.portfolio_rc,
                receipt.marginal_rc_1,
                receipt.diversification_benefit,
                receipt.euler_rc_1,
                receipt.euler_raroc,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

