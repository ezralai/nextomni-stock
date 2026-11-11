"""Franco Modigliani & Merton H. Miller (1958, 1961, 1963) Capital Structure & Dividend Valuation Engine.

Theoretical Foundation:
    1. Modigliani, Franco, & Miller, Merton H. (1958).
       "The Cost of Capital, Corporation Finance and the Theory of Investment".
       The American Economic Review, 48(3), pp. 261-297.

    2. Miller, Merton H., & Modigliani, Franco (1961).
       "Dividend Policy, Growth, and the Valuation of Shares".
       The Journal of Business, 34(4), pp. 411-433.

    3. Modigliani, Franco, & Miller, Merton H. (1963).
       "Corporate Income Taxes and the Cost of Capital: A Correction".
       The American Economic Review, 53(3), pp. 433-443.

Key Concepts:
    - MM Propositions I & II with Corporate Taxes (tau_c):
      V_U = EBIT*(1 - tau_c) / r_0, V_L = V_U + tau_c*D,
      r_E = r_0 + (r_0 - r_D)*(1 - tau_c)*(D / E_L),
      r_WACC = (E_L/V_L)*r_E + (D/V_L)*r_D*(1 - tau_c) == r_0*(1 - tau_c*D/V_L).
    - MM (1961) Dividend Policy Irrelevance & PVGO:
      With real capital reinvestment rate k over horizon T earning incremental ROIC rho*,
      assets-in-place + growth valuation satisfies PVGO == 0 identically across all
      retention/payout choices when rho* == r_0, isolating true economic value creation
      exclusively in positive-NPV spread (rho* - r_0).
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
HMAC_SECRET = b"modigliani-miller-corporate-v156-secret"


@dataclass(frozen=True)
class ModiglianiMillerReceipt:
    """Immutable quantitative audit receipt for Modigliani & Miller (1958, 1961, 1963)."""

    ticker: str
    ebit: float
    debt_d: float
    unlevered_r0: float
    cost_of_debt_rd: float
    corporate_tax_rate_tc: float
    reinvestment_rate_k: float
    incremental_roic: float
    payout_ratio: float
    growth_horizon_years: int
    nopat: float
    unlevered_value_vu: float
    tax_shield_pv: float
    levered_value_vl: float
    levered_equity_el: float
    debt_to_equity_ratio: float
    levered_equity_cost_re: float
    wacc: float
    mm_growth_equity_value: float
    pvgo: float
    pvgo_share_pct: float
    dividend_irrelevance_wedge: float
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


def _compute_finite_horizon_growth_value(
    nopat_y1: float,
    r0: float,
    k_reinvest: float,
    roic: float,
    horizon_years: int,
) -> float:
    """Exact MM (1961) discounted free-cash-flow + terminal asset valuation over finite horizon."""
    pv_total = 0.0
    earnings_t = nopat_y1
    for t in range(1, horizon_years + 1):
        fcf_t = earnings_t * (1.0 - k_reinvest)
        pv_total += fcf_t / ((1.0 + r0) ** t)
        earnings_t = earnings_t * (1.0 + k_reinvest * roic)

    terminal_asset_val = earnings_t / r0
    pv_total += terminal_asset_val / ((1.0 + r0) ** horizon_years)
    return float(pv_total)


def classify_modigliani_miller_regime(
    tax_rate_tc: float,
    debt_to_equity: float,
    roic: float,
    r0: float,
    pvgo_share_pct: float,
) -> str:
    """Classify capital-structure tax-shield and growth-value creation regime."""
    if tax_rate_tc <= 1e-6 and abs(roic - r0) <= 1e-4:
        return "TAX_FREE_MM_PROPOSITION_INVARIANCE"
    if roic < r0 - 0.015:
        return "NEGATIVE_SPREAD_VALUE_DESTRUCTIVE_GROWTH"
    if pvgo_share_pct >= 25.0 and roic > r0:
        return "HIGH_PVGO_COMPOUNDING_FRANCHISE"
    if debt_to_equity >= 1.50 and tax_rate_tc > 0.10:
        return "TAX_SHIELD_LEVERAGED_EXPANSION"
    return "BALANCED_WACC_MATURE_CASH_COW"


def _build_hmac_payload(
    ticker: str,
    ebit: float,
    debt: float,
    r0: float,
    tc: float,
    k_reinvest: float,
    roic: float,
    vl: float,
    wacc: float,
    pvgo: float,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{ebit:.4f}:{debt:.4f}:{r0:.6f}:{tc:.4f}:{k_reinvest:.4f}:"
        f"{roic:.6f}:{vl:.6f}:{wacc:.6f}:{pvgo:.6f}:{regime}:{timestamp}"
    )


def calculate_modigliani_miller_corporate(
    ebit: float = 20.0,
    debt_d: float = 50.0,
    unlevered_r0: float = 0.10,
    cost_of_debt_rd: float = 0.05,
    corporate_tax_rate_tc: float = 0.21,
    reinvestment_rate_k: float = 0.40,
    incremental_roic: float = 0.14,
    payout_ratio: float = 0.60,
    growth_horizon_years: int = 10,
    ticker: str = "MM61",
) -> ModiglianiMillerReceipt:
    """Evaluate Modigliani & Miller (1958, 1961, 1963) Propositions I/II, WACC & PVGO."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    ebit_val = float(ebit)
    debt = float(debt_d)
    r0 = float(unlevered_r0)
    rd = float(cost_of_debt_rd)
    tc = float(corporate_tax_rate_tc)
    k_inv = float(reinvestment_rate_k)
    roic = float(incremental_roic)
    payout = float(payout_ratio)
    horizon = int(growth_horizon_years)

    if ebit_val <= 0.0:
        raise ValueError("ebit must be strictly positive.")
    if debt < 0.0:
        raise ValueError("debt_d must be non-negative.")
    if r0 <= 0.0 or rd < 0.0:
        raise ValueError("unlevered_r0 must be positive and cost_of_debt_rd non-negative.")
    if not (0.0 <= tc < 1.0):
        raise ValueError("corporate_tax_rate_tc must satisfy 0.0 <= tc < 1.0.")
    if not (0.0 <= k_inv <= 1.0) or not (0.0 <= payout <= 1.0):
        raise ValueError("reinvestment_rate_k and payout_ratio must be in [0.0, 1.0].")
    if horizon < 1 or horizon > 50:
        raise ValueError("growth_horizon_years must be between 1 and 50.")

    nopat = ebit_val * (1.0 - tc)
    vu = nopat / r0
    tax_shield = tc * debt
    vl = vu + tax_shield

    if debt >= vl:
        raise ValueError("debt_d exceeds levered firm value V_L (insolvent capital structure).")

    el = vl - debt
    d_to_e = debt / el
    re = r0 + (r0 - rd) * (1.0 - tc) * d_to_e
    wacc = (el / vl) * re + (debt / vl) * rd * (1.0 - tc)

    unlevered_with_growth = _compute_finite_horizon_growth_value(
        nopat_y1=nopat,
        r0=r0,
        k_reinvest=k_inv,
        roic=roic,
        horizon_years=horizon,
    )
    pvgo = unlevered_with_growth - vu
    mm_growth_eq = el + pvgo
    total_growth_firm = max(1e-8, vl + pvgo)
    pvgo_share = (pvgo / total_growth_firm) * 100.0

    # MM (1961): given real investment schedule k_inv, altering dividend payout vs share issuance is neutral (wedge == 0)
    div_wedge = 0.0

    regime = classify_modigliani_miller_regime(
        tax_rate_tc=tc,
        debt_to_equity=d_to_e,
        roic=roic,
        r0=r0,
        pvgo_share_pct=pvgo_share,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        ebit=ebit_val,
        debt=debt,
        r0=r0,
        tc=tc,
        k_reinvest=k_inv,
        roic=roic,
        vl=vl,
        wacc=wacc,
        pvgo=pvgo,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return ModiglianiMillerReceipt(
        ticker=cleaned_ticker,
        ebit=ebit_val,
        debt_d=debt,
        unlevered_r0=r0,
        cost_of_debt_rd=rd,
        corporate_tax_rate_tc=tc,
        reinvestment_rate_k=k_inv,
        incremental_roic=roic,
        payout_ratio=payout,
        growth_horizon_years=horizon,
        nopat=nopat,
        unlevered_value_vu=vu,
        tax_shield_pv=tax_shield,
        levered_value_vl=vl,
        levered_equity_el=el,
        debt_to_equity_ratio=d_to_e,
        levered_equity_cost_re=re,
        wacc=wacc,
        mm_growth_equity_value=mm_growth_eq,
        pvgo=pvgo,
        pvgo_share_pct=pvgo_share,
        dividend_irrelevance_wedge=div_wedge,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_modigliani_miller_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "MM61",
    debt_d: float = 40.0,
    corporate_tax_rate_tc: float = 0.21,
    incremental_roic: float = 0.14,
) -> ModiglianiMillerReceipt:
    """Calibrate unlevered discount rate proxy from trailing asset volatility and evaluate MM framework."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot = float(max(20.0, p_arr[-1]))
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.std(log_rets) * math.sqrt(252.0))
        r0 = float(np.clip(0.06 + 0.15 * ann_vol, 0.065, 0.18))
        # Anchor EBIT proportional to spot scale so debt_d remains well-scaled
        ebit = max(12.0, spot * 0.20)
    else:
        r0 = 0.10
        ebit = 20.0

    return calculate_modigliani_miller_corporate(
        ebit=ebit,
        debt_d=float(debt_d),
        unlevered_r0=r0,
        corporate_tax_rate_tc=float(corporate_tax_rate_tc),
        incremental_roic=float(incremental_roic),
        ticker=ticker,
    )


def persist_modigliani_miller_record(
    receipt: ModiglianiMillerReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Modigliani & Miller corporate valuation record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS modigliani_miller_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        ebit REAL NOT NULL,
        debt_d REAL NOT NULL,
        unlevered_r0 REAL NOT NULL,
        corporate_tax_rate_tc REAL NOT NULL,
        unlevered_value_vu REAL NOT NULL,
        tax_shield_pv REAL NOT NULL,
        levered_value_vl REAL NOT NULL,
        levered_equity_cost_re REAL NOT NULL,
        wacc REAL NOT NULL,
        pvgo REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO modigliani_miller_audits (
        ticker, ebit, debt_d, unlevered_r0, corporate_tax_rate_tc,
        unlevered_value_vu, tax_shield_pv, levered_value_vl,
        levered_equity_cost_re, wacc, pvgo, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.ebit,
                receipt.debt_d,
                receipt.unlevered_r0,
                receipt.corporate_tax_rate_tc,
                receipt.unlevered_value_vu,
                receipt.tax_shield_pv,
                receipt.levered_value_vl,
                receipt.levered_equity_cost_re,
                receipt.wacc,
                receipt.pvgo,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

