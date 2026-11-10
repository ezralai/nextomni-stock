"""Michael C. Jensen & William H. Meckling (1976) Agency Costs & Asset Substitution Engine.

Theoretical Foundation:
    Jensen, Michael C., & Meckling, William H. (1976). "Theory of the Firm: Managerial
    Behavior, Agency Costs and Ownership Structure". Journal of Financial Economics,
    3(4), pp. 305-360. The foundational paper establishing modern corporate governance,
    the agency theory of outside equity, and bondholder-stockholder conflict.

Key Concepts:
    1. Agency Costs of Outside Equity (Perk Consumption & Shirking):
       When an owner-manager sells fraction (1 - alpha) of equity to outside investors,
       the manager captures 100% of private perquisite utility U(F) while bearing only
       fraction alpha of the dollar cost F, inducing overconsumption F*(alpha) > F*(1)
       and a deadweight residual firm value loss AC_equity(alpha).

    2. Agency Costs of Debt (Asset Substitution / Risk-Shifting):
       Levered equity over debt obligation D behaves as a European Call Option E(V, sigma).
       Because option value increases monotonically with volatility (Vega > 0), equity
       holders have an incentive to substitute a safe positive-NPV project (V_L, sigma_L)
       with a lower-NPV high-variance gamble (V_H < V_L, sigma_H > sigma_L) whenever
       E(V_H, sigma_H) > E(V_L, sigma_L), expropriating creditor wealth at the cost of
       total enterprise value destruction.
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
HMAC_SECRET = b"jensen-meckling-agency-v138-secret"


@dataclass(frozen=True)
class JensenMecklingReceipt:
    """Immutable quantitative audit receipt for Jensen & Meckling (1976) corporate agency costs."""

    ticker: str
    firm_potential_value: float         # V_0
    insider_ownership_alpha: float      # alpha in (0, 1]
    debt_face_value: float              # D
    safe_volatility: float              # sigma_L
    risky_volatility: float             # sigma_H
    risky_npv_discount_pct: float       # (V_L - V_H) / V_L
    optimal_perk_consumption: float     # F*(1) first-best perks
    actual_perk_consumption: float      # F*(alpha) induced perks
    equity_agency_loss: float           # AC_equity(alpha) deadweight loss
    safe_equity_value: float            # E(V_L, sigma_L)
    safe_debt_value: float              # D_L = V_L - E_L
    risky_equity_value: float           # E(V_H, sigma_H)
    risky_debt_value: float             # D_H = V_H - E_H
    risk_shifting_triggered: bool       # E_H > E_L and V_L > V_H
    equity_risk_shifting_gain: float    # max(0, E_H - E_L)
    creditor_wealth_expropriation: float# max(0, D_L - D_H) when shifted
    asset_substitution_loss: float      # V_L - V_H when shifted
    total_agency_cost: float            # equity_agency_loss + asset_substitution_loss
    agency_cost_pct_of_value: float     # total_agency_cost / V_0
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


def _bs_equity_call(
    asset_val: float,
    debt_face: float,
    rate: float,
    horizon: float,
    sigma: float,
) -> float:
    """Black-Scholes-Merton call valuation of levered equity over debt face D."""
    if debt_face <= 1e-10:
        return asset_val
    if horizon <= 1e-8 or sigma <= 1e-8:
        return max(0.0, asset_val - debt_face * math.exp(-rate * horizon))
    sqrt_t = math.sqrt(horizon)
    d1 = (math.log(asset_val / debt_face) + (rate + 0.5 * sigma * sigma) * horizon) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return max(0.0, asset_val * _norm_cdf(d1) - debt_face * math.exp(-rate * horizon) * _norm_cdf(d2))


def classify_agency_regime(
    insider_alpha: float,
    risk_shifting_triggered: bool,
    agency_cost_pct: float,
) -> str:
    """Classify the Jensen-Meckling corporate governance and agency conflict regime."""
    if risk_shifting_triggered and agency_cost_pct >= 0.08:
        return "SEVERE_ASSET_SUBSTITUTION_RISK_SHIFTING"
    if risk_shifting_triggered:
        return "DEBT_RISK_SHIFTING_EXPROPRIATION"
    if insider_alpha >= 0.999 and agency_cost_pct < 0.01:
        return "ALIGNED_OWNER_MANAGER_FIRST_BEST"
    if agency_cost_pct >= 0.05:
        return "OUTSIDE_EQUITY_PERK_DILUTION_HIGH"
    return "GOVERNANCE_MONITORED_LOW_AGENCY"


def calculate_jensen_meckling_equilibrium(
    firm_potential_value: float = 100.0,
    insider_ownership_alpha: float = 0.40,
    debt_face_value: float = 75.0,
    safe_volatility: float = 0.20,
    risky_volatility: float = 0.55,
    risky_npv_discount_pct: float = 0.06,
    perk_propensity: float = 0.12,
    risk_free_rate: float = 0.04,
    debt_horizon_years: float = 2.0,
    ticker: str = "FIRM",
) -> JensenMecklingReceipt:
    """Compute Jensen & Meckling (1976) agency costs of equity and debt asset substitution.

    Args:
        firm_potential_value: Unconstrained firm asset potential V_0 (> 0).
        insider_ownership_alpha: Fraction of equity retained by manager alpha in (0.05, 1.0].
        debt_face_value: Debt obligation face value D (>= 0).
        safe_volatility: Volatility of prudent core operating project sigma_L (> 0).
        risky_volatility: Volatility of high-risk substitute gamble sigma_H (>= sigma_L).
        risky_npv_discount_pct: Fractional asset value lost by undertaking risky gamble (V_L - V_H)/V_L.
        perk_propensity: Managerial private perquisite utility appetite parameter (>= 0).
        risk_free_rate: Continuously compounded risk-free rate r.
        debt_horizon_years: Effective debt maturity T in years (> 0).
        ticker: Security ticker symbol.

    Returns:
        JensenMecklingReceipt quantifying outside equity deadweight loss, asset substitution, and audit info.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "FIRM"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    v0 = float(firm_potential_value)
    if not math.isfinite(v0) or v0 <= 0.0:
        raise ValueError(f"firm_potential_value must be finite and positive, got: {firm_potential_value}")

    alpha = float(np.clip(insider_ownership_alpha, 0.05, 1.0))
    d_face = max(0.0, float(debt_face_value))
    sig_l = max(0.05, float(safe_volatility))
    sig_h = max(sig_l, float(risky_volatility))
    npv_disc = float(np.clip(risky_npv_discount_pct, 0.0, 0.50))
    c_perk = max(0.0, float(perk_propensity))
    r = float(risk_free_rate)
    t = max(0.1, float(debt_horizon_years))

    # 1. Agency Costs of Outside Equity (Jensen & Meckling 1976, Section 2):
    # Private perk utility U(F) = 2 * c_perk * sqrt(V_0 * F)
    # Optimal first-best (alpha = 1): d/dF [V_0 - F + U(F)] = 0 -> F*(1) = c_perk^2 * V_0
    # Induced consumption at fractional ownership alpha: F*(alpha) = min(0.8 * V_0, c_perk^2 * V_0 / alpha^2)
    f_opt = min(0.80 * v0, (c_perk ** 2) * v0)
    f_act = min(0.80 * v0, ((c_perk ** 2) * v0) / (alpha ** 2))

    def _total_surplus(f_val: float) -> float:
        return (v0 - f_val) + 2.0 * c_perk * math.sqrt(v0 * max(0.0, f_val))

    equity_agency_loss = max(0.0, _total_surplus(f_opt) - _total_surplus(f_act))

    # 2. Agency Costs of Debt: Asset Substitution / Risk-Shifting (Section 4):
    v_safe = max(1.0, v0 - (f_act - f_opt))
    v_risky = max(0.5, v_safe * (1.0 - npv_disc))

    e_safe = _bs_equity_call(v_safe, d_face, r, t, sig_l)
    d_safe = max(0.0, v_safe - e_safe)

    e_risky = _bs_equity_call(v_risky, d_face, r, t, sig_h)
    d_risky = max(0.0, v_risky - e_risky)

    # Risk-shifting occurs if equity call value is higher under risky project despite lower total NPV
    if e_risky > (e_safe + 1e-8) and v_safe > v_risky:
        risk_shifting = True
        eq_gain = e_risky - e_safe
        cred_exprop = max(0.0, d_safe - d_risky)
        sub_loss = v_safe - v_risky
    else:
        risk_shifting = False
        eq_gain = 0.0
        cred_exprop = 0.0
        sub_loss = 0.0

    total_agency = equity_agency_loss + sub_loss
    agency_pct = total_agency / v0

    regime = classify_agency_regime(
        insider_alpha=alpha,
        risk_shifting_triggered=risk_shifting,
        agency_cost_pct=agency_pct,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{v0:.4f}:{alpha:.4f}:{d_face:.4f}:{sig_l:.4f}:{sig_h:.4f}:"
        f"{equity_agency_loss:.4f}:{sub_loss:.4f}:{total_agency:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return JensenMecklingReceipt(
        ticker=cleaned_ticker,
        firm_potential_value=v0,
        insider_ownership_alpha=alpha,
        debt_face_value=d_face,
        safe_volatility=sig_l,
        risky_volatility=sig_h,
        risky_npv_discount_pct=npv_disc,
        optimal_perk_consumption=f_opt,
        actual_perk_consumption=f_act,
        equity_agency_loss=equity_agency_loss,
        safe_equity_value=e_safe,
        safe_debt_value=d_safe,
        risky_equity_value=e_risky,
        risky_debt_value=d_risky,
        risk_shifting_triggered=risk_shifting,
        equity_risk_shifting_gain=eq_gain,
        creditor_wealth_expropriation=cred_exprop,
        asset_substitution_loss=sub_loss,
        total_agency_cost=total_agency,
        agency_cost_pct_of_value=agency_pct,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_jensen_meckling_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "FIRM",
    insider_alpha: float = 0.40,
    risky_volatility: float = 0.55,
    debt_to_asset_ratio: float = 0.70,
) -> JensenMecklingReceipt:
    """Empirically calibrate Jensen-Meckling agency risk profile from equity price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        sig_l = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.50))
    else:
        spot = 100.0
        sig_l = 0.22

    v0 = spot / max(0.25, 1.0 - float(np.clip(debt_to_asset_ratio, 0.0, 0.75)))
    d_face = v0 * float(np.clip(debt_to_asset_ratio, 0.0, 0.90))

    return calculate_jensen_meckling_equilibrium(
        firm_potential_value=v0,
        insider_ownership_alpha=insider_alpha,
        debt_face_value=d_face,
        safe_volatility=sig_l,
        risky_volatility=max(sig_l + 0.15, float(risky_volatility)),
        risky_npv_discount_pct=0.06,
        ticker=ticker,
    )


def persist_jensen_meckling_record(
    receipt: JensenMecklingReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Jensen-Meckling agency cost record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS jensen_meckling_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        firm_potential_value REAL NOT NULL,
        insider_ownership_alpha REAL NOT NULL,
        debt_face_value REAL NOT NULL,
        safe_volatility REAL NOT NULL,
        risky_volatility REAL NOT NULL,
        risky_npv_discount_pct REAL NOT NULL,
        optimal_perk_consumption REAL NOT NULL,
        actual_perk_consumption REAL NOT NULL,
        equity_agency_loss REAL NOT NULL,
        safe_equity_value REAL NOT NULL,
        safe_debt_value REAL NOT NULL,
        risky_equity_value REAL NOT NULL,
        risky_debt_value REAL NOT NULL,
        risk_shifting_triggered INTEGER NOT NULL,
        equity_risk_shifting_gain REAL NOT NULL,
        creditor_wealth_expropriation REAL NOT NULL,
        asset_substitution_loss REAL NOT NULL,
        total_agency_cost REAL NOT NULL,
        agency_cost_pct_of_value REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_jm_ticker_time ON jensen_meckling_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO jensen_meckling_audits (
        ticker, firm_potential_value, insider_ownership_alpha, debt_face_value,
        safe_volatility, risky_volatility, risky_npv_discount_pct,
        optimal_perk_consumption, actual_perk_consumption, equity_agency_loss,
        safe_equity_value, safe_debt_value, risky_equity_value, risky_debt_value,
        risk_shifting_triggered, equity_risk_shifting_gain, creditor_wealth_expropriation,
        asset_substitution_loss, total_agency_cost, agency_cost_pct_of_value,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.firm_potential_value,
                    receipt.insider_ownership_alpha,
                    receipt.debt_face_value,
                    receipt.safe_volatility,
                    receipt.risky_volatility,
                    receipt.risky_npv_discount_pct,
                    receipt.optimal_perk_consumption,
                    receipt.actual_perk_consumption,
                    receipt.equity_agency_loss,
                    receipt.safe_equity_value,
                    receipt.safe_debt_value,
                    receipt.risky_equity_value,
                    receipt.risky_debt_value,
                    1 if receipt.risk_shifting_triggered else 0,
                    receipt.equity_risk_shifting_gain,
                    receipt.creditor_wealth_expropriation,
                    receipt.asset_substitution_loss,
                    receipt.total_agency_cost,
                    receipt.agency_cost_pct_of_value,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

