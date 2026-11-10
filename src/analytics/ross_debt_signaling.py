"""Stephen A. Ross (1977) Incentive Signaling via Debt Structure & Bankruptcy Penalty Engine.

Theoretical Foundation:
    Ross, Stephen A. (1977). "The Determination of Financial Structure: The Incentive-
    Signalling Approach". The Bell Journal of Economics, 8(1), pp. 23-40.
    The pioneering asymmetric information capital structure theory proving that managers
    use financial debt leverage (F) to credibly signal high future cash flows when
    default imposes personal managerial bankruptcy penalties (L).

Key Concepts:
    1. Managerial Incentive Contract & Single Crossing:
       Firm terminal cash flow X ~ Uniform[0, t], with private quality t >= t_min.
       Manager compensation puts weight gamma_0 on date-0 market value V_0(F) and gamma_1
       on date-1 payoff net of bankruptcy penalty L incurred with probability F / t:
           M_0(t, F) = gamma_0 * V_0(F) + [gamma_1 / (1 + r)] * (t / 2 - L * F / t)
       Because marginal bankruptcy probability (1 / t) falls strictly as firm quality t rises,
       higher-quality firms can signal profitably via higher debt F.

    2. Separating Signaling Equilibrium Schedule:
       With incentive ratio kappa = gamma_1 * L / gamma_0:
           t*(F)  = sqrt(t_min^2 + 4 * kappa * F)
           V_0(F) = t*(F) / [2 * (1 + r)]
       Optimal separating leverage for true quality t_true:
           F*(t_true) = max(0, (t_true^2 - t_min^2) / (4 * kappa))
       Demonstrating strictly positive firm-value response to leverage increases (dV_0 / dF > 0).
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
HMAC_SECRET = b"ross-debt-signaling-v145-secret"


@dataclass(frozen=True)
class RossSignalingReceipt:
    """Immutable quantitative audit receipt for Ross (1977) debt incentive signaling."""

    ticker: str
    floor_quality_t_min: float            # t_min (X ~ U[0, t_min])
    true_quality_t: float                 # t_true
    debt_face_value: float                # F
    bankruptcy_penalty_l: float           # L
    gamma_0_weight: float                 # Weight on date-0 market value
    gamma_1_weight: float                 # Weight on date-1 terminal payoff
    risk_free_rate: float                 # r
    kappa_incentive_coeff: float          # gamma_1 * L / gamma_0
    signaled_quality_t_star: float        # t*(F)
    signaled_market_value: float          # V_0(F)
    floor_market_value: float             # V_0(0)
    signaling_valuation_lift_pct: float   # (V_0(F) - V_0(0)) / V_0(0)
    marginal_value_dv_df: float           # dV_0 / dF
    optimal_separating_debt_f_star: float # F*(t_true)
    true_default_probability: float       # min(1, F / t_true)
    signaled_default_probability: float   # min(1, F / t*(F))
    mimicry_deterrence_gap: float         # M_0(t_true, F*) - M_0(t_true, F) >= 0
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


def _manager_utility(
    t_true: float,
    f_debt: float,
    t_min: float,
    kappa: float,
    g0: float,
    g1: float,
    l_pen: float,
    r: float,
) -> float:
    """Evaluate Ross (1977) expected discounted managerial payoff M_0(t_true, F)."""
    t_sig = math.sqrt((t_min ** 2) + 4.0 * kappa * f_debt)
    v0 = t_sig / (2.0 * (1.0 + r))
    p_bank = min(1.0, f_debt / max(1e-8, t_true))
    return g0 * v0 + (g1 / (1.0 + r)) * (0.5 * t_true - l_pen * p_bank)


def classify_ross_signaling_regime(
    debt_face: float,
    true_quality_t: float,
    signaled_t: float,
    true_default_prob: float,
) -> str:
    """Classify the Ross (1977) debt-signaling equilibrium and mimicry deterrence regime."""
    if debt_face <= 1e-8:
        return "UNLEVERED_LEMON_FLOOR_BASELINE"
    if true_default_prob >= 0.45:
        return "EXCESS_LEVERAGE_BANKRUPTCY_TRAP"
    if signaled_t > true_quality_t * 1.10:
        return "OVER_SIGNALED_COSTLY_MIMICRY_PENALIZED"
    if signaled_t >= true_quality_t * 0.90:
        return "TRUTHFUL_SEPARATING_DEBT_SIGNAL"
    return "CONSERVATIVE_UNDER_LEVERAGED_SIGNAL"


def calculate_ross_debt_signaling(
    floor_quality_t_min: float = 100.0,
    true_quality_t: float = 160.0,
    debt_face_value: float = 65.0,
    bankruptcy_penalty_l: float = 60.0,
    gamma_0_weight: float = 0.50,
    gamma_1_weight: float = 0.50,
    risk_free_rate: float = 0.04,
    ticker: str = "ROSS",
) -> RossSignalingReceipt:
    """Solve Ross (1977) incentive-signaling debt structure equilibrium and bankruptcy deterrence.

    Args:
        floor_quality_t_min: Lowest-quality cash-flow ceiling t_min (> 0).
        true_quality_t: Manager's privately observed true cash-flow ceiling t_true (>= t_min).
        debt_face_value: Proposed face value of debt F (>= 0).
        bankruptcy_penalty_l: Personal bankruptcy penalty suffered by manager if X < F (> 0).
        gamma_0_weight: Compensation weight on date-0 valuation (> 0).
        gamma_1_weight: Compensation weight on date-1 realized payoff (> 0).
        risk_free_rate: Risk-free interest rate r (>= 0).
        ticker: Security ticker symbol.

    Returns:
        RossSignalingReceipt with signaled firm value, separating leverage, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "ROSS"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    t_min = float(floor_quality_t_min)
    if not math.isfinite(t_min) or t_min <= 0.0:
        raise ValueError(f"floor_quality_t_min must be finite and positive, got: {floor_quality_t_min}")

    t_true = max(t_min, float(true_quality_t))
    f_val = max(0.0, float(debt_face_value))
    l_pen = max(0.01, float(bankruptcy_penalty_l))
    g0 = max(0.01, float(gamma_0_weight))
    g1 = max(0.01, float(gamma_1_weight))
    r = max(0.0, float(risk_free_rate))

    kappa = (g1 * l_pen) / g0
    t_sig = math.sqrt((t_min ** 2) + 4.0 * kappa * f_val)
    v_sig = t_sig / (2.0 * (1.0 + r))
    v_floor = t_min / (2.0 * (1.0 + r))
    lift_pct = (v_sig - v_floor) / max(1e-8, v_floor)

    # Marginal signaling response: dV_0 / dF = kappa / [(1 + r) * t*(F)]
    dv_df = kappa / ((1.0 + r) * t_sig)

    # Optimal separating debt F*(t_true) maximizing M_0(t_true, F)
    f_opt = max(0.0, ((t_true ** 2) - (t_min ** 2)) / (4.0 * kappa))

    p_def_true = min(1.0, f_val / t_true)
    p_def_sig = min(1.0, f_val / t_sig)

    m_opt = _manager_utility(t_true, f_opt, t_min, kappa, g0, g1, l_pen, r)
    m_cur = _manager_utility(t_true, f_val, t_min, kappa, g0, g1, l_pen, r)
    deterrence_gap = max(0.0, m_opt - m_cur)

    regime = classify_ross_signaling_regime(
        debt_face=f_val,
        true_quality_t=t_true,
        signaled_t=t_sig,
        true_default_prob=p_def_true,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{t_min:.4f}:{t_true:.4f}:{f_val:.4f}:{l_pen:.4f}:"
        f"{v_sig:.4f}:{f_opt:.4f}:{p_def_true:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return RossSignalingReceipt(
        ticker=cleaned_ticker,
        floor_quality_t_min=t_min,
        true_quality_t=t_true,
        debt_face_value=f_val,
        bankruptcy_penalty_l=l_pen,
        gamma_0_weight=g0,
        gamma_1_weight=g1,
        risk_free_rate=r,
        kappa_incentive_coeff=kappa,
        signaled_quality_t_star=t_sig,
        signaled_market_value=v_sig,
        floor_market_value=v_floor,
        signaling_valuation_lift_pct=lift_pct,
        marginal_value_dv_df=dv_df,
        optimal_separating_debt_f_star=f_opt,
        true_default_probability=p_def_true,
        signaled_default_probability=p_def_sig,
        mimicry_deterrence_gap=deterrence_gap,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_ross_signaling_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "ROSS",
    debt_to_price_ratio: float = 0.65,
    bankruptcy_penalty_ratio: float = 0.60,
) -> RossSignalingReceipt:
    """Empirically calibrate Ross (1977) debt-signaling equilibrium from price history."""
    p_arr = np.asarray(prices, dtype=float)
    spot = float(p_arr[-1]) if len(p_arr) >= 5 else 100.0

    # Baseline cash flow ceiling so expected floor ~ 0.8 * spot, true ceiling ~ 2.08 * spot
    t_min = spot * 1.60
    t_true = spot * 2.50
    debt_f = spot * max(0.0, float(debt_to_price_ratio))
    l_pen = spot * max(0.05, float(bankruptcy_penalty_ratio))

    return calculate_ross_debt_signaling(
        floor_quality_t_min=t_min,
        true_quality_t=t_true,
        debt_face_value=debt_f,
        bankruptcy_penalty_l=l_pen,
        gamma_0_weight=0.50,
        gamma_1_weight=0.50,
        risk_free_rate=0.04,
        ticker=ticker,
    )


def persist_ross_signaling_record(
    receipt: RossSignalingReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Ross (1977) debt-signaling record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS ross_signaling_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        floor_quality_t_min REAL NOT NULL,
        true_quality_t REAL NOT NULL,
        debt_face_value REAL NOT NULL,
        bankruptcy_penalty_l REAL NOT NULL,
        gamma_0_weight REAL NOT NULL,
        gamma_1_weight REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        kappa_incentive_coeff REAL NOT NULL,
        signaled_quality_t_star REAL NOT NULL,
        signaled_market_value REAL NOT NULL,
        floor_market_value REAL NOT NULL,
        signaling_valuation_lift_pct REAL NOT NULL,
        marginal_value_dv_df REAL NOT NULL,
        optimal_separating_debt_f_star REAL NOT NULL,
        true_default_probability REAL NOT NULL,
        signaled_default_probability REAL NOT NULL,
        mimicry_deterrence_gap REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ross_sig_ticker_time ON ross_signaling_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO ross_signaling_audits (
        ticker, floor_quality_t_min, true_quality_t, debt_face_value,
        bankruptcy_penalty_l, gamma_0_weight, gamma_1_weight, risk_free_rate,
        kappa_incentive_coeff, signaled_quality_t_star, signaled_market_value,
        floor_market_value, signaling_valuation_lift_pct, marginal_value_dv_df,
        optimal_separating_debt_f_star, true_default_probability,
        signaled_default_probability, mimicry_deterrence_gap, regime,
        hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.floor_quality_t_min,
                    receipt.true_quality_t,
                    receipt.debt_face_value,
                    receipt.bankruptcy_penalty_l,
                    receipt.gamma_0_weight,
                    receipt.gamma_1_weight,
                    receipt.risk_free_rate,
                    receipt.kappa_incentive_coeff,
                    receipt.signaled_quality_t_star,
                    receipt.signaled_market_value,
                    receipt.floor_market_value,
                    receipt.signaling_valuation_lift_pct,
                    receipt.marginal_value_dv_df,
                    receipt.optimal_separating_debt_f_star,
                    receipt.true_default_probability,
                    receipt.signaled_default_probability,
                    receipt.mimicry_deterrence_gap,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

