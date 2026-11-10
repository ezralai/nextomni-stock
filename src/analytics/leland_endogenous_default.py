"""Hayne E. Leland (1994) Endogenous Default Barrier & Optimal Capital Structure Engine.

Theoretical Foundation:
    Leland, Hayne E. (1994). "Corporate Debt Value, Bond Covenants, and Optimal Capital Structure".
    The Journal of Finance, 49(4), pp. 1213-1252.
    Foundational continuous-time structural model deriving closed-form risky debt pricing,
    endogenous smooth-pasting bankruptcy triggers, credit spreads, and optimal leverage
    via the tradeoff between debt tax shields and deadweight bankruptcy costs.

Key Concepts:
    1. Endogenous Smooth-Pasting Default Barrier (V_B):
       Equity holders inject cash to service perpetual debt coupon C only until asset value V
       hits the endogenous default boundary V_B where dE/dV|_{V=V_B} = 0:
           X = 2r / sigma^2
           V_B = [(1 - tau) * C / r] * [X / (1 + X)]

    2. Closed-Form Perpetual Claims (for V > V_B):
       - Tax Shield Value:      TB(V) = (tau * C / r) * (1 - (V / V_B)^(-X))
       - Bankruptcy Costs:      BC(V) = alpha * V_B * (V / V_B)^(-X)
       - Levered Firm Value:    v(V)  = V + TB(V) - BC(V)
       - Risky Debt Value:      D(V)  = C/r + ((1 - alpha) * V_B - C/r) * (V / V_B)^(-X)
       - Equity Value:          E(V)  = v(V) - D(V)

    3. Closed-Form Optimal Capital Structure (C*, L*):
       Maximizing levered firm value v(V) with respect to coupon C balances marginal
       tax shields against marginal expected bankruptcy losses.
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
HMAC_SECRET = b"leland-endogenous-default-v136-secret"


@dataclass(frozen=True)
class LelandCapitalReceipt:
    """Immutable quantitative audit receipt for Leland (1994) endogenous default & capital structure."""

    ticker: str
    asset_value: float                  # V
    asset_volatility: float             # sigma
    risk_free_rate: float               # r
    corporate_tax_rate: float           # tau
    bankruptcy_cost_alpha: float        # alpha
    contractual_coupon: float           # C
    endogenous_default_barrier: float   # V_B
    default_probability_pv: float       # (V / V_B)^(-X)
    debt_value: float                   # D(V)
    equity_value: float                 # E(V)
    tax_shield_value: float             # TB(V)
    bankruptcy_cost_value: float        # BC(V)
    levered_firm_value: float           # v(V) = V + TB - BC
    leverage_ratio: float               # D(V) / v(V)
    credit_spread_bps: float            # (C/D(V) - r) * 10000
    optimal_coupon: float               # C*
    optimal_leverage_ratio: float       # L*
    optimal_firm_value: float           # v*(V)
    default_triggered: bool             # True if V <= V_B
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


def _compute_leland_claims(
    v: float,
    c: float,
    r: float,
    sigma: float,
    tau: float,
    alpha: float,
) -> tuple[float, float, float, float, float, float, float, bool]:
    """Helper computing closed-form Leland (1994) valuations for a given coupon C."""
    x_exp = (2.0 * r) / (sigma * sigma)
    if c <= 1e-12:
        return 0.0, 0.0, 0.0, v, 0.0, 0.0, v, False

    v_b = ((1.0 - tau) * c / r) * (x_exp / (1.0 + x_exp))
    if v <= v_b:
        debt_val = (1.0 - alpha) * v
        return v_b, 1.0, debt_val, 0.0, 0.0, alpha * v, (1.0 - alpha) * v, True

    ratio_log = math.log(v / v_b)
    exponent = max(-50.0, -x_exp * ratio_log)
    p_b = math.exp(exponent)

    tb_val = (tau * c / r) * (1.0 - p_b)
    bc_val = alpha * v_b * p_b
    firm_val = v + tb_val - bc_val
    debt_val = (c / r) + ((1.0 - alpha) * v_b - (c / r)) * p_b
    equity_val = max(0.0, firm_val - debt_val)
    return v_b, p_b, debt_val, equity_val, tb_val, bc_val, firm_val, False


def classify_leland_regime(
    default_triggered: bool,
    leverage_ratio: float,
    optimal_leverage_ratio: float,
    credit_spread_bps: float,
) -> str:
    """Classify the Leland (1994) structural credit & capital structure regime."""
    if default_triggered:
        return "ENDOGENOUS_DEFAULT_BANKRUPTCY_TRIGGERED"
    if credit_spread_bps >= 350.0 or leverage_ratio > optimal_leverage_ratio + 0.15:
        return "OVERLEVERAGED_HIGH_DISTRESS_SPREAD"
    if leverage_ratio < 0.15:
        return "CONSERVATIVE_UNDERLEVERED_TAX_FORFEIT"
    if abs(leverage_ratio - optimal_leverage_ratio) <= 0.10:
        return "OPTIMAL_TRADEOFF_CAPITAL_EQUILIBRIUM"
    return "INVESTMENT_GRADE_SOLVENT_LEVERAGE"


def calculate_leland_equilibrium(
    asset_value: float = 100.0,
    asset_volatility: float = 0.25,
    risk_free_rate: float = 0.05,
    corporate_tax_rate: float = 0.25,
    bankruptcy_cost_alpha: float = 0.35,
    coupon: float = 4.0,
    ticker: str = "DEBT",
) -> LelandCapitalReceipt:
    """Compute Leland (1994) endogenous default barrier, debt/equity valuation, and optimal leverage.

    Args:
        asset_value: Unlevered firm asset value V (> 0).
        asset_volatility: Annual asset return volatility sigma (> 0).
        risk_free_rate: Continuously compounded risk-free rate r (> 0).
        corporate_tax_rate: Effective corporate tax rate tau in [0, 0.90].
        bankruptcy_cost_alpha: Deadweight bankruptcy loss fraction alpha in [0, 1].
        coupon: Annual perpetual debt coupon C (>= 0).
        ticker: Security ticker symbol.

    Returns:
        LelandCapitalReceipt with closed-form valuations, optimal capital structure, and HMAC signature.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "DEBT"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    v = float(asset_value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"Asset value must be finite and positive, got: {asset_value}")

    sigma = max(0.05, float(asset_volatility))
    r = max(1e-4, float(risk_free_rate))
    tau = float(np.clip(corporate_tax_rate, 0.0, 0.90))
    alpha = float(np.clip(bankruptcy_cost_alpha, 0.0, 1.0))
    c = max(0.0, float(coupon))

    v_b, p_b, debt_val, equity_val, tb_val, bc_val, firm_val, default_triggered = _compute_leland_claims(
        v=v, c=c, r=r, sigma=sigma, tau=tau, alpha=alpha
    )

    leverage_ratio = debt_val / firm_val if firm_val > 1e-12 else 0.0
    if default_triggered:
        credit_spread_bps = 10000.0
    elif debt_val > 1e-8 and c > 0.0:
        yield_rate = c / debt_val
        credit_spread_bps = max(0.0, (yield_rate - r) * 10000.0)
    else:
        credit_spread_bps = 0.0

    # Closed-form optimal coupon C* maximizing firm value v(V):
    x_exp = (2.0 * r) / (sigma * sigma)
    if tau <= 1e-8:
        optimal_coupon = 0.0
        optimal_firm_value = v
        optimal_leverage = 0.0
    else:
        k_coeff = ((1.0 - tau) * x_exp) / (r * (1.0 + x_exp))
        denom_inner = tau * (1.0 + x_exp) + alpha * (1.0 - tau) * x_exp
        h_ratio = tau / denom_inner if denom_inner > 1e-12 else 0.0
        optimal_coupon = (v / k_coeff) * (h_ratio ** (1.0 / x_exp)) if h_ratio > 0.0 else 0.0
        _, _, opt_debt, _, _, _, optimal_firm_value, _ = _compute_leland_claims(
            v=v, c=optimal_coupon, r=r, sigma=sigma, tau=tau, alpha=alpha
        )
        optimal_leverage = opt_debt / optimal_firm_value if optimal_firm_value > 1e-12 else 0.0

    regime = classify_leland_regime(
        default_triggered=default_triggered,
        leverage_ratio=leverage_ratio,
        optimal_leverage_ratio=optimal_leverage,
        credit_spread_bps=credit_spread_bps,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{v:.4f}:{sigma:.4f}:{r:.4f}:{tau:.4f}:{alpha:.4f}:{c:.4f}:"
        f"{v_b:.4f}:{firm_val:.4f}:{optimal_leverage:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return LelandCapitalReceipt(
        ticker=cleaned_ticker,
        asset_value=v,
        asset_volatility=sigma,
        risk_free_rate=r,
        corporate_tax_rate=tau,
        bankruptcy_cost_alpha=alpha,
        contractual_coupon=c,
        endogenous_default_barrier=v_b,
        default_probability_pv=p_b,
        debt_value=debt_val,
        equity_value=equity_val,
        tax_shield_value=tb_val,
        bankruptcy_cost_value=bc_val,
        levered_firm_value=firm_val,
        leverage_ratio=leverage_ratio,
        credit_spread_bps=credit_spread_bps,
        optimal_coupon=optimal_coupon,
        optimal_leverage_ratio=optimal_leverage,
        optimal_firm_value=optimal_firm_value,
        default_triggered=default_triggered,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_leland_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "DEBT",
    corporate_tax_rate: float = 0.25,
    bankruptcy_cost_alpha: float = 0.35,
    target_leverage_proxy: float = 0.40,
    risk_free_rate: float = 0.05,
) -> LelandCapitalReceipt:
    """Empirically evaluate Leland (1994) capital structure profile from equity price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        sigma = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.85))
    else:
        spot = 100.0
        sigma = 0.25

    # Approximate total asset value from equity price and leverage proxy
    v_asset = spot / max(0.20, 1.0 - float(np.clip(target_leverage_proxy, 0.0, 0.75)))
    coupon_proxy = v_asset * float(np.clip(target_leverage_proxy, 0.0, 0.80)) * risk_free_rate

    return calculate_leland_equilibrium(
        asset_value=v_asset,
        asset_volatility=sigma,
        risk_free_rate=risk_free_rate,
        corporate_tax_rate=corporate_tax_rate,
        bankruptcy_cost_alpha=bankruptcy_cost_alpha,
        coupon=coupon_proxy,
        ticker=ticker,
    )


def persist_leland_record(
    receipt: LelandCapitalReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Leland (1994) capital structure record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS leland_capital_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        asset_value REAL NOT NULL,
        asset_volatility REAL NOT NULL,
        risk_free_rate REAL NOT NULL,
        corporate_tax_rate REAL NOT NULL,
        bankruptcy_cost_alpha REAL NOT NULL,
        contractual_coupon REAL NOT NULL,
        endogenous_default_barrier REAL NOT NULL,
        default_probability_pv REAL NOT NULL,
        debt_value REAL NOT NULL,
        equity_value REAL NOT NULL,
        tax_shield_value REAL NOT NULL,
        bankruptcy_cost_value REAL NOT NULL,
        levered_firm_value REAL NOT NULL,
        leverage_ratio REAL NOT NULL,
        credit_spread_bps REAL NOT NULL,
        optimal_coupon REAL NOT NULL,
        optimal_leverage_ratio REAL NOT NULL,
        optimal_firm_value REAL NOT NULL,
        default_triggered INTEGER NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_leland_ticker_time ON leland_capital_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO leland_capital_audits (
        ticker, asset_value, asset_volatility, risk_free_rate, corporate_tax_rate,
        bankruptcy_cost_alpha, contractual_coupon, endogenous_default_barrier,
        default_probability_pv, debt_value, equity_value, tax_shield_value,
        bankruptcy_cost_value, levered_firm_value, leverage_ratio, credit_spread_bps,
        optimal_coupon, optimal_leverage_ratio, optimal_firm_value, default_triggered,
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
                    receipt.asset_value,
                    receipt.asset_volatility,
                    receipt.risk_free_rate,
                    receipt.corporate_tax_rate,
                    receipt.bankruptcy_cost_alpha,
                    receipt.contractual_coupon,
                    receipt.endogenous_default_barrier,
                    receipt.default_probability_pv,
                    receipt.debt_value,
                    receipt.equity_value,
                    receipt.tax_shield_value,
                    receipt.bankruptcy_cost_value,
                    receipt.levered_firm_value,
                    receipt.leverage_ratio,
                    receipt.credit_spread_bps,
                    receipt.optimal_coupon,
                    receipt.optimal_leverage_ratio,
                    receipt.optimal_firm_value,
                    1 if receipt.default_triggered else 0,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

