"""Philippe Aghion & Patrick Bolton (1992) / Oliver Hart & John Moore (1994) Control Rights & Strategic Default Engine.

Theoretical Foundation:
    1. Aghion, Philippe, & Bolton, Patrick (1992). "An Incomplete Contracts Approach to
       Financial Contracting". The Review of Economic Studies, 59(3), pp. 473-494.
       Foundational theory of contingent control allocation: control transfers from equity
       to creditors when cash flow thresholds are breached.

    2. Hart, Oliver, & Moore, John (1994). "A Theory of Debt Based on the Inalienability
       of Human Capital". The Quarterly Journal of Economics, 109(4), pp. 841-879.
       Establishes that inalienable human capital empowers borrowers to default strategically,
       forcing debt renegotiation down to the asset's outside liquidation value.

    3. Hart, Oliver, & Moore, John (1998). "Default and Renegotiation: A Dynamic Model of Debt".
       The Quarterly Journal of Economics, 113(1), pp. 1-41.
       Formalizes Nash bargaining over continuation surplus vs. liquidation threat points.

Key Concepts:
    - State-Contingent Control Shift:
      When cash flow y >= D, manager/equity holds residual control and private benefits B.
      When y < D, liquidity default shifts residual control to senior creditors.
    - Strategic Default Incentive:
      Borrowers strategically threaten default whenever renegotiated payment D* < D,
      extracting rents bounded by the creditor's outside liquidation value L.
    - Nash Debt Renegotiation:
      D* = min(D, alpha * y + (1 - alpha) * L), where alpha is creditor bargaining power.
    - Efficient Liquidation vs. Value-Destroying Foreclosure:
      Assets are liquidated if L >= y + B, otherwise continuation surplus is preserved.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import re
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
HMAC_SECRET = b"aghion-bolton-control-rights-v135-secret"


@dataclass(frozen=True)
class AghionBoltonReceipt:
    """Immutable quantitative audit receipt for Aghion-Bolton & Hart-Moore control rights equilibrium."""

    ticker: str
    realized_cash_flow: float
    debt_face_value: float
    liquidation_value: float
    private_benefit: float
    creditor_bargaining_power: float
    contractual_control_holder: str
    renegotiated_debt_service: float
    debt_haircut_pct: float
    strategic_default_triggered: bool
    liquidation_triggered: bool
    creditor_payoff: float
    entrepreneur_payoff: float
    total_surplus_realized: float
    first_best_surplus: float
    efficiency_loss: float
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


def classify_control_rights_regime(
    control_holder: str,
    strategic_default: bool,
    liquidation_triggered: bool,
    debt_haircut_pct: float,
) -> str:
    """Classify the corporate governance and debt contracting regime."""
    if liquidation_triggered:
        return "EFFICIENT_CREDITOR_LIQUIDATION"
    if strategic_default:
        return "STRATEGIC_DEFAULT_INALIENABLE_HUMAN_CAPITAL"
    if control_holder == "CREDITOR_DEBT":
        if debt_haircut_pct >= 0.25:
            return "CREDITOR_CONTROL_SEVERE_RENEGOTIATION"
        return "CREDITOR_CONTROL_WORKOUT"
    return "EQUITY_CONTROL_SOLVENT_OPERATION"


def calculate_aghion_bolton_equilibrium(
    cash_flow: float = 100.0,
    debt_face_value: float = 80.0,
    liquidation_value: float = 50.0,
    private_benefit: float = 15.0,
    creditor_bargaining_power: float = 0.50,
    ticker: str = "GOV",
) -> AghionBoltonReceipt:
    """Calculate Aghion-Bolton & Hart-Moore control rights, strategic default, and renegotiation equilibrium.

    Args:
        cash_flow: Verifiable monetary output y (>= 0).
        debt_face_value: Contractual debt obligation D (> 0).
        liquidation_value: Asset liquidation threat point value L (>= 0).
        private_benefit: Non-monetary benefit B accruing to entrepreneur under control (>= 0).
        creditor_bargaining_power: Creditor Nash bargaining share alpha in [0, 1].
        ticker: Asset ticker symbol.

    Returns:
        AghionBoltonReceipt containing state-contingent control, payoffs, and audit metrics.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "GOV"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    y = max(0.0, float(cash_flow))
    d = max(0.01, float(debt_face_value))
    l_val = max(0.0, float(liquidation_value))
    b = max(0.0, float(private_benefit))
    alpha = float(np.clip(creditor_bargaining_power, 0.0, 1.0))

    first_best_surplus = max(l_val, y + b)

    # 1. Contractual Control Shift (Aghion-Bolton 1992):
    # Solvency check: if y >= D, equity initially retains contractual control.
    if y >= d:
        control_holder = "ENTREPRENEUR_EQUITY"
        is_liquidity_default = False
    else:
        control_holder = "CREDITOR_DEBT"
        is_liquidity_default = True

    # 2. Strategic Default Condition (Hart-Moore 1994, 1998):
    # With inalienable human capital, the borrower can threaten to withhold labor.
    # Outside option of creditor is liquidation L. Cash flow under continuation is y.
    # Nash renegotiated debt claim: D* = min(D, alpha * y + (1 - alpha) * L)
    # The creditor can never extract more than total cash flow y.
    renegotiated_payment_cap = max(0.0, alpha * y + (1.0 - alpha) * min(y, l_val))
    d_star = min(d, renegotiated_payment_cap)

    # Strategic default occurs if debtor is solvent (y >= D) but can credibly force concessions (D* < D)
    if not is_liquidity_default and d_star < d:
        strategic_default = True
    else:
        strategic_default = False

    # 3. Liquidation vs Continuation Decision:
    # Under creditor control (or insolvency), creditors compare liquidation L vs continuation
    # If L > y + b, continuation is inefficient and creditor forecloses/liquidates.
    if is_liquidity_default and l_val > (y + b):
        liquidation = True
        creditor_payoff = min(d, l_val)
        entrepreneur_payoff = 0.0
        debt_service = creditor_payoff
    else:
        liquidation = False
        # Project continues under renegotiated or contractual debt service
        debt_service = d_star if (strategic_default or is_liquidity_default) else d
        creditor_payoff = debt_service
        # Entrepreneur captures remaining cash plus private non-monetary benefit B
        entrepreneur_payoff = max(0.0, y - debt_service) + b

    debt_haircut_pct = float(np.clip((d - debt_service) / d, 0.0, 1.0))
    total_surplus_realized = creditor_payoff + entrepreneur_payoff
    efficiency_loss = max(0.0, first_best_surplus - total_surplus_realized)

    regime = classify_control_rights_regime(
        control_holder=control_holder,
        strategic_default=strategic_default,
        liquidation_triggered=liquidation,
        debt_haircut_pct=debt_haircut_pct,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{y:.4f}:{d:.4f}:{l_val:.4f}:{b:.4f}:{alpha:.4f}:"
        f"{debt_service:.4f}:{creditor_payoff:.4f}:{entrepreneur_payoff:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return AghionBoltonReceipt(
        ticker=cleaned_ticker,
        realized_cash_flow=y,
        debt_face_value=d,
        liquidation_value=l_val,
        private_benefit=b,
        creditor_bargaining_power=alpha,
        contractual_control_holder=control_holder,
        renegotiated_debt_service=debt_service,
        debt_haircut_pct=debt_haircut_pct,
        strategic_default_triggered=strategic_default,
        liquidation_triggered=liquidation,
        creditor_payoff=creditor_payoff,
        entrepreneur_payoff=entrepreneur_payoff,
        total_surplus_realized=total_surplus_realized,
        first_best_surplus=first_best_surplus,
        efficiency_loss=efficiency_loss,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_aghion_bolton_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "GOV",
    debt_ratio: float = 0.50,
    liquidation_ratio: float = 0.40,
    private_benefit_ratio: float = 0.15,
    creditor_power: float = 0.50,
) -> AghionBoltonReceipt:
    """Empirically evaluate Aghion-Bolton governance profile from historical prices and balance sheet proxies."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        # Cash flow proxy: ~10% enterprise cash yield
        cf = spot * 0.10
    else:
        cf = 100.0

    debt = cf * (debt_ratio / 0.50) if debt_ratio > 0 else cf * 0.80
    liq = cf * (liquidation_ratio / 0.50)
    p_ben = cf * private_benefit_ratio

    return calculate_aghion_bolton_equilibrium(
        cash_flow=cf,
        debt_face_value=debt,
        liquidation_value=liq,
        private_benefit=p_ben,
        creditor_bargaining_power=creditor_power,
        ticker=ticker,
    )


def persist_aghion_bolton_record(
    receipt: AghionBoltonReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Aghion-Bolton control rights record to SQLite database using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS aghion_bolton_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        realized_cash_flow REAL NOT NULL,
        debt_face_value REAL NOT NULL,
        liquidation_value REAL NOT NULL,
        private_benefit REAL NOT NULL,
        creditor_bargaining_power REAL NOT NULL,
        contractual_control_holder TEXT NOT NULL,
        renegotiated_debt_service REAL NOT NULL,
        debt_haircut_pct REAL NOT NULL,
        strategic_default_triggered INTEGER NOT NULL,
        liquidation_triggered INTEGER NOT NULL,
        creditor_payoff REAL NOT NULL,
        entrepreneur_payoff REAL NOT NULL,
        total_surplus_realized REAL NOT NULL,
        first_best_surplus REAL NOT NULL,
        efficiency_loss REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ab_ticker_time ON aghion_bolton_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO aghion_bolton_audits (
        ticker, realized_cash_flow, debt_face_value, liquidation_value,
        private_benefit, creditor_bargaining_power, contractual_control_holder,
        renegotiated_debt_service, debt_haircut_pct, strategic_default_triggered,
        liquidation_triggered, creditor_payoff, entrepreneur_payoff,
        total_surplus_realized, first_best_surplus, efficiency_loss,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.realized_cash_flow,
                    receipt.debt_face_value,
                    receipt.liquidation_value,
                    receipt.private_benefit,
                    receipt.creditor_bargaining_power,
                    receipt.contractual_control_holder,
                    receipt.renegotiated_debt_service,
                    receipt.debt_haircut_pct,
                    1 if receipt.strategic_default_triggered else 0,
                    1 if receipt.liquidation_triggered else 0,
                    receipt.creditor_payoff,
                    receipt.entrepreneur_payoff,
                    receipt.total_surplus_realized,
                    receipt.first_best_surplus,
                    receipt.efficiency_loss,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

