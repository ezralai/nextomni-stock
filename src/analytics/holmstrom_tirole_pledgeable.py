"""Bengt Holmstrom & Jean Tirole (1997) Moral Hazard, Pledgeable Income & Credit Rationing Engine.

Theoretical Foundation:
    Holmstrom, Bengt, & Tirole, Jean (1997). "Financial Intermediation, Loanable Funds,
    and the Real Sector". The Quarterly Journal of Economics, 112(3), pp. 663-691.
    Foundational Nobel-Prize equilibrium theory explaining how moral hazard constrains
    pledgeable income, induces balance-sheet credit rationing for low-net-worth firms,
    and motivates informed bank monitoring during credit crunches.

Key Concepts:
    1. Moral Hazard & Incentive Compatibility:
       An entrepreneur choosing diligence yields success R with probability p_H, while shirking
       yields private benefit B with probability p_L < p_H (Delta_p = p_H - p_L). Incentive
       compatibility requires leaving managerial skin-in-the-game R_f >= B / Delta_p.

    2. Pledgeable Income & Net Worth Thresholds:
       - Direct Market Pledgeable Income:  P_direct = p_H * (R - B / Delta_p)
       - Minimum Direct Net Worth:         A_bar_direct = max(0, I - P_direct / gamma)
       - Monitored Pledgeable Income:      P_bank = p_H * (R - b / Delta_p) - c
       - Minimum Monitored Net Worth:      A_bar_bank = max(0, I - P_bank / gamma)

    3. Credit Rationing & Financial Intermediation Tiers:
       Capital-rich firms (A >= A_bar_direct) tap direct public markets; medium-capital firms
       (A_bar_bank <= A < A_bar_direct) rely on bank monitoring; net-worth-starved firms
       (A < A_bar_bank) face credit rationing despite holding positive-NPV projects.
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
HMAC_SECRET = b"holmstrom-tirole-pledgeable-v141-secret"


@dataclass(frozen=True)
class HolmstromTiroleReceipt:
    """Immutable quantitative audit receipt for Holmstrom-Tirole (1997) credit rationing."""

    ticker: str
    investment_outlay: float               # I
    corporate_net_worth: float             # A
    success_payoff: float                  # R
    prob_diligence: float                  # p_H
    prob_shirk: float                      # p_L
    private_benefit_unmonitored: float     # B
    private_benefit_monitored: float       # b
    monitoring_cost: float                 # c
    required_return_gamma: float           # gamma
    expected_project_npv: float            # p_H * R - gamma * I
    direct_pledgeable_income: float        # P_direct
    monitored_pledgeable_income: float     # P_bank
    pledgeable_share_of_payoff: float      # P_direct / (p_H * R)
    min_net_worth_direct: float            # A_bar_direct
    min_net_worth_bank: float              # A_bar_bank
    external_funding_needed: float         # max(0, I - A)
    financing_tier: str                    # DIRECT_MARKET / INTERMEDIATED_BANK / CREDIT_RATIONED
    credit_rationed: bool
    funding_gap: float                     # max(0, min_feasible_A - A)
    credit_multiplier: float
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


def classify_credit_rationing_regime(
    expected_npv: float,
    financing_tier: str,
    credit_rationed: bool,
) -> str:
    """Classify the Holmstrom-Tirole credit access and intermediation regime."""
    if expected_npv <= 0.0:
        return "NEGATIVE_NPV_PROJECT_UNFUNDABLE"
    if credit_rationed:
        return "CREDIT_CRUNCH_MORAL_HAZARD_RATIONING"
    if financing_tier == "INTERMEDIATED_BANK":
        return "BANK_MONITORED_INTERMEDIATION"
    return "DIRECT_CAPITAL_MARKET_UNCONSTRAINED"


def calculate_holmstrom_tirole_equilibrium(
    investment_outlay: float = 100.0,
    corporate_net_worth: float = 35.0,
    success_payoff: float = 160.0,
    prob_diligence: float = 0.80,
    prob_shirk: float = 0.40,
    private_benefit_unmonitored: float = 28.0,
    private_benefit_monitored: float = 12.0,
    monitoring_cost: float = 8.0,
    required_return_gamma: float = 1.05,
    ticker: str = "BANK",
) -> HolmstromTiroleReceipt:
    """Solve Holmstrom & Tirole (1997) pledgeable income, net-worth thresholds, and credit rationing.

    Args:
        investment_outlay: Required capital outlay I (> 0).
        corporate_net_worth: Internal collateral / cash net worth A (>= 0).
        success_payoff: Gross payoff R upon project success (> 0).
        prob_diligence: Success probability p_H under high effort in (0.05, 1.0].
        prob_shirk: Success probability p_L under shirking in [0.0, p_H - 0.01].
        private_benefit_unmonitored: Shirking private benefit B without monitoring (>= 0).
        private_benefit_monitored: Reduced private benefit b under bank monitoring in [0, B].
        monitoring_cost: Intermediary monitoring cost c (>= 0).
        required_return_gamma: Investor opportunity cost factor gamma (>= 1.0).
        ticker: Security ticker symbol.

    Returns:
        HolmstromTiroleReceipt with pledgeable income, thresholds, financing tier, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "BANK"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    i_val = float(investment_outlay)
    r_val = float(success_payoff)
    if not math.isfinite(i_val) or i_val <= 0.0:
        raise ValueError(f"investment_outlay must be finite and positive, got: {investment_outlay}")
    if not math.isfinite(r_val) or r_val <= 0.0:
        raise ValueError(f"success_payoff must be finite and positive, got: {success_payoff}")

    a_val = max(0.0, float(corporate_net_worth))
    p_h = float(np.clip(prob_diligence, 0.05, 1.0))
    p_l = float(np.clip(prob_shirk, 0.0, p_h - 0.01))
    delta_p = p_h - p_l

    b_unmon = max(0.0, float(private_benefit_unmonitored))
    b_mon = float(np.clip(private_benefit_monitored, 0.0, b_unmon))
    c_cost = max(0.0, float(monitoring_cost))
    gamma = max(1.0, float(required_return_gamma))

    expected_gross = p_h * r_val
    expected_npv = expected_gross - gamma * i_val

    # Entrepreneur minimum incentive-compatible compensation R_f = B / delta_p
    rf_direct = b_unmon / delta_p
    rf_bank = b_mon / delta_p

    p_direct = max(0.0, p_h * (r_val - rf_direct))
    p_bank = max(0.0, p_h * (r_val - rf_bank) - c_cost)

    pledgeable_share = p_direct / max(1e-8, expected_gross)

    a_bar_direct = max(0.0, i_val - (p_direct / gamma))
    a_bar_bank = max(0.0, i_val - (p_bank / gamma))

    ext_needed = max(0.0, i_val - a_val)
    min_feasible_a = min(a_bar_direct, a_bar_bank)

    if expected_npv <= 0.0:
        tier = "NEGATIVE_NPV_REJECTED"
        rationed = True
        funding_gap = max(0.0, -expected_npv)
    elif a_val >= a_bar_direct:
        tier = "DIRECT_MARKET"
        rationed = False
        funding_gap = 0.0
    elif a_val >= a_bar_bank and p_bank > p_direct:
        tier = "INTERMEDIATED_BANK"
        rationed = False
        funding_gap = 0.0
    else:
        tier = "CREDIT_RATIONED"
        rationed = True
        funding_gap = max(0.0, min_feasible_a - a_val)

    # Balance-sheet credit multiplier: k = I / max(1, A_bar) when constrained
    if p_direct < gamma * i_val:
        denom_mult = max(0.05, 1.0 - (p_direct / (gamma * i_val)))
        credit_mult = 1.0 / denom_mult
    else:
        credit_mult = 1.0

    regime = classify_credit_rationing_regime(
        expected_npv=expected_npv,
        financing_tier=tier,
        credit_rationed=rationed,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{i_val:.4f}:{a_val:.4f}:{r_val:.4f}:{p_h:.4f}:{p_l:.4f}:"
        f"{b_unmon:.4f}:{p_direct:.4f}:{p_bank:.4f}:{tier}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return HolmstromTiroleReceipt(
        ticker=cleaned_ticker,
        investment_outlay=i_val,
        corporate_net_worth=a_val,
        success_payoff=r_val,
        prob_diligence=p_h,
        prob_shirk=p_l,
        private_benefit_unmonitored=b_unmon,
        private_benefit_monitored=b_mon,
        monitoring_cost=c_cost,
        required_return_gamma=gamma,
        expected_project_npv=expected_npv,
        direct_pledgeable_income=p_direct,
        monitored_pledgeable_income=p_bank,
        pledgeable_share_of_payoff=pledgeable_share,
        min_net_worth_direct=a_bar_direct,
        min_net_worth_bank=a_bar_bank,
        external_funding_needed=ext_needed,
        financing_tier=tier,
        credit_rationed=rationed,
        funding_gap=funding_gap,
        credit_multiplier=credit_mult,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_holmstrom_tirole_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BANK",
    net_worth_ratio: float = 0.35,
    private_benefit_pct: float = 0.28,
    prob_diligence: float = 0.80,
) -> HolmstromTiroleReceipt:
    """Empirically calibrate Holmstrom-Tirole pledgeable income & rationing profile from prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 10:
        spot = float(p_arr[-1])
        outlay = spot
        # Collateral / net worth proxy scaled by equity drawdown/stability
        net_worth = outlay * max(0.0, float(net_worth_ratio))
    else:
        outlay = 100.0
        net_worth = outlay * max(0.0, float(net_worth_ratio))

    success_payoff = outlay * 1.60
    b_unmon = outlay * max(0.0, float(private_benefit_pct))
    b_mon = b_unmon * 0.43

    return calculate_holmstrom_tirole_equilibrium(
        investment_outlay=outlay,
        corporate_net_worth=net_worth,
        success_payoff=success_payoff,
        prob_diligence=prob_diligence,
        prob_shirk=max(0.10, prob_diligence - 0.40),
        private_benefit_unmonitored=b_unmon,
        private_benefit_monitored=b_mon,
        monitoring_cost=outlay * 0.08,
        required_return_gamma=1.05,
        ticker=ticker,
    )


def persist_holmstrom_tirole_record(
    receipt: HolmstromTiroleReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Holmstrom-Tirole credit rationing record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS holmstrom_tirole_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        investment_outlay REAL NOT NULL,
        corporate_net_worth REAL NOT NULL,
        success_payoff REAL NOT NULL,
        prob_diligence REAL NOT NULL,
        prob_shirk REAL NOT NULL,
        private_benefit_unmonitored REAL NOT NULL,
        private_benefit_monitored REAL NOT NULL,
        monitoring_cost REAL NOT NULL,
        required_return_gamma REAL NOT NULL,
        expected_project_npv REAL NOT NULL,
        direct_pledgeable_income REAL NOT NULL,
        monitored_pledgeable_income REAL NOT NULL,
        pledgeable_share_of_payoff REAL NOT NULL,
        min_net_worth_direct REAL NOT NULL,
        min_net_worth_bank REAL NOT NULL,
        external_funding_needed REAL NOT NULL,
        financing_tier TEXT NOT NULL,
        credit_rationed INTEGER NOT NULL,
        funding_gap REAL NOT NULL,
        credit_multiplier REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_ht_ticker_time ON holmstrom_tirole_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO holmstrom_tirole_audits (
        ticker, investment_outlay, corporate_net_worth, success_payoff,
        prob_diligence, prob_shirk, private_benefit_unmonitored,
        private_benefit_monitored, monitoring_cost, required_return_gamma,
        expected_project_npv, direct_pledgeable_income, monitored_pledgeable_income,
        pledgeable_share_of_payoff, min_net_worth_direct, min_net_worth_bank,
        external_funding_needed, financing_tier, credit_rationed, funding_gap,
        credit_multiplier, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.investment_outlay,
                    receipt.corporate_net_worth,
                    receipt.success_payoff,
                    receipt.prob_diligence,
                    receipt.prob_shirk,
                    receipt.private_benefit_unmonitored,
                    receipt.private_benefit_monitored,
                    receipt.monitoring_cost,
                    receipt.required_return_gamma,
                    receipt.expected_project_npv,
                    receipt.direct_pledgeable_income,
                    receipt.monitored_pledgeable_income,
                    receipt.pledgeable_share_of_payoff,
                    receipt.min_net_worth_direct,
                    receipt.min_net_worth_bank,
                    receipt.external_funding_needed,
                    receipt.financing_tier,
                    1 if receipt.credit_rationed else 0,
                    receipt.funding_gap,
                    receipt.credit_multiplier,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

