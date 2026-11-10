"""Douglas W. Diamond (1984) Financial Intermediation & Delegated Monitoring Engine.

Theoretical Foundation:
    Diamond, Douglas W. (1984). "Financial Intermediation and Delegated Monitoring".
    The Review of Economic Studies, 51(3), pp. 393-414.
    Nobel-Prize theory proving why financial intermediaries arise naturally to eliminate
    duplicative monitoring costs (m * K) across atomistic depositors while resolving the
    "Who monitors the monitor?" delegation cost (d_N) through portfolio diversification
    across N borrowers.

Key Concepts:
    1. Costly Verification & Duplicative Monitoring Trap:
       With m small lenders per borrower and verification cost K, direct monitored lending
       wastes m * K per project, while direct unmonitored debt suffers expected deadweight
       bankruptcy penalty D_direct.

    2. Portfolio Diversification of Delegation Cost (d_N):
       Pooling N loans with pairwise return correlation rho shrinks portfolio return std dev:
           sigma_port(N, rho) = sigma * sqrt((1 + (N - 1) * rho) / N)
       Solving competitive deposit promise E[min(y_bar_N - K, H*)] = R_f * I drives
       non-pecuniary delegation cost d_N = penalty * (H* - R_f * I) -> 0 as N -> infinity
       (for rho < 1).

    3. Intermediation Viability Inequality:
       Delegated banking dominates all direct alternatives iff:
           K + d_N < min(m * K, D_direct)
       At N = 1, K + d_1 > D_direct (single-loan banking is dominated). High systemic loan
       correlation rho destroys diversification benefits and destabilizes delegation.
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
HMAC_SECRET = b"diamond-delegated-monitoring-v146-secret"


@dataclass(frozen=True)
class DiamondMonitoringReceipt:
    """Immutable quantitative audit receipt for Diamond (1984) delegated monitoring."""

    ticker: str
    num_borrowers_n: int
    depositors_per_firm_m: int
    monitoring_cost_k: float
    project_mean_payoff: float           # mu
    project_volatility: float            # sigma
    loan_correlation_rho: float          # rho in [0, 1]
    required_depositor_return: float     # R_f * I
    portfolio_volatility: float          # sigma_port(N, rho)
    direct_duplicative_cost: float       # m * K
    direct_unmonitored_cost: float       # D_direct
    delegation_cost_d_n: float           # d_N
    delegated_total_cost: float          # K + d_N
    optimal_deposit_face_value: float    # H*
    best_direct_cost: float              # min(m * K, D_direct)
    net_intermediation_surplus: float    # best_direct_cost - (K + d_N)
    delegated_intermediation_viable: bool
    diversification_efficiency_pct: float# 1 - sigma_port / sigma
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
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _expected_truncated_debt(mean_y: float, std_y: float, face_h: float) -> float:
    """Compute E[min(Y, H)] for Y ~ Normal(mean_y, std_y^2)."""
    if std_y <= 1e-10:
        return min(mean_y, face_h)
    z = (face_h - mean_y) / std_y
    expected_shortfall = (face_h - mean_y) * _norm_cdf(z) + std_y * _norm_pdf(z)
    return face_h - expected_shortfall


def _solve_deposit_face_and_penalty(
    mean_net: float,
    std_net: float,
    target_return: float,
    penalty_mult: float,
) -> tuple[float, float]:
    """Solve E[min(Y, H*)] = target_return and return (H*, penalty * E[max(0, H* - Y)])."""
    if mean_net <= target_return + 1e-9:
        # Infeasible to promise target_return even at H -> inf; large deadweight default cost
        return mean_net, penalty_mult * max(1.0, (target_return - mean_net) + std_net)

    low = target_return
    high = max(target_return + 1.0, mean_net + 4.0 * max(0.01, std_net))
    while _expected_truncated_debt(mean_net, std_net, high) < target_return and high < mean_net + 50.0 * max(1.0, std_net):
        high *= 2.0

    for _ in range(50):
        mid = 0.5 * (low + high)
        val = _expected_truncated_debt(mean_net, std_net, mid)
        if val < target_return:
            low = mid
        else:
            high = mid

    h_star = 0.5 * (low + high)
    shortfall = max(0.0, h_star - target_return)
    return h_star, penalty_mult * shortfall


def classify_intermediation_regime(
    delegated_viable: bool,
    num_borrowers_n: int,
    loan_correlation_rho: float,
    delegation_cost_d_n: float,
    monitoring_cost_k: float,
) -> str:
    """Classify the Diamond (1984) delegated monitoring and diversification regime."""
    if num_borrowers_n == 1:
        return "SINGLE_LOAN_NO_DIVERSIFICATION_UNVIABLE"
    if not delegated_viable and loan_correlation_rho >= 0.45:
        return "SYSTEMIC_CORRELATION_DELEGATION_BREAKDOWN"
    if not delegated_viable:
        return "DIRECT_FINANCING_DOMINATES_DELEGATION"
    if delegation_cost_d_n <= 0.15 * monitoring_cost_k:
        return "ASYMPTOTIC_ZERO_DELEGATION_BANK_DOMINANCE"
    return "DIVERSIFIED_INTERMEDIARY_VIABLE"


def calculate_diamond_delegated_monitoring(
    num_borrowers_n: int = 50,
    depositors_per_firm_m: int = 10,
    monitoring_cost_k: float = 2.0,
    project_mean_payoff: float = 125.0,
    project_volatility: float = 25.0,
    loan_correlation_rho: float = 0.05,
    required_depositor_return: float = 104.0,
    bankruptcy_penalty_mult: float = 1.60,
    ticker: str = "BANK",
) -> DiamondMonitoringReceipt:
    """Solve Diamond (1984) financial intermediation costs vs direct lending benchmarks.

    Args:
        num_borrowers_n: Number of firms N pooled by the financial intermediary (>= 1).
        depositors_per_firm_m: Number of small lenders m per borrower under direct finance (>= 1).
        monitoring_cost_k: Costly state verification outlay K per monitored project (>= 0).
        project_mean_payoff: Expected gross terminal output per loan mu (> required_depositor_return).
        project_volatility: Single-firm cash flow standard deviation sigma (> 0).
        loan_correlation_rho: Pairwise correlation rho between borrower cash flows in [0, 1].
        required_depositor_return: Competitive gross safe return R_f * I owed to depositors (> 0).
        bankruptcy_penalty_mult: Deadweight default/incentive multiplier (>= 1.0).
        ticker: Financial intermediary ticker identifier.

    Returns:
        DiamondMonitoringReceipt detailing delegation cost, net surplus, and HMAC seal.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "BANK"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    n = max(1, int(num_borrowers_n))
    m = max(1, int(depositors_per_firm_m))
    k = max(0.0, float(monitoring_cost_k))
    mu = float(project_mean_payoff)
    sig = max(0.01, float(project_volatility))
    rho = float(np.clip(loan_correlation_rho, 0.0, 1.0))
    req_ret = float(required_depositor_return)
    pen_mult = max(1.0, float(bankruptcy_penalty_mult))

    if not math.isfinite(mu) or not math.isfinite(req_ret) or req_ret <= 0.0 or mu <= 0.0:
        raise ValueError("project_mean_payoff and required_depositor_return must be finite positive numbers.")

    # Direct lending alternative 1: duplicative monitoring by m small investors
    direct_dup_cost = m * k

    # Direct lending alternative 2: unmonitored debt contracting (N = 1, zero monitoring K)
    _, direct_unmon_cost = _solve_deposit_face_and_penalty(
        mean_net=mu,
        std_net=sig,
        target_return=req_ret,
        penalty_mult=pen_mult,
    )

    best_direct = min(direct_dup_cost, direct_unmon_cost)

    # Delegated bank pooling N borrowers: portfolio volatility sigma_port(N, rho)
    port_vol = sig * math.sqrt((1.0 + (n - 1) * rho) / float(n))
    h_star, d_n = _solve_deposit_face_and_penalty(
        mean_net=mu - k,
        std_net=port_vol,
        target_return=req_ret,
        penalty_mult=pen_mult,
    )

    delegated_total = k + d_n
    net_surplus = best_direct - delegated_total
    viable = net_surplus > 1e-6
    div_eff = max(0.0, 1.0 - (port_vol / sig))

    regime = classify_intermediation_regime(
        delegated_viable=viable,
        num_borrowers_n=n,
        loan_correlation_rho=rho,
        delegation_cost_d_n=d_n,
        monitoring_cost_k=k,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{n}:{m}:{k:.4f}:{mu:.4f}:{sig:.4f}:{rho:.4f}:"
        f"{d_n:.4f}:{delegated_total:.4f}:{net_surplus:.4f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return DiamondMonitoringReceipt(
        ticker=cleaned_ticker,
        num_borrowers_n=n,
        depositors_per_firm_m=m,
        monitoring_cost_k=k,
        project_mean_payoff=mu,
        project_volatility=sig,
        loan_correlation_rho=rho,
        required_depositor_return=req_ret,
        portfolio_volatility=port_vol,
        direct_duplicative_cost=direct_dup_cost,
        direct_unmonitored_cost=direct_unmon_cost,
        delegation_cost_d_n=d_n,
        delegated_total_cost=delegated_total,
        optimal_deposit_face_value=h_star,
        best_direct_cost=best_direct,
        net_intermediation_surplus=net_surplus,
        delegated_intermediation_viable=viable,
        diversification_efficiency_pct=div_eff,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_diamond_monitoring_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "BANK",
    num_borrowers_n: int = 50,
    loan_correlation_rho: float = 0.05,
) -> DiamondMonitoringReceipt:
    """Empirically calibrate Diamond (1984) intermediary diversification profile from prices."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        spot = float(p_arr[-1])
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        ann_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.25, 0.65))
        mu = spot * 1.25
        sig = spot * ann_vol
        req_ret = spot * 1.04
        k_cost = spot * 0.02
    else:
        mu = 125.0
        sig = 25.0
        req_ret = 104.0
        k_cost = 2.0

    return calculate_diamond_delegated_monitoring(
        num_borrowers_n=num_borrowers_n,
        depositors_per_firm_m=10,
        monitoring_cost_k=k_cost,
        project_mean_payoff=mu,
        project_volatility=sig,
        loan_correlation_rho=loan_correlation_rho,
        required_depositor_return=req_ret,
        ticker=ticker,
    )


def persist_diamond_monitoring_record(
    receipt: DiamondMonitoringReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Diamond (1984) delegated monitoring record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS diamond_monitoring_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        num_borrowers_n INTEGER NOT NULL,
        depositors_per_firm_m INTEGER NOT NULL,
        monitoring_cost_k REAL NOT NULL,
        project_mean_payoff REAL NOT NULL,
        project_volatility REAL NOT NULL,
        loan_correlation_rho REAL NOT NULL,
        required_depositor_return REAL NOT NULL,
        portfolio_volatility REAL NOT NULL,
        direct_duplicative_cost REAL NOT NULL,
        direct_unmonitored_cost REAL NOT NULL,
        delegation_cost_d_n REAL NOT NULL,
        delegated_total_cost REAL NOT NULL,
        optimal_deposit_face_value REAL NOT NULL,
        best_direct_cost REAL NOT NULL,
        net_intermediation_surplus REAL NOT NULL,
        delegated_intermediation_viable INTEGER NOT NULL,
        diversification_efficiency_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_dm_ticker_time ON diamond_monitoring_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO diamond_monitoring_audits (
        ticker, num_borrowers_n, depositors_per_firm_m, monitoring_cost_k,
        project_mean_payoff, project_volatility, loan_correlation_rho,
        required_depositor_return, portfolio_volatility, direct_duplicative_cost,
        direct_unmonitored_cost, delegation_cost_d_n, delegated_total_cost,
        optimal_deposit_face_value, best_direct_cost, net_intermediation_surplus,
        delegated_intermediation_viable, diversification_efficiency_pct, regime,
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
                    receipt.num_borrowers_n,
                    receipt.depositors_per_firm_m,
                    receipt.monitoring_cost_k,
                    receipt.project_mean_payoff,
                    receipt.project_volatility,
                    receipt.loan_correlation_rho,
                    receipt.required_depositor_return,
                    receipt.portfolio_volatility,
                    receipt.direct_duplicative_cost,
                    receipt.direct_unmonitored_cost,
                    receipt.delegation_cost_d_n,
                    receipt.delegated_total_cost,
                    receipt.optimal_deposit_face_value,
                    receipt.best_direct_cost,
                    receipt.net_intermediation_surplus,
                    1 if receipt.delegated_intermediation_viable else 0,
                    receipt.diversification_efficiency_pct,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
