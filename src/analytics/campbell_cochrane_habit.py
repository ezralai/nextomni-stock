"""John Y. Campbell & John H. Cochrane (1999) External Habit Formation Asset Pricing Engine.

Theoretical Foundation:
    Campbell, John Y., & Cochrane, John H. (1999). "By Force of Habit: A Consumption-Based
    Explanation of Aggregate Stock Market Behavior". Journal of Political Economy, 107(2),
    pp. 205-251.
    Landmark equilibrium consumption-based asset pricing model resolving the Equity Premium
    Puzzle, Risk-Free Rate Puzzle, counter-cyclical stock volatility, and return predictability
    through slow-moving external habit persistence.

Key Concepts:
    1. External Habit & Surplus Consumption Ratio (S_t):
       Agents derive power utility over excess consumption above a slow-moving external
       social habit X_t, summarized by surplus consumption ratio S_t = (C_t - X_t) / C_t.
       Steady-state surplus ratio balances precautionary savings and intertemporal substitution:
           S_bar = sigma_c * sqrt(gamma / (1 - phi))

    2. Nonlinear Sensitivity Function lambda(s_t) & Constant Risk-Free Rate:
       Log surplus s_t = ln(S_t) evolves with sensitivity:
           lambda(s_t) = (1 / S_bar) * sqrt(1 - 2 * (s_t - s_bar)) - 1   for s_t <= s_max
       where s_max = s_bar + 0.5 * (1 - S_bar^2), and lambda(s_t) = 0 for s_t > s_max.
       Yields a constant real risk-free rate r_f = -ln(beta) + gamma * g - 0.5 * gamma * (1 - phi).

    3. Time-Varying Risk Aversion & Counter-Cyclical Equity Premia:
       Local Relative Risk Aversion equals RRA_t = gamma / S_t, and Maximum Conditional
       Sharpe Ratio equals SR_t = gamma * sigma_c * (1 + lambda(s_t)), surging sharply in
       recessions (when S_t drops toward zero) and falling in boom expansions.
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
HMAC_SECRET = b"campbell-cochrane-habit-v148-secret"


@dataclass(frozen=True)
class CampbellCochraneReceipt:
    """Immutable quantitative audit receipt for Campbell & Cochrane (1999) habit equilibrium."""

    ticker: str
    utility_curvature_gamma: float         # gamma
    habit_persistence_phi: float           # phi
    consumption_growth_mean: float         # g
    consumption_growth_vol: float          # sigma_c
    subjective_beta: float                 # beta
    steady_state_surplus_s_bar: float      # S_bar
    max_surplus_ratio_s_max: float         # S_max = exp(s_max)
    current_surplus_ratio: float           # S_t
    sensitivity_lambda: float              # lambda(s_t)
    local_risk_aversion_rra: float         # gamma / S_t
    conditional_max_sharpe: float          # SR_t = gamma * sigma_c * (1 + lambda)
    implied_risk_free_rate: float          # r_f
    expected_equity_premium: float         # E[R_e - R_f]
    habit_half_life_years: float           # ln(2) / (-ln(phi))
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


def classify_habit_regime(
    current_s: float,
    steady_s: float,
    max_s: float,
    conditional_sharpe: float,
) -> str:
    """Classify the Campbell-Cochrane cyclical habit & surplus consumption regime."""
    if current_s <= 0.60 * steady_s or conditional_sharpe >= 0.70:
        return "SURPLUS_CRISIS_SPIKE_HIGH_RISK_AVERSION"
    if current_s < 0.92 * steady_s:
        return "RECESSION_ELEVATED_EQUITY_PREMIUM"
    if current_s >= max_s * 0.99:
        return "HABIT_COMPLACENCY_LOW_VOLATILITY_BOOM"
    if current_s > 1.08 * steady_s:
        return "EXPANSION_COMPRESSED_RISK_PREMIUM"
    return "STEADY_STATE_HABIT_EQUILIBRIUM"


def calculate_campbell_cochrane_habit(
    utility_curvature_gamma: float = 2.0,
    habit_persistence_phi: float = 0.87,
    consumption_growth_mean: float = 0.0189,
    consumption_growth_vol: float = 0.0150,
    subjective_beta: float = 0.89,
    current_surplus_ratio: Optional[float] = None,
    equity_volatility: float = 0.16,
    equity_consumption_corr: float = 0.75,
    ticker: str = "HABIT",
) -> CampbellCochraneReceipt:
    """Solve Campbell & Cochrane (1999) external habit equilibrium metrics.

    Args:
        utility_curvature_gamma: Utility curvature parameter gamma (> 0).
        habit_persistence_phi: Annual persistence phi of log surplus consumption in (0, 0.999).
        consumption_growth_mean: Expected annual real consumption growth g.
        consumption_growth_vol: Annual real consumption growth std dev sigma_c (> 0).
        subjective_beta: Annual time preference discount factor beta in (0, 1).
        current_surplus_ratio: Current surplus consumption ratio S_t = (C_t - X_t)/C_t.
                               Defaults to steady state S_bar if omitted.
        equity_volatility: Aggregate equity return volatility sigma_eq for premium calibration.
        equity_consumption_corr: Correlation between equity returns and consumption shocks.
        ticker: Security ticker identifier.

    Returns:
        CampbellCochraneReceipt with surplus bounds, sensitivity lambda, RRA, Sharpe ratio, and HMAC.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "HABIT"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    gamma = float(utility_curvature_gamma)
    sigma_c = float(consumption_growth_vol)
    phi = float(np.clip(habit_persistence_phi, 0.01, 0.999))
    g = float(consumption_growth_mean)
    beta = float(np.clip(subjective_beta, 0.50, 0.9999))

    if not math.isfinite(gamma) or gamma <= 0.0:
        raise ValueError(f"utility_curvature_gamma must be finite and positive, got: {utility_curvature_gamma}")
    if not math.isfinite(sigma_c) or sigma_c <= 0.0:
        raise ValueError(f"consumption_growth_vol must be finite and positive, got: {consumption_growth_vol}")

    # Steady-state surplus consumption ratio S_bar = sigma_c * sqrt(gamma / (1 - phi))
    s_bar = min(0.95, sigma_c * math.sqrt(gamma / (1.0 - phi)))
    s_bar_log = math.log(s_bar)
    s_max_log = s_bar_log + 0.5 * (1.0 - s_bar ** 2)
    s_max = min(0.99, math.exp(s_max_log))

    s_t = s_bar if current_surplus_ratio is None else float(np.clip(current_surplus_ratio, 0.002, 0.99))
    s_t_log = math.log(s_t)

    if s_t_log <= s_max_log:
        inner = max(0.0, 1.0 - 2.0 * (s_t_log - s_bar_log))
        lam = max(0.0, (1.0 / s_bar) * math.sqrt(inner) - 1.0)
    else:
        lam = 0.0

    local_rra = gamma / s_t
    cond_sharpe = gamma * sigma_c * (1.0 + lam)
    rf = -math.log(beta) + gamma * g - 0.5 * gamma * (1.0 - phi)

    eq_vol = max(0.01, float(equity_volatility))
    eq_corr = float(np.clip(equity_consumption_corr, 0.0, 1.0))
    exp_eq_prem = eq_corr * eq_vol * cond_sharpe

    half_life = math.log(2.0) / max(1e-6, -math.log(phi))

    regime = classify_habit_regime(
        current_s=s_t,
        steady_s=s_bar,
        max_s=s_max,
        conditional_sharpe=cond_sharpe,
    )
    now_iso = datetime.now(timezone.utc).isoformat()

    sig_payload = (
        f"{cleaned_ticker}:{gamma:.4f}:{phi:.4f}:{sigma_c:.4f}:{s_bar:.6f}:"
        f"{s_t:.6f}:{lam:.6f}:{local_rra:.4f}:{cond_sharpe:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return CampbellCochraneReceipt(
        ticker=cleaned_ticker,
        utility_curvature_gamma=gamma,
        habit_persistence_phi=phi,
        consumption_growth_mean=g,
        consumption_growth_vol=sigma_c,
        subjective_beta=beta,
        steady_state_surplus_s_bar=s_bar,
        max_surplus_ratio_s_max=s_max,
        current_surplus_ratio=s_t,
        sensitivity_lambda=lam,
        local_risk_aversion_rra=local_rra,
        conditional_max_sharpe=cond_sharpe,
        implied_risk_free_rate=rf,
        expected_equity_premium=exp_eq_prem,
        habit_half_life_years=half_life,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_campbell_cochrane_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "HABIT",
    surplus_multiplier: float = 1.0,
    utility_curvature_gamma: float = 2.0,
) -> CampbellCochraneReceipt:
    """Calibrate surplus consumption state from historical drawdown/volatility and run habit engine."""
    p_arr = np.asarray(prices, dtype=float)
    sigma_c = 0.0150
    phi = 0.87
    s_bar = min(0.95, sigma_c * math.sqrt(max(0.1, utility_curvature_gamma) / (1.0 - phi)))

    if len(p_arr) >= 15:
        log_rets = np.diff(np.log(np.maximum(1e-6, p_arr[-min(len(p_arr), 60):])))
        eq_vol = float(np.clip(np.std(log_rets) * math.sqrt(252.0), 0.12, 0.60))
        peak = float(np.max(p_arr))
        drawdown = max(0.0, (peak - float(p_arr[-1])) / max(1e-6, peak))
        # Recessions/drawdowns compress surplus consumption below steady state
        cyclical_factor = max(0.35, 1.0 - 1.5 * drawdown)
        s_t = s_bar * float(surplus_multiplier) * cyclical_factor
    else:
        eq_vol = 0.16
        s_t = s_bar * float(surplus_multiplier)

    return calculate_campbell_cochrane_habit(
        utility_curvature_gamma=utility_curvature_gamma,
        habit_persistence_phi=phi,
        consumption_growth_mean=0.0189,
        consumption_growth_vol=sigma_c,
        subjective_beta=0.89,
        current_surplus_ratio=s_t,
        equity_volatility=eq_vol,
        ticker=ticker,
    )


def persist_campbell_cochrane_record(
    receipt: CampbellCochraneReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Campbell-Cochrane habit equilibrium record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS campbell_cochrane_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        utility_curvature_gamma REAL NOT NULL,
        habit_persistence_phi REAL NOT NULL,
        consumption_growth_mean REAL NOT NULL,
        consumption_growth_vol REAL NOT NULL,
        subjective_beta REAL NOT NULL,
        steady_state_surplus_s_bar REAL NOT NULL,
        max_surplus_ratio_s_max REAL NOT NULL,
        current_surplus_ratio REAL NOT NULL,
        sensitivity_lambda REAL NOT NULL,
        local_risk_aversion_rra REAL NOT NULL,
        conditional_max_sharpe REAL NOT NULL,
        implied_risk_free_rate REAL NOT NULL,
        expected_equity_premium REAL NOT NULL,
        habit_half_life_years REAL NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_cc_habit_ticker_time ON campbell_cochrane_audits(ticker, timestamp);
    """
    insert_sql = """
    INSERT INTO campbell_cochrane_audits (
        ticker, utility_curvature_gamma, habit_persistence_phi, consumption_growth_mean,
        consumption_growth_vol, subjective_beta, steady_state_surplus_s_bar,
        max_surplus_ratio_s_max, current_surplus_ratio, sensitivity_lambda,
        local_risk_aversion_rra, conditional_max_sharpe, implied_risk_free_rate,
        expected_equity_premium, habit_half_life_years, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.utility_curvature_gamma,
                    receipt.habit_persistence_phi,
                    receipt.consumption_growth_mean,
                    receipt.consumption_growth_vol,
                    receipt.subjective_beta,
                    receipt.steady_state_surplus_s_bar,
                    receipt.max_surplus_ratio_s_max,
                    receipt.current_surplus_ratio,
                    receipt.sensitivity_lambda,
                    receipt.local_risk_aversion_rra,
                    receipt.conditional_max_sharpe,
                    receipt.implied_risk_free_rate,
                    receipt.expected_equity_premium,
                    receipt.habit_half_life_years,
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False

