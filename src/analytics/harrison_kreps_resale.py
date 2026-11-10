"""J. Michael Harrison & David M. Kreps (1978, 1979) Speculative Disagreement & Resale Option Premium Engine.

Theoretical Foundation:
    Harrison, J. M., & Kreps, D. M. (1978). "Speculative Investor Behavior in a Stock Market
    with Heterogeneous Expectations". Quarterly Journal of Economics, 92(2), pp. 323-336.
    Foundational to Scheinkman & Xiong (2003) and modern speculative bubble theory.

Key Insights:
    1. Speculative Disagreement:
       Investors have heterogeneous subjective beliefs regarding state transition probabilities
       over an asset that yields dividends D(s) across discrete states s in S.

    2. Short-Sale Constraints & Resale Option:
       In the presence of short-sale constraints, an asset's equilibrium market price strictly
       exceeds the fundamental valuation of even the most optimistic investor's discounted
       buy-and-hold present value of future cash flows:
           P(s) = max_{i in {A, B}} { beta * sum_{s'} pi_i(s, s') [P(s') + D(s')] }
           P(s) = V_max(s) + Delta_resale(s)
       where Delta_resale(s) >= 0 is the pure speculative Resale Option Premium.

    3. Endogenous Trading & Speculative Bubble:
       Investors willingly pay a premium today because they acquire a valuable option to resell
       the asset in the future to another investor whose subjective belief will turn even more
       optimistic in a subsequent state.
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
HMAC_SECRET = b"harrison-kreps-resale-secret-v130"


@dataclass(frozen=True)
class HarrisonKrepsReceipt:
    """Immutable quantitative audit receipt for Harrison-Kreps resale option equilibrium."""

    ticker: str
    discount_factor: float            # beta in (0, 1)
    belief_divergence: float          # delta
    states_count: int
    dividend_states: List[float]      # D(s)
    equilibrium_prices: List[float]   # P*(s)
    fundamental_val_A: List[float]    # V_A(s)
    fundamental_val_B: List[float]    # V_B(s)
    max_fundamental_val: List[float]  # V_max(s)
    resale_option_premia: List[float] # Delta_resale(s) = P*(s) - V_max(s)
    speculative_bubble_ratios: List[float]  # Delta(s) / P*(s)
    mean_resale_premium: float
    mean_bubble_ratio: float
    active_owner_states: List[str]    # "Agent_A" or "Agent_B"
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


def calculate_harrison_kreps_equilibrium(
    dividend_states: Optional[List[float] | np.ndarray] = None,
    transition_prob_A: Optional[List[List[float]] | np.ndarray] = None,
    transition_prob_B: Optional[List[List[float]] | np.ndarray] = None,
    discount_factor: float = 0.95,
    belief_divergence: float = 0.20,
    ticker: str = "HK",
    max_iterations: int = 200,
    tolerance: float = 1e-9,
) -> HarrisonKrepsReceipt:
    """Calculate the Harrison & Kreps (1978) equilibrium prices and resale option premia.

    Args:
        dividend_states: Vector of state dividends D(s), default [1.0, 3.0] (Low vs High).
        transition_prob_A: Transition matrix under Agent A's subjective beliefs (S x S).
        transition_prob_B: Transition matrix under Agent B's subjective beliefs (S x S).
        discount_factor: Time discount factor beta in (0, 1).
        belief_divergence: Parametric divergence spread delta if matrices not provided.
        ticker: Asset symbol.
        max_iterations: Max iterations for Bellman contraction.
        tolerance: Convergence tolerance for fixed-point iteration.

    Returns:
        HarrisonKrepsReceipt containing equilibrium prices and resale premia.
    """
    cleaned_ticker = ticker.strip().upper() if ticker else "HK"
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol format: {ticker}")

    beta = float(np.clip(discount_factor, 0.01, 0.999))
    delta = float(np.clip(belief_divergence, 0.0, 0.45))

    # Default 2-state model: State 0 (Low dividend), State 1 (High dividend)
    if dividend_states is None:
        d = np.array([1.0, 3.0], dtype=float)
    else:
        d = np.maximum(0.0, np.asarray(dividend_states, dtype=float))
        if len(d) == 0:
            d = np.array([1.0, 3.0], dtype=float)

    n_states = len(d)

    # Setup transition matrices
    if transition_prob_A is not None and transition_prob_B is not None:
        pi_A = np.asarray(transition_prob_A, dtype=float)
        pi_B = np.asarray(transition_prob_B, dtype=float)
        # Ensure row-stochastic
        pi_A = pi_A / np.sum(pi_A, axis=1, keepdims=True)
        pi_B = pi_B / np.sum(pi_B, axis=1, keepdims=True)
    else:
        # Canonical Harrison-Kreps belief structure:
        # In state 0 (Low), Agent A is more optimistic about jumping to High state 1.
        # In state 1 (High), Agent B is more optimistic about persisting in High state 1.
        base_prob = 0.50
        pi_A = np.array([
            [1.0 - (base_prob + delta), base_prob + delta],
            [1.0 - (base_prob - delta * 0.5), base_prob - delta * 0.5]
        ], dtype=float)
        pi_B = np.array([
            [1.0 - (base_prob - delta), base_prob - delta],
            [1.0 - (base_prob + delta * 0.5), base_prob + delta * 0.5]
        ], dtype=float)

    # 1. Compute Fundamental Valuations (Buy-and-hold without resale option):
    # V^i = (I - beta * Pi^i)^(-1) * (beta * Pi^i * D)
    eye = np.eye(n_states)
    try:
        inv_A = np.linalg.inv(eye - beta * pi_A)
        val_A = inv_A @ (beta * pi_A @ d)
    except np.linalg.LinAlgError:
        val_A = np.zeros(n_states)

    try:
        inv_B = np.linalg.inv(eye - beta * pi_B)
        val_B = inv_B @ (beta * pi_B @ d)
    except np.linalg.LinAlgError:
        val_B = np.zeros(n_states)

    val_max = np.maximum(val_A, val_B)

    # 2. Compute Equilibrium Prices via Bellman Contraction Mapping:
    # T(P)(s) = max_{i in {A, B}} { beta * sum_{s'} pi^i(s, s') * [P(s') + D(s')] }
    p_curr = np.copy(val_max)
    active_owner = ["Agent_A"] * n_states

    for _ in range(max_iterations):
        # Payoff in next state is P(s') + D(s')
        total_payoff = p_curr + d
        val_next_A = beta * (pi_A @ total_payoff)
        val_next_B = beta * (pi_B @ total_payoff)

        p_next = np.maximum(val_next_A, val_next_B)
        diff = np.max(np.abs(p_next - p_curr))
        p_curr = p_next

        if diff < tolerance:
            break

    # Determine equilibrium owner in each state
    total_payoff = p_curr + d
    val_next_A = beta * (pi_A @ total_payoff)
    val_next_B = beta * (pi_B @ total_payoff)
    active_owner = [
        "Agent_A" if val_next_A[s] >= val_next_B[s] else "Agent_B"
        for s in range(n_states)
    ]

    # 3. Resale Option Premia & Bubble Metrics
    # In Harrison-Kreps theorem: P*(s) >= V_max(s) for all s
    resale_premia = np.maximum(0.0, p_curr - val_max)
    bubble_ratios = np.where(p_curr > 1e-6, resale_premia / p_curr, 0.0)

    mean_resale_prem = float(np.mean(resale_premia))
    mean_bubble_rat = float(np.mean(bubble_ratios))

    regime = classify_harrison_kreps_regime(mean_bubble_rat, delta)
    now_iso = datetime.now(timezone.utc).isoformat()

    signature_payload = (
        f"{cleaned_ticker}:{beta:.6f}:{delta:.6f}:{mean_resale_prem:.6f}:"
        f"{mean_bubble_rat:.6f}:{regime}:{now_iso}"
    )
    hmac_sig = generate_hmac_signature(signature_payload)

    return HarrisonKrepsReceipt(
        ticker=cleaned_ticker,
        discount_factor=beta,
        belief_divergence=delta,
        states_count=n_states,
        dividend_states=[float(x) for x in d],
        equilibrium_prices=[float(x) for x in p_curr],
        fundamental_val_A=[float(x) for x in val_A],
        fundamental_val_B=[float(x) for x in val_B],
        max_fundamental_val=[float(x) for x in val_max],
        resale_option_premia=[float(x) for x in resale_premia],
        speculative_bubble_ratios=[float(x) for x in bubble_ratios],
        mean_resale_premium=mean_resale_prem,
        mean_bubble_ratio=mean_bubble_rat,
        active_owner_states=active_owner,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def classify_harrison_kreps_regime(
    mean_bubble_ratio: float,
    belief_divergence: float,
) -> str:
    """Classify the Harrison-Kreps speculative bubble regime."""
    if mean_bubble_ratio >= 0.20:
        return "SPECULATIVE_BUBBLE_CASCADE"
    if mean_bubble_ratio >= 0.06:
        return "MODERATE_RESALE_PREMIUM"
    if belief_divergence < 0.03 or mean_bubble_ratio < 0.01:
        return "BELIEF_CONVERGENCE_ANCHORED"
    return "HETEROGENEOUS_DISAGREEMENT_STABLE"


def evaluate_harrison_kreps_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "EQUITY",
    discount_factor: float = 0.95,
    belief_spread_override: Optional[float] = None,
) -> HarrisonKrepsReceipt:
    """Empirically evaluate Harrison-Kreps speculative resale profile from price history."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) < 10:
        return calculate_harrison_kreps_equilibrium(
            discount_factor=discount_factor,
            belief_divergence=belief_spread_override or 0.15,
            ticker=ticker,
        )

    # Discretize historical prices into low and high dividend states
    p_median = float(np.median(p_arr))
    d_low = max(0.50, float(np.mean(p_arr[p_arr <= p_median])) * 0.03)
    d_high = max(d_low * 1.5, float(np.mean(p_arr[p_arr > p_median])) * 0.03)
    dividends = [d_low, d_high]

    # Estimate empirical volatility of returns to calibrate belief divergence
    returns = np.diff(p_arr) / p_arr[:-1]
    ret_vol = float(np.std(returns)) if len(returns) > 1 else 0.02
    calibrated_delta = float(np.clip(ret_vol * 4.0, 0.05, 0.35))
    if belief_spread_override is not None:
        calibrated_delta = float(np.clip(belief_spread_override, 0.0, 0.45))

    return calculate_harrison_kreps_equilibrium(
        dividend_states=dividends,
        discount_factor=discount_factor,
        belief_divergence=calibrated_delta,
        ticker=ticker,
    )


def persist_harrison_kreps_record(
    receipt: HarrisonKrepsReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Harrison-Kreps equilibrium record to SQLite using parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS harrison_kreps_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        discount_factor REAL NOT NULL,
        belief_divergence REAL NOT NULL,
        states_count INTEGER NOT NULL,
        dividend_states_json TEXT NOT NULL,
        equilibrium_prices_json TEXT NOT NULL,
        resale_option_premia_json TEXT NOT NULL,
        mean_resale_premium REAL NOT NULL,
        mean_bubble_ratio REAL NOT NULL,
        active_owners_json TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_hk_ticker_time ON harrison_kreps_audits(ticker, timestamp);
    """
    import json
    insert_sql = """
    INSERT INTO harrison_kreps_audits (
        ticker, discount_factor, belief_divergence, states_count,
        dividend_states_json, equilibrium_prices_json, resale_option_premia_json,
        mean_resale_premium, mean_bubble_ratio, active_owners_json,
        regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.executescript(create_table_sql)
            conn.execute(
                insert_sql,
                (
                    receipt.ticker,
                    receipt.discount_factor,
                    receipt.belief_divergence,
                    receipt.states_count,
                    json.dumps(receipt.dividend_states),
                    json.dumps(receipt.equilibrium_prices),
                    json.dumps(receipt.resale_option_premia),
                    receipt.mean_resale_premium,
                    receipt.mean_bubble_ratio,
                    json.dumps(receipt.active_owner_states),
                    receipt.regime,
                    receipt.hmac_signature,
                    receipt.timestamp,
                ),
            )
            conn.commit()
        return True
    except Exception:
        return False
