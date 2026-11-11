"""Yakov Amihud & Haim Mendelson (1986) Bid-Ask Spread & Clientele Asset Pricing Engine.

Theoretical Foundation:
    Amihud, Yakov, & Mendelson, Haim (1986).
    "Asset Pricing and the Bid-Ask Spread".
    Journal of Financial Economics, 17(2), pp. 223-249.

Key Concepts:
    - Piecewise-Linear Concave Spread-Return Equilibrium:
      Expected gross asset return R(S) is an increasing, piecewise-linear concave function
      of proportional bid-ask spread S:
      R(S) = min_k (alpha_k + mu_k * S),
      reflecting marginal investor turnover cohorts mu_1 > mu_2 > mu_3 > 0.
    - Investor Holding-Period Clientele Effect:
      Short-horizon traders (high mu_1) clear liquid low-spread assets (S <= S_1), while
      patient long-term investors (low mu_3) hold illiquid wide-spread assets (S > S_2) to
      amortize execution costs over multi-year horizons and earn excess liquidity rents.
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
HMAC_SECRET = b"amihud-mendelson-clientele-v169-secret"

# Canonical 3-tier investor turnover clientele cohorts (annual trades/yr & spread breakpoints)
COHORT_MU1 = 4.0     # Short-horizon trader: 0.25y holding period
COHORT_MU2 = 1.5     # Institutional active: ~0.67y holding period
COHORT_MU3 = 0.4     # Patient endowment: 2.50y holding period
BREAK_S1 = 0.005     # 50 bps proportional spread boundary
BREAK_S2 = 0.020     # 200 bps proportional spread boundary


@dataclass(frozen=True)
class AmihudMendelsonReceipt:
    """Immutable quantitative audit receipt for Amihud & Mendelson (1986) Clientele Pricing."""

    ticker: str
    proportional_spread_s: float
    spread_bps: float
    base_liquid_return_r0: float
    investor_turnover_mu: float
    investor_holding_years: float
    gross_expected_return: float
    illiquidity_premium_bps: float
    marginal_clientele_mu: float
    optimal_clientele_holding_years: float
    investor_net_return: float
    clientele_net_alpha_bps: float
    spread_compression_upside_pct: float
    equilibrium_clientele_tier: str
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


def _evaluate_concave_spread_curve(spread_s: float, base_r0: float) -> tuple[float, float, str]:
    """Evaluate continuous piecewise-linear concave gross return R(S) and marginal slope mu."""
    if spread_s <= BREAK_S1:
        gross_r = base_r0 + COHORT_MU1 * spread_s
        return gross_r, COHORT_MU1, "TIER1_SHORT_HORIZON_LIQUID_TRADER"
    prem_s1 = COHORT_MU1 * BREAK_S1
    if spread_s <= BREAK_S2:
        gross_r = base_r0 + prem_s1 + COHORT_MU2 * (spread_s - BREAK_S1)
        return gross_r, COHORT_MU2, "TIER2_INSTITUTIONAL_MEDIUM_HORIZON"
    prem_s2 = prem_s1 + COHORT_MU2 * (BREAK_S2 - BREAK_S1)
    gross_r = base_r0 + prem_s2 + COHORT_MU3 * (spread_s - BREAK_S2)
    return gross_r, COHORT_MU3, "TIER3_PATIENT_ENDOWMENT_ILLIQUID_VALUE"


def classify_amihud_mendelson_regime(
    spread_s: float,
    net_alpha_bps: float,
    inv_mu: float,
    marg_mu: float,
) -> str:
    """Classify liquidity-clientele alignment and net transaction-cost efficiency."""
    if spread_s <= 1e-6:
        return "ZERO_SPREAD_FRICTIONLESS_BENCHMARK"
    if inv_mu > marg_mu * 1.15 and net_alpha_bps < 0.0:
        return "CLIENTELE_MISMATCH_OVERTRADING_DRAG"
    if net_alpha_bps >= 50.0:
        return "PATIENT_CAPITAL_ILLIQUIDITY_RENT_CAPTURE"
    if spread_s > BREAK_S2:
        return "WIDE_SPREAD_PATIENT_ENDOWMENT_EQUILIBRIUM"
    return "MATCHED_HORIZON_CLIENTELE_EQUILIBRIUM"


def _build_hmac_payload(
    ticker: str,
    spread_s: float,
    base_r0: float,
    inv_mu: float,
    gross_r: float,
    illiq_bps: float,
    net_r: float,
    tier: str,
    regime: str,
    timestamp: str,
) -> str:
    """Build deterministic canonical string for HMAC cryptographic sealing."""
    return (
        f"{ticker}:{spread_s:.6f}:{base_r0:.6f}:{inv_mu:.4f}:{gross_r:.6f}:"
        f"{illiq_bps:.2f}:{net_r:.6f}:{tier}:{regime}:{timestamp}"
    )


def calculate_amihud_mendelson_clientele(
    proportional_spread_s: float = 0.015,
    base_liquid_return_r0: float = 0.06,
    investor_turnover_mu: float = 1.0,
    target_liquid_spread: float = 0.002,
    ticker: str = "AM86",
) -> AmihudMendelsonReceipt:
    """Evaluate Amihud & Mendelson (1986) piecewise-linear concave pricing & clientele sort."""
    cleaned_ticker = str(ticker).strip().upper()
    if not TICKER_REGEX.match(cleaned_ticker):
        raise ValueError(f"Invalid ticker symbol: {ticker!r}")

    spread_s = float(proportional_spread_s)
    r0 = float(base_liquid_return_r0)
    inv_mu = float(investor_turnover_mu)
    target_s = float(target_liquid_spread)

    if not (0.0 <= spread_s <= 0.25):
        raise ValueError("proportional_spread_s must satisfy 0.0 <= S <= 0.25.")
    if r0 <= 0.0:
        raise ValueError("base_liquid_return_r0 must be strictly positive.")
    if inv_mu <= 0.0:
        raise ValueError("investor_turnover_mu must be strictly positive.")
    if not (0.0 <= target_s <= spread_s + 1e-12):
        # Clamp target_s safely if user evaluates very tight spread below default 20 bps
        target_s = min(max(0.0, target_s), spread_s)

    gross_r, marg_mu, tier = _evaluate_concave_spread_curve(spread_s, r0)
    target_r, _, _ = _evaluate_concave_spread_curve(target_s, r0)

    illiq_prem_bps = max(0.0, (gross_r - r0) * 10000.0)
    inv_net_r = gross_r - inv_mu * spread_s
    alpha_vs_r0_bps = (inv_net_r - r0) * 10000.0
    rerating_pct = max(0.0, ((gross_r - target_r) / max(1e-6, target_r)) * 100.0)

    regime = classify_amihud_mendelson_regime(
        spread_s=spread_s,
        net_alpha_bps=alpha_vs_r0_bps,
        inv_mu=inv_mu,
        marg_mu=marg_mu,
    )
    now_iso = datetime.now(timezone.utc).isoformat()
    sig_payload = _build_hmac_payload(
        ticker=cleaned_ticker,
        spread_s=spread_s,
        base_r0=r0,
        inv_mu=inv_mu,
        gross_r=gross_r,
        illiq_bps=illiq_prem_bps,
        net_r=inv_net_r,
        tier=tier,
        regime=regime,
        timestamp=now_iso,
    )
    hmac_sig = generate_hmac_signature(sig_payload)

    return AmihudMendelsonReceipt(
        ticker=cleaned_ticker,
        proportional_spread_s=spread_s,
        spread_bps=spread_s * 10000.0,
        base_liquid_return_r0=r0,
        investor_turnover_mu=inv_mu,
        investor_holding_years=1.0 / inv_mu,
        gross_expected_return=gross_r,
        illiquidity_premium_bps=illiq_prem_bps,
        marginal_clientele_mu=marg_mu,
        optimal_clientele_holding_years=1.0 / marg_mu,
        investor_net_return=inv_net_r,
        clientele_net_alpha_bps=alpha_vs_r0_bps,
        spread_compression_upside_pct=rerating_pct,
        equilibrium_clientele_tier=tier,
        regime=regime,
        hmac_signature=hmac_sig,
        timestamp=now_iso,
    )


def evaluate_amihud_mendelson_profile(
    prices: List[float] | np.ndarray,
    ticker: str = "AM86",
    investor_turnover_mu: float = 1.0,
    base_liquid_return_r0: float = 0.06,
) -> AmihudMendelsonReceipt:
    """Calibrate effective proportional spread proxy from trailing return volatility/autocovariance."""
    p_arr = np.asarray(prices, dtype=float)
    if len(p_arr) >= 15:
        window = p_arr[-min(len(p_arr), 60):]
        rets = np.diff(window) / np.maximum(1e-6, window[:-1])
        if len(rets) >= 5:
            cov_serial = float(np.cov(rets[1:], rets[:-1])[0, 1])
            roll_spread = 2.0 * math.sqrt(max(0.0, -cov_serial))
            vol_proxy = float(np.std(rets) * 0.25)
            est_s = float(np.clip(max(roll_spread, vol_proxy), 0.001, 0.080))
        else:
            est_s = 0.015
    else:
        est_s = 0.015

    return calculate_amihud_mendelson_clientele(
        proportional_spread_s=est_s,
        base_liquid_return_r0=float(base_liquid_return_r0),
        investor_turnover_mu=float(investor_turnover_mu),
        ticker=ticker,
    )


def persist_amihud_mendelson_record(
    receipt: AmihudMendelsonReceipt,
    db_path: str = "data/nextomni_stock.db",
) -> bool:
    """Persist Amihud & Mendelson (1986) clientele record to SQLite via parameterized queries."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS amihud_mendelson_audits (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        proportional_spread_s REAL NOT NULL,
        investor_turnover_mu REAL NOT NULL,
        gross_expected_return REAL NOT NULL,
        illiquidity_premium_bps REAL NOT NULL,
        investor_net_return REAL NOT NULL,
        clientele_net_alpha_bps REAL NOT NULL,
        equilibrium_clientele_tier TEXT NOT NULL,
        regime TEXT NOT NULL,
        hmac_signature TEXT NOT NULL,
        timestamp TEXT NOT NULL
    )
    """
    insert_sql = """
    INSERT INTO amihud_mendelson_audits (
        ticker, proportional_spread_s, investor_turnover_mu, gross_expected_return,
        illiquidity_premium_bps, investor_net_return, clientele_net_alpha_bps,
        equilibrium_clientele_tier, regime, hmac_signature, timestamp
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(create_table_sql)
        conn.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.proportional_spread_s,
                receipt.investor_turnover_mu,
                receipt.gross_expected_return,
                receipt.illiquidity_premium_bps,
                receipt.investor_net_return,
                receipt.clientele_net_alpha_bps,
                receipt.equilibrium_clientele_tier,
                receipt.regime,
                receipt.hmac_signature,
                receipt.timestamp,
            ),
        )
        conn.commit()
    return True

