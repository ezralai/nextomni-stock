"""Douglas W. Diamond & Philip H. Dybvig (1983) Bank Runs, Liquidity Transformation & Sequential Service Engine.

Theoretical Foundation:
    Diamond, D. W., & Dybvig, P. H. (1983). "Bank Runs, Deposit Insurance, and Liquidity".
    Journal of Political Economy, 91(3), 401-419.
    (Awarded the 2022 Nobel Memorial Prize in Economic Sciences).
    
    Diamond, D. W. (2007). "Banks and Liquidity Creation: A Simple Exposition of the Diamond-Dybvig Model".
    Federal Reserve Bank of Richmond Economic Quarterly, 93(2), 189-200.
    
    Goldstein, I., & Pauzner, A. (2005). "Demand-Deposit Contracts and the Probability of Bank Runs".
    Journal of Finance, 60(3), 1293-1327.

Key Insights:
    - Liquidity Transformation: Financial intermediaries (banks, money market funds, open-end mutual funds)
      pool liquid deposits/shares from investors with uncertain liquidity needs to invest in high-yielding
      illiquid productive assets.
    - Sequential Service Constraint: Redemptions are honored first-come, first-served until reserves and
      salvage liquidation capacity are exhausted.
    - Coordination Failure & Run Equilibrium: If depositors believe others will withdraw early, expected
      payoffs for staying to maturity fall below promised early payoffs (v_2(f_1) < r_1). This triggers a
      self-fulfilling, rational panic run where all depositors scramble to withdraw before assets are wiped out.
    - Deposit Insurance & Backstops: Credible government deposit insurance or central bank discount windows
      guarantee patience payouts, eliminating the run coordination equilibrium entirely.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker, SecurityValidationError

HMAC_SECRET = b"nextomni-diamond-dybvig-run-v124-secret"

DIAMOND_DYBVIG_REGIMES = (
    "SYSTEMIC_COORDINATION_RUN",
    "ELEVATED_RUN_HAZARD",
    "NORMAL_STABILITY",
    "INSURED_SAFETY",
)

DEFAULT_LIQUID_RESERVE = 0.15       # 15% cash/liquid reserve buffer (c)
DEFAULT_EARLY_WITHDRAWALS = 0.20    # 20% fraction of early withdrawers (f_1)
DEFAULT_R1_PAYOUT = 1.05            # 1.05 promised short-term gross payout (r_1)
DEFAULT_R2_YIELD = 1.35             # 1.35 long-term productive asset yield (R_2)
DEFAULT_SALVAGE_L = 0.65            # 0.65 premature liquidation salvage return (L < 1.0)


@dataclass(frozen=True)
class DiamondDybvigReceipt:
    """Immutable audit receipt for Diamond & Dybvig (1983) liquidity transformation and run analysis."""

    ticker: str
    latest_price: float
    liquid_reserve_ratio: float           # Cash reserve buffer c in [0.01, 0.90]
    illiquid_asset_ratio: float           # 1.0 - c in [0.10, 0.99]
    early_withdrawal_ratio: float         # Observed / stress withdrawal fraction f_1 in [0.0, 1.0]
    r1_payout: float                      # Promised gross short-term return r_1 >= 1.0
    R2_yield: float                       # Long-term productive gross return R_2 > r_1
    salvage_liquidation_L: float          # Fire-sale salvage yield L in (0.0, 1.0)
    total_liquidation_capacity: float     # Total t=1 liquidation capacity c + L * (1 - c)
    exhaustion_threshold_fbar: float      # Critical withdrawal fraction f_bar where assets are wiped out
    early_liquidation_required: float     # Illiquid assets liquidated at t=1
    remaining_illiquid_ratio: float       # Illiquid assets preserved until t=2
    patient_t2_expected_payoff: float     # Expected payoff v_2(f_1) for patient depositors at t=2
    liquidity_mismatch_ratio: float       # Ratio of t=1 demands to total liquidation capacity
    fire_sale_loss_fraction: float        # Wealth destroyed through premature salvage liquidation
    run_hazard_probability: float         # Equilibrium run coordination probability in [0.0, 1.0]
    distressed_asset_price: float         # Price markdown under run & fire-sale pressure
    fire_sale_volume_impact: float        # Distressed liquidation volume in ADV shares
    deposit_insurance_active: bool        # Whether full government backstop / insurance is active
    regime: str
    is_systemic_coordination_run: bool
    is_elevated_run_hazard: bool
    is_normal_stability: bool
    is_insured_safety: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_diamond_dybvig_audit_digest(
    ticker: str,
    latest_price: float,
    liquid_reserve_ratio: float,
    early_withdrawal_ratio: float,
    r1_payout: float,
    R2_yield: float,
    salvage_liquidation_L: float,
    total_liquidation_capacity: float,
    exhaustion_threshold_fbar: float,
    patient_t2_expected_payoff: float,
    run_hazard_probability: float,
    distressed_asset_price: float,
    regime: str,
    deposit_insurance_active: bool,
    calculated_at: str,
) -> str:
    """Generate tamper-evident HMAC-SHA256 signature for Diamond-Dybvig evaluation."""
    payload = (
        f"{ticker}|{latest_price:.4f}|{liquid_reserve_ratio:.4f}|{early_withdrawal_ratio:.4f}|"
        f"{r1_payout:.4f}|{R2_yield:.4f}|{salvage_liquidation_L:.4f}|{total_liquidation_capacity:.4f}|"
        f"{exhaustion_threshold_fbar:.4f}|{patient_t2_expected_payoff:.4f}|{run_hazard_probability:.4f}|"
        f"{distressed_asset_price:.4f}|{regime}|{int(deposit_insurance_active)}|{calculated_at}"
    )
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_diamond_dybvig_audit_digest(receipt: DiamondDybvigReceipt) -> bool:
    """Verify cryptographic authenticity and tamper-resistance of a DiamondDybvigReceipt."""
    expected = generate_diamond_dybvig_audit_digest(
        ticker=receipt.ticker,
        latest_price=receipt.latest_price,
        liquid_reserve_ratio=receipt.liquid_reserve_ratio,
        early_withdrawal_ratio=receipt.early_withdrawal_ratio,
        r1_payout=receipt.r1_payout,
        R2_yield=receipt.R2_yield,
        salvage_liquidation_L=receipt.salvage_liquidation_L,
        total_liquidation_capacity=receipt.total_liquidation_capacity,
        exhaustion_threshold_fbar=receipt.exhaustion_threshold_fbar,
        patient_t2_expected_payoff=receipt.patient_t2_expected_payoff,
        run_hazard_probability=receipt.run_hazard_probability,
        distressed_asset_price=receipt.distressed_asset_price,
        regime=receipt.regime,
        deposit_insurance_active=receipt.deposit_insurance_active,
        calculated_at=receipt.calculated_at,
    )
    return hmac.compare_digest(receipt.audit_digest, expected)


def classify_diamond_dybvig_regime(
    run_prob: float,
    early_withdrawals: float,
    fbar: float,
    liquid_reserves: float,
    r1: float,
    deposit_insurance: bool,
) -> str:
    """Classify the liquidity equilibrium into one of four Diamond-Dybvig states."""
    if deposit_insurance:
        return "INSURED_SAFETY"
    if early_withdrawals >= fbar or run_prob >= 0.75:
        return "SYSTEMIC_COORDINATION_RUN"
    if run_prob > 0.0 or (early_withdrawals * r1) > liquid_reserves:
        return "ELEVATED_RUN_HAZARD"
    return "NORMAL_STABILITY"


def calculate_diamond_dybvig_run(
    p_t: float,
    adv: float,
    liquid_reserve_ratio: float = DEFAULT_LIQUID_RESERVE,
    early_withdrawal_ratio: float = DEFAULT_EARLY_WITHDRAWALS,
    r1_payout: float = DEFAULT_R1_PAYOUT,
    R2_yield: float = DEFAULT_R2_YIELD,
    salvage_L: float = DEFAULT_SALVAGE_L,
    deposit_insurance: bool = False,
    ticker: str = "TICKER",
) -> DiamondDybvigReceipt:
    """Solve the Diamond & Dybvig (1983) liquidity transformation and run equilibrium.
    
    Args:
        p_t: Current asset price level (> 0).
        adv: Average daily volume / base liquid capacity (> 0).
        liquid_reserve_ratio: Fraction of assets held in liquid reserves c in [0.01, 0.90].
        early_withdrawal_ratio: Fraction of depositors claiming early withdrawal f_1 in [0.0, 1.0].
        r1_payout: Promised gross short-term payout r_1 >= 1.0.
        R2_yield: Long-term productive investment yield R_2 > r_1.
        salvage_L: Salvage liquidation recovery rate L in (0.0, 1.0).
        deposit_insurance: If True, government guarantees payouts, eliminating run coordination.
        ticker: Normalized asset/fund identifier.
        
    Returns:
        DiamondDybvigReceipt: Complete audited quantitative equilibrium analysis.
    """
    assert_finite_number(p_t, "p_t")
    assert_finite_number(adv, "adv")
    assert_finite_number(liquid_reserve_ratio, "liquid_reserve_ratio")
    assert_finite_number(early_withdrawal_ratio, "early_withdrawal_ratio")
    assert_finite_number(r1_payout, "r1_payout")
    assert_finite_number(R2_yield, "R2_yield")
    assert_finite_number(salvage_L, "salvage_L")

    if p_t <= 0.0:
        raise ValueError(f"Price p_t must be positive, got {p_t}")
    if adv <= 0.0:
        raise ValueError(f"ADV must be positive, got {adv}")
    
    c = float(np.clip(liquid_reserve_ratio, 0.01, 0.90))
    f1 = float(np.clip(early_withdrawal_ratio, 0.0, 1.0))
    r1 = max(1.0, float(r1_payout))
    R2 = max(r1 + 0.01, float(R2_yield))
    L = float(np.clip(salvage_L, 0.05, 0.95))
    
    illiquid_ratio = 1.0 - c
    total_cap = c + L * illiquid_ratio
    
    # Critical sequential service exhaustion threshold
    f_bar = min(1.0, total_cap / r1)
    
    # Early withdrawal cash requirement
    req_1 = f1 * r1
    
    # Illiquid assets liquidation and remaining pool
    if req_1 <= c:
        illiquid_liquidated = 0.0
        cash_rem = c - req_1
        illiquid_rem = illiquid_ratio
    else:
        cash_rem = 0.0
        deficit = req_1 - c
        illiquid_liquidated = min(illiquid_ratio, deficit / L)
        illiquid_rem = max(0.0, illiquid_ratio - illiquid_liquidated)
        
    fire_sale_loss = illiquid_liquidated * (1.0 - L)
    
    # Patient depositor payoff at t=2
    if f1 >= f_bar or f1 >= (1.0 - 1e-6):
        v2 = 0.0
    else:
        assets_t2 = (illiquid_rem * R2) + cash_rem
        v2 = max(0.0, assets_t2 / (1.0 - f1))
        
    # Liquidity Mismatch Ratio (LMR)
    lmr = req_1 / max(0.001, total_cap)
    
    # Run hazard probability
    if deposit_insurance:
        run_prob = 0.0
    elif f1 >= f_bar or v2 <= 1.0:
        run_prob = 1.0
    elif v2 < r1:
        # Expected payoff of waiting falls below promised early payout r_1
        # Coordination failure strictly increases as gap (r_1 - v_2) widens
        run_prob = min(1.0, math.pow((r1 - v2) / max(0.001, r1 - 1.0), 0.75))
    else:
        run_prob = 0.0
        
    # Asset price markdown under distress
    discount_factor = min(0.90, (fire_sale_loss * 1.5) + (run_prob * (1.0 - L) * 0.80))
    p_distressed = max(p_t * 0.10, p_t * (1.0 - discount_factor))
    
    # Fire-sale volume impact
    v_firesale = adv * min(5.0, f1 + (run_prob * (1.0 - f1) * 1.5))
    
    regime = classify_diamond_dybvig_regime(
        run_prob=run_prob,
        early_withdrawals=f1,
        fbar=f_bar,
        liquid_reserves=c,
        r1=r1,
        deposit_insurance=deposit_insurance,
    )
    
    now_iso = datetime.now(timezone.utc).isoformat()
    
    # Round metrics for audit consistency
    p_t_round = round(p_t, 4)
    c_round = round(c, 4)
    illiquid_round = round(illiquid_ratio, 4)
    f1_round = round(f1, 4)
    r1_round = round(r1, 4)
    R2_round = round(R2, 4)
    L_round = round(L, 4)
    total_cap_round = round(total_cap, 4)
    fbar_round = round(f_bar, 4)
    illiquid_liq_round = round(illiquid_liquidated, 4)
    illiquid_rem_round = round(illiquid_rem, 4)
    v2_round = round(v2, 4)
    lmr_round = round(lmr, 4)
    fire_sale_loss_round = round(fire_sale_loss, 4)
    run_prob_round = round(run_prob, 4)
    p_distressed_round = round(p_distressed, 4)
    v_firesale_round = round(v_firesale, 2)
    
    digest = generate_diamond_dybvig_audit_digest(
        ticker=ticker,
        latest_price=p_t_round,
        liquid_reserve_ratio=c_round,
        early_withdrawal_ratio=f1_round,
        r1_payout=r1_round,
        R2_yield=R2_round,
        salvage_liquidation_L=L_round,
        total_liquidation_capacity=total_cap_round,
        exhaustion_threshold_fbar=fbar_round,
        patient_t2_expected_payoff=v2_round,
        run_hazard_probability=run_prob_round,
        distressed_asset_price=p_distressed_round,
        regime=regime,
        deposit_insurance_active=deposit_insurance,
        calculated_at=now_iso,
    )
    
    summary = (
        f"Diamond-Dybvig Run Assessment: Regime={regime} | "
        f"Early Withdrawals={f1_round:.2%} (Exhaustion Threshold={fbar_round:.2%}) | "
        f"Liquid Reserves={c_round:.2%} | Expected t=2 Payout v_2={v2_round:.3f}x vs r_1={r1_round:.3f}x | "
        f"Run Hazard Probability={run_prob_round:.1%} | Distressed Price=${p_distressed_round:.2f} "
        f"(-{discount_factor:.1%}) | Insured={deposit_insurance}"
    )
    
    return DiamondDybvigReceipt(
        ticker=ticker,
        latest_price=p_t_round,
        liquid_reserve_ratio=c_round,
        illiquid_asset_ratio=illiquid_round,
        early_withdrawal_ratio=f1_round,
        r1_payout=r1_round,
        R2_yield=R2_round,
        salvage_liquidation_L=L_round,
        total_liquidation_capacity=total_cap_round,
        exhaustion_threshold_fbar=fbar_round,
        early_liquidation_required=illiquid_liq_round,
        remaining_illiquid_ratio=illiquid_rem_round,
        patient_t2_expected_payoff=v2_round,
        liquidity_mismatch_ratio=lmr_round,
        fire_sale_loss_fraction=fire_sale_loss_round,
        run_hazard_probability=run_prob_round,
        distressed_asset_price=p_distressed_round,
        fire_sale_volume_impact=v_firesale_round,
        deposit_insurance_active=deposit_insurance,
        regime=regime,
        is_systemic_coordination_run=(regime == "SYSTEMIC_COORDINATION_RUN"),
        is_elevated_run_hazard=(regime == "ELEVATED_RUN_HAZARD"),
        is_normal_stability=(regime == "NORMAL_STABILITY"),
        is_insured_safety=(regime == "INSURED_SAFETY"),
        summary=summary,
        audit_digest=digest,
        calculated_at=now_iso,
    )


def evaluate_diamond_dybvig_profile(
    ticker: str,
    liquid_reserve_ratio: Optional[float] = None,
    early_withdrawal_ratio: Optional[float] = None,
    r1_payout: Optional[float] = None,
    R2_yield: Optional[float] = None,
    salvage_L: Optional[float] = None,
    deposit_insurance: bool = False,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> DiamondDybvigReceipt:
    """Evaluate Diamond & Dybvig run equilibrium using historical prices and volume from database."""
    valid_ticker = validate_ticker(ticker)

    if prices is None or len(prices) < 5 or volumes is None or len(volumes) < 5:
        try:
            obs = db.get_observations(valid_ticker, db_path=db_path)
            if obs is not None and not obs.empty:
                if "close" in obs.columns:
                    closes = [
                        float(x)
                        for x in obs["close"].tolist()
                        if math.isfinite(float(x)) and float(x) > 0
                    ]
                    if closes:
                        prices = closes
                if "volume" in obs.columns:
                    vols = [
                        float(x)
                        for x in obs["volume"].tolist()
                        if math.isfinite(float(x)) and float(x) > 0
                    ]
                    if vols:
                        volumes = vols
        except Exception:
            pass

    latest_price = float(prices[-1]) if prices and len(prices) >= 1 else 100.0
    adv = float(np.mean(volumes[-60:])) if volumes and len(volumes) >= 5 else 1_000_000.0

    c = DEFAULT_LIQUID_RESERVE if liquid_reserve_ratio is None else liquid_reserve_ratio
    f1 = DEFAULT_EARLY_WITHDRAWALS if early_withdrawal_ratio is None else early_withdrawal_ratio
    r1 = DEFAULT_R1_PAYOUT if r1_payout is None else r1_payout
    R2 = DEFAULT_R2_YIELD if R2_yield is None else R2_yield
    L = DEFAULT_SALVAGE_L if salvage_L is None else salvage_L

    return calculate_diamond_dybvig_run(
        p_t=latest_price,
        adv=adv,
        liquid_reserve_ratio=c,
        early_withdrawal_ratio=f1,
        r1_payout=r1,
        R2_yield=R2,
        salvage_L=L,
        deposit_insurance=deposit_insurance,
        ticker=valid_ticker,
    )


def persist_diamond_dybvig_record(
    receipt: DiamondDybvigReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> bool:
    """Store Diamond & Dybvig evaluation record in SQLite database table."""
    conn = db.get_connection(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS diamond_dybvig_run_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                latest_price REAL NOT NULL,
                liquid_reserve_ratio REAL NOT NULL,
                illiquid_asset_ratio REAL NOT NULL,
                early_withdrawal_ratio REAL NOT NULL,
                r1_payout REAL NOT NULL,
                R2_yield REAL NOT NULL,
                salvage_liquidation_L REAL NOT NULL,
                total_liquidation_capacity REAL NOT NULL,
                exhaustion_threshold_fbar REAL NOT NULL,
                early_liquidation_required REAL NOT NULL,
                remaining_illiquid_ratio REAL NOT NULL,
                patient_t2_expected_payoff REAL NOT NULL,
                liquidity_mismatch_ratio REAL NOT NULL,
                fire_sale_loss_fraction REAL NOT NULL,
                run_hazard_probability REAL NOT NULL,
                distressed_asset_price REAL NOT NULL,
                fire_sale_volume_impact REAL NOT NULL,
                deposit_insurance_active INTEGER NOT NULL,
                regime TEXT NOT NULL,
                summary TEXT NOT NULL,
                audit_digest TEXT NOT NULL,
                calculated_at TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO diamond_dybvig_run_records (
                ticker, latest_price, liquid_reserve_ratio, illiquid_asset_ratio,
                early_withdrawal_ratio, r1_payout, R2_yield, salvage_liquidation_L,
                total_liquidation_capacity, exhaustion_threshold_fbar, early_liquidation_required,
                remaining_illiquid_ratio, patient_t2_expected_payoff, liquidity_mismatch_ratio,
                fire_sale_loss_fraction, run_hazard_probability, distressed_asset_price,
                fire_sale_volume_impact, deposit_insurance_active, regime, summary,
                audit_digest, calculated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                receipt.ticker,
                receipt.latest_price,
                receipt.liquid_reserve_ratio,
                receipt.illiquid_asset_ratio,
                receipt.early_withdrawal_ratio,
                receipt.r1_payout,
                receipt.R2_yield,
                receipt.salvage_liquidation_L,
                receipt.total_liquidation_capacity,
                receipt.exhaustion_threshold_fbar,
                receipt.early_liquidation_required,
                receipt.remaining_illiquid_ratio,
                receipt.patient_t2_expected_payoff,
                receipt.liquidity_mismatch_ratio,
                receipt.fire_sale_loss_fraction,
                receipt.run_hazard_probability,
                receipt.distressed_asset_price,
                receipt.fire_sale_volume_impact,
                int(receipt.deposit_insurance_active),
                receipt.regime,
                receipt.summary,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        return True
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
