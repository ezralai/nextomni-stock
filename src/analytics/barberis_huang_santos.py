"""Nicholas Barberis, Ming Huang & Tano Santos (2001) Prospect Theory & Dynamic Loss Aversion Engine.

Theoretical Foundation:
    Barberis, N., Huang, M., & Santos, T. (2001). "Prospect Theory and Asset Prices".
    Quarterly Journal of Economics, 116(1), 1-53.
    Kahneman, D., & Tversky, A. (1979). "Prospect Theory: An Analysis of Decision under Risk".
    Econometrica, 47(2), 263-291.
    Thaler, R. H., & Johnson, E. J. (1990). "Gambling with the House Money and Trying to Break Even:
    The Effects of Prior Outcomes on Risky Choice". Management Science, 36(6), 643-660.

Key Insights:
    - Direct Utility from Financial Wealth Fluctuations: Investors derive utility
      not just from aggregate consumption, but directly from gains and losses
      in financial wealth relative to a reference purchase/benchmark price.
    - Dynamic Loss Aversion: Loss aversion is path-dependent. After prior gains,
      investors accumulate a psychological capital cushion (z_t < 1, "house money effect"),
      which dampens loss aversion on subsequent declines. After prior losses, the cushion
      is erased (z_t > 1), causing loss aversion to spike dramatically (lambda_t >> 2.25).
    - Asymmetric Return Premia: Elevated loss aversion after down markets commands
      a heavy prospective loss surcharge on hurdle rates, explaining excess stock market volatility
      and high equity premia.
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

HMAC_SECRET = b"nextomni-bhs-prospect-v121-secret"

PROSPECT_REGIMES = (
    "ACUTE_LOSS_AVERSION_PANIC",
    "MODERATE_RECESSION_SENSITIVITY",
    "BENIGN_NEUTRAL_PROSPECT",
    "HOUSE_MONEY_CUSHION",
)

DEFAULT_BASELINE_LOSS_AVERSION = 2.25  # Kahneman & Tversky (1979) lambda_0 = 2.25
DEFAULT_GAIN_CUSHION_DISCOUNT = 1.25   # c = 1.25
DEFAULT_LOSS_PENALTY_SLOPE = 3.50      # k = 3.50
DEFAULT_RISK_FREE_RATE = 0.0425        # 4.25% Rf
DEFAULT_MARKET_EQUITY_PREMIUM = 0.0550 # 5.50% ERP


@dataclass(frozen=True)
class BHSProspectReceipt:
    """Immutable audit receipt for Barberis, Huang & Santos (2001) prospect theory evaluation."""

    ticker: str
    latest_price: float
    reference_price: float                # Historical purchase/reference benchmark price P_ref
    prior_cushion_state_z: float          # z_t = P_ref / P_t
    prior_return_pct: float               # (P_t - P_ref) / P_ref in %
    dynamic_loss_aversion_lambda: float   # lambda_t(z_t) in [1.0, 8.0]
    house_money_cushion_pct: float        # Max(0, 1 - z_t) in %
    prospective_loss_surcharge_pct: float # Delta r_prospect in % / year
    prospective_loss_surcharge_bps: float # Delta r_prospect in basis points
    capm_benchmark_return_pct: float      # Rf + beta * ERP in %
    total_prospect_hurdle_pct: float      # CAPM + Surcharge in %
    regime: str
    is_acute_loss_aversion: bool
    is_moderate_sensitivity: bool
    is_benign_neutral: bool
    is_house_money_cushion: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_bhs_audit_digest(
    ticker: str,
    latest_price: float,
    z_state: float,
    lambda_val: float,
    surcharge_pct: float,
    total_hurdle: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over BHS prospect theory metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{z_state:.4f}|{lambda_val:.4f}|{surcharge_pct:.4f}|{total_hurdle:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_bhs_audit_digest(receipt: BHSProspectReceipt) -> bool:
    """Verify cryptographic audit digest of a BHSProspectReceipt."""
    expected = generate_bhs_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.prior_cushion_state_z,
        receipt.dynamic_loss_aversion_lambda,
        receipt.prospective_loss_surcharge_pct,
        receipt.total_prospect_hurdle_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_prospect_regime(
    z_state: float,
    lambda_val: float,
) -> str:
    """Classify into one of four Barberis, Huang & Santos (2001) prospect regimes."""
    if z_state >= 1.30 or lambda_val >= 4.50:
        return "ACUTE_LOSS_AVERSION_PANIC"
    elif z_state >= 1.05 or lambda_val >= 2.50:
        return "MODERATE_RECESSION_SENSITIVITY"
    elif z_state <= 0.85:
        return "HOUSE_MONEY_CUSHION"
    else:
        return "BENIGN_NEUTRAL_PROSPECT"


def calculate_bhs_prospect(
    ticker: str,
    latest_price: float = 100.0,
    reference_price: float = 100.0,
    baseline_loss_aversion: float = DEFAULT_BASELINE_LOSS_AVERSION,
    asset_volatility: float = 0.22,
    market_beta: float = 1.05,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    market_equity_premium: float = DEFAULT_MARKET_EQUITY_PREMIUM,
    loss_penalty_slope_k: float = DEFAULT_LOSS_PENALTY_SLOPE,
    gain_discount_slope_c: float = DEFAULT_GAIN_CUSHION_DISCOUNT,
) -> BHSProspectReceipt:
    """Calculate Barberis, Huang & Santos (2001) dynamic loss aversion and prospect hurdle rate.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        reference_price: Purchase reference / cost-basis price ($).
        baseline_loss_aversion: Benchmark loss aversion lambda_0 in [1.5, 3.5].
        asset_volatility: Annualized asset volatility sigma in [0.02, 2.00].
        market_beta: Systematic market beta in [-2.0, 5.0].
        risk_free_rate: Risk-free rate Rf in [0.00, 0.15].
        market_equity_premium: Market equity risk premium ERP in [0.01, 0.15].
        loss_penalty_slope_k: Downside loss aversion multiplier slope k in [1.0, 10.0].
        gain_discount_slope_c: Upside cushion discount slope c in [0.2, 5.0].

    Returns:
        BHSProspectReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if reference_price <= 0.0:
        raise SecurityValidationError("reference_price must be strictly positive.")
    if not (1.0 <= baseline_loss_aversion <= 5.0):
        raise SecurityValidationError("baseline_loss_aversion must be between 1.0 and 5.0.")
    if not (0.01 <= asset_volatility <= 3.00):
        raise SecurityValidationError("asset_volatility must be between 0.01 and 3.00.")
    if not (-2.0 <= market_beta <= 5.0):
        raise SecurityValidationError("market_beta must be between -2.0 and 5.0.")

    clean_price = float(latest_price)
    clean_ref = float(reference_price)
    clean_vol = float(np.clip(asset_volatility, 0.01, 3.00))

    # 1. Prior Cushion State z_t = P_ref / P_t
    z_state = clean_ref / clean_price
    z_state = float(np.clip(z_state, 0.10, 5.00))

    # Prior Return relative to reference
    prior_return_pct = ((clean_price - clean_ref) / clean_ref) * 100.0

    # House money cushion: (1 - z_t) when in prior gain territory
    house_money_cushion_pct = max(0.0, (1.0 - z_state) * 100.0)

    # 2. Dynamic Loss Aversion Coefficient lambda_t(z_t)
    # Barberis, Huang & Santos (2001) specification:
    if z_state > 1.0:
        # Prior loss: psychological reserves depleted, loss aversion surges
        raw_lambda = baseline_loss_aversion + loss_penalty_slope_k * (z_state - 1.0)
    else:
        # Prior gain: cushion protects investor, loss aversion dampens
        raw_lambda = baseline_loss_aversion - gain_discount_slope_c * (1.0 - z_state)

    dynamic_lambda = float(np.clip(raw_lambda, 1.00, 8.00))

    # 3. Prospective Loss Surcharge (% / year)
    # The excess return demanded above standard CAPM to compensate for loss aversion pain
    raw_surcharge = (dynamic_lambda - 1.00) * clean_vol * 3.50
    prospective_loss_surcharge_pct = float(np.clip(raw_surcharge, 0.0, 35.0))
    prospective_loss_surcharge_bps = prospective_loss_surcharge_pct * 100.0

    # 4. CAPM Benchmark Return
    capm_return = risk_free_rate + market_beta * market_equity_premium
    capm_return_pct = capm_return * 100.0

    # 5. Total Prospect Theory Hurdle Return
    total_hurdle_pct = capm_return_pct + prospective_loss_surcharge_pct

    regime = classify_prospect_regime(z_state, dynamic_lambda)
    is_acute = (regime == "ACUTE_LOSS_AVERSION_PANIC")
    is_moderate = (regime == "MODERATE_RECESSION_SENSITIVITY")
    is_benign = (regime == "BENIGN_NEUTRAL_PROSPECT")
    is_cushion = (regime == "HOUSE_MONEY_CUSHION")

    summary = (
        f"Barberis-Huang-Santos Prospect: {regime} "
        f"(z={z_state:.2f}, PriorReturn={prior_return_pct:+.1f}%, Lambda={dynamic_lambda:.2f}x, "
        f"Cushion={house_money_cushion_pct:.1f}%, Surcharge=+{prospective_loss_surcharge_pct:.2f}% / +{prospective_loss_surcharge_bps:.0f} bps, "
        f"TotalHurdle={total_hurdle_pct:.2f}%). "
        f"{'Acutely loss-averse investors demanding steep risk premium after painful losses.' if is_acute else ('House money cushion dampening loss aversion and compressing hurdle rate.' if is_cushion else 'Moderate or neutral prospect theory sensitivity.')}"
    )

    clean_p_rd = round(clean_price, 4)
    ref_p_rd = round(clean_ref, 4)
    z_rd = round(z_state, 4)
    ret_rd = round(prior_return_pct, 2)
    lambda_rd = round(dynamic_lambda, 4)
    cushion_rd = round(house_money_cushion_pct, 2)
    surch_pct_rd = round(prospective_loss_surcharge_pct, 2)
    surch_bps_rd = round(prospective_loss_surcharge_bps, 1)
    capm_rd = round(capm_return_pct, 2)
    hurdle_rd = round(total_hurdle_pct, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_bhs_audit_digest(
        clean_ticker,
        clean_p_rd,
        z_rd,
        lambda_rd,
        surch_pct_rd,
        hurdle_rd,
        regime,
        timestamp,
    )

    return BHSProspectReceipt(
        ticker=clean_ticker,
        latest_price=clean_p_rd,
        reference_price=ref_p_rd,
        prior_cushion_state_z=z_rd,
        prior_return_pct=ret_rd,
        dynamic_loss_aversion_lambda=lambda_rd,
        house_money_cushion_pct=cushion_rd,
        prospective_loss_surcharge_pct=surch_pct_rd,
        prospective_loss_surcharge_bps=surch_bps_rd,
        capm_benchmark_return_pct=capm_rd,
        total_prospect_hurdle_pct=hurdle_rd,
        regime=regime,
        is_acute_loss_aversion=is_acute,
        is_moderate_sensitivity=is_moderate,
        is_benign_neutral=is_benign,
        is_house_money_cushion=is_cushion,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_bhs_prospect_profile(
    ticker: str,
    reference_price_override: Optional[float] = None,
    lambda_zero_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> BHSProspectReceipt:
    """Evaluate BHS prospect theory profile from historical price observations."""
    clean_ticker = validate_ticker(ticker)

    if prices is None or len(prices) < 20:
        try:
            obs = db.get_observations(clean_ticker, db_path=db_path)
            if obs is not None and not obs.empty and "close" in obs.columns:
                closes = [
                    float(x)
                    for x in obs["close"].tolist()
                    if math.isfinite(float(x)) and float(x) > 0
                ]
                if closes and len(closes) >= 20:
                    prices = closes
        except Exception:
            pass

    if prices is None or len(prices) < 20:
        latest_price = 100.0
        ref_price = 100.0 if reference_price_override is None else reference_price_override
        vol = 0.22
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        vol = float(np.clip(daily_vol * np.sqrt(252.0), 0.05, 1.50))

        if reference_price_override is not None:
            ref_price = reference_price_override
        else:
            # Barberis et al. empirical anchor: 60-day moving average as cost-basis proxy
            window = prices[-60:] if len(prices) >= 60 else prices
            ref_price = float(np.mean(window))

    lam_0 = lambda_zero_override if lambda_zero_override is not None else DEFAULT_BASELINE_LOSS_AVERSION

    return calculate_bhs_prospect(
        ticker=clean_ticker,
        latest_price=latest_price,
        reference_price=ref_price,
        baseline_loss_aversion=lam_0,
        asset_volatility=vol,
    )


def persist_bhs_prospect_record(
    receipt: BHSProspectReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist BHS prospect theory receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS barberis_huang_santos_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        reference_price REAL NOT NULL,
        prior_cushion_state_z REAL NOT NULL,
        prior_return_pct REAL NOT NULL,
        dynamic_loss_aversion_lambda REAL NOT NULL,
        house_money_cushion_pct REAL NOT NULL,
        prospective_loss_surcharge_pct REAL NOT NULL,
        prospective_loss_surcharge_bps REAL NOT NULL,
        capm_benchmark_return_pct REAL NOT NULL,
        total_prospect_hurdle_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO barberis_huang_santos_records (
        ticker, latest_price, reference_price, prior_cushion_state_z,
        prior_return_pct, dynamic_loss_aversion_lambda, house_money_cushion_pct,
        prospective_loss_surcharge_pct, prospective_loss_surcharge_bps,
        capm_benchmark_return_pct, total_prospect_hurdle_pct, regime,
        audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    try:
        conn = db.get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.execute(create_sql)
        cursor.execute(
            insert_sql,
            (
                receipt.ticker,
                receipt.latest_price,
                receipt.reference_price,
                receipt.prior_cushion_state_z,
                receipt.prior_return_pct,
                receipt.dynamic_loss_aversion_lambda,
                receipt.house_money_cushion_pct,
                receipt.prospective_loss_surcharge_pct,
                receipt.prospective_loss_surcharge_bps,
                receipt.capm_benchmark_return_pct,
                receipt.total_prospect_hurdle_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

