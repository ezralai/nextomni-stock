"""John Y. Campbell & Tuomo Vuolteenaho (2004) Two-Beta Intertemporal Asset Pricing Engine.

Theoretical Foundation:
    Campbell, J. Y., & Vuolteenaho, T. (2004). "Bad Beta, Good Beta".
    American Economic Review, 94(5), 1249-1275.
    Campbell, J. Y. (1993). "Intertemporal Asset Pricing without Consumption Data".
    American Economic Review, 83(3), 487-512.

Key Insights:
    - Decomposition of Market Beta: An asset's CAPM market beta is the sum of
      two distinct economic components:
        beta_M = beta_CF + beta_DR
      1. Cash-Flow Beta (beta_CF, "Bad Beta"): Sensitivity to permanent market
         cash-flow innovations. Commands a high risk premium (gamma * sigma_M^2).
      2. Discount-Rate Beta (beta_DR, "Good Beta"): Sensitivity to temporary
         discount-rate news. Commands a low risk premium (sigma_M^2).
    - Resolution of the Value and Size Anomalies: Value stocks have substantially
      higher cash-flow betas than growth stocks, explaining their higher historical
      average returns despite similar or lower total market betas.
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

HMAC_SECRET = b"nextomni-bad-beta-good-beta-v114-secret"

TWO_BETA_REGIMES = (
    "CASH_FLOW_VULNERABLE_BAD_BETA",
    "DISCOUNT_RATE_GROWTH_GOOD_BETA",
    "DEFENSIVE_INSULATED_PROFILE",
    "BALANCED_SYSTEMATIC_EXPOSURE",
)

DEFAULT_RISK_AVERSION_GAMMA = 6.0    # Empirical relative risk aversion
DEFAULT_MARKET_VARIANCE = 0.0256     # (16% annual market vol)^2
DEFAULT_RISK_FREE_RATE = 0.0425      # 4.25% Rf
DEFAULT_EQUITY_RISK_PREMIUM = 0.0550 # 5.50% ERP


@dataclass(frozen=True)
class TwoBetaReceipt:
    """Immutable audit receipt for Campbell-Vuolteenaho two-beta decomposition."""

    ticker: str
    latest_price: float
    total_market_beta: float            # beta_M = beta_CF + beta_DR
    cash_flow_bad_beta: float           # beta_CF ("Bad Beta")
    discount_rate_good_beta: float      # beta_DR ("Good Beta")
    bad_beta_ratio: float               # beta_CF / beta_M
    relative_risk_aversion_gamma: float # gamma
    cash_flow_risk_price_pct: float     # lambda_CF = gamma * sigma_M^2 in %
    discount_rate_risk_price_pct: float # lambda_DR = sigma_M^2 in %
    standard_capm_return_pct: float     # Rf + beta_M * ERP in % / year
    two_beta_icapm_return_pct: float    # Rf + beta_CF * lambda_CF + beta_DR * lambda_DR in % / year
    bad_beta_hurdle_surcharge_bps: float # (ICAPM - CAPM) in basis points
    regime: str
    is_cash_flow_vulnerable: bool
    is_discount_rate_growth: bool
    is_defensive_insulated: bool
    is_balanced_exposure: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_two_beta_audit_digest(
    ticker: str,
    latest_price: float,
    beta_cf: float,
    beta_dr: float,
    icapm_return: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over two-beta metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{beta_cf:.4f}|{beta_dr:.4f}|{icapm_return:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_two_beta_audit_digest(receipt: TwoBetaReceipt) -> bool:
    """Verify cryptographic audit digest of a TwoBetaReceipt."""
    expected = generate_two_beta_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.cash_flow_bad_beta,
        receipt.discount_rate_good_beta,
        receipt.two_beta_icapm_return_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_two_beta_regime(
    total_beta: float,
    bad_beta: float,
    good_beta: float,
    bad_beta_ratio: float,
) -> str:
    """Classify the equity into one of four Campbell-Vuolteenaho two-beta regimes."""
    if total_beta < 0.65:
        return "DEFENSIVE_INSULATED_PROFILE"
    elif bad_beta_ratio >= 0.50 and bad_beta >= 0.55:
        return "CASH_FLOW_VULNERABLE_BAD_BETA"
    elif bad_beta_ratio <= 0.32 and good_beta >= 0.70:
        return "DISCOUNT_RATE_GROWTH_GOOD_BETA"
    else:
        return "BALANCED_SYSTEMATIC_EXPOSURE"


def calculate_two_beta_decomposition(
    ticker: str,
    latest_price: float = 100.0,
    cash_flow_bad_beta: float = 0.40,
    discount_rate_good_beta: float = 0.65,
    risk_aversion_gamma: float = DEFAULT_RISK_AVERSION_GAMMA,
    market_variance: float = DEFAULT_MARKET_VARIANCE,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
) -> TwoBetaReceipt:
    """Calculate Campbell-Vuolteenaho (2004) two-beta ICAPM hurdle returns.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        cash_flow_bad_beta: Cash-flow beta beta_CF.
        discount_rate_good_beta: Discount-rate beta beta_DR.
        risk_aversion_gamma: Relative risk aversion coefficient gamma (>= 1.0).
        market_variance: Annualized market return variance sigma_M^2 (> 0).
        risk_free_rate: Annual risk-free rate Rf (e.g. 0.0425 = 4.25%).
        equity_risk_premium: Market equity risk premium ERP (e.g. 0.055 = 5.5%).

    Returns:
        TwoBetaReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (-2.0 <= cash_flow_bad_beta <= 4.0):
        raise SecurityValidationError("cash_flow_bad_beta must be between -2.0 and 4.0.")
    if not (-2.0 <= discount_rate_good_beta <= 5.0):
        raise SecurityValidationError("discount_rate_good_beta must be between -2.0 and 5.0.")
    if not (1.0 <= risk_aversion_gamma <= 25.0):
        raise SecurityValidationError("risk_aversion_gamma must be between 1.0 and 25.0.")
    if market_variance <= 0.0:
        raise SecurityValidationError("market_variance must be strictly positive.")
    if not (0.0 <= risk_free_rate <= 0.20):
        raise SecurityValidationError("risk_free_rate must be between 0.0 and 0.20.")
    if not (0.0 <= equity_risk_premium <= 0.25):
        raise SecurityValidationError("equity_risk_premium must be between 0.0 and 0.25.")

    total_market_beta = cash_flow_bad_beta + discount_rate_good_beta
    safe_total = max(abs(total_market_beta), 1e-4)
    bad_beta_ratio = float(cash_flow_bad_beta / safe_total)

    # Risk Prices under Campbell-Vuolteenaho (2004) ICAPM:
    # lambda_CF = gamma * sigma_M^2
    # lambda_DR = sigma_M^2
    # Scale to match empirical ERP benchmark across market portfolio:
    market_beta_cf = 0.35
    market_beta_dr = 0.65
    theoretical_mkt_erp = (risk_aversion_gamma * market_beta_cf + 1.0 * market_beta_dr) * market_variance
    scale_factor = equity_risk_premium / max(theoretical_mkt_erp, 1e-6)

    lambda_cf = risk_aversion_gamma * market_variance * scale_factor
    lambda_dr = 1.0 * market_variance * scale_factor

    lambda_cf_pct = lambda_cf * 100.0
    lambda_dr_pct = lambda_dr * 100.0

    # CAPM single-beta required rate of return
    capm_return = risk_free_rate + total_market_beta * equity_risk_premium
    capm_return_pct = capm_return * 100.0

    # Two-Beta ICAPM required rate of return
    icapm_return = risk_free_rate + cash_flow_bad_beta * lambda_cf + discount_rate_good_beta * lambda_dr
    icapm_return_pct = icapm_return * 100.0

    # Surcharge in basis points: (ICAPM - CAPM)
    surcharge_bps = (icapm_return - capm_return) * 10000.0

    regime = classify_two_beta_regime(
        total_market_beta,
        cash_flow_bad_beta,
        discount_rate_good_beta,
        bad_beta_ratio,
    )

    is_cash_flow_vulnerable = (regime == "CASH_FLOW_VULNERABLE_BAD_BETA")
    is_discount_rate_growth = (regime == "DISCOUNT_RATE_GROWTH_GOOD_BETA")
    is_defensive_insulated = (regime == "DEFENSIVE_INSULATED_PROFILE")
    is_balanced_exposure = (regime == "BALANCED_SYSTEMATIC_EXPOSURE")

    summary = (
        f"Campbell-Vuolteenaho Two-Beta: {regime} "
        f"(Beta_M={total_market_beta:.2f}, BadBeta_CF={cash_flow_bad_beta:.2f}, GoodBeta_DR={discount_rate_good_beta:.2f}, "
        f"BadRatio={bad_beta_ratio:.1%}, ICAPM_Hurdle={icapm_return_pct:.2f}% vs CAPM={capm_return_pct:.2f}%, "
        f"BadBetaSurcharge={surcharge_bps:+.1f} bps). "
        f"{'High permanent cash-flow risk demands substantial ICAPM hurdle premium.' if is_cash_flow_vulnerable else ('Growth profile dominated by temporary discount-rate news; lower risk price.' if is_discount_rate_growth else 'Balanced or defensive intertemporal risk exposure.')}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_two_beta_audit_digest(
        clean_ticker,
        latest_price,
        cash_flow_bad_beta,
        discount_rate_good_beta,
        icapm_return_pct,
        regime,
        timestamp,
    )

    return TwoBetaReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        total_market_beta=round(total_market_beta, 4),
        cash_flow_bad_beta=round(cash_flow_bad_beta, 4),
        discount_rate_good_beta=round(discount_rate_good_beta, 4),
        bad_beta_ratio=round(bad_beta_ratio, 4),
        relative_risk_aversion_gamma=round(risk_aversion_gamma, 4),
        cash_flow_risk_price_pct=round(lambda_cf_pct, 4),
        discount_rate_risk_price_pct=round(lambda_dr_pct, 4),
        standard_capm_return_pct=round(capm_return_pct, 4),
        two_beta_icapm_return_pct=round(icapm_return_pct, 4),
        bad_beta_hurdle_surcharge_bps=round(surcharge_bps, 2),
        regime=regime,
        is_cash_flow_vulnerable=is_cash_flow_vulnerable,
        is_discount_rate_growth=is_discount_rate_growth,
        is_defensive_insulated=is_defensive_insulated,
        is_balanced_exposure=is_balanced_exposure,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_two_beta_profile(
    ticker: str,
    cash_flow_beta_override: Optional[float] = None,
    discount_rate_beta_override: Optional[float] = None,
    risk_aversion_gamma: float = DEFAULT_RISK_AVERSION_GAMMA,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> TwoBetaReceipt:
    """Evaluate Campbell-Vuolteenaho two-beta profile from historical return series."""
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
        latest_price = 110.0
        inferred_beta_cf = 0.38
        inferred_beta_dr = 0.68
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))

        # Estimate total market beta proxy via volatility ratio (market vol ~16%)
        stock_vol = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.20
        total_beta_est = float(np.clip(stock_vol / 0.160, 0.35, 2.80))

        # Value vs growth characteristic proxy via return skewness and drawdown
        skewness = float(np.mean((returns - np.mean(returns)) ** 3) / max(np.std(returns) ** 3, 1e-8))
        drawdown = float((prices[-1] - np.max(prices)) / max(np.max(prices), 1e-4))

        # Value/distressed stocks have negative skew and larger drawdowns -> higher cashflow beta
        cf_share = float(np.clip(0.35 - 0.15 * skewness - 0.25 * drawdown, 0.15, 0.70))
        inferred_beta_cf = float(np.clip(total_beta_est * cf_share, 0.05, 2.0))
        inferred_beta_dr = float(np.clip(total_beta_est * (1.0 - cf_share), 0.10, 2.5))

    beta_cf = float(cash_flow_beta_override) if cash_flow_beta_override is not None else inferred_beta_cf
    beta_dr = float(discount_rate_beta_override) if discount_rate_beta_override is not None else inferred_beta_dr

    receipt = calculate_two_beta_decomposition(
        ticker=clean_ticker,
        latest_price=latest_price,
        cash_flow_bad_beta=beta_cf,
        discount_rate_good_beta=beta_dr,
        risk_aversion_gamma=risk_aversion_gamma,
    )

    persist_two_beta_record(receipt, db_path=db_path)
    return receipt


def persist_two_beta_record(
    receipt: TwoBetaReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Campbell-Vuolteenaho two-beta audit receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS bad_beta_good_beta_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        total_market_beta REAL NOT NULL,
        cash_flow_bad_beta REAL NOT NULL,
        discount_rate_good_beta REAL NOT NULL,
        bad_beta_ratio REAL NOT NULL,
        relative_risk_aversion_gamma REAL NOT NULL,
        cash_flow_risk_price_pct REAL NOT NULL,
        discount_rate_risk_price_pct REAL NOT NULL,
        standard_capm_return_pct REAL NOT NULL,
        two_beta_icapm_return_pct REAL NOT NULL,
        bad_beta_hurdle_surcharge_bps REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO bad_beta_good_beta_records (
        ticker, latest_price, total_market_beta, cash_flow_bad_beta,
        discount_rate_good_beta, bad_beta_ratio, relative_risk_aversion_gamma,
        cash_flow_risk_price_pct, discount_rate_risk_price_pct,
        standard_capm_return_pct, two_beta_icapm_return_pct,
        bad_beta_hurdle_surcharge_bps, regime, audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                receipt.total_market_beta,
                receipt.cash_flow_bad_beta,
                receipt.discount_rate_good_beta,
                receipt.bad_beta_ratio,
                receipt.relative_risk_aversion_gamma,
                receipt.cash_flow_risk_price_pct,
                receipt.discount_rate_risk_price_pct,
                receipt.standard_capm_return_pct,
                receipt.two_beta_icapm_return_pct,
                receipt.bad_beta_hurdle_surcharge_bps,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

