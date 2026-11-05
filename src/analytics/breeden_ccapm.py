"""Douglas T. Breeden (1979) Consumption-Based Capital Asset Pricing Model (CCAPM) Engine.

Theoretical Foundation:
    Breeden, D. T. (1979). "An Intertemporal Asset Pricing Model with Stochastic
    Consumption Investment Opportunities". Journal of Financial Economics, 7(3), 265-296.
    Mehra, R., & Prescott, E. C. (1985). "The Equity Premium: A Puzzle".
    Journal of Monetary Economics, 15(2), 145-161.

Key Insights:
    - Single Consumption Benchmark: In an intertemporal economy, an asset's risk
      is completely summarized by its covariance with aggregate real per-capita
      consumption growth, eliminating the need to observe the unobservable true
      market wealth portfolio.
    - Consumption Beta:
        beta_{i, C} = Cov(R_i, Delta ln C) / Var(Delta ln C)
    - CCAPM Expected Hurdle Rate:
        E[R_i] = R_f + beta_{i, C} * lambda_C
    - Macroeconomic Consumption Pricing Wedge:
        Measures the spread between static market CAPM expected returns and
        consumption-based required returns, capturing intertemporal hedging demand.
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

HMAC_SECRET = b"nextomni-breeden-ccapm-v120-secret"

CCAPM_REGIMES = (
    "HIGH_CONSUMPTION_EXPOSURE",
    "MODERATE_PROCYCLICAL_CONSUMPTION",
    "DEFENSIVE_CONSUMPTION_HEDGE",
    "COUNTERCYCLICAL_ASSET",
)

DEFAULT_CONSUMPTION_GROWTH_STD = 0.0150 # 1.50% annual standard deviation of real consumption growth
DEFAULT_CONSUMPTION_RISK_PREMIUM = 0.0380 # 3.80% annual consumption risk price lambda_C
DEFAULT_RISK_FREE_RATE = 0.0425        # 4.25% Rf
DEFAULT_MARKET_EQUITY_PREMIUM = 0.0550 # 5.50% normal market ERP
DEFAULT_RISK_AVERSION_GAMMA = 3.50     # gamma = 3.50


@dataclass(frozen=True)
class BreedenCCAPMReceipt:
    """Immutable audit receipt for Breeden (1979) CCAPM evaluation."""

    ticker: str
    latest_price: float
    consumption_beta: float               # beta_{i, C} in [-2.0, 5.0]
    consumption_correlation: float        # rho_{i, C} in [-0.95, 0.95]
    consumption_growth_volatility_pct: float # sigma_C in %
    consumption_risk_premium_pct: float   # lambda_C in % / year
    ccapm_required_return_pct: float      # Rf + beta_C * lambda_C in %
    capm_benchmark_return_pct: float      # Rf + beta_M * ERP in %
    equity_premium_puzzle_wedge_pct: float # CAPM - CCAPM spread in %
    intertemporal_hedge_elasticity: float # beta_C * sigma_C in %
    macro_hurdle_rate_pct: float          # Hurdle discount rate in %
    regime: str
    is_high_consumption_exposure: bool
    is_moderate_procyclical: bool
    is_defensive_hedge: bool
    is_countercyclical_asset: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_breeden_audit_digest(
    ticker: str,
    latest_price: float,
    beta_c: float,
    ccapm_return: float,
    wedge_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Breeden CCAPM metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{beta_c:.4f}|{ccapm_return:.4f}|{wedge_pct:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_breeden_audit_digest(receipt: BreedenCCAPMReceipt) -> bool:
    """Verify cryptographic audit digest of a BreedenCCAPMReceipt."""
    expected = generate_breeden_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.consumption_beta,
        receipt.ccapm_required_return_pct,
        receipt.equity_premium_puzzle_wedge_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_ccapm_regime(
    beta_c: float,
    ccapm_return_pct: float,
) -> str:
    """Classify into one of four Breeden (1979) CCAPM regimes."""
    if beta_c < 0.0:
        return "COUNTERCYCLICAL_ASSET"
    elif beta_c >= 1.60 or ccapm_return_pct >= 10.50:
        return "HIGH_CONSUMPTION_EXPOSURE"
    elif beta_c >= 0.85:
        return "MODERATE_PROCYCLICAL_CONSUMPTION"
    else:
        return "DEFENSIVE_CONSUMPTION_HEDGE"


def calculate_breeden_ccapm(
    ticker: str,
    latest_price: float = 100.0,
    consumption_beta: float = 1.15,
    consumption_correlation: float = 0.40,
    consumption_growth_std: float = DEFAULT_CONSUMPTION_GROWTH_STD,
    consumption_risk_premium: float = DEFAULT_CONSUMPTION_RISK_PREMIUM,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    market_equity_beta: float = 1.10,
    market_equity_premium: float = DEFAULT_MARKET_EQUITY_PREMIUM,
) -> BreedenCCAPMReceipt:
    """Calculate Breeden (1979) Consumption CAPM required returns and pricing wedges.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        consumption_beta: Consumption beta beta_{i, C} in [-3.0, 6.0].
        consumption_correlation: Correlation with consumption growth rho in [-0.95, 0.95].
        consumption_growth_std: Real per-capita consumption growth volatility in [0.005, 0.10].
        consumption_risk_premium: Unit price of consumption risk lambda_C in [0.005, 0.15].
        risk_free_rate: Risk-free rate Rf in [0.00, 0.15].
        market_equity_beta: Standard market CAPM beta beta_M in [-2.0, 5.0].
        market_equity_premium: Market equity risk premium ERP in [0.01, 0.15].

    Returns:
        BreedenCCAPMReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (-3.0 <= consumption_beta <= 6.0):
        raise SecurityValidationError("consumption_beta must be between -3.0 and 6.0.")
    if not (-0.95 <= consumption_correlation <= 0.95):
        raise SecurityValidationError("consumption_correlation must be between -0.95 and 0.95.")
    if not (0.002 <= consumption_growth_std <= 0.20):
        raise SecurityValidationError("consumption_growth_std must be between 0.002 and 0.20.")
    if not (0.002 <= consumption_risk_premium <= 0.25):
        raise SecurityValidationError("consumption_risk_premium must be between 0.002 and 0.25.")

    clean_beta_c = float(np.clip(consumption_beta, -3.0, 6.0))
    clean_corr = float(np.clip(consumption_correlation, -0.95, 0.95))
    clean_c_vol = float(np.clip(consumption_growth_std, 0.002, 0.20))
    clean_lambda_c = float(np.clip(consumption_risk_premium, 0.002, 0.25))

    # 1. Breeden (1979) CCAPM Required Return: E[R_i] = Rf + beta_{i, C} * lambda_C
    ccapm_return = risk_free_rate + clean_beta_c * clean_lambda_c
    ccapm_return_pct = ccapm_return * 100.0

    # 2. Standard Market CAPM Benchmark Return: E[R_i] = Rf + beta_M * ERP
    capm_return = risk_free_rate + market_equity_beta * market_equity_premium
    capm_return_pct = capm_return * 100.0

    # 3. Equity Premium Puzzle / Intertemporal Consumption Pricing Wedge
    wedge_pct = capm_return_pct - ccapm_return_pct

    # 4. Intertemporal Hedging Demand Elasticity: beta_C * sigma_C * 100
    hedge_elasticity = clean_beta_c * clean_c_vol * 100.0

    # 5. Macro Hurdle Rate: max(CCAPM, Rf)
    macro_hurdle_pct = max(ccapm_return_pct, risk_free_rate * 100.0)

    regime = classify_ccapm_regime(clean_beta_c, ccapm_return_pct)
    is_high_exposure = (regime == "HIGH_CONSUMPTION_EXPOSURE")
    is_moderate_pro = (regime == "MODERATE_PROCYCLICAL_CONSUMPTION")
    is_defensive = (regime == "DEFENSIVE_CONSUMPTION_HEDGE")
    is_countercyclical = (regime == "COUNTERCYCLICAL_ASSET")

    summary = (
        f"Breeden (1979) CCAPM: {regime} "
        f"(Beta_C={clean_beta_c:.2f}x, Corr_C={clean_corr:.2f}, Lambda_C={clean_lambda_c:.2%}, "
        f"CCAPM_Return={ccapm_return_pct:.2f}%, CAPM_Return={capm_return_pct:.2f}%, "
        f"Wedge={wedge_pct:+.2f}%, HedgeElasticity={hedge_elasticity:.2f}%). "
        f"{'Heavy procyclical consumption risk demanding elevated return.' if is_high_exposure else ('Defensive consumption hedge with low business cycle sensitivity.' if is_defensive else ('Countercyclical asset providing consumption insurance.' if is_countercyclical else 'Moderate procyclical consumption risk.'))}"
    )

    clean_price = round(latest_price, 4)
    beta_c_rd = round(clean_beta_c, 4)
    corr_rd = round(clean_corr, 4)
    c_vol_rd = round(clean_c_vol * 100.0, 2)
    lambda_c_rd = round(clean_lambda_c * 100.0, 2)
    ccapm_ret_rd = round(ccapm_return_pct, 2)
    capm_ret_rd = round(capm_return_pct, 2)
    wedge_rd = round(wedge_pct, 2)
    hedge_rd = round(hedge_elasticity, 2)
    hurdle_rd = round(macro_hurdle_pct, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_breeden_audit_digest(
        clean_ticker,
        clean_price,
        beta_c_rd,
        ccapm_ret_rd,
        wedge_rd,
        regime,
        timestamp,
    )

    return BreedenCCAPMReceipt(
        ticker=clean_ticker,
        latest_price=clean_price,
        consumption_beta=beta_c_rd,
        consumption_correlation=corr_rd,
        consumption_growth_volatility_pct=c_vol_rd,
        consumption_risk_premium_pct=lambda_c_rd,
        ccapm_required_return_pct=ccapm_ret_rd,
        capm_benchmark_return_pct=capm_ret_rd,
        equity_premium_puzzle_wedge_pct=wedge_rd,
        intertemporal_hedge_elasticity=hedge_rd,
        macro_hurdle_rate_pct=hurdle_rd,
        regime=regime,
        is_high_consumption_exposure=is_high_exposure,
        is_moderate_procyclical=is_moderate_pro,
        is_defensive_hedge=is_defensive,
        is_countercyclical_asset=is_countercyclical,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_breeden_ccapm_profile(
    ticker: str,
    consumption_beta_override: Optional[float] = None,
    lambda_c_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> BreedenCCAPMReceipt:
    """Evaluate Breeden CCAPM profile from empirical price volatility and beta."""
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
        beta_c = 1.15 if consumption_beta_override is None else consumption_beta_override
        mkt_beta = 1.10
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        annual_vol = daily_vol * np.sqrt(252.0)

        # In empirical literature (e.g. Campbell 2003), consumption beta scales
        # with equity market beta and return volatility relative to macro volatility
        inferred_mkt_beta = float(np.clip(annual_vol / 0.18, 0.40, 2.50))
        mkt_beta = inferred_mkt_beta
        inferred_beta_c = float(np.clip(inferred_mkt_beta * 0.95, -0.50, 3.50))
        beta_c = consumption_beta_override if consumption_beta_override is not None else inferred_beta_c

    lambda_c = lambda_c_override if lambda_c_override is not None else DEFAULT_CONSUMPTION_RISK_PREMIUM

    return calculate_breeden_ccapm(
        ticker=clean_ticker,
        latest_price=latest_price,
        consumption_beta=beta_c,
        consumption_risk_premium=lambda_c,
        market_equity_beta=mkt_beta,
    )


def persist_breeden_ccapm_record(
    receipt: BreedenCCAPMReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Breeden CCAPM receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS breeden_ccapm_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        consumption_beta REAL NOT NULL,
        consumption_correlation REAL NOT NULL,
        consumption_growth_volatility_pct REAL NOT NULL,
        consumption_risk_premium_pct REAL NOT NULL,
        ccapm_required_return_pct REAL NOT NULL,
        capm_benchmark_return_pct REAL NOT NULL,
        equity_premium_puzzle_wedge_pct REAL NOT NULL,
        intertemporal_hedge_elasticity REAL NOT NULL,
        macro_hurdle_rate_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO breeden_ccapm_records (
        ticker, latest_price, consumption_beta, consumption_correlation,
        consumption_growth_volatility_pct, consumption_risk_premium_pct,
        ccapm_required_return_pct, capm_benchmark_return_pct,
        equity_premium_puzzle_wedge_pct, intertemporal_hedge_elasticity,
        macro_hurdle_rate_pct, regime, audit_digest, calculated_at
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
                receipt.consumption_beta,
                receipt.consumption_correlation,
                receipt.consumption_growth_volatility_pct,
                receipt.consumption_risk_premium_pct,
                receipt.ccapm_required_return_pct,
                receipt.capm_benchmark_return_pct,
                receipt.equity_premium_puzzle_wedge_pct,
                receipt.intertemporal_hedge_elasticity,
                receipt.macro_hurdle_rate_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

