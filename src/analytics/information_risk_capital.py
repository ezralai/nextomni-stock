"""David Easley, Soeren Hvidkjaer & Maureen O'Hara (2002, 2004) Information Risk Engine.

Theoretical Foundation:
    Easley, D., Hvidkjaer, S., & O'Hara, M. (2002). "Is Information Risk a
    Determinant of Asset Returns?". The Journal of Finance, 57(5), 2185-2221.
    Easley, D., & O'Hara, M. (2004). "Information and the Cost of Capital".
    The Journal of Finance, 59(4), 1553-1583.

Key Insights:
    - Microstructure to Macro Asset Pricing: Private information risk is
      systematic and undiversifiable in rational expectations equilibrium.
      Uninformed investors cannot eliminate the adverse selection risk of holding
      high-information-asymmetry securities through portfolio diversification.
    - Cost of Capital Surcharge: Stocks with higher private information risk (PIN)
      command an equilibrium expected return premium. Easley, Hvidkjaer, and O'Hara
      demonstrated empirically that every 10 percentage point increase in PIN
      increases the required annual cost of equity capital by ~2.5% (+250 bps).
    - Multi-Factor Formulation:
        E[R_i] - R_f = beta_i^MKT * ERP + beta_i^INFO * lambda_INFO
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

HMAC_SECRET = b"nextomni-information-risk-eho-v111-secret"

INFORMATION_RISK_REGIMES = (
    "SEVERE_ADVERSE_SELECTION_SURCHARGE",
    "ELEVATED_INFORMATION_RISK",
    "BENIGN_TRANSPARENT_LIQUIDITY",
    "MINIMAL_DISCLOSURE_DISCOUNT",
)

# Benchmark constants calibrated from Easley, Hvidkjaer & O'Hara (2002, 2004)
BASELINE_MARKET_PIN = 0.1910         # ~19.1% average empirical PIN for US equities
BENCHMARK_PIN_SIGMA = 0.0820         # standard deviation of PIN in cross-section
DEFAULT_LAMBDA_INFO_PCT = 0.2500     # 25.0% return surcharge per 1.0 unit PIN difference (2.50% per 10% PIN)
DEFAULT_RISK_FREE_RATE_PCT = 0.0425  # 4.25% risk-free rate Rf
DEFAULT_EQUITY_RISK_PREMIUM = 0.0550 # 5.50% Market Equity Risk Premium


@dataclass(frozen=True)
class InformationRiskReceipt:
    """Immutable audit receipt for Easley-Hvidkjaer-O'Hara information risk evaluation."""

    ticker: str
    latest_price: float
    pin: float                          # Probability of Informed Trading in [0.02, 0.60]
    market_equity_beta: float           # Beta_i^MKT
    info_risk_beta: float               # Beta_i^INFO = (PIN - PIN_bar) / sigma_PIN
    baseline_market_pin: float          # PIN_bar (~19.1%)
    cost_of_capital_surcharge_pct: float # Delta r_info in % / year (e.g., +2.50%)
    capm_required_return_pct: float     # Rf + Beta * ERP in % / year
    total_required_hurdle_pct: float    # CAPM + Surcharge in % / year
    cost_of_capital_surcharge_bps: float # Surcharge in basis points / year
    lambda_info_factor_premium: float   # Factor risk price lambda_info
    regime: str
    is_severe_surcharge: bool
    is_elevated_risk: bool
    is_benign_transparent: bool
    is_minimal_discount: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_information_risk_audit_digest(
    ticker: str,
    latest_price: float,
    pin: float,
    surcharge_pct: float,
    hurdle_pct: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over information risk metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{pin:.4f}|{surcharge_pct:.4f}|{hurdle_pct:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_information_risk_audit_digest(receipt: InformationRiskReceipt) -> bool:
    """Verify cryptographic audit digest of an InformationRiskReceipt."""
    expected = generate_information_risk_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.pin,
        receipt.cost_of_capital_surcharge_pct,
        receipt.total_required_hurdle_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_information_risk_regime(pin: float, surcharge_pct: float) -> str:
    """Classify asset into one of four EHO (2002, 2004) information risk regimes."""
    if pin >= 0.28 or surcharge_pct >= 2.25:
        return "SEVERE_ADVERSE_SELECTION_SURCHARGE"
    elif pin >= 0.22 or surcharge_pct >= 0.70:
        return "ELEVATED_INFORMATION_RISK"
    elif pin < 0.14 or surcharge_pct <= -1.25:
        return "MINIMAL_DISCLOSURE_DISCOUNT"
    else:
        return "BENIGN_TRANSPARENT_LIQUIDITY"


def calculate_information_risk_premium(
    ticker: str,
    latest_price: float = 100.0,
    pin: float = 0.1910,
    market_equity_beta: float = 1.00,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE_PCT,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    lambda_info: float = DEFAULT_LAMBDA_INFO_PCT,
    baseline_market_pin: float = BASELINE_MARKET_PIN,
) -> InformationRiskReceipt:
    """Calculate Easley, Hvidkjaer & O'Hara (2002, 2004) information risk metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        pin: Probability of informed trading in [0.02, 0.60].
        market_equity_beta: Systematic CAPM market beta.
        risk_free_rate: Annual risk-free rate Rf (e.g. 0.0425 = 4.25%).
        equity_risk_premium: Annual market risk premium ERP (e.g. 0.0550 = 5.50%).
        lambda_info: Factor risk premium per unit PIN (default 0.250 = 25%).
        baseline_market_pin: Cross-sectional average benchmark PIN (default 0.191).

    Returns:
        InformationRiskReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.01 <= pin <= 0.85):
        raise SecurityValidationError("pin must be between 0.01 and 0.85.")
    if not (-2.0 <= market_equity_beta <= 6.0):
        raise SecurityValidationError("market_equity_beta must be between -2.0 and 6.0.")
    if not (0.0 <= risk_free_rate <= 0.20):
        raise SecurityValidationError("risk_free_rate must be between 0.0 and 0.20.")
    if not (0.0 <= equity_risk_premium <= 0.25):
        raise SecurityValidationError("equity_risk_premium must be between 0.0 and 0.25.")
    if not (0.01 <= lambda_info <= 1.00):
        raise SecurityValidationError("lambda_info must be between 0.01 and 1.00.")

    clean_pin = float(np.clip(pin, 0.01, 0.85))

    # Standardized Information Risk Beta: beta_i^INFO = (PIN - PIN_bar) / sigma_PIN
    info_risk_beta = (clean_pin - baseline_market_pin) / BENCHMARK_PIN_SIGMA

    # Cost of Capital Surcharge Delta r_info = lambda_info * (PIN - PIN_bar)
    surcharge = lambda_info * (clean_pin - baseline_market_pin)
    surcharge_pct = surcharge * 100.0
    surcharge_bps = surcharge * 10000.0

    # CAPM Required Rate of Return: r_CAPM = Rf + beta * ERP
    capm_return = risk_free_rate + market_equity_beta * equity_risk_premium
    capm_return_pct = capm_return * 100.0

    # Total Multi-Factor Hurdle Return: r_total = r_CAPM + Delta r_info
    total_hurdle = capm_return + surcharge
    total_hurdle_pct = total_hurdle * 100.0

    regime = classify_information_risk_regime(clean_pin, surcharge_pct)
    is_severe_surcharge = (regime == "SEVERE_ADVERSE_SELECTION_SURCHARGE")
    is_elevated_risk = (regime == "ELEVATED_INFORMATION_RISK")
    is_benign_transparent = (regime == "BENIGN_TRANSPARENT_LIQUIDITY")
    is_minimal_discount = (regime == "MINIMAL_DISCLOSURE_DISCOUNT")

    summary = (
        f"Easley-Hvidkjaer-O'Hara Information Risk: {regime} "
        f"(PIN={clean_pin:.1%}, InfoBeta={info_risk_beta:+.2f}, Surcharge={surcharge_pct:+.2f}% / {surcharge_bps:+.1f} bps, "
        f"Total Hurdle={total_hurdle_pct:.2f}%). "
        f"{'Severe adverse selection demands high cost-of-capital hurdle premium.' if is_severe_surcharge else 'Transparent order flow provides cost-of-capital efficiency.'}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_information_risk_audit_digest(
        clean_ticker,
        latest_price,
        clean_pin,
        surcharge_pct,
        total_hurdle_pct,
        regime,
        timestamp,
    )

    return InformationRiskReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        pin=round(clean_pin, 4),
        market_equity_beta=round(market_equity_beta, 4),
        info_risk_beta=round(info_risk_beta, 4),
        baseline_market_pin=round(baseline_market_pin, 4),
        cost_of_capital_surcharge_pct=round(surcharge_pct, 4),
        capm_required_return_pct=round(capm_return_pct, 4),
        total_required_hurdle_pct=round(total_hurdle_pct, 4),
        cost_of_capital_surcharge_bps=round(surcharge_bps, 2),
        lambda_info_factor_premium=round(lambda_info, 4),
        regime=regime,
        is_severe_surcharge=is_severe_surcharge,
        is_elevated_risk=is_elevated_risk,
        is_benign_transparent=is_benign_transparent,
        is_minimal_discount=is_minimal_discount,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_information_risk_profile(
    ticker: str,
    pin_override: Optional[float] = None,
    market_beta_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE_PCT,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    lambda_info: float = DEFAULT_LAMBDA_INFO_PCT,
    db_path: Optional[Union[str, Path]] = None,
) -> InformationRiskReceipt:
    """Evaluate Easley-Hvidkjaer-O'Hara information risk profile from historical data."""
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
                vols = [
                    float(x)
                    for x in obs["volume"].tolist()
                    if "volume" in obs.columns and math.isfinite(float(x)) and float(x) > 0
                ] if "volume" in obs.columns else []
                if closes and len(closes) >= 20:
                    prices = closes
                    volumes = vols
        except Exception:
            pass

    if prices is None or len(prices) < 20:
        latest_price = 125.0
        inferred_pin = 0.1910
        inferred_beta = 1.05
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))

        # Estimate market beta proxy (scaled by volatility relative to market vol ~16%)
        stock_vol = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.20
        inferred_beta = float(np.clip(stock_vol / 0.160, 0.40, 2.80))

        # Roll (1984) autocovariance effective spread proxy for adverse selection
        cov_lag1 = float(np.cov(returns[:-1], returns[1:])[0, 1])
        roll_spread = 2.0 * math.sqrt(max(-cov_lag1, 0.0))

        # Volume turnover proxy for uninformed liquidity arrival
        turnover_factor = 0.0
        if volumes and len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            turnover_factor = float(np.clip(math.log(max(avg_vol, 1000.0) / 1_000_000.0) * 0.015, -0.05, 0.05))

        # Empirical PIN proxy: BASELINE + spread_sensitivity - turnover_liquidity
        raw_pin = BASELINE_MARKET_PIN + 3.50 * roll_spread - turnover_factor
        inferred_pin = float(np.clip(raw_pin, 0.05, 0.55))

    pin_val = float(pin_override) if pin_override is not None else inferred_pin
    beta_val = float(market_beta_override) if market_beta_override is not None else inferred_beta

    receipt = calculate_information_risk_premium(
        ticker=clean_ticker,
        latest_price=latest_price,
        pin=pin_val,
        market_equity_beta=beta_val,
        risk_free_rate=risk_free_rate,
        equity_risk_premium=equity_risk_premium,
        lambda_info=lambda_info,
    )

    persist_information_risk_record(receipt, db_path=db_path)
    return receipt


def persist_information_risk_record(
    receipt: InformationRiskReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Information Risk audit receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS information_risk_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        pin REAL NOT NULL,
        market_equity_beta REAL NOT NULL,
        info_risk_beta REAL NOT NULL,
        baseline_market_pin REAL NOT NULL,
        cost_of_capital_surcharge_pct REAL NOT NULL,
        capm_required_return_pct REAL NOT NULL,
        total_required_hurdle_pct REAL NOT NULL,
        cost_of_capital_surcharge_bps REAL NOT NULL,
        lambda_info_factor_premium REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO information_risk_records (
        ticker, latest_price, pin, market_equity_beta, info_risk_beta,
        baseline_market_pin, cost_of_capital_surcharge_pct,
        capm_required_return_pct, total_required_hurdle_pct,
        cost_of_capital_surcharge_bps, lambda_info_factor_premium,
        regime, audit_digest, calculated_at
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
                receipt.pin,
                receipt.market_equity_beta,
                receipt.info_risk_beta,
                receipt.baseline_market_pin,
                receipt.cost_of_capital_surcharge_pct,
                receipt.capm_required_return_pct,
                receipt.total_required_hurdle_pct,
                receipt.cost_of_capital_surcharge_bps,
                receipt.lambda_info_factor_premium,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

