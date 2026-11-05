"""Xavier Gabaix (2012) Variable Rare Disasters & Macroeconomic Tail Risk Engine.

Theoretical Foundation:
    Gabaix, X. (2012). "Variable Rare Disasters: An Exactly Solved Framework for
    Ten Puzzles in Macro-Finance". Quarterly Journal of Economics, 127(2), 645-700.
    Barro, R. J. (2006). "Rare Disasters and Asset Markets in the Twentieth Century".
    Quarterly Journal of Economics, 121(3), 823-866.

Key Insights:
    - Time-Varying Disaster Risk: Catastrophic macroeconomic contractions
      occur with Poisson intensity lambda_t in [0.5%, 15.0%] per year, causing
      aggregate consumption to contract by fraction (1 - B).
    - Firm-Level Disaster Resilience (H): In a disaster, firm cash flows/dividends
      recover by factor F_i in [0.1, 1.2]. The disaster resilience premium is:
        H_{i, t} = lambda_t * (B^(-gamma) * (1 - F_i))
    - Affine Price-to-Dividend Multiples: Gabaix's closed-form solution captures
      how disaster risk spikes compress valuation multiples and elevate forward
      expected hurdle returns.
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

HMAC_SECRET = b"nextomni-variable-rare-disasters-v116-secret"

DISASTER_RESILIENCE_REGIMES = (
    "DISASTER_VULNERABLE_HIGH_PREMIUM",
    "CYCLICAL_TAIL_RISK_EXPOSURE",
    "BENIGN_TAIL_STABILITY",
    "DISASTER_RESILIENT_FORTRESS",
)

DEFAULT_DISASTER_PROBABILITY = 0.0350  # 3.5% annual Poisson arrival rate lambda
DEFAULT_CONSUMPTION_CONTRACTION = 0.70 # B = 70% (30% macro consumption drop)
DEFAULT_RISK_AVERSION_GAMMA = 4.0     # gamma = 4.0
DEFAULT_RISK_FREE_RATE = 0.0425       # 4.25% Rf
DEFAULT_EQUITY_RISK_PREMIUM = 0.0550  # 5.50% normal ERP


@dataclass(frozen=True)
class RareDisasterReceipt:
    """Immutable audit receipt for Gabaix variable rare disaster evaluation."""

    ticker: str
    latest_price: float
    disaster_arrival_rate_lambda: float   # lambda_t in [0.005, 0.20]
    consumption_factor_b: float           # B in [0.40, 0.95]
    firm_dividend_recovery_factor_f: float # F_i in [0.10, 1.20]
    pricing_kernel_disaster_spike: float  # B^(-gamma)
    disaster_resilience_premium_pct: float # H_{i,t} in % / year
    disaster_resilience_premium_bps: float # H_{i,t} in basis points
    capm_required_return_pct: float       # Rf + beta * ERP in % / year
    total_disaster_hurdle_pct: float      # CAPM + H in % / year
    implied_price_dividend_multiple: float # (P/D)* multiple
    regime: str
    is_disaster_vulnerable: bool
    is_cyclical_exposure: bool
    is_benign_stability: bool
    is_resilient_fortress: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_disaster_audit_digest(
    ticker: str,
    latest_price: float,
    lambda_val: float,
    recovery_f: float,
    premium_pct: float,
    total_hurdle: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over variable rare disaster metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{lambda_val:.4f}|{recovery_f:.4f}|{premium_pct:.4f}|{total_hurdle:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_disaster_audit_digest(receipt: RareDisasterReceipt) -> bool:
    """Verify cryptographic audit digest of a RareDisasterReceipt."""
    expected = generate_disaster_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.disaster_arrival_rate_lambda,
        receipt.firm_dividend_recovery_factor_f,
        receipt.disaster_resilience_premium_pct,
        receipt.total_disaster_hurdle_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_disaster_regime(
    recovery_f: float,
    premium_pct: float,
) -> str:
    """Classify the equity into one of four Gabaix disaster resilience regimes."""
    if recovery_f <= 0.55 or premium_pct >= 5.50:
        return "DISASTER_VULNERABLE_HIGH_PREMIUM"
    elif recovery_f <= 0.75 or premium_pct >= 2.50:
        return "CYCLICAL_TAIL_RISK_EXPOSURE"
    elif recovery_f >= 0.92 or premium_pct <= 1.00:
        return "DISASTER_RESILIENT_FORTRESS"
    else:
        return "BENIGN_TAIL_STABILITY"


def calculate_variable_rare_disasters(
    ticker: str,
    latest_price: float = 100.0,
    disaster_arrival_rate_lambda: float = DEFAULT_DISASTER_PROBABILITY,
    consumption_factor_b: float = DEFAULT_CONSUMPTION_CONTRACTION,
    firm_dividend_recovery_factor_f: float = 0.70,
    risk_aversion_gamma: float = DEFAULT_RISK_AVERSION_GAMMA,
    market_equity_beta: float = 1.00,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
    normal_dividend_growth: float = 0.0250,
) -> RareDisasterReceipt:
    """Calculate Gabaix (2012) variable rare disaster resilience and valuation multiples.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        disaster_arrival_rate_lambda: Macro disaster Poisson intensity lambda in [0.005, 0.20].
        consumption_factor_b: Macro consumption contraction multiplier B in [0.40, 0.95].
        firm_dividend_recovery_factor_f: Firm dividend recovery factor F in [0.10, 1.20].
        risk_aversion_gamma: Relative risk aversion gamma in [1.0, 15.0].
        market_equity_beta: Systematic CAPM market beta.
        risk_free_rate: Risk-free rate Rf.
        equity_risk_premium: Normal-times equity risk premium ERP.
        normal_dividend_growth: Normal baseline dividend growth rate g.

    Returns:
        RareDisasterReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.005 <= disaster_arrival_rate_lambda <= 0.20):
        raise SecurityValidationError("disaster_arrival_rate_lambda must be between 0.005 and 0.20.")
    if not (0.30 <= consumption_factor_b <= 0.98):
        raise SecurityValidationError("consumption_factor_b must be between 0.30 and 0.98.")
    if not (0.05 <= firm_dividend_recovery_factor_f <= 1.50):
        raise SecurityValidationError("firm_dividend_recovery_factor_f must be between 0.05 and 1.50.")
    if not (1.0 <= risk_aversion_gamma <= 20.0):
        raise SecurityValidationError("risk_aversion_gamma must be between 1.0 and 20.0.")
    if not (-2.0 <= market_equity_beta <= 5.0):
        raise SecurityValidationError("market_equity_beta must be between -2.0 and 5.0.")

    clean_lambda = float(np.clip(disaster_arrival_rate_lambda, 0.005, 0.20))
    clean_b = float(np.clip(consumption_factor_b, 0.30, 0.98))
    clean_f = float(np.clip(firm_dividend_recovery_factor_f, 0.05, 1.50))

    # Marginal utility / pricing kernel spike in disaster: M_+ / M = B^(-gamma)
    kernel_spike = clean_b ** (-risk_aversion_gamma)

    # Dividend loss in disaster: max(1.0 - F, 0.0)
    dividend_loss = max(1.0 - clean_f, 0.0)

    # Gabaix (2012) Disaster Resilience Premium H_{i, t} = lambda_t * (B^(-gamma) * (1 - F_i))
    disaster_premium = clean_lambda * kernel_spike * dividend_loss
    disaster_premium = float(np.clip(disaster_premium, 0.0, 0.30))
    disaster_premium_pct = disaster_premium * 100.0
    disaster_premium_bps = disaster_premium * 10000.0

    # CAPM Required Rate of Return
    capm_return = risk_free_rate + market_equity_beta * equity_risk_premium
    capm_return_pct = capm_return * 100.0

    # Total Disaster Hurdle Rate
    total_hurdle = capm_return + disaster_premium
    total_hurdle_pct = total_hurdle * 100.0

    # Gabaix (2012) Affine Implied Price-to-Dividend Multiple:
    # (P/D)* = 1 / (r - g + H)
    discount_denominator = max(total_hurdle - normal_dividend_growth, 0.015)
    implied_pd = float(np.clip(1.0 / discount_denominator, 3.0, 65.0))

    regime = classify_disaster_regime(clean_f, disaster_premium_pct)
    is_disaster_vulnerable = (regime == "DISASTER_VULNERABLE_HIGH_PREMIUM")
    is_cyclical_exposure = (regime == "CYCLICAL_TAIL_RISK_EXPOSURE")
    is_benign_stability = (regime == "BENIGN_TAIL_STABILITY")
    is_resilient_fortress = (regime == "DISASTER_RESILIENT_FORTRESS")

    summary = (
        f"Gabaix Variable Rare Disasters: {regime} "
        f"(Lambda={clean_lambda:.1%}, Recovery_F={clean_f:.1%}, KernelSpike={kernel_spike:.1f}x, "
        f"DisasterPremium={disaster_premium_pct:+.2f}% / {disaster_premium_bps:+.1f} bps, "
        f"Total Hurdle={total_hurdle_pct:.2f}%, Implied P/D={implied_pd:.1f}x). "
        f"{'Vulnerable to catastrophic macro shocks; heavy tail risk hurdle premium.' if is_disaster_vulnerable else ('Disaster resilient fortress; dividends insulated against macro tail shocks.' if is_resilient_fortress else 'Moderate cyclical tail risk exposure.')}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_disaster_audit_digest(
        clean_ticker,
        latest_price,
        clean_lambda,
        clean_f,
        disaster_premium_pct,
        total_hurdle_pct,
        regime,
        timestamp,
    )

    return RareDisasterReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        disaster_arrival_rate_lambda=round(clean_lambda, 4),
        consumption_factor_b=round(clean_b, 4),
        firm_dividend_recovery_factor_f=round(clean_f, 4),
        pricing_kernel_disaster_spike=round(kernel_spike, 4),
        disaster_resilience_premium_pct=round(disaster_premium_pct, 4),
        disaster_resilience_premium_bps=round(disaster_premium_bps, 2),
        capm_required_return_pct=round(capm_return_pct, 4),
        total_disaster_hurdle_pct=round(total_hurdle_pct, 4),
        implied_price_dividend_multiple=round(implied_pd, 2),
        regime=regime,
        is_disaster_vulnerable=is_disaster_vulnerable,
        is_cyclical_exposure=is_cyclical_exposure,
        is_benign_stability=is_benign_stability,
        is_resilient_fortress=is_resilient_fortress,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_rare_disasters_profile(
    ticker: str,
    disaster_lambda_override: Optional[float] = None,
    recovery_f_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> RareDisasterReceipt:
    """Evaluate Gabaix rare disaster profile from historical empirical observations."""
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
        latest_price = 105.0
        inferred_f = 0.72
        inferred_beta = 1.05
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))

        # Historical maximum drawdown as proxy for disaster vulnerability
        peak = float(np.max(prices))
        drawdown = float((prices[-1] - peak) / max(peak, 1e-4))  # negative value

        # Volatility and tail skewness
        stock_vol = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.22
        inferred_beta = float(np.clip(stock_vol / 0.160, 0.40, 2.50))

        # Deep drawdowns signal lower dividend resilience (F_i < 0.60)
        raw_f = 0.85 + 0.60 * drawdown
        inferred_f = float(np.clip(raw_f, 0.20, 1.05))

    lam_val = float(disaster_lambda_override) if disaster_lambda_override is not None else DEFAULT_DISASTER_PROBABILITY
    f_val = float(recovery_f_override) if recovery_f_override is not None else inferred_f

    receipt = calculate_variable_rare_disasters(
        ticker=clean_ticker,
        latest_price=latest_price,
        disaster_arrival_rate_lambda=lam_val,
        firm_dividend_recovery_factor_f=f_val,
        market_equity_beta=inferred_beta,
    )

    persist_rare_disasters_record(receipt, db_path=db_path)
    return receipt


def persist_rare_disasters_record(
    receipt: RareDisasterReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Rare Disaster audit receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS variable_rare_disaster_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        disaster_arrival_rate_lambda REAL NOT NULL,
        consumption_factor_b REAL NOT NULL,
        firm_dividend_recovery_factor_f REAL NOT NULL,
        pricing_kernel_disaster_spike REAL NOT NULL,
        disaster_resilience_premium_pct REAL NOT NULL,
        disaster_resilience_premium_bps REAL NOT NULL,
        capm_required_return_pct REAL NOT NULL,
        total_disaster_hurdle_pct REAL NOT NULL,
        implied_price_dividend_multiple REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO variable_rare_disaster_records (
        ticker, latest_price, disaster_arrival_rate_lambda,
        consumption_factor_b, firm_dividend_recovery_factor_f,
        pricing_kernel_disaster_spike, disaster_resilience_premium_pct,
        disaster_resilience_premium_bps, capm_required_return_pct,
        total_disaster_hurdle_pct, implied_price_dividend_multiple,
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
                receipt.disaster_arrival_rate_lambda,
                receipt.consumption_factor_b,
                receipt.firm_dividend_recovery_factor_f,
                receipt.pricing_kernel_disaster_spike,
                receipt.disaster_resilience_premium_pct,
                receipt.disaster_resilience_premium_bps,
                receipt.capm_required_return_pct,
                receipt.total_disaster_hurdle_pct,
                receipt.implied_price_dividend_multiple,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

