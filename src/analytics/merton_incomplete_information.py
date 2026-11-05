"""Robert C. Merton (1987) Incomplete Information & Investor Recognition Engine.

Theoretical Foundation:
    Merton, R. C. (1987). "A Simple Model of Capital Market Equilibrium with
    Incomplete Information". The Journal of Finance, 42(3), 483-510.

Key Insights:
    - Limited Investor Attention: In real financial markets, investors do not have
      universal awareness of every security. Each investor knows only a subset
      of available assets.
    - Investor Recognition Ratio (q): For a firm known by only a fraction q in (0, 1]
      of investors, the cognizant investors must hold an undiversified, concentrated
      fraction of their portfolio in that firm.
    - Shadow Cost of Incomplete Information: To compensate cognizant investors for
      bearing undiversified idiosyncratic risk (sigma_epsilon^2), the firm's cost of
      capital commands a Merton shadow cost surcharge:
        Delta r_Merton = lambda_shadow * (sigma_epsilon^2 / q)
    - Neglected Firm Effect: Smaller, less-covered firms command higher expected
      returns due to limited recognition, resolving cross-sectional size and
      coverage anomalies.
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

HMAC_SECRET = b"nextomni-merton-incomplete-info-v115-secret"

MERTON_RECOGNITION_REGIMES = (
    "NEGLECTED_FIRM_HIGH_SURCHARGE",
    "ELEVATED_INCOMPLETE_INFORMATION",
    "BROAD_RECOGNITION_EQUILIBRIUM",
    "UNIVERSAL_COVERAGE_SATURATED",
)

DEFAULT_SHADOW_LAMBDA = 0.25         # Shadow cost multiplier lambda_shadow
DEFAULT_RISK_FREE_RATE = 0.0425      # 4.25% Rf
DEFAULT_EQUITY_RISK_PREMIUM = 0.0550 # 5.50% ERP


@dataclass(frozen=True)
class MertonIncompleteInfoReceipt:
    """Immutable audit receipt for Merton incomplete information evaluation."""

    ticker: str
    latest_price: float
    investor_recognition_ratio_q: float   # q in [0.05, 1.00]
    idiosyncratic_variance: float         # sigma_epsilon^2
    market_equity_beta: float             # beta_M
    shadow_cost_lambda: float             # lambda_shadow
    capm_required_return_pct: float       # Rf + beta * ERP in % / year
    merton_shadow_surcharge_pct: float    # Delta r_Merton in % / year
    merton_shadow_surcharge_bps: float    # Delta r_Merton in basis points
    total_required_hurdle_pct: float      # CAPM + Surcharge in % / year
    regime: str
    is_neglected_firm: bool
    is_elevated_incomplete: bool
    is_broad_recognition: bool
    is_universal_saturated: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_merton_audit_digest(
    ticker: str,
    latest_price: float,
    recognition_q: float,
    surcharge_pct: float,
    total_hurdle: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Merton incomplete information metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{recognition_q:.4f}|{surcharge_pct:.4f}|{total_hurdle:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_merton_audit_digest(receipt: MertonIncompleteInfoReceipt) -> bool:
    """Verify cryptographic audit digest of a MertonIncompleteInfoReceipt."""
    expected = generate_merton_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.investor_recognition_ratio_q,
        receipt.merton_shadow_surcharge_pct,
        receipt.total_required_hurdle_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_merton_regime(
    recognition_q: float,
    surcharge_pct: float,
) -> str:
    """Classify the equity into one of four Merton incomplete information regimes."""
    if recognition_q <= 0.25 and surcharge_pct >= 2.0:
        return "NEGLECTED_FIRM_HIGH_SURCHARGE"
    elif recognition_q <= 0.20 or surcharge_pct >= 3.0:
        return "NEGLECTED_FIRM_HIGH_SURCHARGE"
    elif recognition_q <= 0.50 or surcharge_pct >= 1.5:
        return "ELEVATED_INCOMPLETE_INFORMATION"
    elif recognition_q >= 0.85:
        return "UNIVERSAL_COVERAGE_SATURATED"
    else:
        return "BROAD_RECOGNITION_EQUILIBRIUM"


def calculate_merton_incomplete_information(
    ticker: str,
    latest_price: float = 100.0,
    investor_recognition_ratio_q: float = 0.50,
    idiosyncratic_variance: float = 0.0400,     # e.g., 20% idiosyncratic vol -> 0.0400
    market_equity_beta: float = 1.00,
    shadow_cost_lambda: float = DEFAULT_SHADOW_LAMBDA,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    equity_risk_premium: float = DEFAULT_EQUITY_RISK_PREMIUM,
) -> MertonIncompleteInfoReceipt:
    """Calculate Merton (1987) incomplete information shadow cost surcharge.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        investor_recognition_ratio_q: Ratio of cognizant investors in [0.05, 1.00].
        idiosyncratic_variance: Annualized idiosyncratic return variance (> 0).
        market_equity_beta: Systematic CAPM market beta.
        shadow_cost_lambda: Shadow cost multiplier lambda_shadow in [0.05, 2.0].
        risk_free_rate: Annual risk-free rate Rf (e.g. 0.0425 = 4.25%).
        equity_risk_premium: Market equity risk premium ERP (e.g. 0.055 = 5.5%).

    Returns:
        MertonIncompleteInfoReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.02 <= investor_recognition_ratio_q <= 1.00):
        raise SecurityValidationError("investor_recognition_ratio_q must be between 0.02 and 1.00.")
    if idiosyncratic_variance <= 0.0:
        raise SecurityValidationError("idiosyncratic_variance must be strictly positive.")
    if not (-2.0 <= market_equity_beta <= 5.0):
        raise SecurityValidationError("market_equity_beta must be between -2.0 and 5.0.")
    if not (0.01 <= shadow_cost_lambda <= 5.0):
        raise SecurityValidationError("shadow_cost_lambda must be between 0.01 and 5.0.")
    if not (0.0 <= risk_free_rate <= 0.20):
        raise SecurityValidationError("risk_free_rate must be between 0.0 and 0.20.")
    if not (0.0 <= equity_risk_premium <= 0.25):
        raise SecurityValidationError("equity_risk_premium must be between 0.0 and 0.25.")

    clean_q = float(np.clip(investor_recognition_ratio_q, 0.02, 1.00))

    # Standard CAPM Required Return
    capm_return = risk_free_rate + market_equity_beta * equity_risk_premium
    capm_return_pct = capm_return * 100.0

    # Merton (1987) Incomplete Information Shadow Cost Surcharge:
    # Delta r_Merton = lambda_shadow * (sigma_epsilon^2 / q)
    surcharge = shadow_cost_lambda * (idiosyncratic_variance / clean_q)
    surcharge = float(np.clip(surcharge, 0.0, 0.35))  # max 35% surcharge
    surcharge_pct = surcharge * 100.0
    surcharge_bps = surcharge * 10000.0

    # Total Multi-Factor Hurdle Return
    total_hurdle = capm_return + surcharge
    total_hurdle_pct = total_hurdle * 100.0

    regime = classify_merton_regime(clean_q, surcharge_pct)
    is_neglected_firm = (regime == "NEGLECTED_FIRM_HIGH_SURCHARGE")
    is_elevated_incomplete = (regime == "ELEVATED_INCOMPLETE_INFORMATION")
    is_broad_recognition = (regime == "BROAD_RECOGNITION_EQUILIBRIUM")
    is_universal_saturated = (regime == "UNIVERSAL_COVERAGE_SATURATED")

    summary = (
        f"Merton Incomplete Information: {regime} "
        f"(Recognition_q={clean_q:.1%}, IdioVar={idiosyncratic_variance:.4f}, Beta_M={market_equity_beta:.2f}, "
        f"ShadowSurcharge={surcharge_pct:+.2f}% / {surcharge_bps:+.1f} bps, Total Hurdle={total_hurdle_pct:.2f}%). "
        f"{'Severely neglected firm; small investor base demands high shadow cost hurdle premium.' if is_neglected_firm else ('Universal consensus coverage; minimal incomplete information penalty.' if is_universal_saturated else 'Standard incomplete information equilibrium with moderate coverage.')}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_merton_audit_digest(
        clean_ticker,
        latest_price,
        clean_q,
        surcharge_pct,
        total_hurdle_pct,
        regime,
        timestamp,
    )

    return MertonIncompleteInfoReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        investor_recognition_ratio_q=round(clean_q, 4),
        idiosyncratic_variance=round(idiosyncratic_variance, 6),
        market_equity_beta=round(market_equity_beta, 4),
        shadow_cost_lambda=round(shadow_cost_lambda, 4),
        capm_required_return_pct=round(capm_return_pct, 4),
        merton_shadow_surcharge_pct=round(surcharge_pct, 4),
        merton_shadow_surcharge_bps=round(surcharge_bps, 2),
        total_required_hurdle_pct=round(total_hurdle_pct, 4),
        regime=regime,
        is_neglected_firm=is_neglected_firm,
        is_elevated_incomplete=is_elevated_incomplete,
        is_broad_recognition=is_broad_recognition,
        is_universal_saturated=is_universal_saturated,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_merton_incomplete_info_profile(
    ticker: str,
    recognition_q_override: Optional[float] = None,
    shadow_lambda_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> MertonIncompleteInfoReceipt:
    """Evaluate Merton incomplete information profile from historical observations."""
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
        latest_price = 120.0
        inferred_q = 0.50
        inferred_idio_var = 0.0450
        inferred_beta = 1.05
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))

        stock_vol = float(np.std(returns, ddof=1) * math.sqrt(252.0)) if len(returns) > 1 else 0.22
        total_var = stock_vol ** 2
        inferred_beta = float(np.clip(stock_vol / 0.160, 0.40, 2.50))

        # Systematic variance = beta^2 * sigma_M^2
        mkt_var = 0.0256
        sys_var = (inferred_beta ** 2) * mkt_var
        inferred_idio_var = float(np.clip(total_var - sys_var, 0.20 * total_var, 0.90 * total_var))

        # Volume turnover proxy for investor recognition ratio q
        if volumes and len(volumes) >= 20:
            avg_vol = float(np.mean(volumes[-20:]))
            log_vol = math.log10(max(avg_vol, 1000.0))
            # 1k vol -> q ~ 0.15, 100k -> q ~ 0.50, 10M -> q ~ 0.90
            turnover_factor = float(np.clip((log_vol - 3.0) / 4.0, 0.0, 1.0))
            inferred_q = float(np.clip(0.15 + 0.75 * turnover_factor, 0.08, 0.95))
        else:
            inferred_q = 0.50

    q_val = float(recognition_q_override) if recognition_q_override is not None else inferred_q
    lam_val = float(shadow_lambda_override) if shadow_lambda_override is not None else DEFAULT_SHADOW_LAMBDA

    receipt = calculate_merton_incomplete_information(
        ticker=clean_ticker,
        latest_price=latest_price,
        investor_recognition_ratio_q=q_val,
        idiosyncratic_variance=inferred_idio_var,
        market_equity_beta=inferred_beta,
        shadow_cost_lambda=lam_val,
    )

    persist_merton_incomplete_info_record(receipt, db_path=db_path)
    return receipt


def persist_merton_incomplete_info_record(
    receipt: MertonIncompleteInfoReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Merton incomplete information audit receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS merton_incomplete_info_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        investor_recognition_ratio_q REAL NOT NULL,
        idiosyncratic_variance REAL NOT NULL,
        market_equity_beta REAL NOT NULL,
        shadow_cost_lambda REAL NOT NULL,
        capm_required_return_pct REAL NOT NULL,
        merton_shadow_surcharge_pct REAL NOT NULL,
        merton_shadow_surcharge_bps REAL NOT NULL,
        total_required_hurdle_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO merton_incomplete_info_records (
        ticker, latest_price, investor_recognition_ratio_q,
        idiosyncratic_variance, market_equity_beta, shadow_cost_lambda,
        capm_required_return_pct, merton_shadow_surcharge_pct,
        merton_shadow_surcharge_bps, total_required_hurdle_pct,
        regime, audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                receipt.investor_recognition_ratio_q,
                receipt.idiosyncratic_variance,
                receipt.market_equity_beta,
                receipt.shadow_cost_lambda,
                receipt.capm_required_return_pct,
                receipt.merton_shadow_surcharge_pct,
                receipt.merton_shadow_surcharge_bps,
                receipt.total_required_hurdle_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
