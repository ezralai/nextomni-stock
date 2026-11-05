"""Harrison Hong & Jeremy C. Stein (2003) Differences of Opinion, Short-Sales Constraints & Market Crashes Engine.

Theoretical Foundation:
    Hong, H., & Stein, J. C. (2003). "Differences of Opinion, Short-Sales Constraints,
    and Market Crashes". The Journal of Finance, 58(2), 487-525.
    Chen, J., Hong, H., & Stein, J. C. (2001). "Forecasting Crashes: Trading Volume,
    Past Returns, and Conditional Skewness in Stock Prices". Journal of Financial Economics, 61(3), 345-381.

Key Insights:
    - Sidelined Pessimists: When investors hold heterogeneous beliefs and short
      selling is prohibited or costly, bearish investors sit out of the market.
    - Latent Negative Information Overhang: Unrevealed negative private signals
      accumulate invisibly behind the market price.
    - Clustered Revelation & Negative Skewness: When negative fundamental shocks
      occur, previously sidelined pessimists become marginal price setters,
      suddenly revealing accumulated bearish information in an asymmetric cascade
      that causes severe negative skewness and flash crash vulnerability.
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

HMAC_SECRET = b"nextomni-hong-stein-crash-v119-secret"

CRASH_REGIMES = (
    "ACUTE_CRASH_VULNERABILITY",
    "ELEVATED_LATENT_OVERHANG",
    "SYMMETRIC_UNCONSTRAINED",
    "CONSENSUS_STABILITY",
)

DEFAULT_OPINION_DISPERSION = 0.18      # 18% opinion dispersion sigma_theta
DEFAULT_SHORT_CONSTRAINT = 0.55        # 55% short-sales friction / borrowing impediment
DEFAULT_HISTORICAL_TURNOVER = 1.20     # Normalized turnover ratio


@dataclass(frozen=True)
class HongSteinCrashReceipt:
    """Immutable audit receipt for Hong & Stein (2003) crash vulnerability evaluation."""

    ticker: str
    latest_price: float
    opinion_dispersion_ratio: float       # sigma_theta in [0.01, 1.00]
    short_sale_constraint_tightness: float # C_short in [0.00, 1.00]
    turnover_ratio: float                 # Trading turnover multiplier
    latent_information_overhang: float    # Accumulated unrevealed bearish signals [0.0, 5.0]
    conditional_crash_skewness: float     # Negative return skewness S_crash <= 0
    crash_vulnerability_index: float      # Index [0.0, 100.0]
    downside_volatility_surcharge_pct: float # Extra downside volatility spread in %
    expected_crash_drawdown_pct: float    # Simulated cascade drawdown in %
    regime: str
    is_acute_crash_vulnerability: bool
    is_elevated_latent_overhang: bool
    is_symmetric_unconstrained: bool
    is_consensus_stability: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_hong_stein_audit_digest(
    ticker: str,
    latest_price: float,
    dispersion: float,
    constraint: float,
    overhang: float,
    skewness: float,
    crash_index: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Hong & Stein crash metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{dispersion:.4f}|{constraint:.4f}|{overhang:.4f}|{skewness:.4f}|{crash_index:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_hong_stein_audit_digest(receipt: HongSteinCrashReceipt) -> bool:
    """Verify cryptographic audit digest of a HongSteinCrashReceipt."""
    expected = generate_hong_stein_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.opinion_dispersion_ratio,
        receipt.short_sale_constraint_tightness,
        receipt.latent_information_overhang,
        receipt.conditional_crash_skewness,
        receipt.crash_vulnerability_index,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_crash_regime(
    dispersion: float,
    constraint: float,
    crash_index: float,
) -> str:
    """Classify into one of four Hong & Stein (2003) crash regimes."""
    if (dispersion >= 0.22 and constraint >= 0.55) or crash_index >= 65.0:
        return "ACUTE_CRASH_VULNERABILITY"
    elif (dispersion >= 0.12 and constraint >= 0.35) or crash_index >= 35.0:
        return "ELEVATED_LATENT_OVERHANG"
    elif constraint <= 0.18:
        return "SYMMETRIC_UNCONSTRAINED"
    else:
        return "CONSENSUS_STABILITY"


def calculate_hong_stein_crash(
    ticker: str,
    latest_price: float = 100.0,
    opinion_dispersion_ratio: float = DEFAULT_OPINION_DISPERSION,
    short_sale_constraint_tightness: float = DEFAULT_SHORT_CONSTRAINT,
    turnover_ratio: float = DEFAULT_HISTORICAL_TURNOVER,
    baseline_volatility: float = 0.22,
) -> HongSteinCrashReceipt:
    """Calculate Hong & Stein (2003) latent negative information overhang and crash skewness.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        opinion_dispersion_ratio: Investor belief dispersion sigma_theta in [0.01, 1.00].
        short_sale_constraint_tightness: Short-sales friction C_short in [0.00, 1.00].
        turnover_ratio: Normalized trading volume / turnover in [0.10, 10.0].
        baseline_volatility: Fundamental annualized volatility in [0.02, 2.00].

    Returns:
        HongSteinCrashReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.005 <= opinion_dispersion_ratio <= 1.50):
        raise SecurityValidationError("opinion_dispersion_ratio must be between 0.005 and 1.50.")
    if not (0.00 <= short_sale_constraint_tightness <= 1.00):
        raise SecurityValidationError("short_sale_constraint_tightness must be between 0.00 and 1.00.")
    if not (0.05 <= turnover_ratio <= 20.0):
        raise SecurityValidationError("turnover_ratio must be between 0.05 and 20.0.")
    if not (0.01 <= baseline_volatility <= 3.00):
        raise SecurityValidationError("baseline_volatility must be between 0.01 and 3.00.")

    clean_disp = float(np.clip(opinion_dispersion_ratio, 0.005, 1.50))
    clean_const = float(np.clip(short_sale_constraint_tightness, 0.00, 1.00))
    clean_turn = float(np.clip(turnover_ratio, 0.05, 20.0))
    clean_vol = float(np.clip(baseline_volatility, 0.01, 3.00))

    # 1. Latent Negative Information Overhang (I_latent)
    # Scaled by dispersion, short-sale friction, and turnover (Chen, Hong & Stein 2001)
    raw_overhang = clean_disp * clean_const * (1.0 + 0.35 * np.log(max(clean_turn, 0.1))) * 4.0
    latent_overhang = float(np.clip(raw_overhang, 0.0, 5.0))

    # 2. Conditional Crash Skewness (S_crash <= 0)
    # Hong & Stein: Negative skewness is zero when constraints are absent (clean_const -> 0)
    # or opinions are identical (clean_disp -> 0). It scales non-linearly with overhang.
    raw_skewness = -1.25 * (latent_overhang ** 1.2) * (clean_vol / 0.20)
    conditional_skewness = float(np.clip(raw_skewness, -4.50, 0.0))

    # 3. Crash Vulnerability Index [0, 100]
    raw_index = (latent_overhang / 3.0) * 65.0 + abs(conditional_skewness) * 12.0
    crash_index = float(np.clip(raw_index, 0.0, 100.0))

    # 4. Downside Volatility Surcharge (%)
    downside_vol_surcharge = abs(conditional_skewness) * 2.85 * clean_vol * 10.0
    downside_vol_surcharge = float(np.clip(downside_vol_surcharge, 0.0, 45.0))

    # 5. Expected Cascade Drawdown (%)
    raw_drawdown = (crash_index / 100.0) * (clean_vol * 45.0)
    expected_drawdown = float(np.clip(raw_drawdown, 0.0, 60.0))

    regime = classify_crash_regime(clean_disp, clean_const, crash_index)
    is_acute_crash = (regime == "ACUTE_CRASH_VULNERABILITY")
    is_elevated_overhang = (regime == "ELEVATED_LATENT_OVERHANG")
    is_symmetric = (regime == "SYMMETRIC_UNCONSTRAINED")
    is_consensus = (regime == "CONSENSUS_STABILITY")

    summary = (
        f"Hong-Stein Crash Asymmetry: {regime} "
        f"(Dispersion={clean_disp:.1%}, ShortConstraint={clean_const:.1%}, "
        f"LatentOverhang={latent_overhang:.2f}, Skewness={conditional_skewness:.2f}, "
        f"CrashIndex={crash_index:.1f}/100, DownsideVolSurcharge=+{downside_vol_surcharge:.1f}%, "
        f"ExpectedCascadeDrawdown={expected_drawdown:.1f}%). "
        f"{'Acute crash vulnerability from pent-up negative information.' if is_acute_crash else ('Elevated latent overhang; downside asymmetry present.' if is_elevated_overhang else 'Symmetric or consensus price discovery.')}"
    )

    clean_price = round(latest_price, 4)
    disp_rd = round(clean_disp, 4)
    const_rd = round(clean_const, 4)
    turn_rd = round(clean_turn, 2)
    overhang_rd = round(latent_overhang, 2)
    skewness_rd = round(conditional_skewness, 2)
    crash_index_rd = round(crash_index, 1)
    downside_vol_rd = round(downside_vol_surcharge, 2)
    drawdown_rd = round(expected_drawdown, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_hong_stein_audit_digest(
        clean_ticker,
        clean_price,
        disp_rd,
        const_rd,
        overhang_rd,
        skewness_rd,
        crash_index_rd,
        regime,
        timestamp,
    )

    return HongSteinCrashReceipt(
        ticker=clean_ticker,
        latest_price=clean_price,
        opinion_dispersion_ratio=disp_rd,
        short_sale_constraint_tightness=const_rd,
        turnover_ratio=turn_rd,
        latent_information_overhang=overhang_rd,
        conditional_crash_skewness=skewness_rd,
        crash_vulnerability_index=crash_index_rd,
        downside_volatility_surcharge_pct=downside_vol_rd,
        expected_crash_drawdown_pct=drawdown_rd,
        regime=regime,
        is_acute_crash_vulnerability=is_acute_crash,
        is_elevated_latent_overhang=is_elevated_overhang,
        is_symmetric_unconstrained=is_symmetric,
        is_consensus_stability=is_consensus,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_hong_stein_crash_profile(
    ticker: str,
    opinion_dispersion_override: Optional[float] = None,
    short_constraint_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> HongSteinCrashReceipt:
    """Evaluate Hong & Stein crash profile using empirical return skewness and turnover."""
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
                if "volume" in obs.columns:
                    vols = [
                        float(x)
                        for x in obs["volume"].tolist()
                        if math.isfinite(float(x)) and float(x) >= 0
                    ]
                    if vols and len(vols) >= 20:
                        volumes = vols
        except Exception:
            pass

    if prices is None or len(prices) < 20:
        latest_price = 100.0
        dispersion = 0.16 if opinion_dispersion_override is None else opinion_dispersion_override
        constraint = 0.50 if short_constraint_override is None else short_constraint_override
        turnover = 1.00
        vol = 0.22
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        vol = float(np.clip(daily_vol * np.sqrt(252.0), 0.05, 1.50))

        # Empirical daily return skewness
        if len(returns) >= 20:
            m3 = float(np.mean((returns - np.mean(returns)) ** 3))
            s3 = float(np.std(returns) ** 3)
            emp_skew = m3 / max(s3, 1e-6)
        else:
            emp_skew = -0.25

        # Inferred belief dispersion from volatility and negative skewness
        inferred_disp = float(np.clip(0.10 + abs(emp_skew) * 0.08, 0.02, 0.50))
        dispersion = opinion_dispersion_override if opinion_dispersion_override is not None else inferred_disp
        constraint = short_constraint_override if short_constraint_override is not None else DEFAULT_SHORT_CONSTRAINT

        # Turnover ratio
        if volumes and len(volumes) >= 20:
            recent_v = np.mean(volumes[-10:])
            hist_v = np.mean(volumes[-60:]) if len(volumes) >= 60 else np.mean(volumes)
            turnover = float(np.clip(recent_v / max(hist_v, 1.0), 0.20, 5.0))
        else:
            turnover = 1.10

    return calculate_hong_stein_crash(
        ticker=clean_ticker,
        latest_price=latest_price,
        opinion_dispersion_ratio=dispersion,
        short_sale_constraint_tightness=constraint,
        turnover_ratio=turnover,
        baseline_volatility=vol,
    )


def persist_hong_stein_crash_record(
    receipt: HongSteinCrashReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Hong & Stein crash receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS hong_stein_crash_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        opinion_dispersion_ratio REAL NOT NULL,
        short_sale_constraint_tightness REAL NOT NULL,
        turnover_ratio REAL NOT NULL,
        latent_information_overhang REAL NOT NULL,
        conditional_crash_skewness REAL NOT NULL,
        crash_vulnerability_index REAL NOT NULL,
        downside_volatility_surcharge_pct REAL NOT NULL,
        expected_crash_drawdown_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO hong_stein_crash_records (
        ticker, latest_price, opinion_dispersion_ratio,
        short_sale_constraint_tightness, turnover_ratio,
        latent_information_overhang, conditional_crash_skewness,
        crash_vulnerability_index, downside_volatility_surcharge_pct,
        expected_crash_drawdown_pct, regime, audit_digest, calculated_at
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
                receipt.opinion_dispersion_ratio,
                receipt.short_sale_constraint_tightness,
                receipt.turnover_ratio,
                receipt.latent_information_overhang,
                receipt.conditional_crash_skewness,
                receipt.crash_vulnerability_index,
                receipt.downside_volatility_surcharge_pct,
                receipt.expected_crash_drawdown_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

