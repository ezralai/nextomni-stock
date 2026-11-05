"""John Geanakoplos (2010) The Leverage Cycle, Collateral Margins & De-leveraging Engine.

Theoretical Foundation:
    Geanakoplos, J. (2010). "The Leverage Cycle".
    NBER Macroeconomics Annual, 24(1), 1-65.
    Geanakoplos, J. (2003). "Liquidity, Default, and Crashes: Endogenous Contracts in General Equilibrium".
    Advances in Economics and Econometrics, II, 170-205.
    Fostel, A., & Geanakoplos, J. (2008). "Leverage Cycles and the Anxious Economy".
    American Economic Review, 98(4), 1211-1244.

Key Insights:
    - Collateral Haircuts as Equilibrium Determinants: Asset prices are determined
      not only by interest rates and earnings, but fundamentally by collateral margin
      requirements (haircuts m_t).
    - Asymmetric Leverage Cycles: In euphoric expansions, low haircuts (m_t -> 0.05)
      permit high leverage (L_t = 1/m_t), enabling optimistic buyers to bid prices
      to lofty heights. When uncertainty rises, lenders demand higher haircuts (m_t -> 0.40+),
      forcing massive asset sales to meet margin calls, depressing prices and triggering
      de-leveraging fire-sales.
    - Endogenous Volatility & Credit Freezes: Margin spirals amplify downturns into
      systemic liquidity crashes and credit crunches.
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

HMAC_SECRET = b"nextomni-geanakoplos-leverage-v123-secret"

LEVERAGE_REGIMES = (
    "ACUTE_MARGIN_SQUEEZE_PANIC",
    "ELEVATED_DELEVERAGING_PHASE",
    "EUPHORIC_HIGH_LEVERAGE_EXPANSION",
    "PRUDENT_LOW_LEVERAGE_STABILITY",
)

DEFAULT_COLLATERAL_HAIRCUT = 0.20       # 20% margin requirement m_t
DEFAULT_BASELINE_HAIRCUT = 0.20         # 20% normal baseline haircut m_0
DEFAULT_RISK_FREE_RATE = 0.0425         # 4.25% Rf
DEFAULT_MARKET_EQUITY_PREMIUM = 0.0550  # 5.50% ERP


@dataclass(frozen=True)
class GeanakoplosLeverageReceipt:
    """Immutable audit receipt for Geanakoplos (2010) leverage cycle evaluation."""

    ticker: str
    latest_price: float
    collateral_haircut_m: float           # Current margin requirement m_t in [0.05, 0.85]
    baseline_haircut_m0: float            # Normal baseline haircut m_0 in [0.05, 0.60]
    effective_leverage_ratio: float       # L_t = 1 / m_t in [1.18, 20.0]
    haircut_tightening_ratio: float       # m_t / m_0 >= 0.1x
    forced_deleveraging_pressure: float   # Index [0.0, 100.0]
    collateral_squeeze_markdown_pct: float # Expected price markdown in %
    volatility_amplification_multiple: float # Vol amplification >= 1.0x
    leverage_hurdle_surcharge_pct: float  # Required return premium in % / year
    total_leverage_hurdle_pct: float      # CAPM + Surcharge in % / year
    regime: str
    is_acute_margin_squeeze: bool
    is_elevated_deleveraging: bool
    is_euphoric_expansion: bool
    is_prudent_stability: bool
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_leverage_audit_digest(
    ticker: str,
    latest_price: float,
    haircut_m: float,
    leverage_ratio: float,
    pressure: float,
    total_hurdle: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Geanakoplos leverage cycle metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{haircut_m:.4f}|{leverage_ratio:.4f}|{pressure:.4f}|{total_hurdle:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_leverage_audit_digest(receipt: GeanakoplosLeverageReceipt) -> bool:
    """Verify cryptographic audit digest of a GeanakoplosLeverageReceipt."""
    expected = generate_leverage_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.collateral_haircut_m,
        receipt.effective_leverage_ratio,
        receipt.forced_deleveraging_pressure,
        receipt.total_leverage_hurdle_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_leverage_regime(
    haircut_m: float,
    pressure: float,
) -> str:
    """Classify into one of four Geanakoplos (2010) leverage cycle regimes."""
    if haircut_m >= 0.45 or pressure >= 65.0:
        return "ACUTE_MARGIN_SQUEEZE_PANIC"
    elif haircut_m >= 0.28 or pressure >= 35.0:
        return "ELEVATED_DELEVERAGING_PHASE"
    elif haircut_m <= 0.12 and pressure <= 10.0:
        return "EUPHORIC_HIGH_LEVERAGE_EXPANSION"
    else:
        return "PRUDENT_LOW_LEVERAGE_STABILITY"


def calculate_geanakoplos_leverage(
    ticker: str,
    latest_price: float = 100.0,
    collateral_haircut_m: float = DEFAULT_COLLATERAL_HAIRCUT,
    baseline_haircut_m0: float = DEFAULT_BASELINE_HAIRCUT,
    asset_volatility: float = 0.22,
    market_beta: float = 1.05,
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE,
    market_equity_premium: float = DEFAULT_MARKET_EQUITY_PREMIUM,
) -> GeanakoplosLeverageReceipt:
    """Calculate Geanakoplos (2010) leverage cycle, collateral haircut, and de-leveraging metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        collateral_haircut_m: Margin haircut requirement m_t in [0.05, 0.85].
        baseline_haircut_m0: Normal baseline haircut m_0 in [0.05, 0.60].
        asset_volatility: Annualized asset volatility sigma in [0.02, 2.00].
        market_beta: Systematic market beta in [-2.0, 5.0].
        risk_free_rate: Risk-free rate Rf in [0.00, 0.15].
        market_equity_premium: Market equity risk premium ERP in [0.01, 0.15].

    Returns:
        GeanakoplosLeverageReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.04 <= collateral_haircut_m <= 0.90):
        raise SecurityValidationError("collateral_haircut_m must be between 0.04 and 0.90.")
    if not (0.04 <= baseline_haircut_m0 <= 0.70):
        raise SecurityValidationError("baseline_haircut_m0 must be between 0.04 and 0.70.")
    if not (0.01 <= asset_volatility <= 3.00):
        raise SecurityValidationError("asset_volatility must be between 0.01 and 3.00.")
    if not (-2.0 <= market_beta <= 5.0):
        raise SecurityValidationError("market_beta must be between -2.0 and 5.0.")

    clean_m = float(np.clip(collateral_haircut_m, 0.04, 0.90))
    clean_m0 = float(np.clip(baseline_haircut_m0, 0.04, 0.70))
    clean_vol = float(np.clip(asset_volatility, 0.01, 3.00))

    # 1. Effective Leverage Ratio L_t = 1 / m_t
    leverage_ratio = 1.0 / clean_m
    leverage_ratio = float(np.clip(leverage_ratio, 1.10, 25.0))

    # 2. Haircut Tightening Ratio: m_t / m_0
    tightening_ratio = clean_m / clean_m0
    tightening_ratio = float(np.clip(tightening_ratio, 0.10, 10.0))

    # 3. Forced De-leveraging Margin Pressure Index [0.0, 100.0]
    # Scales when haircut tightens above baseline, multiplied by baseline leverage
    baseline_leverage = 1.0 / clean_m0
    if clean_m > clean_m0:
        raw_pressure = ((clean_m - clean_m0) / clean_m) * baseline_leverage * 20.0
    else:
        raw_pressure = 0.0
    forced_deleveraging_pressure = float(np.clip(raw_pressure, 0.0, 100.0))

    # 4. Collateral Squeeze Fire-Sale Price Markdown (%)
    raw_markdown = (forced_deleveraging_pressure / 100.0) * (clean_vol * 18.0)
    collateral_squeeze_markdown_pct = float(np.clip(raw_markdown, 0.0, 45.0))

    # 5. Volatility Amplification Multiple >= 1.0x
    raw_vol_amp = 1.0 + (leverage_ratio - 1.0) * 0.10 * (forced_deleveraging_pressure / 100.0)
    vol_amp = float(np.clip(raw_vol_amp, 1.00, 4.50))

    # 6. Leverage Cycle Hurdle Surcharge (% / year)
    surcharge_pct = (forced_deleveraging_pressure / 100.0) * clean_vol * 3.80
    surcharge_pct = float(np.clip(surcharge_pct, 0.0, 25.0))

    # CAPM benchmark return
    capm_return = risk_free_rate + market_beta * market_equity_premium
    capm_return_pct = capm_return * 100.0

    # Total Leverage Hurdle Rate
    total_hurdle_pct = capm_return_pct + surcharge_pct

    regime = classify_leverage_regime(clean_m, forced_deleveraging_pressure)
    is_acute = (regime == "ACUTE_MARGIN_SQUEEZE_PANIC")
    is_elevated = (regime == "ELEVATED_DELEVERAGING_PHASE")
    is_euphoric = (regime == "EUPHORIC_HIGH_LEVERAGE_EXPANSION")
    is_prudent = (regime == "PRUDENT_LOW_LEVERAGE_STABILITY")

    summary = (
        f"Geanakoplos Leverage Cycle: {regime} "
        f"(Haircut_m={clean_m:.1%}, Baseline_m0={clean_m0:.1%}, Leverage={leverage_ratio:.2f}x, "
        f"Tightening={tightening_ratio:.2f}x, DeleveragingPressure={forced_deleveraging_pressure:.1f}/100, "
        f"FireSaleMarkdown={collateral_squeeze_markdown_pct:.1f}%, VolAmp={vol_amp:.2f}x, "
        f"Surcharge=+{surcharge_pct:.2f}%, TotalHurdle={total_hurdle_pct:.2f}%). "
        f"{'Acute margin squeeze panic; severe forced collateral liquidation.' if is_acute else ('Euphoric low-margin leverage expansion; speculative boom.' if is_euphoric else 'Prudent low-leverage or moderate de-leveraging stability.')}"
    )

    clean_p_rd = round(latest_price, 4)
    m_rd = round(clean_m, 4)
    m0_rd = round(clean_m0, 4)
    lev_rd = round(leverage_ratio, 2)
    tight_rd = round(tightening_ratio, 2)
    press_rd = round(forced_deleveraging_pressure, 1)
    mark_rd = round(collateral_squeeze_markdown_pct, 2)
    vol_amp_rd = round(vol_amp, 2)
    surch_rd = round(surcharge_pct, 2)
    hurdle_rd = round(total_hurdle_pct, 2)

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_leverage_audit_digest(
        clean_ticker,
        clean_p_rd,
        m_rd,
        lev_rd,
        press_rd,
        hurdle_rd,
        regime,
        timestamp,
    )

    return GeanakoplosLeverageReceipt(
        ticker=clean_ticker,
        latest_price=clean_p_rd,
        collateral_haircut_m=m_rd,
        baseline_haircut_m0=m0_rd,
        effective_leverage_ratio=lev_rd,
        haircut_tightening_ratio=tight_rd,
        forced_deleveraging_pressure=press_rd,
        collateral_squeeze_markdown_pct=mark_rd,
        volatility_amplification_multiple=vol_amp_rd,
        leverage_hurdle_surcharge_pct=surch_rd,
        total_leverage_hurdle_pct=hurdle_rd,
        regime=regime,
        is_acute_margin_squeeze=is_acute,
        is_elevated_deleveraging=is_elevated,
        is_euphoric_expansion=is_euphoric,
        is_prudent_stability=is_prudent,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_geanakoplos_leverage_profile(
    ticker: str,
    haircut_override: Optional[float] = None,
    baseline_haircut_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> GeanakoplosLeverageReceipt:
    """Evaluate Geanakoplos leverage profile using historical price volatility and drawdowns."""
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
        haircut = 0.20 if haircut_override is None else haircut_override
        baseline_m0 = 0.20 if baseline_haircut_override is None else baseline_haircut_override
        vol = 0.22
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        daily_vol = float(np.std(returns)) if len(returns) > 1 else 0.015
        vol = float(np.clip(daily_vol * np.sqrt(252.0), 0.05, 1.50))

        # Infer collateral haircut from asset volatility (lenders set haircuts higher on volatile collateral)
        inferred_m0 = float(np.clip(0.12 + vol * 0.40, 0.10, 0.50))
        baseline_m0 = baseline_haircut_override if baseline_haircut_override is not None else inferred_m0

        # Drawdown shock expands haircut requirement
        recent_drawdown = max(0.0, (float(np.max(prices[-30:])) - latest_price) / float(np.max(prices[-30:]))) if len(prices) >= 30 else 0.0
        inferred_m = float(np.clip(baseline_m0 + recent_drawdown * 0.85, 0.08, 0.75))
        haircut = haircut_override if haircut_override is not None else inferred_m

    return calculate_geanakoplos_leverage(
        ticker=clean_ticker,
        latest_price=latest_price,
        collateral_haircut_m=haircut,
        baseline_haircut_m0=baseline_m0,
        asset_volatility=vol,
    )


def persist_geanakoplos_leverage_record(
    receipt: GeanakoplosLeverageReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Geanakoplos leverage cycle receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS geanakoplos_leverage_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        collateral_haircut_m REAL NOT NULL,
        baseline_haircut_m0 REAL NOT NULL,
        effective_leverage_ratio REAL NOT NULL,
        haircut_tightening_ratio REAL NOT NULL,
        forced_deleveraging_pressure REAL NOT NULL,
        collateral_squeeze_markdown_pct REAL NOT NULL,
        volatility_amplification_multiple REAL NOT NULL,
        leverage_hurdle_surcharge_pct REAL NOT NULL,
        total_leverage_hurdle_pct REAL NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO geanakoplos_leverage_records (
        ticker, latest_price, collateral_haircut_m, baseline_haircut_m0,
        effective_leverage_ratio, haircut_tightening_ratio,
        forced_deleveraging_pressure, collateral_squeeze_markdown_pct,
        volatility_amplification_multiple, leverage_hurdle_surcharge_pct,
        total_leverage_hurdle_pct, regime, audit_digest, calculated_at
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
                receipt.collateral_haircut_m,
                receipt.baseline_haircut_m0,
                receipt.effective_leverage_ratio,
                receipt.haircut_tightening_ratio,
                receipt.forced_deleveraging_pressure,
                receipt.collateral_squeeze_markdown_pct,
                receipt.volatility_amplification_multiple,
                receipt.leverage_hurdle_surcharge_pct,
                receipt.total_leverage_hurdle_pct,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
