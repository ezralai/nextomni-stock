"""Anat R. Admati & Paul Pfleiderer (1988) Volume Clustering & Execution Timing Engine.

Theoretical Foundation:
    Admati, A. R., & Pfleiderer, P. (1988). "A Theory of Intraday Patterns:
    Volume and Price Variability". Review of Financial Studies, 1(1), 3-40.

Key Insights:
    - The Intraday U-Shape Paradox: Trading volume and price variability cluster
      naturally at open and close across global equity markets.
    - Discretionary Liquidity Coordination: Liquidity traders with discretionary
      timing coordinate endogenously to trade at the same time. By clumping
      together, their combined order flow deepens market depth, diluting the
      adverse selection impact of informed traders.
    - Concentrated Trading Equilibrium: Informed traders follow the liquidity
      cluster to camouflage their private signals, yielding a self-fulfilling
      equilibrium of high volume, high depth, and minimized execution slippage.
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

HMAC_SECRET = b"nextomni-admati-pfleiderer-v112-secret"

CLUSTERING_REGIMES = (
    "CONCENTRATED_LIQUIDITY_CLUSTER",
    "BALANCED_INTRADAY_EQUILIBRIUM",
    "DISPERSED_LOW_DEPTH_TRANSITION",
    "HIGH_SLIPPAGE_MIDDAY_DROUGHT",
)


@dataclass(frozen=True)
class AdmatiPfleidererReceipt:
    """Immutable audit receipt for Admati-Pfleiderer volume clustering evaluation."""

    ticker: str
    latest_price: float
    discretionary_liquidity_ratio: float  # sigma_d^2 / (sigma_d^2 + sigma_n^2) in [0.05, 0.95]
    non_discretionary_variance: float     # sigma_n^2 (baseline noise trading)
    informed_trader_count: int            # k informed traders
    clumping_index: float                 # C_AP in [0.0, 1.0]
    market_depth_multiplier: float        # D_AP = 1 / sqrt(1 - C_AP) in [1.0, 5.0]
    kyle_lambda_reduction_pct: float      # % reduction in price impact vs dispersed baseline
    execution_slippage_discount_bps: float # Slippage savings in basis points
    recommended_execution_window: str     # e.g., 'OPEN_CLOSE_AUCTION_CLUSTER'
    regime: str
    is_concentrated_cluster: bool
    is_balanced_equilibrium: bool
    is_midday_drought: bool
    optimal_execution_timing: str
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_admati_audit_digest(
    ticker: str,
    latest_price: float,
    clumping_index: float,
    depth_multiplier: float,
    lambda_reduction: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over Admati-Pfleiderer equilibrium metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{clumping_index:.4f}|{depth_multiplier:.4f}|{lambda_reduction:.4f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_admati_audit_digest(receipt: AdmatiPfleidererReceipt) -> bool:
    """Verify cryptographic audit digest of an AdmatiPfleidererReceipt."""
    expected = generate_admati_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.clumping_index,
        receipt.market_depth_multiplier,
        receipt.kyle_lambda_reduction_pct,
        receipt.regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_clustering_regime(clumping_index: float, depth_multiplier: float) -> str:
    """Classify intraday volume clustering into Admati-Pfleiderer execution regimes."""
    if clumping_index >= 0.60 or depth_multiplier >= 1.58:
        return "CONCENTRATED_LIQUIDITY_CLUSTER"
    elif clumping_index >= 0.35 or depth_multiplier >= 1.24:
        return "BALANCED_INTRADAY_EQUILIBRIUM"
    elif clumping_index >= 0.15 or depth_multiplier >= 1.08:
        return "DISPERSED_LOW_DEPTH_TRANSITION"
    else:
        return "HIGH_SLIPPAGE_MIDDAY_DROUGHT"


def calculate_admati_pfleiderer_clustering(
    ticker: str,
    latest_price: float = 100.0,
    discretionary_liquidity_ratio: float = 0.50,  # 50% discretionary timing
    non_discretionary_variance: float = 1.0,      # baseline noise variance
    informed_trader_count: int = 3,               # number of informed market participants
    baseline_spread_bps: float = 12.0,            # baseline bid-ask spread
) -> AdmatiPfleidererReceipt:
    """Calculate Admati & Pfleiderer (1988) volume clustering and market depth metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        discretionary_liquidity_ratio: Ratio of discretionary liquidity demand in [0.05, 0.95].
        non_discretionary_variance: Variance of exogenous non-discretionary liquidity trades.
        informed_trader_count: Number of informed traders entering the cluster (k >= 1).
        baseline_spread_bps: Baseline execution spread in basis points.

    Returns:
        AdmatiPfleidererReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if not (0.02 <= discretionary_liquidity_ratio <= 0.98):
        raise SecurityValidationError("discretionary_liquidity_ratio must be between 0.02 and 0.98.")
    if non_discretionary_variance <= 0.0:
        raise SecurityValidationError("non_discretionary_variance must be strictly positive.")
    if not (1 <= informed_trader_count <= 50):
        raise SecurityValidationError("informed_trader_count must be between 1 and 50.")
    if baseline_spread_bps <= 0.0:
        raise SecurityValidationError("baseline_spread_bps must be strictly positive.")

    # Admati-Pfleiderer Clumping Index C_AP in [0.0, 1.0]
    clumping_index = float(np.clip(discretionary_liquidity_ratio, 0.02, 0.98))

    # Market Depth Multiplier D_AP = 1 / sqrt(1 - C_AP)
    # Reflects the pooling of discretionary noise liquidity:
    # Var(Total Noise) = sigma_n^2 / (1 - C_AP)
    # Market Depth D = sqrt(Var(Total Noise)) / (k * sigma_u)
    depth_multiplier = 1.0 / math.sqrt(max(1.0 - clumping_index, 0.04))
    depth_multiplier = float(np.clip(depth_multiplier, 1.0, 5.0))

    # Kyle Lambda price impact reduction: lambda_cluster = lambda_baseline / D_AP
    # Percentage reduction in price impact: (1 - 1 / D_AP) * 100%
    lambda_reduction_pct = (1.0 - (1.0 / depth_multiplier)) * 100.0

    # Slippage discount: basis points saved by trading in the concentrated cluster
    slippage_discount_bps = baseline_spread_bps * (lambda_reduction_pct / 100.0)

    regime = classify_clustering_regime(clumping_index, depth_multiplier)
    is_concentrated_cluster = (regime == "CONCENTRATED_LIQUIDITY_CLUSTER")
    is_balanced_equilibrium = (regime == "BALANCED_INTRADAY_EQUILIBRIUM")
    is_midday_drought = (regime == "HIGH_SLIPPAGE_MIDDAY_DROUGHT")

    if is_concentrated_cluster:
        rec_window = "OPEN_CLOSE_AUCTION_CLUSTER"
        timing_advice = "Concentrate 70%+ of VWAP/TWAP order flow into market open and close auction clusters; maximum depth minimizes slippage."
    elif is_balanced_equilibrium:
        rec_window = "CONTINUOUS_EQUILIBRIUM_SLICING"
        timing_advice = "Standard linear slice execution; liquidity is moderately balanced across the session."
    elif is_midday_drought:
        rec_window = "AVOID_MIDDAY_EXECUTION"
        timing_advice = "Halt large block executions during midday liquidity drought; adverse selection and Kyle lambda are elevated."
    else:
        rec_window = "CAUTIOUS_PASSIVE_LIMIT_ORDERS"
        timing_advice = "Use passive post-only limit orders to avoid crossing wide spreads in thinning liquidity."

    summary = (
        f"Admati-Pfleiderer Volume Clustering: {regime} "
        f"(ClumpingIndex={clumping_index:.1%}, DepthMultiplier={depth_multiplier:.2f}x, "
        f"PriceImpactReduction={lambda_reduction_pct:.1f}%, SlippageDiscount={slippage_discount_bps:.1f} bps). "
        f"{timing_advice}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_admati_audit_digest(
        clean_ticker,
        latest_price,
        clumping_index,
        depth_multiplier,
        lambda_reduction_pct,
        regime,
        timestamp,
    )

    return AdmatiPfleidererReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        discretionary_liquidity_ratio=round(clumping_index, 4),
        non_discretionary_variance=round(non_discretionary_variance, 4),
        informed_trader_count=informed_trader_count,
        clumping_index=round(clumping_index, 4),
        market_depth_multiplier=round(depth_multiplier, 4),
        kyle_lambda_reduction_pct=round(lambda_reduction_pct, 4),
        execution_slippage_discount_bps=round(slippage_discount_bps, 2),
        recommended_execution_window=rec_window,
        regime=regime,
        is_concentrated_cluster=is_concentrated_cluster,
        is_balanced_equilibrium=is_balanced_equilibrium,
        is_midday_drought=is_midday_drought,
        optimal_execution_timing=timing_advice,
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_admati_pfleiderer_profile(
    ticker: str,
    discretionary_ratio_override: Optional[float] = None,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    db_path: Optional[Union[str, Path]] = None,
) -> AdmatiPfleidererReceipt:
    """Evaluate Admati & Pfleiderer volume clustering profile from historical observations."""
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
        latest_price = 140.0
        inferred_clumping = 0.52
        inferred_spread = 14.0
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))

        # Effective spread proxy via Roll (1984) autocovariance
        cov_lag1 = float(np.cov(returns[:-1], returns[1:])[0, 1])
        inferred_spread = max(2.0 * math.sqrt(max(-cov_lag1, 0.0)) * 10000.0, 6.0)

        # Volume dispersion proxy for discretionary trading clumping
        if volumes and len(volumes) >= 20:
            vol_arr = np.array(volumes[-60:], dtype=float)
            vol_mean = float(np.mean(vol_arr))
            vol_std = float(np.std(vol_arr, ddof=1)) if len(vol_arr) > 1 else 0.0
            cv = vol_std / max(vol_mean, 1e-4)
            # High coefficient of variation in volume signals strong intraday/interday clumping
            inferred_clumping = float(np.clip(0.25 + 0.45 * min(cv, 1.5), 0.10, 0.85))
        else:
            inferred_clumping = 0.50

    ratio_val = float(discretionary_ratio_override) if discretionary_ratio_override is not None else inferred_clumping

    receipt = calculate_admati_pfleiderer_clustering(
        ticker=clean_ticker,
        latest_price=latest_price,
        discretionary_liquidity_ratio=ratio_val,
        baseline_spread_bps=inferred_spread,
    )

    persist_admati_pfleiderer_record(receipt, db_path=db_path)
    return receipt


def persist_admati_pfleiderer_record(
    receipt: AdmatiPfleidererReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Admati-Pfleiderer volume clustering receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS admati_pfleiderer_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        discretionary_liquidity_ratio REAL NOT NULL,
        clumping_index REAL NOT NULL,
        market_depth_multiplier REAL NOT NULL,
        kyle_lambda_reduction_pct REAL NOT NULL,
        execution_slippage_discount_bps REAL NOT NULL,
        recommended_execution_window TEXT NOT NULL,
        regime TEXT NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO admati_pfleiderer_records (
        ticker, latest_price, discretionary_liquidity_ratio,
        clumping_index, market_depth_multiplier,
        kyle_lambda_reduction_pct, execution_slippage_discount_bps,
        recommended_execution_window, regime, audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                receipt.discretionary_liquidity_ratio,
                receipt.clumping_index,
                receipt.market_depth_multiplier,
                receipt.kyle_lambda_reduction_pct,
                receipt.execution_slippage_discount_bps,
                receipt.recommended_execution_window,
                receipt.regime,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass

