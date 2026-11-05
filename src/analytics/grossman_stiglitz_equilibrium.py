"""Sanford J. Grossman & Joseph E. Stiglitz (1980) Informational Equilibrium Engine.

Theoretical Foundation:
    Grossman, S. J., & Stiglitz, J. E. (1980). "On the Impossibility of
    Informationally Efficient Markets". American Economic Review, 70(3), 393-408.

Key Insights:
    - The Grossman-Stiglitz Paradox: If a market is perfectly informationally
      efficient (prices reveal all private information with zero noise), no
      agent has an incentive to bear the costly expense (c > 0) of acquiring
      information. If no one acquires information, prices cannot be efficient.
    - Equilibrium with Partially Noisy Prices: In rational expectations
      equilibrium, unobserved liquidity/noise supply shocks (x ~ N(0, sigma_x^2))
      prevent uninformed traders from perfectly inverting prices to deduce the
      private signal. This noise creates an informational rent that exactly
      offsets the marginal information acquisition cost (c).
    - Endogenous Informed Fraction: Solves for the equilibrium proportion
      lambda* in [0.01, 0.99] of informed market participants and equilibrium price
      informativeness I_GS in [0.0, 1.0].
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

HMAC_SECRET = b"nextomni-grossman-stiglitz-v109-secret"

INFORMATIONAL_REGIMES = (
    "INFORMATIONALLY_EFFICIENT_SATURATED",
    "NOISY_EQUILIBRIUM_OPPORTUNITY",
    "HIGH_NOISE_INEFFICIENT_DISLOCATION",
    "UNINFORMED_HERD_DOMINATED",
)


@dataclass(frozen=True)
class GrossmanStiglitzReceipt:
    """Immutable audit receipt for Grossman-Stiglitz informational equilibrium."""

    ticker: str
    latest_price: float
    fundamental_variance: float      # sigma_u^2
    signal_noise_variance: float     # sigma_epsilon^2
    noise_trader_variance: float     # sigma_x^2
    risk_aversion: float             # gamma
    information_cost: float          # c
    equilibrium_informed_fraction: float  # lambda* in [0.01, 0.99]
    price_informativeness: float     # I_GS in [0.0, 1.0]
    uninformed_posterior_variance: float  # Var(u | P)
    informed_posterior_variance: float    # Var(u | theta)
    gross_informational_value: float      # E[U_diff]
    net_informational_alpha: float        # Gross value - c
    noise_to_signal_ratio: float
    informational_regime: str
    is_efficient_saturated: bool
    is_noisy_opportunity: bool
    is_uninformed_herd: bool
    expected_alpha_boost_bps: float
    summary: str
    audit_digest: str
    calculated_at: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def generate_grossman_audit_digest(
    ticker: str,
    latest_price: float,
    lambda_star: float,
    informativeness: float,
    net_alpha: float,
    regime: str,
    timestamp: str,
) -> str:
    """Compute HMAC-SHA256 digest over equilibrium metrics."""
    payload = f"{ticker}|{latest_price:.4f}|{lambda_star:.4f}|{informativeness:.4f}|{net_alpha:.6f}|{regime}|{timestamp}"
    return hmac.new(HMAC_SECRET, payload.encode("utf-8"), hashlib.sha256).hexdigest()


def verify_grossman_audit_digest(receipt: GrossmanStiglitzReceipt) -> bool:
    """Verify cryptographic audit digest of a GrossmanStiglitzReceipt."""
    expected = generate_grossman_audit_digest(
        receipt.ticker,
        receipt.latest_price,
        receipt.equilibrium_informed_fraction,
        receipt.price_informativeness,
        receipt.net_informational_alpha,
        receipt.informational_regime,
        receipt.calculated_at,
    )
    return hmac.compare_digest(expected, receipt.audit_digest)


def classify_informational_regime(
    price_informativeness: float,
    lambda_star: float,
    noise_to_signal_ratio: float,
) -> str:
    """Classify the asset into one of four Grossman-Stiglitz informational regimes."""
    if price_informativeness >= 0.80 and lambda_star >= 0.40:
        return "INFORMATIONALLY_EFFICIENT_SATURATED"
    elif noise_to_signal_ratio >= 2.5 and price_informativeness < 0.40:
        return "HIGH_NOISE_INEFFICIENT_DISLOCATION"
    elif lambda_star <= 0.15:
        return "UNINFORMED_HERD_DOMINATED"
    else:
        return "NOISY_EQUILIBRIUM_OPPORTUNITY"


def solve_equilibrium_informed_fraction(
    fundamental_variance: float,
    signal_noise_variance: float,
    noise_trader_variance: float,
    risk_aversion: float,
    information_cost: float,
) -> Tuple[float, float, float, float]:
    """Solve for Grossman-Stiglitz endogenous equilibrium informed fraction lambda*.

    Returns:
        (lambda_star, informativeness, var_u_given_P, var_u_given_theta)
    """
    tau_u = 1.0 / max(fundamental_variance, 1e-8)
    tau_epsilon = 1.0 / max(signal_noise_variance, 1e-8)

    # Posterior variance for informed trader who observes private signal theta = u + epsilon
    # Var(u | theta) = 1 / (tau_u + tau_epsilon)
    var_u_given_theta = 1.0 / (tau_u + tau_epsilon)

    # In equilibrium: E[U(W_I)] / E[U(W_U)] = exp(gamma * c) * sqrt(Var(u|theta) / Var(u|P))
    # Equilibrium equality holds when:
    # Var(u | P(lambda)) / Var(u | theta) = exp(2 * gamma * c)
    target_variance_ratio = math.exp(2.0 * risk_aversion * information_cost)

    def compute_var_u_given_P(lam: float) -> Tuple[float, float]:
        """Compute uninformed posterior variance and informativeness for given lambda."""
        # Noise component in price signal from uninformed perspective:
        # Var(noise_in_price) = sigma_epsilon^2 + (gamma^2 / lam^2) * sigma_x^2
        noise_in_price_var = signal_noise_variance + (risk_aversion ** 2 / (lam ** 2)) * noise_trader_variance
        tau_price = 1.0 / max(noise_in_price_var, 1e-8)

        # Informativeness index I_GS: fraction of private signal precision revealed in price
        # I_GS = tau_price / tau_epsilon = lam^2 / (lam^2 + gamma^2 * (sigma_x^2 / sigma_epsilon^2))
        noise_ratio = (risk_aversion ** 2) * (noise_trader_variance / max(signal_noise_variance, 1e-8))
        informativeness = (lam ** 2) / max((lam ** 2) + noise_ratio, 1e-8)
        informativeness = float(np.clip(informativeness, 0.0, 1.0))

        # Posterior variance for uninformed trader observing noisy price P
        # Var(u | P) = 1 / (tau_u + tau_price)
        var_P = 1.0 / (tau_u + tau_price)
        return var_P, informativeness

    # Monotonic bisection for lambda* in [0.01, 0.99]
    low_lam, high_lam = 0.01, 0.99
    var_low, _ = compute_var_u_given_P(low_lam)
    ratio_low = var_low / var_u_given_theta

    var_high, _ = compute_var_u_given_P(high_lam)
    ratio_high = var_high / var_u_given_theta

    if ratio_low <= target_variance_ratio:
        lambda_star = low_lam
    elif ratio_high >= target_variance_ratio:
        lambda_star = high_lam
    else:
        for _ in range(60):
            mid_lam = 0.5 * (low_lam + high_lam)
            var_mid, _ = compute_var_u_given_P(mid_lam)
            ratio_mid = var_mid / var_u_given_theta
            if abs(ratio_mid - target_variance_ratio) < 1e-6:
                lambda_star = mid_lam
                break
            if ratio_mid > target_variance_ratio:
                low_lam = mid_lam
            else:
                high_lam = mid_lam
        else:
            lambda_star = 0.5 * (low_lam + high_lam)

    lambda_star = float(np.clip(lambda_star, 0.01, 0.99))
    final_var_P, final_informativeness = compute_var_u_given_P(lambda_star)

    return lambda_star, final_informativeness, final_var_P, var_u_given_theta


def calculate_grossman_stiglitz_equilibrium(
    ticker: str,
    latest_price: float = 100.0,
    fundamental_variance: float = 0.0004,      # e.g., daily variance of fundamental cashflow ~ 2% vol
    signal_noise_variance: float = 0.0001,     # analyst signal precision
    noise_trader_variance: float = 0.0002,     # liquidity / order flow supply variance
    risk_aversion: float = 2.50,              # CARA coefficient gamma
    information_cost: float = 0.0020,         # cost c in return terms (20 bps)
    observed_lambda_override: Optional[float] = None,
) -> GrossmanStiglitzReceipt:
    """Calculate Grossman-Stiglitz (1980) informational equilibrium metrics.

    Args:
        ticker: Security ticker symbol.
        latest_price: Current market price ($).
        fundamental_variance: Payoff variance sigma_u^2 (> 0).
        signal_noise_variance: Private signal noise sigma_epsilon^2 (> 0).
        noise_trader_variance: Unobserved noise supply variance sigma_x^2 (> 0).
        risk_aversion: CARA risk aversion gamma (> 0).
        information_cost: Information acquisition cost c (> 0).
        observed_lambda_override: Optional active informed fraction for disequilibrium analysis.

    Returns:
        GrossmanStiglitzReceipt with certified metrics and audit digest.
    """
    clean_ticker = validate_ticker(ticker)

    if latest_price <= 0.0:
        raise SecurityValidationError("latest_price must be strictly positive.")
    if fundamental_variance <= 0.0:
        raise SecurityValidationError("fundamental_variance must be strictly positive.")
    if signal_noise_variance <= 0.0:
        raise SecurityValidationError("signal_noise_variance must be strictly positive.")
    if noise_trader_variance <= 0.0:
        raise SecurityValidationError("noise_trader_variance must be strictly positive.")
    if not (0.1 <= risk_aversion <= 25.0):
        raise SecurityValidationError("risk_aversion must be between 0.1 and 25.0.")
    if not (0.00005 <= information_cost <= 0.20):
        raise SecurityValidationError("information_cost must be between 0.00005 (0.5 bps) and 0.20 (20%).")

    # 1. Equilibrium Solution
    lambda_eq, informativeness_eq, var_u_P_eq, var_u_theta = solve_equilibrium_informed_fraction(
        fundamental_variance,
        signal_noise_variance,
        noise_trader_variance,
        risk_aversion,
        information_cost,
    )

    eval_lambda = observed_lambda_override if observed_lambda_override is not None else lambda_eq
    eval_lambda = float(np.clip(eval_lambda, 0.01, 0.99))

    # Compute parameters at eval_lambda
    tau_u = 1.0 / fundamental_variance
    tau_eps = 1.0 / signal_noise_variance
    noise_in_price_var = signal_noise_variance + (risk_aversion ** 2 / (eval_lambda ** 2)) * noise_trader_variance
    tau_price = 1.0 / max(noise_in_price_var, 1e-8)
    var_u_P = 1.0 / (tau_u + tau_price)

    noise_ratio = (risk_aversion ** 2) * (noise_trader_variance / max(signal_noise_variance, 1e-8))
    price_informativeness = float(np.clip((eval_lambda ** 2) / max((eval_lambda ** 2) + noise_ratio, 1e-8), 0.0, 1.0))

    # Gross Informational Value:
    # Benefit = (1 / (2 * gamma)) * ln( Var(u | P) / Var(u | theta) )
    var_ratio = max(var_u_P / max(var_u_theta, 1e-8), 1.0)
    gross_info_value = (1.0 / (2.0 * risk_aversion)) * math.log(var_ratio)
    net_informational_alpha = gross_info_value - information_cost

    noise_to_signal_ratio = float(noise_trader_variance / max(fundamental_variance, 1e-8))

    regime = classify_informational_regime(price_informativeness, eval_lambda, noise_to_signal_ratio)
    is_efficient_saturated = (regime == "INFORMATIONALLY_EFFICIENT_SATURATED")
    is_noisy_opportunity = (regime == "NOISY_EQUILIBRIUM_OPPORTUNITY")
    is_uninformed_herd = (regime == "UNINFORMED_HERD_DOMINATED")

    if is_noisy_opportunity:
        expected_alpha_boost_bps = float(np.clip(net_informational_alpha * 10000.0 * 252.0, -100.0, 350.0))
    elif is_efficient_saturated:
        expected_alpha_boost_bps = float(np.clip(net_informational_alpha * 10000.0 * 252.0, -50.0, 20.0))
    elif regime == "HIGH_NOISE_INEFFICIENT_DISLOCATION":
        expected_alpha_boost_bps = float(np.clip(net_informational_alpha * 10000.0 * 252.0, 50.0, 500.0))
    else:
        expected_alpha_boost_bps = -25.0

    summary = (
        f"Grossman-Stiglitz Equilibrium: {regime} "
        f"(Lambda*={eval_lambda:.1%}, Informativeness={price_informativeness:.1%}, "
        f"NetAlpha={net_informational_alpha * 10000.0:+.1f} bps/session, Cost={information_cost * 10000.0:.1f} bps). "
        f"{'Prices reflect high private information; informational rents are competitive.' if is_efficient_saturated else 'Partially noisy equilibrium enables informed quantitative intelligence to capture alpha.'}"
    )

    timestamp = datetime.now(timezone.utc).isoformat()
    audit_digest = generate_grossman_audit_digest(
        clean_ticker,
        latest_price,
        eval_lambda,
        price_informativeness,
        net_informational_alpha,
        regime,
        timestamp,
    )

    return GrossmanStiglitzReceipt(
        ticker=clean_ticker,
        latest_price=round(latest_price, 4),
        fundamental_variance=round(fundamental_variance, 8),
        signal_noise_variance=round(signal_noise_variance, 8),
        noise_trader_variance=round(noise_trader_variance, 8),
        risk_aversion=round(risk_aversion, 4),
        information_cost=round(information_cost, 6),
        equilibrium_informed_fraction=round(eval_lambda, 4),
        price_informativeness=round(price_informativeness, 4),
        uninformed_posterior_variance=round(var_u_P, 8),
        informed_posterior_variance=round(var_u_theta, 8),
        gross_informational_value=round(gross_info_value, 6),
        net_informational_alpha=round(net_informational_alpha, 6),
        noise_to_signal_ratio=round(noise_to_signal_ratio, 4),
        informational_regime=regime,
        is_efficient_saturated=is_efficient_saturated,
        is_noisy_opportunity=is_noisy_opportunity,
        is_uninformed_herd=is_uninformed_herd,
        expected_alpha_boost_bps=round(expected_alpha_boost_bps, 2),
        summary=summary,
        audit_digest=audit_digest,
        calculated_at=timestamp,
    )


def evaluate_grossman_stiglitz_profile(
    ticker: str,
    prices: Optional[List[float]] = None,
    volumes: Optional[List[float]] = None,
    risk_aversion: float = 2.50,
    information_cost: float = 0.0020,
    db_path: Optional[Union[str, Path]] = None,
) -> GrossmanStiglitzReceipt:
    """Evaluate Grossman-Stiglitz informational equilibrium profile from empirical prices."""
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
        latest_price = 150.0
        fundamental_variance = 0.00035
        signal_noise_variance = 0.00010
        noise_trader_variance = 0.00022
    else:
        latest_price = float(prices[-1])
        returns = np.diff(np.log(np.array(prices)))
        total_variance = float(np.var(returns))

        # Roll (1984) autocovariance decomposition of microstructure noise
        cov_lag1 = float(np.cov(returns[:-1], returns[1:])[0, 1])
        if cov_lag1 < 0:
            noise_variance = 2.0 * abs(cov_lag1)
        else:
            noise_variance = 0.25 * total_variance

        noise_trader_variance = float(np.clip(noise_variance, 0.05 * total_variance, 0.85 * total_variance))
        fundamental_variance = float(np.clip(total_variance - noise_trader_variance, 0.15 * total_variance, total_variance))
        signal_noise_variance = float(np.clip(0.30 * fundamental_variance, 1e-6, 0.01))

    receipt = calculate_grossman_stiglitz_equilibrium(
        ticker=clean_ticker,
        latest_price=latest_price,
        fundamental_variance=fundamental_variance,
        signal_noise_variance=signal_noise_variance,
        noise_trader_variance=noise_trader_variance,
        risk_aversion=risk_aversion,
        information_cost=information_cost,
    )

    persist_grossman_stiglitz_record(receipt, db_path=db_path)
    return receipt


def persist_grossman_stiglitz_record(
    receipt: GrossmanStiglitzReceipt,
    db_path: Optional[Union[str, Path]] = None,
) -> None:
    """Persist Grossman-Stiglitz audit receipt into SQLite database."""
    create_sql = """
    CREATE TABLE IF NOT EXISTS grossman_stiglitz_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT NOT NULL,
        latest_price REAL NOT NULL,
        fundamental_variance REAL NOT NULL,
        signal_noise_variance REAL NOT NULL,
        noise_trader_variance REAL NOT NULL,
        risk_aversion REAL NOT NULL,
        information_cost REAL NOT NULL,
        equilibrium_informed_fraction REAL NOT NULL,
        price_informativeness REAL NOT NULL,
        uninformed_posterior_variance REAL NOT NULL,
        informed_posterior_variance REAL NOT NULL,
        gross_informational_value REAL NOT NULL,
        net_informational_alpha REAL NOT NULL,
        noise_to_signal_ratio REAL NOT NULL,
        informational_regime TEXT NOT NULL,
        expected_alpha_boost_bps REAL NOT NULL,
        audit_digest TEXT NOT NULL,
        calculated_at TEXT NOT NULL
    );
    """
    insert_sql = """
    INSERT INTO grossman_stiglitz_records (
        ticker, latest_price, fundamental_variance, signal_noise_variance,
        noise_trader_variance, risk_aversion, information_cost,
        equilibrium_informed_fraction, price_informativeness,
        uninformed_posterior_variance, informed_posterior_variance,
        gross_informational_value, net_informational_alpha,
        noise_to_signal_ratio, informational_regime, expected_alpha_boost_bps,
        audit_digest, calculated_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                receipt.fundamental_variance,
                receipt.signal_noise_variance,
                receipt.noise_trader_variance,
                receipt.risk_aversion,
                receipt.information_cost,
                receipt.equilibrium_informed_fraction,
                receipt.price_informativeness,
                receipt.uninformed_posterior_variance,
                receipt.informed_posterior_variance,
                receipt.gross_informational_value,
                receipt.net_informational_alpha,
                receipt.noise_to_signal_ratio,
                receipt.informational_regime,
                receipt.expected_alpha_boost_bps,
                receipt.audit_digest,
                receipt.calculated_at,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
