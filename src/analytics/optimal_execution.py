"""Almgren & Chriss (2000) Optimal Execution Trajectory & Market Impact Engine.

Implements the classic Almgren & Chriss (2000) discrete-time optimal liquidation
and acquisition framework. Minimizes expected transaction costs (permanent and
temporary market impact) and portfolio timing risk (variance of implementation
shortfall) across multi-session execution horizons.

References:
    Almgren, R., & Chriss, N. (2000). Optimal execution of portfolio transactions.
    Journal of Risk, 3(2), 5-40.
    Kissell, R., & Glantz, M. (2003). Optimal Trading Strategies. AMACOM.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import hmac
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker


OPTIMAL_EXECUTION_SIGNING_KEY = b"nextomni_almgren_chriss_optimal_execution_key_v1"


@dataclass
class ExecutionStep:
    """Detailed breakdown of a single trading period in the optimal trajectory."""

    step: int  # Period index 1..N
    time_day: float  # Elapsed trading time in days
    inventory_start: float  # Shares remaining at start of period (x_{j-1})
    trade_shares: float  # Shares executed in this period (n_j)
    inventory_remaining: float  # Shares remaining at end of period (x_j)
    percent_executed_pct: float  # Cumulative percentage executed


@dataclass
class ExecutionTrajectoryReceipt:
    """Cryptographic audit receipt for an Almgren-Chriss optimal execution plan."""

    ticker: str
    target_shares: float  # X_0: Total shares to execute (positive = sell, negative = buy)
    execution_days: int  # N: Total horizon in trading days
    initial_price: float  # S_0: Benchmark initial stock price
    total_market_value: float  # |X_0| * S_0
    daily_volatility_pct: float  # Daily percentage return volatility (sigma)
    daily_volume: float  # Average daily volume (ADV) in shares
    risk_aversion_lambda: float  # Risk aversion parameter lambda
    urgency_parameter_kappa: float  # Kappa: rate of inventory decay
    half_life_days: float  # t_{1/2} = ln(2) / kappa
    expected_permanent_impact_dollars: float  # E_perm = 0.5 * gamma * X_0^2
    expected_temporary_impact_dollars: float  # E_temp = eta * sum(n_j^2)
    expected_total_shortfall_dollars: float  # E[x] = E_perm + E_temp
    expected_total_shortfall_bps: float  # (E[x] / total_market_value) * 10,000 bps
    shortfall_variance_dollars_sq: float  # V[x] = sigma^2 * sum(x_j^2)
    shortfall_std_dev_dollars: float  # sqrt(V[x])
    trajectory_schedule: List[Dict[str, Any]]
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature over trajectory metrics."""
        payload = (
            f"{self.ticker}:{self.target_shares:.2f}:{self.execution_days}:"
            f"{self.initial_price:.4f}:{self.risk_aversion_lambda:.8f}:"
            f"{self.urgency_parameter_kappa:.6f}:{self.half_life_days:.4f}:"
            f"{self.expected_total_shortfall_dollars:.4f}:{self.expected_total_shortfall_bps:.2f}:"
            f"{self.shortfall_std_dev_dollars:.4f}"
        )
        return hmac.new(
            OPTIMAL_EXECUTION_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity of the execution plan and cryptographic signature."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def estimate_market_impact_parameters(
    prices: Union[pd.Series, np.ndarray, List[float]],
    volumes: Union[pd.Series, np.ndarray, List[float]],
) -> Tuple[float, float, float, float]:
    """Estimate daily volatility, average daily volume, permanent, and temporary impact coefficients.

    Formulas:
        sigma: Standard deviation of log daily returns.
        ADV: Rolling median or mean volume.
        gamma: Permanent impact coefficient ($/share):
            gamma = 0.1 * (sigma * S_0) / (0.10 * ADV)
        eta: Temporary impact coefficient ($/(share/day)):
            eta = (0.5 * spread) + 0.1 * (sigma * S_0) / (0.01 * ADV)

    Returns:
        Tuple of (daily_vol_pct, adv_shares, gamma, eta).
    """
    p_arr = np.asarray(prices, dtype=float)
    v_arr = np.asarray(volumes, dtype=float)

    if len(p_arr) < 10:
        raise ValueError(f"Insufficient history ({len(p_arr)}), need at least 10 observations")

    ret = np.diff(np.log(np.maximum(1e-8, p_arr)))
    daily_vol = float(np.std(ret, ddof=1))
    daily_vol = max(0.002, min(0.15, daily_vol))  # Safe bound: 0.2% to 15% daily vol

    s0 = float(p_arr[-1])
    adv = float(np.median(v_arr[-min(len(v_arr), 30):]))
    adv = max(10_000.0, adv)

    # Permanent price impact per share ($/share)
    # Price moves approximately 10% of daily volatility for 10% of ADV
    gamma = 0.10 * (daily_vol * s0) / (0.10 * adv)

    # Temporary impact coefficient ($/(share/day))
    # Includes half bid-ask spread proxy (~5 bps) + liquidity absorption friction
    spread_proxy = 0.0005 * s0
    eta = spread_proxy + (0.10 * (daily_vol * s0) / (0.01 * adv))

    return float(daily_vol * 100.0), float(adv), float(gamma), float(eta)


def compute_almgren_chriss_trajectory(
    target_shares: float,
    execution_days: int,
    initial_price: float,
    daily_volatility_pct: float,
    daily_volume: float,
    risk_aversion_lambda: float = 1e-6,
    gamma: Optional[float] = None,
    eta: Optional[float] = None,
) -> Tuple[List[ExecutionStep], float, float, float, float, float, float]:
    """Calculate the Almgren-Chriss (2000) optimal discrete execution trajectory.

    Args:
        target_shares: Total order size X_0 in shares (positive = sell/liquidation, negative = buy).
        execution_days: Number of trading periods N (in days, tau = 1.0 day).
        initial_price: Current market price S_0.
        daily_volatility_pct: Daily return volatility as percentage (e.g. 1.8 for 1.8%/day).
        daily_volume: Average daily volume ADV.
        risk_aversion_lambda: Investor risk aversion lambda >= 0.
        gamma: Optional permanent impact parameter. If None, estimated from ADV.
        eta: Optional temporary impact parameter. If None, estimated from ADV.

    Returns:
        Tuple of:
            (schedule, kappa, half_life, e_perm, e_temp, e_total, var_shortfall)
    """
    if execution_days < 1:
        raise ValueError(f"Execution days must be at least 1, got {execution_days}")
    if abs(target_shares) < 1e-6:
        raise ValueError("Target shares must be non-zero")
    if initial_price <= 0.0:
        raise ValueError(f"Initial price must be strictly positive, got {initial_price}")
    if risk_aversion_lambda < 0.0:
        raise ValueError(f"Risk aversion lambda must be non-negative, got {risk_aversion_lambda}")

    n_steps = int(execution_days)
    tau = 1.0  # 1 session / day
    tot_t = n_steps * tau
    sigma_daily = (daily_volatility_pct / 100.0) * initial_price  # Dollar volatility per share per day
    x0 = float(target_shares)
    abs_x0 = abs(x0)

    # Default impact coefficients if not provided
    if gamma is None:
        gamma = 0.10 * sigma_daily / (0.10 * max(10_000.0, daily_volume))
    if eta is None:
        eta = (0.0005 * initial_price) + (0.10 * sigma_daily / (0.01 * max(10_000.0, daily_volume)))

    gamma = max(1e-12, float(gamma))
    eta = max(1e-12, float(eta))

    # Single-step edge case
    if n_steps == 1:
        step_obj = ExecutionStep(
            step=1,
            time_day=1.0,
            inventory_start=x0,
            trade_shares=x0,
            inventory_remaining=0.0,
            percent_executed_pct=100.0,
        )
        e_perm = 0.5 * gamma * (abs_x0 ** 2)
        e_temp = eta * (abs_x0 ** 2)
        e_total = e_perm + e_temp
        v_shortfall = 0.0
        return [step_obj], 0.0, 0.5, e_perm, e_temp, e_total, v_shortfall

    # Compute urgency parameter kappa
    # Discrete formula: cosh(kappa * tau) = 1 + (lambda * sigma^2 * tau^2) / (2 * eta)
    if risk_aversion_lambda <= 1e-12:
        # Risk-neutral TWAP trajectory: linear inventory decay
        kappa = 0.0
        half_life = float(n_steps) / 2.0
        x_trajectory = [x0 * (1.0 - (float(j) / float(n_steps))) for j in range(n_steps + 1)]
    else:
        arg = 1.0 + (risk_aversion_lambda * (sigma_daily ** 2) * (tau ** 2)) / (2.0 * eta)
        # Numerical guard for arccosh(x >= 1.0)
        safe_arg = max(1.0, arg)
        kappa = math.acosh(safe_arg) / tau
        half_life = math.log(2.0) / max(1e-8, kappa)

        # Inventory path: x_j = sinh(kappa * (T - t_j)) / sinh(kappa * T) * X_0
        sinh_kt = math.sinh(min(70.0, kappa * tot_t))
        x_trajectory = []
        for j in range(n_steps + 1):
            t_j = float(j) * tau
            if j == n_steps:
                x_j = 0.0
            else:
                sinh_rem = math.sinh(min(70.0, kappa * (tot_t - t_j)))
                x_j = (sinh_rem / sinh_kt) * x0
            x_trajectory.append(x_j)

    # Compute step trades n_j = x_{j-1} - x_j
    schedule: List[ExecutionStep] = []
    e_temp = 0.0
    v_shortfall = 0.0
    cum_traded = 0.0

    for j in range(1, n_steps + 1):
        x_start = x_trajectory[j - 1]
        x_end = x_trajectory[j]
        n_j = x_start - x_end
        cum_traded += abs(n_j)
        pct_exec = min(100.0, (cum_traded / abs_x0) * 100.0)

        step_obj = ExecutionStep(
            step=j,
            time_day=float(j) * tau,
            inventory_start=x_start,
            trade_shares=n_j,
            inventory_remaining=x_end,
            percent_executed_pct=pct_exec,
        )
        schedule.append(step_obj)

        # Temporary impact cost: sum(eta * n_j^2 / tau)
        e_temp += eta * (n_j ** 2) / tau
        # Shortfall variance: sum(tau * sigma^2 * x_j^2)
        v_shortfall += tau * (sigma_daily ** 2) * (x_end ** 2)

    # Permanent impact cost: 0.5 * gamma * X_0^2
    e_perm = 0.5 * gamma * (abs_x0 ** 2)
    e_total = e_perm + e_temp

    return schedule, float(kappa), float(half_life), float(e_perm), float(e_temp), float(e_total), float(v_shortfall)


def evaluate_execution_frontier(
    target_shares: float,
    execution_days: int,
    initial_price: float,
    daily_volatility_pct: float,
    daily_volume: float,
    num_points: int = 10,
) -> List[Dict[str, float]]:
    """Generate the Efficient Execution Frontier across risk aversion parameters.

    Returns a list of points containing (lambda, expected_shortfall, shortfall_std_dev, half_life).
    """
    frontier = []
    # Grid of lambda from 0 (TWAP / risk-neutral) to 1e-4 (highly risk averse)
    lambdas = [0.0] + list(np.logspace(-7, -4, num_points - 1))

    for lam in lambdas:
        _, kappa, half_life, _, _, e_tot, v_short = compute_almgren_chriss_trajectory(
            target_shares=target_shares,
            execution_days=execution_days,
            initial_price=initial_price,
            daily_volatility_pct=daily_volatility_pct,
            daily_volume=daily_volume,
            risk_aversion_lambda=float(lam),
        )
        frontier.append({
            "lambda": float(lam),
            "expected_shortfall_dollars": float(e_tot),
            "shortfall_std_dev_dollars": float(math.sqrt(max(0.0, v_short))),
            "half_life_days": float(half_life),
        })

    return frontier


def evaluate_optimal_execution_plan(
    ticker: str,
    target_shares: float,
    execution_days: int = 5,
    risk_aversion_lambda: float = 1e-6,
    prices: Optional[pd.Series] = None,
    volumes: Optional[pd.Series] = None,
    db_path: Optional[Path] = None,
) -> ExecutionTrajectoryReceipt:
    """Evaluate comprehensive Almgren & Chriss optimal execution plan with cryptographic receipt.

    Args:
        ticker: Target equity ticker symbol.
        target_shares: Desired share quantity (positive for sell/liquidation, negative for buy).
        execution_days: Horizon in trading sessions (default: 5 sessions).
        risk_aversion_lambda: Risk aversion multiplier (default: 1e-6).
        prices: Optional historical price series. If None, queried from DB.
        volumes: Optional historical volume series. If None, queried from DB.
        db_path: Optional SQLite database path.

    Returns:
        ExecutionTrajectoryReceipt with complete trajectory breakdown and audit signature.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(target_shares, "target_shares")
    assert_finite_number(execution_days, "execution_days")
    assert_finite_number(risk_aversion_lambda, "risk_aversion_lambda")

    if abs(target_shares) < 1.0:
        raise ValueError(f"Target shares must be at least 1, got {target_shares}")
    if execution_days < 1 or execution_days > 60:
        raise ValueError(f"Execution days must be between 1 and 60, got {execution_days}")
    if risk_aversion_lambda < 0.0:
        raise ValueError(f"Risk aversion lambda must be >= 0, got {risk_aversion_lambda}")

    # Fetch observations if not provided
    if prices is None or volumes is None or len(prices) < 10:
        df_obs = db.get_observations(clean_ticker, db_path=db_path)
        if df_obs.empty or len(df_obs) < 10:
            raise ValueError(f"Insufficient historical observations for {clean_ticker} in database")
        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
        v_col = "volume"
        prices = df_obs[p_col].astype(float)
        volumes = df_obs[v_col].astype(float)

    # Estimate empirical volatility and market impact parameters
    daily_vol_pct, adv, gamma, eta = estimate_market_impact_parameters(prices, volumes)
    s0 = float(prices.iloc[-1])
    tot_value = abs(float(target_shares)) * s0

    # Compute optimal execution trajectory
    schedule, kappa, half_life, e_perm, e_temp, e_tot, v_short = compute_almgren_chriss_trajectory(
        target_shares=float(target_shares),
        execution_days=int(execution_days),
        initial_price=s0,
        daily_volatility_pct=daily_vol_pct,
        daily_volume=adv,
        risk_aversion_lambda=float(risk_aversion_lambda),
        gamma=gamma,
        eta=eta,
    )

    std_dev_shortfall = math.sqrt(max(0.0, v_short))
    shortfall_bps = (e_tot / max(1.0, tot_value)) * 10_000.0

    schedule_dicts = [asdict(step) for step in schedule]
    action_str = "Liquidation" if target_shares > 0 else "Acquisition"

    summary = (
        f"Almgren-Chriss Optimal {action_str} for {clean_ticker}: {abs(target_shares):,.0f} shares over "
        f"{execution_days} sessions (Market Value: ${tot_value:,.2f}, S_0: ${s0:.2f}). "
        f"Expected Shortfall: ${e_tot:,.2f} ({shortfall_bps:.1f} bps | Perm: ${e_perm:,.2f}, Temp: ${e_temp:,.2f}), "
        f"Timing Risk (StdDev): ${std_dev_shortfall:,.2f}. Urgency (Kappa): {kappa:.4f}, Half-Life: {half_life:.2f}D."
    )

    receipt = ExecutionTrajectoryReceipt(
        ticker=clean_ticker,
        target_shares=float(target_shares),
        execution_days=int(execution_days),
        initial_price=s0,
        total_market_value=tot_value,
        daily_volatility_pct=daily_vol_pct,
        daily_volume=adv,
        risk_aversion_lambda=float(risk_aversion_lambda),
        urgency_parameter_kappa=kappa,
        half_life_days=half_life,
        expected_permanent_impact_dollars=e_perm,
        expected_temporary_impact_dollars=e_temp,
        expected_total_shortfall_dollars=e_tot,
        expected_total_shortfall_bps=shortfall_bps,
        shortfall_variance_dollars_sq=v_short,
        shortfall_std_dev_dollars=std_dev_shortfall,
        trajectory_schedule=schedule_dicts,
        summary=summary,
    )

    return receipt
