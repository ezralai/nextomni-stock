"""Quantitative tail-risk analytics for NextOmni Stock.

Implements Value-at-Risk (Parametric Gaussian, Cornish-Fisher expansion, and Empirical Historical)
and Conditional Value-at-Risk (Expected Shortfall / CVaR) across multi-horizon forecasts.
"""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PRICE_COLUMN


def _clean_returns_series(returns: Union[np.ndarray, pd.Series, List[float]]) -> np.ndarray:
    """Extract clean, non-null, finite numpy array of returns."""
    arr = np.asarray(returns, dtype=float)
    clean = arr[np.isfinite(arr)]
    return clean


def calculate_var_gaussian(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate parametric Gaussian Value-at-Risk (VaR).

    Expresses maximum expected percentage loss over holding period at the specified confidence.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence (e.g., 0.95 for 95% VaR).

    Returns:
        VaR expressed as a positive percentage (e.g. 0.025 for 2.5% loss).
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma <= 1e-9:
        return 0.0

    z = float(stats.norm.ppf(1.0 - confidence_level))
    var = -(mu + z * sigma)
    return round(float(max(var, 0.0)), 6)


def calculate_var_cornish_fisher(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Cornish-Fisher expansion VaR, adjusting for skewness and fat-tailed excess kurtosis.

    Accounts for empirical financial asset return asymmetry and tail risk.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence (e.g., 0.95 or 0.99).

    Returns:
        VaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 10:
        return calculate_var_gaussian(clean, confidence_level)

    mu = float(np.mean(clean))
    sigma = float(np.std(clean, ddof=1))
    if sigma <= 1e-9:
        return 0.0

    s = float(stats.skew(clean))
    k = float(stats.kurtosis(clean))  # Fisher excess kurtosis (normal = 0)

    z = float(stats.norm.ppf(1.0 - confidence_level))
    # Cornish-Fisher quantile polynomial
    z_cf = (
        z
        + (s / 6.0) * (z**2 - 1.0)
        + (k / 24.0) * (z**3 - 3.0 * z)
        - ((s**2) / 36.0) * (2.0 * z**3 - 5.0 * z)
    )

    var = -(mu + z_cf * sigma)
    return round(float(max(var, 0.0)), 6)


def calculate_var_historical(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate empirical Historical Value-at-Risk from the sample quantile.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence level.

    Returns:
        Historical VaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    alpha_pct = (1.0 - confidence_level) * 100.0
    cutoff = float(np.percentile(clean, alpha_pct))
    var = -cutoff
    return round(float(max(var, 0.0)), 6)


def calculate_cvar_historical(
    returns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Conditional Value-at-Risk (Expected Shortfall / CVaR).

    Measures the expected loss given that the loss exceeds the Historical VaR threshold.

    Args:
        returns: Array of historical returns.
        confidence_level: Risk confidence level.

    Returns:
        CVaR expressed as a positive percentage loss.
    """
    clean = _clean_returns_series(returns)
    if len(clean) < 5:
        return 0.0

    alpha_pct = (1.0 - confidence_level) * 100.0
    cutoff = float(np.percentile(clean, alpha_pct))
    tail_losses = clean[clean <= cutoff]

    if len(tail_losses) == 0:
        return calculate_var_historical(clean, confidence_level)

    cvar = -float(np.mean(tail_losses))
    return round(float(max(cvar, 0.0)), 6)


def compute_multi_horizon_risk_profile(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute comprehensive tail risk metrics across 1, 5, and 21 session horizons.

    Args:
        df: Historical market observations with adj_close prices.
        horizons: List of horizon session integers (defaults to [1, 5, 21]).

    Returns:
        Nested dict mapping horizon key (e.g. '1_SESSION') to risk metrics.
    """
    if horizons is None:
        horizons = [1, 5, 21]

    if df.empty or PRICE_COLUMN not in df.columns:
        return {}

    prices = df[PRICE_COLUMN].dropna().values
    if len(prices) < 15:
        return {}

    daily_returns = np.diff(prices) / prices[:-1]
    results: Dict[str, Dict[str, float]] = {}

    horizon_names = {1: "1_SESSION", 5: "5_SESSIONS", 21: "21_SESSIONS"}

    for h in horizons:
        h_key = horizon_names.get(h, f"{h}_SESSIONS")
        # Empirical k-step forward returns
        if len(prices) > h:
            h_returns = (prices[h:] - prices[:-h]) / prices[:-h]
        else:
            # Fallback square root scaling of daily returns: r_h ~ r_1 * sqrt(h)
            h_returns = daily_returns * np.sqrt(h)

        var_95_g = calculate_var_gaussian(h_returns, 0.95)
        var_95_cf = calculate_var_cornish_fisher(h_returns, 0.95)
        var_99_cf = calculate_var_cornish_fisher(h_returns, 0.99)
        var_95_hist = calculate_var_historical(h_returns, 0.95)
        cvar_95_hist = calculate_cvar_historical(h_returns, 0.95)
        cvar_99_hist = calculate_cvar_historical(h_returns, 0.99)

        results[h_key] = {
            "horizon_sessions": float(h),
            "var_95_gaussian_pct": round(var_95_g * 100.0, 3),
            "var_95_cornish_fisher_pct": round(var_95_cf * 100.0, 3),
            "var_99_cornish_fisher_pct": round(var_99_cf * 100.0, 3),
            "var_95_historical_pct": round(var_95_hist * 100.0, 3),
            "cvar_95_historical_pct": round(cvar_95_hist * 100.0, 3),
            "cvar_99_historical_pct": round(cvar_99_hist * 100.0, 3),
            "skewness": round(float(stats.skew(h_returns)), 3) if len(h_returns) > 3 else 0.0,
            "excess_kurtosis": round(float(stats.kurtosis(h_returns)), 3) if len(h_returns) > 3 else 0.0,
        }

    return results


def simulate_macro_stress_scenarios(
    df: pd.DataFrame,
    position_value: float = 10000.0,
    price_col: str = PRICE_COLUMN,
) -> List[Dict[str, Any]]:
    """Simulate impact of acute historical macroeconomic shock scenarios on current position.

    Scenarios modeled:
    1. 2022 Rates & Inflation Shock (-4.5% benchmark drop, 2.1x volatility multiplier)
    2. 2008 Lehman Brothers Insolvency (-8.5% benchmark drop, 3.5x volatility multiplier)
    3. 2020 COVID Liquidity Freeze (-12.0% benchmark drop, 4.2x volatility multiplier)
    4. 1987 Black Monday Tail Event (-20.5% benchmark drop, 5.0x volatility multiplier)

    Args:
        df: Historical observations DataFrame.
        position_value: Current capital exposed in dollars/currency.
        price_col: Closing price column name.

    Returns:
        List of scenario impact summaries including estimated loss percentage and dollars.
    """
    if len(df) < 5 or price_col not in df.columns:
        return []

    prices = df[price_col].dropna().values
    latest_price = float(prices[-1])
    daily_returns = np.diff(prices) / prices[:-1]
    ticker_ann_vol = float(np.std(daily_returns, ddof=1) * np.sqrt(252)) if len(daily_returns) > 1 else 0.20

    market_benchmark_vol = 0.16
    vol_beta = max(0.5, min(2.5, ticker_ann_vol / market_benchmark_vol))

    scenarios = [
        {
            "scenario_name": "2022 Rates & Inflation Shock",
            "benchmark_shock_pct": -4.5,
            "vol_multiplier": 2.1,
            "severity": "Moderate",
            "description": "Persistent rate hike pressure and aggressive multiple compression.",
        },
        {
            "scenario_name": "2008 Lehman Brothers Insolvency",
            "benchmark_shock_pct": -8.5,
            "vol_multiplier": 3.5,
            "severity": "Severe",
            "description": "Systemic financial counterparty credit crisis and collateral lockup.",
        },
        {
            "scenario_name": "2020 COVID Liquidity Freeze",
            "benchmark_shock_pct": -12.0,
            "vol_multiplier": 4.2,
            "severity": "Extreme",
            "description": "Rapid market-wide dash for cash and cross-asset liquidity dry-up.",
        },
        {
            "scenario_name": "1987 Black Monday Tail Event",
            "benchmark_shock_pct": -20.5,
            "vol_multiplier": 5.0,
            "severity": "Catastrophic",
            "description": "Cascading portfolio insurance algorithmic liquidation shock.",
        },
    ]

    results = []
    for sc in scenarios:
        shock = sc["benchmark_shock_pct"] * vol_beta
        loss_pct = round(abs(shock), 2)
        loss_dollars = round(position_value * (loss_pct / 100.0), 2)
        post_stress_price = round(latest_price * (1.0 - (loss_pct / 100.0)), 2)

        results.append(
            {
                "scenario_name": sc["scenario_name"],
                "severity": sc["severity"],
                "benchmark_shock_pct": sc["benchmark_shock_pct"],
                "vol_multiplier": sc["vol_multiplier"],
                "ticker_vol_beta": round(vol_beta, 2),
                "estimated_loss_pct": loss_pct,
                "estimated_loss_dollars": loss_dollars,
                "post_stress_price": post_stress_price,
                "description": sc["description"],
            }
        )

    return results


@dataclass
class PathRiskReceipt:
    """Diagnostic receipt from path-dependent drawdown and CDaR risk evaluation."""
    ticker: str
    sample_sessions: int
    current_price: float
    high_water_mark: float
    current_drawdown_pct: float
    max_drawdown_pct: float
    dar_95_pct: float
    cdar_95_pct: float
    cdar_99_pct: float
    ulcer_index_pct: float
    pain_index_pct: float
    max_underwater_sessions: int
    current_underwater_sessions: int
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def compute_drawdown_series(
    prices: Union[np.ndarray, pd.Series, List[float]],
) -> pd.Series:
    """Compute cumulative percentage drawdown series from historical price trajectory.

    Calculates running high-water mark M_t = max_{s <= t} P_s and percentage drawdown:
    D_t = (M_t - P_t) / M_t >= 0.

    Args:
        prices: Historical settlement price series.

    Returns:
        pd.Series of drawdowns strictly bounded in [0.0, 1.0].
    """
    s = pd.Series(prices, dtype=float).dropna()
    if s.empty:
        return pd.Series(dtype=float)
    clean = s[s > 0]
    if clean.empty:
        return pd.Series(0.0, index=s.index)
    cum_max = clean.cummax()
    dd = (cum_max - clean) / cum_max
    return dd.clip(lower=0.0, upper=1.0)


def calculate_dar(
    drawdowns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Drawdown-at-Risk (DaR_alpha).

    Represents the alpha-quantile of empirical drawdowns.

    Args:
        drawdowns: Array or series of percentage drawdowns in [0.0, 1.0].
        confidence_level: Risk confidence (e.g., 0.95 for 95% DaR).

    Returns:
        DaR expressed as a percentage (e.g., 14.5 for 14.5% drawdown).
    """
    arr = np.asarray(drawdowns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 5:
        return 0.0
    q = float(np.quantile(clean, confidence_level))
    return round(float(max(q * 100.0, 0.0)), 2)


def calculate_cdar(
    drawdowns: Union[np.ndarray, pd.Series, List[float]],
    confidence_level: float = 0.95,
) -> float:
    """Calculate Conditional Drawdown-at-Risk (CDaR_alpha, Chekhlov et al. 2005).

    Measures expected tail drawdown depth exceeding the alpha-quantile threshold:
    CDaR_alpha = E[D | D >= DaR_alpha].

    Args:
        drawdowns: Array or series of percentage drawdowns in [0.0, 1.0].
        confidence_level: Risk confidence (e.g., 0.95 for 95% CDaR).

    Returns:
        CDaR expressed as a positive percentage (e.g., 18.2 for 18.2% conditional drawdown).
    """
    arr = np.asarray(drawdowns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 5:
        return 0.0
    dar_threshold = float(np.quantile(clean, confidence_level))
    tail = clean[clean >= dar_threshold]
    if len(tail) == 0:
        return round(float(dar_threshold * 100.0), 2)
    cdar = float(np.mean(tail)) * 100.0
    return round(float(max(cdar, dar_threshold * 100.0)), 2)


def calculate_ulcer_index(
    drawdowns: Union[np.ndarray, pd.Series, List[float]],
) -> float:
    """Calculate Peter Martin's (1987) Ulcer Index (UI).

    Measures root-mean-square drawdown severity and duration:
    UI = sqrt(mean((D_t * 100)^2)).

    Args:
        drawdowns: Array or series of percentage drawdowns in [0.0, 1.0].

    Returns:
        Ulcer Index as a percentage.
    """
    arr = np.asarray(drawdowns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 5:
        return 0.0
    sq_dd = (clean * 100.0) ** 2
    ui = float(np.sqrt(np.mean(sq_dd)))
    return round(ui, 2)


def calculate_pain_index(
    drawdowns: Union[np.ndarray, pd.Series, List[float]],
) -> float:
    """Calculate Pain Index (mean absolute drawdown percentage across all sessions)."""
    arr = np.asarray(drawdowns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 5:
        return 0.0
    return round(float(np.mean(clean * 100.0)), 2)


def calculate_underwater_durations(
    drawdowns: Union[np.ndarray, pd.Series, List[float]],
) -> Tuple[int, int]:
    """Calculate maximum and current consecutive underwater trading sessions.

    Args:
        drawdowns: Array or series of percentage drawdowns in [0.0, 1.0].

    Returns:
        Tuple of (max_underwater_sessions, current_underwater_sessions).
    """
    arr = np.asarray(drawdowns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) == 0:
        return 0, 0

    max_duration = 0
    current_duration = 0
    for dd in clean:
        if dd > 1e-4:
            current_duration += 1
            if current_duration > max_duration:
                max_duration = current_duration
        else:
            current_duration = 0
    return max_duration, current_duration


def compute_path_risk_profile(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    price_col: str = PRICE_COLUMN,
) -> PathRiskReceipt:
    """Compute comprehensive path-dependent risk, CDaR, and underwater duration profile.

    Args:
        df: Historical market observations DataFrame.
        ticker: Asset ticker symbol.
        price_col: Column name containing settlement prices (default: adj_close).

    Returns:
        PathRiskReceipt with comprehensive path-dependent risk metrics.
    """
    if df.empty or len(df) < 5:
        return PathRiskReceipt(
            ticker=ticker,
            sample_sessions=len(df),
            current_price=0.0,
            high_water_mark=0.0,
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            dar_95_pct=0.0,
            cdar_95_pct=0.0,
            cdar_99_pct=0.0,
            ulcer_index_pct=0.0,
            pain_index_pct=0.0,
            max_underwater_sessions=0,
            current_underwater_sessions=0,
            summary=f"Insufficient history ({len(df)} sessions) for path-dependent risk analysis.",
        )

    clean_df = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    p_col = price_col if price_col in clean_df.columns else "close"
    if p_col not in clean_df.columns:
        p_col = clean_df.columns[1]

    prices = pd.to_numeric(clean_df[p_col], errors="coerce").dropna().values
    n = len(prices)
    if n < 5:
        return PathRiskReceipt(
            ticker=ticker,
            sample_sessions=n,
            current_price=float(prices[-1]) if n > 0 else 0.0,
            high_water_mark=float(np.max(prices)) if n > 0 else 0.0,
            current_drawdown_pct=0.0,
            max_drawdown_pct=0.0,
            dar_95_pct=0.0,
            cdar_95_pct=0.0,
            cdar_99_pct=0.0,
            ulcer_index_pct=0.0,
            pain_index_pct=0.0,
            max_underwater_sessions=0,
            current_underwater_sessions=0,
            summary=f"Insufficient clean price records ({n} sessions) for path risk.",
        )

    drawdowns = compute_drawdown_series(prices)
    current_p = float(prices[-1])
    hwm = float(np.max(prices))
    curr_dd = float(drawdowns.iloc[-1]) * 100.0 if len(drawdowns) > 0 else 0.0
    max_dd = float(drawdowns.max()) * 100.0 if len(drawdowns) > 0 else 0.0

    dar_95 = calculate_dar(drawdowns, confidence_level=0.95)
    cdar_95 = calculate_cdar(drawdowns, confidence_level=0.95)
    cdar_99 = calculate_cdar(drawdowns, confidence_level=0.99)
    ulcer = calculate_ulcer_index(drawdowns)
    pain = calculate_pain_index(drawdowns)
    max_uw, curr_uw = calculate_underwater_durations(drawdowns)

    summary = (
        f"Path Risk ({ticker}, n={n} sessions): Max Drawdown = {max_dd:.2f}%, "
        f"95% CDaR = {cdar_95:.2f}%, 99% CDaR = {cdar_99:.2f}%, Ulcer Index = {ulcer:.2f}%, "
        f"Max Underwater = {max_uw} sessions, Current Underwater = {curr_uw} sessions ({curr_dd:.2f}% off HWM {hwm:.2f})."
    )

    return PathRiskReceipt(
        ticker=ticker,
        sample_sessions=n,
        current_price=round(current_p, 2),
        high_water_mark=round(hwm, 2),
        current_drawdown_pct=round(curr_dd, 2),
        max_drawdown_pct=round(max_dd, 2),
        dar_95_pct=dar_95,
        cdar_95_pct=cdar_95,
        cdar_99_pct=cdar_99,
        ulcer_index_pct=ulcer,
        pain_index_pct=pain,
        max_underwater_sessions=max_uw,
        current_underwater_sessions=curr_uw,
        summary=summary,
    )

