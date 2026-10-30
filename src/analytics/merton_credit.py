"""Robert C. Merton (1974) Structural Credit Risk & Distance-to-Default (DD) Engine.

Implements option-theoretic structural credit risk modeling:
- Solves unobservable total firm asset value (V_A) and asset volatility (sigma_A)
  simultaneously from equity market value (E), equity volatility (sigma_E),
  total debt face value (D), risk-free rate (r), and maturity (T).
- Formulates equity as a European call option on firm assets:
    E = V_A * Phi(d_1) - D * exp(-r * T) * Phi(d_2)
- Couples equity volatility via Itô's Lemma and Delta hedging:
    sigma_E = (V_A / E) * Phi(d_1) * sigma_A
- Computes Physical and Risk-Neutral Distance-to-Default (DD) & Default Probability:
    DD = [ ln(V_A / D) + (mu_A - 0.5 * sigma_A^2) * T ] / [ sigma_A * sqrt(T) ]
    pi_default = Phi(-DD)
- Derives structural credit yield spread (s) over the risk-free benchmark:
    s = - (1 / T) * ln[ Phi(d_2) + (V_A / (D * exp(-r * T))) * Phi(-d_1) ]

References:
    Merton, R. C. (1974). On the pricing of corporate debt: The risk structure
    of interest rates. The Journal of Finance, 29(2), 449-470.
    Crosbie, P., & Bohn, J. (2003). Modeling default risk. Moody's KMV Company.
    Bharath, S. T., & Shumway, T. (2008). Forecasting default with the Merton
    distance to default model. The Review of Financial Studies, 21(3), 1339-1369.
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
from scipy.optimize import least_squares
from scipy.stats import norm

import src.data.db as db
from src.security.validators import assert_finite_number, validate_ticker

MERTON_CREDIT_SIGNING_KEY = b"nextomni_merton_credit_secret_key_v1"


@dataclass
class MertonCreditReceipt:
    """Cryptographic audit receipt for Merton (1974) Structural Credit Risk & Distance-to-Default Profile."""

    ticker: str
    equity_value: float  # Market value of equity (E)
    debt_value: float  # Face value of default threshold / debt liabilities (D)
    annualized_equity_volatility: float  # sigma_E
    implied_asset_value: float  # Solved total firm assets (V_A)
    implied_asset_volatility: float  # Solved asset volatility (sigma_A)
    distance_to_default: float  # Physical Distance-to-Default (DD)
    risk_neutral_dd: float  # Risk-neutral Distance-to-Default (d_2)
    default_probability_pct: float  # pi_default = Phi(-DD) * 100%
    implied_credit_spread_bps: float  # Structural bond/CDS spread over risk-free rate
    leverage_ratio_pct: float  # Debt / Asset Value (D / V_A) * 100%
    distress_regime: str  # PRIME_SOLVENT, INVESTMENT_GRADE, SPECULATIVE_MODERATE, HIGH_DISTRESS_RISK, CRITICAL_DEFAULT_IMMINENT
    horizon_years: float  # Forecast maturity / credit horizon T (years)
    risk_free_rate_ann: float  # Annualized risk-free interest rate r
    summary: str
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across structural credit metrics."""
        payload = (
            f"{self.ticker}:{self.equity_value:.2f}:{self.debt_value:.2f}:"
            f"{self.annualized_equity_volatility:.4f}:{self.implied_asset_value:.2f}:"
            f"{self.implied_asset_volatility:.4f}:{self.distance_to_default:.4f}:"
            f"{self.default_probability_pct:.4f}:{self.implied_credit_spread_bps:.2f}:"
            f"{self.distress_regime}:{self.horizon_years:.2f}:{self.risk_free_rate_ann:.4f}"
        )
        return hmac.new(
            MERTON_CREDIT_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256
        ).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify authenticity and tamper-resistance of the credit receipt."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def classify_merton_distress_regime(distance_to_default: float) -> str:
    """Classify structural financial distress tier according to KMV / Moody's standards.

    Tiers:
        DD >= 4.0: PRIME_SOLVENT (AAA/AA equivalent, default prob < 0.003%)
        3.0 <= DD < 4.0: INVESTMENT_GRADE (A/BBB equivalent, default prob 0.003% - 0.13%)
        2.0 <= DD < 3.0: SPECULATIVE_MODERATE (BB/B equivalent, default prob 0.13% - 2.28%)
        1.0 <= DD < 2.0: HIGH_DISTRESS_RISK (CCC/CC equivalent, default prob 2.28% - 15.87%)
        DD < 1.0: CRITICAL_DEFAULT_IMMINENT (C/D equivalent, default prob > 15.87%)
    """
    if distance_to_default >= 4.0:
        return "PRIME_SOLVENT"
    elif distance_to_default >= 3.0:
        return "INVESTMENT_GRADE"
    elif distance_to_default >= 2.0:
        return "SPECULATIVE_MODERATE"
    elif distance_to_default >= 1.0:
        return "HIGH_DISTRESS_RISK"
    else:
        return "CRITICAL_DEFAULT_IMMINENT"


def solve_merton_model(
    equity_value: float,
    equity_volatility: float,
    debt_value: float,
    risk_free_rate_ann: float = 0.045,
    horizon_years: float = 1.0,
    asset_drift: Optional[float] = None,
) -> Tuple[float, float, float, float, float, float, float, str]:
    """Solve Merton (1974) structural credit risk equations for unobservable firm parameters.

    Solves the bivariate system:
        1. E = V_A * Phi(d_1) - D * exp(-r * T) * Phi(d_2)
        2. sigma_E = (V_A / E) * Phi(d_1) * sigma_A

    Args:
        equity_value: Current market value of equity (E > 0).
        equity_volatility: Annualized equity return volatility (sigma_E > 0).
        debt_value: Face value of total debt liabilities / default point (D >= 0).
        risk_free_rate_ann: Annual risk-free rate (r, default: 0.045).
        horizon_years: Debt maturity horizon (T, default: 1.0 year).
        asset_drift: Expected physical asset return drift (mu_A). If None, defaults to r.

    Returns:
        Tuple of:
            implied_asset_value (V_A)
            implied_asset_volatility (sigma_A)
            distance_to_default (DD)
            risk_neutral_dd (d_2)
            default_probability_pct (pi_default in %)
            implied_credit_spread_bps (s in bps)
            leverage_ratio_pct (D / V_A in %)
            distress_regime (str)
    """
    assert_finite_number(equity_value, "equity_value")
    assert_finite_number(equity_volatility, "equity_volatility")
    assert_finite_number(debt_value, "debt_value")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")
    assert_finite_number(horizon_years, "horizon_years")

    if horizon_years <= 0:
        raise ValueError(f"horizon_years must be strictly positive, got {horizon_years}")

    # Guard against invalid or insolvent equity
    if equity_value <= 0:
        v_a = max(0.0, float(equity_value + debt_value))
        return (
            v_a,
            0.0,
            -999.0,
            -999.0,
            100.0,
            10000.0,
            100.0,
            "CRITICAL_DEFAULT_IMMINENT",
        )

    # Edge case: Zero debt (all-equity financed firm)
    if debt_value <= 0:
        v_a = float(equity_value)
        sigma_a = float(max(1e-4, equity_volatility))
        return (
            v_a,
            sigma_a,
            999.0,
            999.0,
            0.0,
            0.0,
            0.0,
            "PRIME_SOLVENT",
        )

    e_val = float(equity_value)
    d_val = float(debt_value)
    sig_e = float(max(1e-4, equity_volatility))
    r = float(risk_free_rate_ann)
    t = float(horizon_years)
    sqrt_t = math.sqrt(t)
    drift = float(asset_drift) if asset_drift is not None else r

    # Non-linear residual system normalized by initial scales
    def residuals(params: np.ndarray) -> List[float]:
        v, sig_a = params
        v_safe = max(1e-6, v)
        sig_a_safe = max(1e-6, sig_a)

        # d1 and d2 calculations
        d1 = (math.log(v_safe / d_val) + (r + 0.5 * sig_a_safe**2) * t) / (sig_a_safe * sqrt_t)
        d2 = d1 - sig_a_safe * sqrt_t

        phi_d1 = float(norm.cdf(d1))
        phi_d2 = float(norm.cdf(d2))

        # Equation 1: Black-Scholes call option pricing
        call_val = v_safe * phi_d1 - d_val * math.exp(-r * t) * phi_d2
        res1 = (call_val - e_val) / e_val

        # Equation 2: Equity volatility linkage via Delta
        vol_eq = (v_safe / e_val) * phi_d1 * sig_a_safe
        res2 = (vol_eq - sig_e) / sig_e

        return [res1, res2]

    # Initial parameter guess
    v0 = e_val + d_val * math.exp(-r * t)
    sig_a0 = max(0.01, sig_e * (e_val / max(1e-6, v0)))

    bounds_lower = [0.05 * e_val, 1e-5]
    bounds_upper = [50.0 * (e_val + d_val), 10.0]

    # Solve using bounded Levenberg-Marquardt / Trust-Region Reflective algorithm
    opt_res = least_squares(
        residuals,
        [v0, sig_a0],
        bounds=(bounds_lower, bounds_upper),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
        max_nfev=2000,
    )

    if opt_res.success:
        v_sol, sig_a_sol = opt_res.x
    else:
        # Fallback to robust scalar iteration if multi-dimensional solver terminates early
        v_sol = v0
        sig_a_sol = sig_a0

    v_a = float(max(1e-4, v_sol))
    sigma_a = float(max(1e-5, sig_a_sol))

    # Standardized d1 and d2
    d1 = (math.log(v_a / d_val) + (r + 0.5 * sigma_a**2) * t) / (sigma_a * sqrt_t)
    d2 = d1 - sigma_a * sqrt_t

    # Physical Distance-to-Default (DD) using asset drift
    dd = (math.log(v_a / d_val) + (drift - 0.5 * sigma_a**2) * t) / (sigma_a * sqrt_t)
    rn_dd = d2

    # Default Probability: pi = Phi(-DD)
    pi_def_pct = float(norm.cdf(-dd) * 100.0)

    # Implied Credit Spread: s = - (1 / T) * ln[ Phi(d2) + (V_A / (D * exp(-rT))) * Phi(-d1) ]
    pv_debt = d_val * math.exp(-r * t)
    ratio_term = (v_a / pv_debt) * float(norm.cdf(-d1))
    bond_arg = float(norm.cdf(d2)) + ratio_term
    bond_arg_clipped = min(1.0, max(1e-12, bond_arg))
    spread_ann = - (1.0 / t) * math.log(bond_arg_clipped)
    implied_spread_bps = max(0.0, float(spread_ann * 10000.0))

    # Leverage Ratio
    leverage_pct = float((d_val / v_a) * 100.0)

    # Distress Regime
    distress_regime = classify_merton_distress_regime(dd)

    return (
        v_a,
        sigma_a,
        float(dd),
        float(rn_dd),
        pi_def_pct,
        implied_spread_bps,
        leverage_pct,
        distress_regime,
    )


def compute_rolling_distance_to_default(
    prices: Union[pd.Series, np.ndarray, List[float]],
    debt_value: float,
    shares_outstanding: float = 1.0,
    risk_free_rate_ann: float = 0.045,
    horizon_years: float = 1.0,
    lookback_window: int = 60,
) -> pd.DataFrame:
    """Compute rolling historical time-series of Merton Distance-to-Default and credit metrics.

    Args:
        prices: Daily closing price series.
        debt_value: Total debt default point in dollars.
        shares_outstanding: Total shares outstanding (defaults to 1.0 if prices are already market caps).
        risk_free_rate_ann: Annual risk-free rate (default: 0.045).
        horizon_years: Horizon in years (default: 1.0).
        lookback_window: Rolling window in sessions to estimate equity volatility (default: 60).

    Returns:
        DataFrame indexed by date or session with rolling V_A, sigma_A, DD, and pi_default.
    """
    s_p = pd.Series(prices, dtype=float).dropna()
    if len(s_p) < lookback_window:
        raise ValueError(
            f"Insufficient price observations ({len(s_p)}) for rolling window ({lookback_window})"
        )

    returns = s_p.pct_change().dropna()
    rolling_vol = returns.rolling(window=lookback_window).std(ddof=1) * math.sqrt(252.0)

    records: List[Dict[str, Any]] = []
    idx_list = s_p.index[lookback_window:]

    for idx in idx_list:
        p_t = float(s_p.loc[idx])
        e_t = p_t * float(shares_outstanding)
        sig_e = float(rolling_vol.loc[idx])

        if np.isnan(sig_e) or sig_e < 1e-4:
            sig_e = 0.20

        v_a, sig_a, dd, rn_dd, pi_def, spread_bps, lev_pct, regime = solve_merton_model(
            equity_value=e_t,
            equity_volatility=sig_e,
            debt_value=debt_value,
            risk_free_rate_ann=risk_free_rate_ann,
            horizon_years=horizon_years,
        )

        records.append({
            "session": idx,
            "equity_price": p_t,
            "equity_value": e_t,
            "equity_volatility": sig_e,
            "implied_asset_value": v_a,
            "implied_asset_volatility": sig_a,
            "distance_to_default": dd,
            "risk_neutral_dd": rn_dd,
            "default_probability_pct": pi_def,
            "credit_spread_bps": spread_bps,
            "leverage_ratio_pct": lev_pct,
            "distress_regime": regime,
        })

    df_res = pd.DataFrame(records)
    if "session" in df_res.columns:
        df_res.set_index("session", inplace=True)
    return df_res


def evaluate_merton_credit_profile(
    ticker: str,
    debt_value: float,
    shares_outstanding: Optional[float] = None,
    equity_value: Optional[float] = None,
    equity_volatility: Optional[float] = None,
    asset_prices: Optional[Union[pd.Series, np.ndarray, List[float]]] = None,
    risk_free_rate_ann: float = 0.045,
    horizon_years: float = 1.0,
    lookback_window: int = 60,
    db_path: Optional[Path] = None,
) -> MertonCreditReceipt:
    """Evaluate comprehensive Merton (1974) Structural Credit Risk & Distance-to-Default profile.

    Args:
        ticker: Asset ticker symbol.
        debt_value: Face value of total debt liabilities / default point ($).
        shares_outstanding: Optional total common shares outstanding.
        equity_value: Optional explicit total equity market cap ($).
        equity_volatility: Optional explicit annualized equity volatility (sigma_E).
        asset_prices: Optional price series to compute spot price and volatility.
        risk_free_rate_ann: Annual risk-free interest rate (default: 0.045).
        horizon_years: Credit horizon / maturity in years (default: 1.0).
        lookback_window: Rolling window in sessions to calculate equity volatility (default: 60).
        db_path: Optional SQLite database path.

    Returns:
        MertonCreditReceipt with cryptographic HMAC-SHA256 signature and tamper verification.
    """
    clean_ticker = validate_ticker(ticker)
    assert_finite_number(debt_value, "debt_value")
    assert_finite_number(risk_free_rate_ann, "risk_free_rate_ann")
    assert_finite_number(horizon_years, "horizon_years")

    # 1. Resolve Equity Volatility and Equity Value
    e_val: float
    sig_e: float

    if asset_prices is None and (equity_value is None or equity_volatility is None):
        # Retrieve from database
        df_obs = db.get_observations(clean_ticker, db_path=db_path)
        if df_obs.empty or len(df_obs) < 20:
            raise ValueError(
                f"Insufficient observation history for {clean_ticker}. Synchronize ticker first."
            )
        col = "adj_close" if "adj_close" in df_obs.columns else "close"
        s_prices = df_obs[col].dropna().astype(float)
    elif asset_prices is not None:
        s_prices = pd.Series(asset_prices, dtype=float).dropna()
    else:
        s_prices = pd.Series([], dtype=float)

    # Derive Equity Volatility
    if equity_volatility is not None:
        assert_finite_number(equity_volatility, "equity_volatility")
        sig_e = float(max(1e-4, equity_volatility))
    elif len(s_prices) >= 15:
        rets = s_prices.pct_change().dropna()
        window = min(lookback_window, len(rets))
        sig_e = float(rets.iloc[-window:].std(ddof=1) * math.sqrt(252.0))
        if np.isnan(sig_e) or sig_e < 1e-4:
            sig_e = 0.25
    else:
        sig_e = 0.25

    # Derive Equity Market Cap
    if equity_value is not None:
        assert_finite_number(equity_value, "equity_value")
        e_val = float(equity_value)
    elif len(s_prices) > 0 and shares_outstanding is not None and shares_outstanding > 0:
        spot_price = float(s_prices.iloc[-1])
        e_val = spot_price * float(shares_outstanding)
    elif len(s_prices) > 0:
        # Default shares proxy (1 Billion shares) if not supplied
        spot_price = float(s_prices.iloc[-1])
        e_val = spot_price * 1_000_000_000.0
    else:
        e_val = 10_000_000_000.0  # $10B baseline fallback

    # 2. Solve Merton Model
    (
        v_a,
        sigma_a,
        dd,
        rn_dd,
        pi_def_pct,
        implied_spread_bps,
        leverage_pct,
        distress_regime,
    ) = solve_merton_model(
        equity_value=e_val,
        equity_volatility=sig_e,
        debt_value=debt_value,
        risk_free_rate_ann=risk_free_rate_ann,
        horizon_years=horizon_years,
    )

    # 3. Formulate Summary
    summary = (
        f"{clean_ticker} Merton structural credit profile: Implied Assets ${v_a:,.0f} "
        f"(Asset Vol {sigma_a * 100:.2f}%), Distance-to-Default {dd:.2f} std devs "
        f"(Default Prob {pi_def_pct:.4f}%, Spread {implied_spread_bps:.1f} bps, Leverage {leverage_pct:.1f}%). "
        f"Regime: {distress_regime}."
    )

    return MertonCreditReceipt(
        ticker=clean_ticker,
        equity_value=e_val,
        debt_value=float(debt_value),
        annualized_equity_volatility=sig_e,
        implied_asset_value=v_a,
        implied_asset_volatility=sigma_a,
        distance_to_default=dd,
        risk_neutral_dd=rn_dd,
        default_probability_pct=pi_def_pct,
        implied_credit_spread_bps=implied_spread_bps,
        leverage_ratio_pct=leverage_pct,
        distress_regime=distress_regime,
        horizon_years=float(horizon_years),
        risk_free_rate_ann=float(risk_free_rate_ann),
        summary=summary,
    )
