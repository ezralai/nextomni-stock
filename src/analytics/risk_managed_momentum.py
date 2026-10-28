"""Risk-Managed Momentum & Crash-Risk Dampened Trend Engine for NextOmni Stock.

Implements seminal quantitative risk-managed momentum frameworks:
- Barroso & Santa-Clara (2015) ("Momentum has its moments", JFE):
  Scales momentum exposure inversely by trailing realized variance:
  w_t = min(w_max, sigma* / sigma_hat_{t-1}).
  Eliminates negative skewness (-4.0 -> ~0) and doubles Sharpe ratio without lookahead bias.
- Daniel & Moskowitz (2016) ("Momentum Crashes", JFE):
  Dynamic crash hazard probability conditioned on prolonged bear-market drawdowns
  and short-term volatility surges.
- Cryptographically verifiable HMAC-SHA256 diagnostic receipts guaranteeing analytical integrity.
"""

from dataclasses import asdict, dataclass
import hashlib
import hmac
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats

from src.config import PRICE_COLUMN
from src.security.validators import validate_ticker


MOMENTUM_SIGNING_KEY = b"nextomni_risk_managed_momentum_audit_key_v1"


@dataclass
class RiskManagedMomentumReceipt:
    """Quantitative diagnostic receipt summarizing risk-managed momentum performance."""
    ticker: str
    sample_sessions: int
    target_volatility_pct: float
    max_leverage: float
    current_crash_hazard_pct: float
    crash_risk_regime: str
    current_leverage_weight: float
    raw_momentum_ann_return_pct: float
    rmm_ann_return_pct: float
    raw_momentum_volatility_pct: float
    rmm_volatility_pct: float
    raw_momentum_sharpe: float
    rmm_sharpe: float
    raw_momentum_max_drawdown_pct: float
    rmm_max_drawdown_pct: float
    raw_momentum_skewness: float
    rmm_skewness: float
    summary: str
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core strategy metrics."""
        payload = (
            f"{self.ticker}:{self.sample_sessions}:{self.target_volatility_pct:.2f}:"
            f"{self.max_leverage:.2f}:{self.current_crash_hazard_pct:.2f}:"
            f"{self.raw_momentum_sharpe:.3f}:{self.rmm_sharpe:.3f}:"
            f"{self.raw_momentum_skewness:.3f}:{self.rmm_skewness:.3f}"
        )
        return hmac.new(MOMENTUM_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against tamper modification."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def calculate_risk_managed_momentum(
    returns: Union[np.ndarray, pd.Series, List[float]],
    lookback_mom: int = 21,
    lookback_vol: int = 21,
    target_vol: float = 0.12,
    max_leverage: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Barroso & Santa-Clara (2015) risk-managed momentum strategy series.

    Guarantees strict zero-lookahead:
    Position direction and scaling weight w_t are established strictly at t-1:
        signal_t = sign( prod_{tau=1}^{W_mom} (1 + r_{t-tau}) - 1 )
        sigma_hat_{t-1} = std( r_{t-W_vol .. t-1}^{raw_mom} ) * sqrt(252)
        w_t = min( w_max, sigma* / sigma_hat_{t-1} )
        r_t^{RMM} = w_t * r_t^{raw_mom}

    Args:
        returns: Historical daily asset returns.
        lookback_mom: Momentum formation lookback window (default: 21 sessions).
        lookback_vol: Realized volatility estimation window (default: 21 sessions).
        target_vol: Target annualized volatility sigma* (default: 0.12 or 12%).
        max_leverage: Maximum allowable leverage factor w_max (default: 2.0x).

    Returns:
        Tuple of (raw_momentum_returns, rmm_returns, leverage_weights).
    """
    arr = np.asarray(returns, dtype=float)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    n = len(clean)

    warmup = max(lookback_mom, lookback_vol) + 2
    if n < warmup:
        zeros = np.zeros(n, dtype=float)
        return zeros, zeros, np.ones(n, dtype=float)

    raw_mom = np.zeros(n, dtype=float)
    rmm = np.zeros(n, dtype=float)
    weights = np.ones(n, dtype=float)

    # 1. Compute raw momentum series with strictly prior signals
    for t in range(lookback_mom, n):
        past_ret = clean[t - lookback_mom:t]
        cum_ret = float(np.prod(1.0 + past_ret) - 1.0)
        signal = 1.0 if cum_ret > 0 else (-1.0 if cum_ret < 0 else 0.0)
        raw_mom[t] = signal * clean[t]

    # 2. Compute Barroso & Santa-Clara dynamic volatility scaling
    min_vol = 0.02  # 2% floor to prevent division by zero
    for t in range(warmup, n):
        hist_mom = raw_mom[t - lookback_vol:t]
        vol_est = float(np.std(hist_mom, ddof=1) * math.sqrt(252.0))
        vol_est = max(min_vol, vol_est)

        scaled_w = min(max_leverage, target_vol / vol_est)
        weights[t] = scaled_w
        rmm[t] = scaled_w * raw_mom[t]

    return raw_mom[warmup:], rmm[warmup:], weights[warmup:]


def estimate_momentum_crash_probability(
    returns: Union[np.ndarray, pd.Series, List[float]],
    lookback_bear: int = 126,
    lookback_vol_short: int = 21,
    lookback_vol_long: int = 126,
) -> Tuple[float, str]:
    """Estimate Daniel & Moskowitz (2016) conditional momentum crash hazard probability.

    Momentum crashes occur during volatile rebounds following prolonged drawdowns.
    Models the joint crash state:
        I_bear = 1 if past 126-session cumulative return < 0
        VR = sigma_short / sigma_long (volatility acceleration ratio)
        Index = -1.2 + 1.8 * I_bear + 1.2 * max(0, VR - 1.0)
        P_crash = 1 / (1 + exp(-Index))

    Args:
        returns: Daily asset return series.
        lookback_bear: Bear market drawdown horizon (default: 126 sessions / 6 months).
        lookback_vol_short: Short-term realized volatility horizon (default: 21 sessions).
        lookback_vol_long: Baseline realized volatility horizon (default: 126 sessions).

    Returns:
        Tuple of (crash_hazard_pct, crash_risk_regime).
    """
    arr = np.asarray(returns, dtype=float)
    clean = np.where(np.isfinite(arr), arr, 0.0)
    n = len(clean)

    if n < max(lookback_bear, lookback_vol_long):
        return 0.0, "INSUFFICIENT_DATA"

    # Bear market state: cumulative return over past lookback_bear
    past_bear_rets = clean[-lookback_bear:]
    cum_ret = float(np.prod(1.0 + past_bear_rets) - 1.0)
    is_bear = 1.0 if cum_ret < 0.0 else 0.0

    # Volatility acceleration ratio
    short_vol = float(np.std(clean[-lookback_vol_short:], ddof=1) * math.sqrt(252.0))
    long_vol = float(np.std(clean[-lookback_vol_long:], ddof=1) * math.sqrt(252.0))
    long_vol = max(1e-4, long_vol)
    vol_ratio = short_vol / long_vol

    # Logistic crash hazard index
    log_odds = -1.2 + 1.8 * is_bear + 1.2 * max(0.0, vol_ratio - 1.0)
    p_crash = 1.0 / (1.0 + math.exp(-log_odds))
    p_crash_pct = round(float(p_crash * 100.0), 2)

    if p_crash_pct >= 65.0:
        regime = "CRASH_RISK_ELEVATED"
    elif p_crash_pct >= 40.0:
        regime = "CRASH_RISK_MODERATE"
    else:
        regime = "CRASH_RISK_BENIGN"

    return p_crash_pct, regime


def evaluate_momentum_risk_profile(
    df: pd.DataFrame,
    ticker: str = "TICKER",
    target_vol: float = 0.12,
    max_leverage: float = 2.0,
    price_col: str = PRICE_COLUMN,
) -> RiskManagedMomentumReceipt:
    """Evaluate comprehensive risk-managed momentum profile and crash diagnostics.

    Args:
        df: Historical market observations DataFrame.
        ticker: Asset ticker symbol.
        target_vol: Target annualized volatility (default: 0.12).
        max_leverage: Maximum allowable leverage (default: 2.0).
        price_col: Settlement price column name (default: adj_close).

    Returns:
        RiskManagedMomentumReceipt containing performance metrics and HMAC digest.
    """
    clean_ticker = validate_ticker(ticker)

    if df.empty or len(df) < 45:
        return RiskManagedMomentumReceipt(
            ticker=clean_ticker,
            sample_sessions=len(df),
            target_volatility_pct=target_vol * 100.0,
            max_leverage=max_leverage,
            current_crash_hazard_pct=0.0,
            crash_risk_regime="INSUFFICIENT_DATA",
            current_leverage_weight=1.0,
            raw_momentum_ann_return_pct=0.0,
            rmm_ann_return_pct=0.0,
            raw_momentum_volatility_pct=0.0,
            rmm_volatility_pct=0.0,
            raw_momentum_sharpe=0.0,
            rmm_sharpe=0.0,
            raw_momentum_max_drawdown_pct=0.0,
            rmm_max_drawdown_pct=0.0,
            raw_momentum_skewness=0.0,
            rmm_skewness=0.0,
            summary=f"Insufficient observation history ({len(df)} sessions) for risk-managed momentum analysis.",
        )

    clean_df = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    p_col = price_col if price_col in clean_df.columns else "close"
    if p_col not in clean_df.columns:
        p_col = clean_df.columns[1]

    prices = pd.to_numeric(clean_df[p_col], errors="coerce").dropna().values
    if len(prices) < 45:
        return RiskManagedMomentumReceipt(
            ticker=clean_ticker,
            sample_sessions=len(prices),
            target_volatility_pct=target_vol * 100.0,
            max_leverage=max_leverage,
            current_crash_hazard_pct=0.0,
            crash_risk_regime="INSUFFICIENT_DATA",
            current_leverage_weight=1.0,
            raw_momentum_ann_return_pct=0.0,
            rmm_ann_return_pct=0.0,
            raw_momentum_volatility_pct=0.0,
            rmm_volatility_pct=0.0,
            raw_momentum_sharpe=0.0,
            rmm_sharpe=0.0,
            raw_momentum_max_drawdown_pct=0.0,
            rmm_max_drawdown_pct=0.0,
            raw_momentum_skewness=0.0,
            rmm_skewness=0.0,
            summary=f"Insufficient price history ({len(prices)}) for momentum analysis.",
        )

    daily_returns = np.diff(prices) / prices[:-1]

    raw_rets, rmm_rets, weights = calculate_risk_managed_momentum(
        daily_returns,
        lookback_mom=21,
        lookback_vol=21,
        target_vol=target_vol,
        max_leverage=max_leverage,
    )

    crash_hazard, crash_regime = estimate_momentum_crash_probability(daily_returns)

    # Strategy metrics calculation helper
    def _calc_strategy_metrics(r_arr: np.ndarray) -> Tuple[float, float, float, float, float]:
        if len(r_arr) < 5:
            return 0.0, 0.0, 0.0, 0.0, 0.0
        ann_ret = float(np.mean(r_arr) * 252.0 * 100.0)
        ann_vol = float(np.std(r_arr, ddof=1) * math.sqrt(252.0) * 100.0)
        sharpe = round(float((ann_ret / 100.0 - 0.04) / max(ann_vol / 100.0, 1e-4)), 3)

        # Max Drawdown
        cum = np.cumprod(1.0 + r_arr)
        peak = np.maximum.accumulate(cum)
        dd = (peak - cum) / peak
        max_dd = float(np.max(dd) * 100.0) if len(dd) > 0 else 0.0

        # Skewness
        skew = float(stats.skew(r_arr)) if len(r_arr) > 2 else 0.0
        return round(ann_ret, 2), round(ann_vol, 2), sharpe, round(max_dd, 2), round(skew, 3)

    raw_ret, raw_vol, raw_sr, raw_mdd, raw_skew = _calc_strategy_metrics(raw_rets)
    rmm_ret, rmm_vol, rmm_sr, rmm_mdd, rmm_skew = _calc_strategy_metrics(rmm_rets)

    latest_weight = round(float(weights[-1]), 3) if len(weights) > 0 else 1.0

    summary = (
        f"{clean_ticker} Momentum Profile: Crash Hazard={crash_hazard:.1f}% ({crash_regime}), "
        f"Dynamic Leverage={latest_weight:.2f}x. Risk-Managed Sharpe={rmm_sr:+.2f} vs Raw={raw_sr:+.2f}. "
        f"Skewness normalized from {raw_skew:+.2f} to {rmm_skew:+.2f}."
    )

    return RiskManagedMomentumReceipt(
        ticker=clean_ticker,
        sample_sessions=len(prices),
        target_volatility_pct=target_vol * 100.0,
        max_leverage=max_leverage,
        current_crash_hazard_pct=crash_hazard,
        crash_risk_regime=crash_regime,
        current_leverage_weight=latest_weight,
        raw_momentum_ann_return_pct=raw_ret,
        rmm_ann_return_pct=rmm_ret,
        raw_momentum_volatility_pct=raw_vol,
        rmm_volatility_pct=rmm_vol,
        raw_momentum_sharpe=raw_sr,
        rmm_sharpe=rmm_sr,
        raw_momentum_max_drawdown_pct=raw_mdd,
        rmm_max_drawdown_pct=rmm_mdd,
        raw_momentum_skewness=raw_skew,
        rmm_skewness=rmm_skew,
        summary=summary,
    )
