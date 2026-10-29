"""Online Kalman Filter Dynamic Hedge Ratio & Adaptive Spread Engine for NextOmni Stock.

Implements recursive state-space Bayesian filtering for statistical arbitrage pairs trading
and cointegrated asset monitoring. Recursively estimates time-varying intercept alpha_t
and dynamic hedge ratio beta_t with zero forward lookahead.

References:
    Chan, E. (2013). Algorithmic Trading: Winning Strategies and Their Rationale.
    John Wiley & Sons, Chapter 3: "Mean Reversion of Portfolios and Pairs".
"""

import hashlib
import hmac
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

KALMAN_SIGNING_KEY = b"nextomni_kalman_secret_audit_key_v1"


@dataclass
class KalmanSpreadReceipt:
    """Cryptographic audit receipt for online Kalman filter pairs estimation."""

    y_ticker: str
    x_ticker: str
    sample_sessions: int
    delta_process_noise: float
    latest_alpha: float
    latest_beta: float
    latest_spread: float
    latest_spread_std: float
    latest_zscore: float
    current_signal: str
    ols_static_beta: float
    beta_drift_pct: float
    mean_reversion_half_life: float
    spread_history: List[Dict[str, Any]]
    created_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    signature: str = ""

    def __post_init__(self) -> None:
        if not self.signature:
            self.signature = self._compute_signature()

    def _compute_signature(self) -> str:
        """Compute deterministic HMAC-SHA256 signature across core state estimates."""
        payload = (
            f"{self.y_ticker}:{self.x_ticker}:{self.sample_sessions}:"
            f"{self.delta_process_noise:.6e}:{self.latest_alpha:.4f}:{self.latest_beta:.4f}:"
            f"{self.latest_spread:.4f}:{self.latest_zscore:.4f}:{self.current_signal}"
        )
        return hmac.new(KALMAN_SIGNING_KEY, payload.encode("utf-8"), hashlib.sha256).hexdigest()

    def verify_integrity(self) -> bool:
        """Verify receipt provenance against unauthorized tampering."""
        expected = self._compute_signature()
        return hmac.compare_digest(self.signature, expected)

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


class OnlineKalmanFilterPairsTracker:
    """Recursive 2D Kalman Filter for dynamic hedge ratio and spread estimation.

    State Space Formulation:
        State vector: theta_t = [alpha_t, beta_t]^T
        State transition: theta_t = theta_{t-1} + w_t,  w_t ~ N(0, Q)
        Measurement:     y_t = H_t * theta_t + v_t,     v_t ~ N(0, R)
        Measurement matrix: H_t = [1.0, x_t]
    """

    def __init__(
        self,
        delta: float = 1e-4,
        measurement_noise_r: float = 1.0,
        initial_p_variance: float = 1.0,
    ) -> None:
        """Initialize the online Kalman filter.

        Args:
            delta: Process noise factor controlling rate of parameter evolution (default: 1e-4).
            measurement_noise_r: Measurement noise variance R (default: 1.0).
            initial_p_variance: Initial diagonal uncertainty on state covariance P_0.
        """
        self.delta = float(np.clip(delta, 1e-7, 1e-1))
        self.measurement_noise_r = float(max(1e-6, measurement_noise_r))

        # Process noise covariance Q: diag(delta / (1 - delta))
        q_diag = self.delta / (1.0 - self.delta)
        self.q_mat = np.eye(2, dtype=float) * q_diag

        # Initial state theta_0 = [0.0, 0.0]^T and covariance P_0
        self.theta = np.zeros(2, dtype=float)
        self.p_mat = np.eye(2, dtype=float) * initial_p_variance

    def update(self, y_t: float, x_t: float) -> Tuple[float, float, float, float]:
        """Execute one online prediction-update cycle for session t.

        Args:
            y_t: Price of asset Y at session t.
            x_t: Price of asset X at session t.

        Returns:
            Tuple of (alpha_t, beta_t, innovation_error_e_t, z_score_t).
        """
        # 1. State prediction: theta_{t|t-1} = theta_{t-1}
        theta_prior = self.theta.copy()
        # Covariance prediction: P_{t|t-1} = P_{t-1} + Q
        p_prior = self.p_mat + self.q_mat

        # 2. Measurement matrix H_t = [1.0, x_t]
        h_t = np.array([1.0, float(x_t)], dtype=float)

        # 3. Measurement prediction and innovation
        y_hat = float(np.dot(h_t, theta_prior))
        error_e = float(y_t - y_hat)

        # 4. Innovation variance: S_t = H_t * P_{t|t-1} * H_t^T + R
        s_t = float(np.dot(h_t, np.dot(p_prior, h_t)) + self.measurement_noise_r)
        s_t = max(1e-12, s_t)

        # 5. Kalman Gain: K_t = P_{t|t-1} * H_t^T / S_t
        k_t = np.dot(p_prior, h_t) / s_t

        # 6. State update: theta_{t|t} = theta_{t|t-1} + K_t * e_t
        self.theta = theta_prior + k_t * error_e

        # 7. Covariance update: P_{t|t} = (I - K_t * H_t) * P_{t|t-1}
        i_kh = np.eye(2, dtype=float) - np.outer(k_t, h_t)
        self.p_mat = np.dot(i_kh, p_prior)

        # 8. Normalized instantaneous z-score
        z_score = error_e / np.sqrt(s_t)

        return float(self.theta[0]), float(self.theta[1]), error_e, float(z_score)


def estimate_kalman_spread_series(
    y_prices: Union[pd.Series, np.ndarray, List[float]],
    x_prices: Union[pd.Series, np.ndarray, List[float]],
    delta: float = 1e-4,
) -> pd.DataFrame:
    """Run online Kalman filter over aligned time-series of paired assets.

    Args:
        y_prices: Historical closing price sequence for asset Y.
        x_prices: Historical closing price sequence for asset X.
        delta: Process noise factor.

    Returns:
        pd.DataFrame containing columns: [alpha, beta, spread, spread_std, zscore].
    """
    if isinstance(y_prices, pd.Series):
        y_vals = y_prices.values.astype(float)
        idx = y_prices.index
    else:
        y_vals = np.asarray(y_prices, dtype=float)
        idx = pd.RangeIndex(len(y_vals))

    if isinstance(x_prices, pd.Series):
        x_vals = x_prices.values.astype(float)
    else:
        x_vals = np.asarray(x_prices, dtype=float)

    n = min(len(y_vals), len(x_vals))
    if n == 0:
        return pd.DataFrame(columns=["alpha", "beta", "spread", "spread_std", "zscore"])

    tracker = OnlineKalmanFilterPairsTracker(delta=delta)

    alphas = np.zeros(n, dtype=float)
    betas = np.zeros(n, dtype=float)
    spreads = np.zeros(n, dtype=float)
    zscores = np.zeros(n, dtype=float)

    for t in range(n):
        a, b, e, z = tracker.update(y_vals[t], x_vals[t])
        alphas[t] = a
        betas[t] = b
        spreads[t] = e
        zscores[t] = z

    # Rolling standard deviation of spread for robust visualization
    spread_series = pd.Series(spreads, index=idx[:n])
    rolling_std = spread_series.rolling(window=20, min_periods=5).std().bfill().fillna(1.0)

    return pd.DataFrame({
        "alpha": alphas,
        "beta": betas,
        "spread": spreads,
        "spread_std": rolling_std.values,
        "zscore": zscores,
    }, index=idx[:n])


def compute_half_life(spread: pd.Series) -> float:
    """Calculate the Ornstein-Uhlenbeck mean-reversion half-life of a price spread.

    Formula:
        Delta s_t = lambda * s_{t-1} + eps_t
        Half-Life = -ln(2) / lambda

    Args:
        spread: Time-series of spread values.

    Returns:
        Half-life in trading sessions (clamped to [0.5, 252.0]).
    """
    s = spread.dropna().values.astype(float)
    if len(s) < 15:
        return 20.0

    delta_s = np.diff(s)
    s_lag = s[:-1]

    # OLS: delta_s = lambda * s_lag
    denom = np.dot(s_lag, s_lag)
    if denom <= 1e-12:
        return 20.0

    lambda_param = float(np.dot(s_lag, delta_s) / denom)
    if lambda_param >= 0:
        return 252.0  # Divergent or non-mean-reverting

    half_life = -np.log(2.0) / lambda_param
    return float(np.clip(half_life, 0.5, 252.0))


def evaluate_kalman_pairs_profile(
    y_ticker: str,
    x_ticker: str,
    df_y: pd.DataFrame,
    df_x: pd.DataFrame,
    delta: float = 1e-4,
    price_col: str = "adj_close",
) -> KalmanSpreadReceipt:
    """Evaluate full Kalman Filter dynamic pairs profile and issue audit receipt.

    Args:
        y_ticker: Ticker symbol for dependent leg Y (e.g. JPM).
        x_ticker: Ticker symbol for independent leg X (e.g. BAC).
        df_y: Observation dataframe for Y.
        df_x: Observation dataframe for X.
        delta: Filter process noise factor.
        price_col: Column name to use for prices.

    Returns:
        Cryptographically signed KalmanSpreadReceipt.
    """
    col_y = price_col if price_col in df_y.columns else "close"
    col_x = price_col if price_col in df_x.columns else "close"

    # Inner merge on session_date to guarantee strict alignment
    merged = pd.merge(
        df_y[["session_date", col_y]].rename(columns={col_y: "y_price"}),
        df_x[["session_date", col_x]].rename(columns={col_x: "x_price"}),
        on="session_date",
        how="inner",
    ).sort_values("session_date").reset_index(drop=True)

    n_sessions = len(merged)
    if n_sessions < 20:
        return KalmanSpreadReceipt(
            y_ticker=y_ticker.upper().strip(),
            x_ticker=x_ticker.upper().strip(),
            sample_sessions=n_sessions,
            delta_process_noise=delta,
            latest_alpha=0.0,
            latest_beta=1.0,
            latest_spread=0.0,
            latest_spread_std=1.0,
            latest_zscore=0.0,
            current_signal="INSUFFICIENT_DATA",
            ols_static_beta=1.0,
            beta_drift_pct=0.0,
            mean_reversion_half_life=20.0,
            spread_history=[],
        )

    # Static OLS benchmark: y = alpha + beta * x
    x_arr = merged["x_price"].values.astype(float)
    y_arr = merged["y_price"].values.astype(float)
    cov_xy = np.cov(x_arr, y_arr)[0, 1]
    var_x = np.var(x_arr)
    ols_beta = float(cov_xy / var_x) if var_x > 1e-12 else 1.0

    # Run Online Kalman Filter
    kalman_df = estimate_kalman_spread_series(
        y_prices=merged["y_price"],
        x_prices=merged["x_price"],
        delta=delta,
    )

    merged = pd.concat([merged, kalman_df], axis=1)

    latest_row = merged.iloc[-1]
    latest_alpha = float(latest_row["alpha"])
    latest_beta = float(latest_row["beta"])
    latest_spread = float(latest_row["spread"])
    latest_std = float(latest_row["spread_std"])
    latest_z = float(latest_row["zscore"])

    # Trading Signal Categorization
    if latest_z < -2.0:
        signal = "LONG_SPREAD"  # Underpriced: Buy Y, Short beta * X
    elif latest_z > 2.0:
        signal = "SHORT_SPREAD"  # Overpriced: Short Y, Buy beta * X
    elif abs(latest_z) <= 0.5:
        signal = "EXIT_POSITION"
    else:
        signal = "NEUTRAL"

    # Parameter drift vs static OLS
    beta_drift_pct = float(((latest_beta - ols_beta) / abs(ols_beta)) * 100.0) if abs(ols_beta) > 1e-6 else 0.0

    half_life = compute_half_life(merged["spread"])

    # Prepare recent spread history (last 60 sessions)
    history_records = []
    for _, row in merged.tail(60).iterrows():
        history_records.append({
            "session_date": str(row["session_date"]),
            "y_price": round(float(row["y_price"]), 2),
            "x_price": round(float(row["x_price"]), 2),
            "beta": round(float(row["beta"]), 4),
            "spread": round(float(row["spread"]), 4),
            "zscore": round(float(row["zscore"]), 2),
        })

    return KalmanSpreadReceipt(
        y_ticker=y_ticker.upper().strip(),
        x_ticker=x_ticker.upper().strip(),
        sample_sessions=n_sessions,
        delta_process_noise=delta,
        latest_alpha=latest_alpha,
        latest_beta=latest_beta,
        latest_spread=latest_spread,
        latest_spread_std=latest_std,
        latest_zscore=latest_z,
        current_signal=signal,
        ols_static_beta=ols_beta,
        beta_drift_pct=beta_drift_pct,
        mean_reversion_half_life=half_life,
        spread_history=history_records,
    )
