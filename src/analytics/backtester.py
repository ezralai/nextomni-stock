"""Quantitative backtesting and execution simulation engine for NextOmni Stock.

Simulates walk-forward quantitative trading strategies driven by multi-horizon forecast models,
accounting for transaction friction (brokerage commission & slippage), and evaluating
risk-adjusted performance (Annualized Sharpe, Sortino ratio, Drawdown, Profit Factor).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import PRICE_COLUMN
from src.features.builder import build_feature_matrix
from src.models.base_model import BaseForecaster
from src.models.ensemble import ConvexEnsembleForecaster
from src.models.gb_model import GradientBoostingForecaster
from src.models.linear_model import LinearRegressionForecaster


@dataclass
class BacktestMetrics:
    """Summary risk-adjusted metrics for strategy evaluation."""
    total_return_pct: float
    benchmark_return_pct: float
    annualized_return_pct: float
    annualized_sharpe: float
    annualized_sortino: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float
    total_trades: int
    fee_drag_pct: float
    avg_exposure: float = 1.0


@dataclass
class BacktestResult:
    """Complete backtest output including equity curve time series and metrics."""
    metrics: BacktestMetrics
    equity_df: pd.DataFrame


def calculate_sharpe_ratio(
    returns: Union[np.ndarray, pd.Series, List[float]],
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sharpe Ratio.

    Args:
        returns: Daily or periodic strategy returns.
        risk_free_rate: Annualized risk-free benchmark rate.
        periods_per_year: Number of trading sessions per year (252 for stocks).

    Returns:
        Annualized Sharpe ratio (returns 0.0 if standard deviation is zero).
    """
    arr = np.asarray(returns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 2:
        return 0.0

    rf_per_period = risk_free_rate / periods_per_year
    excess = clean - rf_per_period
    std = float(np.std(excess, ddof=1))
    if std <= 1e-8:
        return 0.0

    mean_excess = float(np.mean(excess))
    return float((mean_excess / std) * np.sqrt(periods_per_year))


def calculate_sortino_ratio(
    returns: Union[np.ndarray, pd.Series, List[float]],
    target_return: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """Calculate annualized Sortino Ratio focusing on downside volatility.

    Args:
        returns: Daily or periodic strategy returns.
        target_return: Minimum acceptable return (default: 0.0).
        periods_per_year: Number of trading sessions per year.

    Returns:
        Annualized Sortino ratio (returns 0.0 if downside deviation is zero).
    """
    arr = np.asarray(returns, dtype=float)
    clean = arr[np.isfinite(arr)]
    if len(clean) < 2:
        return 0.0

    target_per_period = target_return / periods_per_year
    excess = clean - target_per_period
    downside_diff = np.minimum(excess, 0.0)
    downside_variance = np.mean(downside_diff ** 2)
    downside_dev = np.sqrt(downside_variance)

    if downside_dev <= 1e-8:
        return 0.0

    mean_excess = float(np.mean(excess))
    return float((mean_excess / downside_dev) * np.sqrt(periods_per_year))


def calculate_max_drawdown(equity: Union[np.ndarray, pd.Series, List[float]]) -> float:
    """Calculate maximum peak-to-trough percentage decline.

    Returns:
        Max drawdown as positive percentage (e.g. 15.5 for 15.5% drawdown).
    """
    eq = np.asarray(equity, dtype=float)
    if len(eq) < 2:
        return 0.0

    running_max = np.maximum.accumulate(eq)
    drawdowns = (running_max - eq) / np.maximum(running_max, 1e-8)
    return float(np.max(drawdowns) * 100.0)


def run_forecast_backtest(
    df: pd.DataFrame,
    forecaster: Optional[BaseForecaster] = None,
    horizon_sessions: int = 1,
    fee_bps: float = 5.0,
    min_train_sessions: int = 120,
    model_family: str = "ensemble",
    threshold_return_bps: float = 0.0,
    sizing_mode: str = "fixed",
    target_annual_vol: float = 0.15,
    max_leverage: float = 1.0,
) -> BacktestResult:
    """Execute an expanding-window walk-forward out-of-sample backtest.

    Strategy Logic:
    - At session t, train model on features P_{<= t}.
    - Forecast expected return r_hat_{t+k}.
    - Determine position size:
      * 'fixed': 1.0 if r_hat > threshold else 0.0.
      * 'vol_target': Scale exposure inversely with historical volatility to target annual volatility.
      * 'half_kelly': Scale exposure proportional to expected return / variance.
    - Deduct transaction costs (fee_bps) on turnover.
    - Track net daily portfolio returns and compounding equity curves.

    Args:
        df: Chronologically sorted observations DataFrame.
        forecaster: Optional pre-configured BaseForecaster.
        horizon_sessions: Step horizon (1, 5, or 21 sessions).
        fee_bps: One-way transaction fee in basis points (5 bps = 0.05%).
        min_train_sessions: Initial burn-in training sessions.
        model_family: Model family if forecaster not provided ("ensemble", "linear", "gradient_boosting").
        threshold_return_bps: Minimum forecasted return to enter long position in bps.
        sizing_mode: Position sizing mechanism ('fixed', 'vol_target', 'half_kelly').
        target_annual_vol: Desired annualized volatility when using 'vol_target' (e.g. 0.15 for 15%).
        max_leverage: Maximum allowed portfolio leverage (default: 1.0).

    Returns:
        BacktestResult with detailed time series and summary metrics.
    """
    valid_sizing = {"fixed", "vol_target", "half_kelly"}
    if sizing_mode not in valid_sizing:
        raise ValueError(f"Invalid sizing_mode '{sizing_mode}'. Must be one of {sorted(list(valid_sizing))}")
    data = df.copy().sort_values("session_date", ascending=True).reset_index(drop=True)
    n_total = len(data)

    if n_total < min_train_sessions + horizon_sessions + 5:
        raise ValueError(
            f"Insufficient sessions ({n_total}) for walk-forward backtest "
            f"with min_train={min_train_sessions} and horizon={horizon_sessions}."
        )

    fee_rate = fee_bps / 10000.0
    threshold = threshold_return_bps / 10000.0

    # Build feature matrix
    feat_df, feat_cols = build_feature_matrix(
        data, include_target=True, horizon_sessions=horizon_sessions
    )
    valid_df = feat_df.dropna(subset=feat_cols).copy().reset_index(drop=True)
    n_valid = len(valid_df)

    dates: List[str] = []
    prices: List[float] = []
    pred_returns: List[float] = []
    signals: List[float] = []
    asset_returns: List[float] = []
    strat_returns: List[float] = []
    fee_costs: List[float] = []

    current_position = 0.0

    for t in range(min_train_sessions, n_valid - horizon_sessions):
        # 1. Training fold strictly up to t (ZERO LOOKAHEAD LEAKAGE)
        train_fold = valid_df.iloc[:t].dropna(subset=["target_price"])
        X_tr = train_fold[feat_cols]
        y_tr = train_fold["target_price"]
        p_tr = train_fold[PRICE_COLUMN]

        # Initialize or clone model
        if forecaster is None:
            if model_family == "linear":
                model = LinearRegressionForecaster(alpha=1.0)
            elif model_family == "gradient_boosting":
                model = GradientBoostingForecaster(max_depth=2, max_iter=30, random_state=42)
            else:
                model = ConvexEnsembleForecaster()
        else:
            # Re-instantiate identical model family
            model = forecaster.__class__()

        model.fit(X_tr, y_tr, p_tr, fit_quantiles=False)

        # 2. Predict on observation t
        eval_row = valid_df.iloc[t : t + 1]
        p_t = float(eval_row[PRICE_COLUMN].iloc[0])
        p_next = float(valid_df.iloc[t + horizon_sessions][PRICE_COLUMN])
        session_date = str(valid_df.iloc[t + horizon_sessions]["session_date"])

        pred_price = float(model.predict(eval_row[feat_cols], p_t)[0])
        pred_ret = (pred_price - p_t) / p_t
        realized_asset_ret = (p_next - p_t) / p_t

        # 3. Position decision based on sizing_mode
        if pred_ret > threshold:
            if sizing_mode == "vol_target":
                # Rolling 20-session historical volatility from training fold (strictly <= t)
                p_hist = p_tr.tail(21)
                if len(p_hist) >= 5:
                    pct_changes = p_hist.pct_change().dropna()
                    hist_vol = float(pct_changes.std(ddof=1) * np.sqrt(252))
                else:
                    hist_vol = target_annual_vol

                safe_vol = max(hist_vol, 1e-4) if np.isfinite(hist_vol) and hist_vol > 0 else target_annual_vol
                raw_pos = target_annual_vol / safe_vol
                target_position = float(np.clip(raw_pos, 0.1, max_leverage))
            elif sizing_mode == "half_kelly":
                p_hist = p_tr.tail(21)
                if len(p_hist) >= 5:
                    pct_changes = p_hist.pct_change().dropna()
                    var_daily = float(pct_changes.var(ddof=1))
                else:
                    var_daily = 0.0004
                safe_var = (
                    max(var_daily * max(horizon_sessions, 1), 1e-5)
                    if np.isfinite(var_daily) and var_daily > 0
                    else 0.0004
                )
                raw_pos = 0.5 * (pred_ret / safe_var)
                target_position = float(np.clip(raw_pos, 0.0, max_leverage))
            else:
                target_position = 1.0
        else:
            target_position = 0.0

        # Transaction cost on turnover
        turnover = abs(target_position - current_position)
        cost = turnover * fee_rate

        # Net strategy return over the horizon period
        net_strat_ret = (current_position * realized_asset_ret) - cost

        dates.append(session_date)
        prices.append(p_next)
        pred_returns.append(pred_ret)
        signals.append(current_position)
        asset_returns.append(realized_asset_ret)
        strat_returns.append(net_strat_ret)
        fee_costs.append(cost)

        current_position = target_position

    # Construct compounding equity series (starting at 100.0)
    strat_equity = [100.0]
    bench_equity = [100.0]

    for s_ret, a_ret in zip(strat_returns, asset_returns):
        strat_equity.append(strat_equity[-1] * (1.0 + s_ret))
        bench_equity.append(bench_equity[-1] * (1.0 + a_ret))

    strat_eq_arr = np.array(strat_equity[1:])
    bench_eq_arr = np.array(bench_equity[1:])

    total_return_pct = float((strat_eq_arr[-1] - 100.0))
    benchmark_return_pct = float((bench_eq_arr[-1] - 100.0))

    n_periods = len(strat_returns)
    years = max(n_periods / (252.0 / horizon_sessions), 0.1)
    ann_return_pct = float(((strat_eq_arr[-1] / 100.0) ** (1.0 / years) - 1.0) * 100.0)

    annualized_sharpe = calculate_sharpe_ratio(
        strat_returns, periods_per_year=int(252 / horizon_sessions)
    )
    annualized_sortino = calculate_sortino_ratio(
        strat_returns, periods_per_year=int(252 / horizon_sessions)
    )
    max_dd_pct = calculate_max_drawdown(strat_eq_arr)

    # Win rate and profit factor on active positions
    active_returns = [r for pos, r in zip(signals, strat_returns) if pos > 0]
    wins = [r for r in active_returns if r > 0]
    losses = [r for r in active_returns if r < 0]

    win_rate_pct = float((len(wins) / len(active_returns) * 100.0)) if active_returns else 0.0
    gross_gains = sum(wins)
    gross_losses = abs(sum(losses))
    profit_factor = (
        float(gross_gains / gross_losses) if gross_losses > 1e-8 else (2.0 if gross_gains > 0 else 1.0)
    )

    trades_count = int(np.sum(np.abs(np.diff([0.0] + signals)) > 0))
    total_fee_drag = float(sum(fee_costs) * 100.0)
    avg_exposure = float(np.mean(signals)) if signals else 0.0

    metrics = BacktestMetrics(
        total_return_pct=round(total_return_pct, 2),
        benchmark_return_pct=round(benchmark_return_pct, 2),
        annualized_return_pct=round(ann_return_pct, 2),
        annualized_sharpe=round(annualized_sharpe, 3),
        annualized_sortino=round(annualized_sortino, 3),
        max_drawdown_pct=round(max_dd_pct, 2),
        win_rate_pct=round(win_rate_pct, 2),
        profit_factor=round(profit_factor, 2),
        total_trades=trades_count,
        fee_drag_pct=round(total_fee_drag, 2),
        avg_exposure=round(avg_exposure, 3),
    )

    equity_df = pd.DataFrame(
        {
            "session_date": dates,
            "settlement_price": prices,
            "forecast_return_pct": [r * 100.0 for r in pred_returns],
            "position": signals,
            "asset_return_pct": [r * 100.0 for r in asset_returns],
            "strategy_return_pct": [r * 100.0 for r in strat_returns],
            "strategy_equity": strat_eq_arr,
            "benchmark_equity": bench_eq_arr,
        }
    )

    return BacktestResult(metrics=metrics, equity_df=equity_df)
