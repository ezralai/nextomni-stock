"""Exploratory Data Analysis (EDA) engine for NextOmni Stock.

Computes statistical distributions, returns, volatility, drawdowns,
and generates interactive Plotly visualizations.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.config import PRICE_COLUMN, PROJECT_ROOT
from src.data.db import get_observations


@dataclass(frozen=True)
class StockStatistics:
    ticker: str
    exchange: str
    total_sessions: int
    start_date: str
    end_date: str
    first_price: float
    last_price: float
    min_price: float
    max_price: float
    mean_price: float
    cumulative_return_pct: float
    annualized_return_pct: float
    annualized_volatility_pct: float
    max_drawdown_pct: float
    skewness: float
    kurtosis: float
    avg_daily_volume: float


def compute_stock_analytics(df: pd.DataFrame) -> Tuple[pd.DataFrame, StockStatistics]:
    """Compute returns, moving averages, rolling volatility, and drawdowns on market observations.

    Args:
        df: DataFrame of market observations sorted chronologically.

    Returns:
        Tuple[pd.DataFrame, StockStatistics]:
            - Enhanced DataFrame with analytical columns.
            - Statistical summary metrics.
    """
    if df.empty or len(df) < 2:
        raise ValueError("DataFrame must contain at least 2 observations for exploratory analysis.")

    data = df.copy()
    data = data.sort_values("session_date", ascending=True).reset_index(drop=True)

    # 1. Price Returns
    # Simple daily return: (P_t - P_{t-1}) / P_{t-1}
    data["daily_return"] = data[PRICE_COLUMN].pct_change()
    # Log return: ln(P_t / P_{t-1}) (useful for statistical modeling)
    data["log_return"] = np.log(data[PRICE_COLUMN] / data[PRICE_COLUMN].shift(1))

    # 2. Moving Averages
    data["sma_5"] = data[PRICE_COLUMN].rolling(window=5, min_periods=1).mean()
    data["sma_20"] = data[PRICE_COLUMN].rolling(window=20, min_periods=1).mean()
    data["sma_50"] = data[PRICE_COLUMN].rolling(window=50, min_periods=1).mean()

    # 3. Rolling Volatility
    # 21 trading sessions (~1 month) rolling standard deviation of daily returns,
    # annualized by multiplying by sqrt(252 trading days per year).
    data["rolling_volatility_21"] = (
        data["daily_return"].rolling(window=21, min_periods=5).std() * np.sqrt(252)
    )

    # 4. Drawdown Analysis
    # Peak-to-trough decline from the all-time rolling high up to time t
    data["running_peak"] = data[PRICE_COLUMN].cummax()
    data["drawdown"] = (data[PRICE_COLUMN] - data["running_peak"]) / data["running_peak"]

    # 5. Summary Statistics Calculation
    valid_returns = data["daily_return"].dropna()
    total_sessions = len(data)
    first_price = float(data[PRICE_COLUMN].iloc[0])
    last_price = float(data[PRICE_COLUMN].iloc[-1])

    # Cumulative return over observed window
    cum_return = (last_price - first_price) / first_price if first_price > 0 else 0.0

    # Annualized return: (1 + cum_return)^(252 / total_sessions) - 1
    if total_sessions > 1 and (1 + cum_return) > 0:
        annualized_return = ((1 + cum_return) ** (252.0 / total_sessions)) - 1.0
    else:
        annualized_return = 0.0

    # Annualized volatility
    ann_volatility = float(valid_returns.std() * np.sqrt(252)) if len(valid_returns) > 1 else 0.0

    # Max Drawdown
    max_drawdown = float(data["drawdown"].min()) if not data["drawdown"].isna().all() else 0.0

    # Distribution Moments: Skewness & Kurtosis
    skew = float(valid_returns.skew()) if len(valid_returns) > 2 else 0.0
    kurt = float(valid_returns.kurtosis()) if len(valid_returns) > 3 else 0.0

    stats = StockStatistics(
        ticker=str(data["ticker"].iloc[0]),
        exchange=str(data["exchange"].iloc[0]),
        total_sessions=total_sessions,
        start_date=str(data["session_date"].iloc[0]),
        end_date=str(data["session_date"].iloc[-1]),
        first_price=round(first_price, 4),
        last_price=round(last_price, 4),
        min_price=round(float(data[PRICE_COLUMN].min()), 4),
        max_price=round(float(data[PRICE_COLUMN].max()), 4),
        mean_price=round(float(data[PRICE_COLUMN].mean()), 4),
        cumulative_return_pct=round(cum_return * 100, 2),
        annualized_return_pct=round(annualized_return * 100, 2),
        annualized_volatility_pct=round(ann_volatility * 100, 2),
        max_drawdown_pct=round(max_drawdown * 100, 2),
        skewness=round(skew, 3),
        kurtosis=round(kurt, 3),
        avg_daily_volume=round(float(data["volume"].mean()), 0),
    )

    return data, stats


def generate_eda_figures(df: pd.DataFrame, ticker: str) -> Dict[str, go.Figure]:
    """Generate interactive Plotly figures for price, returns, volatility, and drawdown."""
    figures = {}

    # 1. Price, Moving Averages and Volume Subplot
    fig_price = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=(f"{ticker} - Adjusted Close & Moving Averages", "Trading Volume"),
        row_heights=[0.7, 0.3],
    )

    # Price Line
    fig_price.add_trace(
        go.Scatter(
            x=df["session_date"],
            y=df[PRICE_COLUMN],
            name="Adjusted Close",
            line=dict(color="#2962FF", width=2),
        ),
        row=1,
        col=1,
    )
    # SMA 20
    fig_price.add_trace(
        go.Scatter(
            x=df["session_date"],
            y=df["sma_20"],
            name="SMA 20 (1 Month)",
            line=dict(color="#FF6D00", width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    # SMA 50
    fig_price.add_trace(
        go.Scatter(
            x=df["session_date"],
            y=df["sma_50"],
            name="SMA 50",
            line=dict(color="#00C853", width=1.5, dash="dot"),
        ),
        row=1,
        col=1,
    )
    # Volume Bar
    fig_price.add_trace(
        go.Bar(
            x=df["session_date"],
            y=df["volume"],
            name="Volume",
            marker=dict(color="#90A4AE"),
        ),
        row=2,
        col=1,
    )

    fig_price.update_layout(
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=50, b=40),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    figures["price_and_volume"] = fig_price

    # 2. Return Distribution & Histogram
    valid_returns = df["daily_return"].dropna() * 100
    fig_dist = go.Figure()
    fig_dist.add_trace(
        go.Histogram(
            x=valid_returns,
            nbinsx=40,
            name="Daily Return %",
            marker=dict(color="#3949AB", line=dict(color="#1A237E", width=1)),
            opacity=0.75,
        )
    )
    fig_dist.update_layout(
        title=f"{ticker} - Daily Return Distribution (%)",
        xaxis_title="Daily Return (%)",
        yaxis_title="Frequency (Sessions)",
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    figures["return_distribution"] = fig_dist

    # 3. Rolling Volatility Chart
    fig_vol = go.Figure()
    fig_vol.add_trace(
        go.Scatter(
            x=df["session_date"],
            y=df["rolling_volatility_21"] * 100,
            name="21-Session Annualized Volatility",
            line=dict(color="#D50000", width=1.8),
        )
    )
    fig_vol.update_layout(
        title=f"{ticker} - 21-Session Rolling Volatility (Annualized %)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    figures["rolling_volatility"] = fig_vol

    # 4. Underwater Drawdown Chart
    fig_dd = go.Figure()
    fig_dd.add_trace(
        go.Scatter(
            x=df["session_date"],
            y=df["drawdown"] * 100,
            fill="tozeroy",
            name="Drawdown %",
            line=dict(color="#C2185B", width=1.5),
            fillcolor="rgba(194, 24, 91, 0.2)",
        )
    )
    fig_dd.update_layout(
        title=f"{ticker} - Historical Drawdown from Peak (%)",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        template="plotly_white",
        height=380,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    figures["drawdown"] = fig_dd

    return figures


def export_eda_report(
    ticker: str,
    output_dir: Optional[Path] = None,
    db_path: Optional[Path] = None,
) -> Tuple[StockStatistics, Path]:
    """Execute EDA and export interactive HTML visualization report."""
    target_dir = output_dir or (PROJECT_ROOT / "reports")
    target_dir.mkdir(parents=True, exist_ok=True)

    df = get_observations(ticker, db_path=db_path)
    if df.empty:
        raise ValueError(f"No historical data found for ticker '{ticker}'. Run 'sync' first.")

    enhanced_df, stats = compute_stock_analytics(df)
    figures = generate_eda_figures(enhanced_df, ticker)

    report_path = target_dir / f"eda_{ticker.replace('.', '_')}.html"

    # Assemble HTML document containing all interactive Plotly figures and statistics
    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>NextOmni Stock EDA - {stats.ticker} ({stats.exchange})</title>
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #F8F9FA; color: #212529; margin: 0; padding: 24px; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: #FFFFFF; padding: 24px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
        h1 {{ margin: 0 0 8px 0; font-size: 24px; color: #1A237E; }}
        .badge {{ background: #E8EAF6; color: #283593; padding: 4px 10px; border-radius: 4px; font-size: 13px; font-weight: 600; display: inline-block; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 20px; }}
        .metric-card {{ background: #FAFAFA; border: 1px solid #EEEEEE; padding: 14px; border-radius: 6px; }}
        .metric-label {{ font-size: 12px; color: #757575; text-transform: uppercase; margin-bottom: 4px; }}
        .metric-val {{ font-size: 18px; font-weight: 700; color: #212121; }}
        .chart-box {{ background: #FFFFFF; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>NextOmni Stock: Exploratory Data Analysis & Statistics</h1>
            <span class="badge">{stats.ticker}</span>
            <span class="badge">Exchange: {stats.exchange}</span>
            <span class="badge">Period: {stats.start_date} to {stats.end_date}</span>
            <div class="metrics-grid">
                <div class="metric-card"><div class="metric-label">Total Sessions</div><div class="metric-val">{stats.total_sessions}</div></div>
                <div class="metric-card"><div class="metric-label">First Price</div><div class="metric-val">{stats.first_price}</div></div>
                <div class="metric-card"><div class="metric-label">Latest Price</div><div class="metric-val">{stats.last_price}</div></div>
                <div class="metric-card"><div class="metric-label">Cumulative Return</div><div class="metric-val">{stats.cumulative_return_pct}%</div></div>
                <div class="metric-card"><div class="metric-label">Ann. Volatility</div><div class="metric-val">{stats.annualized_volatility_pct}%</div></div>
                <div class="metric-card"><div class="metric-label">Max Drawdown</div><div class="metric-val">{stats.max_drawdown_pct}%</div></div>
                <div class="metric-card"><div class="metric-label">Skewness</div><div class="metric-val">{stats.skewness}</div></div>
                <div class="metric-card"><div class="metric-label">Kurtosis</div><div class="metric-val">{stats.kurtosis}</div></div>
            </div>
        </div>

        <div class="chart-box">{figures['price_and_volume'].to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="chart-box">{figures['return_distribution'].to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="chart-box">{figures['rolling_volatility'].to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="chart-box">{figures['drawdown'].to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
</body>
</html>
"""
    report_path.write_text(html_content, encoding="utf-8")
    return stats, report_path
