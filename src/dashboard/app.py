"""NextOmni Stock (nextomni-stock) Interactive Intelligence Dashboard.

A portfolio-grade financial application providing:
1. Live Market Overview (Quotes, session statuses, data latency badges)
2. Stock Analytics & EDA (Interactive Plotly charts, SMAs, volatility, drawdowns)
3. Multi-Horizon Price Forecasting (1, 5, 21 trading sessions direct models)
4. Prediction Tracking & Audit Log (Predicted vs Actual reconciliation)
5. Model Performance & Walk-Forward Validation (Backtest metrics and feature weights)
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.eda import compute_stock_analytics
from src.config import (
    FEED_DELAYED,
    FEED_EOD,
    FEED_REALTIME,
    INITIAL_TICKERS,
    PRICE_COLUMN,
    PROJECT_ROOT,
    SUPPORTED_EXCHANGES,
    SUPPORTED_HORIZONS,
)
from src.data.db import get_connection, get_observations, get_predictions, init_db
from src.data.monitor import MarketMonitorService, MarketQuote
from src.data.pipeline import DataPipeline
from src.data.session_status import ExchangeSessionEngine
from src.evaluation.tracker import PredictionTracker
from src.models.multi_horizon import MultiHorizonForecaster


# Initialize database
init_db()


# -----------------------------------------------------------------------------
# Data Loading & Caching Helpers
# -----------------------------------------------------------------------------
@st.cache_data(ttl=30)
def load_market_quotes(force_refresh: bool = False) -> List[MarketQuote]:
    """Fetch live quotes with operational session state across all default tickers."""
    service = MarketMonitorService()
    return service.get_multi_quotes(force_refresh=force_refresh)


@st.cache_data(ttl=60)
def load_stock_observations(ticker: str, exchange: Optional[str] = None) -> pd.DataFrame:
    """Fetch historical observations dataframe from SQLite."""
    return get_observations(ticker=ticker)


@st.cache_data(ttl=15)
def load_predictions_data(status: Optional[str] = None, ticker: Optional[str] = None) -> pd.DataFrame:
    """Fetch recorded predictions and evaluations from SQLite."""
    return get_predictions(status=status, ticker=ticker)


# -----------------------------------------------------------------------------
# Main Dashboard UI Renderer
# -----------------------------------------------------------------------------
def render_dashboard() -> None:
    """Render the full interactive Streamlit dashboard layout and tabs."""
    # Page configuration
    st.set_page_config(
        page_title="NextOmni Stock | Quantitative Intelligence",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Sidebar: Exchange & Stock Selector
    st.sidebar.title("NextOmni Stock")
    st.sidebar.caption("Dual-Exchange Quantitative Intelligence Service")

    raw_exchange = st.sidebar.selectbox(
        "Select Exchange",
        options=list(SUPPORTED_EXCHANGES.keys()),
        format_func=lambda k: f"{SUPPORTED_EXCHANGES[k].name} ({k})",
    )
    from src.security.validators import validate_exchange, validate_model_family, validate_ticker
    selected_exchange = validate_exchange(raw_exchange)
    exchange_cfg = SUPPORTED_EXCHANGES[selected_exchange]
    available_tickers = INITIAL_TICKERS.get(selected_exchange, [])

    raw_ticker = st.sidebar.selectbox(
        "Select Stock Ticker",
        options=available_tickers,
        index=0,
    )
    selected_ticker = validate_ticker(raw_ticker)

    raw_model = st.sidebar.selectbox(
        "Forecasting Architecture",
        options=["adaptive_ensemble", "ensemble", "gradient_boosting", "linear"],
        format_func=lambda x: {
            "adaptive_ensemble": "🧠 Dynamic Regime-Adaptive (Market-State Conditional)",
            "ensemble": "🛡️⚡ Convex Ensemble (Ridge + GBDT)",
            "gradient_boosting": "⚡ Gradient Boosting (Non-Linear)",
            "linear": "📈 Ridge Regression (Linear)",
        }.get(x, x),
    )

    selected_model_family = validate_model_family(raw_model)

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        **System Parameters:**
        - **Active Model:** `{selected_model_family}`
        - **MIC:** `{exchange_cfg.mic}`
        - **Timezone:** `{exchange_cfg.timezone}`
        - **Currency:** `{exchange_cfg.currency}`
        - **Hours:** `{exchange_cfg.session_open} - {exchange_cfg.session_close}`
        """
    )

    # Main Application Tabs
    tabs = st.tabs([
        "🔴 Live Market Monitor",
        "📊 Stock Analytics & EDA",
        "🔮 Multi-Horizon Forecasts",
        "📋 Prediction Audit Log",
        "🏆 Model Performance",
    ])

    # =========================================================================
    # TAB 1: Live Market Monitor
    # =========================================================================
    with tabs[0]:
        st.header("Real-Time Market Monitoring")
        st.caption("Live quotes, exchange operational states, and data latency badges across NYSE and Bursa Malaysia.")

        # Exchange Operational Status Cards
        col_nyse, col_bursa = st.columns(2)
        with col_nyse:
            nyse_status = ExchangeSessionEngine.get_status("XNYS")
            badge_color = "🟢" if nyse_status.is_open else "🔴"
            st.subheader(f"{badge_color} New York Stock Exchange (XNYS)")
            st.markdown(f"**Operational State:** `{nyse_status.status}`")
            st.markdown(f"**Local Time:** `{nyse_status.local_time}`")
            if nyse_status.is_open and nyse_status.next_close:
                st.info(f"Active Session: Scheduled close at **{nyse_status.next_close}**")
            elif not nyse_status.is_open and nyse_status.next_open:
                st.info(f"Market Closed: Scheduled re-open at **{nyse_status.next_open}**")
            st.caption(nyse_status.details)

        with col_bursa:
            bursa_status = ExchangeSessionEngine.get_status("XKLS")
            badge_color = "🟢" if bursa_status.is_open else ("🟡" if bursa_status.status == "LUNCH_BREAK" else "🔴")
            st.subheader(f"{badge_color} Bursa Malaysia (XKLS)")
            st.markdown(f"**Operational State:** `{bursa_status.status}`")
            st.markdown(f"**Local Time:** `{bursa_status.local_time}`")
            if bursa_status.is_open and bursa_status.next_close:
                st.info(f"Active Session: Scheduled close at **{bursa_status.next_close}**")
            elif bursa_status.status == "LUNCH_BREAK" and bursa_status.next_open:
                st.warning(f"Midday Lunch Break: Afternoon session resumes at **{bursa_status.next_open}**")
            elif not bursa_status.is_open and bursa_status.next_open:
                st.info(f"Market Closed: Scheduled re-open at **{bursa_status.next_open}**")
            st.caption(bursa_status.details)

        st.markdown("---")

        # Refresh controls and quotes table
        col_tbl_header, col_refresh = st.columns([5, 1])
        with col_tbl_header:
            st.subheader("Monitored Securities")
        with col_refresh:
            refresh_clicked = st.button("🔄 Force Refresh", use_container_width=True)

        if refresh_clicked:
            st.cache_data.clear()
            quotes = load_market_quotes(force_refresh=True)
            st.toast("Quotes refreshed directly from providers!", icon="✅")
        else:
            quotes = load_market_quotes()

        records = []
        for q in quotes:
            records.append({
                "Ticker": q.ticker,
                "Exchange": q.exchange,
                "Market": q.exchange_name,
                "Price": f"{q.last_price:.2f} {q.currency}",
                "Change": f"{q.change:+.2f}",
                "Change %": f"{q.pct_change:+.2f}%",
                "Session": q.session_status,
                "Data Latency": q.latency_badge,
                "Cached": "Yes" if q.cached else "No",
                "Provider": q.provider,
                "Operational Detail": q.next_event,
            })

        df_quotes = pd.DataFrame(records)
        st.dataframe(df_quotes, use_container_width=True, hide_index=True)

        st.markdown("---")
        with st.expander("⚡ Cross-Sectional Alpha Universe Screener", expanded=False):
            st.markdown(
                "Scan all synchronized stocks across NYSE and Bursa Malaysia, ranking assets cross-sectionally by expected return, "
                "calibrated directional win probability, Cornish-Fisher 95% downside tail risk, and expected Sharpe ratio."
            )
            col_sc1, col_sc2 = st.columns(2)
            with col_sc1:
                sc_horizon = st.selectbox("Screener Horizon", [1, 5, 21], index=1, format_func=lambda h: f"{h} Session(s)", key="sc_h")
            with col_sc2:
                sc_model = st.selectbox("Screener Model Family", ["ensemble", "linear", "gradient_boosting"], index=0, key="sc_m")

            if st.button("🔍 Scan & Rank Asset Universe", use_container_width=True):
                with st.spinner("Screening cross-sectional asset universe..."):
                    from src.analytics.screener import screen_asset_universe
                    sc_results = screen_asset_universe(horizon_sessions=sc_horizon, model_family=sc_model)
                    if sc_results:
                        sc_df = pd.DataFrame([
                            {
                                "Rank": r.rank,
                                "Ticker": r.ticker,
                                "Exchange": r.exchange,
                                "Company": r.company_name,
                                "Price": f"{r.latest_price:.2f} {r.currency}",
                                "Expected Return": f"{r.expected_return_pct:+.2f}%",
                                "Win Prob": f"{r.directional_win_probability * 100:.1f}%",
                                "Conviction": r.conviction_label,
                                "95% CF-VaR": f"-{r.cornish_fisher_var_pct:.2f}%",
                                "Sharpe Score": f"{r.predicted_sharpe:+.2f}",
                            }
                            for r in sc_results
                        ])
                        st.dataframe(sc_df, use_container_width=True, hide_index=True)

                        # Portfolio Optimizer Section
                        if len(sc_results) >= 2:
                            st.markdown("---")
                            st.subheader("💼 Optimal Multi-Asset Portfolio Allocation")
                            st.caption("Analytical Ledoit-Wolf shrinkage covariance estimation with Markowitz long-only optimization.")
                            opt_c1, opt_c2 = st.columns(2)
                            with opt_c1:
                                opt_mode = st.selectbox(
                                    "Portfolio Objective Mode",
                                    ["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp"],
                                    format_func=lambda m: {
                                        "max_sharpe": "Maximum Sharpe Ratio",
                                        "min_variance": "Minimum Variance",
                                        "risk_parity": "Equal Risk Contribution (Risk Parity)",
                                        "equal_weight": "Equal Weighting (1/N)",
                                        "hrp": "Hierarchical Risk Parity (HRP)",
                                    }[m],
                                    key="opt_mode",
                                )

                            with opt_c2:
                                opt_max_w = st.slider("Max Single-Asset Weight", 0.10, 1.0, 0.40, step=0.05, key="opt_max_w")

                            from src.analytics.portfolio import optimize_alpha_portfolio
                            port_rec = optimize_alpha_portfolio(
                                screener_records=sc_results,
                                mode=opt_mode,
                                max_weight=opt_max_w,
                            )

                            p_kpi1, p_kpi2, p_kpi3, p_kpi4 = st.columns(4)
                            p_kpi1.metric("Exp Annual Return", f"{port_rec.expected_annual_return_pct:+.2f}%")
                            p_kpi2.metric("Annualized Volatility", f"{port_rec.annualized_volatility_pct:.2f}%")
                            p_kpi3.metric("Portfolio Sharpe", f"{port_rec.sharpe_ratio:+.2f}")
                            p_kpi4.metric("LW Shrinkage (δ)", f"{port_rec.shrinkage_intensity:.4f}")

                            w_df = pd.DataFrame([
                                {
                                    "Ticker": t,
                                    "Weight": f"{w * 100:.1f}%",
                                    "Risk Contribution": f"{port_rec.risk_contributions_pct.get(t, 0.0):.1f}%",
                                }
                                for t, w in port_rec.weights.items()
                                if w > 0.001
                            ])
                            st.dataframe(w_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No synchronized assets with sufficient observation history found. Synchronize tickers first.")

    # =========================================================================
    # TAB 2: Stock Analytics & EDA
    # =========================================================================
    with tabs[1]:
        st.header(f"Exploratory Data Analysis: {selected_ticker} ({selected_exchange})")
        df_obs = load_stock_observations(selected_ticker, selected_exchange)

        if df_obs.empty:
            st.warning(f"No historical observations found in SQLite database for {selected_ticker}.")
            if st.button(f"Sync 1-Year History for {selected_ticker} now"):
                with st.spinner("Downloading and syncing historical data..."):
                    pipe = DataPipeline()
                    end_d = datetime.now().strftime("%Y-%m-%d")
                    start_d = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                    res = pipe.sync_stock(selected_ticker, selected_exchange, start_d, end_d)
                    st.success(f"Synced {res['sessions_synced']} trading sessions!")
                    st.cache_data.clear()
                    st.rerun()
        else:
            df_ind, stats = compute_stock_analytics(df_obs)

            from src.analytics.regime import detect_market_regime
            if len(df_obs) >= 20:
                regime_info = detect_market_regime(df_obs)
                st.info(
                    f"🧭 **Market Regime:** `{regime_info.regime_label}` | "
                    f"**Sentiment Score:** `{regime_info.sentiment_score:+.2f}` | "
                    f"**Volatility State:** `{regime_info.volatility_regime}` ({regime_info.rolling_vol_annualized:.1f}% Ann, {regime_info.vol_percentile_1y:.1f}% 1Y Percentile) | "
                    f"**Trend State:** `{regime_info.trend_regime}`"
                )

            # Key Metrics Row
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Latest Adj Close", f"{stats.last_price:.2f} {exchange_cfg.currency}")
            c2.metric("Cumulative Return", f"{stats.cumulative_return_pct:+.2f}%")
            c3.metric("Annualized Volatility", f"{stats.annualized_volatility_pct:.2f}%")
            c4.metric("Max Drawdown", f"{stats.max_drawdown_pct:.2f}%")

            # Interactive Price & Moving Averages Chart
            st.subheader("Price History & Moving Averages (SMA 5, 20, 50)")
            fig_price = go.Figure()
            fig_price.add_trace(go.Scatter(
                x=df_ind["session_date"], y=df_ind[PRICE_COLUMN],
                name="Adjusted Close", line=dict(color="#2962FF", width=2)
            ))
            if "sma_5" in df_ind.columns:
                fig_price.add_trace(go.Scatter(
                    x=df_ind["session_date"], y=df_ind["sma_5"],
                    name="SMA 5 (1-Week)", line=dict(color="#00E676", width=1.5, dash="dot")
                ))
            if "sma_20" in df_ind.columns:
                fig_price.add_trace(go.Scatter(
                    x=df_ind["session_date"], y=df_ind["sma_20"],
                    name="SMA 20 (1-Month)", line=dict(color="#FF9100", width=1.5, dash="dash")
                ))
            if "sma_50" in df_ind.columns:
                fig_price.add_trace(go.Scatter(
                    x=df_ind["session_date"], y=df_ind["sma_50"],
                    name="SMA 50 (Intermediate)", line=dict(color="#E040FB", width=1.5)
                ))
            fig_price.update_layout(
                template="plotly_dark",
                xaxis_title="Session Date",
                yaxis_title=f"Price ({exchange_cfg.currency})",
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig_price, use_container_width=True)

            # Volatility and Drawdown Curves
            col_vol, col_dd = st.columns(2)
            with col_vol:
                st.subheader("21-Session Rolling Volatility")
                fig_vol = go.Figure()
                if "rolling_volatility_21" in df_ind.columns:
                    fig_vol.add_trace(go.Scatter(
                        x=df_ind["session_date"], y=df_ind["rolling_volatility_21"] * 100,
                        name="21-Session Vol %", line=dict(color="#FFD600", width=1.8), fill="tozeroy"
                    ))
                fig_vol.update_layout(
                    template="plotly_dark",
                    yaxis_title="Annualized Volatility (%)",
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_vol, use_container_width=True)

            with col_dd:
                st.subheader("Drawdown from Peak")
                fig_dd = go.Figure()
                if "drawdown" in df_ind.columns:
                    fig_dd.add_trace(go.Scatter(
                        x=df_ind["session_date"], y=df_ind["drawdown"] * 100,
                        name="Drawdown %", line=dict(color="#FF1744", width=1.8), fill="tozeroy"
                    ))
                fig_dd.update_layout(
                    template="plotly_dark",
                    yaxis_title="Drawdown (%)",
                    margin=dict(l=20, r=20, t=30, b=20),
                )
                st.plotly_chart(fig_dd, use_container_width=True)

            # Tail-Risk & Value-at-Risk (VaR) Analytics
            with st.expander("🛡️ Quantitative Tail-Risk & Value-at-Risk (VaR) Profile", expanded=True):
                from src.analytics.risk import compute_multi_horizon_risk_profile
                risk_profile = compute_multi_horizon_risk_profile(df_ind, horizons=[1, 5, 21])
                if risk_profile:
                    r1 = risk_profile.get("1_SESSION", {})
                    r5 = risk_profile.get("5_SESSIONS", {})
                    r21 = risk_profile.get("21_SESSIONS", {})

                    rc1, rc2, rc3, rc4 = st.columns(4)
                    rc1.metric("1-Day VaR (95% Cornish-Fisher)", f"{r1.get('var_95_cornish_fisher_pct', 0.0):.2f}%")
                    rc2.metric("1-Day CVaR (95% Expected Shortfall)", f"{r1.get('cvar_95_historical_pct', 0.0):.2f}%")
                    rc3.metric("5-Day VaR (95% Cornish-Fisher)", f"{r5.get('var_95_cornish_fisher_pct', 0.0):.2f}%")
                    rc4.metric("21-Day VaR (95% Cornish-Fisher)", f"{r21.get('var_95_cornish_fisher_pct', 0.0):.2f}%")

                    risk_rows = [
                        {
                            "Horizon": k,
                            "Gaussian VaR 95%": f"{v['var_95_gaussian_pct']:.2f}%",
                            "Cornish-Fisher VaR 95%": f"{v['var_95_cornish_fisher_pct']:.2f}%",
                            "Cornish-Fisher VaR 99%": f"{v['var_99_cornish_fisher_pct']:.2f}%",
                            "Historical VaR 95%": f"{v['var_95_historical_pct']:.2f}%",
                            "Expected Shortfall (CVaR 95%)": f"{v['cvar_95_historical_pct']:.2f}%",
                            "Expected Shortfall (CVaR 99%)": f"{v['cvar_99_historical_pct']:.2f}%",
                        }
                        for k, v in risk_profile.items()
                    ]
                    st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

                    # Macro Stress-Testing Simulation
                    from src.analytics.risk import simulate_macro_stress_scenarios
                    stress_res = simulate_macro_stress_scenarios(df_ind, position_value=10000.0)
                    if stress_res:
                        st.markdown("**Historical Macro Stress Scenarios ($10,000 Portfolio Base):**")
                        stress_df = pd.DataFrame([
                            {
                                "Macro Scenario": s["scenario_name"],
                                "Severity": s["severity"],
                                "Benchmark Shock": f"{s['benchmark_shock_pct']:+.1f}%",
                                "Estimated Loss": f"-{s['estimated_loss_pct']:.2f}%",
                                "Loss ($)": f"-${s['estimated_loss_dollars']:.2f}",
                                "Stressed Price": f"{s['post_stress_price']:.2f} {exchange_cfg.currency}",
                                "Historical Dynamic": s["description"],
                            }
                            for s in stress_res
                        ])
                        st.dataframe(stress_df, use_container_width=True, hide_index=True)

            # Out-of-Sample Quantitative Strategy Backtest Card
            with st.expander("📈 Out-of-Sample Strategy Backtest (Realized Sharpe & PnL Curve)", expanded=False):
                st.markdown(
                    "Simulates an expanding-window walk-forward execution strategy driven by the multi-horizon forecasting engine, "
                    "incorporating realistic transaction costs and evaluating risk-adjusted equity compounding."
                )
                col_bt1, col_bt2, col_bt3 = st.columns(3)
                with col_bt1:
                    bt_horizon = st.selectbox("Strategy Horizon", [1, 5, 21], format_func=lambda h: f"{h} Session(s)", key="bt_h")
                with col_bt2:
                    bt_sizing = st.selectbox(
                        "Position Sizing",
                        ["fixed", "vol_target", "half_kelly"],
                        format_func=lambda s: {
                            "fixed": "Fixed Binary (1.0x)",
                            "vol_target": "Volatility-Targeted",
                            "half_kelly": "Half-Kelly Sizing",
                        }[s],
                        key="bt_s",
                    )
                with col_bt3:
                    bt_fee = st.number_input("Transaction Fee (bps)", min_value=0.0, max_value=50.0, value=5.0, step=1.0, key="bt_f")

                bt_target_vol = 0.15
                if bt_sizing == "vol_target":
                    bt_target_vol = st.slider("Target Annualized Volatility (%)", min_value=5.0, max_value=40.0, value=15.0, step=1.0, key="bt_tv") / 100.0

                if st.button("🚀 Run Walk-Forward Strategy Backtest", use_container_width=True):
                    with st.spinner("Executing walk-forward strategy backtest..."):
                        from src.analytics.backtester import run_forecast_backtest
                        bt_res = run_forecast_backtest(
                            df_obs,
                            horizon_sessions=bt_horizon,
                            fee_bps=bt_fee,
                            min_train_sessions=min(120, max(30, len(df_obs) // 2)),
                            model_family=selected_model_family,
                            sizing_mode=bt_sizing,
                            target_annual_vol=bt_target_vol,
                        )
                        m = bt_res.metrics

                        bm1, bm2, bm3, bm4 = st.columns(4)
                        bm1.metric("Strategy Net Return", f"{m.total_return_pct:+.2f}%")
                        bm2.metric("Buy & Hold Return", f"{m.benchmark_return_pct:+.2f}%")
                        bm3.metric("Annualized Sharpe", f"{m.annualized_sharpe:.3f}")
                        bm4.metric("Avg Portfolio Exposure", f"{m.avg_exposure:.2f}x")

                        bm5, bm6, bm7, bm8 = st.columns(4)
                        bm5.metric("Max Drawdown", f"{m.max_drawdown_pct:.2f}%")
                        bm6.metric("Win Rate", f"{m.win_rate_pct:.1f}%")
                        bm7.metric("Profit Factor", f"{m.profit_factor:.2f}")
                        bm8.metric("Fee Drag Deducted", f"{m.fee_drag_pct:.2f}%")

                        fig_bt = go.Figure()
                        fig_bt.add_trace(go.Scatter(
                            x=bt_res.equity_df["session_date"],
                            y=bt_res.equity_df["strategy_equity"],
                            name=f"Strategy Equity ({selected_model_family.capitalize()} - {bt_sizing})",
                            line=dict(color="#00E676", width=2.5),
                        ))
                        fig_bt.add_trace(go.Scatter(
                            x=bt_res.equity_df["session_date"],
                            y=bt_res.equity_df["benchmark_equity"],
                            name="Buy & Hold Benchmark",
                            line=dict(color="#90CAF9", width=1.8, dash="dash"),
                        ))
                        fig_bt.update_layout(
                            template="plotly_dark",
                            title=f"Compounding Equity Curve (Base = $100.00 | Sizing: {bt_sizing.upper()})",
                            xaxis_title="Trading Session Date",
                            yaxis_title="Portfolio Equity ($)",
                            margin=dict(l=20, r=20, t=40, b=20),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_bt, use_container_width=True)

            # Cross-Market Lead-Lag & Cointegration Diagnostics
            with st.expander("🔗 Cross-Market Lead-Lag & Cointegration Diagnostics", expanded=False):
                st.markdown(
                    "Evaluate inter-market Granger causality (does an international lead market predict local moves?) "
                    "and Engle-Granger cointegration spread mean-reversion dynamics."
                )
                from src.data.db import get_all_stocks
                all_stocks_df = get_all_stocks()
                all_tickers = [t for t in all_stocks_df["ticker"].tolist() if t != selected_ticker]
                if all_tickers:
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        partner_ticker = st.selectbox("Select Partner Asset", options=all_tickers, key="coint_partner")
                    with col_p2:
                        max_lag_sel = st.slider("Granger Max Lag (Sessions)", 1, 10, 5, key="granger_max_lag")

                    if st.button("🔬 Compute Lead-Lag & Cointegration", use_container_width=True):
                        with st.spinner("Aligning multi-market trading sessions and testing dynamics..."):
                            from src.analytics.causality import evaluate_engle_granger_cointegration, evaluate_granger_causality
                            from src.data.db import get_observations

                            df_partner = get_observations(partner_ticker)
                            if df_partner.empty or len(df_partner) < 30:
                                st.warning(f"Insufficient history for {partner_ticker}.")
                            else:
                                c_obs = df_obs.sort_values("session_date").drop_duplicates("session_date")
                                c_prt = df_partner.sort_values("session_date").drop_duplicates("session_date")

                                ret_sel = c_obs[PRICE_COLUMN].astype(float).pct_change().dropna()
                                ret_sel.index = c_obs["session_date"].iloc[1:]
                                ret_prt = c_prt[PRICE_COLUMN].astype(float).pct_change().dropna()
                                ret_prt.index = c_prt["session_date"].iloc[1:]

                                p_sel = c_obs.set_index("session_date")[PRICE_COLUMN].astype(float)
                                p_prt = c_prt.set_index("session_date")[PRICE_COLUMN].astype(float)

                                g_lead = evaluate_granger_causality(ret_prt, ret_sel, max_lag=max_lag_sel, source_ticker=partner_ticker, target_ticker=selected_ticker)
                                g_lag = evaluate_granger_causality(ret_sel, ret_prt, max_lag=max_lag_sel, source_ticker=selected_ticker, target_ticker=partner_ticker)
                                coint_rec = evaluate_engle_granger_cointegration(p_sel, p_prt, ticker_1=selected_ticker, ticker_2=partner_ticker)

                                col_cg1, col_cg2 = st.columns(2)
                                with col_cg1:
                                    st.markdown(f"**Lead-Lag Direction: {partner_ticker} ➔ {selected_ticker}**")
                                    st.metric("Granger F-Stat", f"{g_lead.f_statistic:.2f}", delta=f"p={g_lead.p_value:.4f}")
                                    st.caption(f"Optimal Lag: {g_lead.optimal_lag} session(s) | {'✅ Statistically Significant (p < 0.05)' if g_lead.is_causal else '❌ Not Significant'}")
                                with col_cg2:
                                    st.markdown(f"**Reverse Direction: {selected_ticker} ➔ {partner_ticker}**")
                                    st.metric("Granger F-Stat", f"{g_lag.f_statistic:.2f}", delta=f"p={g_lag.p_value:.4f}")
                                    st.caption(f"Optimal Lag: {g_lag.optimal_lag} session(s) | {'✅ Statistically Significant (p < 0.05)' if g_lag.is_causal else '❌ Not Significant'}")

                                st.markdown("---")
                                st.markdown("**Engle-Granger Cointegration & Spread Mean-Reversion:**")
                                col_ci1, col_ci2, col_ci3, col_ci4 = st.columns(4)
                                col_ci1.metric("Hedge Ratio (β)", f"{coint_rec.hedge_ratio_beta:.4f}")
                                col_ci2.metric("ADF Spread Stat", f"{coint_rec.adf_statistic:.2f}", delta=f"5% Crit: {coint_rec.critical_values['5%']:.2f}")
                                col_ci3.metric("Half-Life", f"{coint_rec.half_life_sessions:.1f} sessions" if math.isfinite(coint_rec.half_life_sessions) else "Non-stationary")
                                col_ci4.metric("Spread Z-Score", f"{coint_rec.current_spread_zscore:+.2f} σ")

                                if coint_rec.is_cointegrated:
                                    st.success(f"⚖️ {coint_rec.summary}")
                                else:
                                    st.info(f"ℹ️ {coint_rec.summary}")

            # Market Microstructure & Liquidity Profile
            with st.expander("💧 Market Microstructure & Liquidity Profile", expanded=False):
                st.markdown(
                    "Evaluate high-frequency liquidity friction and price impact metrics derived from daily OHLCV quotes: "
                    "Amihud Illiquidity Ratio (price impact per $1M traded) and Corwin-Schultz High-Low bid-ask spread %."
                )
                from src.features.microstructure import evaluate_liquidity_profile
                liq_rec = evaluate_liquidity_profile(df_obs, ticker=selected_ticker)

                col_l1, col_l2, col_l3, col_l4 = st.columns(4)
                col_l1.metric("Corwin-Schultz Spread", f"{liq_rec.latest_corwin_schultz_spread_pct:.2f}%", delta=f"{liq_rec.mean_corwin_schultz_spread_pct:.2f}% mean")
                col_l2.metric("Amihud Illiquidity", f"{liq_rec.latest_amihud_illiq:.4f}", delta=f"{liq_rec.mean_amihud_illiq:.4f} mean")
                col_l3.metric("Daily Turnover ($)", f"${liq_rec.mean_dollar_volume:,.0f}")
                col_l4.metric("Liquidity Regime", liq_rec.liquidity_regime)

                if liq_rec.is_liquid:
                    st.success(f"💧 {liq_rec.summary}")
                else:
                    st.warning(f"⚠️ {liq_rec.summary}")

            # Multi-Estimator Realized Volatility Engine
            with st.expander("⚡ Multi-Estimator Realized Volatility & Extreme Value Engine", expanded=False):
                st.markdown(
                    r"Compare benchmark realized volatility estimators: **Yang-Zhang (2000)** (minimum-variance, jump & drift unbiased, ~14x CC efficiency), "
                    r"**Garman-Klass (1980)** (~7.4x efficiency), **Parkinson (1980)** (~5.2x efficiency), **RiskMetrics EWMA** ($\lambda=0.94$), and **Close-to-Close**."
                )

                from src.analytics.volatility import evaluate_volatility_profile
                vol_rec = evaluate_volatility_profile(df_obs, ticker=selected_ticker, window=20)

                col_v1, col_v2, col_v3, col_v4, col_v5 = st.columns(5)
                col_v1.metric("Yang-Zhang (Optimal)", f"{vol_rec.yang_zhang_ann_pct:.2f}%", delta=f"{vol_rec.volatility_ratio_yz_cc:.2f}x CC")
                col_v2.metric("Garman-Klass (OHLC)", f"{vol_rec.garman_klass_ann_pct:.2f}%")
                col_v3.metric("Parkinson (High-Low)", f"{vol_rec.parkinson_ann_pct:.2f}%")
                col_v4.metric("Close-to-Close", f"{vol_rec.close_to_close_ann_pct:.2f}%")
                col_v5.metric("EWMA (λ=0.94)", f"{vol_rec.ewma_ann_pct:.2f}%")

                st.info(f"⚡ **Volatility Regime:** `{vol_rec.regime_signal}` | **Sample Sessions:** {vol_rec.sample_sessions} | **Window:** {vol_rec.window} sessions")

                if vol_rec.term_structure_yang_zhang:
                    df_ts = pd.DataFrame([
                        {"Horizon": k, "Yang-Zhang Volatility (%)": v}
                        for k, v in vol_rec.term_structure_yang_zhang.items()
                    ])
                    st.dataframe(df_ts, use_container_width=True, hide_index=True)


    # =========================================================================
    # TAB 3: Multi-Horizon Price Forecasting
    # =========================================================================
    with tabs[2]:
        st.header(f"Multi-Horizon Forecasting Engine: {selected_ticker} ({selected_model_family})")
        df_obs = load_stock_observations(selected_ticker, selected_exchange)

        if df_obs.empty:
            st.warning(f"Please synchronize historical data for {selected_ticker} before generating forecasts.")
        else:
            # Load or fit multi-horizon model
            models_dir = PROJECT_ROOT / "models"
            forecaster = MultiHorizonForecaster(model_family=selected_model_family)
            model_loaded = False
            try:
                forecaster = MultiHorizonForecaster.load_models(models_dir, selected_ticker)
                if forecaster.model_family == selected_model_family:
                    model_loaded = True
            except Exception:
                model_loaded = False

            if not model_loaded:
                st.info(f"Trained {selected_model_family} model artifact not found for {selected_ticker}. Training on latest historical sessions...")
                with st.spinner(f"Training Multi-Horizon {selected_model_family} Models (1, 5, 21 sessions)..."):
                    forecaster = MultiHorizonForecaster(model_family=selected_model_family)
                    forecaster.fit(df_obs)
                    forecaster.save_models(models_dir, selected_ticker)
                    st.success("Models trained and artifacts saved!")

            # Generate future forecasts with 80% prediction interval
            forecasts = forecaster.predict_all_horizons(df_obs, selected_ticker, selected_exchange, confidence_level=0.80)
            latest_price = float(df_obs.iloc[-1][PRICE_COLUMN])
            latest_date = str(df_obs.iloc[-1]["session_date"])

            st.markdown(f"**Origin Session Date:** `{latest_date}` | **Origin Price:** `{latest_price:.2f} {exchange_cfg.currency}` | **Model:** `{selected_model_family}`")

            # 3 Horizon Cards
            col_h1, col_h5, col_h21 = st.columns(3)
            h1 = forecasts["1_SESSION"]
            h5 = forecasts["5_SESSIONS"]
            h21 = forecasts["21_SESSIONS"]

            with col_h1:
                st.metric(
                    label=f"Next Session (P_t+1: {h1['target_session_date']})",
                    value=f"{h1['predicted_price']:.2f} {exchange_cfg.currency}",
                    delta=f"{h1['expected_return_pct']:+.2f}%",
                )
                prob_1 = h1.get("directional_win_probability", 0.5) * 100.0
                st.caption(f"80% CI: `[{h1['lower_bound']:.2f} – {h1['upper_bound']:.2f}]`")
                st.caption(f"Win Prob: **{prob_1:.1f}%** ({h1.get('conviction_label', 'NEUTRAL')})")

            with col_h5:
                st.metric(
                    label=f"5 Sessions Ahead (P_t+5: {h5['target_session_date']})",
                    value=f"{h5['predicted_price']:.2f} {exchange_cfg.currency}",
                    delta=f"{h5['expected_return_pct']:+.2f}%",
                )
                prob_5 = h5.get("directional_win_probability", 0.5) * 100.0
                st.caption(f"80% CI: `[{h5['lower_bound']:.2f} – {h5['upper_bound']:.2f}]`")
                st.caption(f"Win Prob: **{prob_5:.1f}%** ({h5.get('conviction_label', 'NEUTRAL')})")

            with col_h21:
                st.metric(
                    label=f"21 Sessions Ahead (P_t+21: {h21['target_session_date']})",
                    value=f"{h21['predicted_price']:.2f} {exchange_cfg.currency}",
                    delta=f"{h21['expected_return_pct']:+.2f}%",
                )
                prob_21 = h21.get("directional_win_probability", 0.5) * 100.0
                st.caption(f"80% CI: `[{h21['lower_bound']:.2f} – {h21['upper_bound']:.2f}]`")
                st.caption(f"Win Prob: **{prob_21:.1f}%** ({h21.get('conviction_label', 'NEUTRAL')})")

            st.markdown("---")

            # Forward Projection Chart with Uncertainty Interval Cone
            st.subheader("Forward Price Trajectory & Calibrated Prediction Intervals")
            recent_df = df_obs.tail(40)
            fig_proj = go.Figure()
            # Historical Path
            fig_proj.add_trace(go.Scatter(
                x=recent_df["session_date"],
                y=recent_df[PRICE_COLUMN],
                mode="lines+markers",
                name="Observed Price History",
                line=dict(color="#90CAF9", width=2),
            ))

            # Forward Projection Line and 80% Interval Cone
            proj_dates = [latest_date, h1["target_session_date"], h5["target_session_date"], h21["target_session_date"]]
            proj_prices = [latest_price, h1["predicted_price"], h5["predicted_price"], h21["predicted_price"]]
            upper_prices = [latest_price, h1["upper_bound"], h5["upper_bound"], h21["upper_bound"]]
            lower_prices = [latest_price, h1["lower_bound"], h5["lower_bound"], h21["lower_bound"]]

            # Upper bound line (invisible)
            fig_proj.add_trace(go.Scatter(
                x=proj_dates,
                y=upper_prices,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))

            # Lower bound line with shaded fill to upper
            fig_proj.add_trace(go.Scatter(
                x=proj_dates,
                y=lower_prices,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(255, 214, 0, 0.18)",
                name="80% Prediction Interval",
            ))

            # Forecast centerline
            fig_proj.add_trace(go.Scatter(
                x=proj_dates,
                y=proj_prices,
                mode="lines+markers",
                name=f"Direct Forecast ({selected_model_family})",
                line=dict(color="#FFD600", width=2.5, dash="dash"),
                marker=dict(size=[6, 10, 10, 12], symbol="diamond"),
            ))

            # Cornish-Fisher VaR 95% downside floor
            try:
                from src.analytics.risk import compute_multi_horizon_risk_profile
                risk_summary = compute_multi_horizon_risk_profile(df_obs, horizons=[1, 5, 21])
                if risk_summary:
                    v1 = risk_summary.get("1_SESSION", {}).get("var_95_cornish_fisher_pct", 0.0) / 100.0
                    v5 = risk_summary.get("5_SESSIONS", {}).get("var_95_cornish_fisher_pct", 0.0) / 100.0
                    v21 = risk_summary.get("21_SESSIONS", {}).get("var_95_cornish_fisher_pct", 0.0) / 100.0
                    floor_prices = [
                        latest_price,
                        latest_price * (1.0 - v1),
                        latest_price * (1.0 - v5),
                        latest_price * (1.0 - v21),
                    ]
                    fig_proj.add_trace(go.Scatter(
                        x=proj_dates,
                        y=floor_prices,
                        mode="lines+markers",
                        name="VaR 95% Downside Floor (Cornish-Fisher)",
                        line=dict(color="#FF5252", width=1.8, dash="dot"),
                        marker=dict(size=6, symbol="x"),
                    ))
            except Exception:
                pass

            fig_proj.update_layout(
                template="plotly_dark",
                xaxis_title="Trading Session Date",
                yaxis_title=f"Price ({exchange_cfg.currency})",
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig_proj, use_container_width=True)

            # Interactive Signal Attribution Explainer
            with st.expander("🔍 Explainable AI: Quantitative Signal Attribution Breakdown", expanded=False):
                st.markdown(
                    "Inspect the local additive drivers contributing to the direct expected return for each horizon. "
                    "Green indicates positive bullish contribution; red indicates bearish drag."
                )
                tab_h1, tab_h5, tab_h21 = st.tabs(["Next Session (1D)", "1-Week (5D)", "1-Month (21D)"])
                for tab_obj, h_k in zip([tab_h1, tab_h5, tab_h21], ["1_SESSION", "5_SESSIONS", "21_SESSIONS"]):
                    with tab_obj:
                        attrs = forecasts.get(h_k, {}).get("signal_attributions", {})
                        if attrs:
                            sorted_signals = sorted(
                                [(k, v) for k, v in attrs.items() if k != "intercept"],
                                key=lambda x: abs(x[1]),
                                reverse=True,
                            )
                            fig_attr = go.Figure()
                            names = [s[0] for s in sorted_signals]
                            vals = [s[1] * 100.0 for s in sorted_signals]
                            colors = ["#00E676" if v >= 0 else "#FF5252" for v in vals]

                            fig_attr.add_trace(go.Bar(
                                x=vals,
                                y=names,
                                orientation="h",
                                marker=dict(color=colors),
                                text=[f"{v:+.2f}%" for v in vals],
                                textposition="outside",
                            ))
                            fig_attr.update_layout(
                                template="plotly_dark",
                                title=f"Additive Signal Contributions to Expected Return ({h_k})",
                                xaxis_title="Contribution to Expected Return (%)",
                                yaxis=dict(autorange="reversed"),
                                margin=dict(l=120, r=40, t=40, b=30),
                                height=380,
                            )
                            st.plotly_chart(fig_attr, use_container_width=True)
                        else:
                            st.info("Signal attributions not available for this forecast.")

            # Action: Register Forecast in Audit Log
            col_reg, _ = st.columns([2, 4])
            with col_reg:
                if st.button("📝 Register These Forecasts in Audit Log", use_container_width=True):
                    tracker = PredictionTracker()
                    for h_key, fc in forecasts.items():
                        tracker.record_prediction(
                            ticker=selected_ticker,
                            exchange=selected_exchange,
                            forecast_origin_session=latest_date,
                            origin_price=latest_price,
                            horizon_sessions=fc["horizon_sessions"],
                            target_session_date=fc["target_session_date"],
                            predicted_price=fc["predicted_price"],
                            model_name=f"{selected_model_family.capitalize()}_Direct_{h_key}",
                            model_version="1.0.0",
                            lower_bound=fc.get("lower_bound"),
                            upper_bound=fc.get("upper_bound"),
                            confidence_level=fc.get("confidence_level"),
                        )
                    st.success("All 3 horizon forecasts registered to prediction audit database!")
                    st.cache_data.clear()

    # =========================================================================
    # TAB 4: Prediction Audit Log & Reconciled Evaluation
    # =========================================================================
    with tabs[3]:
        st.header("Prediction Audit Log & Evaluation")
        st.caption("Auditable tracking finite state machine reconciling predictions with actual settlement prices.")

        # Cryptographic Audit Verification & Database Backup Box
        with st.expander("🛡️ Unified Cryptographic Security & Health Pre-Flight Audit", expanded=True):
            from src.data.db import backup_database, verify_database_integrity
            from src.security.health import run_system_health_audit

            health_rec = run_system_health_audit()

            col_h1, col_h2, col_h3, col_h4 = st.columns(4)
            col_h1.metric("Database B-Trees", health_rec.database_integrity)
            col_h2.metric("Foreign Keys", health_rec.foreign_keys_integrity)
            col_h3.metric("Prediction Audit", health_rec.prediction_audit)
            col_h4.metric("Model Signatures", health_rec.model_signatures)

            if health_rec.is_healthy:
                st.success(f"🔒 High Assurance Security Seal: {health_rec.summary}")
            else:
                st.error(f"⚠️ Security Health Alert: {health_rec.summary}")

            from src.data.rate_limiter import get_quota_tracker
            tracker_td = get_quota_tracker("TwelveData")
            q_tel = tracker_td.get_telemetry()
            st.markdown(
                f"**Provider Rate Limiter Telemetry (Twelve Data):** "
                f"Requests (Last 60s): `{q_tel.requests_last_minute}/{q_tel.max_requests_per_minute}` | "
                f"Quota Remaining: `{q_tel.quota_remaining}` | "
                f"Total Invocations: `{q_tel.total_requests}` | "
                f"Failover Count: `{q_tel.failover_count}`"
            )

            col_bk1, col_bk2 = st.columns(2)
            with col_bk1:
                if st.button("📦 Create Atomic Database Backup", use_container_width=True):
                    with st.spinner("Executing non-blocking online SQLite backup..."):
                        bk_path = backup_database()
                        st.success(f"Snapshot created: `{bk_path.name}` ({bk_path.stat().st_size} bytes)")
            with col_bk2:
                if st.button("🧹 Vacuum Database & Compact Storage", use_container_width=True):
                    with st.spinner("Defragmenting B-trees and reclaiming disk pages..."):
                        from src.data.db import vacuum_database
                        v_res = vacuum_database()
                        st.success(f"Compacted! Reclaimed {v_res['bytes_reclaimed']} bytes. Integrity: {v_res['integrity']}.")

        # Action: Reconcile Pending Button
        col_act, col_filter = st.columns([2, 3])
        with col_act:
            if st.button("⚖️ Run Session-Aware Reconciliation Daemon", use_container_width=True):
                with st.spinner("Checking closed exchange sessions and reconciling prices..."):
                    from src.evaluation.daemon import EvaluationDaemon
                    daemon = EvaluationDaemon()
                    res = daemon.run_once()
                    st.cache_data.clear()
                    m = res["reconciliation_metrics"]
                    st.toast(
                        f"Reconciliation complete: {m.get('evaluated', 0)} evaluated, {m.get('awaiting', 0)} awaiting, {m.get('pending', 0)} pending future.",
                        icon="⚖️"
                    )

        with col_filter:
            status_filter = st.selectbox(
                "Filter by Lifecycle Status",
                options=["ALL", "EVALUATED", "AWAITING_ACTUAL", "PENDING", "FAILED"],
            )

        # Load Predictions Table
        df_preds = load_predictions_data(
            status=None if status_filter == "ALL" else status_filter,
            ticker=None,
        )

        if df_preds.empty:
            st.info("No predictions recorded yet. Generate and register forecasts in the 'Multi-Horizon Forecasts' tab.")
        else:
            # KPI Metric Summary
            total_count = len(df_preds)
            eval_count = (df_preds["status"] == "EVALUATED").sum()
            await_count = (df_preds["status"] == "AWAITING_ACTUAL").sum()
            pend_count = (df_preds["status"] == "PENDING").sum()

            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Predictions", total_count)
            k2.metric("Evaluated", int(eval_count))
            k3.metric("Awaiting Settlement", int(await_count))
            k4.metric("Pending Future Sessions", int(pend_count))

            # Information Coefficient & Alpha Decay Diagnostic Box
            evaluated_subset = df_preds[df_preds["status"] == "EVALUATED"]
            if not evaluated_subset.empty:
                with st.expander("📈 Rolling Information Coefficient (IC) & Alpha Decay Monitor", expanded=False):
                    from src.analytics.drift import calculate_information_coefficient_drift
                    ic_receipt = calculate_information_coefficient_drift(evaluated_subset)

                    ic_col1, ic_col2, ic_col3, ic_col4 = st.columns(4)
                    ic_col1.metric("Mean Rank IC (Spearman)", f"{ic_receipt.mean_rank_ic:+.4f}")
                    ic_col2.metric("Trailing Rank IC", f"{ic_receipt.trailing_rank_ic:+.4f}")
                    ic_col3.metric("Information Ratio (IR)", f"{ic_receipt.information_ratio:+.2f}")
                    ic_col4.metric("Alpha State", ic_receipt.decay_status)

                    if ic_receipt.is_decaying:
                        st.warning(f"⚠️ {ic_receipt.decay_warning}")
                    else:
                        st.success(f"✅ Forecasting skill stable across {ic_receipt.sample_size} evaluated predictions (t-stat: {ic_receipt.t_statistic:+.2f}).")

            # Format Display Table
            disp_df = df_preds.copy()
            disp_df["predicted_price"] = disp_df["predicted_price"].apply(lambda v: f"{float(v):.2f}")
            disp_df["actual_price"] = disp_df["actual_price"].apply(
                lambda v: f"{float(v):.2f}" if pd.notna(v) else "Pending"
            )
            disp_df["percentage_error"] = disp_df["percentage_error"].apply(
                lambda v: f"{float(v):.2f}%" if pd.notna(v) else "-"
            )
            disp_df["direction_correct"] = disp_df["direction_correct"].apply(
                lambda v: "✓ Correct" if pd.notna(v) and bool(v) else ("✗ Incorrect" if pd.notna(v) else "-")
            )

            cols_to_show = [
                "ticker", "exchange", "model_name", "horizon_sessions",
                "forecast_origin_session", "target_session_date",
                "predicted_price", "lower_bound", "upper_bound", "confidence_level",
                "actual_price", "percentage_error", "direction_correct", "status"
            ]
            st.dataframe(
                disp_df[[c for c in cols_to_show if c in disp_df.columns]],
                use_container_width=True,
                hide_index=True,
            )

    # =========================================================================
    # TAB 5: Model Performance & Walk-Forward Validation
    # =========================================================================
    with tabs[4]:
        st.header(f"Model Performance & Validation: {selected_ticker} ({selected_model_family})")
        st.caption("Chronological expanding-window walk-forward backtest results vs persistence baseline.")

        df_obs = load_stock_observations(selected_ticker, selected_exchange)
        if df_obs.empty:
            st.warning(f"No historical observations available for {selected_ticker}.")
        else:
            forecaster = MultiHorizonForecaster(model_family=selected_model_family)
            with st.spinner(f"Computing Walk-Forward Validation metrics ({selected_model_family})..."):
                validation_summary = forecaster.evaluate_walk_forward_all(df_obs, min_train_sessions=120)

            # Performance Comparison Table
            st.subheader(f"Walk-Forward Validation Summary ({selected_model_family})")
            val_rows = []
            for h_key, res in validation_summary.items():
                m_m = res["model_metrics"]
                b_m = res["baseline_metrics"]
                val_rows.append({
                    "Horizon": h_key,
                    "Model MAE": f"{m_m['mae']:.4f}",
                    "Baseline MAE": f"{b_m['mae']:.4f}",
                    "MAE Improvement %": f"{res['mae_improvement_pct']:+.2f}%",
                    "Model RMSE": f"{m_m['rmse']:.4f}",
                    "Baseline RMSE": f"{b_m['rmse']:.4f}",
                    "Directional Accuracy": f"{m_m['directional_accuracy_pct']:.2f}%",
                    "Information Ratio": f"{m_m.get('information_ratio', 0.0):.3f}",
                    "Max Error": f"{m_m.get('max_error', 0.0):.4f}",
                    "Folds Evaluated": res["walk_forward_steps"],
                })

            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

            # Conformal Prediction & Interval Calibration Box
            from src.analytics.calibration import evaluate_interval_calibration
            from src.data.db import get_predictions
            evaluated_preds = get_predictions(ticker=selected_ticker, status="EVALUATED")
            if not evaluated_preds.empty:
                calib_rec = evaluate_interval_calibration(evaluated_preds, confidence_level=0.80)
                if calib_rec.sample_size >= 5:
                    st.markdown("---")
                    st.subheader("🎯 Prediction Interval Calibration & Winkler Sharpness")
                    st.caption("Empirical coverage and sharpness parsimony audit on reconciled out-of-sample predictions.")
                    c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                    c_col1.metric("Empirical Coverage", f"{calib_rec.empirical_coverage_pct:.1f}%", delta=f"{calib_rec.empirical_coverage_pct - 80.0:+.1f}% vs 80% nominal")
                    c_col2.metric("Mean Width", f"{calib_rec.mean_interval_width:.2f}")
                    c_col3.metric("Mean Winkler Score", f"{calib_rec.mean_winkler_score:.2f}")
                    c_col4.metric("Calibration Status", calib_rec.calibration_status)

                # Probabilistic Quality & CRPS Scorecard
                from src.evaluation.probabilistic import evaluate_probabilistic_forecasts
                prob_rec = evaluate_probabilistic_forecasts(evaluated_preds, nominal_confidence=0.80)
                if prob_rec.sample_size >= 3:
                    st.markdown("---")
                    st.subheader("📊 Probabilistic Quality & CRPS Scorecard")
                    st.caption("Proper scoring rules penalizing predictive dispersion and tail inaccuracy.")
                    p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                    p_col1.metric("CRPS", f"{prob_rec.mean_crps:.4f}", delta=f"{prob_rec.mean_crps - prob_rec.mean_mae:+.4f} vs MAE")
                    p_col2.metric("Pinball 10% (Downside)", f"{prob_rec.mean_pinball_loss_q10:.4f}")
                    p_col3.metric("Pinball 50% (Median)", f"{prob_rec.mean_pinball_loss_q50:.4f}")
                    p_col4.metric("Pinball 90% (Upside)", f"{prob_rec.mean_pinball_loss_q90:.4f}")
                    st.info(f"📊 **Probabilistic Scoring Audit:** {prob_rec.summary}")

            st.markdown("---")


            # Quantitative Feature Importance
            st.subheader(f"Quantitative Feature Signals ({selected_model_family})")
            st.caption("Feature signals learned by direct models across forecasting horizons.")

            # Train models to inspect weights
            forecaster.fit(df_obs)
            horizon_choice = st.selectbox("Select Horizon for Feature Breakdown", options=["1_SESSION", "5_SESSIONS", "21_SESSIONS"])
            coefs = forecaster.get_feature_importances(horizon_choice)

            df_coefs = pd.DataFrame([
                {"Feature": k, "Weight": v} for k, v in coefs.items()
            ]).sort_values("Weight", key=abs, ascending=True)

            fig_coef = go.Figure()
            colors = ["#00E676" if w >= 0 else "#FF1744" for w in df_coefs["Weight"]]
            fig_coef.add_trace(go.Bar(
                y=df_coefs["Feature"],
                x=df_coefs["Weight"],
                orientation="h",
                marker=dict(color=colors),
            ))
            fig_coef.update_layout(
                template="plotly_dark",
                xaxis_title="Standardized Ridge Regression Coefficient (Beta)",
                yaxis_title="Quantitative Feature Signal",
                margin=dict(l=20, r=20, t=30, b=20),
            )
            st.plotly_chart(fig_coef, use_container_width=True)


if __name__ == "__main__":
    render_dashboard()
