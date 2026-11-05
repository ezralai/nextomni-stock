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
        options=["stacking", "adaptive_ensemble", "ensemble", "gradient_boosting", "linear"],
        format_func=lambda x: {
            "stacking": "🏆 Stacking Meta-Learner (Out-of-Fold Super Learner)",
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
                                    ["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp", "cvar_min", "cvar_parity"],
                                    format_func=lambda m: {
                                        "max_sharpe": "Maximum Sharpe Ratio",
                                        "min_variance": "Minimum Variance",
                                        "risk_parity": "Equal Risk Contribution (Risk Parity)",
                                        "equal_weight": "Equal Weighting (1/N)",
                                        "hrp": "Hierarchical Risk Parity (HRP)",
                                        "cvar_min": "Minimum Expected Shortfall (CVaR 95%)",
                                        "cvar_parity": "Tail Risk Parity (Equal CVaR Contribution)",
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

                            p_kpi1, p_kpi2, p_kpi3, p_kpi4, p_kpi5 = st.columns(5)
                            p_kpi1.metric("Exp Annual Return", f"{port_rec.expected_annual_return_pct:+.2f}%")
                            p_kpi2.metric("Annualized Volatility", f"{port_rec.annualized_volatility_pct:.2f}%")
                            p_kpi3.metric("Portfolio Sharpe", f"{port_rec.sharpe_ratio:+.2f}")
                            p_kpi4.metric("LW Shrinkage (δ)", f"{port_rec.shrinkage_intensity:.4f}")
                            p_kpi5.metric("95% CVaR (Tail Loss)", f"{port_rec.cvar_95_pct:.2f}%")

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

                            # Asymmetric Lower-Tail Dependence & Joint Crash Fragility
                            st.markdown("---")
                            st.subheader("📉 Asymmetric Lower-Tail Dependence Matrix (λ_L) & Joint Crash Fragility")
                            st.caption(
                                "Empirical quantile co-exceedance and Clayton copula lower-tail dependence (q=5%). "
                                "Measures extreme downside co-crash probability where diversification breaks down."
                            )
                            from src.analytics.copula import evaluate_universe_tail_dependence
                            tail_rec = evaluate_universe_tail_dependence(tickers=list(port_rec.weights.keys()), threshold=0.05)
                            if tail_rec.sample_sessions >= 15:
                                tk1, tk2, tk3 = st.columns(3)
                                tk1.metric("Market Joint Crash State", tail_rec.market_joint_crash_regime)
                                p1, p2, val = tail_rec.highest_risk_pair
                                tk2.metric("Highest Co-Crash Pair", f"{p1} & {p2}", delta=f"λ_L = {val * 100:.1f}%")
                                avg_fragility = float(np.mean(list(tail_rec.systemic_fragility_scores.values()))) if tail_rec.systemic_fragility_scores else 0.0
                                tk3.metric("Mean Tail Dependence", f"{avg_fragility * 100:.1f}%")

                                disp_tail = tail_rec.tail_dependence_matrix.copy()
                                if hasattr(disp_tail, "map"):
                                    disp_tail = disp_tail.map(lambda v: f"{v * 100:.1f}%")
                                else:
                                    disp_tail = disp_tail.applymap(lambda v: f"{v * 100:.1f}%")
                                st.dataframe(disp_tail, use_container_width=True)
                                st.caption(f"🛡️ **Tail Dependence Summary:** {tail_rec.summary}")

                            # Michaud (1998) Resampled Efficiency & Robust Frontier
                            st.markdown("---")
                            st.subheader("🎯 Michaud (1998) Resampled Efficient Frontier")
                            st.caption(
                                "Monte Carlo resampling across estimation uncertainty to eliminate Markowitz 'error maximization'. "
                                "Produces robust, turnover-dampened asset allocations."
                            )
                            from src.analytics.resampled_portfolio import evaluate_resampled_frontier_profile
                            try:
                                resampled_rec = evaluate_resampled_frontier_profile(
                                    tickers=list(port_rec.weights.keys()),
                                    num_simulations=50,
                                    num_frontier_points=12,
                                )
                                rf_c1, rf_c2, rf_c3, rf_c4 = st.columns(4)
                                rf_c1.metric("Resampled Max Sharpe", f"{resampled_rec.resampled_max_sharpe.sharpe_ratio:.3f}")
                                rf_c2.metric("Classical Max Sharpe", f"{resampled_rec.classical_max_sharpe.sharpe_ratio:.3f}")
                                rf_c3.metric("Diversification Gain", f"{resampled_rec.michaud_diversification_gain_pct:+.1f}%")
                                rf_c4.metric("Effective Assets", f"{resampled_rec.resampled_max_sharpe.effective_assets:.2f}")

                                # Dual Frontier Chart
                                c_vols = [p.volatility_ann_pct for p in resampled_rec.classical_frontier]
                                c_rets = [p.expected_return_ann_pct for p in resampled_rec.classical_frontier]
                                r_vols = [p.volatility_ann_pct for p in resampled_rec.resampled_frontier]
                                r_rets = [p.expected_return_ann_pct for p in resampled_rec.resampled_frontier]

                                fig_frontier = go.Figure()
                                fig_frontier.add_trace(go.Scatter(
                                    x=c_vols, y=c_rets, mode="lines+markers",
                                    name="Classical Markowitz Frontier",
                                    line=dict(color="#EF4444", width=2, dash="dash"),
                                    marker=dict(size=6),
                                ))
                                fig_frontier.add_trace(go.Scatter(
                                    x=r_vols, y=r_rets, mode="lines+markers",
                                    name="Michaud Resampled Frontier",
                                    line=dict(color="#10B981", width=3),
                                    marker=dict(size=7),
                                ))
                                fig_frontier.add_trace(go.Scatter(
                                    x=[resampled_rec.resampled_max_sharpe.volatility_ann_pct],
                                    y=[resampled_rec.resampled_max_sharpe.expected_return_ann_pct],
                                    mode="markers+text",
                                    name="Resampled Max Sharpe",
                                    text=["Resampled Max Sharpe"],
                                    textposition="top left",
                                    marker=dict(color="#3B82F6", size=12, symbol="star"),
                                ))
                                fig_frontier.update_layout(
                                    title="Classical Markowitz vs. Michaud Resampled Efficient Frontier",
                                    xaxis_title="Annualized Volatility (%)",
                                    yaxis_title="Expected Annual Return (%)",
                                    template="plotly_dark",
                                    height=380,
                                    margin=dict(l=40, r=40, t=40, b=40),
                                )
                                st.plotly_chart(fig_frontier, use_container_width=True)
                                st.caption(f"🛡️ **Resampled Robustness Summary:** {resampled_rec.summary}")
                            except Exception as ex:
                                st.info(f"Resampled frontier profile deferred: {ex}")

                            # Marchenko-Pastur (1967) Random Matrix Theory Covariance Denoising
                            st.markdown("---")
                            st.subheader("🔬 Marchenko-Pastur (1967) RMT Covariance Denoising")
                            st.caption(
                                "Random Matrix Theory spectral filtering to eliminate in-sample noise eigenvalues "
                                "and dramatically stabilize multi-asset covariance matrices (López de Prado, 2018)."
                            )
                            try:
                                from src.analytics.rmt_denoiser import evaluate_rmt_denoising_profile
                                rmt_rec, raw_cov_df, denoised_cov_df = evaluate_rmt_denoising_profile(
                                    tickers=list(port_rec.weights.keys()),
                                    lookback_sessions=252,
                                )
                                rmt_c1, rmt_c2, rmt_c3, rmt_c4 = st.columns(4)
                                rmt_c1.metric("Signal Modes", f"{rmt_rec.num_signal_eigenvalues} / {rmt_rec.num_assets}", help="Eigenvalues exceeding Marchenko-Pastur Lambda+")
                                rmt_c2.metric("Signal Variance", f"{rmt_rec.signal_variance_ratio_pct:.1f}%")
                                rmt_c3.metric("Noise Ceiling (λ+)", f"{rmt_rec.noise_cutoff_lambda_plus:.3f}")
                                rmt_c4.metric("Condition Impr.", f"{rmt_rec.condition_number_improvement_ratio:.1f}x", delta=f"{rmt_rec.condition_number_raw:.1f} → {rmt_rec.condition_number_denoised:.1f}")

                                st.caption(f"🛡️ **RMT Stability Summary:** {rmt_rec.summary}")
                            except Exception as rmt_ex:
                                st.info(f"RMT covariance denoising deferred: {rmt_ex}")

                            # Black-Litterman (1992) Bayesian Portfolio Optimization & Machine Learning View Integration
                            st.markdown("---")
                            st.subheader("⚖️ Black-Litterman (1992) Bayesian Portfolio Optimization")
                            st.caption(
                                "Combines CAPM implied equilibrium market priors with active quantitative/model views. "
                                "Resolves Markowitz extreme corner solutions and yields stable, robust asset allocations."
                            )
                            try:
                                from src.analytics.black_litterman import evaluate_black_litterman_portfolio
                                bl_rec = evaluate_black_litterman_portfolio(
                                    universe=list(port_rec.weights.keys()),
                                    tau=0.05,
                                    risk_aversion_delta=2.5,
                                    lookback_window=120,
                                    denoise_cov=True,
                                )
                                bl_c1, bl_c2, bl_c3, bl_c4 = st.columns(4)
                                bl_c1.metric("Posterior Return", f"{bl_rec.expected_portfolio_return_ann_pct:+.2f}% Ann", delta=f"{bl_rec.expected_portfolio_return_ann_pct - bl_rec.prior_market_return_ann_pct:+.2f}% vs CAPM")
                                bl_c2.metric("Posterior Volatility", f"{bl_rec.expected_portfolio_vol_ann_pct:.2f}% Ann")
                                bl_c3.metric("Posterior Sharpe", f"{bl_rec.sharpe_ratio:+.2f}")
                                bl_c4.metric("RMT Covariance", "Denoised" if bl_rec.is_rmt_denoised else "Standard", help="Marchenko-Pastur spectral filtering applied")

                                # Allocation comparison table
                                bl_alloc_df = pd.DataFrame([
                                    {
                                        "Ticker": a["ticker"],
                                        "Market Wgt": f"{a['market_weight'] * 100:.1f}%",
                                        "Prior Ret (Π)": f"{a['prior_return_ann_pct']:+.2f}%",
                                        "Posterior Ret (μ)": f"{a['posterior_return_ann_pct']:+.2f}%",
                                        "BL Optimal Wgt": f"{a['optimal_weight'] * 100:.1f}%",
                                        "Shift (Δw)": f"{a['weight_shift'] * 100:+.1f}%",
                                    }
                                    for a in bl_rec.allocations
                                ])
                                st.dataframe(bl_alloc_df, use_container_width=True, hide_index=True)
                                st.caption(f"🛡️ **Black-Litterman Summary:** {bl_rec.summary}")
                            except Exception as bl_ex:
                                st.info(f"Black-Litterman portfolio optimization deferred: {bl_ex}")

                            # Fama & MacBeth (1973) Two-Pass Cross-Sectional Asset Pricing Engine
                            st.markdown("---")
                            st.subheader("🏛️ Fama & MacBeth (1973) Two-Pass Cross-Sectional Asset Pricing")
                            st.caption(
                                "Two-pass cross-sectional regression across universe assets to test multi-factor pricing "
                                "and estimate factor risk premia (λ_k) with Shanken (1992) errors-in-variables corrections."
                            )
                            if len(port_rec.weights.keys()) >= 4:
                                try:
                                    from src.analytics.fama_macbeth import evaluate_fama_macbeth_profile
                                    fm_rec = evaluate_fama_macbeth_profile(
                                        universe=list(port_rec.weights.keys()),
                                        benchmark_ticker="SPY",
                                        lookback_window=120,
                                    )
                                    fm_c1, fm_c2, fm_c3, fm_c4 = st.columns(4)
                                    fm_c1.metric("Cross-Sectional R²", f"{fm_rec.cross_sectional_r_squared * 100:.1f}%")
                                    fm_c2.metric("Pricing RMSE", f"{fm_rec.pricing_error_rmse_ann_pct:.2f}% Ann", help="Root mean square unexplained pricing error alpha")
                                    status_badge = "❌ Rejected" if fm_rec.is_model_rejected_5pct else "✅ Supported"
                                    fm_c3.metric("Model Status", status_badge, help="Wald test of joint pricing error significance")
                                    n_sig = sum(1 for p in fm_rec.factor_premia if p["is_significant_5pct"])
                                    fm_c4.metric("Priced Factors", f"{n_sig} / {len(fm_rec.factor_premia)}", help="Factors with Shanken p < 0.05")

                                    # Factor risk premia table
                                    fm_df = pd.DataFrame([
                                        {
                                            "Factor": p["factor_name"],
                                            "Risk Premium (λ)": f"{p['risk_premium_ann_pct']:+.2f}%/yr",
                                            "t-Stat (FM)": f"{p['t_stat_fm']:+.2f}",
                                            "t-Stat (Shanken)": f"{p['t_stat_shanken']:+.2f}",
                                            "p-Value": f"{p['p_value_shanken']:.4f}",
                                            "Priced (5%)": "YES" if p["is_significant_5pct"] else "no",
                                        }
                                        for p in fm_rec.factor_premia
                                    ])
                                    st.dataframe(fm_df, use_container_width=True, hide_index=True)
                                    st.caption(f"🛡️ **Fama-MacBeth Summary:** {fm_rec.summary}")
                                except Exception as fm_ex:
                                    st.info(f"Fama-MacBeth cross-sectional asset pricing deferred: {fm_ex}")
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
                    # Path-Dependent Drawdown & Conditional Drawdown-at-Risk (CDaR)
                    from src.analytics.risk import compute_path_risk_profile
                    path_profile = compute_path_risk_profile(df_obs, ticker=selected_ticker)
                    if path_profile.sample_sessions >= 5:
                        st.markdown("---")
                        st.markdown("**Path-Dependent Cumulative Drawdown & Conditional Drawdown-at-Risk (CDaR):**")
                        p_col1, p_col2, p_col3, p_col4 = st.columns(4)
                        p_col1.metric("Historical Max Drawdown", f"{path_profile.max_drawdown_pct:.2f}%", delta=f"{path_profile.current_drawdown_pct:.2f}% current")
                        p_col2.metric("95% CDaR (Chekhlov et al.)", f"{path_profile.cdar_95_pct:.2f}%", delta=f"95% DaR: {path_profile.dar_95_pct:.2f}%")
                        p_col3.metric("Peter Martin's Ulcer Index", f"{path_profile.ulcer_index_pct:.2f}%")
                        p_col4.metric("Underwater Duration", f"{path_profile.current_underwater_sessions} sessions", delta=f"Max: {path_profile.max_underwater_sessions} sessions")
                        st.caption(f"🛡️ **Path Risk Summary:** {path_profile.summary}")

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

                                # Online Kalman Filter Dynamic Pairs Tracker
                                st.markdown("---")
                                st.markdown("**Online Kalman Filter Dynamic Hedge Ratio Tracking (Chan 2013):**")
                                from src.analytics.kalman import evaluate_kalman_pairs_profile
                                kal_rec = evaluate_kalman_pairs_profile(
                                    y_ticker=selected_ticker,
                                    x_ticker=partner_ticker,
                                    df_y=c_obs,
                                    df_x=c_prt,
                                    delta=1e-4,
                                )
                                if kal_rec.sample_sessions >= 20:
                                    col_k1, col_k2, col_k3, col_k4 = st.columns(4)
                                    col_k1.metric("Dynamic Beta (β_t)", f"{kal_rec.latest_beta:.4f}", delta=f"{kal_rec.beta_drift_pct:+.1f}% vs OLS")
                                    col_k2.metric("Spread Error (e_t)", f"{kal_rec.latest_spread:+.2f}")
                                    col_k3.metric("Adaptive Z-Score", f"{kal_rec.latest_zscore:+.2f} σ")
                                    col_k4.metric("Arbitrage Signal", kal_rec.current_signal)

                                    if kal_rec.spread_history:
                                        k_df = pd.DataFrame(kal_rec.spread_history)
                                        st.caption("Trailing Dynamic Beta (β_t) and Spread Z-Score:")
                                        st.line_chart(k_df.set_index("session_date")[["beta", "zscore"]])
                                    st.caption(fr"🤖 **Kalman State:** Half-life = {kal_rec.mean_reversion_half_life:.1f} sessions | Process noise $\delta = {kal_rec.delta_process_noise:.1e}$ | **HMAC Digest:** `{kal_rec.signature[:16]}...` (Verified: `{kal_rec.verify_integrity()}`)")

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

                # Volume-Synchronized Probability of Toxicity (VPIN)
                st.markdown("---")
                st.markdown("**Order Flow Toxicity & VPIN (Easley, Lopez de Prado & O'Hara 2012):**")
                from src.features.toxicity import evaluate_flow_toxicity
                tox_rec = evaluate_flow_toxicity(df_obs, ticker=selected_ticker, window=20)

                col_tx1, col_tx2, col_tx3, col_tx4 = st.columns(4)
                col_tx1.metric("Latest VPIN", f"{tox_rec.latest_vpin * 100:.2f}%", delta=f"{tox_rec.mean_vpin * 100:.2f}% mean")
                col_tx2.metric("Buyer Initiation %", f"{tox_rec.latest_buyer_volume_pct:.1f}%")
                col_tx3.metric("Seller Initiation %", f"{tox_rec.latest_seller_volume_pct:.1f}%")
                col_tx4.metric("Toxicity Regime", tox_rec.toxicity_regime)

                if tox_rec.toxicity_regime in ["TOXIC_INFORMED_SURGE", "HIGH_ADVERSE_SELECTION"]:
                    st.error(f"🚨 **Adverse Selection Alert:** {tox_rec.summary}")
                elif tox_rec.toxicity_regime == "MODERATE_FLOW_IMBALANCE":
                    st.warning(f"⚠️ **Flow Imbalance:** {tox_rec.summary}")
                else:
                    st.success(f"🛡️ **Flow State:** {tox_rec.summary}")

                # Kyle's (1985) Informed Trading Lambda & Price Impact Engine
                st.markdown("---")
                st.markdown("**Kyle's (1985) Informed Trading Lambda & Market Price Impact:**")
                from src.analytics.price_impact import evaluate_price_impact_profile
                p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                impact_rec = evaluate_price_impact_profile(
                    ticker=selected_ticker,
                    prices=df_obs[p_col],
                    volumes=df_obs["volume"],
                    window=60,
                )

                col_kp1, col_kp2, col_kp3, col_kp4 = st.columns(4)
                col_kp1.metric("Kyle's Lambda (λ)", f"{impact_rec.kyle_lambda_pct_per_million:.4f}%/$1M", delta=f"${impact_rec.kyle_lambda_abs_per_k_shares:.4f}/1k shares")
                col_kp2.metric("Market Depth (1% shift)", f"${impact_rec.market_depth_millions:.1f}M")
                col_kp3.metric("Square-Root Impact (1% ADV)", f"{impact_rec.sqrt_impact_1pct_adv_bps:.1f} bps")
                col_kp4.metric("Adverse Selection State", impact_rec.adverse_selection_regime)

                if impact_rec.adverse_selection_regime == "HIGH_ADVERSE_SELECTION_FRAGILE":
                    st.error(f"🚨 **Microstructure Liquidity Fragility:** {impact_rec.summary}")
                elif impact_rec.adverse_selection_regime == "MODERATE_IMPACT_NORMAL_LIQUIDITY":
                    st.warning(f"⚠️ **Moderate Price Impact:** {impact_rec.summary}")
                else:
                    st.success(f"🛡️ **Deep Liquidity State:** {impact_rec.summary}")

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

            # Bayesian Online Change-Point Detection (BOCPD) Engine
            with st.expander("🔮 Bayesian Online Change-Point Detection (BOCPD)", expanded=False):
                st.markdown(
                    "Adams & MacKay (2007) exact Bayesian change-point inference on run-length posterior distribution "
                    "$P(r_t | x_{1:t})$ using conjugate Student-t predictive updates. Identifies structural regime shifts "
                    "and mean/volatility breaks in online $O(1)$ memory without moving average lag."
                )
                from src.analytics.changepoints import detect_bayesian_changepoints

                col_cp_cfg, _ = st.columns([2, 4])
                with col_cp_cfg:
                    hazard_val = st.slider("Hazard Timescale (λ Sessions)", min_value=20, max_value=250, value=100, step=10, key="bocpd_lambda_slider")

                cp_receipt = detect_bayesian_changepoints(df_obs, ticker=selected_ticker, hazard_lambda=hazard_val)

                col_cp1, col_cp2, col_cp3, col_cp4 = st.columns(4)
                col_cp1.metric("Current Run-Length", f"{cp_receipt.current_run_length} sessions")
                col_cp2.metric("Break Probability", f"{cp_receipt.current_change_point_prob * 100:.1f}%")
                col_cp3.metric("Regime Stability", cp_receipt.regime_stability_status)
                col_cp4.metric("Breaks Detected", f"{len(cp_receipt.detected_break_dates)}")

                if cp_receipt.regime_stability_status == "STRUCTURAL_BREAK_DETECTED":
                    st.error(f"🚨 **Structural Break Detected:** {cp_receipt.summary}")
                elif cp_receipt.regime_stability_status == "REGIME_UNCERTAINTY":
                    st.warning(f"⚠️ **Regime Uncertainty:** {cp_receipt.summary}")
                else:
                    st.success(f"🛡️ **Regime Stability:** {cp_receipt.summary}")

                if cp_receipt.detected_break_dates:
                    st.markdown("**Historical Structural Break Sessions:**")
                    df_breaks = pd.DataFrame([
                        {"Event #": i + 1, "Break Session Date": d}
                        for i, d in enumerate(cp_receipt.detected_break_dates)
                    ])
                    st.dataframe(df_breaks, use_container_width=True, hide_index=True)

            # Multi-Factor Risk Attribution & Style Exposure
            with st.expander("🔬 Multi-Factor Risk Attribution & Style Exposure", expanded=False):
                st.markdown(
                    "Decompose asset volatility into **Systematic Factor Risk** and **Idiosyncratic Specific Risk** "
                    "using leak-free universe style factors (Market, Momentum, Low Volatility, and Short-Term Reversal) "
                    "with **Newey-West HAC** robust standard errors."
                )
                from src.analytics.factor_attribution import evaluate_factor_attribution
                fa_rec = evaluate_factor_attribution(ticker=selected_ticker, exchange=selected_exchange)

                if fa_rec.sample_sessions >= 15:
                    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
                    col_f1.metric("Regression Fit (R²)", f"{fa_rec.r_squared * 100:.1f}%", delta=f"{fa_rec.adj_r_squared * 100:.1f}% Adj")
                    col_f2.metric("Annualized Alpha (α)", f"{fa_rec.annualized_alpha_pct:+.2f}%", delta=f"t={fa_rec.alpha_t_stat:.2f}")
                    col_f3.metric("Systematic Factor Risk", f"{fa_rec.systemic_risk_share_pct:.1f}%")
                    col_f4.metric("Idiosyncratic Specific Risk", f"{fa_rec.specific_risk_share_pct:.1f}%")

                    if fa_rec.factor_betas:
                        st.markdown("**Style Factor Loadings (Newey-West HAC Robust Estimator):**")
                        fa_df = pd.DataFrame([
                            {
                                "Factor": f,
                                "Beta (β)": f"{fa_rec.factor_betas.get(f, 0.0):.4f}",
                                "t-stat": f"{fa_rec.factor_t_stats.get(f, 0.0):.2f}",
                                "p-value": f"{fa_rec.factor_p_values.get(f, 1.0):.4f}",
                                "Risk Contrib (%)": f"{fa_rec.factor_risk_contributions_pct.get(f, 0.0):.1f}%",
                            }
                            for f in fa_rec.factor_betas.keys()
                        ])
                        st.dataframe(fa_df, use_container_width=True, hide_index=True)

                    st.caption(f"🛡️ **Factor Summary:** {fa_rec.summary} | **HMAC Digest:** `{fa_rec.signature[:16]}...` (Verified: `{fa_rec.verify_integrity()}`)")
                else:
                    st.info(f"ℹ️ {fa_rec.summary}")

            # Multiscale Wavelet Multiresolution & Denoising
            with st.expander("🌊 Multiscale Wavelet Multiresolution & Denoising", expanded=False):
                st.markdown(
                    "Decompose non-stationary prices via shift-invariant **Maximal Overlap Discrete Wavelet Transform (MODWT)**. "
                    "Applies **VisuShrink** adaptive soft-thresholding to isolate transient microstructure noise while "
                    "preserving discrete jump discontinuities with zero phase lag."
                )
                from src.features.wavelet import evaluate_wavelet_profile
                wav_rec = evaluate_wavelet_profile(df_obs, ticker=selected_ticker, levels=3)

                if wav_rec.sample_sessions >= 8:
                    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
                    col_w1.metric("Denoised Jump Trend", f"${wav_rec.latest_denoised_price:.2f}", delta=f"{wav_rec.price_deviation_pct:+.2f}% deviation")
                    col_w2.metric("Estimated Noise (σ)", f"${wav_rec.noise_sigma_hat:.4f}")
                    col_w3.metric("Signal-to-Noise Ratio", f"{wav_rec.snr_db:.1f} dB")
                    col_w4.metric("Timescale Energy Regime", wav_rec.dominant_scale_regime)

                    if wav_rec.energy_shares_pct:
                        st.markdown("**Wavelet Energy Scalogram (Variance across Octaves):**")
                        w_shares_df = pd.DataFrame([
                            {
                                "Octave Scale": k,
                                "Timescale Band": "2-4 sessions (Noise/Bounce)" if "1" in k else ("4-8 sessions (Weekly)" if "2" in k else "8-16 sessions (Swing)"),
                                "Variance Share (%)": f"{v:.1f}%",
                            }
                            for k, v in wav_rec.energy_shares_pct.items()
                        ])
                        st.dataframe(w_shares_df, use_container_width=True, hide_index=True)

                    st.caption(f"🛡️ **Wavelet Summary:** {wav_rec.summary} | **HMAC Digest:** `{wav_rec.signature[:16]}...` (Verified: `{wav_rec.verify_integrity()}`)")
                else:
                    st.info(f"ℹ️ {wav_rec.summary}")

            # Risk-Managed Momentum & Crash Risk Monitor
            with st.expander("🛡️ Risk-Managed Momentum & Crash Risk Monitor", expanded=False):
                st.markdown(
                    "Evaluate **Barroso & Santa-Clara (2015)** inverse-volatility dynamic scaling and **Daniel & Moskowitz (2016)** "
                    "bear-market crash hazard probability. Mitigates severe momentum crashes and negative return skewness."
                )
                from src.analytics.risk_managed_momentum import evaluate_momentum_risk_profile
                mom_rec = evaluate_momentum_risk_profile(df_obs, ticker=selected_ticker, target_vol=0.12, max_leverage=2.0)

                if mom_rec.sample_sessions >= 45:
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("Crash Hazard Probability", f"{mom_rec.current_crash_hazard_pct:.1f}%")
                    col_m2.metric("Crash Risk State", mom_rec.crash_risk_regime)
                    col_m3.metric("Dynamic Leverage", f"{mom_rec.current_leverage_weight:.2f}x", delta=f"{mom_rec.target_volatility_pct:.0f}% target vol")
                    col_m4.metric("RMM Sharpe", f"{mom_rec.rmm_sharpe:+.2f}", delta=f"{mom_rec.rmm_sharpe - mom_rec.raw_momentum_sharpe:+.2f} vs Raw")

                    st.markdown("**Comparative Strategy Performance (Barroso & Santa-Clara 2015):**")
                    mom_df = pd.DataFrame([
                        {"Metric": "Annualized Return", "Unmanaged Momentum": f"{mom_rec.raw_momentum_ann_return_pct:+.2f}%", "Risk-Managed (RMM)": f"{mom_rec.rmm_ann_return_pct:+.2f}%"},
                        {"Metric": "Annualized Volatility", "Unmanaged Momentum": f"{mom_rec.raw_momentum_volatility_pct:.2f}%", "Risk-Managed (RMM)": f"{mom_rec.rmm_volatility_pct:.2f}%"},
                        {"Metric": "Sharpe Ratio (Rf=4%)", "Unmanaged Momentum": f"{mom_rec.raw_momentum_sharpe:+.2f}", "Risk-Managed (RMM)": f"{mom_rec.rmm_sharpe:+.2f}"},
                        {"Metric": "Max Drawdown", "Unmanaged Momentum": f"{mom_rec.raw_momentum_max_drawdown_pct:.2f}%", "Risk-Managed (RMM)": f"{mom_rec.rmm_max_drawdown_pct:.2f}%"},
                        {"Metric": "Return Skewness", "Unmanaged Momentum": f"{mom_rec.raw_momentum_skewness:+.2f}", "Risk-Managed (RMM)": f"{mom_rec.rmm_skewness:+.2f}"},
                    ])
                    st.dataframe(mom_df, use_container_width=True, hide_index=True)
                    st.caption(f"🛡️ **Momentum Summary:** {mom_rec.summary} | **HMAC Digest:** `{mom_rec.signature[:16]}...` (Verified: `{mom_rec.verify_integrity()}`)")
                else:
                    st.info(f"ℹ️ {mom_rec.summary}")

            # Fractional Differentiation & Memory Preservation Analyzer (López de Prado 2018)
            with st.expander("📈 Fractional Differentiation & Memory Preservation Analyzer (FFD)", expanded=False):
                st.markdown(
                    "Evaluate **Marcos López de Prado (2018)** Fixed-Width Window Fractional Differentiation (FFD). "
                    "Determines the minimal degree of differentiation $d^*$ that achieves statistical stationarity ($p_{\\text{ADF}} < 0.05$) "
                    "while preserving maximum memory ($\rho$) from the underlying price series."
                )
                from src.features.fracdiff import evaluate_fracdiff_profile
                fd_price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                fd_prices = df_obs[fd_price_col].dropna()

                if len(fd_prices) >= 40:
                    fd_rec = evaluate_fracdiff_profile(
                        ticker=selected_ticker,
                        prices=fd_prices,
                        threshold=1e-4,
                        auto_d=True,
                    )

                    col_fd1, col_fd2, col_fd3, col_fd4 = st.columns(4)
                    col_fd1.metric("Optimal Stationary Order (d*)", f"{fd_rec.optimal_d:.2f}")
                    col_fd2.metric(
                        "Retained Memory Corr ρ",
                        f"{fd_rec.retained_memory_corr * 100:.1f}%",
                        delta=f"{(fd_rec.retained_memory_corr - fd_rec.raw_to_diff1_corr) * 100:+.1f}% vs ΔP",
                    )
                    col_fd3.metric("ADF Stat at d*", f"{fd_rec.adf_stat:.3f}", delta="5% crit: -2.863")
                    col_fd4.metric("ADF p-value", f"{fd_rec.adf_pvalue:.4f}", delta="Stationary" if fd_rec.is_stationary else "Non-Stationary")

                    if fd_rec.scan_points:
                        st.markdown("**Stationarity vs Memory Trade-off Curve across Differentiation Order $d$:**")
                        scan_df = pd.DataFrame(fd_rec.scan_points)
                        scan_df_disp = scan_df.rename(columns={
                            "d": "Order (d)",
                            "adf_stat": "ADF Stat",
                            "p_value": "ADF p-value",
                            "memory_corr": "Memory Correlation (ρ)",
                            "is_stationary": "Stationary (1/0)",
                        })
                        st.line_chart(scan_df.set_index("d")[["p_value", "memory_corr"]])
                        st.dataframe(scan_df_disp, use_container_width=True, hide_index=True)

                    st.caption(f"📈 **FFD Configuration:** Window Length $l = {fd_rec.window_length}$ sessions | Cutoff $\\tau = {fd_rec.threshold:.1e}$ | **HMAC Digest:** `{fd_rec.signature[:16]}...` (Verified: `{fd_rec.verify_integrity()}`)")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 40 sessions) to evaluate fractional differentiation.")

            # Extreme Value Theory (EVT) Peaks-Over-Threshold Tail Risk Analyzer
            with st.expander("⚡ Extreme Value Theory (EVT) Tail Risk Analyzer", expanded=False):
                st.markdown(
                    "Evaluate **Pickands-Balkema-de Haan (1974, 1975)** Generalized Pareto Distribution (GPD) "
                    "Peaks-Over-Threshold (POT) extreme risk limits. Overcomes Gaussian and polynomial Cornish-Fisher "
                    "underestimations of severe catastrophic market crashes."
                )
                from src.analytics.evt import evaluate_evt_tail_profile
                evt_price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                evt_prices = df_obs[evt_price_col].dropna()

                if len(evt_prices) >= 30:
                    evt_rec = evaluate_evt_tail_profile(
                        ticker=selected_ticker,
                        prices=evt_prices,
                        threshold_quantile=0.95,
                    )

                    col_ev1, col_ev2, col_ev3, col_ev4 = st.columns(4)
                    col_ev1.metric("EVT 99% VaR (1-Day)", f"{evt_rec.var_99_pct:.2f}%")
                    col_ev2.metric("EVT 99% CVaR (ES)", f"{evt_rec.cvar_99_pct:.2f}%")
                    col_ev3.metric("GPD Shape (ξ)", f"{evt_rec.gpd_shape_xi:+.3f}", delta=evt_rec.tail_fatness_regime)
                    col_ev4.metric(
                        "Gaussian Underestimation",
                        f"{evt_rec.tail_risk_underestimation_pct:+.1f}%",
                        delta=f"Gauss: {evt_rec.gaussian_var_99_pct:.2f}% vs EVT: {evt_rec.var_99_pct:.2f}%",
                    )

                    st.markdown("**Comparative Tail Risk Measures (Daily Loss %):**")
                    evt_df = pd.DataFrame([
                        {"Risk Model": "EVT Peaks-Over-Threshold (GPD)", "95% VaR": f"{evt_rec.var_95_pct:.2f}%", "99% VaR": f"{evt_rec.var_99_pct:.2f}%", "99% CVaR (ES)": f"{evt_rec.cvar_99_pct:.2f}%"},
                        {"Risk Model": "Cornish-Fisher Expansion", "95% VaR": "—", "99% VaR": f"{evt_rec.cornish_fisher_var_99_pct:.2f}%", "99% CVaR (ES)": "—"},
                        {"Risk Model": "Standard Gaussian", "95% VaR": "—", "99% VaR": f"{evt_rec.gaussian_var_99_pct:.2f}%", "99% CVaR (ES)": "—"},
                    ])
                    st.dataframe(evt_df, use_container_width=True, hide_index=True)
                    st.caption(fr"⚡ **EVT Threshold:** $u = {evt_rec.threshold_u:.2f}\%$ ({evt_rec.exceedances_count} exceedances) | $\beta = {evt_rec.gpd_scale_beta:.5f}$ | **HMAC Digest:** `{evt_rec.signature[:16]}...` (Verified: `{evt_rec.verify_integrity()}`)")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 30 sessions) to evaluate EVT tail risk.")

            # Market Information Entropy & Ordinal Complexity Monitor (Bandt & Pompe 2002)
            with st.expander("🌌 Market Information Entropy & Ordinal Complexity Monitor", expanded=False):
                st.markdown(
                    "Evaluate **Bandt & Pompe (2002)** Permutation Entropy ($H_P$) and **Richman & Moorman (2000)** Sample Entropy (SampEn). "
                    "Quantifies market disorder and information flow to distinguish predictable alpha structures from high-entropy random walk noise."
                )
                from src.analytics.entropy import evaluate_market_entropy_profile
                ent_price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                ent_prices = df_obs[ent_price_col].dropna()

                if len(ent_prices) >= 30:
                    ent_rec = evaluate_market_entropy_profile(
                        ticker=selected_ticker,
                        prices=ent_prices,
                        window=60,
                        m=3,
                    )

                    col_en1, col_en2, col_en3, col_en4 = st.columns(4)
                    col_en1.metric("Permutation Entropy (H_P)", f"{ent_rec.permutation_entropy:.3f}", delta="0: Min, 1: Max")
                    col_en2.metric("Predictability Index", f"{ent_rec.predictability_index_pct:.1f}%", delta=ent_rec.entropy_regime)
                    col_en3.metric("Sample Entropy (SampEn)", f"{ent_rec.sample_entropy:.3f}")
                    col_en4.metric("Trailing Mean H_P", f"{ent_rec.mean_permutation_entropy:.3f}")

                    if ent_rec.entropy_history:
                        st.markdown("**Trailing 60-Session Permutation Entropy & Predictability Index:**")
                        ent_df = pd.DataFrame(ent_rec.entropy_history)
                        st.line_chart(ent_df.set_index("session_date")[["permutation_entropy", "predictability_index"]])

                    st.caption(fr"🌌 **Complexity Summary:** {ent_rec.summary} | **HMAC Digest:** `{ent_rec.signature[:16]}...` (Verified: `{ent_rec.verify_integrity()}`)")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 30 sessions) to evaluate market entropy.")

            # Asymmetric GJR-GARCH(1, 1) Volatility Forecasting & Leverage Effect Engine
            with st.expander("📉 Asymmetric GJR-GARCH(1, 1) Volatility Forecaster", expanded=False):
                st.markdown(
                    r"Evaluate **Glosten, Jagannathan & Runkle (1993)** Asymmetric GJR-GARCH(1, 1) conditional variance dynamics. "
                    r"Quantifies the equity **leverage effect** ($\gamma$) where negative market returns cause disproportionately higher volatility spikes than positive returns."
                )
                from src.analytics.garch import evaluate_garch_profile
                garch_price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                garch_prices = df_obs[garch_price_col].dropna()

                if len(garch_prices) >= 40:
                    garch_rec = evaluate_garch_profile(
                        ticker=selected_ticker,
                        prices=garch_prices,
                        asymmetric=True,
                    )

                    col_g1, col_g2, col_g3, col_g4 = st.columns(4)
                    col_g1.metric("Conditional Volatility", f"{garch_rec.current_conditional_volatility_ann_pct:.2f}%", delta=f"{garch_rec.current_conditional_volatility_ann_pct - garch_rec.unconditional_volatility_ann_pct:+.2f}% vs Unconditional")
                    col_g2.metric("Leverage Asymmetry (γ)", f"{garch_rec.gamma_asymmetry:.3f}", delta=f"{garch_rec.leverage_asymmetry_ratio:.2f}x bad news ratio")
                    col_g3.metric("GARCH Persistence (φ)", f"{garch_rec.persistence_phi:.3f}", delta="Stationary (<1.0)")
                    col_g4.metric("Unconditional Vol (Mean)", f"{garch_rec.unconditional_volatility_ann_pct:.2f}%")

                    if garch_rec.term_structure_forecast_ann_pct:
                        st.markdown("**Forward Conditional Volatility Term Structure Projections:**")
                        ts_df = pd.DataFrame([
                            {"Horizon": "Next Session (1-Day)", "Projected Annualized Volatility": f"{garch_rec.term_structure_forecast_ann_pct.get('1_session', 0.0):.2f}%"},
                            {"Horizon": "1 Trading Week (5-Day)", "Projected Annualized Volatility": f"{garch_rec.term_structure_forecast_ann_pct.get('5_sessions', 0.0):.2f}%"},
                            {"Horizon": "1 Trading Month (21-Day)", "Projected Annualized Volatility": f"{garch_rec.term_structure_forecast_ann_pct.get('21_sessions', 0.0):.2f}%"},
                        ])
                        st.dataframe(ts_df, use_container_width=True, hide_index=True)

                    st.caption(fr"📉 **Parameters:** $\omega = {garch_rec.omega:.3e}$ | $\alpha = {garch_rec.alpha:.3f}$ | $\gamma = {garch_rec.gamma_asymmetry:.3f}$ | $\beta = {garch_rec.beta:.3f}$ | **HMAC Digest:** `{garch_rec.signature[:16]}...` (Verified: `{garch_rec.verify_integrity()}`)")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 40 sessions) to evaluate GARCH volatility.")

                # Minimum Redundancy Maximum Relevance (mRMR) Feature Selection
                st.markdown("---")
                st.subheader("🧬 Minimum Redundancy Maximum Relevance (mRMR) Feature Selection")
                st.caption(
                    "Identifies the most predictive, non-collinear quantitative signals using Peng, Long & Ding (2005) "
                    "mutual information and redundancy penalties, eliminating multicollinearity before model fitting."
                )
                from src.features.mrmr import evaluate_mrmr_profile
                if len(df_obs) >= 40:
                    try:
                        mrmr_rec = evaluate_mrmr_profile(
                            ticker=selected_ticker,
                            horizon_sessions=1,
                            n_features=6,
                            criterion="MID",
                        )
                        mc1, mc2, mc3, mc4 = st.columns(4)
                        mc1.metric("Selected Signals", f"{mrmr_rec.num_selected} / {mrmr_rec.num_candidates}")
                        mc2.metric("Mean Selected Corr", f"{mrmr_rec.mean_selected_correlation:.3f}")
                        mc3.metric("Candidate Baseline Corr", f"{mrmr_rec.mean_candidate_correlation:.3f}")
                        mc4.metric("Redundancy Reduction", f"{mrmr_rec.redundancy_reduction_pct:+.1f}%")

                        # Bar chart of selected features: Relevance vs Redundancy
                        feat_names = [s["feature"] for s in mrmr_rec.selection_history]
                        rel_vals = [s["relevance"] for s in mrmr_rec.selection_history]
                        red_vals = [s["redundancy"] for s in mrmr_rec.selection_history]
                        scores = [s["mrmr_score"] for s in mrmr_rec.selection_history]

                        fig_mrmr = go.Figure()
                        fig_mrmr.add_trace(go.Bar(
                            x=feat_names, y=rel_vals, name="Relevance I(f;y)",
                            marker_color="#10B981",
                        ))
                        fig_mrmr.add_trace(go.Bar(
                            x=feat_names, y=red_vals, name="Redundancy Penalty R(f;S)",
                            marker_color="#EF4444",
                        ))
                        fig_mrmr.add_trace(go.Scatter(
                            x=feat_names, y=scores, name="Net mRMR Score",
                            mode="lines+markers", line=dict(color="#3B82F6", width=2),
                            yaxis="y",
                        ))
                        fig_mrmr.update_layout(
                            title=f"mRMR Feature Selection Step Trace: {selected_ticker} (Criterion: {mrmr_rec.criterion})",
                            barmode="group",
                            template="plotly_dark",
                            height=360,
                            margin=dict(l=30, r=30, t=40, b=30),
                        )
                        st.plotly_chart(fig_mrmr, use_container_width=True)
                        st.caption(f"🛡️ **Selection Provenance:** {mrmr_rec.summary} | **HMAC Digest:** `{mrmr_rec.signature[:16]}...` (Verified: `{mrmr_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"mRMR feature selection deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 40 sessions) to evaluate mRMR feature selection.")

                # Robust Quantile Tail Shape & Moors Kurtosis Engine
                st.markdown("---")
                st.subheader("📐 Robust Quantile Tail Shape & Moors Kurtosis Engine")
                st.caption(
                    "Quantifies fat-tailedness, Moors (1988) octile kurtosis, and downside crash asymmetry "
                    "without requiring higher-order moment stability, identifying structural crash vulnerability."
                )
                from src.analytics.tail_shape import evaluate_tail_shape_profile
                if len(df_obs) >= 30:
                    try:
                        p_ts_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        tail_rec = evaluate_tail_shape_profile(
                            ticker=selected_ticker,
                            prices=df_obs[p_ts_col],
                            window=120,
                        )
                        tq1, tq2, tq3, tq4 = st.columns(4)
                        tq1.metric("Moors Kurtosis", f"{tail_rec.moors_quantile_kurtosis:.3f}", delta=f"{tail_rec.moors_excess_kurtosis:+.3f} vs Normal 1.233")
                        tq2.metric("Bowley Skewness", f"{tail_rec.bowley_quantile_skewness:+.3f}", delta="Bounded [-1, +1]")
                        tq3.metric("Tail Asymmetry Ratio", f"{tail_rec.tail_asymmetry_ratio:.2f}x", delta=f"L: {tail_rec.left_tail_depth_1pct:.1f}% vs R: {tail_rec.right_tail_depth_1pct:.1f}%")
                        tq4.metric("Tail Shape Regime", tail_rec.tail_shape_regime)

                        if tail_rec.tail_shape_regime == "HEAVY_TAIL_ASYMMETRIC_CRASH_RISK":
                            st.error(f"🚨 **Asymmetric Crash Risk Alert:** {tail_rec.summary}")
                        elif tail_rec.tail_shape_regime == "LEPTOKURTIC_SYMMETRIC":
                            st.warning(f"⚠️ **Fat Tails Warning:** {tail_rec.summary}")
                        else:
                            st.success(f"🛡️ **Tail Shape Profile:** {tail_rec.summary}")

                        st.caption(f"🛡️ **Provenance HMAC Digest:** `{tail_rec.signature[:16]}...` (Verified: `{tail_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Tail shape evaluation deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 30 sessions) to evaluate tail shape.")

                # Hamilton (1989) Two-State Markov-Switching Regime Engine
                st.markdown("---")
                st.subheader("🔄 Hamilton (1989) Two-State Markov-Switching Regime Engine")
                st.caption(
                    "Bayesian hidden Markov state-space inference distinguishing calm low-volatility expansion "
                    "from turbulent high-volatility crisis regimes with transition matrices and expected durations."
                )
                from src.analytics.markov_regime import evaluate_markov_regime_profile
                if len(df_obs) >= 40:
                    try:
                        p_mr_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        markov_rec = evaluate_markov_regime_profile(
                            ticker=selected_ticker,
                            prices=df_obs[p_mr_col],
                            window=120,
                        )
                        mq1, mq2, mq3, mq4 = st.columns(4)
                        mq1.metric("Current Regime", markov_rec.current_regime)
                        mq2.metric("Crisis Probability P(S=1)", f"{markov_rec.current_crisis_probability * 100:.1f}%")
                        mq3.metric("Low-Vol Duration (p00)", f"{markov_rec.expected_low_vol_duration_sessions:.1f} sessions", delta=f"{markov_rec.state_0_volatility_ann_pct:.1f}% Vol")
                        mq4.metric("Crisis Duration (p11)", f"{markov_rec.expected_high_vol_duration_sessions:.1f} sessions", delta=f"{markov_rec.state_1_volatility_ann_pct:.1f}% Vol ({markov_rec.volatility_ratio_state1_to_state0:.1f}x)")

                        if markov_rec.current_regime == "HIGH_VOLATILITY_CRISIS":
                            st.error(f"🚨 **Markov Crisis Regime Alert:** {markov_rec.summary}")
                        else:
                            st.success(f"🛡️ **Markov Regime State:** {markov_rec.summary}")

                        st.caption(f"🛡️ **Markov Audit Digest:** `{markov_rec.signature[:16]}...` (Verified: `{markov_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Markov regime evaluation deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 40 sessions) to evaluate Markov-switching model.")

                # Barndorff-Nielsen & Shephard (2004, 2006) Bipower Jump Decomposition
                st.markdown("---")
                st.subheader("⚡ Barndorff-Nielsen & Shephard (2004, 2006) Jump Detection Engine")
                st.caption(
                    "Nonparametric quadratic variation decomposition isolating continuous hedgeable sample path "
                    "variance from discrete, unhedgeable Poisson jump events."
                )
                from src.analytics.jump_diffusion import evaluate_jump_diffusion_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        jump_rec = evaluate_jump_diffusion_profile(
                            ticker=selected_ticker,
                            prices=df_obs[p_col],
                            window=21,
                        )
                        jq1, jq2, jq3, jq4 = st.columns(4)
                        jq1.metric("Jump Regime", jump_rec.jump_regime)
                        jq2.metric("Total Realized Vol", f"{jump_rec.annualized_total_vol_pct:.1f}% Ann", delta=f"{jump_rec.annualized_continuous_vol_pct:.1f}% Cont / {jump_rec.annualized_jump_vol_pct:.1f}% Jump")
                        jq3.metric("Jump Ratio (JV/RV)", f"{jump_rec.jump_contribution_ratio * 100:.1f}%")
                        jq4.metric("BNS Z-Score", f"{jump_rec.bns_z_statistic:.2f}", delta="Significant Jump (p<0.05)" if jump_rec.has_significant_jump else "Continuous Path")

                        if jump_rec.has_significant_jump:
                            st.warning(f"⚡ **Discrete Jump Alert:** {jump_rec.summary}")
                        else:
                            st.success(f"🛡️ **Continuous Path State:** {jump_rec.summary}")

                        st.caption(f"🛡️ **Jump Audit Digest:** `{jump_rec.signature[:16]}...` (Verified: `{jump_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Jump diffusion evaluation deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate jump diffusion model.")

                # Richard Roll (1984) Effective Spread & Composite Execution Friction
                st.markdown("---")
                st.subheader("💧 Richard Roll (1984) Effective Spread & Trading Friction")
                st.caption(
                    "Serial covariance order bounce model and composite pre-trade slippage estimating round-trip execution costs."
                )
                from src.analytics.effective_spread import evaluate_effective_spread_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        spread_rec = evaluate_effective_spread_profile(
                            ticker=selected_ticker,
                            prices=df_obs[p_col],
                            high=df_obs["high"] if "high" in df_obs.columns else None,
                            low=df_obs["low"] if "low" in df_obs.columns else None,
                            volumes=df_obs["volume"] if "volume" in df_obs.columns else None,
                            window=21,
                        )
                        sq1, sq2, sq3, sq4 = st.columns(4)
                        sq1.metric("Liquidity Tier", spread_rec.liquidity_tier)
                        sq2.metric("Roll Spread", f"{spread_rec.roll_spread_pct:.2f}%", delta=f"${spread_rec.roll_spread_dollars:.3f}")
                        sq3.metric("Corwin-Schultz Spread", f"{spread_rec.corwin_schultz_spread_pct:.2f}%")
                        sq4.metric("Composite Round-Trip", f"{spread_rec.composite_round_trip_bps:.1f} bps", delta=f"Autocov: {spread_rec.serial_autocovariance:.4f}")

                        st.caption(f"🛡️ **Microstructure Cost Summary:** {spread_rec.summary}")
                        st.caption(f"🛡️ **Spread Audit Digest:** `{spread_rec.signature[:16]}...` (Verified: `{spread_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Effective spread evaluation deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate effective spread model.")

                # Ang, Chen & Xing (2006) Downside Beta & Realized Semivariance
                st.markdown("---")
                st.subheader("📉 Ang, Chen & Xing (2006) Downside Beta & Realized Semivariance")
                st.caption(
                    "Asymmetric market co-movement (downside vs. upside beta) and Barndorff-Nielsen realized semivariance decomposition."
                )
                from src.analytics.downside_risk import evaluate_downside_risk_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        down_rec = evaluate_downside_risk_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            benchmark_ticker="SPY",
                            window=min(len(df_obs) - 1, 60),
                        )
                        dq1, dq2, dq3, dq4 = st.columns(4)
                        dq1.metric("Crash Sensitivity", down_rec.crash_sensitivity_tier)
                        dq2.metric("Downside Beta (β⁻)", f"{down_rec.downside_beta:.2f}", delta=f"CAPM: {down_rec.capm_beta:.2f}")
                        dq3.metric("Upside Beta (β⁺)", f"{down_rec.upside_beta:.2f}", delta=f"Spread: {down_rec.beta_asymmetry_spread:+.2f}")
                        dq4.metric("Downside Var Ratio", f"{down_rec.downside_variance_ratio_pct:.1f}%", delta=f"Bad Vol: {down_rec.annualized_downside_vol_pct:.1f}% Ann")

                        st.caption(f"🛡️ **Downside Asymmetry Summary:** {down_rec.summary}")
                        st.caption(f"🛡️ **Downside Risk Audit Digest:** `{down_rec.signature[:16]}...` (Verified: `{down_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Downside risk evaluation deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate downside risk profile.")

                # Fama-French (1993) & Carhart (1997) Multi-Factor Risk & Alpha Attribution
                st.markdown("---")
                st.subheader("🏛️ Fama-French & Carhart (1997) Multi-Factor Risk & Alpha Attribution")
                st.caption(
                    "Multi-factor asset pricing decomposition: Market (MKT), Size (SMB), Value (HML), and Momentum (UMD) with White HC0 robust standard errors."
                )
                from src.analytics.fama_french import evaluate_fama_french_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        ff_rec = evaluate_fama_french_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            benchmark_ticker="SPY",
                            lookback_window=min(len(df_obs) - 1, 60),
                        )
                        fq1, fq2, fq3, fq4 = st.columns(4)
                        sig_alpha = "★ Sig" if ff_rec.is_alpha_significant else "Not Sig"
                        fq1.metric("Annualized Alpha", f"{ff_rec.annualized_alpha_pct:+.2f}%", delta=f"t={ff_rec.alpha_t_stat:+.2f} ({sig_alpha})")
                        fq2.metric("Market Beta (MKT)", f"{ff_rec.market_beta:.2f}", delta=f"Adj R²: {ff_rec.r_squared_adjusted * 100:.1f}%")
                        fq3.metric("Size Tilt (SMB)", f"{ff_rec.size_beta:+.2f}", delta=f"Value (HML): {ff_rec.value_beta:+.2f}")
                        fq4.metric("Idiosyncratic Vol", f"{ff_rec.idiosyncratic_annualized_vol_pct:.1f}% Ann", delta=f"{ff_rec.idiosyncratic_variance_ratio_pct:.1f}% of Var")

                        # Factor loadings table expander
                        with st.expander("📊 Factor Loadings & White HC0 Robust Inference", expanded=False):
                            st.dataframe(pd.DataFrame(ff_rec.factor_loadings), use_container_width=True)

                        st.caption(f"🛡️ **Multi-Factor Summary:** {ff_rec.summary}")
                        st.caption(f"🛡️ **Factor Audit Digest:** `{ff_rec.signature[:16]}...` (Verified: `{ff_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Fama-French factor attribution deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate multi-factor model.")

                # Ang, Hodrick, Xing & Zhang (2006) Cross-Sectional Idiosyncratic Volatility (IVOL) & Higher Moments
                st.markdown("---")
                st.subheader("⚡ Ang et al. (2006) Idiosyncratic Volatility (IVOL) & Higher-Moment Alpha Engine")
                st.caption(
                    "Cross-sectional idiosyncratic volatility puzzle, lottery-demand skewness (Boyer et al. 2010), "
                    "tail kurtosis, downside idiosyncratic semivariance, and expected return anomaly drag."
                )
                from src.analytics.idiosyncratic_volatility import evaluate_idiosyncratic_volatility_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        ivol_rec = evaluate_idiosyncratic_volatility_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            benchmark_ticker="SPY",
                            lookback_window=min(len(df_obs) - 1, 60),
                        )
                        iv1, iv2, iv3, iv4 = st.columns(4)
                        iv1.metric(
                            "Idiosyncratic Vol (IVOL)",
                            f"{ivol_rec.idiosyncratic_vol_ann_pct:.1f}% Ann",
                            delta=f"{ivol_rec.idiosyncratic_variance_ratio_pct:.1f}% of Total Var",
                        )
                        iv2.metric(
                            "Downside IVOL (IVOL⁻)",
                            f"{ivol_rec.downside_idiosyncratic_vol_ann_pct:.1f}% Ann",
                            delta=f"Ratio: {ivol_rec.downside_ivol_ratio:.2f}",
                        )
                        iv3.metric(
                            "Idiosyncratic Skew (ISKEW)",
                            f"{ivol_rec.idiosyncratic_skewness:+.2f}",
                            delta=f"IKURT: {ivol_rec.idiosyncratic_kurtosis:.1f}",
                            help="Boyer et al. (2010) lottery-demand skewness; positive values reflect speculative retail premium",
                        )
                        iv4.metric(
                            "Anomaly Regime",
                            ivol_rec.ivol_anomaly_regime,
                            delta=f"-{ivol_rec.ivol_expected_return_haircut_pct:.2f}%/yr drag" if ivol_rec.ivol_expected_return_haircut_pct > 0 else "No Drag",
                            delta_color="inverse" if ivol_rec.ivol_expected_return_haircut_pct > 0 else "normal",
                        )

                        st.caption(f"🛡️ **IVOL Anomaly Summary:** {ivol_rec.summary}")
                        st.caption(f"🛡️ **IVOL Audit Digest:** `{ivol_rec.signature[:16]}...` (Verified: `{ivol_rec.verify_integrity()}`)")
                    except Exception as iv_ex:
                        st.info(f"Idiosyncratic volatility evaluation deferred: {iv_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate idiosyncratic volatility.")

                # Pástor & Stambaugh (2003) Systematic Liquidity Risk Factor & Liquidity Beta
                st.markdown("---")
                st.subheader("🌊 Pástor & Stambaugh (2003) Systematic Liquidity Risk & Beta Engine")
                st.caption(
                    "Sensitivity to market-wide aggregate liquidity dry-ups (Liquidity Beta β_L), "
                    "order-flow reversal price impact (γ), and expected systematic liquidity premium (7.5% annual decile spread)."
                )
                from src.analytics.pastor_stambaugh import evaluate_pastor_stambaugh_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        ps_rec = evaluate_pastor_stambaugh_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            asset_volumes=df_obs["volume"],
                            benchmark_ticker="SPY",
                            lookback_window=min(len(df_obs) - 1, 60),
                        )
                        ps1, ps2, ps3, ps4 = st.columns(4)
                        sig_b = "★ Sig" if ps_rec.is_liquidity_beta_significant else "Not Sig"
                        ps1.metric(
                            "Liquidity Beta (β_L)",
                            f"{ps_rec.liquidity_beta:+.3f}",
                            delta=f"t={ps_rec.liquidity_beta_t_stat:+.2f} ({sig_b})",
                            help="Sensitivity to aggregate market liquidity innovations; high beta commands return premium but suffers in panics",
                        )
                        ps2.metric(
                            "Market Beta (β_M)",
                            f"{ps_rec.market_beta:.2f}",
                            delta=f"R²: {ps_rec.r_squared * 100:.1f}%",
                        )
                        ps3.metric(
                            "Order Flow Gamma (γ)",
                            f"{ps_rec.average_price_impact_gamma:+.4f}",
                            delta="Reversal Friction",
                            help="Campbell-Grossman-Wang price impact parameter; negative values reflect order flow reversal",
                        )
                        ps4.metric(
                            "Liquidity Regime",
                            ps_rec.liquidity_risk_regime,
                            delta=f"{ps_rec.expected_liquidity_premium_pct:+.2f}%/yr Prem",
                        )

                        st.caption(f"🛡️ **Liquidity Beta Summary:** {ps_rec.summary}")
                        st.caption(f"🛡️ **Liquidity Audit Digest:** `{ps_rec.signature[:16]}...` (Verified: `{ps_rec.verify_integrity()}`)")
                    except Exception as ps_ex:
                        st.info(f"Pástor-Stambaugh liquidity evaluation deferred: {ps_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate systematic liquidity risk.")

                # Harvey & Siddique (2000) Co-Skewness & Dittmar (2002) Co-Kurtosis Higher-Moment Asset Pricing
                st.markdown("---")
                st.subheader("📐 Harvey-Siddique Co-Skewness & Dittmar Co-Kurtosis Higher-Moment Pricing")
                st.caption(
                    "Third and fourth systematic co-moments: Co-Skewness (β_SKS, covariance with market volatility) "
                    "and Co-Kurtosis (β_CKT, covariance with market tail-risk contagion) with Four-Moment Alpha."
                )
                from src.analytics.higher_moments import evaluate_higher_moments_profile
                if len(df_obs) >= 22:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        hm_rec = evaluate_higher_moments_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            benchmark_ticker="SPY",
                            lookback_window=min(len(df_obs) - 1, 60),
                        )
                        hm1, hm2, hm3, hm4 = st.columns(4)
                        sig_sks = "★ Sig" if hm_rec.is_coskewness_significant else "Not Sig"
                        sig_ckt = "★ Sig" if hm_rec.is_cokurtosis_significant else "Not Sig"
                        hm1.metric(
                            "Co-Skewness (β_SKS)",
                            f"{hm_rec.coskewness_beta:+.3f}",
                            delta=f"t={hm_rec.coskewness_t_stat:+.2f} ({sig_sks})",
                            help="Harvey & Siddique (2000) co-skewness; negative values reflect crash-when-market-spikes risk and command positive premium",
                        )
                        hm2.metric(
                            "Co-Kurtosis (β_CKT)",
                            f"{hm_rec.cokurtosis_beta:.3f}",
                            delta=f"t={hm_rec.cokurtosis_t_stat:+.2f} ({sig_ckt})",
                            help="Dittmar (2002) co-kurtosis; measures sensitivity to market extreme fat-tail co-exceedance",
                        )
                        hm3.metric(
                            "4-Moment Alpha",
                            f"{hm_rec.annualized_alpha_pct:+.2f}% Ann",
                            delta=f"R²: {hm_rec.four_moment_r_squared * 100:.1f}%",
                        )
                        hm4.metric(
                            "Higher-Moment Regime",
                            hm_rec.higher_moment_risk_regime,
                            delta=f"{hm_rec.expected_coskewness_premium_pct:+.2f}%/yr SKS Prem",
                        )

                        st.caption(f"🛡️ **Higher Moments Summary:** {hm_rec.summary}")
                        st.caption(f"🛡️ **Higher Moments Audit Digest:** `{hm_rec.signature[:16]}...` (Verified: `{hm_rec.verify_integrity()}`)")
                    except Exception as hm_ex:
                        st.info(f"Higher moments asset pricing evaluation deferred: {hm_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 22 sessions) to evaluate higher moments.")

                st.markdown("---")
                st.subheader("💳 Robert C. Merton (1974) Structural Credit Risk & Distance-to-Default (DD)")
                st.caption(
                    "Option-theoretic structural solvency model solving unobservable firm asset value (V_A) "
                    "and asset volatility (σ_A) to evaluate Distance-to-Default, default probability (π), "
                    "and structural credit spreads."
                )
                from src.analytics.merton_credit import evaluate_merton_credit_profile
                if len(df_obs) >= 20:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        latest_price = float(df_obs[p_col].dropna().iloc[-1])
                        mc_cols = st.columns(3)
                        with mc_cols[0]:
                            debt_input_b = st.number_input(
                                "Total Debt / Default Point ($B)",
                                min_value=0.0,
                                max_value=5000.0,
                                value=round(latest_price * 1.0 * 0.25, 2),
                                step=1.0,
                                key="merton_debt_b",
                                help="Face value of short-term debt + 0.5 * long-term debt (KMV default point in $ Billions)",
                            )
                        with mc_cols[1]:
                            shares_b = st.number_input(
                                "Common Shares Outstanding (Billion shares)",
                                min_value=0.01,
                                max_value=100.0,
                                value=1.0,
                                step=0.1,
                                key="merton_shares_b",
                                help="Total shares outstanding in Billions (defaults to 1.0B shares)",
                            )
                        with mc_cols[2]:
                            rf_rate_pct = st.number_input(
                                "Risk-Free Rate (% / yr)",
                                min_value=0.0,
                                max_value=20.0,
                                value=4.5,
                                step=0.25,
                                key="merton_rf_pct",
                            )

                        mc_rec = evaluate_merton_credit_profile(
                            ticker=selected_ticker,
                            debt_value=debt_input_b * 1_000_000_000.0,
                            shares_outstanding=shares_b * 1_000_000_000.0,
                            asset_prices=df_obs[p_col],
                            risk_free_rate_ann=rf_rate_pct / 100.0,
                            horizon_years=1.0,
                            lookback_window=min(len(df_obs) - 1, 60),
                        )

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(
                            "Distance-to-Default (DD)",
                            f"{mc_rec.distance_to_default:+.2f} σ",
                            delta=f"d2 (RN): {mc_rec.risk_neutral_dd:+.2f}",
                            help="Standard deviations to default point; DD >= 3.0 represents investment grade, DD < 1.0 imminent default",
                        )
                        m2.metric(
                            "Default Probability (π)",
                            f"{mc_rec.default_probability_pct:.4f}%",
                            delta=f"Regime: {mc_rec.distress_regime}",
                            help="Theoretical cumulative 1-year default probability Φ(-DD)",
                        )
                        m3.metric(
                            "Structural Credit Spread",
                            f"{mc_rec.implied_credit_spread_bps:.1f} bps",
                            delta=f"Leverage: {mc_rec.leverage_ratio_pct:.1f}%",
                            help="Merton implied theoretical yield spread over risk-free benchmark",
                        )
                        m4.metric(
                            "Implied Asset Vol (σ_A)",
                            f"{mc_rec.implied_asset_volatility * 100:.2f}%",
                            delta=f"Equity Vol: {mc_rec.annualized_equity_volatility * 100:.2f}%",
                            help="Option-inverted unlevered firm asset volatility",
                        )

                        st.caption(f"💳 **Merton Credit Profile:** {mc_rec.summary}")
                        st.caption(f"🛡️ **Credit Audit Digest:** `{mc_rec.signature[:16]}...` (Verified: `{mc_rec.verify_integrity()}`)")
                    except Exception as mc_ex:
                        st.info(f"Merton structural credit evaluation deferred: {mc_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 20 sessions) to evaluate Merton credit risk.")

                st.markdown("---")
                st.subheader("⚡ Joel Hasbrouck (1991) VAR Price Discovery & Information Share")
                st.caption(
                    "Dynamic Vector Autoregression (VAR) decomposing trade price impact into permanent informed trading "
                    "information content vs transient microstructure friction (inventory bounce), with Random-Walk "
                    "variance decomposition (R_w²)."
                )
                from src.analytics.hasbrouck_var import evaluate_hasbrouck_var_profile
                if len(df_obs) >= 30:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        v_col = "volume"
                        hv_cols = st.columns(2)
                        with hv_cols[0]:
                            var_lags_val = st.slider("VAR Lag Order (p)", min_value=1, max_value=8, value=5, step=1, key="hasbrouck_lags")
                        with hv_cols[1]:
                            irf_h_val = st.slider("IRF Simulation Horizon (H)", min_value=5, max_value=30, value=20, step=5, key="hasbrouck_irf_h")

                        hv_rec = evaluate_hasbrouck_var_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            asset_volumes=df_obs[v_col],
                            var_lags=var_lags_val,
                            irf_horizon=irf_h_val,
                            lookback_window=min(len(df_obs) - 1, 120),
                        )

                        hv1, hv2, hv3, hv4 = st.columns(4)
                        hv1.metric(
                            "Permanent Impact I(∞)",
                            f"{hv_rec.permanent_impact_bps:+.2f} bps",
                            delta=f"{hv_rec.permanent_to_total_ratio_pct:.1f}% of total",
                            help="Permanent informational price revision per 1-sigma unexpected order flow innovation",
                        )
                        hv2.metric(
                            "Contemporaneous Impact (b0)",
                            f"{hv_rec.contemporaneous_impact_bps:+.2f} bps",
                            delta=f"Transient S0: {hv_rec.transient_impact_bps:+.2f} bps",
                            help="Immediate trade execution price response b0",
                        )
                        hv3.metric(
                            "Trade Info Share (R_w²)",
                            f"{hv_rec.information_share_rw2_pct:.1f}%",
                            delta=f"Regime: {hv_rec.market_quality_regime}",
                            help="Percentage of permanent efficient random-walk price variance explained by trade order flow",
                        )
                        hv4.metric(
                            "Public News Volatility",
                            f"{hv_rec.public_innovation_bps:.1f} bps",
                            delta=f"Trade Shock σ: {hv_rec.trade_innovation_std:.2f}",
                            help="Non-trade public information innovation volatility",
                        )

                        st.caption(f"⚡ **Hasbrouck Microstructure Summary:** {hv_rec.summary}")
                        st.caption(f"🛡️ **VAR Audit Digest:** `{hv_rec.signature[:16]}...` (Verified: `{hv_rec.verify_integrity()}`)")
                    except Exception as hv_ex:
                        st.info(f"Hasbrouck VAR price discovery evaluation deferred: {hv_ex}")
                else:
                    st.info("ℹ️ Insufficient price & volume history (minimum 30 sessions) to evaluate Hasbrouck VAR.")

                st.markdown("---")
                st.subheader("📊 Bollerslev, Tauchen & Zhou (2009) Variance Risk Premium (VRP) & Vol-of-Vol")
                st.caption(
                    "Econometric crash-risk insurance premium measuring the spread between risk-neutral expected variance "
                    "(IV²) and physical realized variance (RV²), with Newey-West forward return predictions."
                )
                from src.analytics.variance_risk_premium import evaluate_variance_risk_premium_profile
                if len(df_obs) >= 45:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        vrp_cols = st.columns(2)
                        with vrp_cols[0]:
                            iv_override = st.number_input(
                                "Implied Volatility IV (% Ann, 0=Auto)",
                                min_value=0.0,
                                max_value=200.0,
                                value=0.0,
                                step=1.0,
                                key="vrp_iv_input",
                                help="Option-implied volatility in % (e.g. 28.0). Set 0 for automated empirical proxy.",
                            )
                        with vrp_cols[1]:
                            vrp_horizon_val = st.slider(
                                "Forecast Horizon (Sessions)",
                                min_value=5,
                                max_value=63,
                                value=21,
                                step=1,
                                key="vrp_horizon_slider",
                                help="Number of forward trading sessions (21 sessions = 1 trading month)",
                            )

                        iv_param = (iv_override / 100.0) if iv_override > 0 else None
                        vrp_rec = evaluate_variance_risk_premium_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            implied_volatility=iv_param,
                            horizon_sessions=vrp_horizon_val,
                            lookback_window=min(len(df_obs) - 1, 120),
                        )

                        vp1, vp2, vp3, vp4 = st.columns(4)
                        vp1.metric(
                            "Variance Risk Premium (VRP)",
                            f"{vrp_rec.variance_risk_premium:+.2f} pts",
                            delta=f"Vol Spread: {vrp_rec.volatility_spread_pct_points:+.1f}% pts",
                            help="Bollerslev VRP = (IV² - RV²) * 100",
                        )
                        vp2.metric(
                            "IV vs RV",
                            f"{vrp_rec.implied_volatility_pct:.1f}% IV",
                            delta=f"RV: {vrp_rec.realized_volatility_pct:.1f}%",
                            help="Risk-neutral Implied Volatility vs Physical Realized Volatility",
                        )
                        vp3.metric(
                            "Predicted Forward Return",
                            f"{vrp_rec.predicted_forward_return_pct:+.2f}%",
                            delta=f"β_VRP: {vrp_rec.predictive_beta_vrp:+.4f} (t={vrp_rec.predictive_t_stat:+.2f})",
                            help=f"Newey-West HAC predicted forward {vrp_rec.horizon_sessions}-session return",
                        )
                        vp4.metric(
                            "Vol-of-Vol (VoV)",
                            f"{vrp_rec.volatility_of_volatility_pct:.1f}%",
                            delta=f"Regime: {vrp_rec.vrp_regime}",
                            help="Annualized volatility-of-volatility (VVIX equivalent second-order risk)",
                        )

                        st.caption(f"📊 **VRP Summary:** {vrp_rec.summary}")
                        st.caption(f"🛡️ **VRP Audit Digest:** `{vrp_rec.signature[:16]}...` (Verified: `{vrp_rec.verify_integrity()}`)")
                    except Exception as vrp_ex:
                        st.info(f"Variance Risk Premium evaluation deferred: {vrp_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 45 sessions) to evaluate Variance Risk Premium.")

                st.markdown("---")
                st.subheader("🎰 Bali, Cakici & Whitelaw (2011) 'MAX' Effect & Lottery Demand Alpha")
                st.caption(
                    "Behavioral finance cumulative prospect theory engine quantifying speculative retail lottery preferences, "
                    "extreme upside tail order statistics (MAX_N), and expected structural alpha drag."
                )
                from src.analytics.max_lottery import evaluate_max_lottery_profile
                if len(df_obs) >= 20:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        ml_cols = st.columns(2)
                        with ml_cols[0]:
                            top_n_val = st.slider("Extreme Tail Days (N)", min_value=1, max_value=10, value=5, step=1, key="max_lottery_top_n")
                        with ml_cols[1]:
                            ml_lookback = st.slider("Observation Window (Sessions)", min_value=20, max_value=120, value=60, step=10, key="max_lottery_lookback")

                        ml_rec = evaluate_max_lottery_profile(
                            ticker=selected_ticker,
                            asset_prices=df_obs[p_col],
                            top_n_days=top_n_val,
                            lookback_window=ml_lookback,
                        )

                        ml1, ml2, ml3, ml4 = st.columns(4)
                        ml1.metric(
                            f"MAX_{ml_rec.top_n_days} Avg Return",
                            f"{ml_rec.max_n_pct:+.2f}%",
                            delta=f"Single MAX_1: {ml_rec.max_1_pct:+.2f}%",
                            help=f"Average of the {ml_rec.top_n_days} largest daily return spikes in lookback window",
                        )
                        ml2.metric(
                            f"MIN_{ml_rec.top_n_days} Downside",
                            f"{ml_rec.min_n_pct:+.2f}%",
                            delta=f"Asymmetry: {ml_rec.max_to_min_ratio:.2f}x",
                            help=f"Average of the {ml_rec.top_n_days} worst daily drawdowns; asymmetry ratio = MAX / |MIN|",
                        )
                        ml3.metric(
                            "Lottery Score (Λ)",
                            f"{ml_rec.lottery_demand_score:.1f} / 100",
                            delta=f"Regime: {ml_rec.lottery_regime}",
                            help="Composite behavioral lottery demand index; scores > 75 signal severe retail overpricing",
                        )
                        ml4.metric(
                            "Expected Alpha Drag",
                            f"{ml_rec.expected_alpha_drag_ann_pct:+.2f}% Ann",
                            delta=f"Skew: {ml_rec.return_skewness:+.2f}",
                            help="Bali et al. (2011) expected annualized risk-adjusted performance penalty / compounder premium",
                        )

                        st.caption(f"🎰 **MAX Lottery Summary:** {ml_rec.summary}")
                        st.caption(f"🛡️ **Lottery Audit Digest:** `{ml_rec.signature[:16]}...` (Verified: `{ml_rec.verify_integrity()}`)")
                    except Exception as ml_ex:
                        st.info(f"MAX lottery evaluation deferred: {ml_ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 20 sessions) to evaluate MAX lottery effect.")

                st.markdown("---")
                st.subheader("💎 Robert Novy-Marx (2013) Gross Profitability & QMJ Quality Engine")
                st.caption(
                    "Fundamental quantitative quality engine measuring Gross Profitability to Assets (GP/A), "
                    "operating efficiency, and Asness et al. (2019) QMJ composite scores to immunize portfolios against value traps."
                )
                from src.analytics.gross_profitability import evaluate_gross_profitability_profile
                if len(df_obs) >= 1:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        latest_p = float(df_obs[p_col].dropna().iloc[-1])
                        est_assets_b = round(latest_p * 1.0 * 1.10, 2)
                        est_rev_b = round(est_assets_b * 0.85, 2)
                        est_cogs_b = round(est_rev_b * 0.58, 2)
                        est_debt_b = round(est_assets_b * 0.25, 2)

                        gp_cols = st.columns(4)
                        with gp_cols[0]:
                            rev_input = st.number_input("Annual Revenue ($B)", min_value=0.1, max_value=2000.0, value=max(0.1, est_rev_b), step=5.0, key="gp_rev_b")
                        with gp_cols[1]:
                            cogs_input = st.number_input("COGS ($B)", min_value=0.0, max_value=2000.0, value=max(0.0, est_cogs_b), step=5.0, key="gp_cogs_b")
                        with gp_cols[2]:
                            assets_input = st.number_input("Total Assets ($B)", min_value=0.1, max_value=5000.0, value=max(0.1, est_assets_b), step=5.0, key="gp_assets_b")
                        with gp_cols[3]:
                            debt_input = st.number_input("Total Debt ($B)", min_value=0.0, max_value=2000.0, value=max(0.0, est_debt_b), step=5.0, key="gp_debt_b")

                        gp_rec = evaluate_gross_profitability_profile(
                            ticker=selected_ticker,
                            revenue=rev_input * 1_000_000_000.0,
                            cogs=cogs_input * 1_000_000_000.0,
                            total_assets=assets_input * 1_000_000_000.0,
                            total_debt=debt_input * 1_000_000_000.0,
                        )

                        g1, g2, g3, g4 = st.columns(4)
                        g1.metric(
                            "Gross Profitability (GP/A)",
                            f"{gp_rec.gross_profitability_pct:.1f}%",
                            delta=f"Margin: {gp_rec.gross_margin_pct:.1f}%",
                            help="Novy-Marx (2013) Gross Profit to Total Assets (GP/A); > 30% indicates high productive efficiency",
                        )
                        g2.metric(
                            "Asset Turnover",
                            f"{gp_rec.asset_turnover:.2f}x",
                            delta=f"Leverage: {gp_rec.debt_to_assets_pct:.1f}%",
                            help="Capital turnover = Revenue / Assets; debt-to-assets measures leverage safety",
                        )
                        trap_badge = "⚠️ VALUE TRAP" if gp_rec.is_value_trap else "✅ SAFE VALUE"
                        g3.metric(
                            "QMJ Quality Score",
                            f"{gp_rec.quality_score:.1f} / 100",
                            delta=trap_badge,
                            help="Asness et al. (2019) composite Quality-Minus-Junk score",
                        )
                        g4.metric(
                            "Expected Quality Alpha",
                            f"{gp_rec.expected_quality_alpha_ann_pct:+.2f}% Ann",
                            delta=f"Regime: {gp_rec.quality_regime}",
                            help="Expected annualized risk-adjusted return premium over low-quality junk",
                        )

                        st.caption(f"💎 **Profitability Summary:** {gp_rec.summary}")
                        st.caption(f"🛡️ **QMJ Audit Digest:** `{gp_rec.signature[:16]}...` (Verified: `{gp_rec.verify_integrity()}`)")
                    except Exception as gp_ex:
                        st.info(f"Gross profitability evaluation deferred: {gp_ex}")
                else:
                    st.info("ℹ️ Insufficient price history to evaluate gross profitability.")

                st.markdown("---")
                st.subheader("🚀 Jegadeesh & Titman (1993) / Antonacci Dual Momentum & Industry Relative Strength")
                st.caption(
                    "Intermediate momentum (12-2, 6-1) with 1-month short-term microstructure reversal separation, "
                    "Moskowitz & Grinblatt (1999) idiosyncratic alpha decomposition, and Antonacci dual absolute trend filtering."
                )
                from src.analytics.dual_momentum import evaluate_dual_momentum_profile
                if len(df_obs) >= 42:
                    try:
                        dm_cols = st.columns(2)
                        with dm_cols[0]:
                            bench_input = st.text_input("Benchmark Ticker", value="SPY", key="dm_bench_ticker")
                        with dm_cols[1]:
                            rf_input = st.number_input("Risk-Free Rate (%/yr)", min_value=0.0, max_value=20.0, value=4.0, step=0.5, key="dm_rf_rate")

                        dm_rec = evaluate_dual_momentum_profile(
                            ticker=selected_ticker,
                            benchmark_ticker=bench_input.strip() if bench_input else "SPY",
                            rf_rate_pct=float(rf_input),
                        )

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric(
                            "12-2 Momentum",
                            f"{dm_rec.momentum_12_2_pct:+.1f}%",
                            delta=f"6-1: {dm_rec.momentum_6_1_pct:+.1f}%",
                            help="Jegadeesh & Titman (1993) intermediate momentum skipping recent month (t-252 to t-21)",
                        )
                        m2.metric(
                            "1-Month Reversal",
                            f"{dm_rec.reversal_1_0_pct:+.1f}%",
                            delta="Short-Term Bounce",
                            help="Microstructure return over recent 21 trading days (t-21 to t)",
                        )
                        trend_label = "BULLISH" if dm_rec.is_absolute_trend_positive else "BEARISH"
                        m3.metric(
                            "Relative Spread vs Bench",
                            f"{dm_rec.relative_momentum_spread_pct:+.1f}%",
                            delta=f"Idio Alpha: {dm_rec.idiosyncratic_momentum_pct:+.1f}%",
                            help="Relative momentum spread over benchmark (SPY) and beta-adjusted idiosyncratic alpha",
                        )
                        crash_badge = "⚠️ CRASH RISK" if dm_rec.has_momentum_crash_risk else f"Trend: {trend_label}"
                        m4.metric(
                            "Sharpe Momentum",
                            f"{dm_rec.sharpe_momentum:.2f}",
                            delta=crash_badge,
                            help="Volatility-adjusted intermediate momentum ratio (R_{12,2} / Ann Vol)",
                        )

                        st.caption(f"🚀 **Dual Momentum Summary:** {dm_rec.summary}")
                        st.caption(f"🛡️ **Momentum Audit Digest:** `{dm_rec.signature[:16]}...` (Verified: `{dm_rec.verify_integrity()}`)")
                    except Exception as dm_ex:
                        st.info(f"Dual momentum evaluation deferred: {dm_ex}")
                else:
                    st.info("ℹ️ Requires at least 42 price sessions to evaluate dual momentum.")

                st.markdown("---")
                st.subheader("⚖️ Andrea Frazzini & Lasse Heje Pedersen (2014) Betting Against Beta (BAB)")
                st.caption(
                    "Structural leverage constraint engine: estimates Vasicek (1973) Bayesian shrunk beta, "
                    "Jensen's alpha on the flattened Security Market Line, and optimal zero-beta leverage multipliers (1/beta)."
                )
                from src.analytics.betting_against_beta import evaluate_bab_profile
                if len(df_obs) >= 42:
                    try:
                        b_cols = st.columns(3)
                        with b_cols[0]:
                            bab_bench_input = st.text_input("Benchmark Ticker", value="SPY", key="bab_bench_ticker")
                        with b_cols[1]:
                            bab_rf_input = st.number_input("Risk-Free Rate (%/yr)", min_value=0.0, max_value=20.0, value=4.0, step=0.5, key="bab_rf_rate")
                        with b_cols[2]:
                            bab_erp_input = st.number_input("Equity Risk Premium (%/yr)", min_value=1.0, max_value=15.0, value=6.0, step=0.5, key="bab_erp")

                        bab_rec = evaluate_bab_profile(
                            ticker=selected_ticker,
                            benchmark_ticker=bab_bench_input.strip() if bab_bench_input else "SPY",
                            rf_rate_pct=float(bab_rf_input),
                            equity_risk_premium_pct=float(bab_erp_input),
                        )

                        b1, b2, b3, b4 = st.columns(4)
                        b1.metric(
                            "Vasicek Shrunk Beta",
                            f"{bab_rec.shrunk_beta:.2f}",
                            delta=f"Raw: {bab_rec.raw_beta:.2f} (ρ={bab_rec.correlation_to_benchmark:.2f})",
                            help="Vasicek (1973) Bayesian shrinkage beta toward market unity: 0.60·β_raw + 0.40·1.0",
                        )
                        b2.metric(
                            "Jensen's Alpha",
                            f"{bab_rec.jensen_alpha_pct:+.2f}% Ann",
                            delta=f"Realized: {bab_rec.realized_annualized_return_pct:+.1f}%",
                            help="Excess risk-adjusted return over CAPM expected benchmark return",
                        )
                        b3.metric(
                            "Optimal BAB Multiplier",
                            f"{bab_rec.bab_leverage_multiplier:.2f}x",
                            delta=f"Spread: {bab_rec.expected_bab_spread_pct:+.1f}%/yr",
                            help="Frazzini-Pedersen optimal zero-beta portfolio weighting: 1.0 / beta",
                        )
                        overpriced_badge = "⚠️ OVERPRICED DRAG" if bab_rec.is_high_beta_overpriced else ("💎 LOW-BETA ALPHA" if bab_rec.is_low_beta_compounder else f"Regime: {bab_rec.regime}")
                        b4.metric(
                            "BAB Factor Regime",
                            bab_rec.regime,
                            delta=overpriced_badge,
                            help="Asset classification according to leverage constraints and flattened SML dynamics",
                        )

                        st.caption(f"⚖️ **BAB Summary:** {bab_rec.summary}")
                        st.caption(f"🛡️ **BAB Audit Digest:** `{bab_rec.signature[:16]}...` (Verified: `{bab_rec.verify_integrity()}`)")
                    except Exception as bab_ex:
                        st.info(f"Betting Against Beta evaluation deferred: {bab_ex}")
                else:
                    st.info("ℹ️ Requires at least 42 price sessions to evaluate Betting Against Beta.")

                st.markdown("---")
                st.subheader("🔄 Narasimhan Jegadeesh (1990) & Bruce Lehmann (1990) Short-Term Weekly Reversal")
                st.caption(
                    "Microstructure mean-reversion engine: computes 1-week/2-week returns, daily first-order serial correlation (ρ1), "
                    "Roll (1984) bid-ask bounce spread, and trend confluence to separate high-conviction liquidity dips from falling knives."
                )
                from src.analytics.short_term_reversal import evaluate_short_term_reversal_profile
                if len(df_obs) >= 30:
                    try:
                        r_cols = st.columns(2)
                        with r_cols[0]:
                            rev_bench_input = st.text_input("Benchmark Ticker", value="SPY", key="rev_bench_ticker")
                        with r_cols[1]:
                            st.write("")
                            st.caption("Benchmark reference for relative 1-week spread calculation")

                        rev_rec = evaluate_short_term_reversal_profile(
                            ticker=selected_ticker,
                            benchmark_ticker=rev_bench_input.strip() if rev_bench_input else "SPY",
                        )

                        rv1, rv2, rv3, rv4 = st.columns(4)
                        rv1.metric(
                            "1-Week Return (R_1w)",
                            f"{rev_rec.return_1w_pct:+.2f}%",
                            delta=f"2W: {rev_rec.return_2w_pct:+.1f}%",
                            help="Trailing 5-day return (t-5 to t)",
                        )
                        rv2.metric(
                            "Serial Correlation (ρ1)",
                            f"{rev_rec.serial_correlation_rho1:.2f}",
                            delta=f"Roll Spread: {rev_rec.roll_spread_proxy_pct:.2f}%",
                            help="Daily first-order autocovariance; negative value indicates active mean-reverting dynamics",
                        )
                        dip_badge = "🎯 DIP BUY" if rev_rec.is_liquidity_dip_buy else ("⚠️ FALLING KNIFE" if rev_rec.is_falling_knife else f"Alpha: {rev_rec.expected_contrarian_alpha_pct:+.1f}%/yr")
                        rv3.metric(
                            "Expected Rebound Alpha",
                            f"{rev_rec.expected_contrarian_alpha_pct:+.2f}% Ann",
                            delta=dip_badge,
                            help="Expected annualized contrarian rebound return net of Roll bid-ask bounce friction",
                        )
                        rv4.metric(
                            "Reversal Regime",
                            rev_rec.regime,
                            delta="Trend: " + ("BULLISH" if rev_rec.is_bullish_intermediate_trend else "BEARISH"),
                            help="Short-term contrarian classification paired with 12-2 intermediate momentum",
                        )

                        st.caption(f"🔄 **Reversal Summary:** {rev_rec.summary}")
                        st.caption(f"🛡️ **Reversal Audit Digest:** `{rev_rec.signature[:16]}...` (Verified: `{rev_rec.verify_integrity()}`)")
                    except Exception as rev_ex:
                        st.info(f"Short-term reversal evaluation deferred: {rev_ex}")
                else:
                    st.info("ℹ️ Requires at least 30 price sessions to evaluate short-term reversal.")

                st.markdown("---")
                st.subheader("⚠️ John Y. Campbell et al. (2008) Financial Distress Risk Hazard Engine")
                st.caption(
                    "Empirical multi-period logit hazard model forecasting 12-month corporate failure probability and "
                    "solving the Financial Distress Risk Puzzle to eliminate chronically underperforming zombie value traps."
                )
                from src.analytics.distress_risk import evaluate_distress_risk_profile
                if len(df_obs) >= 1:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        latest_p = float(df_obs[p_col].dropna().iloc[-1])
                        est_mcap_b = round(latest_p * 0.05, 2)
                        est_debt_b = round(est_mcap_b * 0.40, 2)
                        est_ni_b = round(est_mcap_b * 0.07, 2)
                        est_cash_b = round(est_mcap_b * 0.15, 2)

                        dst_cols = st.columns(4)
                        with dst_cols[0]:
                            mcap_input = st.number_input("Market Cap ($B)", min_value=0.1, max_value=3000.0, value=max(0.1, est_mcap_b), step=5.0, key="dst_mcap_b")
                        with dst_cols[1]:
                            debt_input = st.number_input("Total Debt ($B)", min_value=0.0, max_value=2000.0, value=max(0.0, est_debt_b), step=2.0, key="dst_debt_b")
                        with dst_cols[2]:
                            ni_input = st.number_input("Net Income ($B)", min_value=-500.0, max_value=500.0, value=est_ni_b, step=1.0, key="dst_ni_b")
                        with dst_cols[3]:
                            cash_input = st.number_input("Cash Buffer ($B)", min_value=0.0, max_value=1000.0, value=max(0.0, est_cash_b), step=2.0, key="dst_cash_b")

                        dst_rec = evaluate_distress_risk_profile(
                            ticker=selected_ticker,
                            market_cap=mcap_input * 1_000_000_000.0,
                            total_liabilities=debt_input * 1_000_000_000.0,
                            net_income=ni_input * 1_000_000_000.0,
                            cash_equivalents=cash_input * 1_000_000_000.0,
                        )

                        d1, d2, d3, d4 = st.columns(4)
                        hazard_badge = "⚠️ HIGH HAZARD" if dst_rec.is_distress_trap else ("💎 SAFE HAVEN" if dst_rec.is_solvent_safe_haven else "NORMAL")
                        d1.metric(
                            "12M Failure Hazard",
                            f"{dst_rec.distress_probability_12m_pct:.2f}%",
                            delta=hazard_badge,
                            help="Campbell, Hilscher & Szilagyi (2008) multi-period dynamic logit failure probability over 12 months",
                        )
                        d2.metric(
                            "Market Leverage (TLMTA)",
                            f"{dst_rec.tlmta_pct:.1f}%",
                            delta=f"Cash: {dst_rec.cashmta_pct:.1f}%",
                            help="Total liabilities divided by Market-Adjusted Total Assets (MTA = Market Cap + Debt)",
                        )
                        d3.metric(
                            "Capital Return (NIMTA)",
                            f"{dst_rec.nimta_pct:+.2f}%",
                            delta=f"Vol: {dst_rec.annualized_volatility_pct:.1f}%",
                            help="Net income divided by Market-Adjusted Total Assets; key gauge of capital efficiency",
                        )
                        d4.metric(
                            "Expected Distress Alpha",
                            f"{dst_rec.expected_distress_alpha_pct:+.2f}% Ann",
                            delta=f"Regime: {dst_rec.regime}",
                            help="Expected annualized return spread; distressed equities incur heavy -8% to -14% annual underperformance",
                        )

                        st.caption(f"⚠️ **Distress Summary:** {dst_rec.summary}")
                        st.caption(f"🛡️ **Distress Audit Digest:** `{dst_rec.signature[:16]}...` (Verified: `{dst_rec.verify_integrity()}`)")
                    except Exception as dst_ex:
                        st.info(f"Financial distress evaluation deferred: {dst_ex}")
                else:
                    st.info("ℹ️ Insufficient price history to evaluate financial distress hazard.")

                st.markdown("---")
                st.subheader("🎭 Malcolm Baker & Jeffrey Wurgler (2006) Cross-Sectional Investor Sentiment")
                st.caption(
                    "Limits-to-arbitrage & subjectivity engine: calculates Firm Speculative Index (S_i), sentiment beta, "
                    "and predicts cross-sectional sentiment reversal alpha across euphoric and panicked macro sentiment regimes."
                )
                from src.analytics.investor_sentiment import evaluate_investor_sentiment_profile
                if len(df_obs) >= 20:
                    try:
                        s_cols = st.columns(3)
                        with s_cols[0]:
                            sent_slider = st.slider(
                                "Macro Market Sentiment (I_sent)",
                                min_value=-3.0,
                                max_value=3.0,
                                value=0.0,
                                step=0.25,
                                help="Standardized macro sentiment wave: +2.0 = Euphoria/Bubble, -2.0 = Panic/Depression",
                                key="sent_macro_slider",
                            )
                        with s_cols[1]:
                            margin_input = st.number_input("Net Margin (%/yr)", min_value=-100.0, max_value=80.0, value=12.5, step=1.0, key="sent_margin_input")
                        with s_cols[2]:
                            st.write("")
                            st.caption("Adjust macro sentiment to observe cross-sectional reversal alpha")

                        sent_rec = evaluate_investor_sentiment_profile(
                            ticker=selected_ticker,
                            market_sentiment_index=float(sent_slider),
                            net_profit_margin_pct=float(margin_input),
                        )

                        sn1, sn2, sn3, sn4 = st.columns(4)
                        sn1.metric(
                            "Firm Speculative Index",
                            f"{sent_rec.speculative_index:.1f} / 100",
                            delta=f"Vol: {sent_rec.annualized_volatility_pct:.1f}%",
                            help="Baker & Wurgler subjectivity score: high score indicates hard to value and hard to arbitrage",
                        )
                        sn2.metric(
                            "Sentiment Beta",
                            f"{sent_rec.sentiment_beta:+.2f}",
                            delta=f"MAX_5: {sent_rec.max_daily_return_5d_pct:.1f}%",
                            help="Sensitivity to retail sentiment waves: beta_sent = (S_i - 50) / 25",
                        )
                        bubble_badge = "⚠️ BUBBLE CRASH" if sent_rec.is_speculative_bubble else ("💎 CONTRARIAN BUY" if sent_rec.is_contrarian_bargain else f"Regime: {sent_rec.regime}")
                        sn3.metric(
                            "Expected Sentiment Alpha",
                            f"{sent_rec.expected_sentiment_alpha_pct:+.2f}% Ann",
                            delta=bubble_badge,
                            help="Projected annualized cross-sectional sentiment alpha: alpha_sent = -beta_sent * I_sent * 4.5%",
                        )
                        sn4.metric(
                            "Sentiment Regime",
                            sent_rec.regime,
                            delta=f"Macro: {sent_rec.market_sentiment_index:+.2f}σ",
                            help="Cross-sectional sentiment classification based on asset subjectivity and macro sentiment state",
                        )

                        st.caption(f"🎭 **Sentiment Summary:** {sent_rec.summary}")
                        st.caption(f"🛡️ **Sentiment Audit Digest:** `{sent_rec.signature[:16]}...` (Verified: `{sent_rec.verify_integrity()}`)")
                    except Exception as sent_ex:
                        st.info(f"Investor sentiment evaluation deferred: {sent_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate investor sentiment.")

                st.markdown("---")
                st.subheader("⚖️ Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Arbitrage Asymmetry & Short-Sale Impediments")
                st.caption(
                    "Structural short-sale friction and anomaly overpricing engine: models borrow friction, idiosyncratic volatility risk, "
                    "short interest, days-to-cover, and evaluates asymmetric overpricing hazard when positive sentiment interacts with shorting limits."
                )
                from src.analytics.arbitrage_asymmetry import evaluate_arbitrage_asymmetry_profile
                if len(df_obs) >= 20:
                    try:
                        asym_cols = st.columns(3)
                        with asym_cols[0]:
                            asym_borrow = st.number_input("Borrow Fee Rate (%/yr)", min_value=0.25, max_value=80.0, value=1.25, step=0.5, key="asym_borrow_fee")
                        with asym_cols[1]:
                            asym_si = st.number_input("Short Interest (% Float)", min_value=0.0, max_value=80.0, value=5.0, step=1.0, key="asym_short_interest")
                        with asym_cols[2]:
                            asym_dtc = st.number_input("Days to Cover (DTC)", min_value=0.5, max_value=30.0, value=3.0, step=0.5, key="asym_days_to_cover")

                        asym_rec = evaluate_arbitrage_asymmetry_profile(
                            ticker=selected_ticker,
                            market_sentiment_index=float(st.session_state.get("sent_macro_slider", 0.0)),
                            borrow_fee_annual_pct=float(asym_borrow),
                            short_interest_pct_float=float(asym_si),
                            days_to_cover=float(asym_dtc),
                        )

                        ay1, ay2, ay3, ay4 = st.columns(4)
                        ay1.metric(
                            "Short Impediment Index",
                            f"{asym_rec.short_sale_impediment_index:.1f} / 100",
                            delta=f"Borrow: {asym_rec.borrow_fee_annual_pct:.2f}%",
                            help="Stambaugh et al. short-sale friction index: synthesizes borrow fees, idiosyncratic volatility, and lendable supply scarcity",
                        )
                        hazard_badge = "⚠️ SEVERE OVERPRICED" if asym_rec.is_severe_overpriced else f"Hazard: {asym_rec.asymmetric_overpricing_hazard:.1f}"
                        ay2.metric(
                            "Overpricing Hazard Score",
                            f"{asym_rec.asymmetric_overpricing_hazard:.1f} / 100",
                            delta=hazard_badge,
                            help="Asymmetric overpricing hazard: activated when positive sentiment interacts with high shorting impediments",
                        )
                        squeeze_badge = "🔥 SQUEEZE RISK" if asym_rec.is_squeeze_risk else f"DTC: {asym_rec.days_to_cover:.1f}D"
                        ay3.metric(
                            "Squeeze Vulnerability",
                            f"{asym_rec.short_squeeze_vulnerability:.1f} / 100",
                            delta=squeeze_badge,
                            help="Short squeeze susceptibility based on borrow utilization, days to cover, and short interest concentration",
                        )
                        ay4.metric(
                            "Expected Overpricing Drag",
                            f"{asym_rec.expected_asymmetry_alpha_pct:+.2f}% Ann",
                            delta=f"Regime: {asym_rec.regime}",
                            help="Projected annualized underperformance as anomaly short-leg overpricing unravels",
                        )

                        st.caption(f"⚖️ **Arbitrage Summary:** {asym_rec.summary}")
                        st.caption(f"🛡️ **Arbitrage Audit Digest:** `{asym_rec.signature[:16]}...` (Verified: `{asym_rec.verify_integrity()}`)")
                    except Exception as asym_ex:
                        st.info(f"Arbitrage asymmetry evaluation deferred: {asym_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate arbitrage asymmetry.")

                st.markdown("---")
                st.subheader("🏭 Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor & Corporate Investment Engine")
                st.caption(
                    "Neoclassical investment asset pricing engine: links expected returns directly to corporate investment optimality "
                    "(Expected Return = Marginal Profitability / Marginal Cost of Investment). Disentangles high-ROE disciplined compounders "
                    "from capital-destructive over-investors across the corporate cross-section."
                )
                from src.analytics.q_factor import evaluate_q_factor_profile
                if len(df_obs) >= 20:
                    try:
                        q_cols = st.columns(3)
                        with q_cols[0]:
                            q_roe_input = st.number_input("Return on Equity (ROE %)", min_value=-50.0, max_value=80.0, value=18.0, step=1.0, key="q_roe_input")
                        with q_cols[1]:
                            q_inv_input = st.number_input("Investment / Asset Growth (I/A %)", min_value=-20.0, max_value=80.0, value=5.5, step=0.5, key="q_inv_input")
                        with q_cols[2]:
                            q_tobin_input = st.number_input("Tobin's q Proxy", min_value=0.2, max_value=20.0, value=1.75, step=0.1, key="q_tobin_input")

                        q_rec = evaluate_q_factor_profile(
                            ticker=selected_ticker,
                            return_on_equity_pct=float(q_roe_input),
                            investment_to_assets_pct=float(q_inv_input),
                            tobin_q_proxy=float(q_tobin_input),
                        )

                        qy1, qy2, qy3, qy4 = st.columns(4)
                        compounder_badge = "💎 COMPOUNDER" if q_rec.is_disciplined_compounder else ("⚠️ OVER-INVESTOR" if q_rec.is_overinvestor_trap else f"Regime: {q_rec.regime}")
                        qy1.metric(
                            "Composite q-Alpha Score",
                            f"{q_rec.q_alpha_score:.1f} / 100",
                            delta=compounder_badge,
                            help="Hou-Xue-Zhang composite score: balances high ROE profitability against conservative asset growth",
                        )
                        qy2.metric(
                            "Profitability Score (ROE)",
                            f"{q_rec.profitability_score:.1f} / 100",
                            delta=f"ROE: {q_rec.return_on_equity_pct:+.1f}%",
                            help="Marginal corporate profitability: higher ROE demands higher cost of capital and expected return",
                        )
                        qy3.metric(
                            "Investment Score (I/A)",
                            f"{q_rec.investment_score:.1f} / 100",
                            delta=f"Growth: {q_rec.investment_to_assets_pct:+.1f}%",
                            help="Asset expansion rate: lower investment indicates capital discipline and higher future returns",
                        )
                        qy4.metric(
                            "Expected q-Factor Alpha",
                            f"{q_rec.expected_q_alpha_pct:+.2f}% Ann",
                            delta=f"Tobin's q: {q_rec.tobin_q_proxy:.2f}",
                            help="Projected annualized q-factor alpha spread: alpha_q = (S_q - 50) * 0.16%",
                        )

                        st.caption(f"🏭 **q-Factor Summary:** {q_rec.summary}")
                        st.caption(f"🛡️ **q-Factor Audit Digest:** `{q_rec.signature[:16]}...` (Verified: `{q_rec.verify_integrity()}`)")
                    except Exception as q_ex:
                        st.info(f"q-Factor investment evaluation deferred: {q_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate q-factor investment profile.")

                st.markdown("---")
                st.subheader("📊 Richard G. Sloan (1996) Accrual Anomaly & Earnings Quality Engine")
                st.caption(
                    "Earnings quality decomposition engine: separates reported net income into persistent operating cash flows "
                    "and non-cash accounting accruals. Identifies high-cash-flow quality compounders while screening out "
                    "accrual manipulation traps vulnerable to sharp mean-reversion crashes."
                )
                from src.analytics.accrual_anomaly import evaluate_accrual_anomaly_profile
                if len(df_obs) >= 20:
                    try:
                        acc_cols = st.columns(3)
                        with acc_cols[0]:
                            acc_ni_input = st.number_input("Reported Net Income ($M)", min_value=-50_000.0, max_value=200_000.0, value=1_200.0, step=100.0, key="acc_ni_input")
                        with acc_cols[1]:
                            acc_cfo_input = st.number_input("Operating Cash Flow ($M)", min_value=-50_000.0, max_value=250_000.0, value=1_500.0, step=100.0, key="acc_cfo_input")
                        with acc_cols[2]:
                            acc_assets_input = st.number_input("Average Total Assets ($M)", min_value=100.0, max_value=1_000_000.0, value=12_000.0, step=500.0, key="acc_assets_input")

                        acc_rec = evaluate_accrual_anomaly_profile(
                            ticker=selected_ticker,
                            net_income=float(acc_ni_input * 1_000_000.0),
                            operating_cash_flow=float(acc_cfo_input * 1_000_000.0),
                            average_total_assets=float(acc_assets_input * 1_000_000.0),
                        )

                        ay1, ay2, ay3, ay4 = st.columns(4)
                        quality_badge = "💎 CASH QUALITY" if acc_rec.is_cash_quality_compounder else ("⚠️ ACCRUAL TRAP" if acc_rec.is_accrual_trap else f"Regime: {acc_rec.regime}")
                        ay1.metric(
                            "Earnings Quality Index",
                            f"{acc_rec.earnings_quality_index:.1f} / 100",
                            delta=quality_badge,
                            help="Sloan Earnings Quality Index: high index indicates earnings driven by cash flows rather than paper accruals",
                        )
                        ay2.metric(
                            "Accrual Ratio (CFA)",
                            f"{acc_rec.accrual_ratio_pct:+.2f}%",
                            delta=f"Accruals: ${acc_rec.total_accruals / 1e6:,.0f}M",
                            help="Normalized accruals / average assets: negative accruals reflect conservative accounting and cash generation",
                        )
                        ay3.metric(
                            "Cash Conversion Multiplier",
                            f"{acc_rec.cash_flow_to_net_income:.2f}x",
                            delta=f"CFO: ${acc_rec.operating_cash_flow / 1e6:,.0f}M",
                            help="Ratio of operating cash flow to net income (CFNI): healthy franchises consistently convert >1.0x",
                        )
                        ay4.metric(
                            "Expected Accrual Alpha",
                            f"{acc_rec.expected_accrual_alpha_pct:+.2f}% Ann",
                            delta=f"Regime: {acc_rec.regime}",
                            help="Projected annualized accrual reversal alpha spread: alpha = (EQI - 50) * 0.18%",
                        )

                        st.caption(f"📊 **Accrual Summary:** {acc_rec.summary}")
                        st.caption(f"🛡️ **Accrual Audit Digest:** `{acc_rec.signature[:16]}...` (Verified: `{acc_rec.verify_integrity()}`)")
                    except Exception as acc_ex:
                        st.info(f"Accrual anomaly evaluation deferred: {acc_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate accrual anomaly profile.")

                st.markdown("---")
                st.subheader("🕵️ Messod D. Beneish (1999) Forensic Accounting M-Score & Manipulation Engine")
                st.caption(
                    "Forensic accounting probit model: synthesizes 8 structural balance-sheet and income-statement indices "
                    "(DSRI, GMI, AQI, SGI, DEPI, SGAI, LVGI, TATA) to calculate the probability of corporate financial statement manipulation. "
                    "Scores M > -1.78 identify high-risk earnings manipulation and impending restatement crash risks."
                )
                from src.analytics.beneish_mscore import evaluate_beneish_mscore_profile
                if len(df_obs) >= 20:
                    try:
                        ben_cols = st.columns(4)
                        with ben_cols[0]:
                            ben_dsri = st.number_input("Days' Sales in Receivables (DSRI)", min_value=0.2, max_value=5.0, value=1.05, step=0.05, key="ben_dsri")
                        with ben_cols[1]:
                            ben_gmi = st.number_input("Gross Margin Index (GMI)", min_value=0.2, max_value=5.0, value=1.02, step=0.05, key="ben_gmi")
                        with ben_cols[2]:
                            ben_aqi = st.number_input("Asset Quality Index (AQI)", min_value=0.2, max_value=5.0, value=1.04, step=0.05, key="ben_aqi")
                        with ben_cols[3]:
                            ben_sgi = st.number_input("Sales Growth Index (SGI)", min_value=0.2, max_value=5.0, value=1.08, step=0.05, key="ben_sgi")

                        ben_rec = evaluate_beneish_mscore_profile(
                            ticker=selected_ticker,
                            dsri=float(ben_dsri),
                            gmi=float(ben_gmi),
                            aqi=float(ben_aqi),
                            sgi=float(ben_sgi),
                        )

                        by1, by2, by3, by4 = st.columns(4)
                        fraud_badge = "🚨 FRAUD RED FLAG" if ben_rec.is_manipulator_risk else ("🛡️ CLEAN FINANCIALS" if ben_rec.is_clean_profile else f"Regime: {ben_rec.regime}")
                        by1.metric(
                            "Beneish M-Score",
                            f"{ben_rec.m_score:.2f}",
                            delta=fraud_badge,
                            help="Beneish 8-Variable M-Score: scores greater than -1.78 indicate high probability of financial statement manipulation",
                        )
                        by2.metric(
                            "Manipulation Probability",
                            f"{ben_rec.manipulation_probability_pct:.1f}%",
                            delta=f"Cutoff: > 3.8%",
                            help="Probit probability Phi(M) of intentional accounting distortion or fraud",
                        )
                        by3.metric(
                            "Revenue & Margin Flags",
                            f"DSRI: {ben_rec.dsri:.2f}",
                            delta=f"GMI: {ben_rec.gmi:.2f}",
                            help="Days' sales in receivables and gross margin deterioration indices",
                        )
                        by4.metric(
                            "Expected Forensic Drag",
                            f"{ben_rec.expected_forensic_alpha_pct:+.2f}% Ann",
                            delta=f"Regime: {ben_rec.regime}",
                            help="Projected annualized underperformance associated with forensic manipulation risk",
                        )

                        st.caption(f"🕵️ **Beneish Summary:** {ben_rec.summary}")
                        st.caption(f"🛡️ **Beneish Audit Digest:** `{ben_rec.signature[:16]}...` (Verified: `{ben_rec.verify_integrity()}`)")
                    except Exception as ben_ex:
                        st.info(f"Beneish M-Score evaluation deferred: {ben_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Beneish M-Score profile.")

                st.markdown("---")
                st.subheader("🏆 Joseph D. Piotroski (2000) Fundamental Accounting F-Score Engine")
                st.caption(
                    "Value separation & fundamental health engine: scores 9 discrete accounting signals across "
                    "Profitability (4 pts), Leverage/Liquidity (3 pts), and Operating Efficiency (2 pts). "
                    "Isolates high-conviction fundamental compounders (F >= 8) while eliminating value traps (F <= 3)."
                )
                from src.analytics.piotroski_fscore import evaluate_piotroski_fscore_profile
                if len(df_obs) >= 20:
                    try:
                        f_cols = st.columns(3)
                        with f_cols[0]:
                            f_roa = st.number_input("Return on Assets (ROA %)", min_value=-50.0, max_value=80.0, value=12.0, step=1.0, key="f_roa_input")
                        with f_cols[1]:
                            f_droa = st.number_input("Change in ROA (ΔROA %)", min_value=-20.0, max_value=30.0, value=1.2, step=0.2, key="f_droa_input")
                        with f_cols[2]:
                            f_cfo_mult = st.number_input("CFO / Net Income Multiplier", min_value=0.2, max_value=5.0, value=1.25, step=0.05, key="f_cfo_mult")

                        assets_proxy = 10_000_000_000.0
                        ni_proxy = assets_proxy * (float(f_roa) / 100.0)
                        cfo_proxy = ni_proxy * float(f_cfo_mult)

                        f_rec = evaluate_piotroski_fscore_profile(
                            ticker=selected_ticker,
                            net_income=ni_proxy,
                            operating_cash_flow=cfo_proxy,
                            total_assets=assets_proxy,
                            return_on_assets_pct=float(f_roa),
                            delta_roa_pct=float(f_droa),
                        )

                        fy1, fy2, fy3, fy4 = st.columns(4)
                        compounder_badge = "💎 COMPOUNDER" if f_rec.is_strong_compounder else ("⚠️ VALUE TRAP" if f_rec.is_value_trap else f"Regime: {f_rec.regime}")
                        fy1.metric(
                            "Piotroski F-Score",
                            f"{f_rec.f_score} / 9",
                            delta=compounder_badge,
                            help="Aggregate 9-point fundamental accounting score",
                        )
                        fy2.metric(
                            "Profitability Points",
                            f"{f_rec.profitability_points} / 4",
                            delta=f"ROA: {f_rec.return_on_assets_pct:+.1f}%",
                            help="Subtotal: ROA > 0, CFO > 0, ΔROA > 0, and CFO > Net Income",
                        )
                        fy3.metric(
                            "Leverage & Liquidity",
                            f"{f_rec.leverage_liquidity_points} / 3",
                            delta="Deleveraged" if f_rec.f_delta_leverage else "Debt Expanded",
                            help="Subtotal: De-leveraging, current ratio expansion, and zero share dilution",
                        )
                        fy4.metric(
                            "Expected Fundamental Alpha",
                            f"{f_rec.expected_fundamental_alpha_pct:+.2f}% Ann",
                            delta=f"Efficiency: {f_rec.operating_efficiency_points}/2",
                            help="Projected annualized alpha spread: alpha_F = (F - 4.5) * 1.6%",
                        )

                        st.caption(f"🏆 **F-Score Summary:** {f_rec.summary}")
                        st.caption(f"🛡️ **F-Score Audit Digest:** `{f_rec.signature[:16]}...` (Verified: `{f_rec.verify_integrity()}`)")
                    except Exception as f_ex:
                        st.info(f"Piotroski F-Score evaluation deferred: {f_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Piotroski F-Score profile.")

                st.markdown("---")
                st.subheader("🌊 Harrison Hong & Jeremy C. Stein (1999) Gradual Information Diffusion Engine")
                st.caption(
                    "Unified behavioral asset pricing engine (Hong & Stein 1999; Chordia & Shivakumar 2002): "
                    "models gradual news diffusion across boundedly rational newswatchers (underreaction expansion) "
                    "versus extrapolative trend chasing (overreaction peaks, momentum crowding, and crash hazard)."
                )
                from src.analytics.gradual_diffusion import evaluate_gradual_diffusion_profile
                if len(df_obs) >= 20:
                    try:
                        d_cols = st.columns(2)
                        with d_cols[0]:
                            d_cov = st.number_input(
                                "Analyst Coverage Count (Proxy)",
                                min_value=1,
                                max_value=80,
                                value=15,
                                step=1,
                                key="diff_cov_input",
                            )
                        with d_cols[1]:
                            st.write("")
                            st.write("")
                            st.caption("ℹ️ Higher analyst coverage accelerates diffusion velocity (IDV), compressing underreaction lag.")

                        d_rec = evaluate_gradual_diffusion_profile(
                            ticker=selected_ticker,
                            analyst_coverage=int(d_cov),
                        )

                        dy1, dy2, dy3, dy4 = st.columns(4)
                        crowd_badge = (
                            "⚠️ CROWDED PEAK"
                            if d_rec.is_crowded_overreaction
                            else ("🚀 UNDERREACTION" if d_rec.is_early_underreaction else "STABLE DYNAMICS")
                        )
                        dy1.metric(
                            "Diffusion Velocity (IDV)",
                            f"{d_rec.information_diffusion_velocity:.2f} / 1.00",
                            delta=f"Turnover: {d_rec.turnover_ratio_pct:.2f}%",
                            help="Speed at which private information permeates market prices",
                        )
                        dy2.metric(
                            "Trend Maturity (TMI)",
                            f"{d_rec.trend_maturity_index:.2f} / 1.00",
                            delta=f"Momentum: {d_rec.cumulative_momentum_pct:+.1f}%",
                            help="Degree of speculative exhaustion and trend aging",
                        )
                        dy3.metric(
                            "Momentum Crowding Risk",
                            f"{d_rec.momentum_crowding_risk_pct:.1f}%",
                            delta=crowd_badge,
                            delta_color="inverse" if d_rec.is_crowded_overreaction else "normal",
                            help="Estimated probability of sharp momentum reversal / crash",
                        )
                        dy4.metric(
                            "Expected Hong-Stein Alpha",
                            f"{d_rec.expected_hong_stein_alpha_pct:+.2f}% Ann",
                            delta=f"Regime: {d_rec.regime}",
                            help="Projected annualized alpha spread based on diffusion phase",
                        )

                        st.caption(f"🌊 **Diffusion Summary:** {d_rec.summary}")
                        st.caption(f"🛡️ **Diffusion Audit Digest:** `{d_rec.signature[:16]}...` (Verified: `{d_rec.verify_integrity()}`)")
                    except Exception as d_ex:
                        st.info(f"Hong-Stein Diffusion evaluation deferred: {d_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Hong-Stein diffusion profile.")

                st.markdown("---")
                st.subheader("💎 Josef Lakonishok, Andrei Shleifer & Robert W. Vishny (1994) Contrarian Investment Engine")
                st.caption(
                    "Extrapolation bias & glamour overpricing engine (LSV 1994; Fama & French 1998): "
                    "identifies systematic investor over-extrapolation of past growth. Contrarian value equities "
                    "command asymmetric upside as low expectation hurdles are cleared, whereas glamour equities "
                    "suffer severe multiple compression upon earnings normalization."
                )
                from src.analytics.contrarian_extrapolation import evaluate_contrarian_extrapolation_profile
                if len(df_obs) >= 20:
                    try:
                        c_cols = st.columns(3)
                        with c_cols[0]:
                            c_sg = st.number_input(
                                "Sales Growth (3-Yr Ann %)",
                                min_value=-50.0,
                                max_value=120.0,
                                value=8.5,
                                step=1.0,
                                key="lsv_sg_input",
                            )
                        with c_cols[1]:
                            c_eg = st.number_input(
                                "Earnings Growth (3-Yr Ann %)",
                                min_value=-60.0,
                                max_value=150.0,
                                value=10.0,
                                step=1.0,
                                key="lsv_eg_input",
                            )
                        with c_cols[2]:
                            c_pe = st.number_input(
                                "Trailing P/E Ratio",
                                min_value=-50.0,
                                max_value=250.0,
                                value=17.5,
                                step=1.0,
                                key="lsv_pe_input",
                            )

                        c_rec = evaluate_contrarian_extrapolation_profile(
                            ticker=selected_ticker,
                            sales_growth_3yr_pct=float(c_sg),
                            earnings_growth_3yr_pct=float(c_eg),
                            price_to_earnings=float(c_pe),
                        )

                        cy1, cy2, cy3, cy4 = st.columns(4)
                        c_badge = (
                            "💎 CONTRARIAN VALUE"
                            if c_rec.is_contrarian_value
                            else ("⚠️ GLAMOUR TRAP" if c_rec.is_glamour_trap else f"Regime: {c_rec.regime}")
                        )
                        cy1.metric(
                            "Naive Extrapolation (NEI)",
                            f"{c_rec.naive_extrapolation_index:.2f} / 1.00",
                            delta=f"SG: {c_rec.sales_growth_3yr_pct:+.1f}% | EG: {c_rec.earnings_growth_3yr_pct:+.1f}%",
                            help="Degree to which past growth track record is extrapolated forward",
                        )
                        cy2.metric(
                            "Valuation Multiple (VMC)",
                            f"{c_rec.valuation_multiple_composite:.2f} / 1.00",
                            delta=f"P/E: {c_rec.price_to_earnings:.1f} | P/CF: {c_rec.price_to_cash_flow:.1f}",
                            help="Composite percentile of valuation multiples (P/E, P/CF, P/B)",
                        )
                        cy3.metric(
                            "Expectation Discrepancy",
                            f"{c_rec.expectation_discrepancy:>+.2f}",
                            delta=c_badge,
                            delta_color="normal" if c_rec.is_contrarian_value else ("inverse" if c_rec.is_glamour_trap else "off"),
                            help="Valuation Multiple minus Naive Extrapolation Index (VMC - NEI)",
                        )
                        cy4.metric(
                            "Expected Contrarian Alpha",
                            f"{c_rec.expected_contrarian_alpha_pct:+.2f}% Ann",
                            delta="Positive Value Premium" if c_rec.expected_contrarian_alpha_pct > 0 else "Multiple Vulnerability",
                            help="Projected annualized alpha spread from multiple re-rating and expectation mean-reversion",
                        )

                        st.caption(f"💎 **Contrarian Summary:** {c_rec.summary}")
                        st.caption(f"🛡️ **Contrarian Audit Digest:** `{c_rec.signature[:16]}...` (Verified: `{c_rec.verify_integrity()}`)")
                    except Exception as c_ex:
                        st.info(f"LSV Contrarian evaluation deferred: {c_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate LSV contrarian profile.")

                st.markdown("---")
                st.subheader("🧠 Kent Daniel, David Hirshleifer & Avanidhar Subrahmanyam (1998) Overconfidence & Self-Attribution Engine")
                st.caption(
                    "Dual-bias psychological asset pricing engine (DHS 1998, 2001): models cognitive overconfidence "
                    "in private information signals and asymmetric self-attribution of subsequent public earnings outcomes. "
                    "Confirmatory public news accelerates momentum overshooting, whereas contradictory news builds up latent crash fragility."
                )
                from src.analytics.dhs_overconfidence import evaluate_dhs_overconfidence_profile
                if len(df_obs) >= 20:
                    try:
                        dh_cols = st.columns(2)
                        with dh_cols[0]:
                            dh_surp = st.number_input(
                                "Public Earnings Surprise (%)",
                                min_value=-60.0,
                                max_value=80.0,
                                value=4.5,
                                step=0.5,
                                key="dhs_surp_input",
                            )
                        with dh_cols[1]:
                            dh_disp = st.number_input(
                                "Analyst Forecast Dispersion (%)",
                                min_value=1.0,
                                max_value=50.0,
                                value=12.0,
                                step=0.5,
                                key="dhs_disp_input",
                            )

                        dhs_rec = evaluate_dhs_overconfidence_profile(
                            ticker=selected_ticker,
                            earnings_surprise_pct=float(dh_surp),
                            analyst_dispersion_pct=float(dh_disp),
                        )

                        dhy1, dhy2, dhy3, dhy4 = st.columns(4)
                        dh_badge = (
                            "⚠️ CONTRADICTION FRAGILITY"
                            if dhs_rec.is_contradiction_fragile
                            else ("🚀 ATTRIBUTION ACCEL" if dhs_rec.is_attribution_accelerating else f"Regime: {dhs_rec.regime}")
                        )
                        dhy1.metric(
                            "Overconfidence Intensity (OCI)",
                            f"{dhs_rec.overconfidence_intensity:.2f} / 1.00",
                            delta=f"Vol: {dhs_rec.annualized_volatility_pct:.1f}% | Turn: {dhs_rec.turnover_ratio_pct:.2f}%",
                            help="Overconfidence in private signal precision relative to public benchmarks",
                        )
                        dhy2.metric(
                            "Biased Self-Attribution (BAI)",
                            f"{dhs_rec.biased_self_attribution_index:>+.2f}",
                            delta="Confirmatory Amplification" if dhs_rec.biased_self_attribution_index > 0 else "Contradiction Denial",
                            help="Directional confirmation of prior momentum by recent public earnings news",
                        )
                        dhy3.metric(
                            "Reversal Crash Hazard",
                            f"{dhs_rec.reversal_crash_hazard_pct:.1f}%",
                            delta=dh_badge,
                            delta_color="inverse" if dhs_rec.is_contradiction_fragile else "normal",
                            help="Estimated probability of sharp corrective price reversal / crash",
                        )
                        dhy4.metric(
                            "Expected DHS Alpha",
                            f"{dhs_rec.expected_dhs_alpha_pct:+.2f}% Ann",
                            delta="Continuation Expansion" if dhs_rec.expected_dhs_alpha_pct > 0 else "Mean Reversion Drag",
                            help="Projected annualized alpha spread based on psychological life-cycle horizon",
                        )

                        st.caption(f"🧠 **DHS Summary:** {dhs_rec.summary}")
                        st.caption(f"🛡️ **DHS Audit Digest:** `{dhs_rec.signature[:16]}...` (Verified: `{dhs_rec.verify_integrity()}`)")
                    except Exception as dh_ex:
                        st.info(f"DHS Overconfidence evaluation deferred: {dh_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate DHS overconfidence profile.")

                st.markdown("---")
                st.subheader("⚖️ Nicholas Barberis, Andrei Shleifer & Robert Vishny (1998) Dual-Heuristic Sentiment Engine")
                st.caption(
                    "Dual-heuristic behavioral asset pricing engine (BSV 1998): unifies short-term underreaction and "
                    "long-term overreaction via conservatism bias on isolated earnings shocks (PEAD drift) versus the "
                    "representativeness heuristic / law of small numbers on streaks (regime-switching overreaction and reversals)."
                )
                from src.analytics.bsv_sentiment import evaluate_bsv_sentiment_profile
                if len(df_obs) >= 20:
                    try:
                        b_cols = st.columns(2)
                        with b_cols[0]:
                            b_surp = st.number_input(
                                "Latest Earnings Surprise (%)",
                                min_value=-50.0,
                                max_value=80.0,
                                value=5.0,
                                step=0.5,
                                key="bsv_surp_input",
                            )
                        with b_cols[1]:
                            b_streak = st.slider(
                                "Consecutive Earnings Streak (Quarters)",
                                min_value=1,
                                max_value=8,
                                value=2,
                                step=1,
                                key="bsv_streak_input",
                            )

                        b_rec = evaluate_bsv_sentiment_profile(
                            ticker=selected_ticker,
                            latest_surprise_pct=float(b_surp),
                            consecutive_streak=int(b_streak),
                        )

                        by1, by2, by3, by4 = st.columns(4)
                        b_badge = (
                            "⚠️ STREAK OVERREACTION"
                            if b_rec.is_representativeness_peak
                            else ("🚀 PEAD DRIFT" if b_rec.is_conservatism_drift else f"Regime: {b_rec.regime}")
                        )
                        by1.metric(
                            "Trending State Prob (pi_t)",
                            f"{b_rec.trending_state_probability:.2f} / 1.00",
                            delta=f"Streak: {b_rec.consecutive_streak_quarters}Q",
                            help="Subjective Bayesian belief that firm is in a permanent trending regime",
                        )
                        by2.metric(
                            "Conservatism Drift (CDI)",
                            f"{b_rec.conservatism_drift_index:.2f} / 1.00",
                            delta="High PEAD Drift" if b_rec.is_conservatism_drift else "Contained Drift",
                            help="Degree of underreaction drift due to conservative prior belief updating",
                        )
                        by3.metric(
                            "Representativeness (RSS)",
                            f"{b_rec.representativeness_streak_score:.2f} / 1.00",
                            delta=b_badge,
                            delta_color="inverse" if b_rec.is_representativeness_peak else "normal",
                            help="Degree of streak over-extrapolation based on the law of small numbers",
                        )
                        by4.metric(
                            "Expected BSV Alpha",
                            f"{b_rec.expected_bsv_alpha_pct:+.2f}% Ann",
                            delta="Positive Drift Alpha" if b_rec.expected_bsv_alpha_pct > 0 else "Reversal Drag",
                            help="Projected annualized alpha spread balancing conservatism drift versus representativeness reversal",
                        )

                        st.caption(f"⚖️ **BSV Summary:** {b_rec.summary}")
                        st.caption(f"🛡️ **BSV Audit Digest:** `{b_rec.signature[:16]}...` (Verified: `{b_rec.verify_integrity()}`)")
                    except Exception as b_ex:
                        st.info(f"BSV Sentiment evaluation deferred: {b_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate BSV sentiment profile.")

                st.markdown("---")
                st.subheader("💡 Allan C. Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Engine")
                st.caption(
                    "Intangible capital misvaluation engine (Eberhart et al. 2004; Cohen et al. 2013): models conservative "
                    "accounting distortion where 100% immediate expensing of R&D temporarily depresses reported net income "
                    "while accumulating productive, unpriced knowledge capital stocks that deliver persistent abnormal alpha."
                )
                from src.analytics.innovation_capital import evaluate_innovation_capital_profile
                if len(df_obs) >= 20:
                    try:
                        i_cols = st.columns(3)
                        with i_cols[0]:
                            i_rd = (
                                st.number_input(
                                    "Annual R&D Expenditure ($M)",
                                    min_value=0.0,
                                    max_value=80000.0,
                                    value=3500.0,
                                    step=100.0,
                                    key="inno_rd_input",
                                )
                                * 1_000_000.0
                            )
                        with i_cols[1]:
                            i_rev = (
                                st.number_input(
                                    "Annual Revenues ($M)",
                                    min_value=1.0,
                                    max_value=600000.0,
                                    value=45000.0,
                                    step=500.0,
                                    key="inno_rev_input",
                                )
                                * 1_000_000.0
                            )
                        with i_cols[2]:
                            i_assets = (
                                st.number_input(
                                    "Total Assets ($M)",
                                    min_value=1.0,
                                    max_value=800000.0,
                                    value=60000.0,
                                    step=500.0,
                                    key="inno_assets_input",
                                )
                                * 1_000_000.0
                            )

                        i_rec = evaluate_innovation_capital_profile(
                            ticker=selected_ticker,
                            current_rd_expense=float(i_rd),
                            annual_revenues=float(i_rev),
                            total_assets=float(i_assets),
                        )

                        iy1, iy2, iy3, iy4 = st.columns(4)
                        i_badge = (
                            "💎 HIGH INNOVATION ALPHA"
                            if i_rec.is_high_innovation_alpha
                            else ("⚠️ OBSOLESCENCE LAGGARD" if i_rec.is_innovation_laggard else f"Regime: {i_rec.regime}")
                        )
                        iy1.metric(
                            "R&D-to-Market (RD/M)",
                            f"{i_rec.rd_to_market_pct:.2f}%",
                            delta=f"RD/S: {i_rec.rd_to_sales_pct:.1f}%",
                            help="R&D expenditure scaled by equity market capitalization",
                        )
                        iy2.metric(
                            "Knowledge Stock (K)",
                            f"${i_rec.capitalized_knowledge_stock / 1e9:.2f}B",
                            delta=f"K/A: {i_rec.knowledge_to_assets_pct:.1f}%",
                            help="Perpetual inventory knowledge accumulation (20% economic depreciation rate)",
                        )
                        iy3.metric(
                            "Innovation Misvaluation (IMI)",
                            f"{i_rec.innovation_misvaluation_index:.2f} / 1.00",
                            delta=i_badge,
                            delta_color="normal" if i_rec.is_high_innovation_alpha else ("inverse" if i_rec.is_innovation_laggard else "off"),
                            help="Composite percentile score of unpriced corporate intangible innovation intensity",
                        )
                        iy4.metric(
                            "Expected Innovation Alpha",
                            f"{i_rec.expected_innovation_alpha_pct:+.2f}% Ann",
                            delta="Patent Moat Premium" if i_rec.expected_innovation_alpha_pct > 0 else "Obsolescence Drag",
                            help="Projected annualized abnormal return from intangible knowledge stock repricing",
                        )

                        st.caption(f"💡 **Innovation Summary:** {i_rec.summary}")
                        st.caption(f"🛡️ **Innovation Audit Digest:** `{i_rec.signature[:16]}...` (Verified: `{i_rec.verify_integrity()}`)")
                    except Exception as i_ex:
                        st.info(f"Innovation Capital evaluation deferred: {i_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Innovation Capital profile.")

                st.markdown("---")
                st.subheader("🔄 Charles M. C. Lee & Bhaskaran Swaminathan (2000) Momentum Life-Cycle Engine")
                st.caption(
                    "Volume-momentum interaction engine (Lee & Swaminathan 2000): trading volume reveals the persistence "
                    "and reversal hazard of price momentum. Low-volume winners (Stage 1) have prolonged momentum persistence "
                    "(+10% to +14%/yr alpha); high-volume winners (Stage 2) face imminent exhaustion. High-volume losers "
                    "(Stage 3) experience capitulation rebounds; low-volume losers (Stage 4) suffer zombie decay."
                )
                from src.analytics.momentum_lifecycle import evaluate_momentum_lifecycle_profile
                if len(df_obs) >= 20:
                    try:
                        m_cols = st.columns(2)
                        with m_cols[0]:
                            m_to = st.number_input(
                                "Daily Turnover Ratio (% override)",
                                min_value=0.05,
                                max_value=25.0,
                                value=1.20,
                                step=0.1,
                                key="mlc_turnover_input",
                            )
                        with m_cols[1]:
                            st.write("")
                            st.write("")
                            st.caption("ℹ️ Benchmark turnover: Low <= 1.4% (neglected), High >= 2.2% (crowded/capitulation).")

                        m_rec = evaluate_momentum_lifecycle_profile(
                            ticker=selected_ticker,
                            turnover_ratio_override=float(m_to),
                        )

                        my1, my2, my3, my4 = st.columns(4)
                        m_badge = (
                            "💎 STAGE 1 NEGLECTED WINNER"
                            if m_rec.is_neglected_winner
                            else (
                                "⚠️ STAGE 2 CROWDED"
                                if m_rec.is_crowded_winner
                                else (
                                    "🚀 STAGE 3 CAPITULATION"
                                    if m_rec.is_capitulation_loser
                                    else f"Regime: {m_rec.regime}"
                                )
                            )
                        )
                        my1.metric(
                            "12-M Momentum",
                            f"{m_rec.trailing_momentum_pct:+.2f}%",
                            delta=f"Vol: {m_rec.annualized_volatility_pct:.1f}%",
                            help="Trailing cumulative intermediate price return",
                        )
                        my2.metric(
                            "Daily Turnover Ratio",
                            f"{m_rec.daily_turnover_ratio_pct:.2f}%",
                            delta="Low (Neglected)" if m_rec.daily_turnover_ratio_pct <= 1.4 else ("High (Crowded)" if m_rec.daily_turnover_ratio_pct >= 2.2 else "Moderate"),
                            help="Shares traded daily scaled by diluted shares outstanding",
                        )
                        my3.metric(
                            "Volume-Momentum Score",
                            f"{m_rec.volume_momentum_interaction_score:>+.2f}",
                            delta=m_badge,
                            delta_color="normal" if (m_rec.is_neglected_winner or m_rec.is_capitulation_loser) else ("inverse" if m_rec.is_crowded_winner else "off"),
                            help="Interaction score: underreaction continuation vs. crowded exhaustion",
                        )
                        my4.metric(
                            "Expected MLC Alpha",
                            f"{m_rec.expected_mlc_alpha_pct:+.2f}% Ann",
                            delta="Continuation Spread" if m_rec.expected_mlc_alpha_pct > 0 else "Reversal Drag",
                            help="Projected annualized abnormal return based on momentum life-cycle quadrant",
                        )

                        st.caption(f"🔄 **MLC Summary:** {m_rec.summary}")
                        st.caption(f"🛡️ **MLC Audit Digest:** `{m_rec.signature[:16]}...` (Verified: `{m_rec.verify_integrity()}`)")
                    except Exception as m_ex:
                        st.info(f"Momentum Life Cycle evaluation deferred: {m_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Momentum Life Cycle profile.")

                st.markdown("---")
                st.subheader("🛡️ Edward I. Altman (1968, 2000) Z-Score & James A. Ohlson (1980) O-Score Solvency Engine")
                st.caption(
                    "Structural solvency and probabilistic bankruptcy risk engine: unifies Altman's multi-discriminant "
                    "Z-Score (manufacturing) and Z''-Score (services/emerging markets) with Ohlson's 9-factor conditional logit "
                    "O-Score. Isolates solvent fortress compounders while eliminating acute credit distress and default traps (Campbell et al. 2008)."
                )
                from src.analytics.altman_zscore import evaluate_solvency_profile
                if len(df_obs) >= 20:
                    try:
                        az_cols = st.columns(3)
                        with az_cols[0]:
                            az_assets = (
                                st.number_input(
                                    "Total Assets ($M)",
                                    min_value=1.0,
                                    max_value=1_000_000.0,
                                    value=10_000.0,
                                    step=500.0,
                                    key="az_assets_input",
                                )
                            )
                        with az_cols[1]:
                            az_liab = (
                                st.number_input(
                                    "Total Liabilities ($M)",
                                    min_value=0.1,
                                    max_value=1_000_000.0,
                                    value=4_500.0,
                                    step=250.0,
                                    key="az_liab_input",
                                )
                            )
                        with az_cols[2]:
                            az_ebit = (
                                st.number_input(
                                    "EBIT Operating Profit ($M)",
                                    min_value=-50_000.0,
                                    max_value=100_000.0,
                                    value=1_200.0,
                                    step=100.0,
                                    key="az_ebit_input",
                                )
                            )

                        az_rec = evaluate_solvency_profile(
                            ticker=selected_ticker,
                            total_assets=float(az_assets),
                            total_liabilities=float(az_liab),
                            ebit=float(az_ebit),
                        )

                        zy1, zy2, zy3, zy4 = st.columns(4)
                        z_badge = (
                            "💎 SAFE ZONE"
                            if az_rec.is_safe_zone
                            else (
                                "⚠️ GREY ZONE"
                                if az_rec.is_grey_zone
                                else "🚨 DISTRESS ZONE"
                            )
                        )
                        zy1.metric(
                            "Altman Z-Score",
                            f"{az_rec.altman_z_score:.2f}",
                            delta=z_badge,
                            delta_color="normal" if az_rec.is_safe_zone else ("inverse" if az_rec.is_distress_zone else "off"),
                            help="Original 5-factor manufacturing discriminant score (Safe > 2.99, Distress < 1.81)",
                        )
                        zy2.metric(
                            "Altman Non-Mfg Z''",
                            f"{az_rec.altman_z_double_prime:.2f}",
                            delta="Safe > 2.60" if az_rec.altman_z_double_prime > 2.60 else ("Distress < 1.10" if az_rec.altman_z_double_prime < 1.10 else "Grey Zone"),
                            help="4-factor emerging markets and services score (Safe > 2.60, Distress < 1.10)",
                        )
                        zy3.metric(
                            "Ohlson Default Prob P(Def)",
                            f"{az_rec.default_probability_pct:.1f}%",
                            delta=f"O-Score: {az_rec.ohlson_o_score:+.2f}",
                            delta_color="inverse" if az_rec.default_probability_pct >= 20.0 else "normal",
                            help="Ohlson (1980) 9-factor logistic probability of 2-year structural default",
                        )
                        zy4.metric(
                            "Expected Solvency Alpha",
                            f"{az_rec.expected_solvency_alpha_pct:+.2f}% Ann",
                            delta="Solvency Premium" if az_rec.expected_solvency_alpha_pct > 0 else "Distress Drag",
                            help="Projected annualized alpha: quality fortress premium vs. Campbell et al. (2008) distress discount",
                        )

                        st.caption(f"🛡️ **Solvency Summary:** {az_rec.summary}")
                        st.caption(f"🔒 **Solvency Audit Digest:** `{az_rec.signature[:16]}...` (Verified: `{az_rec.verify_integrity()}`)")
                    except Exception as az_ex:
                        st.info(f"Altman Solvency evaluation deferred: {az_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Altman & Ohlson Solvency profile.")

                st.markdown("---")
                st.subheader("🌐 Clifford S. Asness, Tobias J. Moskowitz & Lasse Heje Pedersen (2013) 'Value and Momentum Everywhere' (VME) Engine")
                st.caption(
                    "Unified multi-style asset pricing engine (Asness et al. 2013): combines standardized Value (B/M, E/P) "
                    "and 12-2 Skip-Month Momentum to exploit their strong negative correlation (rho in [-0.40, -0.65]). "
                    "The 50/50 volatility-matched VME combination neutralizes cyclical style crashes and nearly doubles the Sharpe ratio."
                )
                from src.analytics.vme_unified import evaluate_vme_profile
                if len(df_obs) >= 25:
                    try:
                        vme_cols = st.columns(3)
                        with vme_cols[0]:
                            vme_bm = st.number_input(
                                "Book-to-Market (B/M)",
                                min_value=0.05,
                                max_value=3.00,
                                value=0.62,
                                step=0.05,
                                key="vme_bm_input",
                            )
                        with vme_cols[1]:
                            vme_ep = st.number_input(
                                "Earnings Yield (E/P %)",
                                min_value=-20.0,
                                max_value=30.0,
                                value=6.20,
                                step=0.50,
                                key="vme_ep_input",
                            )
                        with vme_cols[2]:
                            vme_corr = st.slider(
                                "Value-Momentum Correlation (rho)",
                                min_value=-0.85,
                                max_value=0.50,
                                value=-0.52,
                                step=0.05,
                                key="vme_corr_input",
                            )

                        vme_rec = evaluate_vme_profile(
                            ticker=selected_ticker,
                            book_to_market=float(vme_bm),
                            earnings_yield_pct=float(vme_ep),
                            value_momentum_correlation=float(vme_corr),
                        )

                        vy1, vy2, vy3, vy4 = st.columns(4)
                        vme_badge = (
                            "💎 TWIN-ENGINE COMPOUNDER"
                            if vme_rec.is_twin_engine_compounder
                            else (
                                "🚨 HEADWIND TRAP"
                                if vme_rec.is_dual_headwind_trap
                                else f"Regime: {vme_rec.regime}"
                            )
                        )
                        vy1.metric(
                            "Unified 50/50 VME Score",
                            f"{vme_rec.vme_combo_z_score:+.2f}",
                            delta=vme_badge,
                            delta_color="normal" if vme_rec.vme_combo_z_score >= 0.25 else ("inverse" if vme_rec.is_dual_headwind_trap else "off"),
                            help="Covariance-adjusted 50/50 combination of Value and Momentum z-scores",
                        )
                        vy2.metric(
                            "Style Z-Scores (Val / Mom)",
                            f"V:{vme_rec.value_z_score:+.2f} | M:{vme_rec.momentum_z_score:+.2f}",
                            delta=f"12-2 Mom: {vme_rec.momentum_12_2_pct:+.1f}%",
                            help="Standardized Value z-score and 12-2 skip-month intermediate Momentum z-score",
                        )
                        vy3.metric(
                            "Diversification Benefit",
                            f"{vme_rec.diversification_benefit_ratio:.2f}x",
                            delta=f"Implied SR: {vme_rec.implied_vme_sharpe:.2f}",
                            help="Portfolio volatility reduction factor sqrt(2 / (1 + rho)) from negative style correlation",
                        )
                        vy4.metric(
                            "Expected VME Alpha",
                            f"{vme_rec.expected_vme_alpha_pct:+.2f}% Ann",
                            delta="Anti-Fragile Premium" if vme_rec.expected_vme_alpha_pct > 0 else "Dual Style Drag",
                            help="Projected annualized abnormal return from unified Value + Momentum exposure",
                        )

                        st.caption(f"🌐 **VME Summary:** {vme_rec.summary}")
                        st.caption(f"🛡️ **VME Audit Digest:** `{vme_rec.signature[:16]}...` (Verified: `{vme_rec.verify_integrity()}`)")
                    except Exception as vme_ex:
                        st.info(f"VME evaluation deferred: {vme_ex}")
                else:
                    st.info("ℹ️ Requires at least 25 price sessions to evaluate VME profile.")

                st.markdown("---")
                st.subheader("📊 Victor L. Bernard & Jacob K. Thomas (1989, 1990) PEAD & SUE Autocorrelation Engine")
                st.caption(
                    "Post-Earnings-Announcement Drift (PEAD) & Standardized Unexpected Earnings (SUE) engine: models "
                    "seasonal quarterly earnings autocorrelation (phi_1 = +0.34 persistence vs. phi_4 = -0.24 seasonal reversal). "
                    "Captures 60-trading-day underreaction drift as market participants anchor on naive seasonal random walks."
                )
                from src.analytics.pead_sue import evaluate_pead_sue_profile
                if len(df_obs) >= 20:
                    try:
                        p_cols = st.columns(3)
                        with p_cols[0]:
                            p_cur_eps = st.number_input(
                                "Current Quarter EPS ($)",
                                min_value=-50.0,
                                max_value=100.0,
                                value=1.80,
                                step=0.05,
                                key="pead_cur_eps_input",
                            )
                        with p_cols[1]:
                            p_yr_eps = st.number_input(
                                "Year-Ago Quarter EPS ($)",
                                min_value=-50.0,
                                max_value=100.0,
                                value=1.50,
                                step=0.05,
                                key="pead_yr_eps_input",
                            )
                        with p_cols[2]:
                            p_days = st.slider(
                                "Sessions Since Announcement (d)",
                                min_value=0,
                                max_value=90,
                                value=10,
                                step=5,
                                key="pead_days_input",
                            )

                        p_rec = evaluate_pead_sue_profile(
                            ticker=selected_ticker,
                            current_eps=float(p_cur_eps),
                            year_ago_eps=float(p_yr_eps),
                            days_since_announcement=int(p_days),
                        )

                        py1, py2, py3, py4 = st.columns(4)
                        p_badge = (
                            "🚀 ACTIVE UPWARD PEAD"
                            if p_rec.is_active_positive_drift
                            else (
                                "📉 ACTIVE DOWNWARD SLUMP"
                                if p_rec.is_active_negative_slump
                                else f"Regime: {p_rec.regime}"
                            )
                        )
                        py1.metric(
                            "Standardized Surprise (SUE)",
                            f"{p_rec.sue_score:+.2f}",
                            delta=f"Decile {p_rec.sue_decile} / 10",
                            delta_color="normal" if p_rec.sue_score >= 0 else "inverse",
                            help="Standardized Unexpected Earnings relative to seasonal random walk with drift",
                        )
                        py2.metric(
                            "Next-Quarter AR(1) SUE",
                            f"{p_rec.predicted_next_quarter_sue:+.2f}",
                            delta=f"Lag-4 Reversal: {p_rec.predicted_lag4_reversal_sue:+.2f}",
                            help="Bernard-Thomas seasonal earnings autocorrelation forecast (phi_1 = +0.34, phi_4 = -0.24)",
                        )
                        py3.metric(
                            "Remaining 60-Day Drift",
                            f"{p_rec.remaining_drift_pct:+.2f}%",
                            delta=p_badge,
                            delta_color="normal" if p_rec.is_active_positive_drift else ("inverse" if p_rec.is_active_negative_slump else "off"),
                            help="Decay-adjusted remaining cumulative abnormal return over the 60-session window",
                        )
                        py4.metric(
                            "Expected PEAD Alpha",
                            f"{p_rec.expected_pead_alpha_pct:+.2f}% Ann",
                            delta="Underreaction Drift" if p_rec.expected_pead_alpha_pct > 0 else "Negative Drift Drag",
                            help="Projected annualized abnormal return from post-earnings announcement drift",
                        )

                        st.caption(f"📊 **PEAD Summary:** {p_rec.summary}")
                        st.caption(f"🛡️ **PEAD Audit Digest:** `{p_rec.signature[:16]}...` (Verified: `{p_rec.verify_integrity()}`)")
                    except Exception as p_ex:
                        st.info(f"PEAD & SUE evaluation deferred: {p_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate PEAD & SUE profile.")

                st.markdown("---")
                st.subheader("🏗️ Cooper, Gulen & Schill (2008) / Titman, Wei & Xie (2004) Asset Growth & Empire-Building Engine")
                st.caption(
                    "Corporate asset expansion and abnormal capital investment engine: measures YoY Total Asset Growth (AG%) "
                    "and Abnormal Capital Investment (ACI) conditioned on Jensen (1986) Free Cash Flow agency discretion. "
                    "Separates disciplined capital allocators from value-destructive empire-building overinvestors."
                )
                from src.analytics.asset_growth_empire import evaluate_asset_growth_profile
                if len(df_obs) >= 20:
                    try:
                        em_cols = st.columns(3)
                        with em_cols[0]:
                            em_cur_ta = st.number_input(
                                "Current Total Assets ($M)",
                                min_value=10.0,
                                max_value=1_000_000.0,
                                value=10_400.0,
                                step=200.0,
                                key="emp_cur_ta_input",
                            )
                        with em_cols[1]:
                            em_pri_ta = st.number_input(
                                "Prior Total Assets ($M)",
                                min_value=10.0,
                                max_value=1_000_000.0,
                                value=10_000.0,
                                step=200.0,
                                key="emp_pri_ta_input",
                            )
                        with em_cols[2]:
                            em_capex = st.number_input(
                                "Current CapEx / Sales (%)",
                                min_value=0.0,
                                max_value=80.0,
                                value=5.8,
                                step=0.5,
                                key="emp_capex_input",
                            )

                        em_rec = evaluate_asset_growth_profile(
                            ticker=selected_ticker,
                            current_total_assets=float(em_cur_ta),
                            prior_total_assets=float(em_pri_ta),
                            current_capex_to_sales_pct=float(em_capex),
                        )

                        ey1, ey2, ey3, ey4 = st.columns(4)
                        em_badge = (
                            "💎 DISCIPLINED ALLOCATOR"
                            if em_rec.is_disciplined_allocator
                            else (
                                "🚨 EMPIRE BUILDER TRAP"
                                if em_rec.is_empire_builder_trap
                                else f"Regime: {em_rec.regime}"
                            )
                        )
                        ey1.metric(
                            "Total Asset Growth (AG)",
                            f"{em_rec.asset_growth_pct:+.2f}%",
                            delta=em_badge,
                            delta_color="normal" if em_rec.is_disciplined_allocator else ("inverse" if em_rec.is_empire_builder_trap else "off"),
                            help="Cooper, Gulen & Schill (2008) YoY balance sheet asset expansion",
                        )
                        ey2.metric(
                            "Abnormal CapEx (ACI)",
                            f"{em_rec.abnormal_capital_investment:+.2f}",
                            delta=f"3Yr Avg: {em_rec.historical_3yr_capex_to_sales_pct:.1f}%",
                            delta_color="inverse" if em_rec.abnormal_capital_investment > 0.30 else "normal",
                            help="Titman, Wei & Xie (2004) CapEx/Sales deviation from 3-year baseline",
                        )
                        ey3.metric(
                            "Empire-Building Score",
                            f"{em_rec.empire_building_score:+.2f}",
                            delta=f"Jensen Agency: {em_rec.jensen_agency_discretion_index:.2f}",
                            delta_color="inverse" if em_rec.empire_building_score >= 1.25 else "normal",
                            help="Composite overinvestment score amplified by free cash flow agency discretion",
                        )
                        ey4.metric(
                            "Expected Allocation Alpha",
                            f"{em_rec.expected_allocation_alpha_pct:+.2f}% Ann",
                            delta="Capital Restraint" if em_rec.expected_allocation_alpha_pct > 0 else "Overinvestment Drag",
                            help="Projected annualized abnormal return from corporate capital allocation discipline",
                        )

                        st.caption(f"🏗️ **Allocation Summary:** {em_rec.summary}")
                        st.caption(f"🛡️ **Allocation Audit Digest:** `{em_rec.signature[:16]}...` (Verified: `{em_rec.verify_integrity()}`)")
                    except Exception as em_ex:
                        st.info(f"Asset Growth & Empire-Building evaluation deferred: {em_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Asset Growth & Empire-Building profile.")

                st.markdown("---")
                st.subheader("💰 Boudoukh, Michaely, Richardson & Roberts (2007) / Pontiff & Woodgate (2008) Net Payout & Share Issuance Engine")
                st.caption(
                    "Total shareholder yield and share cannibalization engine: synthesizes Cash Dividends, Gross Share "
                    "Repurchases, and Equity Issuance into Net Payout Yield (NPY) and Composite Share Issuance (CSI), "
                    "guarded by Free Cash Flow Payout Coverage (PCR)."
                )
                from src.analytics.net_payout_yield import evaluate_net_payout_profile
                if len(df_obs) >= 20:
                    try:
                        np_cols = st.columns(3)
                        with np_cols[0]:
                            np_buybacks = st.number_input(
                                "Annual Share Repurchases ($M)",
                                min_value=0.0,
                                max_value=200_000.0,
                                value=380.0,
                                step=20.0,
                                key="npy_buybacks_input",
                            )
                        with np_cols[1]:
                            np_issuance = st.number_input(
                                "Annual Equity Issuance ($M)",
                                min_value=0.0,
                                max_value=200_000.0,
                                value=60.0,
                                step=20.0,
                                key="npy_issuance_input",
                            )
                        with np_cols[2]:
                            np_cur_sh = st.number_input(
                                "Current Diluted Shares (M, Prior=100M)",
                                min_value=10.0,
                                max_value=1_000.0,
                                value=97.0,
                                step=1.0,
                                key="npy_shares_input",
                            )

                        np_rec = evaluate_net_payout_profile(
                            ticker=selected_ticker,
                            market_cap=10_000.0,
                            dividends=180.0,
                            share_repurchases=float(np_buybacks),
                            equity_issuance=float(np_issuance),
                            current_shares=float(np_cur_sh),
                            prior_shares=100.0,
                            free_cash_flow=750.0,
                        )

                        ny1, ny2, ny3, ny4 = st.columns(4)
                        np_badge = (
                            "💎 SHARE CANNIBAL"
                            if np_rec.is_share_cannibal
                            else (
                                "🚨 DILUTION TRAP"
                                if np_rec.is_dilution_trap
                                else (
                                    "⚠️ DEBT-FUNDED PAYOUT"
                                    if np_rec.is_unsustainable_debt_payout
                                    else f"Regime: {np_rec.regime}"
                                )
                            )
                        )
                        ny1.metric(
                            "Net Payout Yield (NPY)",
                            f"{np_rec.net_payout_yield_pct:+.2f}%",
                            delta=np_badge,
                            delta_color="normal" if np_rec.is_share_cannibal else ("inverse" if (np_rec.is_dilution_trap or np_rec.is_unsustainable_debt_payout) else "off"),
                            help="Boudoukh et al. (2007) Total Net Payout = (Dividends + Buybacks - Issuance) / Market Cap",
                        )
                        ny2.metric(
                            "Composite Share Issuance",
                            f"{np_rec.composite_share_issuance_pct:+.2f}%",
                            delta=f"Buyback Yld: {np_rec.gross_buyback_yield_pct:.1f}%",
                            delta_color="inverse" if np_rec.composite_share_issuance_pct > 0 else "normal",
                            help="Pontiff & Woodgate (2008) log change in split-adjusted shares outstanding",
                        )
                        ny3.metric(
                            "FCF Payout Coverage",
                            f"{np_rec.fcf_payout_coverage_ratio:.2f}x",
                            delta="Sustainable" if np_rec.fcf_payout_coverage_ratio >= 1.0 else "Uncovered",
                            delta_color="normal" if np_rec.fcf_payout_coverage_ratio >= 0.75 else "inverse",
                            help="Ratio of annual Free Cash Flow to total gross distributions (Dividends + Repurchases)",
                        )
                        ny4.metric(
                            "Expected Shareholder Alpha",
                            f"{np_rec.expected_payout_alpha_pct:+.2f}% Ann",
                            delta="Cannibalization Premium" if np_rec.expected_payout_alpha_pct > 0 else "Dilution Drag",
                            help="Projected annualized abnormal return from total net payout yield and share shrinkage",
                        )

                        st.caption(f"💰 **Payout Summary:** {np_rec.summary}")
                        st.caption(f"🛡️ **Payout Audit Digest:** `{np_rec.signature[:16]}...` (Verified: `{np_rec.verify_integrity()}`)")
                    except Exception as np_ex:
                        st.info(f"Net Payout Yield evaluation deferred: {np_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Net Payout Yield profile.")

                st.markdown("---")
                st.subheader("🧾 Hirshleifer, Hou, Teoh & Zhang (2004) Net Operating Assets (NOA) & Balance Sheet Bloat Engine")
                st.caption(
                    "Cumulative accounting distortion and balance sheet bloat engine: measures Net Operating Assets "
                    "(NOA = Operating Assets - Operating Liabilities) scaled by lagged Total Assets (SNOA). "
                    "Identifies lean cash-generative fortresses while eliminating acute balance-sheet accrual bloat traps."
                )
                from src.analytics.noa_bloat import evaluate_noa_bloat_profile
                if len(df_obs) >= 20:
                    try:
                        noa_cols = st.columns(3)
                        with noa_cols[0]:
                            noa_cash = st.number_input(
                                "Cash & Equivalents ($M)",
                                min_value=0.0,
                                max_value=500_000.0,
                                value=2_400.0,
                                step=100.0,
                                key="noa_cash_input",
                            )
                        with noa_cols[1]:
                            noa_debt = st.number_input(
                                "Total Debt ($M)",
                                min_value=0.0,
                                max_value=500_000.0,
                                value=2_200.0,
                                step=100.0,
                                key="noa_debt_input",
                            )
                        with noa_cols[2]:
                            noa_equity = st.number_input(
                                "Book Equity ($M)",
                                min_value=-100_000.0,
                                max_value=500_000.0,
                                value=4_200.0,
                                step=100.0,
                                key="noa_equity_input",
                            )

                        noa_rec = evaluate_noa_bloat_profile(
                            ticker=selected_ticker,
                            total_assets=10_500.0,
                            prior_total_assets=10_000.0,
                            cash_and_equivalents=float(noa_cash),
                            total_debt=float(noa_debt),
                            book_equity=float(noa_equity),
                            prior_snoa=0.41,
                        )

                        hy1, hy2, hy3, hy4 = st.columns(4)
                        noa_badge = (
                            "💎 LEAN CASH FORTRESS"
                            if noa_rec.is_lean_cash_fortress
                            else (
                                "🚨 BLOAT TRAP"
                                if noa_rec.is_bloated_balance_sheet_trap
                                else f"Regime: {noa_rec.regime}"
                            )
                        )
                        hy1.metric(
                            "Scaled NOA Ratio (SNOA)",
                            f"{noa_rec.scaled_noa_ratio:.2f}",
                            delta=noa_badge,
                            delta_color="normal" if noa_rec.is_lean_cash_fortress else ("inverse" if noa_rec.is_bloated_balance_sheet_trap else "off"),
                            help="Hirshleifer, Hou, Teoh & Zhang (2004) NOA / Prior Total Assets (Lean <= 0.45, Bloated >= 0.85)",
                        )
                        hy2.metric(
                            "Net Operating Assets",
                            f"${noa_rec.net_operating_assets:,.0f}M",
                            delta=f"dSNOA: {noa_rec.delta_snoa:+.2f}",
                            delta_color="inverse" if noa_rec.delta_snoa > 0.08 else "normal",
                            help="Operating Assets minus Operating Liabilities (cumulative accounting profit minus cash flow)",
                        )
                        hy3.metric(
                            "Cash Conversion Eff.",
                            f"{noa_rec.cash_conversion_efficiency:.2f} / 1.00",
                            delta=f"OA: ${noa_rec.operating_assets:,.0f}M",
                            help="Complement of cumulative balance sheet accrual bloat",
                        )
                        hy4.metric(
                            "Expected NOA Alpha",
                            f"{noa_rec.expected_noa_alpha_pct:+.2f}% Ann",
                            delta="Cash Quality Premium" if noa_rec.expected_noa_alpha_pct > 0 else "Sustainability Wall Drag",
                            help="Projected annualized abnormal return from balance-sheet accrual discipline",
                        )

                        st.caption(f"🧾 **NOA Summary:** {noa_rec.summary}")
                        st.caption(f"🛡️ **NOA Audit Digest:** `{noa_rec.signature[:16]}...` (Verified: `{noa_rec.verify_integrity()}`)")
                    except Exception as noa_ex:
                        st.info(f"NOA Bloat evaluation deferred: {noa_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate NOA & Balance Sheet Bloat profile.")

                st.divider()
                st.markdown("#### 🔮 Diether, Malloy & Scherbina (2002) & Miller (1977) Analyst Forecast Dispersion & Divergence of Opinion")
                if len(prices) >= 20:
                    try:
                        from src.analytics.belief_dispersion import evaluate_belief_dispersion_profile

                        with st.expander("⚙️ Optional I/B/E/S Consensus & Short-Sale Friction Inputs (Diether-Malloy-Scherbina 2002)", expanded=False):
                            bd_c1, bd_c2, bd_c3 = st.columns(3)
                            bd_mean_eps = bd_c1.number_input("Consensus Mean EPS ($)", value=0.0, step=0.25, key="bd_mean_eps")
                            bd_std_eps = bd_c2.number_input("Analyst EPS Std Dev ($)", value=0.0, min_value=0.0, step=0.05, key="bd_std_eps")
                            bd_analysts = bd_c3.number_input("Analyst Coverage (N)", value=0, min_value=0, step=1, key="bd_analysts")
                            bd_c4, bd_c5 = st.columns(2)
                            bd_borrow_fee = bd_c4.number_input("Annualized Stock Loan Borrow Fee (%)", value=0.0, min_value=0.0, step=0.25, key="bd_borrow_fee")
                            bd_inst_own = bd_c5.number_input("Institutional Ownership (%)", value=0.0, min_value=0.0, max_value=100.0, step=5.0, key="bd_inst_own")

                        use_custom_bd = (bd_mean_eps != 0.0 or bd_std_eps > 0.0)
                        bd_rec = evaluate_belief_dispersion_profile(
                            ticker=selected_ticker,
                            consensus_mean_eps=bd_mean_eps if use_custom_bd else None,
                            forecast_std_eps=bd_std_eps if use_custom_bd else None,
                            analyst_count=int(bd_analysts) if bd_analysts >= 2 else None,
                            short_borrow_fee_pct=bd_borrow_fee if bd_borrow_fee > 0 else None,
                            institutional_ownership_pct=bd_inst_own if bd_inst_own > 0 else None,
                        )

                        bd1, bd2, bd3, bd4 = st.columns(4)
                        bd_badge = (
                            "🟢 HIGH-CONSENSUS COMPOUNDER"
                            if bd_rec.is_high_consensus_compounder
                            else (
                                "🔴 MILLER OVERPRICING TRAP"
                                if bd_rec.is_miller_overpricing_trap
                                else f"Regime: {bd_rec.regime}"
                            )
                        )
                        bd1.metric(
                            "Analyst Dispersion (D_t)",
                            f"{bd_rec.dispersion_ratio:.4f}",
                            delta=bd_badge,
                            delta_color="normal" if bd_rec.is_high_consensus_compounder else ("inverse" if bd_rec.is_miller_overpricing_trap else "off"),
                            help="Diether, Malloy & Scherbina (2002) Analyst EPS Forecast Dispersion sigma(EPS) / max(|mean(EPS)|, 0.10) (Tight <= 0.05, Trap >= 0.28)",
                        )
                        bd2.metric(
                            "Miller Overpricing Wedge",
                            f"{bd_rec.miller_overpricing_wedge:.2f} / 3.50",
                            delta=f"SSFI: {bd_rec.short_sale_friction_index:.2f}",
                            delta_color="inverse" if bd_rec.miller_overpricing_wedge >= 1.35 else "normal",
                            help="Miller (1977) Overpricing Wedge: interaction of belief dispersion and short-sale binding friction",
                        )
                        bd3.metric(
                            "Short-Sale Friction",
                            f"{bd_rec.short_sale_friction_index:.2f} / 1.00",
                            delta=f"Coverage: {bd_rec.analyst_count} Analysts",
                            help="Composite short-sale binding impediment score from borrow fee and institutional lendable float",
                        )
                        bd4.metric(
                            "Expected Dispersion Alpha",
                            f"{bd_rec.expected_dispersion_alpha_pct:+.2f}% Ann",
                            delta="Consensus Premium" if bd_rec.expected_dispersion_alpha_pct > 0 else "Optimist Overpricing Drag",
                            help="Projected annualized abnormal return from cross-analyst belief dispersion and Miller (1977) friction",
                        )

                        st.caption(f"🧾 **Dispersion Summary:** {bd_rec.summary}")
                        st.caption(f"🛡️ **Dispersion Audit Digest:** `{bd_rec.signature[:16]}...` (Verified: `{bd_rec.verify_integrity()}`)")
                    except Exception as bd_ex:
                        st.info(f"Belief Dispersion evaluation deferred: {bd_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Analyst Forecast Dispersion profile.")

                st.divider()
                st.markdown("#### 🏗️ Polk & Sapienza (2009) Catering Theory of Corporate Investment & Valuation Mispricing")
                if len(prices) >= 20:
                    try:
                        from src.analytics.catering_investment import evaluate_catering_investment_profile

                        with st.expander("⚙️ Optional Capital Allocation & Valuation Mispricing Inputs (Polk-Sapienza 2009)", expanded=False):
                            ci_c1, ci_c2, ci_c3 = st.columns(3)
                            ci_capex = ci_c1.number_input("Annual CapEx ($M)", value=0.0, min_value=0.0, step=25.0, key="ci_capex")
                            ci_ppe = ci_c2.number_input("Lagged Net PPE K_{t-1} ($M)", value=0.0, min_value=0.0, step=100.0, key="ci_ppe")
                            ci_q = ci_c3.number_input("Tobin's Q Ratio", value=0.0, min_value=0.0, step=0.10, key="ci_q")
                            ci_c4, ci_c5, ci_c6 = st.columns(3)
                            ci_cf = ci_c4.number_input("Cash Flow / Lagged PPE Ratio", value=0.0, step=0.05, key="ci_cf")
                            ci_dmp = ci_c5.number_input("Discretionary Mispricing Score [-1..1]", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key="ci_dmp")
                            ci_turn = ci_c6.number_input("Annual Share Turnover (%)", value=0.0, min_value=0.0, step=10.0, key="ci_turn")

                        use_custom_ci = (ci_capex > 0.0 and ci_ppe > 0.0)
                        ci_rec = evaluate_catering_investment_profile(
                            ticker=selected_ticker,
                            capex_amount=ci_capex if use_custom_ci else None,
                            lagged_net_ppe=ci_ppe if use_custom_ci else None,
                            tobins_q=ci_q if ci_q > 0 else None,
                            cash_flow_to_capital=ci_cf if ci_cf != 0.0 else None,
                            discretionary_mispricing_score=ci_dmp if ci_dmp != 0.0 else None,
                            annual_turnover_pct=ci_turn if ci_turn > 0 else None,
                        )

                        ci1, ci2, ci3, ci4 = st.columns(4)
                        ci_badge = (
                            "🟢 DISCIPLINED ALLOCATOR"
                            if ci_rec.is_disciplined_allocator
                            else (
                                "🔴 CATERING CAPEX TRAP"
                                if ci_rec.is_catering_capex_trap
                                else f"Regime: {ci_rec.regime}"
                            )
                        )
                        ci1.metric(
                            "Investment / Capital (IK_t)",
                            f"{ci_rec.investment_to_capital_ratio:.3f}",
                            delta=ci_badge,
                            delta_color="normal" if ci_rec.is_disciplined_allocator else ("inverse" if ci_rec.is_catering_capex_trap else "off"),
                            help="Polk & Sapienza (2009) Physical Capital Investment Rate CapEx_t / K_{t-1}",
                        )
                        ci2.metric(
                            "Residual Abnormal Inv.",
                            f"{ci_rec.residual_abnormal_investment:+.3f}",
                            delta=f"Q Benchmark: {ci_rec.fundamental_q_investment_rate:.3f}",
                            delta_color="inverse" if ci_rec.residual_abnormal_investment >= 0.15 else "normal",
                            help="Excess investment above neoclassical Tobin's Q and internal cash flow benchmark",
                        )
                        ci3.metric(
                            "Catering Distortion (CDI)",
                            f"{ci_rec.catering_distortion_index:+.2f}",
                            delta=f"Turnover Mult: {ci_rec.short_horizon_turnover_multiplier:.2f}x",
                            delta_color="inverse" if ci_rec.catering_distortion_index >= 1.10 else "normal",
                            help="Interaction of abnormal CapEx, discretionary equity mispricing, and short-horizon share turnover",
                        )
                        ci4.metric(
                            "Expected Catering Alpha",
                            f"{ci_rec.expected_catering_alpha_pct:+.2f}% Ann",
                            delta="Capital Discipline" if ci_rec.expected_catering_alpha_pct > 0 else "Catering Over-CapEx Drag",
                            help="Projected annualized abnormal return from Polk & Sapienza (2009) investment catering dynamics",
                        )

                        st.caption(f"🧾 **Catering Summary:** {ci_rec.summary}")
                        st.caption(f"🛡️ **Catering Audit Digest:** `{ci_rec.signature[:16]}...` (Verified: `{ci_rec.verify_integrity()}`)")
                    except Exception as ci_ex:
                        st.info(f"Catering Investment evaluation deferred: {ci_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Catering Theory of Investment profile.")

                st.divider()
                st.markdown("#### 🌊 Chordia, Roll & Subrahmanyam (2000, 2001) Commonality in Liquidity & Order Imbalance")
                if len(prices) >= 20:
                    try:
                        from src.analytics.liquidity_commonality import evaluate_liquidity_commonality_profile

                        with st.expander("⚙️ Optional Microstructure Co-Movement & Order Imbalance Inputs (Chordia-Roll-Subrahmanyam)", expanded=False):
                            lc_c1, lc_c2, lc_c3 = st.columns(3)
                            lc_sb = lc_c1.number_input("Spread Commonality Beta", value=0.0, step=0.10, key="lc_sb")
                            lc_db = lc_c2.number_input("Depth Commonality Beta", value=0.0, step=0.10, key="lc_db")
                            lc_r2 = lc_c3.number_input("Market Liquidity R^2 [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="lc_r2")
                            lc_c4, lc_c5, lc_c6 = st.columns(3)
                            lc_oib = lc_c4.number_input("Signed Order Imbalance [-1..1]", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key="lc_oib")
                            lc_ac = lc_c5.number_input("OIB Autocorrelation [-1..1]", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key="lc_ac")
                            lc_surge = lc_c6.number_input("Down-Market Illiquidity Surge (>=0.5)", value=0.0, min_value=0.0, step=0.10, key="lc_surge")

                        lc_rec = evaluate_liquidity_commonality_profile(
                            ticker=selected_ticker,
                            spread_commonality_beta=lc_sb if lc_sb != 0.0 else None,
                            depth_commonality_beta=lc_db if lc_db != 0.0 else None,
                            liquidity_market_r2=lc_r2 if lc_r2 > 0.0 else None,
                            signed_order_imbalance=lc_oib if lc_oib != 0.0 else None,
                            oib_autocorrelation=lc_ac if lc_ac != 0.0 else None,
                            down_market_illiquidity_surge=lc_surge if lc_surge >= 0.5 else None,
                        )

                        lc1, lc2, lc3, lc4 = st.columns(4)
                        lc_badge = (
                            "🟢 IDIOSYNCRATIC HAVEN"
                            if lc_rec.is_idiosyncratic_liquidity_haven
                            else (
                                "🔴 CONTAGION TRAP"
                                if lc_rec.is_systemic_contagion_trap
                                else f"Regime: {lc_rec.regime}"
                            )
                        )
                        lc1.metric(
                            "Effective Liquidity Beta",
                            f"{lc_rec.effective_commonality_beta:+.3f}",
                            delta=lc_badge,
                            delta_color="normal" if lc_rec.is_idiosyncratic_liquidity_haven else ("inverse" if lc_rec.is_systemic_contagion_trap else "off"),
                            help="Chordia, Roll & Subrahmanyam (2000) composite spread and depth co-movement beta",
                        )
                        lc2.metric(
                            "Signed Order Imbalance",
                            f"{lc_rec.signed_order_imbalance:+.3f}",
                            delta=f"Autocorr: {lc_rec.oib_autocorrelation:+.2f}",
                            delta_color="inverse" if lc_rec.order_splitting_pressure_score >= 0.38 else "normal",
                            help="Chordia, Roll & Subrahmanyam (2002) net signed order imbalance and first-order autocorrelation",
                        )
                        lc3.metric(
                            "Inventory Contagion (SICI)",
                            f"{lc_rec.systemic_inventory_contagion_index:.2f} / 3.50",
                            delta=f"Down Surge: {lc_rec.down_market_illiquidity_surge:.2f}x",
                            delta_color="inverse" if lc_rec.systemic_inventory_contagion_index >= 1.75 else "normal",
                            help="Systemic Inventory Contagion Index combining spread commonality, market R^2, and down-market illiquidity surge",
                        )
                        lc4.metric(
                            "Expected Commonality Alpha",
                            f"{lc_rec.expected_commonality_alpha_pct:+.2f}% Ann",
                            delta="Liquidity Haven" if lc_rec.expected_commonality_alpha_pct > 0 else "Contagion Drag",
                            help="Projected annualized execution-adjusted alpha from market-wide liquidity commonality",
                        )

                        st.caption(f"🧾 **Commonality Summary:** {lc_rec.summary}")
                        st.caption(f"🛡️ **Commonality Audit Digest:** `{lc_rec.signature[:16]}...` (Verified: `{lc_rec.verify_integrity()}`)")
                    except Exception as lc_ex:
                        st.info(f"Liquidity Commonality evaluation deferred: {lc_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Liquidity Commonality profile.")

                st.divider()
                st.markdown("#### 🔗 Cohen & Frazzini (2008) & Ahern (2013) Economic Links & Supply-Chain Customer Momentum")
                if len(prices) >= 20:
                    try:
                        from src.analytics.supply_chain_momentum import evaluate_supply_chain_momentum_profile

                        with st.expander("⚙️ Optional Supply-Chain Customer Link & Network Centrality Inputs (Cohen-Frazzini 2008)", expanded=False):
                            sc_c1, sc_c2, sc_c3 = st.columns(3)
                            sc_cust_ret = sc_c1.number_input("Principal Customer Return (decimal)", value=0.0, min_value=-0.95, max_value=5.0, step=0.02, key="sc_cust_ret")
                            sc_supp_ret = sc_c2.number_input("Supplier Own Return (decimal)", value=0.0, min_value=-0.95, max_value=5.0, step=0.02, key="sc_supp_ret")
                            sc_rev_share = sc_c3.number_input("Customer Revenue Share [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="sc_rev_share")
                            sc_c4, sc_c5 = st.columns(2)
                            sc_cov = sc_c4.number_input("Shared Analyst Coverage Ratio [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="sc_cov")
                            sc_cent = sc_c5.number_input("Ahern Network Centrality Score [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="sc_cent")

                        sc_rec = evaluate_supply_chain_momentum_profile(
                            ticker=selected_ticker,
                            customer_portfolio_return=sc_cust_ret if sc_cust_ret != 0.0 else None,
                            supplier_own_return=sc_supp_ret if sc_supp_ret != 0.0 else None,
                            customer_revenue_share=sc_rev_share if sc_rev_share > 0.0 else None,
                            shared_coverage_ratio=sc_cov if sc_cov > 0.0 else None,
                            network_centrality_score=sc_cent if sc_cent > 0.0 else None,
                        )

                        sc1, sc2, sc3, sc4 = st.columns(4)
                        sc_badge = (
                            "🟢 CUSTOMER SPILLOVER LEADER"
                            if sc_rec.is_positive_spillover_leader
                            else (
                                "🔴 UPSTREAM DEMAND TRAP"
                                if sc_rec.is_upstream_demand_collapse_trap
                                else f"Regime: {sc_rec.regime}"
                            )
                        )
                        sc1.metric(
                            "Customer Return Wedge",
                            f"{sc_rec.customer_supplier_return_wedge:+.2%}",
                            delta=sc_badge,
                            delta_color="normal" if sc_rec.is_positive_spillover_leader else ("inverse" if sc_rec.is_upstream_demand_collapse_trap else "off"),
                            help="Cohen & Frazzini (2008) unpriced return gap between principal customer portfolio and supplier",
                        )
                        sc2.metric(
                            "Spillover Impulse (LASI)",
                            f"{sc_rec.limited_attention_spillover_impulse:+.3f}",
                            delta=f"Rev Share: {sc_rec.customer_revenue_share:.0%}",
                            delta_color="normal" if sc_rec.limited_attention_spillover_impulse >= 0.035 else ("inverse" if sc_rec.limited_attention_spillover_impulse <= -0.035 else "off"),
                            help="Limited-Attention Spillover Impulse scaled by customer revenue share and inattention friction",
                        )
                        sc3.metric(
                            "Network Centrality",
                            f"{sc_rec.network_centrality_score:.2f} / 1.00",
                            delta=f"Inattention: {sc_rec.inattention_diffusion_multiplier:.2f}x",
                            help="Ahern (2013) intersectoral production network eigenvector centrality score",
                        )
                        sc4.metric(
                            "Expected Supply-Chain Alpha",
                            f"{sc_rec.expected_supply_chain_alpha_pct:+.2f}% Ann",
                            delta="Customer Momentum" if sc_rec.expected_supply_chain_alpha_pct > 0 else "Demand Shock Drag",
                            help="Projected annualized abnormal return from supply-chain customer momentum and network centrality",
                        )

                        st.caption(f"🧾 **Supply-Chain Summary:** {sc_rec.summary}")
                        st.caption(f"🛡️ **Supply-Chain Audit Digest:** `{sc_rec.signature[:16]}...` (Verified: `{sc_rec.verify_integrity()}`)")
                    except Exception as sc_ex:
                        st.info(f"Supply-Chain Momentum evaluation deferred: {sc_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Supply-Chain Customer Momentum profile.")

                st.divider()
                st.markdown("#### 🏛️ Gompers, Ishii & Metrick (2003) & Bebchuk et al. (2009) Corporate Governance & Managerial Entrenchment (G-Index / E-Index)")
                if len(prices) >= 20:
                    try:
                        from src.analytics.governance_entrenchment import evaluate_governance_entrenchment_profile

                        with st.expander("⚙️ Optional Constitutional Governance & Agency Inputs (Gompers-Bebchuk)", expanded=False):
                            ge_c1, ge_c2, ge_c3 = st.columns(3)
                            ge_e = ge_c1.number_input("Bebchuk E-Index [0..6]", value=1, min_value=0, max_value=6, step=1, key="ge_e")
                            ge_g = ge_c2.number_input("Gompers G-Index [0..24] (>= E)", value=5, min_value=0, max_value=24, step=1, key="ge_g")
                            ge_wedge = ge_c3.number_input("Dual-Class Voting Wedge [0..0.95]", value=0.0, min_value=0.0, max_value=0.95, step=0.05, key="ge_wedge")
                            ge_c4, ge_c5 = st.columns(2)
                            ge_indep = ge_c4.number_input("Board Independence Ratio [0..1]", value=0.85, min_value=0.0, max_value=1.0, step=0.05, key="ge_indep")
                            ge_fcf = ge_c5.number_input("Unreturned FCF Retention [0..1]", value=0.22, min_value=0.0, max_value=1.0, step=0.05, key="ge_fcf")

                        ge_rec = evaluate_governance_entrenchment_profile(
                            ticker=selected_ticker,
                            bebchuk_e_index=int(ge_e),
                            gompers_g_index=max(int(ge_g), int(ge_e)),
                            dual_class_voting_wedge=float(ge_wedge),
                            board_independence_ratio=float(ge_indep),
                            unreturned_fcf_ratio=float(ge_fcf),
                        )

                        ge1, ge2, ge3, ge4 = st.columns(4)
                        ge_badge = (
                            "🟢 SHAREHOLDER DEMOCRACY"
                            if ge_rec.is_shareholder_democracy
                            else (
                                "🔴 DICTATORSHIP TRAP"
                                if ge_rec.is_managerial_dictatorship_trap
                                else f"Regime: {ge_rec.regime}"
                            )
                        )
                        ge1.metric(
                            "Bebchuk E-Index",
                            f"{ge_rec.bebchuk_e_index} / 6",
                            delta=ge_badge,
                            delta_color="normal" if ge_rec.is_shareholder_democracy else ("inverse" if ge_rec.is_managerial_dictatorship_trap else "off"),
                            help="Bebchuk, Cohen & Ferrell (2009) 6-provision Entrenchment Index (Democracy <= 1, Dictatorship >= 5)",
                        )
                        ge2.metric(
                            "Gompers G-Index",
                            f"{ge_rec.gompers_g_index} / 24",
                            delta=f"Other IRRC: {ge_rec.other_irrc_provisions} / 18",
                            delta_color="inverse" if ge_rec.gompers_g_index >= 14 else "normal",
                            help="Gompers, Ishii & Metrick (2003) 24-provision Governance Index",
                        )
                        ge3.metric(
                            "Entrenchment (CMES)",
                            f"{ge_rec.composite_entrenchment_score:.2f} / 1.00",
                            delta=f"Agency Wedge: {ge_rec.agency_cost_wedge_index:.2f}",
                            delta_color="inverse" if ge_rec.composite_entrenchment_score >= 0.72 else "normal",
                            help="Composite Managerial Entrenchment Score combining E-Index, G-Index, and dual-class voting wedge",
                        )
                        ge4.metric(
                            "Expected Governance Alpha",
                            f"{ge_rec.expected_governance_alpha_pct:+.2f}% Ann",
                            delta="Democracy Premium" if ge_rec.expected_governance_alpha_pct > 0 else "Entrenchment Drag",
                            help="Projected annualized abnormal return from Gompers-Bebchuk shareholder rights discipline",
                        )

                        st.caption(f"🧾 **Governance Summary:** {ge_rec.summary}")
                        st.caption(f"🛡️ **Governance Audit Digest:** `{ge_rec.signature[:16]}...` (Verified: `{ge_rec.verify_integrity()}`)")
                    except Exception as ge_ex:
                        st.info(f"Corporate Governance evaluation deferred: {ge_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Corporate Governance profile.")

                st.divider()
                st.markdown("#### 🐸 Da, Gurun & Warachka (2014) 'Frog in the Pan' (FIP) Information Discreteness & Momentum Quality")
                if len(prices) >= 20:
                    try:
                        from src.analytics.frog_in_the_pan import evaluate_frog_in_the_pan_profile

                        with st.expander("⚙️ Optional Formation Return & Daily Sign Frequency Inputs (Da-Gurun-Warachka 2014)", expanded=False):
                            fp_c1, fp_c2, fp_c3, fp_c4 = st.columns(4)
                            fp_pret = fp_c1.number_input("Past Return PRET (decimal)", value=0.0, min_value=-0.95, max_value=10.0, step=0.02, key="fp_pret")
                            fp_pos = fp_c2.number_input("% Positive Days [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="fp_pos")
                            fp_neg = fp_c3.number_input("% Negative Days [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="fp_neg")
                            fp_jump = fp_c4.number_input("Top 3-Day Jump Share [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="fp_jump")

                        fp_rec = evaluate_frog_in_the_pan_profile(
                            ticker=selected_ticker,
                            past_cumulative_return=fp_pret if fp_pret != 0.0 else None,
                            pct_positive_days=fp_pos if fp_pos > 0.0 else None,
                            pct_negative_days=fp_neg if fp_neg > 0.0 else None,
                            max_three_day_jump_share=fp_jump if fp_jump > 0.0 else None,
                        )

                        fp1, fp2, fp3, fp4 = st.columns(4)
                        fp_badge = (
                            "🟢 CONTINUOUS FROG WINNER"
                            if fp_rec.is_continuous_frog_winner
                            else (
                                "🔴 CONTINUOUS LOSER TRAP"
                                if fp_rec.is_continuous_frog_loser_trap
                                else f"Regime: {fp_rec.regime}"
                            )
                        )
                        fp1.metric(
                            "Info Discreteness (ID_t)",
                            f"{fp_rec.information_discreteness_id:+.3f}",
                            delta=fp_badge,
                            delta_color="normal" if fp_rec.is_continuous_frog_winner else ("inverse" if fp_rec.is_continuous_frog_loser_trap else "off"),
                            help="Da, Gurun & Warachka (2014) ID = sign(PRET) * (%Neg - %Pos) (Continuous <= -0.15, Discrete > 0)",
                        )
                        fp2.metric(
                            "FIP Momentum Quality",
                            f"{fp_rec.fip_momentum_quality_score:.2f} / 1.00",
                            delta=f"PRET: {fp_rec.past_cumulative_return:+.1%}",
                            delta_color="normal" if fp_rec.fip_momentum_quality_score >= 0.62 else "off",
                            help="Composite continuous-arrival quality score penalizing lumpy 3-day jumps",
                        )
                        fp3.metric(
                            "Daily Sign Split",
                            f"{fp_rec.pct_positive_days:.0%} Up / {fp_rec.pct_negative_days:.0%} Dn",
                            delta=f"3-Day Jump: {fp_rec.max_three_day_jump_share:.0%}",
                            help="Proportion of positive vs. negative return sessions and top-3 day jump concentration",
                        )
                        fp4.metric(
                            "Expected FIP Alpha",
                            f"{fp_rec.expected_fip_alpha_pct:+.2f}% Ann",
                            delta="Boiling Frog Drift" if fp_rec.expected_fip_alpha_pct > 0 else "Downward Drift / Stall",
                            help="Projected annualized momentum continuation return from continuous information arrival",
                        )

                        st.caption(f"🧾 **FIP Summary:** {fp_rec.summary}")
                        st.caption(f"🛡️ **FIP Audit Digest:** `{fp_rec.signature[:16]}...` (Verified: `{fp_rec.verify_integrity()}`)")
                    except Exception as fp_ex:
                        st.info(f"Frog in the Pan evaluation deferred: {fp_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Frog in the Pan profile.")

                st.divider()
                st.markdown("#### 🔥 Barber & Odean (2008) & Da, Engelberg & Gao (2011) Retail Attention Spike (ASVI) & Buying Pressure Reversal")
                if len(prices) >= 20:
                    try:
                        from src.analytics.attention_salience import evaluate_attention_salience_profile

                        with st.expander("⚙️ Optional Retail Attention & Salience Inputs (Barber-Odean / Da-Engelberg-Gao)", expanded=False):
                            at_c1, at_c2, at_c3, at_c4 = st.columns(4)
                            at_asvi = at_c1.number_input("Abnormal SVI Log Change [-3..5]", value=0.0, min_value=-3.0, max_value=5.0, step=0.10, key="at_asvi")
                            at_vol = at_c2.number_input("Abnormal Volume Ratio (>0)", value=0.0, min_value=0.0, step=0.25, key="at_vol")
                            at_ret = at_c3.number_input("Extreme 1-Day Return (decimal)", value=0.0, min_value=-0.95, max_value=5.0, step=0.01, key="at_ret")
                            at_imb = at_c4.number_input("Retail Buy Imbalance [-1..1]", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key="at_imb")

                        at_rec = evaluate_attention_salience_profile(
                            ticker=selected_ticker,
                            abnormal_svi_log_change=at_asvi if at_asvi != 0.0 else None,
                            abnormal_volume_ratio=at_vol if at_vol > 0.0 else None,
                            extreme_day_return=at_ret if at_ret != 0.0 else None,
                            retail_buy_imbalance=at_imb if at_imb != 0.0 else None,
                        )

                        at1, at2, at3, at4 = st.columns(4)
                        at_badge = (
                            "🟢 UNDER-THE-RADAR COMPOUNDER"
                            if at_rec.is_under_the_radar_compounder
                            else (
                                "🔴 GLITTER BUYING CLIMAX TRAP"
                                if at_rec.is_glitter_buying_climax_trap
                                else f"Regime: {at_rec.regime}"
                            )
                        )
                        at1.metric(
                            "Barber-Odean Salience",
                            f"{at_rec.barber_odean_salience_score:.2f} / 3.00",
                            delta=at_badge,
                            delta_color="normal" if at_rec.is_under_the_radar_compounder else ("inverse" if at_rec.is_glitter_buying_climax_trap else "off"),
                            help="Barber & Odean (2008) 3-pillar attention salience score combining abnormal volume, extreme return, and ASVI",
                        )
                        at2.metric(
                            "Abnormal Search (ASVI)",
                            f"{at_rec.abnormal_svi_log_change:+.2f}",
                            delta=f"1-2Wk Boost: +{at_rec.short_term_pressure_bps:.0f}bps",
                            delta_color="inverse" if at_rec.abnormal_svi_log_change >= 0.45 else "normal",
                            help="Da, Engelberg & Gao (2011) log Search Volume Index change vs 8-week median",
                        )
                        at3.metric(
                            "Reversal Overhang (ARRO)",
                            f"{at_rec.retail_reversal_overhang_index:.2f} / 3.50",
                            delta=f"Vol Ratio: {at_rec.abnormal_volume_ratio:.2f}x",
                            delta_color="inverse" if at_rec.retail_reversal_overhang_index >= 1.85 else "normal",
                            help="Asymmetric retail buying climax overhang index",
                        )
                        at4.metric(
                            "Expected Reversal Alpha",
                            f"{at_rec.expected_reversal_alpha_pct:+.2f}% Ann",
                            delta="Quiet Compounding" if at_rec.expected_reversal_alpha_pct > 0 else "Post-Hype Reversal Drag",
                            help="Projected annualized abnormal return net of post-attention retail reversal",
                        )

                        st.caption(f"🧾 **Attention Summary:** {at_rec.summary}")
                        st.caption(f"🛡️ **Attention Audit Digest:** `{at_rec.signature[:16]}...` (Verified: `{at_rec.verify_integrity()}`)")
                    except Exception as at_ex:
                        st.info(f"Attention Salience evaluation deferred: {at_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Retail Attention Salience profile.")

                st.divider()
                st.markdown("#### 🎯 Baker, Bradley & Wurgler (2011) Institutional Benchmark Mandate & Tracking-Error Limits to Arbitrage")
                if len(prices) >= 20:
                    try:
                        from src.analytics.benchmark_mandate import evaluate_benchmark_mandate_profile

                        with st.expander("⚙️ Optional Beta, Volatility & Benchmark Mandate Inputs (Baker-Bradley-Wurgler 2011)", expanded=False):
                            bm_c1, bm_c2, bm_c3 = st.columns(3)
                            bm_beta = bm_c1.number_input("CAPM Beta [-1..4]", value=0.0, min_value=-1.0, max_value=4.0, step=0.05, key="bm_beta")
                            bm_mvol = bm_c2.number_input("Market Volatility (decimal)", value=0.0, min_value=0.0, max_value=2.0, step=0.02, key="bm_mvol")
                            bm_ivol = bm_c3.number_input("Idiosyncratic Vol (decimal)", value=0.0, min_value=0.0, max_value=3.0, step=0.02, key="bm_ivol")
                            bm_c4, bm_c5 = st.columns(2)
                            bm_erp = bm_c4.number_input("Expected Market Risk Premium (%)", value=0.0, min_value=0.0, max_value=25.0, step=0.50, key="bm_erp")
                            bm_inst = bm_c5.number_input("Benchmarked Institutional Share [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="bm_inst")

                        bm_rec = evaluate_benchmark_mandate_profile(
                            ticker=selected_ticker,
                            capm_beta=bm_beta if bm_beta != 0.0 else None,
                            market_volatility_ann=bm_mvol if bm_mvol > 0.0 else None,
                            idiosyncratic_volatility_ann=bm_ivol if bm_ivol > 0.0 else None,
                            expected_market_risk_premium_pct=bm_erp if bm_erp > 0.0 else None,
                            benchmarked_institutional_share=bm_inst if bm_inst > 0.0 else None,
                        )

                        bm1, bm2, bm3, bm4 = st.columns(4)
                        bm_badge = (
                            "🟢 NEGLECTED LOW-BETA HAVEN"
                            if bm_rec.is_neglected_low_beta_haven
                            else (
                                "🔴 CROWDED HIGH-BETA TRAP"
                                if bm_rec.is_crowded_high_beta_trap
                                else f"Regime: {bm_rec.regime}"
                            )
                        )
                        bm1.metric(
                            "Single-Stock Tracking Error",
                            f"{bm_rec.single_stock_tracking_error:.2%}",
                            delta=bm_badge,
                            delta_color="normal" if bm_rec.is_neglected_low_beta_haven else ("inverse" if bm_rec.is_crowded_high_beta_trap else "off"),
                            help="Baker, Bradley & Wurgler (2011) TE_i = sqrt((beta-1)^2 * sigma_M^2 + sigma_e^2)",
                        )
                        bm2.metric(
                            "Active Return Alpha Hurdle",
                            f"{bm_rec.active_return_alpha_hurdle_pct:+.2f}% / yr",
                            delta=f"CAPM Beta: {bm_rec.capm_beta:.2f}",
                            delta_color="normal" if bm_rec.capm_beta <= 0.75 else ("inverse" if bm_rec.capm_beta >= 1.35 else "off"),
                            help="Required CAPM alpha (1 - beta)*ERP for an unlevered manager to avoid lagging the benchmark",
                        )
                        bm3.metric(
                            "Benchmark Crowding Wedge",
                            f"{bm_rec.institutional_crowding_wedge:+.2f}",
                            delta=f"IVOL: {bm_rec.idiosyncratic_volatility_ann:.1%}",
                            delta_color="inverse" if bm_rec.institutional_crowding_wedge >= 0.85 else "normal",
                            help="Institutional neglect (< 0) vs. high-beta leverage-substitute crowding (> 0)",
                        )
                        bm4.metric(
                            "Unconstrained Alpha",
                            f"{bm_rec.expected_unconstrained_alpha_pct:+.2f}% Ann",
                            delta="Low-Risk Harvest" if bm_rec.expected_unconstrained_alpha_pct > 0 else "High-Beta Crowding Drag",
                            help="Projected annualized abnormal return for unconstrained portfolios harvesting the benchmark mandate distortion",
                        )

                        st.caption(f"🧾 **Mandate Summary:** {bm_rec.summary}")
                        st.caption(f"🛡️ **Mandate Audit Digest:** `{bm_rec.signature[:16]}...` (Verified: `{bm_rec.verify_integrity()}`)")
                    except Exception as bm_ex:
                        st.info(f"Benchmark Mandate evaluation deferred: {bm_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Institutional Benchmark Mandate profile.")

                st.divider()
                st.markdown("#### ⚓ Grinblatt & Han (2005) & Frazzini (2006) Capital Gains Overhang (CGO) & Disposition Effect")
                if len(prices) >= 20:
                    try:
                        from src.analytics.capital_gains_overhang import evaluate_capital_gains_overhang_profile

                        with st.expander("⚙️ Optional Reference Cost Basis & News Surprise Inputs (Grinblatt-Han / Frazzini)", expanded=False):
                            cg_c1, cg_c2, cg_c3, cg_c4 = st.columns(4)
                            cg_ref = cg_c1.number_input("Reference Cost Basis R_t ($)", value=0.0, min_value=0.0, step=5.0, key="cg_ref")
                            cg_turn = cg_c2.number_input("Avg Daily Turnover (0..0.5]", value=0.0, min_value=0.0, max_value=0.50, step=0.005, key="cg_turn")
                            cg_news = cg_c3.number_input("News Surprise Score [-5..5]", value=0.0, min_value=-5.0, max_value=5.0, step=0.25, key="cg_news")
                            cg_shld = cg_c4.number_input("Institutional Rationality Shield [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="cg_shld")

                        cg_rec = evaluate_capital_gains_overhang_profile(
                            ticker=selected_ticker,
                            aggregate_reference_price=cg_ref if cg_ref > 0.0 else None,
                            avg_daily_turnover=cg_turn if cg_turn > 0.0 else None,
                            news_surprise_score=cg_news if cg_news != 0.0 else None,
                            institutional_rationality_shield=cg_shld if cg_shld > 0.0 else None,
                        )

                        cg1, cg2, cg3, cg4 = st.columns(4)
                        cg_badge = (
                            "🟢 GAIN OVERHANG LEADER"
                            if cg_rec.is_gain_overhang_leader
                            else (
                                "🔴 LOSS LOCK-IN TRAP"
                                if cg_rec.is_loss_lock_in_trap
                                else f"Regime: {cg_rec.regime}"
                            )
                        )
                        cg1.metric(
                            "Capital Gains Overhang",
                            f"{cg_rec.capital_gains_overhang:+.2%}",
                            delta=cg_badge,
                            delta_color="normal" if cg_rec.is_gain_overhang_leader else ("inverse" if cg_rec.is_loss_lock_in_trap else "off"),
                            help="Grinblatt & Han (2005) CGO_t = (P_t - R_t) / P_t relative to turnover-weighted reference cost basis",
                        )
                        cg2.metric(
                            "Reference Cost Basis (R_t)",
                            f"${cg_rec.aggregate_reference_price:,.2f}",
                            delta=f"Spot P_t: ${cg_rec.latest_price:,.2f}",
                            help="Aggregate mental-accounting reference cost basis across current shareholders",
                        )
                        cg3.metric(
                            "Frazzini Impulse (FDUI)",
                            f"{cg_rec.frazzini_disposition_impulse:+.3f}",
                            delta=f"News Align: {cg_rec.news_overhang_alignment_mult:.2f}x",
                            delta_color="normal" if cg_rec.frazzini_disposition_impulse >= 0.18 else ("inverse" if cg_rec.frazzini_disposition_impulse <= -0.22 else "off"),
                            help="Frazzini (2006) disposition underreaction impulse interacting CGO with news surprise",
                        )
                        cg4.metric(
                            "Expected Disposition Alpha",
                            f"{cg_rec.expected_cgo_alpha_pct:+.2f}% Ann",
                            delta="Disposition Drift" if cg_rec.expected_cgo_alpha_pct > 0 else "Loss Lock-In Drag",
                            help="Projected annualized abnormal return from Grinblatt-Han & Frazzini disposition underreaction",
                        )

                        st.caption(f"🧾 **CGO Summary:** {cg_rec.summary}")
                        st.caption(f"🛡️ **CGO Audit Digest:** `{cg_rec.signature[:16]}...` (Verified: `{cg_rec.verify_integrity()}`)")
                    except Exception as cg_ex:
                        st.info(f"Capital Gains Overhang evaluation deferred: {cg_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Capital Gains Overhang profile.")

                st.divider()
                st.markdown("#### 🎯 George & Hwang (2004) 52-Week High Psychological Anchoring Engine")
                st.caption(
                    "Evaluates nearness to the 52-week high ($NH_{52,t} = P_t / H_{52,t}$), psychological anchor resistance, "
                    "non-reversal stability, and George-Hwang anchoring continuation vs. dead-cat bounce divergence."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.fifty_two_week_high import (
                            evaluate_fifty_two_week_high_profile,
                        )

                        with st.expander("⚙️ Customize 52-Week High & Past Return Parameters (Optional)", expanded=False):
                            gh_c1, gh_c2, gh_c3, gh_c4 = st.columns(4)
                            gh_high = gh_c1.number_input("52-Week High H_52 ($ >= P_t, 0=Auto)", value=0.0, min_value=0.0, max_value=100000.0, step=1.0, key="gh_high")
                            gh_low = gh_c2.number_input("52-Week Low L_52 ($ <= P_t, 0=Auto)", value=0.0, min_value=0.0, max_value=100000.0, step=1.0, key="gh_low")
                            gh_pret = gh_c3.number_input("Past Return PRET_t (-0.95..5.0]", value=0.0, min_value=-0.90, max_value=5.0, step=0.05, key="gh_pret")
                            gh_vol = gh_c4.number_input("Breakout Volume Ratio [0.1..10]", value=0.0, min_value=0.0, max_value=10.0, step=0.1, key="gh_vol")

                        gh_rec = evaluate_fifty_two_week_high_profile(
                            ticker=selected_ticker,
                            high_52w=gh_high if gh_high > 0.0 else None,
                            low_52w=gh_low if gh_low > 0.0 else None,
                            past_return_20d=gh_pret if gh_pret != 0.0 else None,
                            breakout_volume_ratio=gh_vol if gh_vol > 0.0 else None,
                        )

                        gh1, gh2, gh3, gh4 = st.columns(4)
                        gh_badge = (
                            "🟢 52W-HIGH BREAKOUT LEADER"
                            if gh_rec.is_52w_high_breakout_leader
                            else (
                                "🔴 DEEP ANCHOR OVERHANG TRAP"
                                if gh_rec.is_deep_anchor_overhang_trap
                                else f"Regime: {gh_rec.regime}"
                            )
                        )
                        gh1.metric(
                            "Nearness to 52W High",
                            f"{gh_rec.nearness_to_52w_high:.2%}",
                            delta=gh_badge,
                            delta_color="normal" if gh_rec.is_52w_high_breakout_leader else ("inverse" if gh_rec.is_deep_anchor_overhang_trap else "off"),
                            help="George & Hwang (2004) NH_52,t = P_t / H_52,t nearness ratio to 252-day maximum price",
                        )
                        gh2.metric(
                            "52-Week High Anchor",
                            f"${gh_rec.high_52w:,.2f}",
                            delta=f"Range Pos: {gh_rec.range_position_52w:.1%}",
                            help="Rolling 252-session maximum price reference point and position in 52-week range",
                        )
                        gh3.metric(
                            "George-Hwang Impulse (GHAI)",
                            f"{gh_rec.george_hwang_anchoring_impulse:+.3f}",
                            delta=f"Vol Ratio: {gh_rec.breakout_volume_ratio:.2f}x",
                            delta_color="normal" if gh_rec.george_hwang_anchoring_impulse >= 0.45 else ("inverse" if gh_rec.george_hwang_anchoring_impulse <= -0.35 else "off"),
                            help="George & Hwang (2004) anchoring continuation impulse and breakout volume ratio",
                        )
                        gh4.metric(
                            "Expected Anchoring Alpha",
                            f"{gh_rec.expected_anchoring_alpha_pct:+.2f}% Ann",
                            delta="Anchor Underreaction Drift" if gh_rec.expected_anchoring_alpha_pct > 0 else "Distant-From-High Drag",
                            help="Projected annualized abnormal return from George & Hwang (2004) 52-week high anchoring",
                        )

                        st.caption(f"🧾 **52W-High Summary:** {gh_rec.summary}")
                        st.caption(f"🛡️ **52W-High Audit Digest:** `{gh_rec.signature[:16]}...` (Verified: `{gh_rec.verify_integrity()}`)")
                    except Exception as gh_ex:
                        st.info(f"52-Week High Anchoring evaluation deferred: {gh_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate 52-Week High Anchoring profile.")

                st.divider()
                st.markdown("#### 🔬 Pástor & Veronesi (2003, 2006) Bayesian Learning & Uncertainty Convexity Engine")
                st.caption(
                    "Evaluates Bayesian posterior profitability uncertainty ($\\sigma_{\\psi,\\text{post}}$), Jensen's inequality "
                    "convexity multiplier ($C_{PV}$), warranted Market-to-Book ($(M/B)^*_{PV}$), and uncertainty bubble risk."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.pastor_veronesi_uncertainty import (
                            evaluate_pastor_veronesi_uncertainty_profile,
                        )

                        with st.expander("⚙️ Customize Bayesian Learning & Convexity Parameters (Optional)", expanded=False):
                            pv_c1, pv_c2, pv_c3, pv_c4, pv_c5 = st.columns(5)
                            pv_psi = pv_c1.number_input("Excess ROE psi_hat [-0.5..0.8]", value=0.0, min_value=-0.50, max_value=0.80, step=0.02, key="pv_psi")
                            pv_sig = pv_c2.number_input("Prior Uncertainty [0.005..0.5]", value=0.0, min_value=0.0, max_value=0.50, step=0.02, key="pv_sig")
                            pv_age = pv_c3.number_input("Firm Age (Years, 0=Auto)", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="pv_age")
                            pv_ret = pv_c4.number_input("Retention Ratio [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="pv_ret")
                            pv_mb = pv_c5.number_input("Observed M/B [0.2..50]", value=0.0, min_value=0.0, max_value=50.0, step=0.5, key="pv_mb")

                        pv_rec = evaluate_pastor_veronesi_uncertainty_profile(
                            ticker=selected_ticker,
                            expected_excess_profitability=pv_psi if pv_psi != 0.0 else None,
                            prior_profitability_uncertainty=pv_sig if pv_sig > 0.0 else None,
                            firm_learning_age_years=pv_age if pv_age > 0.0 else None,
                            retention_ratio=pv_ret if pv_ret > 0.0 else None,
                            observed_market_to_book=pv_mb if pv_mb > 0.0 else None,
                        )

                        pv1, pv2, pv3, pv4 = st.columns(4)
                        pv_badge = (
                            "🟢 CONVEX LEARNING LEADER"
                            if pv_rec.is_convex_learning_leader
                            else (
                                "🔴 UNCERTAINTY BUBBLE TRAP"
                                if pv_rec.is_uncertainty_bubble_trap
                                else f"Regime: {pv_rec.regime}"
                            )
                        )
                        pv1.metric(
                            "Warranted vs. Obs M/B",
                            f"{pv_rec.warranted_market_to_book:.2f}x",
                            delta=f"Obs: {pv_rec.observed_market_to_book:.2f}x ({pv_rec.valuation_mispricing_gap_pct:+.1%})",
                            delta_color="normal" if pv_rec.is_convex_learning_leader else ("inverse" if pv_rec.is_uncertainty_bubble_trap else "off"),
                            help="Pástor & Veronesi (2003) warranted uncertainty-adjusted Market-to-Book ratio vs. observed M/B",
                        )
                        pv2.metric(
                            "Convexity Multiplier (C_PV)",
                            f"{pv_rec.convexity_valuation_multiplier:.2f}x",
                            delta=f"Post Sigma: {pv_rec.posterior_profitability_uncertainty:.1%}",
                            help="Jensen's inequality valuation multiplier from posterior profitability uncertainty",
                        )
                        pv3.metric(
                            "Learning Impulse (PVLI)",
                            f"{pv_rec.pastor_veronesi_learning_impulse:+.3f}",
                            delta=pv_badge,
                            delta_color="normal" if pv_rec.pastor_veronesi_learning_impulse >= 0.35 else ("inverse" if pv_rec.pastor_veronesi_learning_impulse <= -0.40 else "off"),
                            help="Pástor-Veronesi Bayesian learning & convexity valuation impulse",
                        )
                        pv4.metric(
                            "Expected Learning Alpha",
                            f"{pv_rec.expected_learning_alpha_pct:+.2f}% Ann",
                            delta="Convexity Undervaluation" if pv_rec.expected_learning_alpha_pct > 0 else "Bubble Compression Risk",
                            help="Projected annualized abnormal return from Pástor-Veronesi learning & convexity convergence",
                        )

                        st.caption(f"🧾 **Pástor-Veronesi Summary:** {pv_rec.summary}")
                        st.caption(f"🛡️ **Pástor-Veronesi Audit Digest:** `{pv_rec.signature[:16]}...` (Verified: `{pv_rec.verify_integrity()}`)")
                    except Exception as pv_ex:
                        st.info(f"Pástor-Veronesi Uncertainty evaluation deferred: {pv_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Pástor-Veronesi Uncertainty profile.")

                st.divider()
                st.markdown("#### 🌊 Bansal & Yaron (2004) Long-Run Risks (LRR) & Epstein-Zin Cash-Flow Beta Engine")
                st.caption(
                    "Evaluates Epstein-Zin preference for early resolution of uncertainty ($\\Theta_{EZ} = \\gamma - 1/\\psi$), "
                    "persistent cash-flow growth beta ($\\beta_{x,i}$), macroeconomic volatility shock beta ($\\beta_{\\sigma,i}$), and LRR alpha."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.long_run_risks import (
                            evaluate_long_run_risks_profile,
                        )

                        with st.expander("⚙️ Customize Bansal-Yaron Long-Run Risks Parameters (Optional)", expanded=False):
                            lr_c1, lr_c2, lr_c3, lr_c4, lr_c5 = st.columns(5)
                            lr_bx = lr_c1.number_input("Cash-Flow Beta [-2..4]", value=0.0, min_value=-2.0, max_value=4.0, step=0.1, key="lr_bx")
                            lr_bs = lr_c2.number_input("Uncertainty Beta [-3..3]", value=0.0, min_value=-3.0, max_value=3.0, step=0.1, key="lr_bs")
                            lr_rho = lr_c3.number_input("Persistence rho [0.5..0.995]", value=0.0, min_value=0.0, max_value=0.995, step=0.01, key="lr_rho")
                            lr_ies = lr_c4.number_input("Epstein-Zin IES [0.5..3.0]", value=0.0, min_value=0.0, max_value=3.0, step=0.1, key="lr_ies")
                            lr_rra = lr_c5.number_input("Risk Aversion [1.5..20]", value=0.0, min_value=0.0, max_value=20.0, step=0.5, key="lr_rra")

                        lr_rec = evaluate_long_run_risks_profile(
                            ticker=selected_ticker,
                            cashflow_growth_beta=lr_bx if lr_bx != 0.0 else None,
                            uncertainty_shock_beta=lr_bs if lr_bs != 0.0 else None,
                            growth_persistence_rho=lr_rho if lr_rho >= 0.50 else None,
                            intertemporal_elasticity_ies=lr_ies if lr_ies >= 0.50 else None,
                            relative_risk_aversion=lr_rra if lr_rra >= 1.50 else None,
                        )

                        lr1, lr2, lr3, lr4 = st.columns(4)
                        lr_badge = (
                            "🟢 HIGH LRR GROWTH LEADER"
                            if lr_rec.is_high_lrr_growth_premium_leader
                            else (
                                "🛡️ MACRO SAFE-HAVEN HEDGE"
                                if lr_rec.is_macro_uncertainty_safe_haven
                                else f"Regime: {lr_rec.regime}"
                            )
                        )
                        lr1.metric(
                            "Cash-Flow Growth Beta",
                            f"{lr_rec.cashflow_growth_beta:+.2f}",
                            delta=f"Growth Prem: {lr_rec.persistent_growth_premium_pct:+.2f}%",
                            delta_color="normal" if lr_rec.is_high_lrr_growth_premium_leader else "off",
                            help="Bansal, Dittmar & Lundblad (2005) persistent cash-flow growth beta and annualized growth risk premium",
                        )
                        lr2.metric(
                            "Uncertainty Shock Beta",
                            f"{lr_rec.uncertainty_shock_beta:+.2f}",
                            delta=f"Vol Prem: {lr_rec.uncertainty_volatility_premium_pct:+.2f}%",
                            help="Exposure to macroeconomic consumption volatility shocks under Epstein-Zin preferences",
                        )
                        lr3.metric(
                            "Bansal-Yaron Impulse (LRRI)",
                            f"{lr_rec.bansal_yaron_lrr_impulse:+.3f}",
                            delta=lr_badge,
                            delta_color="normal" if lr_rec.bansal_yaron_lrr_impulse >= 0.45 else ("inverse" if lr_rec.bansal_yaron_lrr_impulse <= -0.45 else "off"),
                            help="Composite Bansal & Yaron (2004) Long-Run Risks pricing impulse",
                        )
                        lr4.metric(
                            "Expected LRR Risk Premium",
                            f"{lr_rec.expected_lrr_alpha_pct:+.2f}% Ann",
                            delta=f"A_x: {lr_rec.wealth_growth_sensitivity_ax:+.1f} | Theta: {lr_rec.early_resolution_preference_theta:.1f}",
                            help="Total projected annualized Long-Run Risks equilibrium risk premium",
                        )

                        st.caption(f"🧾 **Long-Run Risks Summary:** {lr_rec.summary}")
                        st.caption(f"🛡️ **LRR Audit Digest:** `{lr_rec.signature[:16]}...` (Verified: `{lr_rec.verify_integrity()}`)")
                    except Exception as lr_ex:
                        st.info(f"Long-Run Risks evaluation deferred: {lr_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Long-Run Risks profile.")

                st.divider()
                st.markdown("#### 🏭 Lu Zhang (2005) Costly Reversibility of Capital & Countercyclical Value Beta Engine")
                st.caption(
                    "Evaluates physical capital adjustment asymmetry ($\\theta^- / \\theta^+$), disinvestment friction ($\\Phi_{\\text{disinv}}$), "
                    "countercyclical value beta ($\\beta_{\\text{Zhang},t}$), and stranded-capital recession risk."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.costly_reversibility import (
                            evaluate_costly_reversibility_profile,
                        )

                        with st.expander("⚙️ Customize Lu Zhang Costly Reversibility Parameters (Optional)", expanded=False):
                            lz_c1, lz_c2, lz_c3, lz_c4 = st.columns(4)
                            lz_bm = lz_c1.number_input("Book-to-Market [0.05..5.0]", value=0.0, min_value=0.0, max_value=5.0, step=0.1, key="lz_bm")
                            lz_as = lz_c2.number_input("Cost Asymmetry [1..25]", value=0.0, min_value=0.0, max_value=25.0, step=1.0, key="lz_as")
                            lz_fk = lz_c3.number_input("Fixed Capital f_K [0..1]", value=0.0, min_value=0.0, max_value=1.0, step=0.05, key="lz_fk")
                            lz_mc = lz_c4.number_input("Macro State Z [-3..3]", value=0.0, min_value=-3.0, max_value=3.0, step=0.25, key="lz_mc")

                        lz_rec = evaluate_costly_reversibility_profile(
                            ticker=selected_ticker,
                            book_to_market_ratio=lz_bm if lz_bm > 0.0 else None,
                            downward_upward_cost_asymmetry=lz_as if lz_as >= 1.0 else None,
                            fixed_capital_intensity=lz_fk if lz_fk > 0.0 else None,
                            macro_state_zscore=lz_mc if lz_mc != 0.0 else None,
                        )

                        lz1, lz2, lz3, lz4 = st.columns(4)
                        lz_badge = (
                            "🟢 COMPENSATED VALUE LEADER"
                            if lz_rec.is_compensated_irreversible_value_leader
                            else (
                                "🔴 STRANDED CAPITAL TRAP"
                                if lz_rec.is_stranded_capital_recession_trap
                                else f"Regime: {lz_rec.regime}"
                            )
                        )
                        lz1.metric(
                            "Countercyclical Value Beta",
                            f"{lz_rec.countercyclical_value_beta:.2f}",
                            delta=f"B/M: {lz_rec.book_to_market_ratio:.2f}x",
                            delta_color="normal" if lz_rec.is_compensated_irreversible_value_leader else ("inverse" if lz_rec.is_stranded_capital_recession_trap else "off"),
                            help="Lu Zhang (2005) conditional market beta amplified by costly capital reversibility in contractions",
                        )
                        lz2.metric(
                            "Capital Irreversibility (CII)",
                            f"{lz_rec.capital_irreversibility_index:.2f}",
                            delta=f"Friction Phi: {lz_rec.disinvestment_friction_mult:.2f}x",
                            help="Asymmetric disinvestment adjustment cost index and contraction friction multiplier",
                        )
                        lz3.metric(
                            "Lu Zhang Impulse (LZVI)",
                            f"{lz_rec.lu_zhang_value_impulse:+.3f}",
                            delta=lz_badge,
                            delta_color="normal" if lz_rec.lu_zhang_value_impulse >= 0.40 else ("inverse" if lz_rec.lu_zhang_value_impulse <= -0.35 else "off"),
                            help="Composite Lu Zhang (2005) costly reversibility value pricing impulse",
                        )
                        lz4.metric(
                            "Expected Reversibility Alpha",
                            f"{lz_rec.expected_reversibility_alpha_pct:+.2f}% Ann",
                            delta="Value Risk Premium" if lz_rec.expected_reversibility_alpha_pct > 0 else "Stranded Overhead Drag",
                            help="Projected annualized abnormal return from Lu Zhang costly capital reversibility",
                        )

                        st.caption(f"🧾 **Costly Reversibility Summary:** {lz_rec.summary}")
                        st.caption(f"🛡️ **Reversibility Audit Digest:** `{lz_rec.signature[:16]}...` (Verified: `{lz_rec.verify_integrity()}`)")
                    except Exception as lz_ex:
                        st.info(f"Costly Reversibility evaluation deferred: {lz_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Costly Reversibility profile.")

                st.divider()
                st.markdown("#### ⚖️ Campbell & Shiller (1988) / Cochrane (2008, 2011) Present-Value & Discount-Rate Decomposition Engine")
                st.caption(
                    "Evaluates the exact present-value log-linearization identity ($1 = S_{DR} + S_{CF}$), decomposing valuation ratio "
                    "movements into time-varying discount rates ($S_{DR} = b_r^{LR}$) versus cash-flow growth ($S_{CF} = -b_d^{LR}$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.campbell_shiller_cochrane import (
                            evaluate_campbell_shiller_cochrane_profile,
                        )

                        with st.expander("⚙️ Customize Present-Value Decomposition Parameters (Optional)", expanded=False):
                            cs_c1, cs_c2, cs_c3, cs_c4 = st.columns(4)
                            cs_dp = cs_c1.number_input("Log Yield dp_t [-6..0]", value=0.0, min_value=-6.0, max_value=0.0, step=0.1, key="cs_dp")
                            cs_bar = cs_c2.number_input("Mean Yield bar{dp} [-6..0]", value=0.0, min_value=-6.0, max_value=0.0, step=0.1, key="cs_bar")
                            cs_phi = cs_c3.number_input("Persistence phi [0.5..0.995]", value=0.0, min_value=0.0, max_value=0.995, step=0.01, key="cs_phi")
                            cs_br = cs_c4.number_input("Return Slope b_r1 [-0.5..1.0]", value=0.0, min_value=-0.50, max_value=1.0, step=0.01, key="cs_br")

                        cs_rec = evaluate_campbell_shiller_cochrane_profile(
                            ticker=selected_ticker,
                            log_dividend_price_ratio=cs_dp if cs_dp != 0.0 else None,
                            mean_log_dividend_price=cs_bar if cs_bar != 0.0 else None,
                            valuation_persistence_phi=cs_phi if cs_phi >= 0.50 else None,
                            one_period_return_slope=cs_br if cs_br != 0.0 else None,
                        )

                        cs1, cs2, cs3, cs4 = st.columns(4)
                        cs_badge = (
                            "🟢 DISCOUNT-RATE OPPORTUNITY"
                            if cs_rec.is_discount_rate_opportunity
                            else (
                                "🔴 COMPRESSED DR TRAP"
                                if cs_rec.is_compressed_discount_rate_trap
                                else f"Regime: {cs_rec.regime}"
                            )
                        )
                        cs1.metric(
                            "Discount-Rate Share (S_DR)",
                            f"{cs_rec.discount_rate_variance_share:.1%}",
                            delta=f"Cash-Flow: {cs_rec.cashflow_variance_share:.1%}",
                            delta_color="normal" if cs_rec.is_discount_rate_opportunity else ("inverse" if cs_rec.is_compressed_discount_rate_trap else "off"),
                            help="Cochrane (2008, 2011) long-run discount-rate elasticity S_DR = b_r^LR = b_r1 / (1 - rho * phi)",
                        )
                        cs2.metric(
                            "Log Yield (dp_t)",
                            f"{cs_rec.log_dividend_price_ratio:.2f}",
                            delta=f"Dev: {cs_rec.yield_deviation:+.2f} (bar: {cs_rec.mean_log_dividend_price:.2f})",
                            help="Current log dividend/payout-to-price ratio and deviation from long-run historical mean",
                        )
                        cs3.metric(
                            "Cochrane Impulse (CSCI)",
                            f"{cs_rec.cochrane_discount_rate_impulse:+.3f}",
                            delta=cs_badge,
                            delta_color="normal" if cs_rec.cochrane_discount_rate_impulse >= 0.40 else ("inverse" if cs_rec.cochrane_discount_rate_impulse <= -0.40 else "off"),
                            help="Campbell-Shiller & Cochrane present-value return forecasting impulse",
                        )
                        cs4.metric(
                            "Expected Mean-Reversion Alpha",
                            f"{cs_rec.expected_discount_rate_alpha_pct:+.2f}% Ann",
                            delta="Discount-Rate Drift" if cs_rec.expected_discount_rate_alpha_pct > 0 else "Multiple Compression Drag",
                            help="Projected annualized abnormal return from discount-rate present-value mean-reversion",
                        )

                        st.caption(f"🧾 **Decomposition Summary:** {cs_rec.summary}")
                        st.caption(f"🛡️ **Decomposition Audit Digest:** `{cs_rec.signature[:16]}...` (Verified: `{cs_rec.verify_integrity()}`)")
                    except Exception as cs_ex:
                        st.info(f"Campbell-Shiller Cochrane evaluation deferred: {cs_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Campbell-Shiller Cochrane profile.")

                st.divider()
                st.markdown("#### 🧩 Lamont & Thaler (2003) & Mitchell-Pulvino-Stafford (2002) Stub Value Arbitrage Engine")
                st.caption(
                    "Evaluates equity carve-out stub value ($S_{\\text{stub}} = P_{\\text{parent}} - \\alpha \\cdot P_{\\text{sub}}$), "
                    "Law of One Price (LOOP) violations, and short-sale borrow fee arbitrage limits."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.stub_value_arbitrage import (
                            evaluate_stub_value_arbitrage_profile,
                        )

                        with st.expander("⚙️ Customize Equity Carve-Out & Borrow Friction Parameters (Optional)", expanded=False):
                            st_c1, st_c2, st_c3, st_c4 = st.columns(4)
                            st_sub = st_c1.text_input("Carved Subsidiary Ticker", value="SUB", key="st_sub")
                            st_prc = st_c2.number_input("Subsidiary Price ($)", value=0.0, min_value=0.0, max_value=100000.0, step=1.0, key="st_prc")
                            st_rat = st_c3.number_input("Shares Per Parent (alpha)", value=0.0, min_value=0.0, max_value=10.0, step=0.05, key="st_rat")
                            st_fee = st_c4.number_input("Short Borrow Fee (% Ann)", value=0.0, min_value=0.0, max_value=100.0, step=1.0, key="st_fee")

                        st_rec = evaluate_stub_value_arbitrage_profile(
                            ticker=selected_ticker,
                            subsidiary_ticker=st_sub if st_sub and st_sub != "SUB" else None,
                            subsidiary_price=st_prc if st_prc > 0.0 else None,
                            shares_per_parent_ratio=st_rat if st_rat > 0.0 else None,
                            annual_borrow_fee_pct=st_fee if st_fee > 0.0 else None,
                        )

                        st1, st2, st3, st4 = st.columns(4)
                        st_badge = (
                            "🔴 NEGATIVE STUB VIOLATION"
                            if st_rec.is_negative_stub_violation
                            else (
                                "🟢 COMPRESSED STUB OPPORTUNITY"
                                if st_rec.is_compressed_stub_opportunity
                                else f"Regime: {st_rec.regime}"
                            )
                        )
                        st1.metric(
                            "Implied Stub Value",
                            f"${st_rec.implied_stub_value:,.2f}",
                            delta=f"Fraction: {st_rec.stub_fraction_of_parent:+.1%}",
                            delta_color="inverse" if st_rec.is_negative_stub_violation else ("normal" if st_rec.is_compressed_stub_opportunity else "off"),
                            help="Lamont & Thaler (2003) implied market value of parent core business S_stub = P_parent - alpha * P_sub",
                        )
                        st2.metric(
                            "Carved Sub Value / Parent",
                            f"${st_rec.implied_subsidiary_value_per_share:,.2f}",
                            delta=f"Sub P_sub: ${st_rec.subsidiary_price:.2f} ({st_rec.shares_per_parent_ratio:.2f}x)",
                            help="Embedded market value of subsidiary holding per parent share",
                        )
                        st3.metric(
                            "Arbitrage Impulse (LTAI)",
                            f"{st_rec.lamont_thaler_arbitrage_impulse:+.3f}",
                            delta=st_badge,
                            delta_color="normal" if st_rec.is_compressed_stub_opportunity else ("inverse" if st_rec.is_negative_stub_violation else "off"),
                            help="Lamont-Thaler & Mitchell-Pulvino-Stafford stub arbitrage and short friction impulse",
                        )
                        st4.metric(
                            "Expected Convergence Alpha",
                            f"{st_rec.expected_convergence_alpha_pct:+.2f}% Ann",
                            delta=f"Borrow Fee: {st_rec.annual_borrow_fee_pct:.1f}% | Friction: {st_rec.short_sale_friction_multiplier:.2f}x",
                            help="Projected annualized abnormal return from stub parity convergence net of short borrowing frictions",
                        )

                        st.caption(f"🧾 **Stub Value Summary:** {st_rec.summary}")
                        st.caption(f"🛡️ **Stub Audit Digest:** `{st_rec.signature[:16]}...` (Verified: `{st_rec.verify_integrity()}`)")
                    except Exception as st_ex:
                        st.info(f"Stub Value Arbitrage evaluation deferred: {st_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Stub Value Arbitrage profile.")

                st.divider()
                st.markdown("#### 🌪️ Brunnermeier & Pedersen (2009) Market & Funding Liquidity Spirals Engine")
                st.caption(
                    "Evaluates the interaction between prime brokerage margin haircuts ($m_t / m_0$), trader equity margin buffers ($B_{\\text{fund}}$), "
                    "the liquidity spiral multiplier ($\\mathcal{M}_{\\text{spiral}}$), and fire-sale liquidation risk."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.liquidity_spiral import (
                            evaluate_liquidity_spiral_profile,
                        )

                        with st.expander("⚙️ Customize Margin Haircut & Funding Liquidity Parameters (Optional)", expanded=False):
                            ls_c1, ls_c2, ls_c3, ls_c4 = st.columns(4)
                            ls_mt = ls_c1.number_input("Margin Haircut m_t [0.02..0.85]", value=0.0, min_value=0.0, max_value=0.85, step=0.02, key="ls_mt")
                            ls_m0 = ls_c2.number_input("Base Margin m_0 [0.02..0.50]", value=0.0, min_value=0.0, max_value=0.50, step=0.02, key="ls_m0")
                            ls_buf = ls_c3.number_input("Funding Buffer [-1.0..1.0]", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key="ls_buf")
                            ls_vol = ls_c4.number_input("Annualized Vol [0.05..1.50]", value=0.0, min_value=0.0, max_value=1.50, step=0.02, key="ls_vol")

                        ls_rec = evaluate_liquidity_spiral_profile(
                            ticker=selected_ticker,
                            current_margin_haircut=ls_mt if ls_mt > 0.0 else None,
                            baseline_margin_haircut=ls_m0 if ls_m0 > 0.0 else None,
                            funding_margin_buffer=ls_buf if ls_buf != 0.0 else None,
                            annualized_volatility=ls_vol if ls_vol > 0.0 else None,
                        )

                        ls1, ls2, ls3, ls4 = st.columns(4)
                        ls_badge = (
                            "🔴 FIRE-SALE SPIRAL TRAP"
                            if ls_rec.is_fire_sale_spiral_trap
                            else (
                                "🟢 FUNDING SAFE HAVEN"
                                if ls_rec.is_funding_liquidity_safe_haven
                                else f"Regime: {ls_rec.regime}"
                            )
                        )
                        ls1.metric(
                            "Spiral Multiplier",
                            f"{ls_rec.liquidity_spiral_multiplier:.2f}x",
                            delta=f"Margin: {ls_rec.margin_tightness_ratio:.2f}x",
                            delta_color="inverse" if ls_rec.is_fire_sale_spiral_trap else ("normal" if ls_rec.is_funding_liquidity_safe_haven else "off"),
                            help="Brunnermeier & Pedersen (2009) liquidity spiral amplification multiplier M_spiral",
                        )
                        ls2.metric(
                            "Margin Haircut (m_t)",
                            f"{ls_rec.current_margin_haircut:.1%}",
                            delta=f"Base m_0: {ls_rec.baseline_margin_haircut:.1%}",
                            help="Current prime brokerage margin requirement vs normal-times baseline",
                        )
                        ls3.metric(
                            "Funding Buffer (B_fund)",
                            f"{ls_rec.funding_margin_buffer:+.1%}",
                            delta=ls_badge,
                            delta_color="normal" if ls_rec.is_funding_liquidity_safe_haven else ("inverse" if ls_rec.is_fire_sale_spiral_trap else "off"),
                            help="Trader equity margin buffer; negative values denote margin calls and forced liquidation",
                        )
                        ls4.metric(
                            "Expected Liquidity Alpha",
                            f"{ls_rec.expected_liquidity_alpha_pct:+.2f}% Ann",
                            delta=f"Fragility: {ls_rec.funding_fragility_score:.2f} | Impulse: {ls_rec.brunnermeier_pedersen_impulse:+.2f}",
                            help="Projected annualized abnormal return net of fire-sale discount and liquidity dry-up risk",
                        )

                        st.caption(f"🧾 **Liquidity Spiral Summary:** {ls_rec.summary}")
                        st.caption(f"🛡️ **Spiral Audit Digest:** `{ls_rec.signature[:16]}...` (Verified: `{ls_rec.verify_integrity()}`)")
                    except Exception as ls_ex:
                        st.info(f"Liquidity Spiral evaluation deferred: {ls_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Liquidity Spiral profile.")

                st.divider()
                st.markdown("#### 🏛️ He-Kelly-Manela (2017) Intermediary Asset Pricing & Dealer Capital Capacity Engine")
                st.caption(
                    "Evaluates primary dealer aggregate equity capital ratio ($\\eta_t$), balance sheet capacity ($\\mathcal{C}_{\\text{dealer}}$), "
                    "intermediary capital beta ($\\beta_{\\eta, i}$), and leverage constraints under Adrian-Shin (2010) and He-Kelly-Manela (2017)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.intermediary_asset_pricing import (
                            evaluate_intermediary_pricing_profile,
                        )

                        with st.expander("⚙️ Customize Primary Dealer Capital & Intermediary Beta Parameters (Optional)", expanded=False):
                            iap_c1, iap_c2, iap_c3, iap_c4 = st.columns(4)
                            iap_eta = iap_c1.number_input("Dealer Capital Ratio eta_t [0.01..0.40]", value=0.0, min_value=0.0, max_value=0.40, step=0.01, key="iap_eta")
                            iap_eta0 = iap_c2.number_input("Base Capital Ratio eta_0 [0.02..0.30]", value=0.0, min_value=0.0, max_value=0.30, step=0.01, key="iap_eta0")
                            iap_lev = iap_c3.number_input("Dealer Leverage Lev_t [5.0..50.0]", value=0.0, min_value=0.0, max_value=50.0, step=1.0, key="iap_lev")
                            iap_beta = iap_c4.number_input("Intermediary Beta beta_eta [-2.0..4.0]", value=0.0, min_value=-2.0, max_value=4.0, step=0.1, key="iap_beta")

                        iap_rec = evaluate_intermediary_pricing_profile(
                            ticker=selected_ticker,
                            dealer_capital_ratio=iap_eta if iap_eta > 0.0 else None,
                            baseline_capital_ratio=iap_eta0 if iap_eta0 > 0.0 else None,
                            dealer_leverage_ratio=iap_lev if iap_lev > 0.0 else None,
                            intermediary_capital_beta=iap_beta if iap_beta != 0.0 else None,
                        )

                        iap1, iap2, iap3, iap4 = st.columns(4)
                        iap_badge = (
                            "🔴 LEVERAGE CONSTRAINED"
                            if iap_rec.regime == "LEVERAGE_CONSTRAINED_DEALER_OVERHANG"
                            else (
                                "🟢 INTERMEDIARY SAFE HAVEN"
                                if iap_rec.is_intermediary_distress_safe_haven
                                else (
                                    "⚡ RISK CAPITAL LEADER"
                                    if iap_rec.is_high_intermediary_risk_leader
                                    else f"Regime: {iap_rec.regime}"
                                )
                            )
                        )
                        iap1.metric(
                            "Dealer Capital Ratio (η_t)",
                            f"{iap_rec.dealer_capital_ratio:.1%}",
                            delta=f"Base η_0: {iap_rec.baseline_capital_ratio:.1%}",
                            delta_color="normal" if iap_rec.dealer_capital_ratio >= iap_rec.baseline_capital_ratio else "inverse",
                            help="He-Kelly-Manela (2017) market equity of primary dealers divided by total dealer assets",
                        )
                        iap2.metric(
                            "Dealer Capacity (C_dealer)",
                            f"{iap_rec.balance_sheet_capacity_index:.2f}x",
                            delta=f"Lev: {iap_rec.dealer_leverage_ratio:.1f}x",
                            delta_color="inverse" if iap_rec.regime == "LEVERAGE_CONSTRAINED_DEALER_OVERHANG" else "normal",
                            help="Dealer balance sheet capacity scaling factor C_dealer = (eta_t / eta_0) * (1 - 0.75 * max(0, (Lev_t - Lev_0)/Lev_0))",
                        )
                        iap3.metric(
                            "Intermediary Beta (β_η)",
                            f"{iap_rec.intermediary_capital_beta:+.2f}",
                            delta=iap_badge,
                            delta_color="normal" if iap_rec.is_intermediary_distress_safe_haven else ("inverse" if iap_rec.regime == "LEVERAGE_CONSTRAINED_DEALER_OVERHANG" else "off"),
                            help="Stock return sensitivity to dealer equity capital ratio growth rate",
                        )
                        iap4.metric(
                            "Expected Intermediary Alpha",
                            f"{iap_rec.expected_intermediary_alpha_pct:+.2f}% Ann",
                            delta=f"Impulse: {iap_rec.intermediary_pricing_impulse:+.2f} | Premium: {iap_rec.intermediary_risk_premium_pct:+.2f}%",
                            help="Projected annualized abnormal return reflecting intermediary capital risk compensation",
                        )

                        st.caption(f"🧾 **Intermediary Pricing Summary:** {iap_rec.summary}")
                        st.caption(f"🛡️ **Intermediary Audit Digest:** `{iap_rec.signature[:16]}...` (Verified: `{iap_rec.verify_integrity()}`)")
                    except Exception as iap_ex:
                        st.info(f"Intermediary Asset Pricing evaluation deferred: {iap_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Intermediary Asset Pricing profile.")

                st.divider()
                st.markdown("#### ⚖️ Gârleanu-Pedersen (2011) Margin-Based Asset Pricing & Basis Spread Engine")
                st.caption(
                    "Evaluates the Margin-CAPM return identity ($E[R_i] - R_f = \\beta_i \\lambda_m + m_i \\psi$), "
                    "cash-synthetic basis spread discount ($D_i$), and Frazzini-Pedersen (2014) Betting Against Beta (BAB) compression."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.margin_asset_pricing import (
                            evaluate_margin_pricing_profile,
                        )

                        with st.expander("⚙️ Customize Margin Requirement & Shadow Cost Parameters (Optional)", expanded=False):
                            map_c1, map_c2, map_c3, map_c4 = st.columns(4)
                            map_mi = map_c1.number_input("Margin Haircut m_i [0.05..0.90]", value=0.0, min_value=0.0, max_value=0.90, step=0.05, key="map_mi")
                            map_psi = map_c2.number_input("Shadow Cost psi [0.0..0.25]", value=0.0, min_value=0.0, max_value=0.25, step=0.01, key="map_psi")
                            map_beta = map_c3.number_input("Market Beta beta_i [-1.0..3.5]", value=0.0, min_value=-1.0, max_value=3.5, step=0.1, key="map_beta")
                            map_mrp = map_c4.number_input("Market Risk Premium [2.0..12.0]", value=0.0, min_value=0.0, max_value=12.0, step=0.5, key="map_mrp")

                        map_rec = evaluate_margin_pricing_profile(
                            ticker=selected_ticker,
                            margin_requirement=map_mi if map_mi > 0.0 else None,
                            margin_shadow_cost_psi=map_psi if map_psi > 0.0 else None,
                            market_beta=map_beta if map_beta != 0.0 else None,
                            market_risk_premium_pct=map_mrp if map_mrp > 0.0 else None,
                        )

                        map1, map2, map3, map4 = st.columns(4)
                        map_badge = (
                            "🔴 HIGH-BETA OVERPRICING TRAP"
                            if map_rec.is_margin_overpricing_trap
                            else (
                                "🟢 MARGIN BASIS DISCOUNT"
                                if map_rec.is_margin_constrained_discount
                                else (
                                    "⚡ BAB ALPHA OPPORTUNITY"
                                    if map_rec.is_bab_alpha_opportunity
                                    else f"Regime: {map_rec.regime}"
                                )
                            )
                        )
                        map1.metric(
                            "Margin Haircut (m_i)",
                            f"{map_rec.margin_requirement:.1%}",
                            delta=f"Base m_0: {map_rec.baseline_margin_requirement:.1%}",
                            delta_color="normal" if map_rec.margin_requirement <= map_rec.baseline_margin_requirement else "inverse",
                            help="Gârleanu & Pedersen (2011) prime brokerage margin / haircut requirement m_i",
                        )
                        map2.metric(
                            "Shadow Cost (ψ)",
                            f"{map_rec.margin_shadow_cost_psi:.1%}",
                            delta=map_badge,
                            delta_color="inverse" if map_rec.is_margin_overpricing_trap else ("normal" if map_rec.is_bab_alpha_opportunity else "off"),
                            help="Lagrange multiplier shadow cost psi of capital margin constraints",
                        )
                        map3.metric(
                            "Margin-CAPM Expected Return",
                            f"{map_rec.margin_capm_expected_return_pct:.2f}% Ann",
                            delta=f"Premium: {map_rec.margin_premium_contribution_pct:+.2f}% vs CAPM",
                            help="Expected return under Margin-CAPM: E[R_i] = R_f + beta_i * lambda_m + m_i * psi",
                        )
                        map4.metric(
                            "Expected Margin Alpha",
                            f"{map_rec.expected_margin_alpha_pct:+.2f}% Ann",
                            delta=f"Basis Spread: {map_rec.basis_spread_discount_pct:+.2f}% | BAB: {map_rec.bab_leverage_compression_score:+.2f}",
                            help="Projected annualized abnormal return reflecting basis spread discount and BAB compression",
                        )

                        st.caption(f"🧾 **Margin Pricing Summary:** {map_rec.summary}")
                        st.caption(f"🛡️ **Margin Audit Digest:** `{map_rec.signature[:16]}...` (Verified: `{map_rec.verify_integrity()}`)")
                    except Exception as map_ex:
                        st.info(f"Margin Asset Pricing evaluation deferred: {map_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Margin Asset Pricing profile.")

                st.divider()
                st.markdown("#### 🏛️ Acharya et al. (2017) Systemic Expected Shortfall (SES) & Capital Surcharge Engine")
                st.caption(
                    "Evaluates Marginal Expected Shortfall ($\\text{MES}_i = -\\mathbb{E}[R_i \\mid R_m \\le -C]$), "
                    "Systemic Expected Shortfall ($\\text{SES}_i$), and Pigouvian systemic externality capital surcharge ($\\tau_i$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.systemic_expected_shortfall import (
                            evaluate_ses_profile,
                        )

                        with st.expander("⚙️ Customize Systemic Tail Drop & Institution Leverage Parameters (Optional)", expanded=False):
                            ses_c1, ses_c2, ses_c3, ses_c4 = st.columns(4)
                            ses_c_in = ses_c1.number_input("Market Drop C [1.0..15.0%]", value=0.0, min_value=0.0, max_value=15.0, step=0.5, key="ses_c_in")
                            ses_mes_in = ses_c2.number_input("Marginal Shortfall MES [-5.0..30.0%]", value=0.0, min_value=-5.0, max_value=30.0, step=0.5, key="ses_mes_in")
                            ses_lev_in = ses_c3.number_input("Leverage Ratio [1.0..45.0x]", value=0.0, min_value=0.0, max_value=45.0, step=1.0, key="ses_lev_in")
                            ses_lev0_in = ses_c4.number_input("Base Leverage [1.0..25.0x]", value=0.0, min_value=0.0, max_value=25.0, step=1.0, key="ses_lev0_in")

                        ses_rec = evaluate_ses_profile(
                            ticker=selected_ticker,
                            market_tail_threshold_pct=ses_c_in if ses_c_in > 0.0 else None,
                            marginal_expected_shortfall_pct=ses_mes_in if ses_mes_in != 0.0 else None,
                            institution_leverage=ses_lev_in if ses_lev_in > 0.0 else None,
                            baseline_leverage=ses_lev0_in if ses_lev0_in > 0.0 else None,
                        )

                        ses1, ses2, ses3, ses4 = st.columns(4)
                        ses_badge = (
                            "🔴 CRITICAL SYSTEMIC THREAT"
                            if ses_rec.is_critical_systemic_threat
                            else (
                                "🟢 DEFENSIVE RESILIENCE"
                                if ses_rec.is_defensive_systemic_resilience
                                else (
                                    "⚡ ELEVATED TAIL SENSITIVITY"
                                    if ses_rec.is_elevated_tail_sensitivity
                                    else f"Regime: {ses_rec.regime}"
                                )
                            )
                        )
                        ses1.metric(
                            "Marginal Shortfall (MES_i)",
                            f"{ses_rec.marginal_expected_shortfall_pct:.2f}%",
                            delta=f"Tail Ratio: {ses_rec.tail_dependence_ratio:.2f}x (vs C={ses_rec.market_tail_threshold_pct:.1f}%)",
                            delta_color="inverse" if ses_rec.marginal_expected_shortfall_pct >= 4.0 else "normal",
                            help="Expected loss conditional on market tail crash R_m <= -C",
                        )
                        ses2.metric(
                            "Systemic Shortfall (SES_i)",
                            f"{ses_rec.systemic_expected_shortfall_pct:.2f}%",
                            delta=ses_badge,
                            delta_color="inverse" if ses_rec.is_critical_systemic_threat else ("normal" if ses_rec.is_defensive_systemic_resilience else "off"),
                            help="Expected contribution to aggregate financial system undercapitalization crisis",
                        )
                        ses3.metric(
                            "Capital Surcharge (τ_i)",
                            f"{ses_rec.systemic_capital_surcharge_pct:.2f}%",
                            delta=f"Lev: {ses_rec.institution_leverage:.1f}x (Base: {ses_rec.baseline_leverage:.1f}x)",
                            delta_color="inverse" if ses_rec.systemic_capital_surcharge_pct > 2.0 else "normal",
                            help="Pigouvian systemic externality capital buffer requirement tau_i",
                        )
                        ses4.metric(
                            "Expected Systemic Alpha",
                            f"{ses_rec.expected_systemic_alpha_pct:+.2f}% Ann",
                            delta=f"Fragility Impulse: {ses_rec.systemic_fragility_impulse:+.2f}",
                            help="Projected annualized abnormal return reflecting systemic tail risk drag or defensive resilience",
                        )

                        st.caption(f"🧾 **Systemic Risk Summary:** {ses_rec.summary}")
                        st.caption(f"🛡️ **Systemic Audit Digest:** `{ses_rec.signature[:16]}...` (Verified: `{ses_rec.verify_integrity()}`)")
                    except Exception as ses_ex:
                        st.info(f"Systemic Expected Shortfall evaluation deferred: {ses_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Systemic Expected Shortfall profile.")

                st.divider()
                st.markdown("#### 🌪️ Adrian & Brunnermeier (2016) CoVaR, Delta-CoVaR & Systemic Spillover Engine")
                st.caption(
                    "Evaluates conditional systemic Value-at-Risk ($\\text{CoVaR}_{m \\mid i}$), "
                    "marginal contagion spillover ($\\Delta\\text{CoVaR}_i$), and systemic exposure ($\\Delta\\text{CoVaR}_{i \\mid m}$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.covar_systemic_spillover import (
                            evaluate_covar_profile,
                        )

                        with st.expander("⚙️ Customize CoVaR Tail Quantile & Regression Beta Parameters (Optional)", expanded=False):
                            cov_c1, cov_c2, cov_c3, cov_c4 = st.columns(4)
                            cov_q_in = cov_c1.number_input("Tail Quantile q [0.01..0.15]", value=0.0, min_value=0.0, max_value=0.15, step=0.01, key="cov_q_in")
                            cov_var_in = cov_c2.number_input("Stock VaR_i [1.0..30.0%]", value=0.0, min_value=0.0, max_value=30.0, step=0.5, key="cov_var_in")
                            cov_varm_in = cov_c3.number_input("System VaR_m [1.0..20.0%]", value=0.0, min_value=0.0, max_value=20.0, step=0.5, key="cov_varm_in")
                            cov_beta_in = cov_c4.number_input("Tail Beta [0.0..3.5]", value=0.0, min_value=0.0, max_value=3.5, step=0.1, key="cov_beta_in")

                        cov_rec = evaluate_covar_profile(
                            ticker=selected_ticker,
                            tail_quantile=cov_q_in if cov_q_in > 0.0 else None,
                            standalone_var_pct=cov_var_in if cov_var_in > 0.0 else None,
                            system_unconditional_var_pct=cov_varm_in if cov_varm_in > 0.0 else None,
                            tail_beta=cov_beta_in if cov_beta_in > 0.0 else None,
                        )

                        cov1, cov2, cov3, cov4 = st.columns(4)
                        cov_badge = (
                            "🔴 HIGH SPILLOVER THREAT"
                            if cov_rec.is_critical_spillover_threat
                            else (
                                "🟢 DISTRESS ABSORBER"
                                if cov_rec.is_systemic_distress_absorber
                                else (
                                    "⚡ ELEVATED SPILLOVER"
                                    if cov_rec.is_elevated_spillover
                                    else f"Regime: {cov_rec.regime}"
                                )
                            )
                        )
                        cov1.metric(
                            "System CoVaR (CoVaR_{m|i})",
                            f"{cov_rec.conditional_system_covar_pct:.2f}%",
                            delta=f"Uncond VaR_m: {cov_rec.system_unconditional_var_pct:.2f}%",
                            delta_color="inverse" if cov_rec.conditional_system_covar_pct > cov_rec.system_unconditional_var_pct * 1.5 else "normal",
                            help="Financial system Value-at-Risk conditional on target asset distress",
                        )
                        cov2.metric(
                            "Marginal Contagion (ΔCoVaR)",
                            f"{cov_rec.delta_covar_pct:.2f}%",
                            delta=cov_badge,
                            delta_color="inverse" if cov_rec.is_critical_spillover_threat else ("normal" if cov_rec.is_systemic_distress_absorber else "off"),
                            help="Difference in financial system VaR between asset distress vs median state",
                        )
                        cov3.metric(
                            "Systemic Exposure (ΔCoVaR_{i|m})",
                            f"{cov_rec.systemic_exposure_covar_pct:.2f}%",
                            delta=f"Spillover: {cov_rec.systemic_spillover_multiplier:.2f}x",
                            delta_color="inverse" if cov_rec.systemic_spillover_multiplier > 1.3 else "normal",
                            help="Asset VaR shift when overall financial system is in distress",
                        )
                        cov4.metric(
                            "Expected CoVaR Alpha",
                            f"{cov_rec.expected_covar_alpha_pct:+.2f}% Ann",
                            delta=f"Fragility Impulse: {cov_rec.covar_fragility_impulse:+.2f}",
                            help="Projected annualized abnormal return reflecting contagion spillover drag or absorber resilience",
                        )

                        st.caption(f"🧾 **CoVaR Summary:** {cov_rec.summary}")
                        st.caption(f"🛡️ **CoVaR Audit Digest:** `{cov_rec.signature[:16]}...` (Verified: `{cov_rec.verify_integrity()}`)")
                    except Exception as cov_ex:
                        st.info(f"CoVaR Spillover evaluation deferred: {cov_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate CoVaR Spillover profile.")

                st.divider()
                st.markdown("#### 🧬 Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis & Evolutionary Alpha Engine")
                st.caption(
                    "Evaluates time-varying Adaptive Market Efficiency ($\\mathcal{E}_{\\text{AMH}}$), "
                    "Lo-MacKinlay Variance Ratio ($VR_k$), Arbitrage Carrying Capacity ($\\mathcal{K}_{\\text{arb}}$), "
                    "and behavioral punctuated equilibrium under Andrew W. Lo (2004, 2005, 2017)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.adaptive_markets import (
                            evaluate_amh_profile,
                        )

                        with st.expander("⚙️ Customize Adaptive Efficiency & Evolutionary Stress Parameters (Optional)", expanded=False):
                            amh_c1, amh_c2, amh_c3, amh_c4 = st.columns(4)
                            amh_vr_in = amh_c1.number_input("Variance Ratio VR(k) [0.30..2.50]", value=0.0, min_value=0.0, max_value=2.50, step=0.05, key="amh_vr_in")
                            amh_rho_in = amh_c2.number_input("Autocorrelation rho_1 [-0.80..0.80]", value=0.0, min_value=-0.80, max_value=0.80, step=0.02, key="amh_rho_in")
                            amh_dd_in = amh_c3.number_input("Peak Drawdown [0.0..0.85]", value=0.0, min_value=0.0, max_value=0.85, step=0.05, key="amh_dd_in")
                            amh_vol_in = amh_c4.number_input("Annualized Volatility [0.05..1.50]", value=0.0, min_value=0.0, max_value=1.50, step=0.02, key="amh_vol_in")

                        amh_rec = evaluate_amh_profile(
                            ticker=selected_ticker,
                            lo_mackinlay_variance_ratio=amh_vr_in if amh_vr_in > 0.0 else None,
                            serial_autocorrelation_rho1=amh_rho_in if amh_rho_in != 0.0 else None,
                            ecological_stress_drawdown=amh_dd_in if amh_dd_in > 0.0 else None,
                            realized_volatility=amh_vol_in if amh_vol_in > 0.0 else None,
                        )

                        amh1, amh2, amh3, amh4 = st.columns(4)
                        amh_badge = (
                            "🔴 BEHAVIORAL DISLOCATION"
                            if amh_rec.is_punctuated_equilibrium
                            else (
                                "🟢 COMPETITIVE SATURATION"
                                if amh_rec.is_competitive_saturation
                                else (
                                    "⚡ ADAPTIVE RECOVERY"
                                    if amh_rec.is_adaptive_recovery
                                    else f"Regime: {amh_rec.regime}"
                                )
                            )
                        )
                        amh1.metric(
                            "Adaptive Efficiency (E_AMH)",
                            f"{amh_rec.adaptive_market_efficiency_index:.2f}",
                            delta=f"VR: {amh_rec.lo_mackinlay_variance_ratio:.2f} (Base=1.00)",
                            delta_color="normal" if amh_rec.adaptive_market_efficiency_index >= 0.70 else "inverse",
                            help="Andrew W. Lo (2004) time-varying market efficiency index: E_AMH = 1 - 0.70|VR - 1| - 0.30|rho_1|",
                        )
                        amh2.metric(
                            "Carrying Capacity (K_arb)",
                            f"{amh_rec.arbitrage_carrying_capacity:.2f}x",
                            delta=amh_badge,
                            delta_color="normal" if amh_rec.is_adaptive_recovery else ("inverse" if amh_rec.is_punctuated_equilibrium else "off"),
                            help="Ecology arbitrage carrying capacity: higher values denote fertile alpha opportunities before saturation",
                        )
                        amh3.metric(
                            "Adaptation Pressure (Π_evol)",
                            f"{amh_rec.evolutionary_adaptation_pressure:+.2f}",
                            delta=f"Autocorr: {amh_rec.serial_autocorrelation_rho1:+.2f}",
                            delta_color="inverse" if amh_rec.evolutionary_adaptation_pressure > 1.0 else "normal",
                            help="Evolutionary natural selection pressure on quantitative strategies from environmental stress and inefficiency",
                        )
                        amh4.metric(
                            "Expected Adaptive Alpha",
                            f"{amh_rec.expected_adaptive_alpha_pct:+.2f}% Ann",
                            delta=f"Arbitrage Impulse: {amh_rec.amh_arbitrage_impulse:+.2f}",
                            help="Projected annualized abnormal alpha potential net of participant crowding and market ecology saturation",
                        )

                        st.caption(f"🧾 **Adaptive Markets Summary:** {amh_rec.summary}")
                        st.caption(f"🛡️ **Adaptive Audit Digest:** `{amh_rec.signature[:16]}...` (Verified: `{amh_rec.verify_integrity()}`)")
                    except Exception as amh_ex:
                        st.info(f"Adaptive Markets Hypothesis evaluation deferred: {amh_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Adaptive Markets profile.")

                st.divider()
                st.markdown("#### 🤝 Duffie-Gârleanu-Pedersen (2005) OTC Search-and-Bargaining Illiquidity Engine")
                st.caption(
                    "Evaluates counterparty search arrival intensity ($\\lambda$), investor Nash bargaining power ($q$), "
                    "endogenous bid-ask spread ($\\Delta P$), and the Search Illiquidity Discount ($S_{\\text{DGP}}$) relative to frictionless Walrasian benchmarks."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.search_bargaining_illiquidity import (
                            evaluate_search_bargaining_profile,
                        )

                        with st.expander("⚙️ Customize Counterparty Search & Nash Bargaining Parameters (Optional)", expanded=False):
                            dgp_c1, dgp_c2, dgp_c3, dgp_c4 = st.columns(4)
                            dgp_lam_in = dgp_c1.number_input("Search Intensity lambda [0.1..50.0]", value=0.0, min_value=0.0, max_value=50.0, step=1.0, key="dgp_lam_in")
                            dgp_q_in = dgp_c2.number_input("Bargaining Power q [0.05..0.95]", value=0.0, min_value=0.0, max_value=0.95, step=0.05, key="dgp_q_in")
                            dgp_chi_in = dgp_c3.number_input("Dealer Inventory Cost chi [0.0..0.25]", value=0.0, min_value=0.0, max_value=0.25, step=0.01, key="dgp_chi_in")
                            dgp_d_in = dgp_c4.number_input("Asset Cash Flow Rate D [0.005..0.20]", value=0.0, min_value=0.0, max_value=0.20, step=0.01, key="dgp_d_in")

                        dgp_rec = evaluate_search_bargaining_profile(
                            ticker=selected_ticker,
                            search_intensity_lambda=dgp_lam_in if dgp_lam_in > 0.0 else None,
                            investor_bargaining_power=dgp_q_in if dgp_q_in > 0.0 else None,
                            dealer_inventory_cost=dgp_chi_in if dgp_chi_in > 0.0 else None,
                            asset_cash_flow_rate=dgp_d_in if dgp_d_in > 0.0 else None,
                        )

                        dgp1, dgp2, dgp3, dgp4 = st.columns(4)
                        dgp_badge = (
                            "🔴 SEVERE SEARCH FREEZE"
                            if dgp_rec.is_severe_search_freeze
                            else (
                                "🟢 CENTRALIZED MATCHING"
                                if dgp_rec.is_centralized_high_liquidity
                                else (
                                    "⚡ ELEVATED FRICTION"
                                    if dgp_rec.is_elevated_search_friction
                                    else f"Regime: {dgp_rec.regime}"
                                )
                            )
                        )
                        dgp1.metric(
                            "Search Intensity (λ)",
                            f"{dgp_rec.search_intensity_lambda:.1f} / yr",
                            delta=f"Power q: {dgp_rec.investor_bargaining_power:.0%}",
                            delta_color="normal" if dgp_rec.search_intensity_lambda >= 15.0 else "inverse",
                            help="Duffie-Gârleanu-Pedersen (2005) Poisson contact arrival rate lambda for counterparty matching",
                        )
                        dgp2.metric(
                            "Search Discount (S_DGP)",
                            f"{dgp_rec.search_illiquidity_discount_pct:.2f}%",
                            delta=dgp_badge,
                            delta_color="inverse" if dgp_rec.is_severe_search_freeze else ("normal" if dgp_rec.is_centralized_high_liquidity else "off"),
                            help="Percentage price markdown relative to frictionless Walrasian benchmark price: S_DGP = ((r + lambda*(1-q))/(r+lambda)) * (chi/D)",
                        )
                        dgp3.metric(
                            "Endogenous Spread (ΔP)",
                            f"{dgp_rec.endogenous_bid_ask_spread_pct:.2f}%",
                            delta=f"Holding Cost χ: {dgp_rec.dealer_inventory_cost:.1%}",
                            delta_color="inverse" if dgp_rec.endogenous_bid_ask_spread_pct > 1.2 else "normal",
                            help="Equilibrium bid-ask spread emerging from sequential search delays and dealer inventory holding costs",
                        )
                        dgp4.metric(
                            "Expected Search Alpha",
                            f"{dgp_rec.expected_search_alpha_pct:+.2f}% Ann",
                            delta=f"Walrasian: ${dgp_rec.walrasian_frictionless_price:.2f}",
                            help="Projected annualized abnormal return reflecting illiquidity recovery and dealer search premia",
                        )

                        st.caption(f"🧾 **Search Illiquidity Summary:** {dgp_rec.summary}")
                        st.caption(f"🛡️ **Search Audit Digest:** `{dgp_rec.signature[:16]}...` (Verified: `{dgp_rec.verify_integrity()}`)")
                    except Exception as dgp_ex:
                        st.info(f"Duffie-Gârleanu-Pedersen Search Illiquidity evaluation deferred: {dgp_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Search-and-Bargaining profile.")

                st.divider()
                st.markdown("#### 📉 Campbell-Hilscher-Szilagyi (2008) Financial Distress Puzzle & Failure Engine")
                st.caption(
                    "Evaluates dynamic multi-period failure probability ($P_t(\\text{Failure})$), "
                    "CHS logit distress score, and the empirical Distress Anomaly alpha drag under Campbell, Hilscher & Szilagyi (2008)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.campbell_distress_puzzle import (
                            evaluate_campbell_distress_profile,
                        )

                        with st.expander("⚙️ Customize CHS Accounting Ratios & Distress Parameters (Optional)", expanded=False):
                            chs_c1, chs_c2, chs_c3, chs_c4 = st.columns(4)
                            chs_nimta_in = chs_c1.number_input("NIMTA (Net Income / MTA) [-1.0..0.50]", value=0.0, min_value=-1.0, max_value=0.50, step=0.02, key="chs_nimta_in")
                            chs_tlmta_in = chs_c2.number_input("TLMTA (Total Liab / MTA) [0.0..0.98]", value=0.0, min_value=0.0, max_value=0.98, step=0.05, key="chs_tlmta_in")
                            chs_cash_in = chs_c3.number_input("CASHMTA (Cash / MTA) [0.0..0.80]", value=0.0, min_value=0.0, max_value=0.80, step=0.05, key="chs_cash_in")
                            chs_sigma_in = chs_c4.number_input("Idiosyncratic Vol SIGMA [0.05..1.50]", value=0.0, min_value=0.0, max_value=1.50, step=0.02, key="chs_sigma_in")

                        chs_rec = evaluate_campbell_distress_profile(
                            ticker=selected_ticker,
                            net_income_to_market_assets=chs_nimta_in if chs_nimta_in != 0.0 else None,
                            total_liabilities_to_market_assets=chs_tlmta_in if chs_tlmta_in > 0.0 else None,
                            cash_to_market_assets=chs_cash_in if chs_cash_in > 0.0 else None,
                            idiosyncratic_volatility_sigma=chs_sigma_in if chs_sigma_in > 0.0 else None,
                        )

                        chs1, chs2, chs3, chs4 = st.columns(4)
                        chs_badge = (
                            "🔴 OVERPRICED DISTRESS TRAP"
                            if chs_rec.is_distress_anomaly_trap
                            else (
                                "🟢 FINANCIAL FORTRESS"
                                if chs_rec.is_solvent_financial_fortress
                                else (
                                    "⚡ ELEVATED DISTRESS WATCH"
                                    if chs_rec.is_elevated_distress_watch
                                    else f"Regime: {chs_rec.regime}"
                                )
                            )
                        )
                        chs1.metric(
                            "Failure Probability (12M)",
                            f"{chs_rec.failure_probability_12m_pct:.3f}%",
                            delta=f"CHS Score: {chs_rec.chs_logit_score:.2f}",
                            delta_color="inverse" if chs_rec.failure_probability_12m_pct > 1.0 else "normal",
                            help="Campbell-Hilscher-Szilagyi (2008) dynamic multi-period logit probability of corporate failure/bankruptcy over next 12 months",
                        )
                        chs2.metric(
                            "Distress Anomaly Drag",
                            f"{chs_rec.distress_anomaly_alpha_drag_pct:+.2f}% Ann",
                            delta=chs_badge,
                            delta_color="inverse" if chs_rec.is_distress_anomaly_trap else ("normal" if chs_rec.is_solvent_financial_fortress else "off"),
                            help="Expected negative alpha drag from the financial distress puzzle anomaly (distressed lottery speculation)",
                        )
                        chs3.metric(
                            "Liabilities (TLMTA)",
                            f"{chs_rec.total_liabilities_to_market_assets:.1%}",
                            delta=f"Cash: {chs_rec.cash_to_market_assets:.1%}",
                            delta_color="inverse" if chs_rec.total_liabilities_to_market_assets > 0.60 else "normal",
                            help="Total liabilities divided by market total assets (MTA = Total Liabilities + Market Cap)",
                        )
                        chs4.metric(
                            "Expected Distress Alpha",
                            f"{chs_rec.expected_distress_alpha_pct:+.2f}% Ann",
                            delta=f"Fragility Impulse: {chs_rec.chs_fragility_impulse:+.2f}",
                            help="Projected annualized abnormal alpha net of failure probability drag and liquidity buffer",
                        )

                        st.caption(f"🧾 **Distress Puzzle Summary:** {chs_rec.summary}")
                        st.caption(f"🛡️ **Distress Audit Digest:** `{chs_rec.signature[:16]}...` (Verified: `{chs_rec.verify_integrity()}`)")
                    except Exception as chs_ex:
                        st.info(f"Campbell Distress Puzzle evaluation deferred: {chs_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Campbell Distress profile.")

                st.divider()
                st.markdown("#### ⚖️ Grossman-Stiglitz (1980) Informational Equilibrium & Rent Engine")
                st.caption(
                    "Evaluates the Grossman-Stiglitz (1980) paradox, equilibrium informed trader fraction ($\\lambda^*$), "
                    "price informativeness index ($\\mathcal{I}_{\\text{GS}}$), and net informational alpha available to quantitative intelligence."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.grossman_stiglitz_equilibrium import (
                            evaluate_grossman_stiglitz_profile,
                            verify_grossman_audit_digest,
                        )

                        with st.expander("⚙️ Customize Grossman-Stiglitz Parameters (Optional)", expanded=False):
                            gs_c1, gs_c2 = st.columns(2)
                            gs_cost_bps = gs_c1.slider("Information Analysis Cost c (bps)", min_value=1.0, max_value=100.0, value=20.0, step=1.0, key="gs_cost_bps")
                            gs_gamma = gs_c2.slider("Investor CARA Risk Aversion (gamma)", min_value=0.5, max_value=10.0, value=2.5, step=0.25, key="gs_gamma")

                        gs_rec = evaluate_grossman_stiglitz_profile(
                            ticker=selected_ticker,
                            risk_aversion=float(gs_gamma),
                            information_cost=float(gs_cost_bps / 10000.0),
                        )

                        gs1, gs2, gs3, gs4 = st.columns(4)
                        gs_badge = (
                            "🟢 SATURATED EFFICIENCY"
                            if gs_rec.is_efficient_saturated
                            else (
                                "⚡ NOISY OPPORTUNITY"
                                if gs_rec.is_noisy_opportunity
                                else (
                                    "👥 UNINFORMED HERD"
                                    if gs_rec.is_uninformed_herd
                                    else f"{gs_rec.informational_regime}"
                                )
                            )
                        )

                        gs1.metric(
                            "Price Informativeness (I_GS)",
                            f"{gs_rec.price_informativeness:.1%}",
                            delta=gs_badge,
                            delta_color="normal" if gs_rec.is_efficient_saturated else ("off" if gs_rec.is_noisy_opportunity else "inverse"),
                            help="Grossman-Stiglitz equilibrium price informativeness index measuring the fraction of private signal precision revealed in market prices",
                        )
                        gs2.metric(
                            "Informed Fraction (Lambda*)",
                            f"{gs_rec.equilibrium_informed_fraction:.1%}",
                            delta="Competitive Equilibrium",
                            help="Endogenous equilibrium fraction of active informed market participants balancing analytical discovery costs against gross informational value",
                        )
                        gs3.metric(
                            "Net Informational Alpha",
                            f"{gs_rec.net_informational_alpha * 10000.0:+.1f} bps/d",
                            delta=f"Cost: {gs_rec.information_cost * 10000.0:.1f} bps",
                            delta_color="normal" if gs_rec.net_informational_alpha >= 0 else "inverse",
                            help="Expected daily informational rent per session after deducting analytical cost c",
                        )
                        gs4.metric(
                            "Expected Annual Boost",
                            f"{gs_rec.expected_alpha_boost_bps:+.1f} bps",
                            delta=f"Noise/Signal: {gs_rec.noise_to_signal_ratio:.2f}",
                            help="Annualized portfolio alpha boost from disciplined quantitative research in partially noisy market equilibrium",
                        )

                        st.caption(f"🧾 **Informational Equilibrium Summary:** {gs_rec.summary}")
                        st.caption(f"🛡️ **Grossman Audit Digest:** `{gs_rec.audit_digest[:16]}...` (Verified: `{verify_grossman_audit_digest(gs_rec)}`)")
                    except Exception as gs_ex:
                        st.info(f"Grossman-Stiglitz Informational Equilibrium evaluation deferred: {gs_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Grossman-Stiglitz profile.")

                st.divider()
                st.markdown("#### 🏛️ Easley-Hvidkjaer-O'Hara (2002, 2004) Information Risk & Cost of Capital Engine")
                st.caption(
                    "Evaluates the multi-factor asset pricing cost of capital under Easley, Hvidkjaer & O'Hara (2002, 2004), "
                    "measuring required expected return surcharges ($\\Delta r_{\\text{info}}$) from systematic, undiversifiable private information risk ($\\text{PIN}$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.information_risk_capital import (
                            evaluate_information_risk_profile,
                            verify_information_risk_audit_digest,
                        )

                        with st.expander("⚙️ Customize Information Risk & Hurdle Parameters (Optional)", expanded=False):
                            eho_c1, eho_c2, eho_c3 = st.columns(3)
                            eho_pin_override = eho_c1.slider("Override PIN (%) [0 = Auto]", min_value=0.0, max_value=60.0, value=0.0, step=1.0, key="eho_pin_override")
                            eho_erp_pct = eho_c2.slider("Equity Risk Premium ERP (%)", min_value=3.0, max_value=9.0, value=5.5, step=0.25, key="eho_erp_pct")
                            eho_lambda_pct = eho_c3.slider("Info Factor Premium lambda_INFO (%)", min_value=10.0, max_value=50.0, value=25.0, step=2.5, key="eho_lambda_pct")

                        eho_rec = evaluate_information_risk_profile(
                            ticker=selected_ticker,
                            pin_override=float(eho_pin_override / 100.0) if eho_pin_override > 0.0 else None,
                            equity_risk_premium=float(eho_erp_pct / 100.0),
                            lambda_info=float(eho_lambda_pct / 100.0),
                        )

                        eho1, eho2, eho3, eho4 = st.columns(4)
                        eho_badge = (
                            "🔴 SEVERE ADVERSE SELECTION"
                            if eho_rec.is_severe_surcharge
                            else (
                                "⚡ ELEVATED INFO RISK"
                                if eho_rec.is_elevated_risk
                                else (
                                    "🟢 BENIGN TRANSPARENT"
                                    if eho_rec.is_benign_transparent
                                    else "💎 DISCLOSURE DISCOUNT"
                                )
                            )
                        )

                        eho1.metric(
                            "Info Risk Surcharge",
                            f"{eho_rec.cost_of_capital_surcharge_pct:+.2f}% Ann",
                            delta=f"{eho_rec.cost_of_capital_surcharge_bps:+.0f} bps",
                            delta_color="inverse" if eho_rec.cost_of_capital_surcharge_pct > 1.0 else "normal",
                            help="Easley-Hvidkjaer-O'Hara (2002) annual cost of capital hurdle premium demanded by uninformed investors for bearing asymmetric information risk",
                        )
                        eho2.metric(
                            "Informed Trading (PIN)",
                            f"{eho_rec.pin:.1%}",
                            delta=f"Benchmark: {eho_rec.baseline_market_pin:.1%}",
                            delta_color="inverse" if eho_rec.pin > eho_rec.baseline_market_pin else "normal",
                            help="Probability of informed trading (PIN) measuring asymmetric order flow relative to the US market average (~19.1%)",
                        )
                        eho3.metric(
                            "Information Beta (Beta_INFO)",
                            f"{eho_rec.info_risk_beta:+.2f}",
                            delta=eho_badge,
                            delta_color="inverse" if eho_rec.info_risk_beta > 1.0 else ("normal" if eho_rec.info_risk_beta <= 0 else "off"),
                            help="Standardized information risk beta measuring cross-sectional factor loading on asymmetric information",
                        )
                        eho4.metric(
                            "Total Required Hurdle Return",
                            f"{eho_rec.total_required_hurdle_pct:.2f}% Ann",
                            delta=f"CAPM: {eho_rec.capm_required_return_pct:.2f}%",
                            help="Multi-factor cost of capital hurdle return combining CAPM systematic risk and EHO information risk surcharge",
                        )

                        st.caption(f"🧾 **Information Risk Summary:** {eho_rec.summary}")
                        st.caption(f"🛡️ **EHO Audit Digest:** `{eho_rec.audit_digest[:16]}...` (Verified: `{verify_information_risk_audit_digest(eho_rec)}`)")
                    except Exception as eho_ex:
                        st.info(f"Information Risk & Cost of Capital evaluation deferred: {eho_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Information Risk profile.")

                st.divider()
                st.markdown("#### ⏳ Admati-Pfleiderer (1988) Volume Clustering & Execution Timing Engine")
                st.caption(
                    "Evaluates intraday concentrated trading equilibria under Admati & Pfleiderer (1988), "
                    "measuring liquidity coordination, clumping index ($\\mathcal{C}_{\\text{AP}}$), effective market depth multiplier ($\\mathcal{D}_{\\text{AP}}$), and optimal execution windows."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.admati_pfleiderer_clustering import (
                            evaluate_admati_pfleiderer_profile,
                            verify_admati_audit_digest,
                        )

                        with st.expander("⚙️ Customize Admati-Pfleiderer Parameters (Optional)", expanded=False):
                            ap_c1, ap_c2 = st.columns(2)
                            ap_disc_ratio = ap_c1.slider("Discretionary Liquidity Ratio (%) [0 = Auto]", min_value=0.0, max_value=95.0, value=0.0, step=5.0, key="ap_disc_ratio")
                            ap_k_informed = ap_c2.slider("Informed Traders Count (k)", min_value=1, max_value=10, value=3, step=1, key="ap_k_informed")

                        ap_rec = evaluate_admati_pfleiderer_profile(
                            ticker=selected_ticker,
                            discretionary_ratio_override=float(ap_disc_ratio / 100.0) if ap_disc_ratio > 0.0 else None,
                        )

                        ap1, ap2, ap3, ap4 = st.columns(4)
                        ap_badge = (
                            "🟢 CONCENTRATED CLUSTER"
                            if ap_rec.is_concentrated_cluster
                            else (
                                "⚡ BALANCED EQUILIBRIUM"
                                if ap_rec.is_balanced_equilibrium
                                else (
                                    "🔴 MIDDAY DROUGHT"
                                    if ap_rec.is_midday_drought
                                    else f"{ap_rec.regime}"
                                )
                            )
                        )

                        ap1.metric(
                            "Market Depth Multiplier",
                            f"{ap_rec.market_depth_multiplier:.2f}x",
                            delta=ap_badge,
                            delta_color="normal" if ap_rec.is_concentrated_cluster else ("off" if ap_rec.is_balanced_equilibrium else "inverse"),
                            help="Admati-Pfleiderer equilibrium market depth multiplier relative to dispersed trading; higher depth substantially compresses execution slippage",
                        )
                        ap2.metric(
                            "Clumping Index (C_AP)",
                            f"{ap_rec.clumping_index:.1%}",
                            delta=f"Discretionary: {ap_rec.discretionary_liquidity_ratio:.1%}",
                            help="Ratio of discretionary liquidity traders coordinating execution timing in peak market depth windows",
                        )
                        ap3.metric(
                            "Price Impact Reduction",
                            f"{ap_rec.kyle_lambda_reduction_pct:.1f}%",
                            delta=f"-{ap_rec.execution_slippage_discount_bps:.1f} bps slippage",
                            delta_color="normal" if ap_rec.kyle_lambda_reduction_pct >= 20.0 else "off",
                            help="Expected reduction in Kyle lambda price impact parameter and execution slippage during concentrated liquidity clusters",
                        )
                        ap4.metric(
                            "Execution Timing",
                            f"{ap_rec.recommended_execution_window}",
                            delta="Optimal Window",
                            help="Recommended algorithmic execution timing window based on intraday depth equilibrium",
                        )

                        st.caption(f"🧾 **Admati-Pfleiderer Summary:** {ap_rec.summary}")
                        st.caption(f"🛡️ **Admati Audit Digest:** `{ap_rec.audit_digest[:16]}...` (Verified: `{verify_admati_audit_digest(ap_rec)}`)")
                    except Exception as ap_ex:
                        st.info(f"Admati-Pfleiderer Volume Clustering evaluation deferred: {ap_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Admati-Pfleiderer profile.")

                st.divider()
                st.markdown("#### 🎯 Black (1986) Noise Valuation Bounds & Mean-Reversion Engine")
                st.caption(
                    "Evaluates the Black (1986) factor-of-two valuation envelope ($[0.50 P^*, 2.00 P^*]$), "
                    "noise-to-information ratio, and expected annual fundamental mean-reversion alpha."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.noise_valuation_bands import (
                            evaluate_noise_valuation_profile,
                            verify_noise_valuation_audit_digest,
                        )

                        with st.expander("⚙️ Customize Valuation Bounds & Convergence Speed (Optional)", expanded=False):
                            nb_c1, nb_c2 = st.columns(2)
                            nb_fund_override = nb_c1.number_input("Override Intrinsic Value P* ($) [0 = Auto]", min_value=0.0, max_value=10000.0, value=0.0, step=5.0, key="nb_fund_override")
                            nb_kappa_pct = nb_c2.slider("Annual Mean-Reversion Speed kappa (%)", min_value=10.0, max_value=80.0, value=35.0, step=5.0, key="nb_kappa_pct")

                        nb_rec = evaluate_noise_valuation_profile(
                            ticker=selected_ticker,
                            fundamental_value_override=float(nb_fund_override) if nb_fund_override > 0.0 else None,
                        )

                        nb1, nb2, nb3, nb4 = st.columns(4)
                        nb_badge = (
                            "🔴 EXTREME DISLOCATION"
                            if nb_rec.is_extreme_dislocation
                            else (
                                "⚠️ NOISE OVERVALUATION"
                                if nb_rec.is_overvalued
                                else (
                                    "💎 DEEP VALUE DISCOUNT"
                                    if nb_rec.is_undervalued
                                    else "🟢 EQUILIBRIUM CORRIDOR"
                                )
                            )
                        )

                        nb1.metric(
                            "Factor-of-Two Bounds",
                            f"${nb_rec.lower_factor2_bound:.1f} - ${nb_rec.upper_factor2_bound:.1f}",
                            delta="Inside Envelope" if nb_rec.is_within_factor2_bounds else "Outside Envelope",
                            delta_color="normal" if nb_rec.is_within_factor2_bounds else "inverse",
                            help="Black (1986) factor-of-two efficiency boundaries: [0.50 * P*, 2.00 * P*]",
                        )
                        nb2.metric(
                            "Pricing Deviation from P*",
                            f"{nb_rec.noise_deviation_pct:+.1f}%",
                            delta=nb_badge,
                            delta_color="inverse" if abs(nb_rec.noise_deviation_pct) > 25.0 else "normal",
                            help="Percentage deviation of latest market price from fundamental intrinsic value P*",
                        )
                        nb3.metric(
                            "Mean-Reversion Alpha",
                            f"{nb_rec.expected_reversion_alpha_pct:+.1f}% Ann",
                            delta=f"P* = ${nb_rec.fundamental_value:.2f}",
                            delta_color="normal" if nb_rec.expected_reversion_alpha_pct > 0 else ("inverse" if nb_rec.expected_reversion_alpha_pct < -5.0 else "off"),
                            help="Projected annualized alpha from fundamental mean-reversion toward intrinsic value",
                        )
                        nb4.metric(
                            "Noise-to-Info Ratio",
                            f"{nb_rec.noise_to_information_ratio:.2f}",
                            delta=f"Info Share: {nb_rec.information_share_pct:.1f}%",
                            help="Ratio of return variance driven by noise trading vs fundamental information",
                        )

                        st.caption(f"🧾 **Valuation Bounds Summary:** {nb_rec.summary}")
                        st.caption(f"🛡️ **Noise Audit Digest:** `{nb_rec.audit_digest[:16]}...` (Verified: `{verify_noise_valuation_audit_digest(nb_rec)}`)")
                    except Exception as nb_ex:
                        st.info(f"Noise Valuation Bounds evaluation deferred: {nb_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Noise Valuation profile.")

                st.divider()
                st.markdown("#### ⚖️ Campbell-Vuolteenaho (2004) Two-Beta ('Bad Beta, Good Beta') Engine")
                st.caption(
                    "Decomposes systematic market risk into permanent Cash-Flow Beta ($\\beta_{\\text{CF}}$, 'Bad Beta') "
                    "and temporary Discount-Rate Beta ($\\beta_{\\text{DR}}$, 'Good Beta'), calculating ICAPM hurdle rates and bad-beta risk surcharges."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.bad_beta_good_beta import (
                            evaluate_two_beta_profile,
                            verify_two_beta_audit_digest,
                        )

                        with st.expander("⚙️ Customize Two-Beta Parameters & Risk Aversion (Optional)", expanded=False):
                            tb_c1, tb_c2, tb_c3 = st.columns(3)
                            tb_cf_override = tb_c1.number_input("Override Bad Beta (beta_CF) [0 = Auto]", min_value=0.0, max_value=4.0, value=0.0, step=0.05, key="tb_cf_override")
                            tb_dr_override = tb_c2.number_input("Override Good Beta (beta_DR) [0 = Auto]", min_value=0.0, max_value=5.0, value=0.0, step=0.05, key="tb_dr_override")
                            tb_gamma = tb_c3.slider("Relative Risk Aversion (gamma)", min_value=2.0, max_value=12.0, value=6.0, step=0.5, key="tb_gamma")

                        tb_rec = evaluate_two_beta_profile(
                            ticker=selected_ticker,
                            cash_flow_beta_override=float(tb_cf_override) if tb_cf_override > 0.0 else None,
                            discount_rate_beta_override=float(tb_dr_override) if tb_dr_override > 0.0 else None,
                            risk_aversion_gamma=float(tb_gamma),
                        )

                        tb1, tb2, tb3, tb4 = st.columns(4)
                        tb_badge = (
                            "🔴 CASH-FLOW VULNERABLE"
                            if tb_rec.is_cash_flow_vulnerable
                            else (
                                "⚡ DISCOUNT-RATE GROWTH"
                                if tb_rec.is_discount_rate_growth
                                else (
                                    "🛡️ DEFENSIVE INSULATED"
                                    if tb_rec.is_defensive_insulated
                                    else f"{tb_rec.regime}"
                                )
                            )
                        )

                        tb1.metric(
                            "Bad Beta (beta_CF)",
                            f"{tb_rec.cash_flow_bad_beta:.2f}x",
                            delta=tb_badge,
                            delta_color="inverse" if tb_rec.cash_flow_bad_beta > 0.60 else "normal",
                            help="Cash-Flow Beta ('Bad Beta') measuring sensitivity to permanent market cashflow shocks; carries high risk price",
                        )
                        tb2.metric(
                            "Good Beta (beta_DR)",
                            f"{tb_rec.discount_rate_good_beta:.2f}x",
                            delta=f"Total Beta: {tb_rec.total_market_beta:.2f}x",
                            help="Discount-Rate Beta ('Good Beta') measuring sensitivity to temporary discount-rate fluctuations",
                        )
                        tb3.metric(
                            "Bad Beta Surcharge",
                            f"{tb_rec.bad_beta_hurdle_surcharge_bps:+.1f} bps",
                            delta=f"ICAPM: {tb_rec.two_beta_icapm_return_pct:.2f}%",
                            delta_color="inverse" if tb_rec.bad_beta_hurdle_surcharge_bps > 50.0 else "normal",
                            help="Required hurdle rate difference between two-beta ICAPM and single-beta CAPM",
                        )
                        tb4.metric(
                            "Bad Beta Ratio",
                            f"{tb_rec.bad_beta_ratio:.1%}",
                            delta="Cash-Flow Share",
                            help="Fraction of systematic market risk driven by permanent cash-flow innovations",
                        )

                        st.caption(f"🧾 **Two-Beta Summary:** {tb_rec.summary}")
                        st.caption(f"🛡️ **Two-Beta Audit Digest:** `{tb_rec.audit_digest[:16]}...` (Verified: `{verify_two_beta_audit_digest(tb_rec)}`)")
                    except Exception as tb_ex:
                        st.info(f"Campbell-Vuolteenaho Two-Beta evaluation deferred: {tb_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Two-Beta profile.")

                st.divider()
                st.markdown("#### 🔭 Merton (1987) Incomplete Information & Investor Recognition Engine")
                st.caption(
                    "Evaluates the Merton (1987) incomplete information equilibrium, measuring the investor recognition ratio ($q$), "
                    "shadow cost of limited attention, and required hurdle rate surcharges on neglected securities."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.merton_incomplete_information import (
                            evaluate_merton_incomplete_info_profile,
                            verify_merton_audit_digest,
                        )

                        with st.expander("⚙️ Customize Investor Recognition & Shadow Cost (Optional)", expanded=False):
                            mi_c1, mi_c2 = st.columns(2)
                            mi_q_override = mi_c1.slider("Override Recognition Ratio q (%) [0 = Auto]", min_value=0, max_value=100, value=0, step=5, key="mi_q_override")
                            mi_lambda = mi_c2.slider("Shadow Cost Multiplier lambda", min_value=0.05, max_value=1.00, value=0.25, step=0.05, key="mi_lambda")

                        mi_rec = evaluate_merton_incomplete_info_profile(
                            ticker=selected_ticker,
                            recognition_q_override=float(mi_q_override / 100.0) if mi_q_override > 0 else None,
                            shadow_lambda_override=float(mi_lambda),
                        )

                        mi1, mi2, mi3, mi4 = st.columns(4)
                        mi_badge = (
                            "🔴 NEGLECTED FIRM SURCHARGE"
                            if mi_rec.is_neglected_firm
                            else (
                                "⚡ ELEVATED INCOMPLETE"
                                if mi_rec.is_elevated_incomplete
                                else (
                                    "🟢 BROAD RECOGNITION"
                                    if mi_rec.is_broad_recognition
                                    else "🌐 UNIVERSAL SATURATED"
                                )
                            )
                        )

                        mi1.metric(
                            "Shadow Surcharge",
                            f"{mi_rec.merton_shadow_surcharge_pct:+.2f}% Ann",
                            delta=f"{mi_rec.merton_shadow_surcharge_bps:+.0f} bps",
                            delta_color="inverse" if mi_rec.merton_shadow_surcharge_pct > 1.5 else "normal",
                            help="Merton (1987) cost of capital surcharge demanded by cognizant investors for holding undiversified positions in limited-recognition stocks",
                        )
                        mi2.metric(
                            "Investor Recognition (q)",
                            f"{mi_rec.investor_recognition_ratio_q:.1%}",
                            delta=mi_badge,
                            delta_color="inverse" if mi_rec.investor_recognition_ratio_q <= 0.25 else "normal",
                            help="Fraction of investors in the economy who are cognizant of and invest in the security",
                        )
                        mi3.metric(
                            "Total Hurdle Return",
                            f"{mi_rec.total_required_hurdle_pct:.2f}% Ann",
                            delta=f"CAPM: {mi_rec.capm_required_return_pct:.2f}%",
                            help="Total multi-factor required return combining CAPM systematic risk and Merton shadow cost surcharge",
                        )
                        mi4.metric(
                            "Idiosyncratic Variance",
                            f"{mi_rec.idiosyncratic_variance:.4f}",
                            delta=f"Beta_M: {mi_rec.market_equity_beta:.2f}x",
                            help="Annualized idiosyncratic variance representing undiversified risk borne by cognizant investors",
                        )

                        st.caption(f"🧾 **Incomplete Information Summary:** {mi_rec.summary}")
                        st.caption(f"🛡️ **Merton Audit Digest:** `{mi_rec.audit_digest[:16]}...` (Verified: `{verify_merton_audit_digest(mi_rec)}`)")
                    except Exception as mi_ex:
                        st.info(f"Merton Incomplete Information evaluation deferred: {mi_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Merton Incomplete Information profile.")

                st.markdown("---")
                st.subheader("🌪️ Xavier Gabaix (2012) Variable Rare Disasters & Macro Tail Risk Engine")
                st.caption(
                    "Evaluates the Gabaix (2012) variable rare disaster framework, modeling time-varying macroeconomic tail risk "
                    "intensity ($\\lambda_t$), pricing kernel disaster spikes ($B^{-\\gamma}$), firm cash-flow resilience ($F_i$), "
                    "and affine price-to-dividend multiples."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.variable_rare_disasters import (
                            evaluate_rare_disasters_profile,
                            verify_disaster_audit_digest,
                        )

                        with st.expander("⚙️ Customize Macro Disaster & Resilience Parameters (Optional)", expanded=False):
                            vrd_c1, vrd_c2 = st.columns(2)
                            vrd_lambda = vrd_c1.slider("Macro Disaster Arrival Rate lambda (%) [0 = Auto]", min_value=0.0, max_value=15.0, value=0.0, step=0.5, key="vrd_lambda")
                            vrd_rec_f = vrd_c2.slider("Dividend Recovery Factor F (%) [0 = Auto]", min_value=0, max_value=120, value=0, step=5, key="vrd_rec_f")

                        vrd_receipt = evaluate_rare_disasters_profile(
                            ticker=selected_ticker,
                            disaster_lambda_override=float(vrd_lambda / 100.0) if vrd_lambda > 0 else None,
                            recovery_f_override=float(vrd_rec_f / 100.0) if vrd_rec_f > 0 else None,
                        )

                        vrd1, vrd2, vrd3, vrd4 = st.columns(4)
                        vrd_badge = (
                            "🔴 DISASTER VULNERABLE"
                            if vrd_receipt.is_disaster_vulnerable
                            else (
                                "⚡ CYCLICAL EXPOSURE"
                                if vrd_receipt.is_cyclical_exposure
                                else (
                                    "🟢 BENIGN STABILITY"
                                    if vrd_receipt.is_benign_stability
                                    else "🏰 RESILIENT FORTRESS"
                                )
                            )
                        )

                        vrd1.metric(
                            "Disaster Premium (H)",
                            f"{vrd_receipt.disaster_resilience_premium_pct:+.2f}% Ann",
                            delta=f"{vrd_receipt.disaster_resilience_premium_bps:+.0f} bps",
                            delta_color="inverse" if vrd_receipt.disaster_resilience_premium_pct > 3.0 else "normal",
                            help="Gabaix (2012) disaster resilience premium demanded by investors for exposure to macro disaster tail risk",
                        )
                        vrd2.metric(
                            "Dividend Recovery (F)",
                            f"{vrd_receipt.firm_dividend_recovery_factor_f:.1%}",
                            delta=vrd_badge,
                            delta_color="inverse" if vrd_receipt.firm_dividend_recovery_factor_f <= 0.60 else "normal",
                            help="Firm cash flow and dividend recovery factor during a catastrophic macroeconomic contraction",
                        )
                        vrd3.metric(
                            "Total Tail Hurdle",
                            f"{vrd_receipt.total_disaster_hurdle_pct:.2f}% Ann",
                            delta=f"CAPM: {vrd_receipt.capm_required_return_pct:.2f}%",
                            help="Total hurdle discount rate accounting for normal market risk and variable disaster resilience",
                        )
                        vrd4.metric(
                            "Implied P/D Multiple",
                            f"{vrd_receipt.implied_price_dividend_multiple:.1f}x",
                            delta=f"Kernel Spike: {vrd_receipt.pricing_kernel_disaster_spike:.1f}x",
                            help="Gabaix closed-form implied price-to-dividend valuation multiple under disaster risk",
                        )

                        st.caption(f"🧾 **Variable Rare Disasters Summary:** {vrd_receipt.summary}")
                        st.caption(f"🛡️ **Gabaix Audit Digest:** `{vrd_receipt.audit_digest[:16]}...` (Verified: `{verify_disaster_audit_digest(vrd_receipt)}`)")
                    except Exception as vrd_ex:
                        st.info(f"Variable Rare Disasters evaluation deferred: {vrd_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Variable Rare Disasters profile.")

                st.markdown("---")
                st.subheader("🌊 Albert S. Kyle & Wei Xiong (2001) Contagion as a Wealth Effect Engine")
                st.caption(
                    "Evaluates the Kyle & Xiong (2001) financial contagion framework, analyzing how arbitrageur equity wealth shocks "
                    "contract risk-bearing capacity ($\\kappa_t$), forcing multi-asset liquidation, non-fundamental excess co-movement "
                    "($\\Delta \\rho_{\\text{wealth}}$), and fire-sale discounts."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.kyle_xiong_contagion import (
                            evaluate_kyle_xiong_profile,
                            verify_kyle_xiong_audit_digest,
                        )

                        with st.expander("⚙️ Customize Arbitrageur Wealth Shock & Risk Parameters (Optional)", expanded=False):
                            kx_c1, kx_c2 = st.columns(2)
                            kx_wealth_dd = kx_c1.slider("Arbitrageur Wealth Drawdown (%) [0 = Auto]", min_value=0, max_value=85, value=0, step=5, key="kx_wealth_dd")
                            kx_gamma = kx_c2.slider("Arbitrageur Risk Aversion gamma", min_value=1.0, max_value=8.0, value=2.5, step=0.5, key="kx_gamma")

                        kx_rec = evaluate_kyle_xiong_profile(
                            ticker=selected_ticker,
                            wealth_drawdown_override=float(kx_wealth_dd) if kx_wealth_dd > 0 else None,
                            gamma_override=float(kx_gamma),
                        )

                        kx1, kx2, kx3, kx4 = st.columns(4)
                        kx_badge = (
                            "🔴 SYSTEMIC CONTAGION"
                            if kx_rec.is_systemic_contagion
                            else (
                                "⚡ MODERATE SPILLOVER"
                                if kx_rec.is_moderate_spillover
                                else (
                                    "🟢 BENIGN DECOUPLED"
                                    if kx_rec.is_benign_decoupled
                                    else "🛡️ CAPITAL SURPLUS"
                                )
                            )
                        )

                        kx1.metric(
                            "Spillover Beta",
                            f"{kx_rec.contagion_spillover_beta:.2f}x",
                            delta=f"{kx_rec.forced_liquidation_pressure:.1f} / 100 Pressure",
                            delta_color="inverse" if kx_rec.contagion_spillover_beta > 0.40 else "normal",
                            help="Kyle & Xiong (2001) cross-market contagion sensitivity beta induced by arbitrageur wealth liquidation",
                        )
                        kx2.metric(
                            "Excess Co-Movement",
                            f"{kx_rec.non_fundamental_excess_corr:.2f}",
                            delta=kx_badge,
                            delta_color="inverse" if kx_rec.non_fundamental_excess_corr > 0.35 else "normal",
                            help="Non-fundamental correlation increase Delta rho_wealth arising purely from convergence traders unwinding unrelated positions",
                        )
                        kx3.metric(
                            "Risk Capacity (kappa)",
                            f"{kx_rec.risk_bearing_capacity:.2f}",
                            delta=f"Drawdown: -{kx_rec.wealth_drawdown_pct:.1f}%",
                            delta_color="inverse" if kx_rec.wealth_drawdown_pct > 20.0 else "normal",
                            help="Effective arbitrageur risk-bearing capacity kappa_t = W_t / (gamma * sigma^2)",
                        )
                        kx4.metric(
                            "Fire-Sale Discount",
                            f"{kx_rec.liquidation_discount_pct:.2f}%",
                            delta=f"Vol Amp: {kx_rec.volatility_amplification_factor:.2f}x",
                            delta_color="inverse" if kx_rec.liquidation_discount_pct > 2.0 else "normal",
                            help="Expected price markdown required to absorb forced arbitrageur liquidation in illiquid conditions",
                        )

                        st.caption(f"🧾 **Contagion Summary:** {kx_rec.summary}")
                        st.caption(f"🛡️ **Kyle-Xiong Audit Digest:** `{kx_rec.audit_digest[:16]}...` (Verified: `{verify_kyle_xiong_audit_digest(kx_rec)}`)")
                    except Exception as kx_ex:
                        st.info(f"Kyle & Xiong Contagion evaluation deferred: {kx_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Kyle & Xiong Contagion profile.")

                st.markdown("---")
                st.subheader("🦈 Markus K. Brunnermeier & Lasse Heje Pedersen (2005) Predatory Trading Engine")
                st.caption(
                    "Evaluates the Brunnermeier & Pedersen (2005) predatory trading framework, detecting strategic front-running of "
                    "distressed institutional liquidation, predatory price overshoot ($\\Omega_{\\text{overshoot}}$), liquidity evaporation, "
                    "and post-trough rebound potential."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.predatory_trading import (
                            evaluate_predatory_trading_profile,
                            verify_predatory_audit_digest,
                        )

                        with st.expander("⚙️ Customize Distressed Liquidation & Predator Dynamics (Optional)", expanded=False):
                            pt_c1, pt_c2 = st.columns(2)
                            pt_size = pt_c1.slider("Distressed Liquidation (% ADV) [0 = Auto]", min_value=0.0, max_value=15.0, value=0.0, step=0.5, key="pt_size")
                            pt_pred = pt_c2.slider("Strategic Predator Share (%)", min_value=10, max_value=90, value=50, step=5, key="pt_pred")

                        pt_rec = evaluate_predatory_trading_profile(
                            ticker=selected_ticker,
                            distressed_size_override=float(pt_size) if pt_size > 0 else None,
                            predator_ratio_override=float(pt_pred / 100.0),
                        )

                        pt1, pt2, pt3, pt4 = st.columns(4)
                        pt_badge = (
                            "🔴 AGGRESSIVE PREDATORY RUN"
                            if pt_rec.is_aggressive_predatory_run
                            else (
                                "⚡ COORDINATED FRONT-RUNNING"
                                if pt_rec.is_coordinated_front_running
                                else (
                                    "🛡️ COMPETITIVE ABSORPTION"
                                    if pt_rec.is_competitive_absorption
                                    else "🟢 BENIGN LIQUIDITY"
                                )
                            )
                        )

                        pt1.metric(
                            "Price Overshoot",
                            f"{pt_rec.price_overshoot_ratio:.2f}x",
                            delta=f"Rebound: +{pt_rec.expected_rebound_potential_pct:.1f}%",
                            delta_color="inverse" if pt_rec.price_overshoot_ratio > 1.25 else "normal",
                            help="Brunnermeier & Pedersen (2005) price overshoot ratio induced by strategic predators front-running distressed liquidation",
                        )
                        pt2.metric(
                            "Predatory Shorting",
                            f"{pt_rec.predatory_shorting_volume_pct_adv:.2f}% ADV",
                            delta=pt_badge,
                            delta_color="inverse" if pt_rec.predatory_shorting_volume_pct_adv > 2.0 else "normal",
                            help="Volume sold or shorted by strategic predators alongside the distressed trader",
                        )
                        pt3.metric(
                            "Liquidity Drain",
                            f"{pt_rec.predatory_liquidity_drain_pct:.1f}%",
                            delta=f"Distressed: {pt_rec.distressed_liquidation_size_pct_adv:.1f}% ADV",
                            delta_color="inverse" if pt_rec.predatory_liquidity_drain_pct > 30.0 else "normal",
                            help="Fraction of order book depth drained by opportunistic predators",
                        )
                        pt4.metric(
                            "Slippage Inflation",
                            f"{pt_rec.distressed_slippage_inflation_pct:+.1f}%",
                            delta=f"Depth: {pt_rec.market_depth_multiplier:.2f}x",
                            delta_color="inverse" if pt_rec.distressed_slippage_inflation_pct > 25.0 else "normal",
                            help="Execution slippage cost penalty borne by the distressed entity due to predatory front-running",
                        )

                        st.caption(f"🧾 **Predatory Trading Summary:** {pt_rec.summary}")
                        st.caption(f"🛡️ **Predatory Audit Digest:** `{pt_rec.audit_digest[:16]}...` (Verified: `{verify_predatory_audit_digest(pt_rec)}`)")
                    except Exception as pt_ex:
                        st.info(f"Predatory Trading evaluation deferred: {pt_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Predatory Trading profile.")

                st.markdown("---")
                st.subheader("⚡ Harrison Hong & Jeremy C. Stein (2003) Differences of Opinion & Crash Asymmetry Engine")
                st.caption(
                    "Evaluates the Hong & Stein (2003) market crash framework, modeling how short-sale constraints sideline pessimistic "
                    "investors, accumulating latent negative information overhang ($I_{\\text{latent}}$) that suddenly reveals in a cascading "
                    "downturn to produce severe negative return skewness ($\\mathcal{S}_{\\text{crash}}$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.hong_stein_crash import (
                            evaluate_hong_stein_crash_profile,
                            verify_hong_stein_audit_digest,
                        )

                        with st.expander("⚙️ Customize Opinion Dispersion & Short-Sale Constraints (Optional)", expanded=False):
                            hs_c1, hs_c2 = st.columns(2)
                            hs_disp = hs_c1.slider("Opinion Dispersion sigma_theta (%) [0 = Auto]", min_value=0, max_value=60, value=0, step=5, key="hs_disp")
                            hs_const = hs_c2.slider("Short-Sale Constraint Tightness (%)", min_value=0, max_value=100, value=55, step=5, key="hs_const")

                        hs_rec = evaluate_hong_stein_crash_profile(
                            ticker=selected_ticker,
                            opinion_dispersion_override=float(hs_disp / 100.0) if hs_disp > 0 else None,
                            short_constraint_override=float(hs_const / 100.0),
                        )

                        hs1, hs2, hs3, hs4 = st.columns(4)
                        hs_badge = (
                            "🔴 ACUTE CRASH VULNERABILITY"
                            if hs_rec.is_acute_crash_vulnerability
                            else (
                                "⚡ ELEVATED OVERHANG"
                                if hs_rec.is_elevated_latent_overhang
                                else (
                                    "🟢 SYMMETRIC UNCONSTRAINED"
                                    if hs_rec.is_symmetric_unconstrained
                                    else "🛡️ CONSENSUS STABILITY"
                                )
                            )
                        )

                        hs1.metric(
                            "Crash Skewness",
                            f"{hs_rec.conditional_crash_skewness:.2f}",
                            delta=f"{hs_rec.crash_vulnerability_index:.1f} / 100 Index",
                            delta_color="inverse" if hs_rec.conditional_crash_skewness < -0.80 else "normal",
                            help="Hong & Stein (2003) conditional negative return skewness induced by unrevealed bearish overhang",
                        )
                        hs2.metric(
                            "Latent Overhang",
                            f"{hs_rec.latent_information_overhang:.2f}",
                            delta=hs_badge,
                            delta_color="inverse" if hs_rec.latent_information_overhang > 0.80 else "normal",
                            help="Accumulated pent-up negative private information held by sidelined pessimists",
                        )
                        hs3.metric(
                            "Downside Surcharge",
                            f"{hs_rec.downside_volatility_surcharge_pct:+.1f}%",
                            delta=f"Constraint: {hs_rec.short_sale_constraint_tightness:.0%}",
                            delta_color="inverse" if hs_rec.downside_volatility_surcharge_pct > 5.0 else "normal",
                            help="Asymmetric downside volatility surcharge expected when negative signals cascade into price",
                        )
                        hs4.metric(
                            "Expected Cascade DD",
                            f"{hs_rec.expected_crash_drawdown_pct:.1f}%",
                            delta=f"Dispersion: {hs_rec.opinion_dispersion_ratio:.1%}",
                            delta_color="inverse" if hs_rec.expected_crash_drawdown_pct > 5.0 else "normal",
                            help="Simulated price drawdown during a clustered revelation of latent bearish signals",
                        )

                        st.caption(f"🧾 **Crash Asymmetry Summary:** {hs_rec.summary}")
                        st.caption(f"🛡️ **Hong-Stein Audit Digest:** `{hs_rec.audit_digest[:16]}...` (Verified: `{verify_hong_stein_audit_digest(hs_rec)}`)")
                    except Exception as hs_ex:
                        st.info(f"Hong-Stein Crash Asymmetry evaluation deferred: {hs_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Hong-Stein Crash Asymmetry profile.")

                st.markdown("---")
                st.subheader("☕ Douglas T. Breeden (1979) Consumption CAPM (CCAPM) Engine")
                st.caption(
                    "Evaluates the Breeden (1979) Consumption-Based Asset Pricing Model, measuring continuous-time consumption beta "
                    "($\\beta_{i, C}$), intertemporal hedging demand, consumption risk premia ($\\lambda_C$), and the equity premium puzzle "
                    "pricing wedge ($E[R_i]_{\\text{CAPM}} - E[R_i]_{\\text{CCAPM}}$)."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.breeden_ccapm import (
                            evaluate_breeden_ccapm_profile,
                            verify_breeden_audit_digest,
                        )

                        with st.expander("⚙️ Customize Macro Consumption Risk Parameters (Optional)", expanded=False):
                            cc_c1, cc_c2 = st.columns(2)
                            cc_beta = cc_c1.slider("Override Consumption Beta beta_C [0 = Auto]", min_value=-1.0, max_value=3.5, value=0.0, step=0.1, key="cc_beta")
                            cc_lambda = cc_c2.slider("Consumption Risk Premium lambda_C (%)", min_value=1.0, max_value=8.0, value=3.8, step=0.2, key="cc_lambda")

                        cc_rec = evaluate_breeden_ccapm_profile(
                            ticker=selected_ticker,
                            consumption_beta_override=float(cc_beta) if cc_beta != 0.0 else None,
                            lambda_c_override=float(cc_lambda / 100.0),
                        )

                        cc1, cc2, cc3, cc4 = st.columns(4)
                        cc_badge = (
                            "🔴 HIGH CONSUMPTION RISK"
                            if cc_rec.is_high_consumption_exposure
                            else (
                                "⚡ MODERATE PROCYCLICAL"
                                if cc_rec.is_moderate_procyclical
                                else (
                                    "🛡️ DEFENSIVE HEDGE"
                                    if cc_rec.is_defensive_hedge
                                    else "🟢 COUNTERCYCLICAL ASSET"
                                )
                            )
                        )

                        cc1.metric(
                            "CCAPM Required Return",
                            f"{cc_rec.ccapm_required_return_pct:.2f}% Ann",
                            delta=f"Beta_C: {cc_rec.consumption_beta:.2f}x",
                            delta_color="inverse" if cc_rec.consumption_beta > 1.5 else "normal",
                            help="Breeden (1979) consumption-based required hurdle return Rf + beta_C * lambda_C",
                        )
                        cc2.metric(
                            "Consumption Wedge",
                            f"{cc_rec.equity_premium_puzzle_wedge_pct:+.2f}%",
                            delta=cc_badge,
                            delta_color="normal" if abs(cc_rec.equity_premium_puzzle_wedge_pct) < 2.0 else "inverse",
                            help="Pricing wedge between standard market CAPM and macro CCAPM required returns",
                        )
                        cc3.metric(
                            "Macro Hurdle Rate",
                            f"{cc_rec.macro_hurdle_rate_pct:.2f}% Ann",
                            delta=f"CAPM: {cc_rec.capm_benchmark_return_pct:.2f}%",
                            help="Discount rate benchmark reflecting aggregate real consumption smoothing requirements",
                        )
                        cc4.metric(
                            "Hedge Elasticity",
                            f"{cc_rec.intertemporal_hedge_elasticity:.2f}%",
                            delta=f"Lambda_C: {cc_rec.consumption_risk_premium_pct:.1f}%",
                            help="Intertemporal consumption hedge elasticity beta_C * sigma_C",
                        )

                        st.caption(f"🧾 **Breeden CCAPM Summary:** {cc_rec.summary}")
                        st.caption(f"🛡️ **Breeden Audit Digest:** `{cc_rec.audit_digest[:16]}...` (Verified: `{verify_breeden_audit_digest(cc_rec)}`)")
                    except Exception as cc_ex:
                        st.info(f"Breeden CCAPM evaluation deferred: {cc_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Breeden CCAPM profile.")

                st.markdown("---")
                st.subheader("🎲 Nicholas Barberis, Ming Huang & Tano Santos (2001) Prospect Theory & Loss Aversion Engine")
                st.caption(
                    "Evaluates the Barberis, Huang & Santos (2001) behavioral asset pricing framework, modeling path-dependent dynamic loss "
                    "aversion ($\\lambda_t$), prior gain/loss cushion states ($z_t = P_{\\text{ref}} / P_t$), house money effects, "
                    "and prospect theory hurdle rate surcharges."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.barberis_huang_santos import (
                            evaluate_bhs_prospect_profile,
                            verify_bhs_audit_digest,
                        )

                        with st.expander("⚙️ Customize Reference Benchmark & Baseline Loss Aversion (Optional)", expanded=False):
                            bhs_c1, bhs_c2 = st.columns(2)
                            bhs_ref = bhs_c1.number_input("Override Reference Purchase Price ($) [0 = Auto]", min_value=0.0, max_value=10000.0, value=0.0, step=1.0, key="bhs_ref")
                            bhs_lam = bhs_c2.slider("Baseline Loss Aversion lambda_0", min_value=1.5, max_value=3.5, value=2.25, step=0.1, key="bhs_lam")

                        bhs_rec = evaluate_bhs_prospect_profile(
                            ticker=selected_ticker,
                            reference_price_override=float(bhs_ref) if bhs_ref > 0.0 else None,
                            lambda_zero_override=float(bhs_lam),
                        )

                        bhs1, bhs2, bhs3, bhs4 = st.columns(4)
                        bhs_badge = (
                            "🔴 ACUTE LOSS AVERSION"
                            if bhs_rec.is_acute_loss_aversion
                            else (
                                "⚡ RECESSION SENSITIVITY"
                                if bhs_rec.is_moderate_sensitivity
                                else (
                                    "🟢 NEUTRAL PROSPECT"
                                    if bhs_rec.is_benign_neutral
                                    else "🏰 HOUSE MONEY CUSHION"
                                )
                            )
                        )

                        bhs1.metric(
                            "Dynamic Loss Aversion",
                            f"{bhs_rec.dynamic_loss_aversion_lambda:.2f}x",
                            delta=f"z_state: {bhs_rec.prior_cushion_state_z:.2f}",
                            delta_color="inverse" if bhs_rec.dynamic_loss_aversion_lambda > 3.0 else "normal",
                            help="Barberis, Huang & Santos (2001) path-dependent loss aversion coefficient lambda_t(z_t)",
                        )
                        bhs2.metric(
                            "Loss Surcharge",
                            f"{bhs_rec.prospective_loss_surcharge_pct:+.2f}% Ann",
                            delta=bhs_badge,
                            delta_color="inverse" if bhs_rec.prospective_loss_surcharge_pct > 1.5 else "normal",
                            help="Prospective loss surcharge required to compensate loss-averse investors",
                        )
                        bhs3.metric(
                            "Total Prospect Hurdle",
                            f"{bhs_rec.total_prospect_hurdle_pct:.2f}% Ann",
                            delta=f"CAPM: {bhs_rec.capm_benchmark_return_pct:.2f}%",
                            help="Required hurdle return incorporating CAPM systematic risk and behavioral loss aversion penalty",
                        )
                        bhs4.metric(
                            "House Money Cushion",
                            f"{bhs_rec.house_money_cushion_pct:.1f}%",
                            delta=f"Prior: {bhs_rec.prior_return_pct:+.1f}%",
                            delta_color="normal" if bhs_rec.house_money_cushion_pct > 10.0 else "inverse",
                            help="Accumulated psychological capital cushion from prior gains dampening pain of future losses",
                        )

                        st.caption(f"🧾 **Prospect Theory Summary:** {bhs_rec.summary}")
                        st.caption(f"🛡️ **BHS Audit Digest:** `{bhs_rec.audit_digest[:16]}...` (Verified: `{verify_bhs_audit_digest(bhs_rec)}`)")
                    except Exception as bhs_ex:
                        st.info(f"BHS Prospect Theory evaluation deferred: {bhs_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate BHS Prospect Theory profile.")

                st.markdown("---")
                st.subheader("🕸️ Franklin Allen & Douglas Gale (2000) Financial Network Contagion Engine")
                st.caption(
                    "Evaluates the Allen & Gale (2000) financial network contagion framework, analyzing how cross-holding network "
                    "completeness ($\\gamma_{\\text{net}}$) vs. incompleteness governs localized liquidity deficit absorption, "
                    "contagion cascade multipliers ($\\mathcal{M}_{\\text{cascade}}$), and systemic fire-sale price markdowns."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.allen_gale_network import (
                            evaluate_allen_gale_network_profile,
                            verify_allen_gale_audit_digest,
                        )

                        with st.expander("⚙️ Customize Network Completeness & Liquidity Shock (Optional)", expanded=False):
                            ag_c1, ag_c2 = st.columns(2)
                            ag_comp = ag_c1.slider("Network Completeness gamma_net (%) [0 = Auto]", min_value=0, max_value=100, value=0, step=5, key="ag_comp")
                            ag_shock = ag_c2.slider("Localized Liquidity Shock theta (%) [0 = Auto]", min_value=0, max_value=100, value=0, step=5, key="ag_shock")

                        ag_rec = evaluate_allen_gale_network_profile(
                            ticker=selected_ticker,
                            network_completeness_override=float(ag_comp / 100.0) if ag_comp > 0 else None,
                            liquidity_shock_override=float(ag_shock / 100.0) if ag_shock > 0 else None,
                        )

                        ag1, ag2, ag3, ag4 = st.columns(4)
                        ag_badge = (
                            "🔴 SYSTEMIC CASCADE"
                            if ag_rec.is_systemic_cascade
                            else (
                                "⚡ FRAGILE NETWORK"
                                if ag_rec.is_incomplete_fragile
                                else (
                                    "🟢 RESILIENT SHARING"
                                    if ag_rec.is_resilient_sharing
                                    else "🏰 NETWORK FORTRESS"
                                )
                            )
                        )

                        ag1.metric(
                            "Cascade Multiplier",
                            f"{ag_rec.contagion_cascade_multiplier:.2f}x",
                            delta=f"{ag_rec.systemic_contagion_index:.1f} / 100 Index",
                            delta_color="inverse" if ag_rec.contagion_cascade_multiplier > 1.3 else "normal",
                            help="Allen & Gale (2000) contagion cascade multiplier amplifying localized liquidity deficits",
                        )
                        ag2.metric(
                            "Network Completeness",
                            f"{ag_rec.network_completeness_ratio:.0%}",
                            delta=ag_badge,
                            delta_color="normal" if ag_rec.network_completeness_ratio > 0.70 else "inverse",
                            help="Degree of cross-holding interconnectedness enabling global liquidity dispersion",
                        )
                        ag3.metric(
                            "Liquidity Deficit",
                            f"{ag_rec.unabsorbed_liquidity_deficit:.2f}",
                            delta=f"Buffer: {ag_rec.cross_holding_absorption_capacity:.2f}",
                            delta_color="inverse" if ag_rec.unabsorbed_liquidity_deficit > 0.0 else "normal",
                            help="Unabsorbed localized liquidity strain spilling over into neighboring network nodes",
                        )
                        ag4.metric(
                            "Fire-Sale Markdown",
                            f"{ag_rec.network_fire_sale_markdown_pct:.1f}%",
                            delta=f"Surcharge: +{ag_rec.systemic_hurdle_surcharge_pct:.1f}%",
                            delta_color="inverse" if ag_rec.network_fire_sale_markdown_pct > 3.0 else "normal",
                            help="Expected asset liquidation price markdown resulting from cascading cross-market fire-sales",
                        )

                        st.caption(f"🧾 **Network Contagion Summary:** {ag_rec.summary}")
                        st.caption(f"🛡️ **Allen-Gale Audit Digest:** `{ag_rec.audit_digest[:16]}...` (Verified: `{verify_allen_gale_audit_digest(ag_rec)}`)")
                    except Exception as ag_ex:
                        st.info(f"Allen & Gale Network Contagion evaluation deferred: {ag_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Allen & Gale Network Contagion profile.")

                st.markdown("---")
                st.subheader("⚖️ John Geanakoplos (2010) The Leverage Cycle & Collateral Margin Engine")
                st.caption(
                    "Evaluates the Geanakoplos (2010) leverage cycle framework, analyzing how fluctuations in collateral margin haircuts "
                    "($m_t$) drive system leverage ($\\mathcal{L}_t = 1/m_t$), forced de-leveraging pressure, collateral squeeze markdowns, "
                    "and credit cycle volatility spirals."
                )
                if len(df_eda) >= 20:
                    try:
                        from src.analytics.geanakoplos_leverage_cycle import (
                            evaluate_geanakoplos_leverage_profile,
                            verify_leverage_audit_digest,
                        )

                        with st.expander("⚙️ Customize Collateral Haircuts & Leverage Parameters (Optional)", expanded=False):
                            gl_c1, gl_c2 = st.columns(2)
                            gl_m = gl_c1.slider("Current Collateral Haircut m (%) [0 = Auto]", min_value=0, max_value=80, value=0, step=5, key="gl_m")
                            gl_m0 = gl_c2.slider("Baseline Normal Haircut m0 (%)", min_value=5, max_value=50, value=20, step=5, key="gl_m0")

                        gl_rec = evaluate_geanakoplos_leverage_profile(
                            ticker=selected_ticker,
                            haircut_override=float(gl_m / 100.0) if gl_m > 0 else None,
                            baseline_haircut_override=float(gl_m0 / 100.0),
                        )

                        gl1, gl2, gl3, gl4 = st.columns(4)
                        gl_badge = (
                            "🔴 MARGIN SQUEEZE"
                            if gl_rec.is_acute_margin_squeeze
                            else (
                                "⚡ DELEVERAGING PHASE"
                                if gl_rec.is_elevated_deleveraging
                                else (
                                    "🚀 EUPHORIC EXPANSION"
                                    if gl_rec.is_euphoric_expansion
                                    else "🟢 PRUDENT STABILITY"
                                )
                            )
                        )

                        gl1.metric(
                            "Collateral Haircut",
                            f"{gl_rec.collateral_haircut_m:.1%}",
                            delta=f"Leverage: {gl_rec.effective_leverage_ratio:.1f}x",
                            delta_color="inverse" if gl_rec.collateral_haircut_m > 0.35 else "normal",
                            help="Current collateral margin haircut requirement m_t governing borrowing capacity",
                        )
                        gl2.metric(
                            "Deleveraging Pressure",
                            f"{gl_rec.forced_deleveraging_pressure:.1f} / 100",
                            delta=gl_badge,
                            delta_color="inverse" if gl_rec.forced_deleveraging_pressure > 30.0 else "normal",
                            help="Forced liquidation pressure triggered by margin calls when current haircut exceeds baseline",
                        )
                        gl3.metric(
                            "Squeeze Markdown",
                            f"{gl_rec.collateral_squeeze_markdown_pct:.1f}%",
                            delta=f"Vol Amp: {gl_rec.volatility_amplification_multiple:.2f}x",
                            delta_color="inverse" if gl_rec.collateral_squeeze_markdown_pct > 2.0 else "normal",
                            help="Expected fire-sale price discount required to absorb forced de-leveraging liquidation",
                        )
                        gl4.metric(
                            "Total Leverage Hurdle",
                            f"{gl_rec.total_leverage_hurdle_pct:.2f}% Ann",
                            delta=f"Surcharge: +{gl_rec.leverage_hurdle_surcharge_pct:.2f}%",
                            help="Required hurdle discount rate accounting for systematic risk and leverage cycle liquidation risk",
                        )

                        st.caption(f"🧾 **Leverage Cycle Summary:** {gl_rec.summary}")
                        st.caption(f"🛡️ **Geanakoplos Audit Digest:** `{gl_rec.audit_digest[:16]}...` (Verified: `{verify_leverage_audit_digest(gl_rec)}`)")
                    except Exception as gl_ex:
                        st.info(f"Geanakoplos Leverage Cycle evaluation deferred: {gl_ex}")
                else:
                    st.info("ℹ️ Requires at least 20 price sessions to evaluate Geanakoplos Leverage Cycle profile.")

                st.markdown("---")
                st.subheader("🏦 Douglas W. Diamond & Philip H. Dybvig (1983) Bank Runs & Liquidity Transformation Engine")
                st.caption(
                    "Evaluates the Diamond & Dybvig (1983) liquidity transformation framework, modeling sequential service constraints, "
                    "illiquid asset fire-sale haircuts, early withdrawal panic dynamics ($f_1$), asset exhaustion thresholds ($\\bar{f}$), "
                    "and coordination run hazard probabilities."
                )
                if len(df_eda) >= 5:
                    try:
                        from src.analytics.diamond_dybvig_run import (
                            calculate_diamond_dybvig_run,
                            verify_diamond_dybvig_audit_digest,
                        )

                        dd_col1, dd_col2, dd_col3 = st.columns(3)
                        with dd_col1:
                            dd_withdraw = st.slider(
                                "Early Withdrawal Demand (f_1)",
                                min_value=0.0,
                                max_value=1.0,
                                value=0.20,
                                step=0.01,
                                format="%.2f",
                                help="Fraction of depositors/shareholders redeeming shares prematurely at t=1",
                            )
                        with dd_col2:
                            dd_reserve = st.slider(
                                "Liquid Cash Buffer (c)",
                                min_value=0.05,
                                max_value=0.50,
                                value=0.15,
                                step=0.01,
                                format="%.2f",
                                help="Fraction of portfolio held in liquid cash / short-term storage",
                            )
                        with dd_col3:
                            dd_salvage = st.slider(
                                "Illiquid Salvage Yield (L)",
                                min_value=0.20,
                                max_value=0.95,
                                value=0.65,
                                step=0.05,
                                format="%.2f",
                                help="Premature fire-sale salvage recovery rate for illiquid productive assets",
                            )

                        dd_rec = calculate_diamond_dybvig_run(
                            p_t=float(df_eda["close"].iloc[-1]),
                            adv=float(df_eda["volume"].mean()) if "volume" in df_eda.columns else 1_000_000.0,
                            liquid_reserve_ratio=dd_reserve,
                            early_withdrawal_ratio=dd_withdraw,
                            salvage_L=dd_salvage,
                            ticker=selected_ticker,
                        )

                        dd_m1, dd_m2, dd_m3, dd_m4 = st.columns(4)
                        dd_m1.metric(
                            "Run Coordination Hazard",
                            f"{dd_rec.run_hazard_probability:.1%}",
                            delta=f"Regime: {dd_rec.regime}",
                            delta_color="inverse" if dd_rec.run_hazard_probability > 0 else "normal",
                        )
                        dd_m2.metric(
                            "Exhaustion Cutoff (f_bar)",
                            f"{dd_rec.exhaustion_threshold_fbar:.1%}",
                            delta=f"Buffer Gap: {dd_rec.exhaustion_threshold_fbar - dd_rec.early_withdrawal_ratio:+.1%}",
                        )
                        dd_m3.metric(
                            "Patient Payoff v_2(f_1)",
                            f"{dd_rec.patient_t2_expected_payoff:.3f}x",
                            delta=f"vs r_1: {dd_rec.patient_t2_expected_payoff - dd_rec.r1_payout:+.3f}x",
                        )
                        dd_m4.metric(
                            "Distressed Price",
                            f"${dd_rec.distressed_asset_price:.2f}",
                            delta=f"-{dd_rec.fire_sale_loss_fraction:.1%}",
                            delta_color="inverse",
                        )

                        st.caption(f"🧾 **Diamond-Dybvig Summary:** {dd_rec.summary}")
                        st.caption(f"🛡️ **Diamond-Dybvig Audit Digest:** `{dd_rec.audit_digest[:16]}...` (Verified: `{verify_diamond_dybvig_audit_digest(dd_rec)}`)")
                    except Exception as dd_ex:
                        st.info(f"Diamond & Dybvig Run evaluation deferred: {dd_ex}")
                else:
                    st.info("ℹ️ Requires at least 5 price sessions to evaluate Diamond & Dybvig Run profile.")

                st.markdown("---")
                st.subheader("🌐 Robert C. Merton (1973) Intertemporal CAPM & State Variable Hedging Engine")
                st.caption(
                    "Evaluates the Merton (1973) continuous-time multi-beta equilibrium framework, isolating sensitivity to "
                    "market excess return ($\\beta_M$), term structure shifts ($\\beta_r$), and market volatility innovations ($\\beta_{\\sigma}$), "
                    "quantifying state-variable hedging premiums and ICAPM intertemporal alpha."
                )
                if len(df_eda) >= 15:
                    try:
                        from src.analytics.merton_icapm import (
                            evaluate_merton_icapm_profile,
                            verify_merton_icapm_audit_digest,
                        )

                        ic_col1, ic_col2, ic_col3 = st.columns(3)
                        with ic_col1:
                            mkt_prem_input = st.slider(
                                "Market Risk Premium (lambda_M %)",
                                min_value=2.0,
                                max_value=10.0,
                                value=5.5,
                                step=0.25,
                                format="%.2f%%",
                                help="Expected annual market equity risk premium",
                            )
                        with ic_col2:
                            rate_prem_input = st.slider(
                                "Interest Rate Risk Price (lambda_r %)",
                                min_value=-5.0,
                                max_value=2.0,
                                value=-2.0,
                                step=0.25,
                                format="%.2f%%",
                                help="Price of unexpected term structure / interest rate shocks",
                            )
                        with ic_col3:
                            vol_prem_input = st.slider(
                                "Volatility Risk Price (lambda_sigma %)",
                                min_value=-8.0,
                                max_value=0.0,
                                value=-3.5,
                                step=0.25,
                                format="%.2f%%",
                                help="Price of market volatility / uncertainty innovations",
                            )

                        ic_rec = evaluate_merton_icapm_profile(
                            ticker=selected_ticker,
                            prices=[float(x) for x in df_eda["close"].tolist()],
                            lambda_market=mkt_prem_input / 100.0,
                            lambda_rate=rate_prem_input / 100.0,
                            lambda_vol=vol_prem_input / 100.0,
                        )

                        ic_m1, ic_m2, ic_m3, ic_m4 = st.columns(4)
                        ic_m1.metric(
                            "Market Beta (beta_M)",
                            f"{ic_rec.market_beta:.3f}",
                            delta=f"Contrib: {ic_rec.market_risk_contribution_pct:+.2f}%",
                        )
                        ic_m2.metric(
                            "State Hedging Premium",
                            f"{ic_rec.total_state_hedging_premium_pct:+.2f}%",
                            delta=f"Vol: {ic_rec.vol_hedging_contribution_pct:+.2f}% | Rate: {ic_rec.rate_hedging_contribution_pct:+.2f}%",
                        )
                        ic_m3.metric(
                            "ICAPM Hurdle Rate",
                            f"{ic_rec.icapm_hurdle_rate_pct:.2f}%",
                            delta=f"Rf: {ic_rec.risk_free_rate_pct:.2f}%",
                        )
                        ic_m4.metric(
                            "ICAPM Alpha",
                            f"{ic_rec.icapm_alpha_pct:+.2f}%",
                            delta=f"Regime: {ic_rec.regime}",
                            delta_color="normal" if ic_rec.icapm_alpha_pct >= 0 else "inverse",
                        )

                        st.caption(f"🧾 **Merton ICAPM Summary:** {ic_rec.summary}")
                        st.caption(f"🛡️ **Merton ICAPM Audit Digest:** `{ic_rec.audit_digest[:16]}...` (Verified: `{verify_merton_icapm_audit_digest(ic_rec)}`)")
                    except Exception as ic_ex:
                        st.info(f"Merton ICAPM evaluation deferred: {ic_ex}")
                else:
                    st.info("ℹ️ Requires at least 15 price sessions to evaluate Merton ICAPM profile.")

                st.markdown("---")
                st.subheader("⚖️ Stephen A. Ross (1976) Arbitrage Pricing Theory (APT) Engine")
                st.caption(
                    "Evaluates the Ross (1976) Arbitrage Pricing Theory framework (Chen, Roll & Ross 1986), estimating multi-factor "
                    "macroeconomic sensitivities ($b_{\\text{growth}}, b_{\\text{inf}}, b_{\\text{def}}, b_{\\text{term}}$), "
                    "no-arbitrage expected returns ($E[R_i]$), systematic vs idiosyncratic variance decomposition, and arbitrage mispricing spreads ($\\Delta_{\\text{APT}}$)."
                )
                if len(df_eda) >= 15:
                    try:
                        from src.analytics.ross_apt import (
                            evaluate_ross_apt_profile,
                            verify_ross_apt_audit_digest,
                        )

                        apt_col1, apt_col2, apt_col3 = st.columns(3)
                        with apt_col1:
                            apt_growth_input = st.slider(
                                "Growth Factor Price (lambda_growth %)",
                                min_value=1.0,
                                max_value=8.0,
                                value=4.5,
                                step=0.25,
                                format="%.2f%%",
                                help="Expected annual risk price for real economic output growth",
                            )
                        with apt_col2:
                            apt_inf_input = st.slider(
                                "Inflation Shock Price (lambda_inf %)",
                                min_value=-6.0,
                                max_value=1.0,
                                value=-2.5,
                                step=0.25,
                                format="%.2f%%",
                                help="Risk price for unexpected inflation innovations",
                            )
                        with apt_col3:
                            apt_def_input = st.slider(
                                "Credit Default Price (lambda_def %)",
                                min_value=-6.0,
                                max_value=0.0,
                                value=-3.0,
                                step=0.25,
                                format="%.2f%%",
                                help="Risk price for credit default spread widening",
                            )

                        apt_rec = evaluate_ross_apt_profile(
                            ticker=selected_ticker,
                            prices=[float(x) for x in df_eda["close"].tolist()],
                            lambda_growth=apt_growth_input / 100.0,
                            lambda_inflation=apt_inf_input / 100.0,
                            lambda_default=apt_def_input / 100.0,
                        )

                        apt_m1, apt_m2, apt_m3, apt_m4 = st.columns(4)
                        apt_m1.metric(
                            "APT Expected Return",
                            f"{apt_rec.apt_expected_return_pct:.2f}%",
                            delta=f"Sys Prem: {apt_rec.total_systematic_risk_premium_pct:+.2f}%",
                        )
                        apt_m2.metric(
                            "Arbitrage Spread (Delta)",
                            f"{apt_rec.apt_arbitrage_spread_pct:+.2f}%",
                            delta=f"Regime: {apt_rec.regime}",
                            delta_color="normal" if apt_rec.apt_arbitrage_spread_pct >= 0 else "inverse",
                        )
                        apt_m3.metric(
                            "Systematic Variance Ratio",
                            f"{apt_rec.systematic_variance_ratio_pct:.1f}%",
                            delta=f"Idio: {apt_rec.idiosyncratic_variance_ratio_pct:.1f}%",
                        )
                        apt_m4.metric(
                            "Growth Beta (b_growth)",
                            f"{apt_rec.growth_factor_beta:.3f}",
                            delta=f"Def Beta: {apt_rec.default_factor_beta:.3f}",
                        )

                        st.caption(f"🧾 **Ross APT Summary:** {apt_rec.summary}")
                        st.caption(f"🛡️ **Ross APT Audit Digest:** `{apt_rec.audit_digest[:16]}...` (Verified: `{verify_ross_apt_audit_digest(apt_rec)}`)")
                    except Exception as apt_ex:
                        st.info(f"Ross APT evaluation deferred: {apt_ex}")
                else:
                    st.info("ℹ️ Requires at least 15 price sessions to evaluate Ross APT profile.")

                st.markdown("---")
                st.subheader("👥 De Long, Shleifer, Summers & Waldmann (1990) Noise Trader Risk Engine")
                st.caption(
                    "Evaluates the DSSW (1990) limits-of-arbitrage framework, modeling how unpredictable noise trader sentiment "
                    "($S_t$, $\\sigma_S^2$) generates resale price risk for finite-horizon rational arbitrageurs, drives the create-space "
                    "volatility discount, and enables noise traders to potentially outperform arbitrageurs."
                )
                if len(df_eda) >= 15:
                    try:
                        from src.analytics.dssw_noise_trader_risk import (
                            evaluate_dssw_profile,
                            verify_dssw_audit_digest,
                        )

                        dssw_col1, dssw_col2, dssw_col3 = st.columns(3)
                        with dssw_col1:
                            dssw_shock_input = st.slider(
                                "Active Sentiment Shock (S_t)",
                                min_value=-0.30,
                                max_value=0.30,
                                value=0.08,
                                step=0.01,
                                format="%.2f",
                                help="Current period unexpected sentiment shock",
                            )
                        with dssw_col2:
                            dssw_var_input = st.slider(
                                "Sentiment Variance (sigma_S^2)",
                                min_value=0.002,
                                max_value=0.060,
                                value=0.015,
                                step=0.002,
                                format="%.3f",
                                help="Unpredictability/variance of future noise trader sentiment",
                            )
                        with dssw_col3:
                            dssw_gamma_input = st.slider(
                                "Arbitrageur Risk Aversion (gamma)",
                                min_value=0.5,
                                max_value=4.0,
                                value=1.25,
                                step=0.25,
                                format="%.2f",
                                help="Absolute risk aversion coefficient of rational arbitrageurs",
                            )

                        dssw_rec = evaluate_dssw_profile(
                            ticker=selected_ticker,
                            prices=[float(x) for x in df_eda["close"].tolist()],
                            sentiment_shock_override=dssw_shock_input,
                            sentiment_variance_override=dssw_var_input,
                            risk_aversion=dssw_gamma_input,
                        )

                        dssw_m1, dssw_m2, dssw_m3, dssw_m4 = st.columns(4)
                        dssw_m1.metric(
                            "DSSW Market Price",
                            f"${dssw_rec.dssw_market_price:.2f}",
                            delta=f"Wedge: {dssw_rec.mispricing_percentage_pct:+.1f}%",
                            delta_color="normal" if abs(dssw_rec.mispricing_percentage_pct) <= 15.0 else "inverse",
                        )
                        dssw_m2.metric(
                            "Create-Space Discount",
                            f"${dssw_rec.create_space_discount_dollar:.2f}",
                            delta=f"Bias Prem: +${dssw_rec.persistent_bias_premium_dollar:.2f}",
                        )
                        dssw_m3.metric(
                            "Return Diff (Noise - Arbs)",
                            f"{dssw_rec.expected_return_differential_pct:+.2f}%",
                            delta="Noise Outperforms" if dssw_rec.is_noise_trader_outperforming else "Arbs Outperform",
                            delta_color="normal" if dssw_rec.is_noise_trader_outperforming else "off",
                        )
                        dssw_m4.metric(
                            "Active Regime",
                            dssw_rec.regime,
                            delta=f"Fundamental: ${dssw_rec.fundamental_price:.2f}",
                        )

                        st.caption(f"🧾 **DSSW Summary:** {dssw_rec.summary}")
                        st.caption(f"🛡️ **DSSW Audit Digest:** `{dssw_rec.audit_digest[:16]}...` (Verified: `{verify_dssw_audit_digest(dssw_rec)}`)")
                    except Exception as dssw_ex:
                        st.info(f"DSSW Noise Trader Risk evaluation deferred: {dssw_ex}")
                else:
                    st.info("ℹ️ Requires at least 15 price sessions to evaluate DSSW Noise Trader Risk profile.")


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

            # Almgren & Chriss (2000) Optimal Execution Schedule
            st.markdown("---")
            st.subheader("⚡ Almgren & Chriss (2000) Optimal Execution Schedule")
            st.caption(
                "Dynamic inventory trajectory minimizing implementation shortfall (permanent & temporary impact) and timing risk."
            )
            from src.analytics.optimal_execution import evaluate_optimal_execution_plan
            with st.expander("⚙️ Execution Parameter Controls & Optimal Schedule", expanded=False):
                col_ex1, col_ex2, col_ex3 = st.columns(3)
                target_shares_input = col_ex1.number_input(
                    "Target Shares (positive=Sell, negative=Buy)",
                    value=1000.0,
                    step=100.0,
                    key="ac_shares",
                )
                exec_days_input = col_ex2.slider(
                    "Execution Horizon (Sessions)",
                    min_value=1,
                    max_value=21,
                    value=5,
                    key="ac_days",
                )
                risk_aversion_input = col_ex3.select_slider(
                    "Risk Aversion (λ)",
                    options=[0.0, 1e-7, 1e-6, 1e-5, 1e-4],
                    value=1e-6,
                    format_func=lambda x: "0.0 (TWAP)" if x == 0.0 else f"{x:.1e}",
                    key="ac_lambda",
                )

                if len(df_obs) >= 10:
                    try:
                        p_col = "adj_close" if "adj_close" in df_obs.columns else "close"
                        plan_rec = evaluate_optimal_execution_plan(
                            ticker=selected_ticker,
                            target_shares=float(target_shares_input),
                            execution_days=int(exec_days_input),
                            risk_aversion_lambda=float(risk_aversion_input),
                            prices=df_obs[p_col],
                            volumes=df_obs["volume"] if "volume" in df_obs.columns else None,
                        )

                        eq1, eq2, eq3, eq4 = st.columns(4)
                        eq1.metric("Order Value", f"${plan_rec.total_market_value:,.0f}", delta=f"{abs(plan_rec.target_shares):,.0f} shs")
                        eq2.metric("Expected Shortfall", f"${plan_rec.expected_total_shortfall_dollars:,.2f}", delta=f"{plan_rec.expected_total_shortfall_bps:.1f} bps")
                        eq3.metric("Execution Half-Life", f"{plan_rec.half_life_days:.2f} Days", delta=f"κ = {plan_rec.urgency_parameter_kappa:.4f}")
                        eq4.metric("Timing Risk (StdDev)", f"${plan_rec.shortfall_std_dev_dollars:,.2f}", delta=f"Vol: {plan_rec.daily_volatility_pct:.2f}%/D")

                        # Trajectory DataFrame table
                        st.dataframe(pd.DataFrame(plan_rec.trajectory_schedule), use_container_width=True)
                        st.caption(f"🛡️ **Optimal Execution Summary:** {plan_rec.summary}")
                        st.caption(f"🛡️ **Execution Audit Digest:** `{plan_rec.signature[:16]}...` (Verified: `{plan_rec.verify_integrity()}`)")
                    except Exception as ex:
                        st.info(f"Execution planning deferred: {ex}")
                else:
                    st.info("ℹ️ Insufficient price history (minimum 10 sessions) to evaluate optimal execution trajectory.")

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

            # Burke Ratio, Becker Pain Index & Ulcer Performance Scorecard
            from src.analytics.drawdown_analytics import evaluate_drawdown_profile
            p_dd_col = "adj_close" if "adj_close" in df_obs.columns else "close"
            if len(df_obs) >= 15:
                try:
                    dd_rec = evaluate_drawdown_profile(
                        ticker=selected_ticker,
                        prices=df_obs[p_dd_col],
                        risk_free_rate=0.04,
                    )
                    st.markdown("---")
                    st.subheader("🛡️ Path-Dependent Drawdown & Becker Pain Index Scorecard")
                    st.caption("Martin (1989) Ulcer Index, Burke (1994) Ratio, and Becker (2001) Pain Index quantifying drawdown severity and recovery duration.")
                    dd_col1, dd_col2, dd_col3, dd_col4, dd_col5 = st.columns(5)
                    dd_col1.metric("Pain Index", f"{dd_rec.pain_index_pct:.2f}%", delta=f"{dd_rec.max_drawdown_pct:.1f}% MaxDD")
                    dd_col2.metric("Pain Ratio", f"{dd_rec.pain_ratio:.2f}", delta="Excess/Pain")
                    dd_col3.metric("Burke Ratio", f"{dd_rec.burke_ratio:.2f}", delta="Quadratic Valleys")
                    dd_col4.metric("Ulcer Index (UI)", f"{dd_rec.ulcer_index_pct:.2f}%", delta=f"UPI: {dd_rec.ulcer_performance_index:.2f}")
                    dd_col5.metric("Longest Under Water", f"{dd_rec.max_drawdown_duration_sessions} sessions", delta=f"{dd_rec.num_discrete_drawdowns} valleys")
                    st.info(f"🛡️ **Path Drawdown Profile:** {dd_rec.summary} | **HMAC:** `{dd_rec.signature[:16]}...` (Verified: `{dd_rec.verify_integrity()}`)")
                except Exception as ex:
                    st.caption(f"Drawdown scorecard deferred: {ex}")

            # Bailey & López de Prado (2014) Deflated Sharpe Ratio & Overfitting Audit
            from src.evaluation.deflated_sharpe import evaluate_deflated_sharpe_profile
            if len(df_obs) >= 20:
                try:
                    p_ret = df_obs[p_dd_col].astype(float).pct_change().dropna()
                    dsr_rec = evaluate_deflated_sharpe_profile(
                        ticker=selected_ticker,
                        returns=p_ret,
                        num_trials=10,
                        benchmark_sharpe_ann=0.0,
                    )
                    st.markdown("---")
                    st.subheader("🎯 Bailey & López de Prado (2014) Deflated Sharpe Ratio")
                    st.caption("Multiple testing, non-normality (skew/kurtosis), and backtest overfitting audit against data snooping bias.")
                    dsr_c1, dsr_c2, dsr_c3, dsr_c4, dsr_c5 = st.columns(5)
                    dsr_c1.metric("Classical Sharpe", f"{dsr_rec.annualized_sharpe_ratio:.2f}", delta=f"Daily: {dsr_rec.daily_sharpe_ratio:.3f}")
                    dsr_c2.metric("Probabilistic SR (PSR)", f"{dsr_rec.probabilistic_sharpe_ratio * 100:.1f}%", help="Corrected for skewness and excess kurtosis")
                    dsr_c3.metric("Deflated SR (DSR)", f"{dsr_rec.deflated_sharpe_ratio * 100:.1f}%", delta=f"{dsr_rec.overfitting_status}")
                    dsr_c4.metric("Overfit Probability", f"{dsr_rec.overfitting_probability_pct:.1f}%", delta="1 - DSR")
                    mintr_disp = f"{dsr_rec.min_track_record_sessions:.0f} sessions" if not math.isinf(dsr_rec.min_track_record_sessions) else "Infinite (SR <= 0)"
                    dsr_c5.metric("Min Track Record", mintr_disp, delta=f"{dsr_rec.num_trials} trials")

                    if dsr_rec.is_statistically_significant:
                        st.success(f"🛡️ **Genuine Alpha Confirmed:** {dsr_rec.summary}")
                    else:
                        st.warning(f"⚠️ **Backtest Overfitting Warning:** {dsr_rec.summary}")

                    st.caption(f"🛡️ **DSR Audit Digest:** `{dsr_rec.signature[:16]}...` (Verified: `{dsr_rec.verify_integrity()}`)")
                except Exception as dsr_ex:
                    st.caption(f"Deflated Sharpe scorecard deferred: {dsr_ex}")

            # Marcos López de Prado (2018) Combinatorial Purged Cross-Validation (CPCV) & PBO
            from src.evaluation.cpcv import evaluate_cpcv_profile
            if len(df_obs) >= 80:
                try:
                    cpcv_rec = evaluate_cpcv_profile(
                        ticker=selected_ticker,
                        df=df_obs,
                        horizon_sessions=1,
                        num_groups=6,
                        test_groups=2,
                        embargo_pct=0.01,
                    )
                    st.markdown("---")
                    st.subheader("🔁 Marcos López de Prado (2018) Combinatorial Purged Cross-Validation (CPCV)")
                    st.caption("Boundary purging, post-test embargoing, combinatorial multiple OOS paths, and Probability of Backtest Overfitting (PBO).")
                    cp_c1, cp_c2, cp_c3, cp_c4, cp_c5 = st.columns(5)
                    cp_c1.metric("Combinatorial Folds", f"{cpcv_rec.num_folds} folds", delta=f"{cpcv_rec.num_paths} OOS Paths")
                    cp_c2.metric("MAE Outperformance", f"{cpcv_rec.mae_improvement_pct:+.2f}%", delta=f"{cpcv_rec.mean_model_mae:.4f} vs {cpcv_rec.mean_baseline_mae:.4f}")
                    cp_c3.metric("Directional Accuracy", f"{cpcv_rec.directional_accuracy_pct:.1f}%")
                    cp_c4.metric("OOS Sharpe", f"{cpcv_rec.mean_oos_sharpe_ratio:.2f}", delta=f"[{cpcv_rec.min_oos_sharpe_ratio:.2f}, {cpcv_rec.max_oos_sharpe_ratio:.2f}]")
                    cp_c5.metric("PBO Overfit Risk", f"{cpcv_rec.probability_of_backtest_overfitting_pct:.1f}%", delta=cpcv_rec.cpcv_status)

                    if cpcv_rec.cpcv_status == "ROBUST_LOW_OVERFITTING":
                        st.success(f"🛡️ **CPCV Low Overfitting Certified:** {cpcv_rec.summary}")
                    else:
                        st.warning(f"⚠️ **CPCV Overfitting Notice:** {cpcv_rec.summary}")

                    st.caption(f"🛡️ **CPCV Audit Digest:** `{cpcv_rec.signature[:16]}...` (Verified: `{cpcv_rec.verify_integrity()}`)")
                except Exception as cpcv_ex:
                    st.caption(f"CPCV validation scorecard deferred: {cpcv_ex}")

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
