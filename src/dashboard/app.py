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
        options=["gradient_boosting", "linear"],
        format_func=lambda x: "⚡ Gradient Boosting (Non-Linear)" if x == "gradient_boosting" else "📈 Ridge Regression (Linear)",
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
                st.caption(f"80% CI: `[{h1['lower_bound']:.2f} – {h1['upper_bound']:.2f}]`")

            with col_h5:
                st.metric(
                    label=f"5 Sessions Ahead (P_t+5: {h5['target_session_date']})",
                    value=f"{h5['predicted_price']:.2f} {exchange_cfg.currency}",
                    delta=f"{h5['expected_return_pct']:+.2f}%",
                )
                st.caption(f"80% CI: `[{h5['lower_bound']:.2f} – {h5['upper_bound']:.2f}]`")

            with col_h21:
                st.metric(
                    label=f"21 Sessions Ahead (P_t+21: {h21['target_session_date']})",
                    value=f"{h21['predicted_price']:.2f} {exchange_cfg.currency}",
                    delta=f"{h21['expected_return_pct']:+.2f}%",
                )
                st.caption(f"80% CI: `[{h21['lower_bound']:.2f} – {h21['upper_bound']:.2f}]`")

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

            fig_proj.update_layout(
                template="plotly_dark",
                xaxis_title="Trading Session Date",
                yaxis_title=f"Price ({exchange_cfg.currency})",
                margin=dict(l=20, r=20, t=30, b=20),
                hovermode="x unified",
            )
            st.plotly_chart(fig_proj, use_container_width=True)

            # Action: Register Forecast in Audit Log
            col_reg, _ = st.columns([2, 4])
            with col_reg:
                if st.button("📝 Register These Forecasts in Audit Log", use_container_width=True):
                    tracker = PredictionTracker()
                    for h_key, fc in forecasts.items():
                        tracker.record_prediction(
                            ticker=selected_ticker,
                            exchange=selected_exchange,
                            model_name=f"Ridge_Direct_{h_key}",
                            horizon_sessions=fc["horizon_sessions"],
                            origin_date=latest_date,
                            target_date=fc["target_session_date"],
                            predicted_price=fc["predicted_price"],
                            features_snapshot=fc.get("features_snapshot", {}),
                        )
                    st.success("All 3 horizon forecasts registered to prediction database!")
                    st.cache_data.clear()

    # =========================================================================
    # TAB 4: Prediction Audit Log & Reconciled Evaluation
    # =========================================================================
    with tabs[3]:
        st.header("Prediction Audit Log & Evaluation")
        st.caption("Auditable tracking finite state machine reconciling predictions with actual settlement prices.")

        # Action: Reconcile Pending Button
        col_act, col_filter = st.columns([2, 3])
        with col_act:
            if st.button("⚖️ Reconcile Past Predictions with Actual Prices", use_container_width=True):
                with st.spinner("Fetching confirmed exchange settlement prices and evaluating..."):
                    tracker = PredictionTracker()
                    res = tracker.evaluate_all_pending()
                    st.cache_data.clear()
                    st.toast(
                        f"Reconciled: {res['evaluated']} evaluated, {res['awaiting']} awaiting actuals, {res['pending']} future targets.",
                        icon="📊"
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
                    "Folds Evaluated": res["walk_forward_steps"],
                })

            st.dataframe(pd.DataFrame(val_rows), use_container_width=True, hide_index=True)

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
