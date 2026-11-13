"""Command-line interface for NextOmni Stock (nextomni-stock).

Allows syncing historical stock data, inspecting database state,
running quantitative analytics, training multi-horizon models,
generating forecast predictions, and monitoring live market feeds.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Optional

import pandas as pd

from src.config import INITIAL_TICKERS, SUPPORTED_EXCHANGES
from src.analytics.probability import calculate_directional_probability
from src.data.db import get_connection, get_observations, init_db, prune_raw_payloads, vacuum_database
from src.data.pipeline import DataPipeline
from src.security.validators import (
    SUPPORTED_MODEL_FAMILIES,
    assert_safe_model_path,
    sanitize_path_component,
    validate_exchange,
    validate_model_family,
    validate_session_date,
    validate_ticker,
)


def sync_command(args: argparse.Namespace) -> None:
    """Sync historical data for a given ticker or all default initial tickers."""
    pipeline = DataPipeline()

    if args.all:
        for exchange, tickers in INITIAL_TICKERS.items():
            for ticker in tickers:
                print(f"Syncing default ticker {ticker} ({exchange})...")
                res = pipeline.sync_stock(
                    ticker=ticker,
                    exchange=exchange,
                    start_date=args.start,
                    end_date=args.end,
                )
                print(
                    f"  -> {ticker}: Synced {res['sessions_synced']} sessions. "
                    f"Missing calendar sessions: {res['missing_calendar_sessions_count']}"
                )
        return

    if not args.ticker or not args.exchange:
        print("Error: --ticker and --exchange are required unless --all is specified.")
        sys.exit(1)

    ticker = validate_ticker(args.ticker)
    exchange = validate_exchange(args.exchange)
    start_date = validate_session_date(args.start)
    end_date = validate_session_date(args.end)

    print(f"Syncing {ticker} on {exchange} from {start_date} to {end_date}...")
    result = pipeline.sync_stock(
        ticker=ticker,
        exchange=exchange,
        start_date=start_date,
        end_date=end_date,
        company_name=args.name,
    )
    print("\nSync Result:")
    for k, v in result.items():
        print(f"  {k}: {v}")


def status_command(args: argparse.Namespace) -> None:
    """Display current database summary: stocks, observation counts, and date ranges."""
    init_db()
    conn = get_connection()

    stocks = conn.execute("SELECT * FROM stocks;").fetchall()
    if not stocks:
        print("No stocks registered in database yet. Run 'sync' to import data.")
        conn.close()
        return

    print("\nNextOmni Stock Database Status:")
    print(f"{'Ticker':<10} {'Exchange':<8} {'Currency':<8} {'Sessions':<10} {'Earliest':<12} {'Latest':<12}")
    print("-" * 65)

    for s in stocks:
        ticker = s["ticker"]
        stats = conn.execute(
            """
            SELECT COUNT(*) as cnt, MIN(session_date) as min_date, MAX(session_date) as max_date
            FROM market_observations
            WHERE ticker = ?;
            """,
            (ticker,),
        ).fetchone()

        cnt = stats["cnt"] if stats else 0
        min_date = stats["min_date"] if stats and stats["min_date"] else "N/A"
        max_date = stats["max_date"] if stats and stats["max_date"] else "N/A"

        print(
            f"{ticker:<10} {s['exchange']:<8} {s['currency']:<8} {cnt:<10} {min_date:<12} {max_date:<12}"
        )
    conn.close()


def analyze_command(args: argparse.Namespace) -> None:
    """Run exploratory data analysis and export interactive visual report."""
    from src.analytics.eda import export_eda_report

    ticker = validate_ticker(args.ticker)
    print(f"\nRunning Exploratory Data Analysis for {ticker}...")
    try:
        stats, report_path = export_eda_report(ticker=ticker)
    except Exception as e:
        print(f"Error during analysis: {e}")
        sys.exit(1)

    print("\n--- Summary Statistics ---")
    print(f"  Ticker:                 {stats.ticker} ({stats.exchange})")
    print(f"  Period:                 {stats.start_date} to {stats.end_date} ({stats.total_sessions} sessions)")
    print(f"  Price Range:            {stats.min_price} to {stats.max_price} (Mean: {stats.mean_price})")
    print(f"  First -> Latest Close:  {stats.first_price} -> {stats.last_price}")
    print(f"  Cumulative Return:      {stats.cumulative_return_pct}%")
    print(f"  Annualized Return:      {stats.annualized_return_pct}%")
    print(f"  Annualized Volatility:  {stats.annualized_volatility_pct}%")
    print(f"  Max Drawdown:           {stats.max_drawdown_pct}%")
    print(f"  Return Skewness:        {stats.skewness}")
    print(f"  Return Kurtosis:        {stats.kurtosis}")
    print(f"  Average Daily Volume:   {int(stats.avg_daily_volume):,}")

    try:
        from src.analytics.regime import detect_market_regime
        df_obs = get_observations(ticker)
        if len(df_obs) >= 20:
            regime_rec = detect_market_regime(df_obs)
            print("\n--- Market Regime Diagnostic ---")
            print(f"  Quantitative Regime:    {regime_rec.regime} [{regime_rec.regime_label}]")
            print(f"  Macro Sentiment Score:  {regime_rec.sentiment_score:+.3f} (Range [-1.0, +1.0])")
            print(f"  Volatility State:       {regime_rec.volatility_regime} ({regime_rec.rolling_vol_annualized:.1f}% Ann, {regime_rec.vol_percentile_1y:.1f}% 1Y Percentile)")
            print(f"  Trend State:            {regime_rec.trend_regime} (SMA50: {regime_rec.sma_50:.2f}" + (f", SMA200: {regime_rec.sma_200:.2f})" if regime_rec.sma_200 else ")"))
    except Exception:
        pass

    print(f"\nInteractive visual report generated at: {report_path.resolve()}\n")


def forecast_baseline_command(args: argparse.Namespace) -> None:
    """Generate persistence baseline forecast for next trading session and evaluate test performance."""
    from src.data.db import get_observations
    from src.models.baseline import PersistenceBaselineModel

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    model = PersistenceBaselineModel()

    # 1. Evaluate historical performance
    eval_result = model.evaluate(df, horizon_sessions=1, train_ratio=0.8)
    metrics = eval_result["metrics"]

    # 2. Generate future forecast
    future = model.generate_future_forecast(ticker, exchange, horizon_sessions=1)

    print(f"\n=======================================================")
    print(f"  NextOmni Stock: Next-Session Persistence Baseline")
    print(f"=======================================================")
    print(f"  Ticker:                 {future['ticker']} ({future['exchange']})")
    print(f"  Forecast Horizon:       {future['forecast_horizon']} (Next Trading Session)")
    print(f"  Forecast Origin Date:   {future['forecast_origin_date']}")
    print(f"  Origin Observed Price:  {future['forecast_origin_price']:.4f} {future['currency']}")
    print(f"  Target Session Date:    {future['target_session_date']}")
    print(f"  Predicted Price:        {future['predicted_price']:.4f} {future['currency']}")
    print(f"  Model Name / Version:   {future['model_name']} (v{future['model_version']})")
    print(f"  Forecast Status:        {future['status']}")

    print(f"\n--- Out-of-Sample Historical Test Performance (Horizon = 1 Session) ---")
    print(f"  Test Period:            {eval_result['test_period_start']} to {eval_result['test_period_end']} ({eval_result['test_sessions_count']} sessions)")
    print(f"  Mean Absolute Error:    {metrics['mae']} {future['currency']}")
    print(f"  Root Mean Squared Error:{metrics['rmse']} {future['currency']}")
    print(f"  Mean Absolute Pct Error:{metrics['mape_pct']}%")
    print(f"  Directional Accuracy:   {metrics['directional_accuracy_pct']}%")
    print(f"\nEducational Context:\n  {eval_result['educational_note']}\n")


def train_linear_command(args: argparse.Namespace) -> None:
    """Execute walk-forward validation and train linear regression model."""
    from src.data.db import get_observations
    from src.evaluation.validation import WalkForwardValidator
    from src.features.builder import build_feature_matrix
    from src.models.linear_model import LinearRegressionForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    currency = "USD" if df["exchange"].iloc[0] == "XNYS" else "MYR"
    validator = WalkForwardValidator(min_train_sessions=args.min_train, step_size=1)

    print(f"\nRunning Chronological Walk-Forward Validation for {ticker}...")
    result = validator.evaluate(df, horizon_sessions=1, alpha=args.alpha)

    m_mod = result["model_metrics"]
    m_base = result["baseline_metrics"]

    print("\n" + "=" * 75)
    print(f"  NextOmni Stock: Linear Regression vs Persistence Baseline (Walk-Forward)")
    print("=" * 75)
    print(f"  Ticker:               {result['ticker']} ({result['exchange']})")
    print(f"  Evaluation Window:    {result['evaluation_period_start']} to {result['evaluation_period_end']}")
    print(f"  Walk-Forward Steps:   {result['walk_forward_steps']} chronological out-of-sample steps")
    print("-" * 75)
    print(f"{'Metric':<25} {'Linear Model':<20} {'Persistence Baseline':<20} {'Improvement':<12}")
    print("-" * 75)
    mod_mape_str = f"{m_mod['mape_pct']:.2f}%"
    base_mape_str = f"{m_base['mape_pct']:.2f}%"
    mod_da_str = f"{m_mod['directional_accuracy_pct']:.2f}%"
    base_da_str = f"{m_base['directional_accuracy_pct']:.2f}%"

    print(f"{'MAE (' + currency + ')':<25} {m_mod['mae']:<20.4f} {m_base['mae']:<20.4f} {result['mae_improvement_pct']:>+8.2f}%")
    print(f"{'RMSE (' + currency + ')':<25} {m_mod['rmse']:<20.4f} {m_base['rmse']:<20.4f} {result['rmse_improvement_pct']:>+8.2f}%")
    print(f"{'MAPE (%)':<25} {mod_mape_str:<20} {base_mape_str:<20}")
    print(f"{'Directional Acc (%)':<25} {mod_da_str:<20} {base_da_str:<20}")
    print("-" * 75)

    print("\nFeature Coefficients (Interpretable Model Drivers):")
    for feat, coef in result["linear_model_coefficients"].items():
        print(f"  {feat:<22}: {coef:>+10.6f}")

    # Persist model artifact
    models_dir = Path("models").resolve()
    models_dir.mkdir(exist_ok=True)
    safe_name = f"linear_{sanitize_path_component(ticker)}.joblib"
    model_path = assert_safe_model_path(models_dir / safe_name, models_dir)

    # Train final production model on full feature set
    feature_df, feat_cols = build_feature_matrix(df, include_target=True, horizon_sessions=1)
    trainable = feature_df.dropna(subset=["target_price"])
    final_model = LinearRegressionForecaster(alpha=args.alpha)
    final_model.fit(
        X=trainable[feat_cols],
        y_target_price=trainable["target_price"],
        p_origin=trainable["adj_close"],
    )
    final_model.save(model_path)
    print(f"\nModel artifact saved to: {model_path.resolve()}\n")


def train_all_horizons_command(args: argparse.Namespace) -> None:
    """Train direct models for 1-session, 5-session, and 21-session horizons with walk-forward validation."""
    from src.data.db import get_observations
    from src.models.multi_horizon import MultiHorizonForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    model_family = validate_model_family(getattr(args, "model", "linear"))

    print(f"\n==========================================================================")
    print(f"  NextOmni Stock: Multi-Horizon Direct Model Training ({ticker}) [{model_family}]")
    print(f"==========================================================================")
    print("Training dedicated direct models for:")
    print("  • 1 Session  (P_{t+1})  - Next Trading Day")
    print("  • 5 Sessions (P_{t+5})  - Next Trading Week (1 Calendar Week)")
    print("  • 21 Sessions (P_{t+21}) - Next Trading Month (~1 Calendar Month)")
    print("\nRunning Walk-Forward Out-of-Sample Validation across all horizons...")

    forecaster = MultiHorizonForecaster(model_family=model_family, alpha=args.alpha)
    validation_results = forecaster.evaluate_walk_forward_all(df, min_train_sessions=args.min_train)

    print("\n" + "=" * 114)
    print(f"  Multi-Horizon Out-of-Sample Performance Comparison ({ticker}) [{model_family}]")
    print("=" * 114)
    print(f"{'Horizon':<15} {'Steps':<8} {'Model MAE':<12} {'Base MAE':<12} {'Improvement':<13} {'Dir Acc (%)':<13} {'Info Ratio':<12} {'Max Error':<12}")
    print("-" * 114)

    for h_key, res in validation_results.items():
        m_m = res["model_metrics"]
        b_m = res["baseline_metrics"]
        mae_imp = f"{res['mae_improvement_pct']:>+6.2f}%"
        da = f"{m_m['directional_accuracy_pct']:.2f}%"
        ir = f"{m_m.get('information_ratio', 0.0):.3f}"
        me = f"{m_m.get('max_error', 0.0):.4f}"
        print(
            f"{h_key:<15} {res['walk_forward_steps']:<8} "
            f"{m_m['mae']:<12.4f} {b_m['mae']:<12.4f} {mae_imp:<13} {da:<13} {ir:<12} {me:<12}"
        )
    print("-" * 114)

    # Train final models on full feature sets and save
    forecaster.fit(df)
    models_dir = Path("models").resolve()
    save_path = forecaster.save_models(models_dir, ticker)
    print(f"\nMulti-horizon models saved to: {save_path.resolve()}")

    # Generate current future predictions
    future_forecasts = forecaster.predict_all_horizons(df, ticker, exchange)
    print("\nCurrent Multi-Horizon Forecasts & 80% Confidence Bands (from latest observed session):")
    print(f"{'Horizon':<15} {'Target Date':<14} {'Origin Price':<14} {'Predicted Price':<16} {'80% Confidence Band':<24} {'Exp Return':<12}")
    print("-" * 98)
    for h_key, fc in future_forecasts.items():
        interval_str = f"[{fc['lower_bound']:.2f} – {fc['upper_bound']:.2f}]"
        print(
            f"{h_key:<15} {fc['target_session_date']:<14} "
            f"{fc['origin_price']:<14.4f} {fc['predicted_price']:<16.4f} {interval_str:<24} {fc['expected_return_pct']:>+6.2f}%"
        )
    print("-" * 98 + "\n")


def predict_command(args: argparse.Namespace) -> None:
    """Generate and record a live forecast prediction into SQLite."""
    import exchange_calendars as xcals
    from src.data.db import get_observations
    from src.evaluation.tracker import PredictionTracker
    from src.features.builder import build_feature_matrix
    from src.models.base_model import BaseForecaster
    from src.models.baseline import PersistenceBaselineModel
    from src.models.linear_model import LinearRegressionForecaster
    from src.models.multi_horizon import MultiHorizonForecaster

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = validate_exchange(args.exchange) if args.exchange else df["exchange"].iloc[0]
    horizon = int(args.horizon)
    tracker = PredictionTracker()

    latest_obs = df.iloc[-1]
    origin_session = str(latest_obs["session_date"])
    origin_price = float(latest_obs["adj_close"])

    # Resolve target session date using exchange calendar
    cal = xcals.get_calendar(exchange)
    origin_ts = pd.Timestamp(origin_session)
    sessions_ahead = cal.sessions_in_range(origin_ts, origin_ts + pd.Timedelta(days=horizon * 7))
    try:
        idx = list(sessions_ahead).index(origin_ts)
        target_ts = sessions_ahead[idx + horizon]
    except Exception:
        target_ts = cal.next_session(origin_ts)
    target_date_str = target_ts.strftime("%Y-%m-%d")

    safe_ticker = sanitize_path_component(ticker)
    models_dir = Path("models").resolve()
    multi_dir = models_dir / f"multi_horizon_{safe_ticker}"
    linear_file = models_dir / f"linear_{safe_ticker}.joblib"

    lower_bound = None
    upper_bound = None
    confidence_level = 0.80

    if args.model == "baseline":
        model_name = "PersistenceBaseline"
        model_version = "1.0.0"
        pred_price = origin_price
        repro_ref = "persistence_t"
    elif multi_dir.exists():
        multi_forecaster = MultiHorizonForecaster.load_models(models_dir, ticker)
        horizon_map = {1: "1_SESSION", 5: "5_SESSIONS", 21: "21_SESSIONS"}
        h_key = horizon_map.get(horizon, "1_SESSION")
        forecaster = multi_forecaster.models[h_key]
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_features = feature_df.iloc[-1][feat_cols].to_frame().T
        pred_arr, low_arr, up_arr = forecaster.predict_interval(
            latest_features, p_origin=origin_price, confidence_level=confidence_level
        )
        pred_price = float(pred_arr[0])
        lower_bound = float(low_arr[0])
        upper_bound = float(up_arr[0])
        model_name = f"{forecaster.model_name}_Direct_{h_key}"
        model_version = forecaster.model_version
        repro_ref = str(multi_dir / f"{h_key}.joblib")
    elif linear_file.exists():
        forecaster = LinearRegressionForecaster.load(linear_file)
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_features = feature_df.iloc[-1][feat_cols].to_frame().T
        pred_arr, low_arr, up_arr = forecaster.predict_interval(
            latest_features, p_origin=origin_price, confidence_level=confidence_level
        )
        pred_price = float(pred_arr[0])
        lower_bound = float(low_arr[0])
        upper_bound = float(up_arr[0])
        model_name = forecaster.model_name
        model_version = forecaster.model_version
        repro_ref = str(linear_file)
    else:
        print(f"Model artifact not found for {ticker}. Run 'train-all-horizons --ticker {ticker}' first.")
        sys.exit(1)

    record = tracker.record_prediction(
        ticker=ticker,
        exchange=exchange,
        forecast_origin_session=origin_session,
        origin_price=origin_price,
        horizon_sessions=horizon,
        target_session_date=target_date_str,
        predicted_price=pred_price,
        model_name=model_name,
        model_version=model_version,
        reproducibility_ref=repro_ref,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        confidence_level=confidence_level if lower_bound is not None else None,
    )

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Forecast Prediction Registered")
    print("=" * 65)
    print(f"  Prediction ID:          {record['prediction_id']}")
    print(f"  Ticker:                 {record['ticker']} ({record['exchange']})")
    print(f"  Forecast Horizon:       {record['horizon_type']}")
    print(f"  Origin Session Date:    {record['forecast_origin_session']}")
    print(f"  Origin Observed Price:  {record['origin_price']:.4f} {record['currency']}")
    print(f"  Target Session Date:    {record['target_session_date']}")
    print(f"  Predicted Price:        {record['predicted_price']:.4f} {record['currency']}")
    if record.get("lower_bound") is not None and record.get("upper_bound") is not None:
        print(f"  80% Confidence Band:    [{record['lower_bound']:.4f} – {record['upper_bound']:.4f}] {record['currency']}")
    exp_ret_pct = ((pred_price - origin_price) / origin_price) * 100.0
    res_std = getattr(forecaster, "residual_std_", 0.02)
    win_prob, conviction_label = calculate_directional_probability(
        expected_return=exp_ret_pct / 100.0,
        residual_std=res_std if res_std > 0 else 0.02,
    )
    print(f"  Expected Return:        {exp_ret_pct:>+6.2f}%")
    print(f"  Directional Win Prob:   {win_prob * 100:.1f}% [{conviction_label}]")
    print(f"  Model:                  {record['model_name']} (v{record['model_version']})")
    print(f"  Status:                 {record['status']}")

    if getattr(args, "explain", False) and hasattr(forecaster, "explain_forecast"):
        try:
            attributions = forecaster.explain_forecast(latest_features)
            if attributions:
                print("  " + "-" * 61)
                print("  Signal Attribution Breakdown (Additive Return Driver):")
                sorted_attrs = sorted(
                    [(k, v) for k, v in attributions.items() if k != "intercept"],
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )
                intercept_val = attributions.get("intercept", 0.0)
                print(f"    {'[Baseline Intercept]':<25} {intercept_val:>+8.4f} ({intercept_val*100:>+6.2f}%)")
                for feat, val in sorted_attrs:
                    direction = "▲" if val > 0 else ("▼" if val < 0 else " ")
                    print(f"    {feat:<25} {val:>+8.4f} ({val*100:>+6.2f}%) {direction}")
        except Exception:
            pass

    print("=" * 65 + "\n")


def evaluate_pending_command(args: argparse.Namespace) -> None:
    """Trigger idempotent evaluation workflow on pending predictions."""
    from src.evaluation.tracker import PredictionTracker

    tracker = PredictionTracker()
    print("\nRunning Idempotent Prediction Evaluation Workflow...")
    summary = tracker.evaluate_pending_predictions()

    print("\nEvaluation Summary:")
    print(f"  Newly Evaluated:        {summary['evaluated']}")
    print(f"  Awaiting Actual Data:   {summary['awaiting']}")
    print(f"  Pending Future Target:  {summary['pending']}")
    print(f"  Failed:                 {summary['failed']}\n")


def list_predictions_command(args: argparse.Namespace) -> None:
    """Display table of recorded predictions and evaluations."""
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if args.ticker else None
    df = get_predictions(status=args.status, ticker=ticker)
    if df.empty:
        print("\nNo prediction records found.\n")
        return

    print("\n" + "=" * 115)
    print(f"  NextOmni Stock: Predictions Audit Log ({len(df)} records)")
    print("=" * 115)
    print(
        f"{'Ticker':<8} {'Model':<22} {'Origin Date':<12} {'Target Date':<12} "
        f"{'Pred Price':<11} {'Actual Price':<13} {'Error':<10} {'Dir':<5} {'Status':<10}"
    )
    print("-" * 115)

    for _, r in df.iterrows():
        pred_p = f"{float(r['predicted_price']):.2f}"
        act_p = f"{float(r['actual_price']):.2f}" if pd.notna(r["actual_price"]) else "Pending"
        err = f"{float(r['percentage_error']):.2f}%" if pd.notna(r["percentage_error"]) else "-"
        dir_ok = "✓" if pd.notna(r["direction_correct"]) and bool(r["direction_correct"]) else ("✗" if pd.notna(r["direction_correct"]) else "-")
        status = str(r["status"])

        print(
            f"{r['ticker']:<8} {r['model_name'][:20]:<22} {r['forecast_origin_session']:<12} {r['target_session_date']:<12} "
            f"{pred_p:<11} {act_p:<13} {err:<10} {dir_ok:<5} {status:<10}"
        )
    print("-" * 115 + "\n")

    if getattr(args, "export", None):
        from src.security.audit import generate_audit_manifest

        export_path = Path(args.export)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        if export_path.suffix.lower() == ".json":
            df.to_json(export_path, orient="records", indent=2)
        else:
            df.to_csv(export_path, index=False)

        manifest_hash = generate_audit_manifest(df.to_dict("records"))
        manifest_file = export_path.with_suffix(export_path.suffix + ".manifest")
        manifest_file.write_text(f"SHA256_MANIFEST:{manifest_hash}\nRECORDS_COUNT:{len(df)}\n")

        print(f"Exported {len(df)} records to: {export_path}")
        print(f"Cryptographic Manifest:     {manifest_file} ({manifest_hash[:16]}...)\n")



def monitor_command(args: argparse.Namespace) -> None:
    """Monitor real-time quotes with session status and latency badges."""
    from src.data.monitor import MarketMonitorService

    ticker = validate_ticker(args.ticker) if args.ticker else None
    exchange = validate_exchange(args.exchange) if args.exchange else None

    service = MarketMonitorService()
    quotes = service.get_multi_quotes(
        tickers=[ticker] if ticker else None,
        exchange=exchange,
        force_refresh=args.refresh,
    )

    print("\nNextOmni Stock Market Monitor:")
    print(f"{'Ticker':<10} {'Exchange':<8} {'Price':<14} {'Change':<10} {'Change %':<10} {'Status':<14} {'Latency Badge':<16} {'Cached':<8}")
    print("-" * 96)
    for q in quotes:
        chg_str = f"{q.change:+.2f}"
        pct_str = f"{q.pct_change:+.2f}%"
        price_str = f"{q.last_price:.2f} {q.currency}"
        print(f"{q.ticker:<10} {q.exchange:<8} {price_str:<14} {chg_str:<10} {pct_str:<10} {q.session_status:<14} {q.latency_badge:<16} {str(q.cached):<8}")
    print("-" * 96)
    for q in quotes:
        print(f"[{q.ticker} ({q.exchange})]: Local Time: {q.local_time} | Status: {q.next_event}")
        if q.note:
            print(f"  Note: {q.note}")
    print()


def session_status_command(args: argparse.Namespace) -> None:
    """Display real-time operational status of supported exchanges."""
    from src.data.session_status import ExchangeSessionEngine

    exchange = validate_exchange(args.exchange) if args.exchange else None
    exchanges = [exchange] if exchange else list(SUPPORTED_EXCHANGES.keys())

    print("\nNextOmni Stock Exchange Session Status:")
    print(f"{'Exchange':<8} {'Name':<26} {'Status':<14} {'Open?':<8} {'Local Time':<30}")
    print("-" * 90)
    for exc in exchanges:
        status_info = ExchangeSessionEngine.get_status(exc)
        print(f"{status_info.exchange:<8} {status_info.name:<26} {status_info.status:<14} {str(status_info.is_open):<8} {status_info.local_time:<30}")
        if status_info.is_open and status_info.next_close:
            print(f"  -> Active Session: Scheduled Close at {status_info.next_close}")
        elif not status_info.is_open and status_info.next_open:
            print(f"  -> Closed Session: Scheduled Re-open at {status_info.next_open}")
        print(f"  -> Operational Details: {status_info.details}\n")


def daemon_reconcile_command(args: argparse.Namespace) -> None:
    """Run automated session-aware reconciliation daemon."""
    from src.evaluation.daemon import EvaluationDaemon

    daemon = EvaluationDaemon(poll_interval_seconds=args.interval)
    if args.once:
        print("\nExecuting single reconciliation daemon pass...")
        res = daemon.run_once()
        metrics = res["reconciliation_metrics"]
        print(f"Timestamp:              {res['timestamp']}")
        print(f"Newly Evaluated:        {metrics.get('evaluated', 0)}")
        print(f"Awaiting Actual Data:   {metrics.get('awaiting', 0)}")
        print(f"Pending Future Targets: {metrics.get('pending', 0)}")
        print(f"Failed Records:         {metrics.get('failed', 0)}\n")
    else:
        print(f"\nStarting NextOmni Stock Evaluation Daemon (Poll Interval: {args.interval}s)...")
        daemon.run_loop(interval_seconds=args.interval, max_iterations=args.max_iterations)


def verify_audit_command(args: argparse.Namespace) -> None:
    """Audit SQLite cryptographic SHA-256 signatures for tamper detection."""
    from src.data.db import verify_database_integrity

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Cryptographic Audit Trail Verification")
    print("=" * 65)
    report = verify_database_integrity()

    print(f"  Predictions Audited:    {report['predictions_audited']}")
    print(f"  Valid Predictions:      {report['predictions_valid']}")
    print(f"  Evaluations Audited:    {report['evaluations_audited']}")
    print(f"  Valid Evaluations:      {report['evaluations_valid']}")
    print("-" * 65)

    if report["is_intact"]:
        print("  ✓ AUDIT STATUS: 100% INTACT. Zero tampering detected.")
        print("=" * 65 + "\n")
    else:
        print("  ✗ AUDIT STATUS: TAMPERING DETECTED!")
        if report["tampered_predictions"]:
            print(f"    Tampered Prediction IDs: {report['tampered_predictions']}")
        if report["tampered_evaluations"]:
            print(f"    Tampered Evaluation IDs: {report['tampered_evaluations']}")
        print("=" * 65 + "\n")
        sys.exit(1)


def backup_db_command(args: argparse.Namespace) -> None:
    """Execute online atomic SQLite backup snapshot with cryptographic verification."""
    from src.data.db import backup_database

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Online Atomic Database Backup")
    print("=" * 65)
    try:
        dest = args.dest if getattr(args, "dest", None) else None
        saved_path = backup_database(destination_path=dest)
        print(f"  ✓ Snapshot Created & Verified: {saved_path}")
        print("  ✓ Cryptographic Integrity:      100% INTACT")
        print("=" * 65 + "\n")
    except Exception as e:
        print(f"  ✗ Backup Failed: {e}")
        print("=" * 65 + "\n")
        sys.exit(1)


def risk_report_command(args: argparse.Namespace) -> None:
    """Generate comprehensive Value-at-Risk (VaR) and Expected Shortfall (CVaR) report."""
    from src.analytics.risk import compute_multi_horizon_risk_profile, compute_path_risk_profile
    from src.data.db import get_observations

    ticker = validate_ticker(args.ticker)
    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    risk_summary = compute_multi_horizon_risk_profile(df, horizons=[1, 5, 21])
    if not risk_summary:
        print(f"Error: Insufficient data to compute risk distribution for {ticker}.")
        sys.exit(1)

    path_rec = compute_path_risk_profile(df, ticker=ticker)

    stress_results = []
    if getattr(args, "stress_test", False):
        from src.analytics.risk import simulate_macro_stress_scenarios
        stress_results = simulate_macro_stress_scenarios(df)

    if getattr(args, "json", False):
        import json
        payload = {
            "ticker": ticker,
            "multi_horizon_var": risk_summary,
            "path_risk_profile": path_rec.to_dict(),
            "stress_scenarios": stress_results if stress_results else None,
        }
        print(json.dumps(payload, indent=2))
        return

    print("\n" + "=" * 116)
    print(f"  NextOmni Stock: Multi-Horizon Quantitative Tail Risk Report ({ticker})")
    print("=" * 116)

    print(
        f"{'Horizon':<15} {'VaR 95% (G)':<14} {'VaR 95% (CF)':<14} {'VaR 99% (CF)':<14} "
        f"{'VaR 95% (Hist)':<16} {'CVaR 95% (Hist)':<17} {'CVaR 99% (Hist)':<17}"
    )
    print("-" * 116)

    for h_key, r in risk_summary.items():
        v95_g = f"{r['var_95_gaussian_pct']:.2f}%"
        v95_cf = f"{r['var_95_cornish_fisher_pct']:.2f}%"
        v99_cf = f"{r['var_99_cornish_fisher_pct']:.2f}%"
        v95_h = f"{r['var_95_historical_pct']:.2f}%"
        c95_h = f"{r['cvar_95_historical_pct']:.2f}%"
        c99_h = f"{r['cvar_99_historical_pct']:.2f}%"
        print(
            f"{h_key:<15} {v95_g:<14} {v95_cf:<14} {v99_cf:<14} "
            f"{v95_h:<16} {c95_h:<17} {c99_h:<17}"
        )
    print("-" * 116)
    latest_price = float(df["adj_close"].iloc[-1])
    h1 = risk_summary.get("1_SESSION", {})
    var95_1d = h1.get("var_95_cornish_fisher_pct", 0.0)
    cvar95_1d = h1.get("cvar_95_historical_pct", 0.0)
    print(f"Latest Settlement Price: {latest_price:.2f}")
    print(f"1-Day Maximum Expected Loss (95% Cornish-Fisher VaR): {var95_1d:.2f}% (~{latest_price * var95_1d / 100:.2f})")
    print(f"1-Day Extreme Tail Shortfall (95% Historical CVaR):     {cvar95_1d:.2f}% (~{latest_price * cvar95_1d / 100:.2f})")
    print("-" * 116)
    print("  Path-Dependent Cumulative Drawdown & CDaR Risk Profile:")
    print(f"  Current Price / High-Water Mark: {path_rec.current_price:.2f} / {path_rec.high_water_mark:.2f} (Current Drawdown: {path_rec.current_drawdown_pct:.2f}%)")
    print(f"  Historical Maximum Drawdown:     {path_rec.max_drawdown_pct:.2f}%")
    print(f"  Drawdown-at-Risk (95% DaR):      {path_rec.dar_95_pct:.2f}%")
    print(f"  Conditional DaR (95% CDaR):      {path_rec.cdar_95_pct:.2f}%  (Expected tail drawdown depth)")
    print(f"  Conditional DaR (99% CDaR):      {path_rec.cdar_99_pct:.2f}%")
    print(f"  Peter Martin's Ulcer Index:      {path_rec.ulcer_index_pct:.2f}%  (Quadratic depth & duration penalty)")
    print(f"  Underwater Sessions:             Current = {path_rec.current_underwater_sessions} sessions | Historical Max = {path_rec.max_underwater_sessions} sessions")

    if stress_results:
        print("\n  Historical Macroeconomic Stress-Testing Scenarios ($10,000 Portfolio Base):")
        print("  " + "-" * 112)
        print(
            f"  {'Scenario':<36} {'Severity':<14} {'Benchmark':<12} {'Ticker Beta':<14} "
            f"{'Est. Loss %':<14} {'Est. Loss ($)':<15} {'Post-Stress Price'}"
        )
        print("  " + "-" * 112)
        for s in stress_results:
            print(
                f"  {s['scenario_name']:<36} {s['severity']:<14} {s['benchmark_shock_pct']:>+6.1f}%     "
                f"{s['ticker_vol_beta']:>8.2f}       {s['estimated_loss_pct']:>8.2f}%     "
                f"${s['estimated_loss_dollars']:>9.2f}      {s['post_stress_price']:>10.2f}"
            )
        print("  " + "-" * 112)

    print("=" * 116 + "\n")


def vacuum_db_command(args: argparse.Namespace) -> None:
    """Run SQLite database integrity check, foreign key verification, and VACUUM defragmentation."""
    print("\nInitiating SQLite database verification and VACUUM compaction...")
    res = vacuum_database()
    kb_before = res["size_before_bytes"] / 1024.0
    kb_after = res["size_after_bytes"] / 1024.0
    kb_reclaimed = res["bytes_reclaimed"] / 1024.0
    print(f"  Foreign Key Consistency: {res['foreign_keys']}")
    print(f"  B-Tree Page Integrity:   {res['integrity']}")
    print(f"  Storage Before:          {kb_before:.2f} KB")
    print(f"  Storage After:           {kb_after:.2f} KB")
    print(f"  Space Reclaimed:         {kb_reclaimed:.2f} KB")
    print("VACUUM and integrity verification completed successfully.\n")


def prune_db_command(args: argparse.Namespace) -> None:
    """Safely prune expired raw provider payloads older than specified days."""
    days = args.days
    if days < 1:
        print("Error: --days must be greater than or equal to 1.")
        sys.exit(1)

    print(f"Pruning raw provider payloads older than {days} days from database...")
    deleted_count = prune_raw_payloads(retention_days=days)
    print(f"Pruning complete. Removed {deleted_count} expired payload record(s).")


def backtest_command(args: argparse.Namespace) -> None:
    """Run walk-forward strategy backtest and output risk-adjusted performance scorecard."""
    from src.analytics.backtester import run_forecast_backtest

    ticker = validate_ticker(args.ticker)
    model_family = validate_model_family(args.model)
    horizon = args.horizon
    fee_bps = float(args.fee_bps)

    df = get_observations(ticker)
    if df.empty:
        print(f"Error: No observations found for {ticker}. Run 'sync' first.")
        sys.exit(1)

    sizing_mode = getattr(args, "sizing", "fixed")
    target_vol = float(getattr(args, "target_vol", 0.15))
    max_leverage = float(getattr(args, "max_leverage", 1.0))

    print(f"\nRunning out-of-sample walk-forward strategy backtest for {ticker}...")
    print(f"  Model Family:      {model_family}")
    print(f"  Forecast Horizon:  {horizon} session(s)")
    print(f"  Position Sizing:   {sizing_mode.upper()} (Target Vol: {target_vol:.1%}, Max Leverage: {max_leverage:.1f}x)")
    print(f"  Transaction Fee:   {fee_bps} bps")
    print(f"  Min Train Burn-in: {args.min_train} sessions\n")

    res = run_forecast_backtest(
        df,
        horizon_sessions=horizon,
        fee_bps=fee_bps,
        min_train_sessions=args.min_train,
        model_family=model_family,
        sizing_mode=sizing_mode,
        target_annual_vol=target_vol,
        max_leverage=max_leverage,
    )
    m = res.metrics

    print("=" * 65)
    print("  NextOmni Stock: Out-of-Sample Backtest Scorecard")
    print("=" * 65)
    print(f"  Strategy Net Return:       {m.total_return_pct:>+8.2f}%")
    print(f"  Buy & Hold Benchmark:      {m.benchmark_return_pct:>+8.2f}%")
    print(f"  Annualized Strategy Return:{m.annualized_return_pct:>+8.2f}%")
    print(f"  Annualized Sharpe Ratio:   {m.annualized_sharpe:>8.3f}")
    print(f"  Annualized Sortino Ratio:  {m.annualized_sortino:>8.3f}")
    print(f"  Maximum Drawdown:          {m.max_drawdown_pct:>8.2f}%")
    print(f"  Win Rate (Active Trades):  {m.win_rate_pct:>8.2f}%")
    print(f"  Profit Factor:             {m.profit_factor:>8.2f}")
    print(f"  Total Trades (Turnover):   {m.total_trades:>8d}")
    print(f"  Average Portfolio Exposure:{m.avg_exposure:>8.2f}x")
    print(f"  Total Fee Drag Deducted:   {m.fee_drag_pct:>8.2f}%")
    print("=" * 65 + "\n")


def system_health_command(args: argparse.Namespace) -> None:
    """Run unified security and system health pre-flight audit."""
    import json
    from src.security.health import run_system_health_audit

    receipt = run_system_health_audit()

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 65)
    print("  NextOmni Stock: Unified Security & Health Audit")
    print("=" * 65)
    print(f"  Audit Timestamp:         {receipt.timestamp}")
    print(f"  Database Storage:        {receipt.database_path}")
    print(f"  B-Tree Integrity:        {receipt.database_integrity}")
    print(f"  Foreign Key Checks:      {receipt.foreign_keys_integrity}")
    print(f"  Prediction Audit Trail:  {receipt.prediction_audit} ({receipt.audited_predictions_count} preds, {receipt.audited_evaluations_count} evals)")
    print(f"  Model Signatures:        {receipt.model_signatures} ({receipt.verified_models_count} verified)")
    if receipt.tampered_models:
        print(f"  Tampered Models:         {receipt.tampered_models}")
    print(f"  Secret Masking Hygiene:  {receipt.secret_hygiene}")
    print(f"  Overall Health Status:   {'HEALTHY (PASS)' if receipt.is_healthy else 'ISSUES DETECTED (FAIL)'}")
    print(f"  Summary:                 {receipt.summary}")
    print("=" * 65 + "\n")


def screen_command(args: argparse.Namespace) -> None:
    """Run cross-sectional alpha universe screener and display ranked asset leaderboard."""
    import json
    from src.analytics.screener import screen_asset_universe

    exchange = validate_exchange(args.exchange) if args.exchange else None
    model_family = validate_model_family(args.model)
    horizon = args.horizon
    top_n = int(args.top) if getattr(args, "top", None) else None

    records = screen_asset_universe(
        exchange=exchange,
        horizon_sessions=horizon,
        model_family=model_family,
    )

    if getattr(args, "json", False):
        data = [r.to_dict() for r in records[:top_n]] if records else []
        print(json.dumps(data, indent=2))
        return

    if not records:
        print("\nNo qualified assets found with sufficient history for screening.\n")
        return

    display_records = records[:top_n] if top_n else records
    print("\n" + "=" * 105)
    print(f"  NextOmni Stock: Cross-Sectional Alpha Universe Screener ({len(display_records)} Assets)")
    print(f"  Horizon: {horizon} Session(s) | Model: {model_family.upper()} | Filter: {exchange or 'ALL EXCHANGES'}")
    print("=" * 105)
    print(
        f"{'Rank':<5} {'Ticker':<8} {'Exchange':<8} {'Price':<14} {'Exp Return':<12} "
        f"{'Win Prob':<10} {'Conviction':<16} {'95% CF-VaR':<12} {'Sharpe':<8}"
    )
    print("-" * 105)

    for r in display_records:
        price_str = f"{r.latest_price:.2f} {r.currency}"
        ret_str = f"{r.expected_return_pct:>+6.2f}%"
        prob_str = f"{r.directional_win_probability * 100:.1f}%"
        var_str = f"-{r.cornish_fisher_var_pct:.2f}%"
        print(
            f"{r.rank:<5} {r.ticker:<8} {r.exchange:<8} {price_str:<14} {ret_str:<12} "
            f"{prob_str:<10} {r.conviction_label:<16} {var_str:<12} {r.predicted_sharpe:>+6.2f}"
        )
    print("=" * 105 + "\n")


def ic_drift_command(args: argparse.Namespace) -> None:
    """Evaluate Information Coefficient drift and detect alpha decay on reconciled predictions."""
    import json
    from src.analytics.drift import calculate_information_coefficient_drift
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = f"{args.horizon}d" if getattr(args, "horizon", None) else None
    window = int(args.window) if getattr(args, "window", None) else 20

    preds_df = get_predictions(
        ticker=ticker,
        horizon_type=horizon,
        status="EVALUATED",
    )

    receipt = calculate_information_coefficient_drift(preds_df, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Information Coefficient & Alpha Drift Monitor")
    print("=" * 70)
    print(f"  Ticker Filter:       {receipt.ticker or 'ALL TICKERS'}")
    print(f"  Horizon Filter:      {receipt.horizon_type or 'ALL HORIZONS'}")
    print(f"  Sample Size:         {receipt.sample_size} evaluated predictions")
    print(f"  Mean IC (Pearson):   {receipt.mean_ic:>+7.4f}")
    print(f"  Mean Rank IC (Spear):{receipt.mean_rank_ic:>+7.4f}")
    print(f"  Rank IC Std Dev:     {receipt.ic_std:>7.4f}")
    print(f"  Information Ratio:   {receipt.information_ratio:>+7.3f}")
    print(f"  IC t-Statistic:      {receipt.t_statistic:>+7.3f}")
    print(f"  Trailing Rank IC:    {receipt.trailing_rank_ic:>+7.4f}")
    print(f"  Alpha Status:        {receipt.decay_status}")
    if receipt.decay_warning:
        print(f"  Alert Warning:       {receipt.decay_warning}")
def portfolio_command(args: argparse.Namespace) -> None:
    """Optimize multi-asset alpha portfolio using Ledoit-Wolf shrinkage covariance."""
    import json
    from src.analytics.portfolio import optimize_alpha_portfolio
    from src.analytics.screener import screen_asset_universe

    exchange = validate_exchange(args.exchange) if args.exchange else None
    horizon = args.horizon
    mode = args.mode
    max_weight = float(args.max_weight)

    records = screen_asset_universe(
        exchange=exchange,
        horizon_sessions=horizon,
        model_family="ensemble",
    )

    if not records:
        print("\nNo qualified assets found with sufficient history for portfolio optimization.\n")
        return

    receipt = optimize_alpha_portfolio(
        screener_records=records,
        mode=mode,
        max_weight=max_weight,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"  NextOmni Stock: Alpha Portfolio Optimizer ({receipt.mode.upper()})")
    print(f"  Horizon: {receipt.horizon_sessions} Sessions | Exchange: {exchange or 'ALL EXCHANGES'}")
    print("=" * 80)
    print(f"  Expected Annual Return:  {receipt.expected_annual_return_pct:>+7.2f}%")
    print(f"  Annualized Volatility:   {receipt.annualized_volatility_pct:>7.2f}%")
    print(f"  Sharpe Ratio (Rf=4%):    {receipt.sharpe_ratio:>+7.3f}")
    print(f"  Expected Shortfall (CVaR 95%): {receipt.cvar_95_pct:>6.2f}% daily")
    print(f"  Ledoit-Wolf Shrinkage:   {receipt.shrinkage_intensity:>7.4f}")
    print(f"  Effective Asset Count:   {receipt.effective_number_of_assets:>7.2f} / {len(receipt.tickers)}")
    print("-" * 80)
    print(f"  {'Ticker':<10} {'Weight %':<12} {'Risk Contribution %':<22}")
    print("-" * 80)
    for ticker, w in receipt.weights.items():
        rc = receipt.risk_contributions_pct.get(ticker, 0.0)
        print(f"  {ticker:<10} {w * 100:>8.2f}%    {rc:>12.2f}%")
    print("=" * 80 + "\n")


def calibration_report_command(args: argparse.Namespace) -> None:
    """Evaluate prediction interval empirical coverage and Winkler sharpness scores."""
    import json
    from src.analytics.calibration import evaluate_interval_calibration
    from src.data.db import get_predictions

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = f"{args.horizon}d" if getattr(args, "horizon", None) else None
    confidence = float(args.confidence) if getattr(args, "confidence", None) else 0.80

    preds_df = get_predictions(
        ticker=ticker,
        horizon_type=horizon,
        status="EVALUATED",
    )

    receipt = evaluate_interval_calibration(preds_df, confidence_level=confidence)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Prediction Interval Calibration & Winkler Report")
    print("=" * 70)
    print(f"  Ticker Filter:           {receipt.ticker or 'ALL TICKERS'}")
    print(f"  Horizon Filter:          {receipt.horizon_type or 'ALL HORIZONS'}")
    print(f"  Evaluated Samples:       {receipt.sample_size}")
    print(f"  Nominal Coverage Target: {receipt.nominal_confidence_level * 100:.1f}%")
    print(f"  Empirical Coverage:      {receipt.empirical_coverage_pct:>6.2f}%")
    print(f"  Coverage Error:          {receipt.coverage_error_pct:>6.2f}%")
    print(f"  Under-Breach Rate:       {receipt.under_breach_pct:>6.2f}%")
    print(f"  Over-Breach Rate:        {receipt.over_breach_pct:>6.2f}%")
    print(f"  Mean Interval Width:     {receipt.mean_interval_width:>6.2f}")
    print(f"  Mean Winkler Score:      {receipt.mean_winkler_score:>6.2f} (lower is sharper)")
    print(f"  Calibration Status:      {receipt.calibration_status}")
    print("=" * 70 + "\n")


def lead_lag_command(args: argparse.Namespace) -> None:
    """Evaluate cross-market lead-lag Granger causality between two equities."""
    import json
    from src.analytics.causality import evaluate_granger_causality
    from src.config import PRICE_COLUMN
    from src.data.db import get_observations

    source_ticker = validate_ticker(args.source)
    target_ticker = validate_ticker(args.target)
    max_lag = int(args.max_lag) if getattr(args, "max_lag", None) else 5

    df_src = get_observations(source_ticker)
    df_tgt = get_observations(target_ticker)

    if df_src.empty or df_tgt.empty:
        print(f"\nError: Insufficient data for {source_ticker} or {target_ticker}. Please synchronize both first.\n")
        return

    clean_src = df_src.sort_values("session_date").drop_duplicates(subset=["session_date"])
    clean_tgt = df_tgt.sort_values("session_date").drop_duplicates(subset=["session_date"])

    ret_src = clean_src[PRICE_COLUMN].astype(float).pct_change().dropna()
    ret_src.index = clean_src["session_date"].iloc[1:]

    ret_tgt = clean_tgt[PRICE_COLUMN].astype(float).pct_change().dropna()
    ret_tgt.index = clean_tgt["session_date"].iloc[1:]

    receipt = evaluate_granger_causality(
        source_series=ret_src,
        target_series=ret_tgt,
        max_lag=max_lag,
        source_ticker=source_ticker,
        target_ticker=target_ticker,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Cross-Market Lead-Lag Granger Causality Test")
    print("=" * 70)
    print(f"  Source (Leading Hypothesis): {receipt.source_ticker}")
    print(f"  Target (Lagging Hypothesis): {receipt.target_ticker}")
    print(f"  Aligned Sample Size:         {receipt.sample_size} sessions")
    print(f"  Optimal Evaluated Lag:       {receipt.optimal_lag} session(s)")
    print(f"  Fisher-Snedecor F-Statistic: {receipt.f_statistic:>7.4f}")
    print(f"  Granger Causality p-Value:   {receipt.p_value:>7.4f}")
    print(f"  Unrestricted R-Squared:      {receipt.r_squared_unrestricted:>7.4f}")
    print(f"  Restricted (AR) R-Squared:   {receipt.r_squared_restricted:>7.4f}")
    print(f"  Causality Verdict:           {'CAUSAL (PASS, p < 0.05)' if receipt.is_causal else 'NOT SIGNIFICANT (FAIL)'}")
    print(f"  Summary:                     {receipt.summary}")
    print("=" * 70 + "\n")


def cointegrate_command(args: argparse.Namespace) -> None:
    """Evaluate Engle-Granger cointegration and mean-reversion spread dynamics between two assets."""
    import json
    from src.analytics.causality import evaluate_engle_granger_cointegration
    from src.config import PRICE_COLUMN
    from src.data.db import get_observations

    ticker1 = validate_ticker(args.ticker1)
    ticker2 = validate_ticker(args.ticker2)

    df1 = get_observations(ticker1)
    df2 = get_observations(ticker2)

    if df1.empty or df2.empty:
        print(f"\nError: Insufficient data for {ticker1} or {ticker2}. Please synchronize both first.\n")
        return

    clean_1 = df1.sort_values("session_date").drop_duplicates(subset=["session_date"])
    clean_2 = df2.sort_values("session_date").drop_duplicates(subset=["session_date"])

    p1 = clean_1.set_index("session_date")[PRICE_COLUMN].astype(float)
    p2 = clean_2.set_index("session_date")[PRICE_COLUMN].astype(float)

    receipt = evaluate_engle_granger_cointegration(p1, p2, ticker_1=ticker1, ticker_2=ticker2)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Engle-Granger Cointegration & Mean-Reversion Engine")
    print("=" * 75)
    print(f"  Asset 1 (Dependent):       {receipt.ticker_1}")
    print(f"  Asset 2 (Independent):     {receipt.ticker_2}")
    print(f"  Aligned Sample Size:       {receipt.sample_size} sessions")
    print(f"  Hedge Ratio (Beta):        {receipt.hedge_ratio_beta:>7.4f}")
    print(f"  Intercept (Alpha):         {receipt.intercept_alpha:>7.4f}")
    print(f"  ADF Spread t-Statistic:    {receipt.adf_statistic:>7.4f}")
    print(f"  Engle-Granger 5% Critical: {receipt.critical_values['5%']:>7.2f}")
    print(f"  p-Value (Approx):          {receipt.p_value_approx:>7.4f}")
    print(f"  Cointegration Status:      {'COINTEGRATED (PASS)' if receipt.is_cointegrated else 'NOT COINTEGRATED (FAIL)'}")
    print(f"  Mean-Reversion Half-Life:  {receipt.half_life_sessions:.1f} sessions")
    print(f"  Current Spread Value:      {receipt.current_spread:>+7.4f}")
    print(f"  Current Spread Z-Score:    {receipt.current_spread_zscore:>+7.2f} sigma")
    print(f"  Summary:                   {receipt.summary}")
    print("=" * 75 + "\n")


def liquidity_report_command(args: argparse.Namespace) -> None:
    """Evaluate market microstructure liquidity and price impact diagnostics."""
    import json
    from src.data.db import get_observations
    from src.features.microstructure import evaluate_liquidity_profile

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 21

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_liquidity_profile(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 70)
    print("  NextOmni Stock: Market Microstructure & Liquidity Profile")
    print("=" * 70)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_size} sessions")
    print(f"  Latest Corwin-Schultz Sprd:{receipt.latest_corwin_schultz_spread_pct:>6.2f}%")
    print(f"  Mean Corwin-Schultz Spread:{receipt.mean_corwin_schultz_spread_pct:>6.2f}%")
    print(f"  Latest Amihud Illiquidity: {receipt.latest_amihud_illiq:>7.4f}")
    print(f"  Mean Amihud Illiquidity:   {receipt.mean_amihud_illiq:>7.4f}")
    print(f"  Mean Daily Turnover:       ${receipt.mean_dollar_volume:>13,.0f}")
    print(f"  Liquidity Regime:          {receipt.liquidity_regime}")
    print(f"  Summary:                   {receipt.summary}")
    print("=" * 70 + "\n")


def volatility_report_command(args: argparse.Namespace) -> None:
    """Evaluate multi-estimator realized volatility and extreme value diagnostics."""
    import json
    from src.data.db import get_observations
    from src.analytics.volatility import evaluate_volatility_profile

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 20

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_volatility_profile(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Multi-Estimator Realized Volatility Diagnostics")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Estimation Window:         {receipt.window} sessions")
    print("-" * 75)
    print(f"  Close-to-Close (Standard): {receipt.close_to_close_ann_pct:>6.2f}% ann.  (Efficiency:  1.0x baseline)")
    print(f"  Parkinson (1980 High-Low): {receipt.parkinson_ann_pct:>6.2f}% ann.  (Efficiency: ~5.2x baseline)")
    print(f"  Garman-Klass (1980 OHLC):  {receipt.garman_klass_ann_pct:>6.2f}% ann.  (Efficiency: ~7.4x baseline)")
    print(f"  Rogers-Satchell (1991):    {receipt.rogers_satchell_ann_pct:>6.2f}% ann.  (Drift-independent)")
    print(f"  Yang-Zhang (2000 Optimal): {receipt.yang_zhang_ann_pct:>6.2f}% ann.  (Efficiency: ~14.0x baseline)")
    print(f"  RiskMetrics EWMA (λ=0.94): {receipt.ewma_ann_pct:>6.2f}% ann.  (Exponential clustering)")
    print("-" * 75)
    print(f"  Volatility Ratio (YZ / CC):{receipt.volatility_ratio_yz_cc:>7.3f}")
    print(f"  Regime Classification:     {receipt.regime_signal}")
    if receipt.term_structure_yang_zhang:
        ts_str = ", ".join(f"{k}: {v:.1f}%" for k, v in receipt.term_structure_yang_zhang.items())
        print(f"  Term Structure (YZ):       {ts_str}")
    print("=" * 75 + "\n")


def crps_eval_command(args: argparse.Namespace) -> None:
    """Evaluate probabilistic forecasts using proper scoring rules (CRPS, Pinball Loss, Winkler Score)."""
    import json
    from src.data.db import get_predictions
    from src.evaluation.probabilistic import evaluate_probabilistic_forecasts

    ticker = validate_ticker(args.ticker) if getattr(args, "ticker", None) else None
    horizon = int(args.horizon) if getattr(args, "horizon", None) else None
    model_family = getattr(args, "model", None)

    preds_df = get_predictions(ticker=ticker)
    if preds_df.empty:
        msg = f"No predictions recorded in database" + (f" for {ticker}" if ticker else "")
        print(f"\nError: {msg}.\n")
        return

    receipt = evaluate_probabilistic_forecasts(
        preds_df,
        horizon_sessions=horizon,
        model_family=model_family,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Probabilistic Forecast Quality & CRPS Scorecard")
    print("=" * 75)
    if receipt.ticker:
        print(f"  Ticker Filter:             {receipt.ticker}")
    if receipt.horizon_sessions:
        print(f"  Horizon Filter:            {receipt.horizon_sessions} sessions")
    if receipt.model_family:
        print(f"  Model Family Filter:       {receipt.model_family}")
    print(f"  Sample Reconciled Folds:   {receipt.sample_size}")
    print("-" * 75)
    print(f"  Continuous Ranked Prob (CRPS): {receipt.mean_crps:>8.4f}  (Proper continuous scoring rule)")
    print(f"  Mean Absolute Error (MAE):     {receipt.mean_mae:>8.4f}  (Deterministic point baseline)")
    print(f"  Quantile Pinball Loss (10%):   {receipt.mean_pinball_loss_q10:>8.4f}  (Downside tail risk loss)")
    print(f"  Quantile Pinball Loss (50%):   {receipt.mean_pinball_loss_q50:>8.4f}  (Median check loss)")
    print(f"  Quantile Pinball Loss (90%):   {receipt.mean_pinball_loss_q90:>8.4f}  (Upside tail opportunity loss)")
    print(f"  Winkler Sharpness (80% CI):    {receipt.mean_winkler_score_80:>8.4f}  (Width + breach penalty)")
    print(f"  Empirical 80% Coverage:        {receipt.empirical_coverage_80_pct:>7.2f}%")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")


def changepoint_detect_command(args: argparse.Namespace) -> None:
    """Detect structural breaks and regime change points using Bayesian Online Change-Point Detection."""
    import json
    from src.data.db import get_observations
    from src.analytics.changepoints import detect_bayesian_changepoints

    ticker = validate_ticker(args.ticker)
    hazard = int(args.hazard) if getattr(args, "hazard", None) else 100

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = detect_bayesian_changepoints(df_obs, ticker=ticker, hazard_lambda=hazard)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Bayesian Online Change-Point & Structural Break Audit")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Hazard Timescale (λ):      {receipt.hazard_lambda} sessions")
    print("-" * 75)
    print(f"  Current Run-Length:        {receipt.current_run_length} sessions since last break")
    print(f"  Change-Point Probability:  {receipt.current_change_point_prob * 100:>6.2f}%")
    print(f"  Regime Stability Status:   {receipt.regime_stability_status}")
    if receipt.detected_break_dates:
        breaks_str = ", ".join(receipt.detected_break_dates[-5:])
        print(f"  Recent Structural Breaks:  {breaks_str}")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")


def toxicity_report_command(args: argparse.Namespace) -> None:
    """Audit order flow toxicity and Volume-Synchronized Probability of Toxicity (VPIN)."""
    import json
    from src.data.db import get_observations
    from src.features.toxicity import evaluate_flow_toxicity

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 20

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observations found for {ticker}. Please synchronize data first.\n")
        return

    receipt = evaluate_flow_toxicity(df_obs, ticker=ticker, window=window)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 75)
    print("  NextOmni Stock: Order Flow Toxicity & VPIN Microstructure Audit")
    print("=" * 75)
    print(f"  Ticker Symbol:             {receipt.ticker}")
    print(f"  Sample Size:               {receipt.sample_sessions} sessions")
    print(f"  Observation Window:        {receipt.window} sessions")
    print("-" * 75)
    print(f"  Latest VPIN:               {receipt.latest_vpin * 100:>6.2f}%")
    print(f"  Trailing Mean VPIN:        {receipt.mean_vpin * 100:>6.2f}%")
    print(f"  Historical Peak VPIN:      {receipt.max_vpin * 100:>6.2f}%")
    print("-" * 75)
    print(f"  Latest Volume Initiation:  {receipt.latest_buyer_volume_pct:.1f}% Buyer / {receipt.latest_seller_volume_pct:.1f}% Seller")
    print(f"  Toxicity Flow Regime:      {receipt.toxicity_regime}")
    print("-" * 75)
    print(f"  Summary: {receipt.summary}")
    print("=" * 75 + "\n")


def tail_dependence_command(args: argparse.Namespace) -> None:
    """Evaluate multi-asset asymmetric lower-tail dependence and joint crash fragility."""
    import json
    from src.analytics.copula import evaluate_universe_tail_dependence
    from src.security.validators import validate_exchange

    exchange = validate_exchange(args.exchange) if getattr(args, "exchange", None) else None
    threshold = float(args.threshold) if getattr(args, "threshold", None) else 0.05

    receipt = evaluate_universe_tail_dependence(exchange=exchange, threshold=threshold)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asymmetric Lower-Tail Dependence & Joint Crash Fragility")
    print("=" * 80)
    print(f"  Assets Evaluated:          {len(receipt.tickers)} securities ({', '.join(receipt.tickers)})")
    print(f"  Overlapping Sessions:      {receipt.sample_sessions} sessions")
    print(f"  Tail Quantile Cutoff (q):  {receipt.tail_threshold * 100:.1f}% extreme tail")
    print("-" * 80)
    print(f"  Market Joint Crash Regime: {receipt.market_joint_crash_regime}")
    p1, p2, val = receipt.highest_risk_pair
    print(f"  Highest Co-Crash Pair:     {p1} & {p2} (λ_L = {val * 100:.1f}%)")
    print("-" * 80)
    print("  Systemic Joint Fragility Scores (Mean Off-Diagonal Co-Crash Risk):")
    for t, score in receipt.systemic_fragility_scores.items():
        print(f"    - {t:<10}: {score * 100:>5.1f}%")
    print("-" * 80)
    print("  Lower-Tail Dependence Matrix (λ_L):")
    header = "  " + f"{'':<8}" + "".join(f"{t:>10}" for t in receipt.tickers)
    print(header)
    for t1 in receipt.tickers:
        row_str = "  " + f"{t1:<8}" + "".join(f"{receipt.tail_dependence_matrix.loc[t1, t2] * 100:>9.1f}%" for t2 in receipt.tickers)
        print(row_str)
    print("-" * 80)
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def factor_attribution_command(args: argparse.Namespace) -> None:
    """Evaluate multi-factor risk attribution and style factor exposures."""
    import json
    import math
    from src.analytics.factor_attribution import evaluate_factor_attribution
    from src.security.validators import validate_exchange, validate_ticker

    ticker = validate_ticker(args.ticker)
    exchange = validate_exchange(args.exchange) if getattr(args, "exchange", None) else "XNYS"

    receipt = evaluate_factor_attribution(ticker=ticker, exchange=exchange)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Multi-Factor Risk Attribution & Style Exposure Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker} ({receipt.exchange})")
    print(f"  Sample History:            {receipt.sample_sessions} aligned sessions")
    print(f"  Regression Fit (R²):       {receipt.r_squared * 100:.1f}% (Adj R²: {receipt.adj_r_squared * 100:.1f}%)")
    print(f"  Annualized Alpha (α):      {receipt.annualized_alpha_pct:+.2f}% (t={receipt.alpha_t_stat:.2f}, p={receipt.alpha_p_value:.3f})")
    print("-" * 80)
    print("  Euler Variance Decomposition:")
    sys_vol = receipt.total_volatility_ann_pct * math.sqrt(receipt.systemic_risk_share_pct / 100.0) if receipt.total_volatility_ann_pct > 0 else 0.0
    print(f"    - Systematic Factor Risk: {receipt.systemic_risk_share_pct:>5.1f}% (Ann Vol: {sys_vol:.1f}%)")
    print(f"    - Idiosyncratic Specific: {receipt.specific_risk_share_pct:>5.1f}% (Ann Vol: {receipt.idiosyncratic_volatility_ann_pct:.1f}%)")
    print(f"    - Total Asset Volatility: {receipt.total_volatility_ann_pct:>5.1f}% annualized")
    print("-" * 80)
    print("  Style Factor Loadings (Newey-West HAC Robust Estimator):")
    header = f"    {'Factor':<10} {'Beta (β)':>10} {'t-stat':>10} {'p-value':>10} {'% Risk Contrib':>16}"
    print(header)
    print("    " + "-" * 60)
    for f, beta in receipt.factor_betas.items():
        t_stat = receipt.factor_t_stats.get(f, 0.0)
        p_val = receipt.factor_p_values.get(f, 1.0)
        p_pct = receipt.factor_risk_contributions_pct.get(f, 0.0)
        print(f"    {f:<10} {beta:>10.4f} {t_stat:>10.2f} {p_val:>10.3f} {p_pct:>15.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def wavelet_denoise_command(args: argparse.Namespace) -> None:
    """Evaluate multiscale wavelet multiresolution decomposition and adaptive denoising."""
    import json
    from src.data.db import get_observations
    from src.features.wavelet import evaluate_wavelet_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    levels = int(args.levels) if getattr(args, "levels", None) else 3

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    receipt = evaluate_wavelet_profile(df=df_obs, ticker=ticker, levels=levels)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Multiscale Wavelet Denoising & Energy Scalogram Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker}")
    print(f"  Sample History:            {receipt.sample_sessions} sessions")
    print(f"  MODWT Decomposition Depth: {receipt.levels} octave levels (Haar shift-invariant)")
    print(f"  Estimated Noise (σ_hat):   ${receipt.noise_sigma_hat:.4f}")
    print(f"  Universal Threshold (λ):   ${receipt.universal_threshold:.4f} (VisuShrink)")
    print(f"  Signal-to-Noise Ratio:     {receipt.snr_db:.2f} dB")
    print("-" * 80)
    print(f"  Latest Settlement Price:   ${receipt.latest_raw_price:.2f}")
    print(f"  Denoised Jump-Clean Trend: ${receipt.latest_denoised_price:.2f}")
    print(f"  Noise Spike Deviation:     {receipt.price_deviation_pct:+.2f}%")
    print(f"  Timescale Energy Regime:   {receipt.dominant_scale_regime}")
    print("-" * 80)
    print("  Multiscale Wavelet Energy Scalogram:")
    for scale_lbl, share_pct in receipt.energy_shares_pct.items():
        print(f"    - {scale_lbl:<20}: {share_pct:>6.2f}% of detail variance")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def momentum_crash_report_command(args: argparse.Namespace) -> None:
    """Evaluate Barroso & Santa-Clara risk-managed momentum and Daniel & Moskowitz crash hazard."""
    import json
    from src.analytics.risk_managed_momentum import evaluate_momentum_risk_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    target_vol = float(args.target_vol) if getattr(args, "target_vol", None) else 0.12
    max_leverage = float(args.max_leverage) if getattr(args, "max_leverage", None) else 2.0

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    receipt = evaluate_momentum_risk_profile(
        df=df_obs,
        ticker=ticker,
        target_vol=target_vol,
        max_leverage=max_leverage,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Risk-Managed Momentum & Crash-Risk Dampened Trend Engine")
    print("=" * 80)
    print(f"  Security / Ticker:         {receipt.ticker}")
    print(f"  Sample History:            {receipt.sample_sessions} sessions")
    print(f"  Target Volatility (σ*):    {receipt.target_volatility_pct:.1f}% annualized")
    print(f"  Max Leverage Cap:          {receipt.max_leverage:.2f}x")
    print(f"  Current Dynamic Leverage:  {receipt.current_leverage_weight:.2f}x")
    print("-" * 80)
    print(f"  Crash Hazard Probability:  {receipt.current_crash_hazard_pct:.1f}%")
    print(f"  Crash Risk State:          {receipt.crash_risk_regime}")
    print("-" * 80)
    print("  Comparative Strategy Performance (Barroso & Santa-Clara 2015):")
    print(f"    {'Metric':<26} {'Unmanaged Momentum':>20} {'Risk-Managed (RMM)':>20}")
    print("    " + "-" * 68)
    print(f"    {'Annualized Return':<26} {receipt.raw_momentum_ann_return_pct:>+19.2f}% {receipt.rmm_ann_return_pct:>+19.2f}%")
    print(f"    {'Annualized Volatility':<26} {receipt.raw_momentum_volatility_pct:>19.2f}% {receipt.rmm_volatility_pct:>19.2f}%")
    print(f"    {'Sharpe Ratio (Rf=4%)':<26} {receipt.raw_momentum_sharpe:>+19.2f} {receipt.rmm_sharpe:>+19.2f}")
    print(f"    {'Max Drawdown':<26} {receipt.raw_momentum_max_drawdown_pct:>19.2f}% {receipt.rmm_max_drawdown_pct:>19.2f}%")
    print(f"    {'Return Skewness':<26} {receipt.raw_momentum_skewness:>+19.2f} {receipt.rmm_skewness:>+19.2f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):       {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fracdiff_command(args: argparse.Namespace) -> None:
    """Evaluate memory-preserving fractional differentiation and minimal stationary order d*."""
    import json
    from src.data.db import get_observations
    from src.features.fracdiff import evaluate_fracdiff_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    threshold = float(args.threshold) if getattr(args, "threshold", None) else 1e-4
    target_d = float(args.d) if getattr(args, "d", None) is not None else None
    auto_d = getattr(args, "auto_d", True) if target_d is None else False

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_fracdiff_profile(
        ticker=ticker,
        prices=prices,
        threshold=threshold,
        auto_d=auto_d,
        target_d=target_d,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Memory-Preserving Fractional Differentiation Engine (FFD)")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Truncation Threshold (tau):   {receipt.threshold:.1e}")
    print(f"  Filter Window Length (l):     {receipt.window_length} sessions")
    print("-" * 80)
    print(f"  Selected Order (d):           {receipt.selected_d:.3f}")
    print(f"  Optimal Stationary Order (d*):{receipt.optimal_d:.3f}")
    print(f"  ADF Test Statistic:           {receipt.adf_stat:.4f} (5% Critical: -2.863)")
    print(f"  ADF p-value:                  {receipt.adf_pvalue:.4f}")
    print(f"  Stationary (p < 0.05):        {'YES' if receipt.is_stationary else 'NO'}")
    print("-" * 80)
    print("  Memory Preservation vs Integer Differencing:")
    print(f"    Memory Retained Correlation rho(P, P^(d)):      {receipt.retained_memory_corr:>+8.4f} ({receipt.retained_memory_corr * 100:.1f}%)")
    print(f"    Integer Diff Memory Correlation rho(P, Delta P): {receipt.raw_to_diff1_corr:>+8.4f} ({receipt.raw_to_diff1_corr * 100:.1f}%)")
    if receipt.is_stationary:
        print(f"    Alpha Retention Advantage:                     {receipt.retained_memory_corr - receipt.raw_to_diff1_corr:>+8.4f} correlation units preserved!")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print("=" * 80 + "\n")


def kalman_spread_command(args: argparse.Namespace) -> None:
    """Evaluate Online Kalman Filter dynamic hedge ratio and adaptive spread tracking."""
    import json
    from src.analytics.kalman import evaluate_kalman_pairs_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    y_ticker = validate_ticker(args.y_ticker)
    x_ticker = validate_ticker(args.x_ticker)
    delta = float(args.delta) if getattr(args, "delta", None) else 1e-4

    df_y = get_observations(y_ticker)
    df_x = get_observations(x_ticker)

    if df_y.empty or df_x.empty:
        missing = y_ticker if df_y.empty else x_ticker
        print(f"\nError: No observation history found for {missing}. Please synchronize first.\n")
        return

    receipt = evaluate_kalman_pairs_profile(
        y_ticker=y_ticker,
        x_ticker=x_ticker,
        df_y=df_y,
        df_x=df_x,
        delta=delta,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Online Kalman Filter Dynamic Hedge Ratio & Spread Tracker")
    print("=" * 80)
    print(f"  Paired Assets:                Y={receipt.y_ticker} vs X={receipt.x_ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} aligned sessions")
    print(f"  Process Noise (delta):        {receipt.delta_process_noise:.1e}")
    print("-" * 80)
    print(f"  Latest State Intercept (alpha_t):   {receipt.latest_alpha:>+10.4f}")
    print(f"  Latest Dynamic Hedge Ratio (beta_t):{receipt.latest_beta:>10.4f}")
    print(f"  Static Benchmark OLS Beta:          {receipt.ols_static_beta:>10.4f}")
    print(f"  Hedge Ratio Parameter Drift:        {receipt.beta_drift_pct:>+9.2f}% vs static OLS")
    print("-" * 80)
    print(f"  Current Spread (e_t):               {receipt.latest_spread:>+10.4f}")
    print(f"  Adaptive Spread Volatility:         {receipt.latest_spread_std:>10.4f}")
    print(f"  Spread Z-Score:                     {receipt.latest_zscore:>+10.2f}")
    print(f"  Mean-Reversion Half-Life:           {receipt.mean_reversion_half_life:.1f} sessions")
    print(f"  Current Arbitrage Signal:           {receipt.current_signal}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print("=" * 80 + "\n")


def evt_report_command(args: argparse.Namespace) -> None:
    """Evaluate Extreme Value Theory (EVT) Peaks-Over-Threshold tail risk and GPD parameters."""
    import json
    from src.analytics.evt import evaluate_evt_tail_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    threshold_q = float(args.threshold_quantile) if getattr(args, "threshold_quantile", None) else 0.95

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_evt_tail_profile(
        ticker=ticker,
        prices=prices,
        threshold_quantile=threshold_q,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Extreme Value Theory (EVT) Peaks-Over-Threshold Tail Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Threshold Quantile:           {receipt.threshold_quantile * 100:.1f}% percentile")
    print(f"  Loss Cutoff Threshold (u):    {receipt.threshold_u:.2f}% daily loss")
    print(f"  Tail Exceedances (N_u):       {receipt.exceedances_count} sessions")
    print("-" * 80)
    print("  Generalized Pareto Distribution (GPD) Parameters:")
    print(f"    Shape Parameter (xi / ξ):   {receipt.gpd_shape_xi:>+10.4f} ({receipt.tail_fatness_regime})")
    print(f"    Scale Parameter (beta / β): {receipt.gpd_scale_beta:>10.6f}")
    print("-" * 80)
    print("  Comparative Tail Risk Measures (Daily Loss %):")
    print(f"    {'Risk Model':<30} {'95% VaR':>12} {'99% VaR':>12} {'99% CVaR (ES)':>15}")
    print("    " + "-" * 71)
    print(f"    {'EVT Peaks-Over-Threshold (GPD)':<30} {receipt.var_95_pct:>11.2f}% {receipt.var_99_pct:>11.2f}% {receipt.cvar_99_pct:>14.2f}%")
    print(f"    {'Cornish-Fisher Expansion':<30} {'—':>12} {receipt.cornish_fisher_var_99_pct:>11.2f}% {'—':>15}")
    print(f"    {'Standard Gaussian':<30} {'—':>12} {receipt.gaussian_var_99_pct:>11.2f}% {'—':>15}")
    print("-" * 80)
    print(f"  Gaussian Tail Underestimation: {receipt.tail_risk_underestimation_pct:>+8.1f}% severe under-prediction")
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def entropy_report_command(args: argparse.Namespace) -> None:
    """Evaluate Market Information Entropy, Permutation Complexity, and Predictability Index."""
    import json
    from src.analytics.entropy import evaluate_market_entropy_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(args.window) if getattr(args, "window", None) else 60
    embedding_dim = int(args.embedding_dim) if getattr(args, "embedding_dim", None) else 3

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_market_entropy_profile(
        ticker=ticker,
        prices=prices,
        window=window,
        m=embedding_dim,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Market Information Entropy & Ordinal Complexity Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Analysis Window:              {receipt.window} sessions")
    print(f"  Embedding Dimension (m):      {receipt.embedding_dim}")
    print("-" * 80)
    print("  Information Entropy & Complexity Measures:")
    print(f"    Permutation Entropy (H_P):    {receipt.permutation_entropy:>10.4f} (0 = Deterministic, 1 = White Noise)")
    print(f"    Sample Entropy (SampEn):      {receipt.sample_entropy:>10.4f}")
    print(f"    Mean Trailing Entropy:        {receipt.mean_permutation_entropy:>10.4f}")
    print(f"    Market Predictability Index:  {receipt.predictability_index_pct:>9.2f}%")
    print("-" * 80)
    print(f"  Information Regime State:     {receipt.entropy_regime}")
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def garch_volatility_command(args: argparse.Namespace) -> None:
    """Evaluate Asymmetric GJR-GARCH(1, 1) volatility forecasting and forward term structure."""
    import json
    from src.analytics.garch import evaluate_garch_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    model_choice = str(getattr(args, "model", "gjr")).lower().strip()
    asymmetric = model_choice != "standard"

    df_obs = get_observations(ticker)
    if df_obs.empty:
        print(f"\nError: No observation history found for {ticker}. Please synchronize first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col].dropna()

    receipt = evaluate_garch_profile(
        ticker=ticker,
        prices=prices,
        asymmetric=asymmetric,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asymmetric GJR-GARCH(1, 1) Volatility Forecasting Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions")
    print(f"  Specification Model:          {receipt.model_type}")
    print("-" * 80)
    print("  QMLE Conditional Variance Parameters:")
    print(f"    Baseline Intercept (omega / ω):       {receipt.omega:.8e}")
    print(f"    Symmetric Shock (alpha / α):          {receipt.alpha:>10.4f}")
    print(f"    Asymmetric Leverage (gamma / γ):      {receipt.gamma_asymmetry:>10.4f}")
    print(f"    Persistence GARCH (beta / β):         {receipt.beta:>10.4f}")
    print(f"    Variance Persistence (phi / φ):       {receipt.persistence_phi:>10.4f}")
    print(f"    Leverage Asymmetry Ratio (α+γ)/α:     {receipt.leverage_asymmetry_ratio:>9.2f}x bad news volatility amplification")
    print("-" * 80)
    print("  Annualized Volatility Levels & Forward Term Structure:")
    print(f"    Current Conditional Volatility:       {receipt.current_conditional_volatility_ann_pct:>9.2f}% annualized")
    print(f"    Unconditional Long-Term Volatility:   {receipt.unconditional_volatility_ann_pct:>9.2f}% annualized")
    print("    Forward Term Structure Projections:")
    for h_label, h_val in receipt.term_structure_forecast_ann_pct.items():
        print(f"      {h_label:<20} {h_val:>10.2f}% annualized")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def resampled_frontier_command(args: argparse.Namespace) -> None:
    """Evaluate Michaud (1998) Resampled Efficient Frontier and compare with Classical Markowitz."""
    import json
    from src.analytics.resampled_portfolio import evaluate_resampled_frontier_profile

    raw_tickers = args.tickers
    if isinstance(raw_tickers, str):
        ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
    else:
        ticker_list = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]

    simulations = int(getattr(args, "simulations", 100))
    points = int(getattr(args, "points", 15))
    rf = float(getattr(args, "risk_free_rate", 0.04))

    receipt = evaluate_resampled_frontier_profile(
        tickers=ticker_list,
        num_simulations=simulations,
        num_frontier_points=points,
        risk_free_rate=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Michaud (1998) Resampled Efficiency & Robust Frontier")
    print("=" * 80)
    print(f"  Asset Universe:               {', '.join(receipt.tickers)}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Monte Carlo Simulations:      {receipt.num_simulations} paths")
    print(f"  Frontier Discrete Points:     {receipt.num_frontier_points}")
    print(f"  Shrinkage Intensity:          {receipt.shrinkage_intensity:.4f}")
    print(f"  Risk-Free Rate:               {receipt.risk_free_rate_pct:.2f}%")
    print("-" * 80)
    print("  Maximum Sharpe Ratio Allocations Comparison:")
    print("    Metric                       Classical Markowitz         Michaud Resampled")
    print(f"    Expected Annual Return:      {receipt.classical_max_sharpe.expected_return_ann_pct:>18.2f}%      {receipt.resampled_max_sharpe.expected_return_ann_pct:>18.2f}%")
    print(f"    Annualized Volatility:       {receipt.classical_max_sharpe.volatility_ann_pct:>18.2f}%      {receipt.resampled_max_sharpe.volatility_ann_pct:>18.2f}%")
    print(f"    Sharpe Ratio:                {receipt.classical_max_sharpe.sharpe_ratio:>19.3f}       {receipt.resampled_max_sharpe.sharpe_ratio:>19.3f}")
    print(f"    Effective Assets (N_eff):    {receipt.classical_max_sharpe.effective_assets:>19.2f}       {receipt.resampled_max_sharpe.effective_assets:>19.2f}")
    print(f"    Michaud Diversification Gain: {receipt.michaud_diversification_gain_pct:>+18.1f}%")
    print("-" * 80)
    print("  Asset Weights Comparison at Maximum Sharpe:")
    print("    Ticker            Classical Markowitz         Michaud Resampled         Dispersion (Std)")
    for t in receipt.tickers:
        cw = receipt.classical_max_sharpe.weights.get(t, 0.0) * 100.0
        rw = receipt.resampled_max_sharpe.weights.get(t, 0.0) * 100.0
        sd = receipt.weight_stability_dispersion.get(t, 0.0) * 100.0
        print(f"    {t:<14}    {cw:>16.1f}%         {rw:>16.1f}%         {sd:>15.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def mrmr_select_command(args: argparse.Namespace) -> None:
    """Execute Minimum Redundancy Maximum Relevance (mRMR) quantitative feature selection."""
    import json
    from src.features.mrmr import evaluate_mrmr_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    horizon = int(getattr(args, "horizon", 1))
    n_features = int(getattr(args, "n_features", 5))
    criterion = str(getattr(args, "criterion", "MID")).upper()

    receipt = evaluate_mrmr_profile(
        ticker=ticker,
        horizon_sessions=horizon,
        n_features=n_features,
        criterion=criterion,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Minimum Redundancy Maximum Relevance (mRMR) Feature Selection")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Forecast Horizon:             {receipt.horizon_sessions} trading session(s)")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Optimization Criterion:       {receipt.criterion}")
    print(f"  Candidate Signals:            {receipt.num_candidates} features")
    print(f"  Selected Signals:             {receipt.num_selected} features")
    print("-" * 80)
    print("  Greedy Forward Selection History:")
    print("    Step   Selected Feature          Relevance I(f;y)    Redundancy R(f;S)    mRMR Score")
    for step in receipt.selection_history:
        print(
            f"    {step['step']:<6} {step['feature']:<25} {step['relevance']:>16.4f} "
            f"{step['redundancy']:>20.4f} {step['mrmr_score']:>13.4f}"
        )
    print("-" * 80)
    print("  Collinearity & Redundancy Suppression:")
    print(f"    Mean Candidate Collinearity:  {receipt.mean_candidate_correlation:.4f}")
    print(f"    Mean Selected Collinearity:   {receipt.mean_selected_correlation:.4f}")
    print(f"    Multicollinearity Reduction:  {receipt.redundancy_reduction_pct:+.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def price_impact_command(args: argparse.Namespace) -> None:
    """Evaluate Kyle's (1985) Informed Trading Lambda and market price impact."""
    import json
    from src.analytics.price_impact import evaluate_price_impact_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 60))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 20:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_price_impact_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        volumes=df_obs["volume"],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Kyle's (1985) Informed Trading Lambda & Price Impact Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample History:               {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Adverse Selection Regime:     {receipt.adverse_selection_regime}")
    print("-" * 80)
    print("  Kyle's Lambda & Market Depth:")
    print(f"    Kyle's Lambda (Percent):      {receipt.kyle_lambda_pct_per_million:.6f}% per $1,000,000 net order flow")
    print(f"    Kyle's Lambda (Dollar/Shares): ${receipt.kyle_lambda_abs_per_k_shares:.6f} per 1,000 shares")
    print(f"    Market Depth (1% Price Move): ${receipt.market_depth_millions:.2f} Million order flow required")
    print(f"    Huber Regression R-Squared:   {receipt.r_squared:.4f}")
    print("-" * 80)
    print("  Trading Volume & Liquidity Baseline:")
    print(f"    21-Day Average Daily Volume:  {receipt.average_daily_volume_shares:,.0f} shares")
    print(f"    Average Daily Dollar Volume:  ${receipt.average_daily_dollar_volume:,.2f}")
    print("-" * 80)
    print("  Square-Root Law Projected Execution Price Impact:")
    print(f"    1% ADV Order Size:            {receipt.sqrt_impact_1pct_adv_bps:>8.2f} basis points (bps)")
    print(f"    5% ADV Order Size:            {receipt.sqrt_impact_5pct_adv_bps:>8.2f} basis points (bps)")
    print(f"    10% ADV Order Size:           {receipt.sqrt_impact_10pct_adv_bps:>8.2f} basis points (bps)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def tail_shape_command(args: argparse.Namespace) -> None:
    """Evaluate robust quantile tail shape, Moors kurtosis, and crash asymmetry."""
    import json
    from src.analytics.tail_shape import evaluate_tail_shape_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 120))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 30:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_tail_shape_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Robust Quantile Tail Shape & Moors Kurtosis Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Tail Shape Regime:            {receipt.tail_shape_regime}")
    print("-" * 80)
    print("  Quantile Fat-Tailedness & Kurtosis:")
    print(f"    Moors Quantile Kurtosis:      {receipt.moors_quantile_kurtosis:.4f} (Normal Benchmark: 1.2330)")
    print(f"    Moors Excess Kurtosis:        {receipt.moors_excess_kurtosis:+.4f}")
    print(f"    Hogg Tail Weight Indicator:   {receipt.hogg_tail_weight:.4f} (Normal Benchmark: 2.5800)")
    print("-" * 80)
    print("  Quantile Asymmetry & Crash Susceptibility:")
    print(f"    Bowley Quantile Skewness:     {receipt.bowley_quantile_skewness:+.4f} ([-1.0, +1.0])")
    print(f"    Tail Asymmetry Ratio (L/R):   {receipt.tail_asymmetry_ratio:.3f}x downside vs upside tail depth")
    print(f"    Left 1% Tail Depth (Loss):    {receipt.left_tail_depth_1pct:.2f}%")
    print(f"    Right 1% Tail Depth (Gain):   {receipt.right_tail_depth_1pct:.2f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def pain_index_command(args: argparse.Namespace) -> None:
    """Evaluate Burke Ratio, Becker Pain Index, and Ulcer Performance Index."""
    import json
    from src.analytics.drawdown_analytics import evaluate_drawdown_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rf = float(getattr(args, "risk_free_rate", 0.04))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 15:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_drawdown_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        risk_free_rate=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Burke Ratio, Becker Pain Index & Drawdown Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions")
    print(f"  Annualized Return:            {receipt.annualized_return_pct:+.2f}% (Risk-Free: {receipt.risk_free_rate_pct:.2f}%)")
    print(f"  Maximum Drawdown:             {receipt.max_drawdown_pct:.2f}% (Current: {receipt.current_drawdown_pct:.2f}%)")
    print(f"  Under-Water State:            {'ACTIVE DRAWDOWN' if receipt.is_in_drawdown else 'AT HIGH-WATER MARK'}")
    print(f"  Longest Drawdown Duration:    {receipt.max_drawdown_duration_sessions} trading sessions")
    print(f"  Discrete Drawdown Episodes:   {receipt.num_discrete_drawdowns} valleys")
    print("-" * 80)
    print("  Path-Dependent Pain & Drawdown Metrics:")
    print(f"    Becker (2001) Pain Index:     {receipt.pain_index_pct:>8.2f}% average time under water")
    print(f"    Becker Pain Ratio:            {receipt.pain_ratio:>8.3f} excess return per unit average drawdown")
    print(f"    Burke (1994) Ratio:           {receipt.burke_ratio:>8.3f} quadratic penalty on drawdown valleys")
    print(f"    Martin (1989) Ulcer Index:    {receipt.ulcer_index_pct:>8.2f}% RMS drawdown")
    print(f"    Ulcer Performance Index (UPI):{receipt.ulcer_performance_index:>8.3f} Martin ratio")
    print(f"    Calmar Ratio (Ann/MaxDD):     {receipt.calmar_ratio:>8.3f}")
    print(f"    Sterling Ratio (Ann/Top5 DD): {receipt.sterling_ratio:>8.3f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def markov_regime_command(args: argparse.Namespace) -> None:
    """Evaluate Hamilton (1989) two-state Markov-switching dynamic regime model."""
    import json
    from src.analytics.markov_regime import evaluate_markov_regime_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 120))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 40:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_markov_regime_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hamilton (1989) Two-State Markov-Switching Regime Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Window: {receipt.rolling_window}D)")
    print(f"  Current Inferred Regime:      {receipt.current_regime}")
    print(f"  Crisis State Probability:     {receipt.current_crisis_probability * 100:>6.2f}% (P(S_T = 1))")
    print(f"  Model Log-Likelihood:         {receipt.log_likelihood:>8.2f}")
    print("-" * 80)
    print("  State Characteristics (Annualized Moments):")
    print(f"    State 0 (Low Vol Expansion):  {receipt.state_0_volatility_ann_pct:>6.2f}% Ann Vol, {receipt.state_0_mean_ann_pct:>+6.2f}% Ann Mean")
    print(f"    State 1 (High Vol Crisis):    {receipt.state_1_volatility_ann_pct:>6.2f}% Ann Vol, {receipt.state_1_mean_ann_pct:>+6.2f}% Ann Mean ({receipt.volatility_ratio_state1_to_state0:.1f}x Vol Amplification)")
    print("-" * 80)
    print("  Regime Transition Probabilities & Expected Persistence:")
    print(f"    P(Stay in State 0 | S0):      {receipt.p00_persistence:>6.4f} (Expected Duration: {receipt.expected_low_vol_duration_sessions:.1f} sessions)")
    print(f"    P(Stay in State 1 | S1):      {receipt.p11_persistence:>6.4f} (Expected Duration: {receipt.expected_high_vol_duration_sessions:.1f} sessions)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def rmt_denoise_command(args: argparse.Namespace) -> None:
    """Evaluate Marchenko-Pastur (1967) RMT covariance matrix denoising and detrending."""
    import json
    from src.analytics.rmt_denoiser import evaluate_rmt_denoising_profile

    raw_tickers = getattr(args, "tickers", None)
    ticker_list = None
    if raw_tickers:
        if isinstance(raw_tickers, str):
            ticker_list = [t.strip().upper() for t in raw_tickers.split(",") if t.strip()]
        else:
            ticker_list = [str(t).strip().upper() for t in raw_tickers if str(t).strip()]

    exchange = getattr(args, "exchange", None)
    lookback = int(getattr(args, "lookback", 252))
    method = getattr(args, "method", "constant_residual")
    detrend = bool(getattr(args, "detrend", False))

    receipt, raw_cov, denoised_cov = evaluate_rmt_denoising_profile(
        tickers=ticker_list,
        exchange=exchange,
        lookback_sessions=lookback,
        method=method,
        detrend=detrend,
    )

    if getattr(args, "json", False):
        out = {
            "receipt": receipt.to_dict(),
            "raw_covariance": raw_cov.to_dict(),
            "denoised_covariance": denoised_cov.to_dict(),
        }
        print(json.dumps(out, indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Marchenko-Pastur (1967) RMT Covariance Denoising Engine")
    print("=" * 80)
    print(f"  Asset Universe:               {', '.join(receipt.tickers)}")
    print(f"  Sample Dimensions:            {receipt.num_assets} assets x {receipt.sample_sessions} sessions (q = {receipt.q_ratio:.2f})")
    print(f"  Denoising Method:             {receipt.method} (Detrend Market Mode: {receipt.is_detrended})")
    print(f"  Fitted Noise Variance:        {receipt.fitted_noise_variance:.4f}")
    print(f"  Marchenko-Pastur Null Band:   [{receipt.noise_floor_lambda_minus:.4f}, {receipt.noise_cutoff_lambda_plus:.4f}] (Lambda+ Cutoff)")
    print("-" * 80)
    print(f"  Signal Eigenvalues (> L+):    {receipt.num_signal_eigenvalues} modes ({receipt.signal_variance_ratio_pct:.1f}% Total Variance)")
    print(f"  Noise Eigenvalues (<= L+):    {receipt.num_noise_eigenvalues} modes (Shrunk / Constant Residual)")
    print("-" * 80)
    print("  Condition Number (Matrix Stability):")
    print(f"    Raw Empirical Matrix:       {receipt.condition_number_raw:>10.2f}")
    print(f"    RMT Denoised Matrix:        {receipt.condition_number_denoised:>10.2f}")
    print(f"    Stability Improvement:      {receipt.condition_number_improvement_ratio:>10.1f}x reduction")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def jump_test_command(args: argparse.Namespace) -> None:
    """Evaluate Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation and Jump Decomposition."""
    import json
    from src.analytics.jump_diffusion import evaluate_jump_diffusion_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 21))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    receipt = evaluate_jump_diffusion_profile(
        ticker=ticker,
        prices=df_obs[price_col],
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barndorff-Nielsen & Shephard (2004, 2006) Jump Detection Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.window}D)")
    print(f"  Jump Regime:                  {receipt.jump_regime}")
    print(f"  Statistically Significant:    {'YES (p < 0.05)' if receipt.has_significant_jump else 'NO (Continuous Brownian Path)'}")
    print("-" * 80)
    print("  Quadratic Variation Volatility Decomposition (Annualized):")
    print(f"    Total Realized Volatility:  {receipt.annualized_total_vol_pct:>6.2f}% Ann (RV = {receipt.realized_variance:.6f})")
    print(f"    Continuous Path Volatility: {receipt.annualized_continuous_vol_pct:>6.2f}% Ann (BV = {receipt.bipower_variation:.6f})")
    print(f"    Jump Component Volatility:  {receipt.annualized_jump_vol_pct:>6.2f}% Ann (JV = {receipt.jump_variation:.6f})")
    print(f"    Jump Variance Proportion:   {receipt.jump_contribution_ratio * 100:>6.2f}% of Total Quadratic Variation")
    print("-" * 80)
    print("  BNS Asymptotic Hypothesis Test Statistics:")
    print(f"    BNS Z-Score:                {receipt.bns_z_statistic:>8.4f} (Standard Normal N(0, 1))")
    print(f"    One-Sided p-value:          {receipt.p_value:>8.4f}")
    print(f"    Tri-power Quarticity (TQ):  {receipt.tripower_quarticity:>12.8f}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def deflated_sharpe_command(args: argparse.Namespace) -> None:
    """Evaluate Bailey & López de Prado (2014) Probabilistic & Deflated Sharpe Ratio."""
    import json
    import math
    from src.evaluation.deflated_sharpe import evaluate_deflated_sharpe_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    trials = int(getattr(args, "trials", 10))
    benchmark_sr = float(getattr(args, "benchmark_sr", 0.0))
    rf = float(getattr(args, "risk_free_rate", 0.04))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 15:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    ret = df_obs[price_col].astype(float).pct_change().dropna()

    receipt = evaluate_deflated_sharpe_profile(
        ticker=ticker,
        returns=ret,
        num_trials=trials,
        benchmark_sharpe_ann=benchmark_sr,
        risk_free_rate_ann=rf,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bailey & López de Prado (2014) Deflated Sharpe Ratio Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Trials Evaluated: {receipt.num_trials})")
    print(f"  Overfitting Status:           {receipt.overfitting_status}")
    print(f"  Statistically Significant:    {'YES (DSR >= 95%)' if receipt.is_statistically_significant else 'NO (Likely Backtest Overfitting)'}")
    print("-" * 80)
    print("  Performance & Statistical Moments:")
    print(f"    Classical Annualized Sharpe:  {receipt.annualized_sharpe_ratio:>8.2f} (Daily SR: {receipt.daily_sharpe_ratio:.4f})")
    print(f"    Skewness (Crash Asymmetry):   {receipt.skewness:>8.2f} (gamma_3)")
    print(f"    Kurtosis (Tail Heaviness):    {receipt.kurtosis:>8.2f} (gamma_4, Normal = 3.0)")
    print("-" * 80)
    print("  Probabilistic & Deflated Sharpe Ratios:")
    print(f"    Benchmark Hurdle Sharpe:      {receipt.benchmark_sharpe_ann:>8.2f} Ann")
    print(f"    Probabilistic Sharpe (PSR):   {receipt.probabilistic_sharpe_ratio * 100:>7.1f}% (Corrected for Skew & Kurtosis)")
    print(f"    Expected Max SR from Trials:  {receipt.expected_max_sharpe_ann:>8.2f} Ann (Null E[max_N SR] under Data Snooping)")
    print(f"    Deflated Sharpe Ratio (DSR):  {receipt.deflated_sharpe_ratio * 100:>7.1f}% (Multiple Testing & Snooping Adjusted)")
    print(f"    Backtest Overfit Probability: {receipt.overfitting_probability_pct:>7.1f}% (1 - DSR)")
    mintr_str = f"{receipt.min_track_record_sessions:.0f} sessions" if not math.isinf(receipt.min_track_record_sessions) else "Infinite (SR <= Benchmark)"
    print(f"    Minimum Track Record Length:  {mintr_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def effective_spread_command(args: argparse.Namespace) -> None:
    """Evaluate Richard Roll (1984) Effective Bid-Ask Spread and Composite Execution Friction."""
    import json
    import math
    from src.analytics.effective_spread import evaluate_effective_spread_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    window = int(getattr(args, "window", 21))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]
    high = df_obs["high"] if "high" in df_obs.columns else None
    low = df_obs["low"] if "low" in df_obs.columns else None
    volumes = df_obs["volume"] if "volume" in df_obs.columns else None

    receipt = evaluate_effective_spread_profile(
        ticker=ticker,
        prices=prices,
        high=high,
        low=low,
        volumes=volumes,
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Richard Roll (1984) Effective Bid-Ask Spread Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.window}D)")
    print(f"  Mean Price Level:             ${receipt.mean_price:.2f}")
    print(f"  Liquidity Friction Tier:      {receipt.liquidity_tier}")
    print(f"  Autocovariance Regime:        {'Positive (Trending / Momentum)' if receipt.has_positive_autocovariance else 'Negative (Dealer Order Bounce)'}")
    print("-" * 80)
    print("  Implicit Bid-Ask Spread Metrics:")
    print(f"    Roll Dollar Spread:         ${receipt.roll_spread_dollars:.4f}")
    print(f"    Roll Percentage Spread:      {receipt.roll_spread_pct:>6.2f}%")
    print(f"    Signed Continuous Spread:    {receipt.signed_roll_spread_pct:>6.2f}% (Chu, Stivers & Truong)")
    print(f"    Corwin-Schultz Spread:       {receipt.corwin_schultz_spread_pct:>6.2f}% (High-Low 2-Day Range)")
    print(f"    Serial Autocovariance:       {receipt.serial_autocovariance:>10.6f} (gamma_1)")
    print("-" * 80)
    print("  Composite Trading Execution Friction:")
    print(f"    Inferred Round-Trip Cost:    {receipt.composite_round_trip_bps:>6.1f} bps ({receipt.composite_round_trip_bps / 100.0:.2f}%)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def cpcv_validate_command(args: argparse.Namespace) -> None:
    """Evaluate Marcos López de Prado (2018) Combinatorial Purged & Embargoed Cross-Validation."""
    import json
    import math
    from src.evaluation.cpcv import evaluate_cpcv_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    horizon = int(getattr(args, "horizon", 1))
    groups = int(getattr(args, "groups", 6))
    test_groups = int(getattr(args, "test_groups", 2))
    embargo = float(getattr(args, "embargo", 0.01))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < 80:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    receipt = evaluate_cpcv_profile(
        ticker=ticker,
        df=df_obs,
        horizon_sessions=horizon,
        num_groups=groups,
        test_groups=test_groups,
        embargo_pct=embargo,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Marcos López de Prado (2018) CPCV & PBO Validation Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Horizon: {receipt.horizon_sessions}D)")
    print(f"  Combinatorial Topology:       N = {receipt.num_groups} Groups, k = {receipt.test_groups} Test Groups")
    print(f"  Folds Evaluated:              C({receipt.num_groups}, {receipt.test_groups}) = {receipt.num_folds} Purged Folds ({receipt.num_paths} Complete Paths)")
    print(f"  Leakage Protection:           Embargo = {receipt.embargo_sessions} sessions, Mean Purged = {receipt.mean_purged_sessions_per_fold:.1f} sessions/fold")
    print(f"  CPCV Overfitting Status:      {receipt.cpcv_status}")
    print("-" * 80)
    print("  Out-of-Sample Performance Benchmarking:")
    print(f"    Model MAE:                  {receipt.mean_model_mae:>8.4f}")
    print(f"    Baseline Persistence MAE:   {receipt.mean_baseline_mae:>8.4f}")
    print(f"    MAE Outperformance:         {receipt.mae_improvement_pct:>+7.2f}%")
    print(f"    Directional Accuracy:       {receipt.directional_accuracy_pct:>7.2f}%")
    print("-" * 80)
    print("  Combinatorial Path Sharpe Distribution & PBO:")
    print(f"    Mean OOS Sharpe Ratio:      {receipt.mean_oos_sharpe_ratio:>8.2f}")
    print(f"    Sharpe Dispersion [Min,Max]:[{receipt.min_oos_sharpe_ratio:.2f}, {receipt.max_oos_sharpe_ratio:.2f}]")
    print(f"    Prob. Backtest Overfit(PBO):{receipt.probability_of_backtest_overfitting_pct:>7.1f}%")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def downside_beta_command(args: argparse.Namespace) -> None:
    """Evaluate Ang, Chen & Xing (2006) Downside Beta and Realized Semivariance."""
    import json
    import math
    from src.analytics.downside_risk import evaluate_downside_risk_profile
    from src.data.db import get_observations
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    window = int(getattr(args, "window", 60))

    df_obs = get_observations(ticker)
    if df_obs.empty or len(df_obs) < window + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    receipt = evaluate_downside_risk_profile(
        ticker=ticker,
        asset_prices=prices,
        benchmark_ticker=benchmark,
        window=window,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Ang, Chen & Xing (2006) Downside Beta & Semivariance Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback Window: {receipt.lookback_window}D)")
    print(f"  Crash Sensitivity Tier:       {receipt.crash_sensitivity_tier}")
    print("-" * 80)
    print("  Asymmetric Beta Attributions:")
    print(f"    Standard CAPM Beta:         {receipt.capm_beta:>8.2f} (Unconditional Market Co-Movement)")
    print(f"    Downside Beta (Beta^-):     {receipt.downside_beta:>8.2f} (Down-Market Co-Movement: r_m < mean)")
    print(f"    Upside Beta (Beta^+):       {receipt.upside_beta:>8.2f} (Up-Market Co-Movement: r_m > mean)")
    print(f"    Beta Asymmetry Spread:      {receipt.beta_asymmetry_spread:>+8.2f} (Delta Beta = Beta^- - Beta^+)")
    print("-" * 80)
    print("  Realized Semivariance Decomposition (Annualized):")
    print(f"    Total Realized Volatility:  {receipt.annualized_total_vol_pct:>6.2f}% Ann (RV = {receipt.realized_variance:.6f})")
    print(f"    Good Volatility (Upside):   {receipt.annualized_upside_vol_pct:>6.2f}% Ann (RS^+ = {receipt.upside_semivariance:.6f})")
    print(f"    Bad Volatility (Downside):  {receipt.annualized_downside_vol_pct:>6.2f}% Ann (RS^- = {receipt.downside_semivariance:.6f})")
    print(f"    Downside Variance Ratio:    {receipt.downside_variance_ratio_pct:>6.2f}% of Total Variance")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def optimal_execution_command(args: argparse.Namespace) -> None:
    """Evaluate Almgren & Chriss (2000) optimal execution trajectory and market impact."""
    import json
    from src.analytics.optimal_execution import evaluate_optimal_execution_plan
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    shares = float(args.shares)
    days = int(getattr(args, "days", 5))
    lam = float(getattr(args, "risk_aversion", 1e-6))

    receipt = evaluate_optimal_execution_plan(
        ticker=ticker,
        target_shares=shares,
        execution_days=days,
        risk_aversion_lambda=lam,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    action = "Liquidation (Sell)" if receipt.target_shares > 0 else "Acquisition (Buy)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Almgren & Chriss (2000) Optimal Execution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} ({action})")
    print(f"  Order Size:                   {abs(receipt.target_shares):,.0f} shares (Benchmark S_0: ${receipt.initial_price:.2f})")
    print(f"  Total Market Value:           ${receipt.total_market_value:,.2f}")
    print(f"  Execution Horizon:            {receipt.execution_days} sessions (Urgency Kappa: {receipt.urgency_parameter_kappa:.4f})")
    print(f"  Execution Half-Life:          {receipt.half_life_days:.2f} trading days")
    print(f"  Risk Aversion (Lambda):       {receipt.risk_aversion_lambda:.8f}")
    print("-" * 80)
    print("  Transaction Cost & Market Impact Metrics:")
    print(f"    Expected Total Shortfall:   ${receipt.expected_total_shortfall_dollars:>10.2f} ({receipt.expected_total_shortfall_bps:>6.2f} bps)")
    print(f"    Permanent Impact Cost:      ${receipt.expected_permanent_impact_dollars:>10.2f}")
    print(f"    Temporary Impact Cost:      ${receipt.expected_temporary_impact_dollars:>10.2f}")
    print(f"    Timing Risk (StdDev):       ${receipt.shortfall_std_dev_dollars:>10.2f}")
    print("-" * 80)
    print("  Discrete Execution Trajectory Schedule:")
    print(f"    {'Step':<6} {'Time(D)':<8} {'Start Shares':<14} {'Trade (n_j)':<14} {'Remaining':<14} {'Executed %':<10}")
    for step in receipt.trajectory_schedule:
        print(
            f"    {step['step']:<6} {step['time_day']:<8.1f} {step['inventory_start']:<14,.0f} "
            f"{step['trade_shares']:<14,.0f} {step['inventory_remaining']:<14,.0f} {step['percent_executed_pct']:<10.1f}%"
        )
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fama_french_command(args: argparse.Namespace) -> None:
    """Evaluate Fama-French (1993) and Carhart (1997) multi-factor risk and alpha attribution."""
    import json
    from src.analytics.fama_french import evaluate_fama_french_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    receipt = evaluate_fama_french_profile(
        ticker=ticker,
        asset_prices=prices,
        benchmark_ticker=benchmark,
        lookback_window=lookback,
    )

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_label = "YES (p < 0.05)" if receipt.is_alpha_significant else "NO (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fama-French & Carhart (1997) Multi-Factor Attribution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: SPY)")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total History: {receipt.sample_sessions} sessions)")
    print(f"  Adjusted R-Squared:           {receipt.r_squared_adjusted * 100:.2f}% (R^2: {receipt.r_squared * 100:.2f}%)")
    print("-" * 80)
    print("  Alpha & Factor Loadings (White HC0 Robust Standard Errors):")
    print(f"    {'Factor':<22} {'Beta / Loading':<16} {'Std Error':<12} {'t-Stat':<12} {'p-Value':<10} {'Sig (5%)':<8}")
    for f in receipt.factor_loadings:
        sig_str = "YES" if f["is_significant_5pct"] else "no"
        print(
            f"    {f['factor_name']:<22} {f['beta']:>12.4f}    {f['std_error']:>10.4f}  "
            f"{f['t_stat']:>10.2f}  {f['p_value']:>8.4f}  {sig_str:>6}"
        )
    print("-" * 80)
    print("  Jensen's Alpha & Risk Decomposition (Annualized):")
    print(f"    Annualized Alpha:           {receipt.annualized_alpha_pct:>+8.2f}% ({receipt.alpha_daily_bps:>+6.1f} bps/day | t={receipt.alpha_t_stat:+.2f}, Sig: {sig_label})")
    print(f"    Total Return Volatility:    {receipt.total_annualized_vol_pct:>8.2f}% Ann")
    print(f"    Systematic Factor Vol:      {receipt.systematic_annualized_vol_pct:>8.2f}% Ann")
    print(f"    Idiosyncratic (Specific):   {receipt.idiosyncratic_annualized_vol_pct:>8.2f}% Ann ({receipt.idiosyncratic_variance_ratio_pct:.1f}% of Total Variance)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def black_litterman_command(args: argparse.Namespace) -> None:
    """Execute Black-Litterman (1992) Bayesian portfolio optimization & active view blending."""
    import json
    from src.analytics.black_litterman import evaluate_black_litterman_portfolio
    from src.security.validators import validate_ticker

    raw_tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    universe = [validate_ticker(t) for t in raw_tickers]
    if len(universe) < 2:
        print("\nError: Universe must contain at least 2 distinct tickers.\n")
        return

    views_dict = {}
    if getattr(args, "views", None):
        raw_views = args.views.strip()
        if raw_views.startswith("{"):
            try:
                parsed = json.loads(raw_views)
                views_dict = {validate_ticker(k): float(v) for k, v in parsed.items()}
            except Exception as e:
                print(f"\nError: Invalid JSON views format: {e}\n")
                return
        else:
            for item in raw_views.split(","):
                if ":" in item:
                    k, v = item.split(":", 1)
                    views_dict[validate_ticker(k.strip())] = float(v.strip())

    tau = float(getattr(args, "tau", 0.05))
    delta = float(getattr(args, "risk_aversion", 2.5))
    lookback = int(getattr(args, "lookback", 120))
    denoise = not getattr(args, "no_denoise", False)

    try:
        receipt = evaluate_black_litterman_portfolio(
            universe=universe,
            views_dict=views_dict,
            tau=tau,
            risk_aversion_delta=delta,
            lookback_window=lookback,
            denoise_cov=denoise,
        )
    except Exception as ex:
        print(f"\nError: Black-Litterman optimization failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    denoise_status = "ACTIVE (RMT Marchenko-Pastur)" if receipt.is_rmt_denoised else "INACTIVE (Sample Cov)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black-Litterman (1992) Bayesian Portfolio Engine")
    print("=" * 80)
    print(f"  Asset Universe ({len(receipt.universe)}):        {', '.join(receipt.universe)}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lookback: {receipt.lookback_window}d)")
    print(f"  Parameters:                   tau = {receipt.tau:.3f} | delta = {receipt.risk_aversion_delta:.2f} | Views = {receipt.views_count}")
    print(f"  Covariance Denoising:         {denoise_status}")
    print("-" * 80)
    print("  Bayesian Return & Constrained Allocation Decomposition:")
    print(f"    {'Ticker':<10} {'Market Wgt':<12} {'Prior Ret (Pi)':<16} {'Posterior Ret (μ)':<18} {'BL Optimal Wgt':<16} {'Shift (Δw)':<10}")
    for alloc in receipt.allocations:
        shift_sign = "+" if alloc["weight_shift"] >= 0 else ""
        print(
            f"    {alloc['ticker']:<10} {alloc['market_weight'] * 100:>10.2f}%  "
            f"{alloc['prior_return_ann_pct']:>14.2f}%  "
            f"{alloc['posterior_return_ann_pct']:>16.2f}%  "
            f"{alloc['optimal_weight'] * 100:>14.2f}%  "
            f"{shift_sign}{alloc['weight_shift'] * 100:>8.2f}%"
        )
    print("-" * 80)
    print("  Portfolio Risk & Return Aggregate Performance:")
    print(f"    Expected Annual Return:     {receipt.expected_portfolio_return_ann_pct:>+8.2f}% Ann (Market Prior: {receipt.prior_market_return_ann_pct:>+8.2f}%)")
    print(f"    Portfolio Annualized Vol:   {receipt.expected_portfolio_vol_ann_pct:>8.2f}% Ann")
    print(f"    Portfolio Sharpe Ratio:     {receipt.sharpe_ratio:>+8.2f} (Rf = 4.0%)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def idiosyncratic_vol_command(args: argparse.Namespace) -> None:
    """Evaluate Ang, Hodrick, Xing & Zhang (2006, 2009) Idiosyncratic Volatility & Higher Moments."""
    import json
    from src.analytics.idiosyncratic_volatility import evaluate_idiosyncratic_volatility_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    try:
        receipt = evaluate_idiosyncratic_volatility_profile(
            ticker=ticker,
            asset_prices=prices,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Idiosyncratic volatility evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Ang et al. (2006) Idiosyncratic Volatility & Higher Moments Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Annualized Alpha:             {receipt.annualized_alpha_pct:>+8.2f}% Ann")
    print(f"  Total Return Volatility:      {receipt.total_annualized_vol_pct:>8.2f}% Ann")
    print(f"  R-Squared (Carhart 4-Factor): {receipt.r_squared * 100:>8.2f}%")
    print("-" * 80)
    print("  Idiosyncratic Dispersion & Downside Asymmetry:")
    print(f"    Idiosyncratic Vol (IVOL):   {receipt.idiosyncratic_vol_ann_pct:>8.2f}% Ann ({receipt.idiosyncratic_variance_ratio_pct:.1f}% of Total Variance)")
    print(f"    Downside IVOL (IVOL^-):     {receipt.downside_idiosyncratic_vol_ann_pct:>8.2f}% Ann (Downside Ratio: {receipt.downside_ivol_ratio:.2f})")
    print(f"    Idiosyncratic Skewness:     {receipt.idiosyncratic_skewness:>+8.2f} (Boyer et al. 2010 Lottery Bias)")
    print(f"    Idiosyncratic Kurtosis:     {receipt.idiosyncratic_kurtosis:>8.2f} (Fat-Tail Excess)")
    print(f"    Idiosyncratic Info Ratio:   {receipt.idiosyncratic_information_ratio:>+8.2f} (Alpha / IVOL)")
    print("-" * 80)
    print("  Ang et al. (2006) Cross-Sectional Pricing Anomaly Assessment:")
    print(f"    Anomaly Risk Regime:        {receipt.ivol_anomaly_regime}")
    print(f"    Expected Return Haircut:    -{receipt.ivol_expected_return_haircut_pct:.2f}% / year (Cross-sectional penalty)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def liquidity_beta_command(args: argparse.Namespace) -> None:
    """Evaluate Pástor & Stambaugh (2003) Systematic Liquidity Risk Factor & Liquidity Beta."""
    import json
    from src.analytics.pastor_stambaugh import evaluate_pastor_stambaugh_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]
    volumes = df_obs["volume"]

    try:
        receipt = evaluate_pastor_stambaugh_profile(
            ticker=ticker,
            asset_prices=prices,
            asset_volumes=volumes,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Pástor-Stambaugh liquidity evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_str = "YES (p < 0.05)" if receipt.is_liquidity_beta_significant else "NO (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Pástor & Stambaugh (2003) Systematic Liquidity Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Annualized Alpha:             {receipt.annualized_alpha_pct:>+8.2f}% Ann")
    print(f"  Market Beta (β_M):            {receipt.market_beta:>8.2f}")
    print(f"  Model R-Squared:              {receipt.r_squared * 100:>8.2f}%")
    print("-" * 80)
    print("  Systematic Liquidity Sensitivity & Risk Premium:")
    print(f"    Liquidity Beta (β_L):       {receipt.liquidity_beta:>+8.3f} (t={receipt.liquidity_beta_t_stat:>+6.2f}, Sig: {sig_str})")
    print(f"    Order Flow Gamma (γ):       {receipt.average_price_impact_gamma:>+8.4f} (Price Impact Reversal)")
    print(f"    Liquidity Risk Regime:      {receipt.liquidity_risk_regime}")
    print(f"    Expected Liquidity Premium: {receipt.expected_liquidity_premium_pct:>+8.2f}% / year (Cross-sectional premium)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def fama_macbeth_command(args: argparse.Namespace) -> None:
    """Execute Fama & MacBeth (1973) Two-Pass Cross-Sectional Asset Pricing & Factor Premia estimation."""
    import json
    from src.analytics.fama_macbeth import evaluate_fama_macbeth_profile
    from src.security.validators import validate_ticker

    raw_tickers = [t.strip().upper() for t in args.universe.split(",") if t.strip()]
    universe = [validate_ticker(t) for t in raw_tickers]
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 120))

    if len(universe) < 4:
        print("\nError: Universe must contain at least 4 distinct tickers for cross-sectional regression.\n")
        return

    try:
        receipt = evaluate_fama_macbeth_profile(
            universe=universe,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Fama-MacBeth evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    status_str = "REJECTED (p < 0.05)" if receipt.is_model_rejected_5pct else "SUPPORTED (p >= 0.05)"
    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fama & MacBeth (1973) Two-Pass Asset Pricing Engine")
    print("=" * 80)
    print(f"  Asset Universe ({len(receipt.universe)}):        {', '.join(receipt.universe)}")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Cross-Sectional R-Squared:    {receipt.cross_sectional_r_squared * 100:>8.2f}%")
    print(f"  Pricing Error RMSE:           {receipt.pricing_error_rmse_ann_pct:>8.2f}% Ann (Unexplained Alpha)")
    print(f"  Pricing Model Test:           {status_str}")
    print("-" * 80)
    print("  Cross-Sectional Factor Risk Premia (with Shanken 1992 Errors-in-Variables):")
    print(f"    {'Factor':<12} {'Premium (Ann)':<16} {'t-Stat (FM)':<14} {'t-Stat (Shanken)':<18} {'p-Val':<10} {'Sig (5%)':<8}")
    for p in receipt.factor_premia:
        sig_str = "YES" if p["is_significant_5pct"] else "no"
        print(
            f"    {p['factor_name']:<12} {p['risk_premium_ann_pct']:>12.2f}%  "
            f"{p['t_stat_fm']:>12.2f}  "
            f"{p['t_stat_shanken']:>16.2f}  "
            f"{p['p_value_shanken']:>8.4f}  {sig_str:>6}"
        )
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")

def higher_moments_command(args: argparse.Namespace) -> None:
    """Evaluate Harvey-Siddique (2000) Co-Skewness & Dittmar (2002) Co-Kurtosis Higher-Moment Asset Pricing."""
    import json
    from src.analytics.higher_moments import evaluate_higher_moments_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    benchmark = validate_ticker(getattr(args, "benchmark", "SPY"))
    lookback = int(getattr(args, "lookback", 60))

    df_obs = db.get_observations(ticker)
    if df_obs.empty or len(df_obs) < lookback + 1:
        print(f"\nError: Insufficient observation history for {ticker}. Synchronize ticker first.\n")
        return

    price_col = "adj_close" if "adj_close" in df_obs.columns else "close"
    prices = df_obs[price_col]

    try:
        receipt = evaluate_higher_moments_profile(
            ticker=ticker,
            asset_prices=prices,
            benchmark_ticker=benchmark,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Higher-moments asset pricing evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_sks = "YES (p < 0.05)" if receipt.is_coskewness_significant else "no (p >= 0.05)"
    sig_ckt = "YES (p < 0.05)" if receipt.is_cokurtosis_significant else "no (p >= 0.05)"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Harvey-Siddique & Dittmar Higher-Moment Asset Pricing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Total Aligned: {receipt.sample_sessions} sessions)")
    print(f"  Market Beta (β_M):            {receipt.market_beta:>8.2f}")
    print(f"  Four-Moment Alpha:            {receipt.annualized_alpha_pct:>+8.2f}% Ann (R²: {receipt.four_moment_r_squared * 100:.1f}%)")
    print("-" * 80)
    print("  Higher-Order Systematic Co-Moments:")
    print(f"    Co-Skewness (β_SKS):        {receipt.coskewness_beta:>+8.3f} (t={receipt.coskewness_t_stat:>+6.2f}, Sig: {sig_sks})")
    print(f"    Co-Kurtosis (β_CKT):        {receipt.cokurtosis_beta:>8.3f} (t={receipt.cokurtosis_t_stat:>+6.2f}, Sig: {sig_ckt})")
    print("-" * 80)
    print("  Higher-Moment Risk Assessment & Expected Return Premia:")
    print(f"    Risk Regime:                {receipt.higher_moment_risk_regime}")
    print(f"    Co-Skewness Premium:        {receipt.expected_coskewness_premium_pct:>+8.2f}% / year (-3.6% * β_SKS)")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def merton_credit_command(args: argparse.Namespace) -> None:
    """Evaluate Robert C. Merton (1974) Structural Credit Risk & Distance-to-Default (DD)."""
    import json
    from src.analytics.merton_credit import evaluate_merton_credit_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    debt = float(getattr(args, "debt", 0.0))
    shares = getattr(args, "shares", None)
    if shares is not None:
        shares = float(shares)
    equity_val = getattr(args, "market_cap", None)
    if equity_val is not None:
        equity_val = float(equity_val)
    vol = getattr(args, "volatility", None)
    if vol is not None:
        vol = float(vol)
    rf = float(getattr(args, "risk_free", 0.045))
    horizon = float(getattr(args, "horizon", 1.0))
    lookback = int(getattr(args, "lookback", 60))

    try:
        receipt = evaluate_merton_credit_profile(
            ticker=ticker,
            debt_value=debt,
            shares_outstanding=shares,
            equity_value=equity_val,
            equity_volatility=vol,
            risk_free_rate_ann=rf,
            horizon_years=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Merton structural credit evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Merton (1974) Structural Credit Risk & Distance-to-Default Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Equity Market Value (E):      ${receipt.equity_value:,.0f} (Vol: {receipt.annualized_equity_volatility * 100:.2f}%)")
    print(f"  Debt Liabilities (D):         ${receipt.debt_value:,.0f} (Horizon T: {receipt.horizon_years:.1f} yr, Rf: {receipt.risk_free_rate_ann * 100:.2f}%)")
    print("-" * 80)
    print("  Structural Implied Asset State (Option Inversion):")
    print(f"    Implied Firm Assets (V_A):  ${receipt.implied_asset_value:,.0f}")
    print(f"    Implied Asset Vol (σ_A):    {receipt.implied_asset_volatility * 100:>8.2f}%")
    print(f"    Leverage Ratio (D / V_A):   {receipt.leverage_ratio_pct:>8.2f}%")
    print("-" * 80)
    print("  Default Probability & Structural Credit Spread:")
    print(f"    Distance-to-Default (DD):   {receipt.distance_to_default:>+8.2f} std devs (Risk-Neutral d2: {receipt.risk_neutral_dd:>+8.2f})")
    print(f"    Default Probability (π):    {receipt.default_probability_pct:>8.4f}%")
    print(f"    Implied Credit Spread:      {receipt.implied_credit_spread_bps:>8.1f} bps")
    print(f"    Credit Distress Regime:     {receipt.distress_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def hasbrouck_var_command(args: argparse.Namespace) -> None:
    """Evaluate Joel Hasbrouck (1991) Vector Autoregression (VAR) Price Discovery & Information Share."""
    import json
    from src.analytics.hasbrouck_var import evaluate_hasbrouck_var_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    lags = int(getattr(args, "lags", 5))
    horizon = int(getattr(args, "horizon", 20))
    lookback = int(getattr(args, "lookback", 120))

    try:
        receipt = evaluate_hasbrouck_var_profile(
            ticker=ticker,
            var_lags=lags,
            irf_horizon=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Hasbrouck VAR evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hasbrouck (1991) VAR Price Discovery & Information Share")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Observations:          {receipt.sample_sessions} sessions (Lags p={receipt.var_lags}, IRF H={receipt.irf_horizon})")
    print("-" * 80)
    print("  Price Impact Decomposition (Unexpected 1-σ Trade Shock):")
    print(f"    Contemporaneous Impact:     {receipt.contemporaneous_impact_bps:>+8.2f} bps (b0 initial trade response)")
    print(f"    Permanent Impact I(∞):      {receipt.permanent_impact_bps:>+8.2f} bps (Private Information Content)")
    print(f"    Transient Friction S0:      {receipt.transient_impact_bps:>+8.2f} bps (Bid-Ask Bounce / Inventory Decay)")
    print(f"    Permanent Ratio:            {receipt.permanent_to_total_ratio_pct:>8.1f}%")
    print("-" * 80)
    print("  Hasbrouck Random-Walk Variance Decomposition:")
    print(f"    Trade Info Share (R_w²):    {receipt.information_share_rw2_pct:>8.2f}% of random walk volatility")
    print(f"    Public News Volatility:     {receipt.public_innovation_bps:>8.2f} bps (Non-trade macro innovations)")
    print(f"    Market Quality Regime:      {receipt.market_quality_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def vrp_command(args: argparse.Namespace) -> None:
    """Evaluate Bollerslev, Tauchen & Zhou (2009) Variance Risk Premium (VRP) & Volatility-of-Volatility."""
    import json
    from src.analytics.variance_risk_premium import evaluate_variance_risk_premium_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    iv = getattr(args, "iv", None)
    if iv is not None:
        iv = float(iv)
    horizon = int(getattr(args, "horizon", 21))
    lookback = int(getattr(args, "lookback", 120))

    try:
        receipt = evaluate_variance_risk_premium_profile(
            ticker=ticker,
            implied_volatility=iv,
            horizon_sessions=horizon,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: Variance Risk Premium evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sig_str = "YES (p < 0.05)" if receipt.predictive_p_value < 0.05 else f"no (p = {receipt.predictive_p_value:.3f})"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bollerslev et al. (2009) Variance Risk Premium Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Forecast Horizon:             {receipt.horizon_sessions} sessions (~1 trading month)")
    print(f"  Implied Volatility (IV):      {receipt.implied_volatility_pct:>8.2f}% (Variance: {receipt.implied_variance:.4f})")
    print(f"  Realized Volatility (RV):     {receipt.realized_volatility_pct:>8.2f}% (Variance: {receipt.realized_variance:.4f})")
    print("-" * 80)
    print("  Variance Risk Premium Spread & Volatility of Volatility:")
    print(f"    Variance Risk Premium (VRP):{receipt.variance_risk_premium:>+8.2f} pts ((IV² - RV²) * 100)")
    print(f"    Volatility Spread:          {receipt.volatility_spread_pct_points:>+8.2f} percentage points (IV - RV)")
    print(f"    Volatility-of-Vol (VoV):    {receipt.volatility_of_volatility_pct:>8.2f}% (Annualized VVIX proxy)")
    print(f"    VRP Regime:                 {receipt.vrp_regime}")
    print("-" * 80)
    print(f"  Bollerslev Forward Return Prediction ({receipt.horizon_sessions}-Session Forward Return):")
    print(f"    Predicted Forward Return:   {receipt.predicted_forward_return_pct:>+8.2f}% (R² = {receipt.r_squared_pct:.1f}%)")
    print(f"    Predictive Beta (β_VRP):    {receipt.predictive_beta_vrp:>+8.4f} (t={receipt.predictive_t_stat:>+6.2f}, Sig: {sig_str})")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def max_lottery_command(args: argparse.Namespace) -> None:
    """Evaluate Bali, Cakici & Whitelaw (2011) Extreme Daily Return MAX Effect & Lottery Demand."""
    import json
    from src.analytics.max_lottery import evaluate_max_lottery_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    top_n = int(getattr(args, "top_n", 5))
    lookback = int(getattr(args, "lookback", 60))

    try:
        receipt = evaluate_max_lottery_profile(
            ticker=ticker,
            top_n_days=top_n,
            lookback_window=lookback,
        )
    except Exception as ex:
        print(f"\nError: MAX lottery evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bali et al. (2011) MAX Effect & Lottery Demand Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Sample Window:                {receipt.lookback_window} sessions (Top N Days = {receipt.top_n_days})")
    print(f"  Daily Volatility:             {receipt.daily_volatility_pct:>8.1f}% Ann (Skewness: {receipt.return_skewness:>+6.2f})")
    print("-" * 80)
    print("  Extreme Daily Return Tail Order Statistics:")
    print(f"    Single Highest Return (MAX_1):{receipt.max_1_pct:>+8.2f}%")
    print(f"    Average Top {receipt.top_n_days} Return (MAX_{receipt.top_n_days}):{receipt.max_n_pct:>+8.2f}%")
    print(f"    Average Bottom {receipt.top_n_days} Return (MIN_{receipt.top_n_days}):{receipt.min_n_pct:>+8.2f}%")
    print(f"    Max-to-Min Asymmetry Ratio:   {receipt.max_to_min_ratio:>8.2f}x")
    print("-" * 80)
    print("  Behavioral Lottery Attraction & Expected Performance Penalty:")
    print(f"    Lottery Demand Score (Λ):     {receipt.lottery_demand_score:>8.1f} / 100")
    print(f"    Expected Annualized Alpha Drag:{receipt.expected_alpha_drag_ann_pct:>+8.2f}% / year")
    print(f"    Lottery Demand Regime:        {receipt.lottery_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def profitability_command(args: argparse.Namespace) -> None:
    """Evaluate Robert Novy-Marx (2013) Gross Profitability Premium & Asness et al. (2019) QMJ."""
    import json
    from src.analytics.gross_profitability import evaluate_gross_profitability_profile
    import src.data.db as db
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rev = getattr(args, "revenue", None)
    if rev is not None:
        rev = float(rev)
    cost = getattr(args, "cogs", None)
    if cost is not None:
        cost = float(cost)
    assets = getattr(args, "assets", None)
    if assets is not None:
        assets = float(assets)
    debt = getattr(args, "debt", None)
    if debt is not None:
        debt = float(debt)

    try:
        receipt = evaluate_gross_profitability_profile(
            ticker=ticker,
            revenue=rev,
            cogs=cost,
            total_assets=assets,
            total_debt=debt,
        )
    except Exception as ex:
        print(f"\nError: Gross profitability evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    trap_str = "YES (Unprofitable/High Leverage)" if receipt.is_value_trap else "NO (Quality Compounder)"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Novy-Marx (2013) Gross Profitability & QMJ Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Total Revenue:                ${receipt.revenue:,.0f} (COGS: ${receipt.cogs:,.0f})")
    print(f"  Gross Profit:                 ${receipt.gross_profit:,.0f} (Margin: {receipt.gross_margin_pct:>5.1f}%)")
    print(f"  Total Book Assets:            ${receipt.total_assets:,.0f} (Debt: ${receipt.total_debt:,.0f})")
    print("-" * 80)
    print("  Novy-Marx Profitability to Capital & Turnover:")
    print(f"    Gross Profitability (GP/A): {receipt.gross_profitability_pct:>8.2f}% (Gross Profit / Assets)")
    print(f"    Asset Turnover (ATO):       {receipt.asset_turnover:>8.2f}x (Revenue / Assets)")
    print(f"    Debt-to-Assets Leverage:    {receipt.debt_to_assets_pct:>8.2f}%")
    print("-" * 80)
    print("  Asness et al. (2019) QMJ Composite Quality & Value-Trap Screening:")
    print(f"    Quality Score (Q):          {receipt.quality_score:>8.1f} / 100")
    print(f"    Value-Trap Risk Hazard:     {trap_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_quality_alpha_ann_pct:>+8.2f}% / year")
    print(f"    Quality Factor Regime:      {receipt.quality_regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dual_momentum_command(args: argparse.Namespace) -> None:
    """Evaluate Jegadeesh & Titman (1993, 2001) / Antonacci (2014) Dual Momentum Profile."""
    import json
    from src.analytics.dual_momentum import evaluate_dual_momentum_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")
    rf_rate = float(getattr(args, "rf_rate", 4.0) or 4.0)

    try:
        receipt = evaluate_dual_momentum_profile(
            ticker=ticker,
            benchmark_ticker=bench,
            rf_rate_pct=rf_rate,
        )
    except Exception as ex:
        print(f"\nError: Dual momentum evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    crash_str = (
        "YES (High Past Winner + Severe 1M Drop)"
        if receipt.has_momentum_crash_risk
        else "NO (Stable Drift)"
    )
    abs_str = (
        "BULLISH (Positive Trend > Rf, >= SMA200)"
        if receipt.is_absolute_trend_positive
        else "BEARISH (Defensive / Cash Hedge)"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Jegadeesh & Titman (1993) / Antonacci Dual Momentum Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f} (SMA 200: ${receipt.sma_200:.2f})")
    print("-" * 80)
    print("  Jegadeesh & Titman (1993, 2001) Intermediate Momentum & Microstructure Reversal:")
    print(f"    12-2 Intermediate Momentum: {receipt.momentum_12_2_pct:>+8.2f}% (t-252 to t-21, skipping 1M)")
    print(f"    6-1 Intermediate Momentum:  {receipt.momentum_6_1_pct:>+8.2f}% (t-126 to t-21, skipping 1M)")
    print(f"    1-Month Reversal (R_1,0):   {receipt.reversal_1_0_pct:>+8.2f}% (Microstructure bounce check)")
    print(f"    Unlagged 12M Total Return:  {receipt.total_return_12m_pct:>+8.2f}%")
    print("-" * 80)
    print("  Cross-Sectional Relative Strength & Moskowitz-Grinblatt Decomposition:")
    print(f"    Benchmark 12-2 Momentum:    {receipt.benchmark_12_2_pct:>+8.2f}%")
    print(f"    Relative Momentum Spread:   {receipt.relative_momentum_spread_pct:>+8.2f}% vs {receipt.benchmark_ticker}")
    print(f"    Idiosyncratic Alpha Mom:    {receipt.idiosyncratic_momentum_pct:>+8.2f}% (Market-beta adjusted)")
    print(f"    Realized Market Beta:       {receipt.beta_to_benchmark:>8.2f}")
    print(f"    Volatility-Adjusted Sharpe: {receipt.sharpe_momentum:>8.2f} (Ann Vol: {receipt.annualized_volatility_pct:.1f}%)")
    print("-" * 80)
    print("  Gary Antonacci Dual Momentum & Crash Risk Overlay:")
    print(f"    Absolute Trend Filter:      {abs_str}")
    print(f"    Momentum Crash Risk:        {crash_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_momentum_alpha_pct:>+8.2f}% / year")
    print(f"    Momentum Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bab_command(args: argparse.Namespace) -> None:
    """Evaluate Andrea Frazzini & Lasse Heje Pedersen (2014) Betting Against Beta (BAB) Profile."""
    import json
    from src.analytics.betting_against_beta import evaluate_bab_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")
    rf_rate = float(getattr(args, "rf_rate", 4.0) or 4.0)
    erp = float(getattr(args, "market_premium", 6.0) or 6.0)

    try:
        receipt = evaluate_bab_profile(
            ticker=ticker,
            benchmark_ticker=bench,
            rf_rate_pct=rf_rate,
            equity_risk_premium_pct=erp,
        )
    except Exception as ex:
        print(f"\nError: Betting Against Beta evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Frazzini & Pedersen (2014) Betting Against Beta (BAB) Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Systematic Market Risk & Vasicek (1973) Bayesian Shrinkage:")
    print(f"    Raw Sample Beta (β_raw):    {receipt.raw_beta:>8.2f} (ρ = {receipt.correlation_to_benchmark:.2f})")
    print(f"    Vasicek Shrunk Beta (β*):   {receipt.shrunk_beta:>8.2f} (0.60·β_raw + 0.40·1.0)")
    print(f"    Asset Realized Volatility:  {receipt.asset_annualized_volatility_pct:>8.1f}% Ann (Bench Vol: {receipt.benchmark_annualized_volatility_pct:.1f}%)")
    print("-" * 80)
    print("  Security Market Line (SML) & Jensen's Alpha:")
    print(f"    Realized Annual Return:     {receipt.realized_annualized_return_pct:>+8.2f}% / year")
    print(f"    CAPM Expected Return:       {receipt.capm_expected_return_pct:>+8.2f}% / year (Rf={rf_rate:.1f}%, ERP={erp:.1f}%)")
    print(f"    Jensen's Alpha (α_i):       {receipt.jensen_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print("  Frazzini-Pedersen Leverage Multiplier & BAB Spread:")
    print(f"    Optimal BAB Multiplier:     {receipt.bab_leverage_multiplier:>8.2f}x (1.0 / β*)")
    print(f"    Expected BAB Factor Alpha:  {receipt.expected_bab_spread_pct:>+8.2f}% / year")
    print(f"    Low-Beta Compounder:        {'YES' if receipt.is_low_beta_compounder else 'NO'}")
    print(f"    High-Beta Overpriced Drag:  {'YES (Overpriced)' if receipt.is_high_beta_overpriced else 'NO'}")
    print(f"    BAB Factor Regime:          {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def reversal_command(args: argparse.Namespace) -> None:
    """Evaluate Narasimhan Jegadeesh (1990) & Bruce Lehmann (1990) Short-Term Reversal Profile."""
    import json
    from src.analytics.short_term_reversal import evaluate_short_term_reversal_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bench = validate_ticker(getattr(args, "benchmark", "SPY") or "SPY")

    try:
        receipt = evaluate_short_term_reversal_profile(
            ticker=ticker,
            benchmark_ticker=bench,
        )
    except Exception as ex:
        print(f"\nError: Short-term reversal evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    dip_str = (
        "YES (High-Conviction Pullback in Bull Trend)"
        if receipt.is_liquidity_dip_buy
        else "NO"
    )
    knife_str = (
        "YES (Avoid - Falling Knife in Bear Trend)"
        if receipt.is_falling_knife
        else "NO"
    )
    trend_str = (
        "BULLISH (12-2 Mom > 0, Price >= SMA200)"
        if receipt.is_bullish_intermediate_trend
        else "BEARISH (Downtrend)"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Jegadeesh & Lehmann (1990) Short-Term Reversal Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker} (Benchmark: {receipt.benchmark_ticker})")
    print(f"  Sessions Evaluated:           {receipt.sessions_evaluated} trading days")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Short-Term Weekly Returns & Microstructure Serial Correlation:")
    print(f"    1-Week Return (R_1w):       {receipt.return_1w_pct:>+8.2f}% (t-5 to t)")
    print(f"    2-Week Return (R_2w):       {receipt.return_2w_pct:>+8.2f}% (t-10 to t)")
    print(f"    Benchmark 1-Week Return:    {receipt.benchmark_1w_pct:>+8.2f}%")
    print(f"    Excess 1-Week Return:       {receipt.excess_return_1w_pct:>+8.2f}% vs {receipt.benchmark_ticker}")
    print(f"    Serial Correlation (ρ_1):   {receipt.serial_correlation_rho1:>8.2f} (Daily first-order autocovariance)")
    print(f"    Roll (1984) Spread Proxy:   {receipt.roll_spread_proxy_pct:>8.2f}% (Bid-ask bounce cost)")
    print(f"    Volume Shock Z-Score:       {receipt.volume_shock_zscore:>8.2f}σ (Liquidity selling intensity)")
    print("-" * 80)
    print("  Trend Confluence & Liquidity Dip Filtering:")
    print(f"    12-2 Intermediate Momentum: {receipt.intermediate_momentum_12_2_pct:>+8.2f}%")
    print(f"    Macro Trend Status:         {trend_str}")
    print(f"    Liquidity Dip Buy Signal:   {dip_str}")
    print(f"    Falling Knife Warning:      {knife_str}")
    print(f"    Expected Rebound Alpha:     {receipt.expected_contrarian_alpha_pct:>+8.2f}% / year")
    print(f"    Reversal Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def distress_command(args: argparse.Namespace) -> None:
    """Evaluate John Y. Campbell et al. (2008) Financial Distress Risk Hazard Profile."""
    import json
    from src.analytics.distress_risk import evaluate_distress_risk_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)
    debt = getattr(args, "debt", None)
    if debt is not None:
        debt = float(debt)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cash = getattr(args, "cash", None)
    if cash is not None:
        cash = float(cash)

    try:
        receipt = evaluate_distress_risk_profile(
            ticker=ticker,
            market_cap=mcap,
            total_liabilities=debt,
            net_income=ni,
            cash_equivalents=cash,
        )
    except Exception as ex:
        print(f"\nError: Financial distress evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    safe_str = "YES (Impeccable Solvency)" if receipt.is_solvent_safe_haven else "NO"
    trap_str = "YES (High Failure Risk - Avoid Zombie)" if receipt.is_distress_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell, Hilscher & Szilagyi (2008) Distress Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print(f"  Market-Adjusted Assets (MTA): ${receipt.market_adjusted_assets:,.0f}")
    print("-" * 80)
    print("  Campbell et al. Dynamic Multi-Period Balance Sheet & Market Covariates:")
    print(f"    Total Liabilities / MTA:    {receipt.tlmta_pct:>8.2f}% (Market-implied leverage)")
    print(f"    Net Income / MTA (NIMTA):   {receipt.nimta_pct:>+8.2f}% (Operating capital efficiency)")
    print(f"    Cash Buffer / MTA (CASHMTA):{receipt.cashmta_pct:>8.2f}% (Liquidity cushion)")
    print(f"    Annualized Volatility:      {receipt.annualized_volatility_pct:>8.1f}% Ann (SIGMA)")
    print(f"    Excess Trailing Return:     {receipt.excess_trailing_return_pct:>+8.2f}% (EXRET momentum)")
    print("-" * 80)
    print("  Dynamic Failure Hazard & Financial Distress Puzzle Spread:")
    print(f"    CHS Linear Logit Score (z): {receipt.chs_linear_index:>8.2f}")
    print(f"    12-Month Failure Hazard (P):{receipt.distress_probability_12m_pct:>8.2f}% (Default/Bankruptcy probability)")
    print(f"    Solvent Safe Haven:         {safe_str}")
    print(f"    Distress Risk Trap:         {trap_str}")
    print(f"    Expected Annualized Alpha:  {receipt.expected_distress_alpha_pct:>+8.2f}% / year")
    print(f"    Solvency Factor Regime:     {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def sentiment_command(args: argparse.Namespace) -> None:
    """Evaluate Malcolm Baker & Jeffrey Wurgler (2006) Cross-Sectional Sentiment Profile."""
    import json
    from src.analytics.investor_sentiment import evaluate_investor_sentiment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sent_idx = float(getattr(args, "market_sentiment", 0.0) or 0.0)
    margin = getattr(args, "net_margin", None)
    if margin is not None:
        margin = float(margin)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)

    try:
        receipt = evaluate_investor_sentiment_profile(
            ticker=ticker,
            market_sentiment_index=sent_idx,
            net_profit_margin_pct=margin,
            market_cap=mcap,
        )
    except Exception as ex:
        print(f"\nError: Investor sentiment evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    bubble_str = (
        "YES (Euphoric Sentiment + High Speculation)"
        if receipt.is_speculative_bubble
        else "NO"
    )
    bargain_str = (
        "YES (Panicked Sentiment + Oversold Speculation)"
        if receipt.is_contrarian_bargain
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Baker & Wurgler (2006) Investor Sentiment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Baker & Wurgler Subjectivity & Limits to Arbitrage Covariates:")
    print(f"    Annualized Volatility:      {receipt.annualized_volatility_pct:>8.1f}% Ann (Arbitrage risk)")
    print(f"    Net Profit Margin:          {receipt.net_profit_margin_pct:>+8.1f}% (Cash flow subjectivity)")
    print(f"    MAX_5 Lottery Demand:       {receipt.max_daily_return_5d_pct:>8.2f}% (Retail speculation)")
    print(f"    Firm Speculative Index (S): {receipt.speculative_index:>8.1f} / 100")
    print("-" * 80)
    print("  Cross-Sectional Sentiment Sensitivity & Reversal Dynamics:")
    print(f"    Macro Sentiment Index (I):  {receipt.market_sentiment_index:>+8.2f}σ (Standardized wave)")
    print(f"    Equity Sentiment Beta:      {receipt.sentiment_beta:>+8.2f} (Sensitivity to sentiment)")
    print(f"    Speculative Bubble Risk:    {bubble_str}")
    print(f"    Contrarian Bargain Rebound: {bargain_str}")
    print(f"    Expected Sentiment Alpha:   {receipt.expected_sentiment_alpha_pct:>+8.2f}% / year")
    print(f"    Sentiment Factor Regime:    {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def arbitrage_asymmetry_command(args: argparse.Namespace) -> None:
    """Evaluate Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Short-Sale Impediments & Arbitrage Asymmetry."""
    import json
    from src.analytics.arbitrage_asymmetry import evaluate_arbitrage_asymmetry_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sent_idx = float(getattr(args, "market_sentiment", 0.0) or 0.0)
    borrow_fee = getattr(args, "borrow_fee", None)
    if borrow_fee is not None:
        borrow_fee = float(borrow_fee)
    short_interest = getattr(args, "short_interest", None)
    if short_interest is not None:
        short_interest = float(short_interest)
    dtc = getattr(args, "days_to_cover", None)
    if dtc is not None:
        dtc = float(dtc)

    try:
        receipt = evaluate_arbitrage_asymmetry_profile(
            ticker=ticker,
            market_sentiment_index=sent_idx,
            borrow_fee_annual_pct=borrow_fee,
            short_interest_pct_float=short_interest,
            days_to_cover=dtc,
        )
    except Exception as ex:
        print(f"\nError: Arbitrage asymmetry evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    severe_str = (
        "YES (High Overpricing Hazard)"
        if receipt.is_severe_overpriced
        else "NO"
    )
    squeeze_str = (
        "YES (Acute Short Squeeze Risk)"
        if receipt.is_squeeze_risk
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Stambaugh, Yu & Yuan (2012) Arbitrage Asymmetry Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Short-Sale Impediment & Friction Covariates:")
    print(f"    Idiosyncratic Volatility:   {receipt.idiosyncratic_volatility_pct:>8.1f}% Ann (Arbitrage risk)")
    print(f"    Annualized Borrow Fee:      {receipt.borrow_fee_annual_pct:>8.2f}% / yr (Cost to borrow)")
    print(f"    Short Interest (% Float):   {receipt.short_interest_pct_float:>8.1f}% of float")
    print(f"    Days to Cover (DTC):        {receipt.days_to_cover:>8.1f} sessions")
    print(f"    Daily Share Turnover:       {receipt.daily_turnover_pct:>8.2f}% per session")
    print(f"    Short Impediment Index (I): {receipt.short_sale_impediment_index:>8.1f} / 100")
    print("-" * 80)
    print("  Arbitrage Asymmetry & Overpricing Dynamics:")
    print(f"    Macro Sentiment Index (I):  {receipt.market_sentiment_index:>+8.2f}σ (Standardized)")
    print(f"    Overpricing Hazard Score:   {receipt.asymmetric_overpricing_hazard:>8.1f} / 100")
    print(f"    Short Squeeze Vulnerability:{receipt.short_squeeze_vulnerability:>8.1f} / 100")
    print(f"    Severe Overpriced Drag:     {severe_str}")
    print(f"    Short Squeeze Powder Keg:   {squeeze_str}")
    print(f"    Projected Mispricing Drag:  {receipt.expected_asymmetry_alpha_pct:>+8.2f}% / year")
    print(f"    Arbitrage Asymmetry Regime: {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def q_factor_command(args: argparse.Namespace) -> None:
    """Evaluate Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor Investment Profile."""
    import json
    from src.analytics.q_factor import evaluate_q_factor_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    roe = getattr(args, "roe", None)
    if roe is not None:
        roe = float(roe)
    inv = getattr(args, "investment_rate", None)
    if inv is None:
        inv = getattr(args, "asset_growth", None)
    if inv is not None:
        inv = float(inv)
    tobin = getattr(args, "tobin_q", None)
    if tobin is not None:
        tobin = float(tobin)

    try:
        receipt = evaluate_q_factor_profile(
            ticker=ticker,
            return_on_equity_pct=roe,
            investment_to_assets_pct=inv,
            tobin_q_proxy=tobin,
        )
    except Exception as ex:
        print(f"\nError: q-Factor evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (High ROE + Conservative Investment)"
        if receipt.is_disciplined_compounder
        else "NO"
    )
    overinvestor_str = (
        "YES (Low/Negative ROE + Aggressive Asset Expansion)"
        if receipt.is_overinvestor_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hou, Xue & Zhang (2015) q-Factor Investment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.0f}")
    print("-" * 80)
    print("  Neoclassical Investment & Profitability Covariates:")
    print(f"    Return on Equity (ROE):     {receipt.return_on_equity_pct:>+8.1f}% (Marginal profitability)")
    print(f"    Investment-to-Assets (I/A): {receipt.investment_to_assets_pct:>+8.1f}% Ann (Asset growth rate)")
    print(f"    Tobin's q Proxy:            {receipt.tobin_q_proxy:>8.2f} (Market-to-book assets)")
    print(f"    Operating Cash Flow/Assets: {receipt.cash_flow_to_assets_pct:>+8.1f}%")
    print(f"    Profitability Score (ROE):  {receipt.profitability_score:>8.1f} / 100")
    print(f"    Investment Score (I/A):     {receipt.investment_score:>8.1f} / 100 (Lower = More Conservative)")
    print("-" * 80)
    print("  Hou-Xue-Zhang Factor Dynamics & Alpha Projections:")
    print(f"    Composite q-Alpha Score:    {receipt.q_alpha_score:>8.1f} / 100")
    print(f"    Disciplined Compounder:     {compounder_str}")
    print(f"    Over-Investment Trap:       {overinvestor_str}")
    print(f"    Expected q-Factor Alpha:    {receipt.expected_q_alpha_pct:>+8.2f}% / year")
    print(f"    Investment Factor Regime:   {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def accruals_command(args: argparse.Namespace) -> None:
    """Evaluate Richard G. Sloan (1996) Accrual Anomaly & Earnings Quality Decomposition."""
    import json
    from src.analytics.accrual_anomaly import evaluate_accrual_anomaly_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cfo = getattr(args, "operating_cash_flow", None)
    if cfo is None:
        cfo = getattr(args, "cash_flow", None)
    if cfo is not None:
        cfo = float(cfo)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)

    try:
        receipt = evaluate_accrual_anomaly_profile(
            ticker=ticker,
            net_income=ni,
            operating_cash_flow=cfo,
            average_total_assets=assets,
        )
    except Exception as ex:
        print(f"\nError: Accrual anomaly evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (High Cash Conversion & Persistent Earnings)"
        if receipt.is_cash_quality_compounder
        else "NO"
    )
    trap_str = (
        "YES (Aggressive Accrual Distortion Trap)"
        if receipt.is_accrual_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Richard G. Sloan (1996) Accrual Anomaly Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Average Total Assets:         ${receipt.average_total_assets:,.0f}")
    print("-" * 80)
    print("  Earnings Decomposition & Cash Conversion Covariates:")
    print(f"    Reported Net Income:        ${receipt.net_income:,.0f}")
    print(f"    Operating Cash Flow:        ${receipt.operating_cash_flow:,.0f}")
    print(f"    Total Accounting Accruals:  ${receipt.total_accruals:>+15,.0f} (Net Income - Cash Flow)")
    print(f"    Accruals / Assets (CFA):    {receipt.accrual_ratio_pct:>+8.2f}% (Normalized Accrual Ratio)")
    print(f"    Cash Flow / Net Income:     {receipt.cash_flow_to_net_income:>8.2f}x (Cash Conversion Multiplier)")
    print(f"    Earnings Quality Index:     {receipt.earnings_quality_index:>8.1f} / 100")
    print("-" * 80)
    print("  Sloan Accrual Fixation & Alpha Spread Dynamics:")
    print(f"    Cash Quality Compounder:    {compounder_str}")
    print(f"    Accrual Inflation Trap:     {trap_str}")
    print(f"    Expected Accrual Alpha:     {receipt.expected_accrual_alpha_pct:>+8.2f}% / year")
    print(f"    Earnings Quality Regime:    {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def beneish_command(args: argparse.Namespace) -> None:
    """Evaluate Messod D. Beneish (1999) Forensic Accounting M-Score & Manipulation Risk."""
    import json
    from src.analytics.beneish_mscore import evaluate_beneish_mscore_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    dsri = getattr(args, "dsri", None)
    if dsri is not None:
        dsri = float(dsri)
    gmi = getattr(args, "gmi", None)
    if gmi is not None:
        gmi = float(gmi)
    aqi = getattr(args, "aqi", None)
    if aqi is not None:
        aqi = float(aqi)
    sgi = getattr(args, "sgi", None)
    if sgi is not None:
        sgi = float(sgi)
    depi = getattr(args, "depi", None)
    if depi is not None:
        depi = float(depi)
    sgai = getattr(args, "sgai", None)
    if sgai is not None:
        sgai = float(sgai)
    lvgi = getattr(args, "lvgi", None)
    if lvgi is not None:
        lvgi = float(lvgi)
    tata = getattr(args, "tata", None)
    if tata is not None:
        tata = float(tata)

    try:
        receipt = evaluate_beneish_mscore_profile(
            ticker=ticker,
            dsri=dsri,
            gmi=gmi,
            aqi=aqi,
            sgi=sgi,
            depi=depi,
            sgai=sgai,
            lvgi=lvgi,
            tata=tata,
        )
    except Exception as ex:
        print(f"\nError: Beneish M-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    fraud_str = (
        "YES (HIGH MANIPULATION / FRAUD PROBABILITY)"
        if receipt.is_manipulator_risk
        else "NO"
    )
    clean_str = (
        "YES (PRISTINE UNMANIPULATED FINANCIALS)"
        if receipt.is_clean_profile
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Messod D. Beneish (1999) Forensic Accounting M-Score")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Beneish 8-Variable Forensic Covariates:")
    print(f"    Days' Sales in Receivables: {receipt.dsri:>8.2f} (DSRI: Revenue acceleration)")
    print(f"    Gross Margin Index:         {receipt.gmi:>8.2f} (GMI: Margin deterioration)")
    print(f"    Asset Quality Index:        {receipt.aqi:>8.2f} (AQI: Expense capitalization)")
    print(f"    Sales Growth Index:         {receipt.sgi:>8.2f} (SGI: Deceleration pressure)")
    print(f"    Depreciation Index:         {receipt.depi:>8.2f} (DEPI: Asset life extension)")
    print(f"    SG&A Expense Index:         {receipt.sgai:>8.2f} (SGAI: Operational inefficiency)")
    print(f"    Leverage Index:             {receipt.lvgi:>8.2f} (LVGI: Debt covenant pressure)")
    print(f"    Total Accruals to Assets:   {receipt.tata:>+8.3f} (TATA: Sloan paper profits)")
    print("-" * 80)
    print("  Forensic Model Inferences & Manipulation Probabilities:")
    print(f"    Beneish M-Score:            {receipt.m_score:>+8.2f} (Cutoff: > -1.78)")
    print(f"    Manipulation Probability:   {receipt.manipulation_probability_pct:>8.1f}% (Phi(M))")
    print(f"    Manipulator Red Flag:       {fraud_str}")
    print(f"    Clean Financial Profile:    {clean_str}")
    print(f"    Expected Forensic Alpha:    {receipt.expected_forensic_alpha_pct:>+8.2f}% / year")
    print(f"    Forensic Regime:            {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fscore_command(args: argparse.Namespace) -> None:
    """Evaluate Joseph D. Piotroski (2000) 9-Point Fundamental Accounting F-Score."""
    import json
    from src.analytics.piotroski_fscore import evaluate_piotroski_fscore_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    cfo = getattr(args, "operating_cash_flow", None)
    if cfo is None:
        cfo = getattr(args, "cash_flow", None)
    if cfo is not None:
        cfo = float(cfo)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)
    roa = getattr(args, "roa", None)
    if roa is not None:
        roa = float(roa)
    d_roa = getattr(args, "delta_roa", None)
    if d_roa is not None:
        d_roa = float(d_roa)
    d_debt = getattr(args, "delta_debt", None)
    if d_debt is not None:
        d_debt = float(d_debt)
    d_liq = getattr(args, "delta_current_ratio", None)
    if d_liq is not None:
        d_liq = float(d_liq)
    dilut = getattr(args, "shares_dilution", None)
    if dilut is not None:
        dilut = float(dilut)
    d_margin = getattr(args, "delta_gross_margin", None)
    if d_margin is not None:
        d_margin = float(d_margin)
    d_turnover = getattr(args, "delta_asset_turnover", None)
    if d_turnover is not None:
        d_turnover = float(d_turnover)

    try:
        receipt = evaluate_piotroski_fscore_profile(
            ticker=ticker,
            net_income=ni,
            operating_cash_flow=cfo,
            total_assets=assets,
            return_on_assets_pct=roa,
            delta_roa_pct=d_roa,
            delta_long_term_debt=d_debt,
            delta_current_ratio=d_liq,
            shares_dilution_pct=dilut,
            delta_gross_margin_pct=d_margin,
            delta_asset_turnover=d_turnover,
        )
    except Exception as ex:
        print(f"\nError: Piotroski F-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    compounder_str = (
        "YES (HIGH CONVICTION FUNDAMENTAL COMPOUNDER)"
        if receipt.is_strong_compounder
        else "NO"
    )
    trap_str = (
        "YES (HIGH RISK VALUE TRAP / DISTRESS)"
        if receipt.is_value_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Joseph D. Piotroski (2000) Fundamental F-Score Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Aggregate Piotroski F-Score:  {receipt.f_score} / 9 Points ({receipt.regime})")
    print("  Fundamental Subscore Breakdown:")
    print(f"    1. Profitability Signals:   {receipt.profitability_points} / 4 Points")
    print(f"       - Return on Assets > 0:       {'PASS [1]' if receipt.f_roa else 'FAIL [0]'} (ROA: {receipt.return_on_assets_pct:+.1f}%)")
    print(f"       - Cash Flow from Ops > 0:     {'PASS [1]' if receipt.f_cfo else 'FAIL [0]'} (CFO: ${receipt.operating_cash_flow:,.0f})")
    print(f"       - Improving ROA (ΔROA > 0):   {'PASS [1]' if receipt.f_delta_roa else 'FAIL [0]'} (ΔROA: {receipt.delta_roa_pct:+.1f}%)")
    print(f"       - Accrual Quality (CFO > NI): {'PASS [1]' if receipt.f_accrual else 'FAIL [0]'} (Net Income: ${receipt.net_income:,.0f})")
    print(f"    2. Leverage & Liquidity:    {receipt.leverage_liquidity_points} / 3 Points")
    print(f"       - De-leveraging (ΔDebt <= 0): {'PASS [1]' if receipt.f_delta_leverage else 'FAIL [0]'}")
    print(f"       - Expanding Liquidity (ΔCR):  {'PASS [1]' if receipt.f_delta_liquidity else 'FAIL [0]'}")
    print(f"       - Zero Share Dilution:        {'PASS [1]' if receipt.f_no_dilution else 'FAIL [0]'}")
    print(f"    3. Operating Efficiency:    {receipt.operating_efficiency_points} / 2 Points")
    print(f"       - Expanding Margin (ΔMargin): {'PASS [1]' if receipt.f_delta_margin else 'FAIL [0]'}")
    print(f"       - Higher Turnover (ΔTurnover):{'PASS [1]' if receipt.f_delta_turnover else 'FAIL [0]'}")
    print("-" * 80)
    print("  Valuation Health & Alpha Spread Dynamics:")
    print(f"    Strong Compounder Quality:  {compounder_str}")
    print(f"    Vulnerable Value Trap Trap: {trap_str}")
    print(f"    Expected Fundamental Alpha: {receipt.expected_fundamental_alpha_pct:>+8.2f}% / year")
    print(f"    Fundamental Health Regime:  {receipt.regime}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def diffusion_command(args: argparse.Namespace) -> None:
    """Evaluate Hong & Stein (1999) / Chordia & Shivakumar (2002) Gradual Information Diffusion."""
    import json
    from src.analytics.gradual_diffusion import evaluate_gradual_diffusion_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    coverage = getattr(args, "analyst_coverage", None)
    if coverage is not None:
        coverage = int(coverage)

    try:
        receipt = evaluate_gradual_diffusion_profile(
            ticker=ticker,
            analyst_coverage=coverage,
        )
    except Exception as ex:
        print(f"\nError: Hong-Stein Diffusion evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    underreaction_str = (
        "YES (HIGH CONVICTION CONTINUATION MOMENTUM)"
        if receipt.is_early_underreaction
        else "NO"
    )
    overreaction_str = (
        "YES (HIGH REVERSAL / CROWDING CRASH RISK)"
        if receipt.is_crowded_overreaction
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hong & Stein (1999) Gradual Information Diffusion Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Cumulative Momentum:          {receipt.cumulative_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Daily Turnover Ratio:         {receipt.turnover_ratio_pct:.2f}%")
    print(f"  Information Diffusion (IDV):  {receipt.information_diffusion_velocity:.2f} / 1.00 (Speed of news permeation)")
    print(f"  Trend Maturity Index (TMI):   {receipt.trend_maturity_index:.2f} / 1.00 (Speculative exhaustion)")
    print(f"  Momentum Crowding Risk (MCR): {receipt.momentum_crowding_risk_pct:.1f}%")
    print("-" * 80)
    print("  Behavioral Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Early Underreaction Phase:  {underreaction_str}")
    print(f"    Crowded Overreaction Peak:  {overreaction_str}")
    print(f"    Expected Hong-Stein Alpha:  {receipt.expected_hong_stein_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def contrarian_command(args: argparse.Namespace) -> None:
    """Evaluate Josef Lakonishok, Andrei Shleifer & Robert W. Vishny (1994) Contrarian Investment."""
    import json
    from src.analytics.contrarian_extrapolation import evaluate_contrarian_extrapolation_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    sg = getattr(args, "sales_growth", None)
    if sg is not None:
        sg = float(sg)
    eg = getattr(args, "earnings_growth", None)
    if eg is not None:
        eg = float(eg)
    pe = getattr(args, "pe", None)
    if pe is not None:
        pe = float(pe)
    pcf = getattr(args, "pcf", None)
    if pcf is not None:
        pcf = float(pcf)
    pb = getattr(args, "pb", None)
    if pb is not None:
        pb = float(pb)

    try:
        receipt = evaluate_contrarian_extrapolation_profile(
            ticker=ticker,
            sales_growth_3yr_pct=sg,
            earnings_growth_3yr_pct=eg,
            price_to_earnings=pe,
            price_to_cash_flow=pcf,
            price_to_book=pb,
        )
    except Exception as ex:
        print(f"\nError: LSV Contrarian evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    contrarian_str = (
        "YES (HIGH CONVICTION CONTRARIAN VALUE BUY)"
        if receipt.is_contrarian_value
        else "NO"
    )
    glamour_str = (
        "YES (VULNERABLE GLAMOUR OVERPRICING TRAP)"
        if receipt.is_glamour_trap
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lakonishok, Shleifer & Vishny (1994) Contrarian Investment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Sales Growth (3-Yr Ann):      {receipt.sales_growth_3yr_pct:>+8.2f}%")
    print(f"  Earnings Growth (3-Yr Ann):   {receipt.earnings_growth_3yr_pct:>+8.2f}%")
    print(f"  Trailing 1-Yr Momentum:       {receipt.trailing_momentum_pct:>+8.2f}%")
    print(f"  Valuation Multiples:          P/E: {receipt.price_to_earnings:.1f} | P/CF: {receipt.price_to_cash_flow:.1f} | P/B: {receipt.price_to_book:.1f}")
    print("-" * 80)
    print(f"  Naive Extrapolation (NEI):    {receipt.naive_extrapolation_index:.2f} / 1.00 (Degree of past growth extrapolation)")
    print(f"  Valuation Multiple (VMC):     {receipt.valuation_multiple_composite:.2f} / 1.00 (Normalized multiple composite)")
    print(f"  Expectation Discrepancy (ED): {receipt.expectation_discrepancy:>+8.2f} (VMC - NEI)")
    print("-" * 80)
    print("  Contrarian Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Contrarian Value Pick:      {contrarian_str}")
    print(f"    Vulnerable Glamour Trap:    {glamour_str}")
    print(f"    Expected LSV Alpha Spread:  {receipt.expected_contrarian_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dhs_command(args: argparse.Namespace) -> None:
    """Evaluate Daniel, Hirshleifer & Subrahmanyam (1998) Overconfidence & Self-Attribution."""
    import json
    from src.analytics.dhs_overconfidence import evaluate_dhs_overconfidence_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    surprise = getattr(args, "earnings_surprise", None)
    if surprise is not None:
        surprise = float(surprise)
    dispersion = getattr(args, "dispersion", None)
    if dispersion is not None:
        dispersion = float(dispersion)

    try:
        receipt = evaluate_dhs_overconfidence_profile(
            ticker=ticker,
            earnings_surprise_pct=surprise,
            analyst_dispersion_pct=dispersion,
        )
    except Exception as ex:
        print(f"\nError: DHS Overconfidence evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    accel_str = (
        "YES (CONFIRMATORY PUBLIC NEWS AMPLIFICATION)"
        if receipt.is_attribution_accelerating
        else "NO"
    )
    fragile_str = (
        "YES (CONTRADICTORY PUBLIC NEWS CRASH HAZARD)"
        if receipt.is_contradiction_fragile
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Daniel, Hirshleifer & Subrahmanyam (1998) DHS Behavioral Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing Momentum:            {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Earnings Surprise:            {receipt.earnings_surprise_pct:>+8.2f}% (Analyst Dispersion: {receipt.analyst_dispersion_pct:.1f}%)")
    print(f"  Turnover Ratio:               {receipt.turnover_ratio_pct:.2f}%")
    print("-" * 80)
    print(f"  Overconfidence Intensity:     {receipt.overconfidence_intensity:.2f} / 1.00 (Private signal overestimation)")
    print(f"  Biased Self-Attribution (BAI):{receipt.biased_self_attribution_index:>+8.2f} (Confirmation > 0, Contradiction < 0)")
    print(f"  Reversal Crash Hazard:        {receipt.reversal_crash_hazard_pct:.1f}%")
    print("-" * 80)
    print("  Behavioral Life-Cycle & Alpha Dynamics:")
    print(f"    Life-Cycle Regime:          {receipt.regime}")
    print(f"    Self-Attribution Accelerating: {accel_str}")
    print(f"    Contradiction Fragility:       {fragile_str}")
    print(f"    Expected DHS Alpha Spread:  {receipt.expected_dhs_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bsv_command(args: argparse.Namespace) -> None:
    """Evaluate Barberis, Shleifer & Vishny (1998) Dual-Heuristic Investor Sentiment."""
    import json
    from src.analytics.bsv_sentiment import evaluate_bsv_sentiment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    surprise = getattr(args, "earnings_surprise", None)
    if surprise is not None:
        surprise = float(surprise)
    streak = getattr(args, "streak", None)
    if streak is not None:
        streak = int(streak)

    try:
        receipt = evaluate_bsv_sentiment_profile(
            ticker=ticker,
            latest_surprise_pct=surprise,
            consecutive_streak=streak,
        )
    except Exception as ex:
        print(f"\nError: BSV Sentiment evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    pead_str = (
        "YES (HIGH POST-EARNINGS UNDERREACTION DRIFT)"
        if receipt.is_conservatism_drift
        else "NO"
    )
    overreact_str = (
        "YES (SPECULATIVE STREAK OVERREACTION / REVERSAL)"
        if receipt.is_representativeness_peak
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing Momentum:            {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Latest Earnings Surprise:     {receipt.latest_earnings_surprise_pct:>+8.2f}%")
    print(f"  Consecutive Streak Length:    {receipt.consecutive_streak_quarters} quarters")
    print("-" * 80)
    print(f"  Trending State Prob (pi_t):   {receipt.trending_state_probability:.2f} / 1.00 (Subjective belief in persistent trend)")
    print(f"  Conservatism Drift Index:     {receipt.conservatism_drift_index:.2f} / 1.00 (PEAD underreaction magnitude)")
    print(f"  Representativeness Score:     {receipt.representativeness_streak_score:.2f} / 1.00 (Streak over-extrapolation)")
    print("-" * 80)
    print("  Cognitive Bias Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    Conservatism Underreaction: {pead_str}")
    print(f"    Representativeness Peak:    {overreact_str}")
    print(f"    Expected BSV Alpha Spread:  {receipt.expected_bsv_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def innovation_command(args: argparse.Namespace) -> None:
    """Evaluate Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Misvaluation."""
    import json
    from src.analytics.innovation_capital import evaluate_innovation_capital_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    rd = getattr(args, "rd_expense", None)
    if rd is not None:
        rd = float(rd)
    rev = getattr(args, "revenues", None)
    if rev is not None:
        rev = float(rev)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)

    try:
        receipt = evaluate_innovation_capital_profile(
            ticker=ticker,
            current_rd_expense=rd,
            annual_revenues=rev,
            total_assets=assets,
        )
    except Exception as ex:
        print(f"\nError: Innovation Capital evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    innovator_str = (
        "YES (HIGH CONVICTION INTANGIBLE INNOVATION ALPHA)"
        if receipt.is_high_innovation_alpha
        else "NO"
    )
    laggard_str = (
        "YES (HIGH OBSOLESCENCE RISK / R&D DEFICIT)"
        if receipt.is_innovation_laggard
        else "NO"
    )

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_capitalization:,.0f}")
    print("-" * 80)
    print(f"  Annual R&D Expenditure:       ${receipt.current_rd_expense:,.0f}")
    print(f"  Capitalized Knowledge Stock:  ${receipt.capitalized_knowledge_stock:,.0f} (Perpetual inventory, delta=20%)")
    print(f"  R&D-to-Market Ratio (RD/M):   {receipt.rd_to_market_pct:.2f}%")
    print(f"  R&D-to-Sales Intensity (RD/S):{receipt.rd_to_sales_pct:.2f}%")
    print(f"  Knowledge-to-Assets Ratio:    {receipt.knowledge_to_assets_pct:.2f}%")
    print("-" * 80)
    print(f"  Innovation Misvaluation Index:{receipt.innovation_misvaluation_index:.2f} / 1.00 (Unpriced intangible patent power)")
    print("  Asset Pricing Regime & Alpha Dynamics:")
    print(f"    Behavioral Regime:          {receipt.regime}")
    print(f"    High Conviction Innovator:  {innovator_str}")
    print(f"    Innovation-Starved Laggard: {laggard_str}")
    print(f"    Expected Innovation Alpha:  {receipt.expected_innovation_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def lifecycle_command(args: argparse.Namespace) -> None:
    """Evaluate Charles M. C. Lee & Bhaskaran Swaminathan (2000) Momentum Life Cycle."""
    import json
    from src.analytics.momentum_lifecycle import evaluate_momentum_lifecycle_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    turnover = getattr(args, "turnover", None)
    if turnover is not None:
        turnover = float(turnover)

    try:
        receipt = evaluate_momentum_lifecycle_profile(
            ticker=ticker,
            turnover_ratio_override=turnover,
        )
    except Exception as ex:
        print(f"\nError: Momentum Life Cycle evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    s1_str = "YES (STAGE 1 NEGLECTED WINNER - HIGH PERSISTENCE)" if receipt.is_neglected_winner else "NO"
    s2_str = "YES (STAGE 2 CROWDED WINNER - EXHAUSTION REVERSAL)" if receipt.is_crowded_winner else "NO"
    s3_str = "YES (STAGE 3 CAPITULATION LOSER - REBOUND BOUNCE)" if receipt.is_capitulation_loser else "NO"
    s4_str = "YES (STAGE 4 NEGLECTED ZOMBIE - PERSISTENT DECAY)" if receipt.is_neglected_loser else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lee & Swaminathan (2000) Momentum Life-Cycle Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print(f"  Trailing 12-M Momentum:       {receipt.trailing_momentum_pct:>+8.2f}% (Annual Volatility: {receipt.annualized_volatility_pct:.1f}%)")
    print(f"  Daily Turnover Ratio:         {receipt.daily_turnover_ratio_pct:.2f}% (Shares Transacted Daily)")
    print(f"  Volume-Momentum Score (VMIS): {receipt.volume_momentum_interaction_score:>+8.2f} (Underreaction > 0, Crowding < 0)")
    print("-" * 80)
    print("  Momentum Life-Cycle Stage & Alpha Dynamics:")
    print(f"    Life-Cycle Stage:           {receipt.regime}")
    print(f"    Stage 1 Neglected Winner:   {s1_str}")
    print(f"    Stage 2 Crowded Winner:     {s2_str}")
    print(f"    Stage 3 Capitulation Loser: {s3_str}")
    print(f"    Stage 4 Neglected Zombie:   {s4_str}")
    print(f"    Expected MLC Alpha Spread:  {receipt.expected_mlc_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def altman_command(args: argparse.Namespace) -> None:
    """Evaluate Edward I. Altman (1968, 2000) Z-Score and James A. Ohlson (1980) O-Score Solvency."""
    import json
    from src.analytics.altman_zscore import evaluate_solvency_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    assets = getattr(args, "total_assets", None)
    if assets is not None:
        assets = float(assets)
    liab = getattr(args, "total_liabilities", None)
    if liab is not None:
        liab = float(liab)
    wc = getattr(args, "working_capital", None)
    if wc is not None:
        wc = float(wc)
    re = getattr(args, "retained_earnings", None)
    if re is not None:
        re = float(re)
    ebit = getattr(args, "ebit", None)
    if ebit is not None:
        ebit = float(ebit)
    sales = getattr(args, "sales", None)
    if sales is not None:
        sales = float(sales)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)
    ca = getattr(args, "current_assets", None)
    if ca is not None:
        ca = float(ca)
    cl = getattr(args, "current_liabilities", None)
    if cl is not None:
        cl = float(cl)
    ni = getattr(args, "net_income", None)
    if ni is not None:
        ni = float(ni)
    ffo = getattr(args, "funds_from_operations", None)
    if ffo is not None:
        ffo = float(ffo)

    try:
        receipt = evaluate_solvency_profile(
            ticker=ticker,
            total_assets=assets,
            total_liabilities=liab,
            working_capital=wc,
            retained_earnings=re,
            ebit=ebit,
            sales=sales,
            market_cap=mcap,
            current_assets=ca,
            current_liabilities=cl,
            net_income=ni,
            funds_from_operations=ffo,
        )
    except Exception as ex:
        print(f"\nError: Altman Z-Score & Ohlson O-Score evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    safe_str = "YES (Z > 2.99 - MINIMAL DEFAULT HAZARD)" if receipt.is_safe_zone else "NO"
    grey_str = "YES (1.81 <= Z <= 2.99 - MONITOR BALANCE SHEET)" if receipt.is_grey_zone else "NO"
    distress_str = "YES (Z < 1.81 - ACUTE INSOLVENCY TRAP HAZARD)" if receipt.is_distress_zone else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Altman (1968, 2000) Z-Score & Ohlson (1980) O-Score Solvency Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Altman Multi-Discriminant Factors:")
    print(f"    X1 (Working Capital / Assets):  {receipt.x1_working_capital_to_assets:>+8.4f}")
    print(f"    X2 (Retained Earnings / Assets):{receipt.x2_retained_earnings_to_assets:>+8.4f}")
    print(f"    X3 (EBIT / Total Assets):       {receipt.x3_ebit_to_assets:>+8.4f}")
    print(f"    X4 (Market Equity / Liab):      {receipt.x4_market_equity_to_liabilities:>8.4f}")
    print(f"    X5 (Sales / Total Assets):      {receipt.x5_sales_to_assets:>8.4f}")
    print("-" * 80)
    print("  Solvency & Bankruptcy Hazard Inferences:")
    print(f"    Altman Manufacturing Z-Score:   {receipt.altman_z_score:>8.2f} (Safe > 2.99, Distress < 1.81)")
    print(f"    Altman Non-Mfg Z''-Score:       {receipt.altman_z_double_prime:>8.2f} (Safe > 2.60, Distress < 1.10)")
    print(f"    Ohlson O-Score Index:           {receipt.ohlson_o_score:>+8.2f}")
    print(f"    Implied 2-Yr Default Prob:      {receipt.default_probability_pct:>8.1f}% (P(Default))")
    print(f"    Safe Zone Classification:       {safe_str}")
    print(f"    Grey Zone Classification:       {grey_str}")
    print(f"    Distress Zone Classification:   {distress_str}")
    print("-" * 80)
    print("  Structural Solvency Regime & Alpha Dynamics:")
    print(f"    Solvency Regime:                {receipt.regime}")
    print(f"    Expected Solvency Alpha Spread: {receipt.expected_solvency_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def vme_command(args: argparse.Namespace) -> None:
    """Evaluate Asness, Moskowitz & Pedersen (2013) 'Value and Momentum Everywhere' (VME)."""
    import json
    from src.analytics.vme_unified import evaluate_vme_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    bm = getattr(args, "book_to_market", None)
    if bm is not None:
        bm = float(bm)
    ep = getattr(args, "earnings_yield", None)
    if ep is not None:
        ep = float(ep)
    corr = getattr(args, "correlation", None)
    if corr is not None:
        corr = float(corr)

    try:
        receipt = evaluate_vme_profile(
            ticker=ticker,
            book_to_market=bm,
            earnings_yield_pct=ep,
            value_momentum_correlation=corr,
        )
    except Exception as ex:
        print(f"\nError: Value and Momentum Everywhere (VME) evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    comp_str = "YES (TWIN-ENGINE VALUE + MOMENTUM COMPOUNDER)" if receipt.is_twin_engine_compounder else "NO"
    trap_str = "YES (EXPENSIVE + CRASHING HEADWIND TRAP)" if receipt.is_dual_headwind_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Asness, Moskowitz & Pedersen (2013) VME Unified Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Value & Momentum Raw Covariates:")
    print(f"    Book-to-Market Ratio (B/M):     {receipt.book_to_market:>8.4f}")
    print(f"    Earnings Yield (E/P):           {receipt.earnings_yield_pct:>8.2f}%")
    print(f"    12-2 Skip-Month Momentum:       {receipt.momentum_12_2_pct:>+8.2f}% (1-M Skipped: {receipt.short_term_1m_pct:+.2f}%)")
    print(f"    Annualized Volatility:          {receipt.annualized_volatility_pct:>8.2f}%")
    print("-" * 80)
    print("  Standardized Style Z-Scores & Negative Covariance Engine:")
    print(f"    Standardized Value Z-Score:     {receipt.value_z_score:>+8.2f}")
    print(f"    Standardized Momentum Z-Score:  {receipt.momentum_z_score:>+8.2f}")
    print(f"    Value-Momentum Correlation:     {receipt.value_momentum_correlation:>+8.2f} (Asness Benchmark ~ -0.52)")
    print(f"    Unified 50/50 VME Combo Score:  {receipt.vme_combo_z_score:>+8.2f}")
    print(f"    Diversification Benefit Ratio:  {receipt.diversification_benefit_ratio:>8.2f}x (sqrt(2 / (1 + rho)))")
    print(f"    Implied VME Portfolio Sharpe:   {receipt.implied_vme_sharpe:>8.2f}")
    print("-" * 80)
    print("  VME Regime & Alpha Dynamics:")
    print(f"    VME Portfolio Regime:           {receipt.regime}")
    print(f"    Twin-Engine Compounder:         {comp_str}")
    print(f"    Dual-Factor Headwind Trap:      {trap_str}")
    print(f"    Expected VME Alpha Spread:      {receipt.expected_vme_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def pead_command(args: argparse.Namespace) -> None:
    """Evaluate Victor L. Bernard & Jacob K. Thomas (1989, 1990) PEAD & SUE Autocorrelation."""
    import json
    from src.analytics.pead_sue import evaluate_pead_sue_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    cur_eps = getattr(args, "current_eps", None)
    if cur_eps is not None:
        cur_eps = float(cur_eps)
    yr_eps = getattr(args, "year_ago_eps", None)
    if yr_eps is not None:
        yr_eps = float(yr_eps)
    std_eps = getattr(args, "surprise_std", None)
    if std_eps is not None:
        std_eps = float(std_eps)
    days = getattr(args, "days_since", None)
    if days is not None:
        days = int(days)

    try:
        receipt = evaluate_pead_sue_profile(
            ticker=ticker,
            current_eps=cur_eps,
            year_ago_eps=yr_eps,
            surprise_std_eps=std_eps,
            days_since_announcement=days,
        )
    except Exception as ex:
        print(f"\nError: PEAD & SUE evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    pos_str = "YES (ACTIVE 60-SESSION UPWARD PEAD)" if receipt.is_active_positive_drift else "NO"
    neg_str = "YES (ACTIVE 60-SESSION DOWNWARD SLUMP)" if receipt.is_active_negative_slump else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bernard & Thomas (1989, 1990) PEAD & SUE Autocorrelation Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Quarterly Earnings & Surprise Standardization:")
    print(f"    Current Quarter EPS (t):        ${receipt.current_eps:>8.4f}")
    print(f"    Year-Ago Quarter EPS (t-4):     ${receipt.year_ago_eps:>8.4f}")
    print(f"    Seasonal Drift Mean (mu):       ${receipt.drift_mean_eps:>+8.4f}")
    print(f"    Surprise Volatility (sigma):    ${receipt.surprise_std_eps:>8.4f}")
    print(f"    Standardized Unexpected (SUE):  {receipt.sue_score:>+8.2f} (Decile Rank: {receipt.sue_decile} / 10)")
    print(f"    Sessions Since Announcement:    {receipt.days_since_announcement:>8d} trading days")
    print("-" * 80)
    print("  Bernard-Thomas AR(4) Seasonal Autocorrelation & Drift Dynamics:")
    print(f"    Predicted Next-Quarter SUE:     {receipt.predicted_next_quarter_sue:>+8.2f} (phi_1 = +0.34 persistence)")
    print(f"    Predicted Lag-4 Seasonal SUE:   {receipt.predicted_lag4_reversal_sue:>+8.2f} (phi_4 = -0.24 reversal)")
    print(f"    Initial 60-Day Drift Potential: {receipt.initial_60d_drift_pct:>+8.2f}%")
    print(f"    Remaining 60-Day Drift (CAR):   {receipt.remaining_drift_pct:>+8.2f}%")
    print("-" * 80)
    print("  PEAD Regime & Alpha Projections:")
    print(f"    PEAD Lifecycle Regime:          {receipt.regime}")
    print(f"    Active Positive Drift:          {pos_str}")
    print(f"    Active Negative Slump:          {neg_str}")
    print(f"    Expected PEAD Alpha Spread:     {receipt.expected_pead_alpha_pct:>+8.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def empire_command(args: argparse.Namespace) -> None:
    """Evaluate Cooper et al. (2008) & Titman et al. (2004) Asset Growth & Empire-Building."""
    import json
    from src.analytics.asset_growth_empire import evaluate_asset_growth_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    cur_assets = getattr(args, "current_assets", None)
    if cur_assets is not None:
        cur_assets = float(cur_assets)
    prior_assets = getattr(args, "prior_assets", None)
    if prior_assets is not None:
        prior_assets = float(prior_assets)
    cur_capex = getattr(args, "current_capex", None)
    if cur_capex is not None:
        cur_capex = float(cur_capex)
    hist_capex = getattr(args, "historical_capex", None)
    if hist_capex is not None:
        hist_capex = float(hist_capex)
    fcf_pct = getattr(args, "fcf_to_assets", None)
    if fcf_pct is not None:
        fcf_pct = float(fcf_pct)
    debt_pct = getattr(args, "debt_to_assets", None)
    if debt_pct is not None:
        debt_pct = float(debt_pct)

    try:
        receipt = evaluate_asset_growth_profile(
            ticker=ticker,
            current_total_assets=cur_assets,
            prior_total_assets=prior_assets,
            current_capex_to_sales_pct=cur_capex,
            historical_3yr_capex_to_sales_pct=hist_capex,
            fcf_to_assets_pct=fcf_pct,
            debt_to_assets_pct=debt_pct,
        )
    except Exception as ex:
        print(f"\nError: Asset Growth & Empire-Building evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    disc_str = "YES (DISCIPLINED CAPITAL RESTRAINT / RETURN)" if receipt.is_disciplined_allocator else "NO"
    trap_str = "YES (EMPIRE-BUILDING OVERINVESTMENT TRAP)" if receipt.is_empire_builder_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Cooper et al. (2008) & Titman et al. (2004) Asset Growth Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Balance Sheet Asset Growth & CapEx Intensity:")
    print(f"    Current Total Assets (t):       ${receipt.current_total_assets:>10,.2f}M")
    print(f"    Prior Total Assets (t-1):       ${receipt.prior_total_assets:>10,.2f}M")
    print(f"    YoY Total Asset Growth (AG):    {receipt.asset_growth_pct:>+10.2f}% (Cooper-Gulen-Schill)")
    print(f"    Current CapEx / Sales:          {receipt.current_capex_to_sales_pct:>10.2f}%")
    print(f"    3-Yr Benchmark CapEx / Sales:   {receipt.historical_3yr_capex_to_sales_pct:>10.2f}%")
    print(f"    Abnormal Capital Invest (ACI):  {receipt.abnormal_capital_investment:>+10.4f} (Titman-Wei-Xie)")
    print("-" * 80)
    print("  Jensen (1986) Agency Discretion & Empire-Building Inferences:")
    print(f"    Jensen Agency Discretion Index: {receipt.jensen_agency_discretion_index:>10.2f} / 1.00")
    print(f"    Empire-Building Overinv Score:  {receipt.empire_building_score:>+10.2f}")
    print("-" * 80)
    print("  Capital Allocation Regime & Alpha Projections:")
    print(f"    Capital Allocation Regime:      {receipt.regime}")
    print(f"    Disciplined Capital Allocator:  {disc_str}")
    print(f"    Empire-Builder Overinv Trap:    {trap_str}")
    print(f"    Expected Allocation Alpha:      {receipt.expected_allocation_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def payout_command(args: argparse.Namespace) -> None:
    """Evaluate Boudoukh et al. (2007) & Pontiff & Woodgate (2008) Net Payout Yield & Share Issuance."""
    import json
    from src.analytics.net_payout_yield import evaluate_net_payout_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    mcap = getattr(args, "market_cap", None)
    if mcap is not None:
        mcap = float(mcap)
    div = getattr(args, "dividends", None)
    if div is not None:
        div = float(div)
    buybacks = getattr(args, "repurchases", None)
    if buybacks is not None:
        buybacks = float(buybacks)
    issuance = getattr(args, "issuance", None)
    if issuance is not None:
        issuance = float(issuance)
    cur_sh = getattr(args, "current_shares", None)
    if cur_sh is not None:
        cur_sh = float(cur_sh)
    pri_sh = getattr(args, "prior_shares", None)
    if pri_sh is not None:
        pri_sh = float(pri_sh)
    fcf = getattr(args, "free_cash_flow", None)
    if fcf is not None:
        fcf = float(fcf)

    try:
        receipt = evaluate_net_payout_profile(
            ticker=ticker,
            market_cap=mcap,
            dividends=div,
            share_repurchases=buybacks,
            equity_issuance=issuance,
            current_shares=cur_sh,
            prior_shares=pri_sh,
            free_cash_flow=fcf,
        )
    except Exception as ex:
        print(f"\nError: Net Payout Yield & Share Issuance evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    can_str = "YES (SHARE-CANNIBAL COMPOUNDER)" if receipt.is_share_cannibal else "NO"
    uns_str = "YES (UNCOVERED DEBT-FUNDED PAYOUT TRAP)" if receipt.is_unsustainable_debt_payout else "NO"
    dil_str = "YES (NET EQUITY ISSUANCE / DILUTION TRAP)" if receipt.is_dilution_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Boudoukh et al. (2007) & Pontiff-Woodgate (2008) Payout Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print(f"  Market Capitalization:        ${receipt.market_cap:,.2f}M")
    print("-" * 80)
    print("  Corporate Distribution & Net Payout Yield Components:")
    print(f"    Cash Dividend Yield (D/P):      {receipt.dividend_yield_pct:>10.2f}%")
    print(f"    Gross Buyback Yield (B/P):      {receipt.gross_buyback_yield_pct:>10.2f}%")
    print(f"    Gross Issuance Dilution (I/P):  {receipt.gross_issuance_yield_pct:>10.2f}%")
    print(f"    Total Net Payout Yield (NPY):   {receipt.net_payout_yield_pct:>+10.2f}% (Boudoukh et al.)")
    print("-" * 80)
    print("  Share Count Dynamics & Free Cash Flow Coverage:")
    print(f"    Current Diluted Shares (t):     {receipt.current_shares:>10.2f}M")
    print(f"    Prior Diluted Shares (t-1):     {receipt.prior_shares:>10.2f}M")
    print(f"    Composite Share Issuance (CSI): {receipt.composite_share_issuance_pct:>+10.2f}% (Pontiff-Woodgate)")
    print(f"    FCF Payout Coverage Ratio:      {receipt.fcf_payout_coverage_ratio:>10.2f}x (Safe >= 1.00x)")
    print("-" * 80)
    print("  Payout Regime & Shareholder-Yield Alpha Dynamics:")
    print(f"    Capital Distribution Regime:    {receipt.regime}")
    print(f"    Share-Cannibal Compounder:      {can_str}")
    print(f"    Unsustainable Debt Payout:      {uns_str}")
    print(f"    Shareholder Dilution Trap:      {dil_str}")
    print(f"    Expected Shareholder Alpha:     {receipt.expected_payout_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def noa_command(args: argparse.Namespace) -> None:
    """Evaluate Hirshleifer, Hou, Teoh & Zhang (2004) Net Operating Assets (NOA) & Balance Sheet Bloat."""
    import json
    from src.analytics.noa_bloat import evaluate_noa_bloat_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ta = getattr(args, "total_assets", None)
    if ta is not None:
        ta = float(ta)
    prior_ta = getattr(args, "prior_assets", None)
    if prior_ta is not None:
        prior_ta = float(prior_ta)
    cash = getattr(args, "cash", None)
    if cash is not None:
        cash = float(cash)
    debt = getattr(args, "debt", None)
    if debt is not None:
        debt = float(debt)
    equity = getattr(args, "equity", None)
    if equity is not None:
        equity = float(equity)
    prev_snoa = getattr(args, "prior_snoa", None)
    if prev_snoa is not None:
        prev_snoa = float(prev_snoa)

    try:
        receipt = evaluate_noa_bloat_profile(
            ticker=ticker,
            total_assets=ta,
            prior_total_assets=prior_ta,
            cash_and_equivalents=cash,
            total_debt=debt,
            book_equity=equity,
            prior_snoa=prev_snoa,
        )
    except Exception as ex:
        print(f"\nError: NOA & Balance Sheet Bloat evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lean_str = "YES (LEAN CASH-GENERATIVE FORTRESS)" if receipt.is_lean_cash_fortress else "NO"
    bloat_str = "YES (ACUTE BALANCE SHEET BLOAT TRAP)" if receipt.is_bloated_balance_sheet_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hirshleifer, Hou, Teoh & Zhang (2004) NOA Bloat Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Balance Sheet Operating Decomposition ($M):")
    print(f"    Current Total Assets (t):       ${receipt.total_assets:>10,.2f}M")
    print(f"    Prior Total Assets (t-1):       ${receipt.prior_total_assets:>10,.2f}M")
    print(f"    Cash & Short-Term Investments:  ${receipt.cash_and_equivalents:>10,.2f}M")
    print(f"    Total Short & Long-Term Debt:   ${receipt.total_debt:>10,.2f}M")
    print(f"    Shareholders' Book Equity:      ${receipt.book_equity:>10,.2f}M")
    print(f"    Operating Assets (OA):          ${receipt.operating_assets:>10,.2f}M")
    print(f"    Operating Liabilities (OL):     ${receipt.operating_liabilities:>10,.2f}M")
    print(f"    Net Operating Assets (NOA):     ${receipt.net_operating_assets:>10,.2f}M (OA - OL)")
    print("-" * 80)
    print("  Scaled NOA Ratios & Cash Conversion Efficiency:")
    print(f"    Scaled NOA Ratio (SNOA):        {receipt.scaled_noa_ratio:>+10.4f} (NOA / Prior TA)")
    print(f"    YoY Change in SNOA (dSNOA):     {receipt.delta_snoa:>+10.4f}")
    print(f"    Cash Conversion Efficiency:     {receipt.cash_conversion_efficiency:>10.2f} / 1.00")
    print("-" * 80)
    print("  Balance Sheet Bloat Regime & Alpha Projections:")
    print(f"    NOA Accounting Regime:          {receipt.regime}")
    print(f"    Lean Cash-Generative Fortress:  {lean_str}")
    print(f"    Acute Balance Sheet Bloat Trap: {bloat_str}")
    print(f"    Expected NOA Alpha Spread:      {receipt.expected_noa_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dispersion_command(args: argparse.Namespace) -> None:
    """Evaluate Diether, Malloy & Scherbina (2002) & Miller (1977) Analyst Forecast Dispersion."""
    import json
    from src.analytics.belief_dispersion import evaluate_belief_dispersion_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    mean_eps = getattr(args, "mean_eps", None)
    if mean_eps is not None:
        mean_eps = float(mean_eps)
    std_eps = getattr(args, "std_eps", None)
    if std_eps is not None:
        std_eps = float(std_eps)
    analysts = getattr(args, "analysts", None)
    if analysts is not None:
        analysts = int(analysts)
    fee = getattr(args, "borrow_fee", None)
    if fee is not None:
        fee = float(fee)
    inst_own = getattr(args, "inst_ownership", None)
    if inst_own is not None:
        inst_own = float(inst_own)

    try:
        receipt = evaluate_belief_dispersion_profile(
            ticker=ticker,
            consensus_mean_eps=mean_eps,
            forecast_std_eps=std_eps,
            analyst_count=analysts,
            short_borrow_fee_pct=fee,
            institutional_ownership_pct=inst_own,
        )
    except Exception as ex:
        print(f"\nError: Analyst Forecast Dispersion evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    cons_str = "YES (HIGH-CONSENSUS COMPOUNDER)" if receipt.is_high_consensus_compounder else "NO"
    trap_str = "YES (MILLER OPTIMISM OVERPRICING TRAP)" if receipt.is_miller_overpricing_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Diether et al. (2002) & Miller (1977) Belief Dispersion Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Analyst Consensus & Short-Sale Friction Inputs:")
    print(f"    Consensus Mean EPS Forecast:    ${receipt.consensus_mean_eps:>10.4f}")
    print(f"    Forecast Std Deviation (sigma): ${receipt.forecast_std_eps:>10.4f}")
    print(f"    Covering Analyst Count (N):     {receipt.analyst_count:>10d} analysts")
    print(f"    Annual Short Borrow Fee:        {receipt.short_borrow_fee_pct:>10.2f}%")
    print(f"    Institutional Ownership:        {receipt.institutional_ownership_pct:>10.2f}%")
    print("-" * 80)
    print("  Belief Dispersion & Miller (1977) Overpricing Inferences:")
    print(f"    Diether-Malloy Dispersion (Dt): {receipt.dispersion_ratio:>10.4f} (sigma / |Mean EPS|)")
    print(f"    Short-Sale Friction Index:      {receipt.short_sale_friction_index:>10.2f} / 1.00")
    print(f"    Miller Overpricing Wedge:       {receipt.miller_overpricing_wedge:>10.2f} / 3.50")
    print("-" * 80)
    print("  Dispersion Regime & Alpha Projections:")
    print(f"    Belief Dispersion Regime:       {receipt.regime}")
    print(f"    High-Consensus Compounder:      {cons_str}")
    print(f"    Miller Overpricing Trap:        {trap_str}")
    print(f"    Expected Dispersion Alpha:      {receipt.expected_dispersion_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def catering_command(args: argparse.Namespace) -> None:
    """Evaluate Polk & Sapienza (2009) Catering Theory of Investment & Valuation Mispricing."""
    import json
    from src.analytics.catering_investment import evaluate_catering_investment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    capex = getattr(args, "capex", None)
    if capex is not None:
        capex = float(capex)
    ppent_lag = getattr(args, "ppent_lag", None)
    if ppent_lag is not None:
        ppent_lag = float(ppent_lag)
    tobins_q = getattr(args, "tobins_q", None)
    if tobins_q is not None:
        tobins_q = float(tobins_q)
    cf_ratio = getattr(args, "cash_flow_ratio", None)
    if cf_ratio is not None:
        cf_ratio = float(cf_ratio)
    mispricing = getattr(args, "mispricing_score", None)
    if mispricing is not None:
        mispricing = float(mispricing)
    turnover = getattr(args, "turnover_pct", None)
    if turnover is not None:
        turnover = float(turnover)

    try:
        receipt = evaluate_catering_investment_profile(
            ticker=ticker,
            capex_amount=capex,
            lagged_net_ppe=ppent_lag,
            tobins_q=tobins_q,
            cash_flow_to_capital=cf_ratio,
            discretionary_mispricing_score=mispricing,
            annual_turnover_pct=turnover,
        )
    except Exception as ex:
        print(f"\nError: Catering Investment evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    disc_str = "YES (DISCIPLINED FUNDAMENTAL ALLOCATOR)" if receipt.is_disciplined_allocator else "NO"
    trap_str = "YES (OVERVALUED CATERING CAPEX TRAP)" if receipt.is_catering_capex_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Polk & Sapienza (2009) Catering Theory of Investment Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Corporate Investment & Valuation Mispricing Inputs:")
    print(f"    Annual Capital Expenditure:     ${receipt.capex_amount:>10,.2f}M")
    print(f"    Lagged Net PPE (K_t-1):         ${receipt.lagged_net_ppe:>10,.2f}M")
    print(f"    Tobin's Q Valuation Ratio:      {receipt.tobins_q:>10.2f}x")
    print(f"    Cash Flow / Lagged PPE Ratio:   {receipt.cash_flow_to_capital:>10.2f}")
    print(f"    Discretionary Mispricing (DMP): {receipt.discretionary_mispricing_score:>+10.2f}")
    print(f"    Annual Share Turnover:          {receipt.annual_turnover_pct:>10.1f}%")
    print("-" * 80)
    print("  Polk-Sapienza (2009) Catering Decomposition:")
    print(f"    Investment-to-Capital (IK_t):   {receipt.investment_to_capital_ratio:>10.4f}")
    print(f"    Fundamental Q Benchmark Rate:   {receipt.fundamental_q_investment_rate:>10.4f}")
    print(f"    Residual Abnormal Investment:   {receipt.residual_abnormal_investment:>+10.4f}")
    print(f"    Short-Horizon Turnover Mult.:   {receipt.short_horizon_turnover_multiplier:>10.2f}x")
    print(f"    Catering Distortion Index:      {receipt.catering_distortion_index:>+10.4f}")
    print("-" * 80)
    print("  Catering Regime & Alpha Projections:")
    print(f"    Catering Investment Regime:     {receipt.regime}")
    print(f"    Disciplined Allocator:          {disc_str}")
    print(f"    Overvalued Catering CapEx Trap: {trap_str}")
    print(f"    Expected Catering Alpha:        {receipt.expected_catering_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def commonality_command(args: argparse.Namespace) -> None:
    """Evaluate Chordia, Roll & Subrahmanyam (2000, 2001, 2002) Commonality in Liquidity & Order Imbalance."""
    import json
    from src.analytics.liquidity_commonality import evaluate_liquidity_commonality_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    spread_beta = getattr(args, "spread_beta", None)
    if spread_beta is not None:
        spread_beta = float(spread_beta)
    depth_beta = getattr(args, "depth_beta", None)
    if depth_beta is not None:
        depth_beta = float(depth_beta)
    liq_r2 = getattr(args, "liquidity_r2", None)
    if liq_r2 is not None:
        liq_r2 = float(liq_r2)
    oib = getattr(args, "order_imbalance", None)
    if oib is not None:
        oib = float(oib)
    oib_ac = getattr(args, "oib_autocorr", None)
    if oib_ac is not None:
        oib_ac = float(oib_ac)
    down_surge = getattr(args, "down_market_surge", None)
    if down_surge is not None:
        down_surge = float(down_surge)

    try:
        receipt = evaluate_liquidity_commonality_profile(
            ticker=ticker,
            spread_commonality_beta=spread_beta,
            depth_commonality_beta=depth_beta,
            liquidity_market_r2=liq_r2,
            signed_order_imbalance=oib,
            oib_autocorrelation=oib_ac,
            down_market_illiquidity_surge=down_surge,
        )
    except Exception as ex:
        print(f"\nError: Liquidity Commonality evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    haven_str = "YES (IDIOSYNCRATIC LIQUIDITY HAVEN)" if receipt.is_idiosyncratic_liquidity_haven else "NO"
    trap_str = "YES (SYSTEMIC LIQUIDITY CONTAGION TRAP)" if receipt.is_systemic_contagion_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Chordia et al. (2000, 2001) Liquidity Commonality Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Microstructure Co-Movement & Order Imbalance Inputs:")
    print(f"    Spread Commonality Beta:        {receipt.spread_commonality_beta:>+10.4f}")
    print(f"    Depth Commonality Beta:         {receipt.depth_commonality_beta:>+10.4f}")
    print(f"    Market Liquidity Model R^2:     {receipt.liquidity_market_r2:>10.4f}")
    print(f"    Signed Order Imbalance (OIB):   {receipt.signed_order_imbalance:>+10.4f}")
    print(f"    OIB First-Order Autocorrelation:{receipt.oib_autocorrelation:>+10.4f}")
    print(f"    Down-Market Illiquidity Surge:  {receipt.down_market_illiquidity_surge:>10.2f}x")
    print("-" * 80)
    print("  Chordia-Roll-Subrahmanyam Microstructure Inferences:")
    print(f"    Effective Commonality Beta:     {receipt.effective_commonality_beta:>+10.4f}")
    print(f"    Order-Splitting Pressure Score: {receipt.order_splitting_pressure_score:>10.4f} / 2.00")
    print(f"    Systemic Inventory Contagion:   {receipt.systemic_inventory_contagion_index:>10.4f} / 3.50")
    print("-" * 80)
    print("  Commonality Regime & Execution Alpha Projections:")
    print(f"    Liquidity Commonality Regime:   {receipt.regime}")
    print(f"    Idiosyncratic Liquidity Haven:  {haven_str}")
    print(f"    Systemic Contagion Trap:        {trap_str}")
    print(f"    Expected Commonality Alpha:     {receipt.expected_commonality_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def supply_chain_command(args: argparse.Namespace) -> None:
    """Evaluate Cohen & Frazzini (2008) & Ahern (2013) Economic Links & Supply-Chain Customer Momentum."""
    import json
    from src.analytics.supply_chain_momentum import evaluate_supply_chain_momentum_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    cust_ret = getattr(args, "customer_return", None)
    if cust_ret is not None:
        cust_ret = float(cust_ret)
    supp_ret = getattr(args, "supplier_return", None)
    if supp_ret is not None:
        supp_ret = float(supp_ret)
    rev_share = getattr(args, "revenue_share", None)
    if rev_share is not None:
        rev_share = float(rev_share)
    shared_cov = getattr(args, "shared_coverage", None)
    if shared_cov is not None:
        shared_cov = float(shared_cov)
    centrality = getattr(args, "network_centrality", None)
    if centrality is not None:
        centrality = float(centrality)

    try:
        receipt = evaluate_supply_chain_momentum_profile(
            ticker=ticker,
            customer_portfolio_return=cust_ret,
            supplier_own_return=supp_ret,
            customer_revenue_share=rev_share,
            shared_coverage_ratio=shared_cov,
            network_centrality_score=centrality,
        )
    except Exception as ex:
        print(f"\nError: Supply-Chain Momentum evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (POSITIVE CUSTOMER SPILLOVER LEADER)" if receipt.is_positive_spillover_leader else "NO"
    trap_str = "YES (UPSTREAM DEMAND COLLAPSE TRAP)" if receipt.is_upstream_demand_collapse_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Cohen & Frazzini (2008) & Ahern (2013) Supply-Chain Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Economic Links & Production Network Inputs:")
    print(f"    Principal Customer Return:      {receipt.customer_portfolio_return:>+10.2%}")
    print(f"    Supplier Own Return:            {receipt.supplier_own_return:>+10.2%}")
    print(f"    Customer Revenue Share (w_C):   {receipt.customer_revenue_share:>10.1%}")
    print(f"    Shared Analyst Coverage Ratio:  {receipt.shared_coverage_ratio:>10.1%}")
    print(f"    Ahern Network Centrality Score: {receipt.network_centrality_score:>10.2f} / 1.00")
    print("-" * 80)
    print("  Cohen-Frazzini Limited-Attention Spillover Inferences:")
    print(f"    Economic Link Passthrough Beta: {receipt.economic_link_beta:>10.4f}")
    print(f"    Unpriced Return Wedge (CSRW):   {receipt.customer_supplier_return_wedge:>+10.4f}")
    print(f"    Inattention Diffusion Mult.:    {receipt.inattention_diffusion_multiplier:>10.2f}x")
    print(f"    Limited-Attention Impulse (LASI):{receipt.limited_attention_spillover_impulse:>+9.4f}")
    print("-" * 80)
    print("  Supply-Chain Regime & Alpha Projections:")
    print(f"    Supply-Chain Momentum Regime:   {receipt.regime}")
    print(f"    Positive Spillover Leader:      {lead_str}")
    print(f"    Upstream Demand Collapse Trap:  {trap_str}")
    print(f"    Expected Supply-Chain Alpha:    {receipt.expected_supply_chain_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def governance_command(args: argparse.Namespace) -> None:
    """Evaluate Gompers, Ishii & Metrick (2003) & Bebchuk et al. (2009) Corporate Governance & Entrenchment."""
    import json
    from src.analytics.governance_entrenchment import evaluate_governance_entrenchment_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    e_idx = getattr(args, "e_index", None)
    if e_idx is not None:
        e_idx = int(e_idx)
    g_idx = getattr(args, "g_index", None)
    if g_idx is not None:
        g_idx = int(g_idx)
    wedge = getattr(args, "dual_class_wedge", None)
    if wedge is not None:
        wedge = float(wedge)
    indep = getattr(args, "board_independence", None)
    if indep is not None:
        indep = float(indep)
    fcf_ret = getattr(args, "fcf_retention", None)
    if fcf_ret is not None:
        fcf_ret = float(fcf_ret)

    try:
        receipt = evaluate_governance_entrenchment_profile(
            ticker=ticker,
            bebchuk_e_index=e_idx,
            gompers_g_index=g_idx,
            dual_class_voting_wedge=wedge,
            board_independence_ratio=indep,
            unreturned_fcf_ratio=fcf_ret,
        )
    except Exception as ex:
        print(f"\nError: Corporate Governance evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    dem_str = "YES (SHAREHOLDER DEMOCRACY COMPOUNDER)" if receipt.is_shareholder_democracy else "NO"
    dict_str = "YES (MANAGERIAL DICTATORSHIP TRAP)" if receipt.is_managerial_dictatorship_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Gompers et al. (2003) & Bebchuk et al. (2009) Governance Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Constitutional Charter & Managerial Agency Inputs:")
    print(f"    Bebchuk Entrenchment (E-Index): {receipt.bebchuk_e_index:>10d} / 6")
    print(f"    Gompers Governance (G-Index):   {receipt.gompers_g_index:>10d} / 24")
    print(f"    Other IRRC Provisions (G - E):  {receipt.other_irrc_provisions:>10d} / 18")
    print(f"    Dual-Class Voting Wedge:        {receipt.dual_class_voting_wedge:>10.1%}")
    print(f"    Board Independence Ratio:       {receipt.board_independence_ratio:>10.1%}")
    print(f"    Unreturned Free Cash Flow:      {receipt.unreturned_fcf_ratio:>10.1%}")
    print("-" * 80)
    print("  Entrenchment & Jensen (1986) Agency Cost Inferences:")
    print(f"    Composite Entrenchment (CMES):  {receipt.composite_entrenchment_score:>10.4f} / 1.00")
    print(f"    Agency Cost Wedge Index (ACWI): {receipt.agency_cost_wedge_index:>10.4f} / 2.50")
    print("-" * 80)
    print("  Governance Regime & Alpha Projections:")
    print(f"    Corporate Governance Regime:    {receipt.regime}")
    print(f"    Shareholder Democracy:          {dem_str}")
    print(f"    Managerial Dictatorship Trap:   {dict_str}")
    print(f"    Expected Governance Alpha:      {receipt.expected_governance_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def fip_command(args: argparse.Namespace) -> None:
    """Evaluate Da, Gurun & Warachka (2014) 'Frog in the Pan' Information Discreteness & Momentum Quality."""
    import json
    from src.analytics.frog_in_the_pan import evaluate_frog_in_the_pan_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    pret = getattr(args, "pret", None)
    if pret is not None:
        pret = float(pret)
    pct_pos = getattr(args, "pct_positive", None)
    if pct_pos is not None:
        pct_pos = float(pct_pos)
    pct_neg = getattr(args, "pct_negative", None)
    if pct_neg is not None:
        pct_neg = float(pct_neg)
    jump_share = getattr(args, "max_jump_share", None)
    if jump_share is not None:
        jump_share = float(jump_share)

    try:
        receipt = evaluate_frog_in_the_pan_profile(
            ticker=ticker,
            past_cumulative_return=pret,
            pct_positive_days=pct_pos,
            pct_negative_days=pct_neg,
            max_three_day_jump_share=jump_share,
        )
    except Exception as ex:
        print(f"\nError: Frog in the Pan evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    win_str = "YES (CONTINUOUS BOILING FROG WINNER)" if receipt.is_continuous_frog_winner else "NO"
    trap_str = "YES (CONTINUOUS BOILING FROG LOSER TRAP)" if receipt.is_continuous_frog_loser_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Da, Gurun & Warachka (2014) 'Frog in the Pan' (FIP) Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Formation Return & Daily Sign Frequency Inputs:")
    print(f"    Past Cumulative Return (PRET):  {receipt.past_cumulative_return:>+10.2%}")
    print(f"    Positive Sessions (%Pos):       {receipt.pct_positive_days:>10.1%}")
    print(f"    Negative Sessions (%Neg):       {receipt.pct_negative_days:>10.1%}")
    print(f"    Flat/Zero Sessions (%Zero):     {receipt.pct_zero_days:>10.1%}")
    print(f"    Top 3-Day Jump Concentration:   {receipt.max_three_day_jump_share:>10.1%}")
    print("-" * 80)
    print("  Da-Gurun-Warachka (2014) Information Discreteness Inferences:")
    print(f"    Information Discreteness (ID):  {receipt.information_discreteness_id:>+10.4f} [-1=Cont, +1=Disc]")
    print(f"    FIP Momentum Quality (FMQS):    {receipt.fip_momentum_quality_score:>10.4f} / 1.00")
    print("-" * 80)
    print("  FIP Regime & Momentum Continuation Projections:")
    print(f"    Frog in the Pan Regime:         {receipt.regime}")
    print(f"    Continuous Frog Winner:         {win_str}")
    print(f"    Continuous Frog Loser Trap:     {trap_str}")
    print(f"    Expected FIP Continuation Alpha:{receipt.expected_fip_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def attention_command(args: argparse.Namespace) -> None:
    """Evaluate Barber & Odean (2008) & Da, Engelberg & Gao (2011) Retail Attention Spike & Reversal."""
    import json
    from src.analytics.attention_salience import evaluate_attention_salience_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    asvi = getattr(args, "asvi", None)
    if asvi is not None:
        asvi = float(asvi)
    abn_vol = getattr(args, "abnormal_volume", None)
    if abn_vol is not None:
        abn_vol = float(abn_vol)
    ext_ret = getattr(args, "extreme_return", None)
    if ext_ret is not None:
        ext_ret = float(ext_ret)
    ret_imb = getattr(args, "retail_buy_imbalance", None)
    if ret_imb is not None:
        ret_imb = float(ret_imb)

    try:
        receipt = evaluate_attention_salience_profile(
            ticker=ticker,
            abnormal_svi_log_change=asvi,
            abnormal_volume_ratio=abn_vol,
            extreme_day_return=ext_ret,
            retail_buy_imbalance=ret_imb,
        )
    except Exception as ex:
        print(f"\nError: Retail Attention Salience evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    comp_str = "YES (UNDER-THE-RADAR STEADY COMPOUNDER)" if receipt.is_under_the_radar_compounder else "NO"
    trap_str = "YES (GLITTER ATTENTION BUYING CLIMAX TRAP)" if receipt.is_glitter_buying_climax_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barber & Odean (2008) & Da et al. (2011) Attention Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Retail Search Attention & Salience Inputs:")
    print(f"    Abnormal Search Volume (ASVI):  {receipt.abnormal_svi_log_change:>+10.4f} (log vs 8-wk median)")
    print(f"    Abnormal Trading Volume Ratio:  {receipt.abnormal_volume_ratio:>10.2f}x")
    print(f"    Extreme 1-Day Salient Return:   {receipt.extreme_day_return:>+10.2%}")
    print(f"    Retail Buy Order Imbalance:     {receipt.retail_buy_imbalance:>+10.2f}")
    print("-" * 80)
    print("  Barber-Odean & Da-Engelberg-Gao Inferences:")
    print(f"    Barber-Odean Salience (BASS):   {receipt.barber_odean_salience_score:>10.4f} / 3.00")
    print(f"    Retail Reversal Overhang (ARRO):{receipt.retail_reversal_overhang_index:>10.4f} / 3.50")
    print(f"    1-2 Wk Transient Price Boost:   +{receipt.short_term_pressure_bps:>9.1f} bps")
    print("-" * 80)
    print("  Attention Regime & Post-Spike Reversal Projections:")
    print(f"    Attention Salience Regime:      {receipt.regime}")
    print(f"    Under-the-Radar Compounder:     {comp_str}")
    print(f"    Glitter Buying Climax Trap:     {trap_str}")
    print(f"    Expected Reversal-Adj. Alpha:   {receipt.expected_reversal_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def benchmark_mandate_command(args: argparse.Namespace) -> None:
    """Evaluate Baker, Bradley & Wurgler (2011) Institutional Benchmark Mandate & Tracking-Error Engine."""
    import json
    from src.analytics.benchmark_mandate import evaluate_benchmark_mandate_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    beta = getattr(args, "beta", None)
    if beta is not None:
        beta = float(beta)
    mkt_vol = getattr(args, "market_vol", None)
    if mkt_vol is not None:
        mkt_vol = float(mkt_vol)
    idio_vol = getattr(args, "idio_vol", None)
    if idio_vol is not None:
        idio_vol = float(idio_vol)
    erp = getattr(args, "market_premium", None)
    if erp is not None:
        erp = float(erp)
    inst_share = getattr(args, "inst_benchmarked_share", None)
    if inst_share is not None:
        inst_share = float(inst_share)

    try:
        receipt = evaluate_benchmark_mandate_profile(
            ticker=ticker,
            capm_beta=beta,
            market_volatility_ann=mkt_vol,
            idiosyncratic_volatility_ann=idio_vol,
            expected_market_risk_premium_pct=erp,
            benchmarked_institutional_share=inst_share,
        )
    except Exception as ex:
        print(f"\nError: Institutional Benchmark Mandate evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    haven_str = "YES (NEGLECTED LOW-BETA ALPHA HAVEN)" if receipt.is_neglected_low_beta_haven else "NO"
    trap_str = "YES (CROWDED HIGH-BETA LEVERAGE TRAP)" if receipt.is_crowded_high_beta_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Baker, Bradley & Wurgler (2011) Benchmark Mandate Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price:                 ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Beta, Volatility & Institutional Mandate Inputs:")
    print(f"    CAPM Beta (beta_i):             {receipt.capm_beta:>10.2f}")
    print(f"    Market Volatility (sigma_M):    {receipt.market_volatility_ann:>10.1%}")
    print(f"    Idiosyncratic Vol (sigma_e):    {receipt.idiosyncratic_volatility_ann:>10.1%}")
    print(f"    Expected Market Risk Premium:   {receipt.expected_market_risk_premium_pct:>10.2f}%")
    print(f"    Benchmarked Institutional Share:{receipt.benchmarked_institutional_share:>10.1%}")
    print("-" * 80)
    print("  Baker-Bradley-Wurgler (2011) Tracking-Error Inferences:")
    print(f"    Single-Stock Tracking Error:    {receipt.single_stock_tracking_error:>10.2%}")
    print(f"    Active Return Alpha Hurdle:     {receipt.active_return_alpha_hurdle_pct:>+10.2f}% / yr")
    print(f"    Institutional Crowding Wedge:   {receipt.institutional_crowding_wedge:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Mandate Regime & Unconstrained Alpha Projections:")
    print(f"    Benchmark Mandate Regime:       {receipt.regime}")
    print(f"    Neglected Low-Beta Haven:       {haven_str}")
    print(f"    Crowded High-Beta Trap:         {trap_str}")
    print(f"    Expected Unconstrained Alpha:   {receipt.expected_unconstrained_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def cgo_command(args: argparse.Namespace) -> None:
    """Evaluate Grinblatt & Han (2005) & Frazzini (2006) Capital Gains Overhang & Disposition Effect."""
    import json
    from src.analytics.capital_gains_overhang import evaluate_capital_gains_overhang_profile
    from src.security.validators import validate_ticker

    ticker = validate_ticker(args.ticker)
    ref_price = getattr(args, "reference_price", None)
    if ref_price is not None:
        ref_price = float(ref_price)
    turnover = getattr(args, "daily_turnover", None)
    if turnover is not None:
        turnover = float(turnover)
    news = getattr(args, "news_surprise", None)
    if news is not None:
        news = float(news)
    shield = getattr(args, "institutional_disposition_shield", None)
    if shield is not None:
        shield = float(shield)

    try:
        receipt = evaluate_capital_gains_overhang_profile(
            ticker=ticker,
            aggregate_reference_price=ref_price,
            avg_daily_turnover=turnover,
            news_surprise_score=news,
            institutional_rationality_shield=shield,
        )
    except Exception as ex:
        print(f"\nError: Capital Gains Overhang evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (GAIN OVERHANG CONTINUATION LEADER)" if receipt.is_gain_overhang_leader else "NO"
    trap_str = "YES (LOSS-AVERSION LOCK-IN TRAP)" if receipt.is_loss_lock_in_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Grinblatt & Han (2005) & Frazzini (2006) CGO Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Mental-Accounting Cost Basis & Fundamental News Inputs:")
    print(f"    Aggregate Reference Price (R_t):${receipt.aggregate_reference_price:>10.2f}")
    print(f"    Average Daily Share Turnover:   {receipt.avg_daily_turnover:>10.2%}")
    print(f"    Fundamental News Surprise:      {receipt.news_surprise_score:>+10.2f} SD")
    print(f"    Institutional Rationality Shield:{receipt.institutional_rationality_shield:>9.1%}")
    print("-" * 80)
    print("  Grinblatt-Han & Frazzini Disposition Inferences:")
    print(f"    Capital Gains Overhang (CGO_t): {receipt.capital_gains_overhang:>+10.2%}")
    print(f"    News x CGO Alignment Mult.:     {receipt.news_overhang_alignment_mult:>10.2f}x")
    print(f"    Disposition Impulse (FDUI_t):   {receipt.frazzini_disposition_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Disposition Regime & Drift Alpha Projections:")
    print(f"    Capital Gains Overhang Regime:  {receipt.regime}")
    print(f"    Gain Overhang Leader:           {lead_str}")
    print(f"    Loss-Aversion Lock-In Trap:     {trap_str}")
    print(f"    Expected Disposition Alpha:     {receipt.expected_cgo_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def high52_command(args: argparse.Namespace) -> None:
    """CLI handler for George & Hwang (2004) 52-Week High Psychological Anchoring Engine."""
    import json
    from src.analytics.fifty_two_week_high import evaluate_fifty_two_week_high_profile

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    high_52w = getattr(args, "high_52w", None)
    if high_52w is not None:
        high_52w = float(high_52w)
    low_52w = getattr(args, "low_52w", None)
    if low_52w is not None:
        low_52w = float(low_52w)
    past_ret = getattr(args, "past_return", None)
    if past_ret is not None:
        past_ret = float(past_ret)
    vol_ratio = getattr(args, "breakout_volume_ratio", None)
    if vol_ratio is not None:
        vol_ratio = float(vol_ratio)

    try:
        receipt = evaluate_fifty_two_week_high_profile(
            ticker=ticker,
            high_52w=high_52w,
            low_52w=low_52w,
            past_return_20d=past_ret,
            breakout_volume_ratio=vol_ratio,
        )
    except Exception as ex:
        print(f"\nError: 52-Week High Anchoring evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (52W-HIGH ANCHOR BREAKOUT LEADER)" if receipt.is_52w_high_breakout_leader else "NO"
    trap_str = "YES (DEEP 52W-ANCHOR OVERHANG TRAP)" if receipt.is_deep_anchor_overhang_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: George & Hwang (2004) 52-Week High Anchoring Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  52-Week Extremum & Past Return Inputs:")
    print(f"    52-Week High Anchor (H_52,t):   ${receipt.high_52w:>10.2f}")
    print(f"    52-Week Low Bound (L_52,t):     ${receipt.low_52w:>10.2f}")
    print(f"    Recent 20d Return (PRET_t):     {receipt.past_return_20d:>+10.2%}")
    print(f"    Breakout Volume Ratio:          {receipt.breakout_volume_ratio:>10.2f}x")
    print("-" * 80)
    print("  George-Hwang Psychological Anchoring Inferences:")
    print(f"    Nearness to 52W High (NH_52,t): {receipt.nearness_to_52w_high:>10.2%}")
    print(f"    52-Week Range Position:         {receipt.range_position_52w:>10.2%}")
    print(f"    George-Hwang Impulse (GHAI_t):  {receipt.george_hwang_anchoring_impulse:>+10.4f} [-2.0..+2.5]")
    print("-" * 80)
    print("  Anchoring Regime & Drift Alpha Projections:")
    print(f"    52-Week High Regime:            {receipt.regime}")
    print(f"    52W-High Breakout Leader:       {lead_str}")
    print(f"    Deep Anchor Overhang Trap:      {trap_str}")
    print(f"    Expected Anchoring Alpha:       {receipt.expected_anchoring_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def pastor_veronesi_command(args: argparse.Namespace) -> None:
    """CLI handler for Pástor & Veronesi (2003, 2006) Learning & Uncertainty Convexity Engine."""
    import json
    from src.analytics.pastor_veronesi_uncertainty import (
        evaluate_pastor_veronesi_uncertainty_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    psi = getattr(args, "expected_excess_profitability", None)
    if psi is not None:
        psi = float(psi)
    sigma0 = getattr(args, "prior_uncertainty", None)
    if sigma0 is not None:
        sigma0 = float(sigma0)
    age = getattr(args, "firm_age_years", None)
    if age is not None:
        age = float(age)
    ret = getattr(args, "retention_ratio", None)
    if ret is not None:
        ret = float(ret)
    mb = getattr(args, "observed_mb", None)
    if mb is not None:
        mb = float(mb)

    try:
        receipt = evaluate_pastor_veronesi_uncertainty_profile(
            ticker=ticker,
            expected_excess_profitability=psi,
            prior_profitability_uncertainty=sigma0,
            firm_learning_age_years=age,
            retention_ratio=ret,
            observed_market_to_book=mb,
        )
    except Exception as ex:
        print(f"\nError: Pástor-Veronesi Uncertainty evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (CONVEX GROWTH LEARNING LEADER)" if receipt.is_convex_learning_leader else "NO"
    trap_str = "YES (EXTRAPOLATIVE UNCERTAINTY BUBBLE TRAP)" if receipt.is_uncertainty_bubble_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Pástor & Veronesi (2003, 2006) Learning & Convexity Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Fundamental Profitability & Bayesian Learning Inputs:")
    print(f"    Expected Excess ROE (psi_hat):  {receipt.expected_excess_profitability:>+10.2%}")
    print(f"    Prior Uncertainty (sigma_0):    {receipt.prior_profitability_uncertainty:>10.2%}")
    print(f"    Firm Learning Age (T_age):      {receipt.firm_learning_age_years:>10.2f} yrs")
    print(f"    Earnings Retention (b_ret):     {receipt.retention_ratio:>10.2%}")
    print(f"    Observed Market-to-Book (M/B):  {receipt.observed_market_to_book:>10.2f}x")
    print("-" * 80)
    print("  Pástor-Veronesi Bayesian & Convexity Inferences:")
    print(f"    Posterior Uncertainty (sigma):  {receipt.posterior_profitability_uncertainty:>10.2%}")
    print(f"    Convexity Multiplier (C_PV):    {receipt.convexity_valuation_multiplier:>10.2f}x")
    print(f"    Warranted M/B ((M/B)^*_PV):     {receipt.warranted_market_to_book:>10.2f}x")
    print(f"    Valuation Mispricing Gap:       {receipt.valuation_mispricing_gap_pct:>+10.2%}")
    print(f"    Learning Impulse (PVLI_t):      {receipt.pastor_veronesi_learning_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Learning Regime & Valuation Alpha Projections:")
    print(f"    Pástor-Veronesi Regime:         {receipt.regime}")
    print(f"    Convex Learning Leader:         {lead_str}")
    print(f"    Uncertainty Bubble Trap:        {trap_str}")
    print(f"    Expected Learning Alpha:        {receipt.expected_learning_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def lrr_command(args: argparse.Namespace) -> None:
    """CLI handler for Bansal & Yaron (2004) Long-Run Risks (LRR) Pricing Engine."""
    import json
    from src.analytics.long_run_risks import evaluate_long_run_risks_profile

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    bx = getattr(args, "cashflow_growth_beta", None)
    if bx is not None:
        bx = float(bx)
    bsig = getattr(args, "uncertainty_shock_beta", None)
    if bsig is not None:
        bsig = float(bsig)
    rho = getattr(args, "growth_persistence", None)
    if rho is not None:
        rho = float(rho)
    ies = getattr(args, "ies", None)
    if ies is not None:
        ies = float(ies)
    rra = getattr(args, "risk_aversion", None)
    if rra is not None:
        rra = float(rra)

    try:
        receipt = evaluate_long_run_risks_profile(
            ticker=ticker,
            cashflow_growth_beta=bx,
            uncertainty_shock_beta=bsig,
            growth_persistence_rho=rho,
            intertemporal_elasticity_ies=ies,
            relative_risk_aversion=rra,
        )
    except Exception as ex:
        print(f"\nError: Long-Run Risks evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (HIGH LRR CASH-FLOW GROWTH LEADER)" if receipt.is_high_lrr_growth_premium_leader else "NO"
    hedge_str = "YES (MACRO UNCERTAINTY SAFE-HAVEN HEDGE)" if receipt.is_macro_uncertainty_safe_haven else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Bansal & Yaron (2004) Long-Run Risks (LRR) Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Cash-Flow Betas & Epstein-Zin Preference Inputs:")
    print(f"    Cash-Flow Growth Beta (beta_x): {receipt.cashflow_growth_beta:>+10.2f}")
    print(f"    Uncertainty Shock Beta (beta_s):{receipt.uncertainty_shock_beta:>+10.2f}")
    print(f"    Growth Persistence (rho_x):     {receipt.growth_persistence_rho:>10.3f}")
    print(f"    Epstein-Zin IES (psi_IES):      {receipt.intertemporal_elasticity_ies:>10.2f}")
    print(f"    Relative Risk Aversion (gamma): {receipt.relative_risk_aversion:>10.2f}")
    print("-" * 80)
    print("  Bansal-Yaron Long-Run Risks Inferences:")
    print(f"    Early Resolution Pref (Theta):  {receipt.early_resolution_preference_theta:>+10.2f}")
    print(f"    Wealth-Growth Sensitivity (A_x):{receipt.wealth_growth_sensitivity_ax:>+10.2f}")
    print(f"    Persistent Growth Premium:      {receipt.persistent_growth_premium_pct:>+10.2f}% / yr")
    print(f"    Uncertainty Volatility Premium: {receipt.uncertainty_volatility_premium_pct:>+10.2f}% / yr")
    print(f"    Bansal-Yaron Impulse (LRRI_t):  {receipt.bansal_yaron_lrr_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Long-Run Risks Regime & Alpha Projections:")
    print(f"    Long-Run Risks Regime:          {receipt.regime}")
    print(f"    High LRR Growth Leader:         {lead_str}")
    print(f"    Macro Uncertainty Safe Haven:   {hedge_str}")
    print(f"    Expected LRR Risk Premium:      {receipt.expected_lrr_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def reversibility_command(args: argparse.Namespace) -> None:
    """CLI handler for Lu Zhang (2005) Costly Reversibility of Capital & Value Beta Engine."""
    import json
    from src.analytics.costly_reversibility import evaluate_costly_reversibility_profile

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    bm = getattr(args, "book_to_market", None)
    if bm is not None:
        bm = float(bm)
    asym = getattr(args, "cost_asymmetry", None)
    if asym is not None:
        asym = float(asym)
    fk = getattr(args, "fixed_capital_intensity", None)
    if fk is not None:
        fk = float(fk)
    macro = getattr(args, "macro_state_zscore", None)
    if macro is not None:
        macro = float(macro)

    try:
        receipt = evaluate_costly_reversibility_profile(
            ticker=ticker,
            book_to_market_ratio=bm,
            downward_upward_cost_asymmetry=asym,
            fixed_capital_intensity=fk,
            macro_state_zscore=macro,
        )
    except Exception as ex:
        print(f"\nError: Costly Reversibility evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (COMPENSATED IRREVERSIBLE VALUE LEADER)" if receipt.is_compensated_irreversible_value_leader else "NO"
    trap_str = "YES (STRANDED CAPITAL RECESSION TRAP)" if receipt.is_stranded_capital_recession_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lu Zhang (2005) Costly Reversibility & Value Beta Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Physical Capital Adjustment & Macroeconomic Inputs:")
    print(f"    Book-to-Market Ratio (B/M):     {receipt.book_to_market_ratio:>10.2f}x")
    print(f"    Adjustment Asymmetry (th-/th+): {receipt.downward_upward_cost_asymmetry:>10.2f}x")
    print(f"    Fixed Capital Intensity (f_K):  {receipt.fixed_capital_intensity:>10.2%}")
    print(f"    Macroeconomic State (S_macro):  {receipt.macro_state_zscore:>+10.2f} SD")
    print("-" * 80)
    print("  Lu Zhang Costly Reversibility Inferences:")
    print(f"    Capital Irreversibility (CII):  {receipt.capital_irreversibility_index:>10.2f}")
    print(f"    Disinvestment Friction (Phi):   {receipt.disinvestment_friction_mult:>10.2f}x")
    print(f"    Countercyclical Value Beta:     {receipt.countercyclical_value_beta:>10.2f}")
    print(f"    Lu Zhang Value Impulse (LZVI):  {receipt.lu_zhang_value_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Costly Reversibility Regime & Alpha Projections:")
    print(f"    Costly Reversibility Regime:    {receipt.regime}")
    print(f"    Compensated Value Leader:       {lead_str}")
    print(f"    Stranded Capital Trap:          {trap_str}")
    print(f"    Expected Reversibility Alpha:   {receipt.expected_reversibility_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def cochrane_command(args: argparse.Namespace) -> None:
    """CLI handler for Campbell-Shiller & Cochrane (2008, 2011) Discount-Rate Decomposition Engine."""
    import json
    from src.analytics.campbell_shiller_cochrane import (
        evaluate_campbell_shiller_cochrane_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    dp = getattr(args, "log_dividend_price", None)
    if dp is not None:
        dp = float(dp)
    dp_bar = getattr(args, "mean_log_dividend_price", None)
    if dp_bar is not None:
        dp_bar = float(dp_bar)
    phi = getattr(args, "valuation_persistence", None)
    if phi is not None:
        phi = float(phi)
    br1 = getattr(args, "return_slope", None)
    if br1 is not None:
        br1 = float(br1)
    bd1 = getattr(args, "cashflow_slope", None)
    if bd1 is not None:
        bd1 = float(bd1)

    try:
        receipt = evaluate_campbell_shiller_cochrane_profile(
            ticker=ticker,
            log_dividend_price_ratio=dp,
            mean_log_dividend_price=dp_bar,
            valuation_persistence_phi=phi,
            one_period_return_slope=br1,
            cashflow_growth_forecast_slope=bd1,
        )
    except Exception as ex:
        print(f"\nError: Campbell-Shiller Cochrane evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    opp_str = "YES (DISCOUNT-RATE MEAN-REVERSION OPPORTUNITY)" if receipt.is_discount_rate_opportunity else "NO"
    trap_str = "YES (COMPRESSED DISCOUNT-RATE TRAP)" if receipt.is_compressed_discount_rate_trap else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell-Shiller & Cochrane (2008, 2011) Decomposition")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Valuation & Present-Value Identity Inputs:")
    print(f"    Log Dividend-Price (dp_t):      {receipt.log_dividend_price_ratio:>10.2f}")
    print(f"    Mean Log Yield (bar{{dp}}):       {receipt.mean_log_dividend_price:>10.2f}")
    print(f"    Yield Deviation (dp - bar{{dp}}): {receipt.yield_deviation:>+10.2f}")
    print(f"    Valuation Persistence (phi):    {receipt.valuation_persistence_phi:>10.3f}")
    print(f"    Campbell-Shiller rho:           {receipt.campbell_shiller_rho:>10.3f}")
    print(f"    Return Forecast Slope (b_r1):   {receipt.one_period_return_slope:>+10.4f}")
    print(f"    Cash-Flow Slope (b_d1):         {receipt.cashflow_growth_forecast_slope:>+10.4f}")
    print("-" * 80)
    print("  Cochrane Present-Value Variance Decomposition:")
    print(f"    Discount-Rate Share (S_DR):     {receipt.discount_rate_variance_share:>10.1%}")
    print(f"    Cash-Flow Share (S_CF):         {receipt.cashflow_variance_share:>10.1%}")
    print(f"    Cochrane Impulse (CSCI_t):      {receipt.cochrane_discount_rate_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Decomposition Regime & Return Projections:")
    print(f"    Present-Value Regime:           {receipt.regime}")
    print(f"    Discount-Rate Opportunity:      {opp_str}")
    print(f"    Compressed Discount-Rate Trap:  {trap_str}")
    print(f"    Expected Mean-Reversion Alpha:  {receipt.expected_discount_rate_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def stub_command(args: argparse.Namespace) -> None:
    """CLI handler for Lamont & Thaler (2003) Stub Value Arbitrage Engine."""
    import json
    from src.analytics.stub_value_arbitrage import evaluate_stub_value_arbitrage_profile

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    sub_ticker = getattr(args, "subsidiary_ticker", None)
    sub_price = getattr(args, "subsidiary_price", None)
    if sub_price is not None:
        sub_price = float(sub_price)
    ratio = getattr(args, "shares_per_parent", None)
    if ratio is not None:
        ratio = float(ratio)
    fee = getattr(args, "borrow_fee", None)
    if fee is not None:
        fee = float(fee)

    try:
        receipt = evaluate_stub_value_arbitrage_profile(
            ticker=ticker,
            subsidiary_ticker=sub_ticker,
            subsidiary_price=sub_price,
            shares_per_parent_ratio=ratio,
            annual_borrow_fee_pct=fee,
        )
    except Exception as ex:
        print(f"\nError: Stub Value Arbitrage evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    neg_str = "YES (FLAGRANT LOOP VIOLATION / SHORT SQUEEZE TRAP)" if receipt.is_negative_stub_violation else "NO"
    comp_str = "YES (COMPRESSED STUB CONVERGENCE OPPORTUNITY)" if receipt.is_compressed_stub_opportunity else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Lamont & Thaler (2003) Stub Value Arbitrage Engine")
    print("=" * 80)
    print(f"  Parent Security:              {receipt.parent_ticker}")
    print(f"  Latest Parent Price (P_p):    ${receipt.latest_parent_price:.2f}")
    print(f"  Carved Subsidiary:            {receipt.subsidiary_ticker}")
    print(f"  Subsidiary Price (P_sub):     ${receipt.subsidiary_price:.2f}")
    print("-" * 80)
    print("  Carve-Out Structure & Borrow Friction Inputs:")
    print(f"    Shares per Parent (alpha):      {receipt.shares_per_parent_ratio:>10.2f}x")
    print(f"    Implied Sub Value / Parent:    ${receipt.implied_subsidiary_value_per_share:>10.2f}")
    print(f"    Subsidiary Short Borrow Fee:    {receipt.annual_borrow_fee_pct:>10.2f}% / yr")
    print("-" * 80)
    print("  Lamont-Thaler & Mitchell-Pulvino-Stafford Inferences:")
    print(f"    Implied Stub Value (S_stub):   ${receipt.implied_stub_value:>10.2f}")
    print(f"    Stub Fraction of Parent:        {receipt.stub_fraction_of_parent:>+10.2%}")
    print(f"    Short-Sale Friction Multiplier: {receipt.short_sale_friction_multiplier:>10.2f}x")
    print(f"    Arbitrage Impulse (LTAI_t):     {receipt.lamont_thaler_arbitrage_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Arbitrage Regime & Convergence Projections:")
    print(f"    Stub Arbitrage Regime:          {receipt.regime}")
    print(f"    Negative Stub Violation:        {neg_str}")
    print(f"    Compressed Stub Opportunity:    {comp_str}")
    print(f"    Expected Convergence Alpha:     {receipt.expected_convergence_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def liquidity_spiral_command(args: argparse.Namespace) -> None:
    """CLI handler for Brunnermeier & Pedersen (2009) Liquidity Spiral Engine."""
    import json
    from src.analytics.liquidity_spiral import evaluate_liquidity_spiral_profile

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    mt = getattr(args, "margin_haircut", None)
    if mt is not None:
        mt = float(mt)
    m0 = getattr(args, "baseline_margin", None)
    if m0 is not None:
        m0 = float(m0)
    buf = getattr(args, "funding_buffer", None)
    if buf is not None:
        buf = float(buf)
    vol = getattr(args, "volatility", None)
    if vol is not None:
        vol = float(vol)
    vol0 = getattr(args, "baseline_volatility", None)
    if vol0 is not None:
        vol0 = float(vol0)

    try:
        receipt = evaluate_liquidity_spiral_profile(
            ticker=ticker,
            current_margin_haircut=mt,
            baseline_margin_haircut=m0,
            funding_margin_buffer=buf,
            annualized_volatility=vol,
            baseline_volatility=vol0,
        )
    except Exception as ex:
        print(f"\nError: Liquidity Spiral evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    spiral_str = "YES (DESTABILIZING LIQUIDITY SPIRAL FIRE-SALE TRAP)" if receipt.is_fire_sale_spiral_trap else "NO"
    haven_str = "YES (ROBUST FUNDING LIQUIDITY SAFE HAVEN)" if receipt.is_funding_liquidity_safe_haven else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Brunnermeier & Pedersen (2009) Liquidity Spiral Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Market Liquidity & Margin Funding Inputs:")
    print(f"    Current Margin Haircut (m_t):   {receipt.current_margin_haircut:>10.2%}")
    print(f"    Baseline Margin (m_0):          {receipt.baseline_margin_haircut:>10.2%}")
    print(f"    Funding Margin Buffer (B_fund): {receipt.funding_margin_buffer:>+10.2%}")
    print(f"    Current Volatility (sigma_t):   {receipt.annualized_volatility:>10.2%}")
    print(f"    Baseline Volatility (sigma_0):  {receipt.baseline_volatility:>10.2%}")
    print("-" * 80)
    print("  Brunnermeier-Pedersen Fragility & Spiral Inferences:")
    print(f"    Margin Tightness (kappa):       {receipt.margin_tightness_ratio:>10.2f}x")
    print(f"    Funding Fragility Score:        {receipt.funding_fragility_score:>10.2f}")
    print(f"    Liquidity Spiral Multiplier:    {receipt.liquidity_spiral_multiplier:>10.2f}x")
    print(f"    Spiral Impulse (BPFI_t):        {receipt.brunnermeier_pedersen_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Liquidity Spiral Regime & Alpha Projections:")
    print(f"    Liquidity Spiral Regime:        {receipt.regime}")
    print(f"    Fire-Sale Spiral Trap:          {spiral_str}")
    print(f"    Funding Safe Haven:             {haven_str}")
    print(f"    Expected Liquidity Alpha:       {receipt.expected_liquidity_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def intermediary_command(args: argparse.Namespace) -> None:
    """CLI handler for He-Kelly-Manela (2017) Intermediary Asset Pricing Engine."""
    import json
    from src.analytics.intermediary_asset_pricing import (
        evaluate_intermediary_pricing_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    eta = getattr(args, "dealer_capital_ratio", None)
    if eta is not None:
        eta = float(eta)
    eta0 = getattr(args, "baseline_capital_ratio", None)
    if eta0 is not None:
        eta0 = float(eta0)
    lev = getattr(args, "dealer_leverage", None)
    if lev is not None:
        lev = float(lev)
    lev0 = getattr(args, "baseline_leverage", None)
    if lev0 is not None:
        lev0 = float(lev0)
    beta = getattr(args, "capital_beta", None)
    if beta is not None:
        beta = float(beta)

    try:
        receipt = evaluate_intermediary_pricing_profile(
            ticker=ticker,
            dealer_capital_ratio=eta,
            baseline_capital_ratio=eta0,
            dealer_leverage_ratio=lev,
            baseline_leverage_ratio=lev0,
            intermediary_capital_beta=beta,
        )
    except Exception as ex:
        print(f"\nError: Intermediary Asset Pricing evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    lead_str = "YES (HIGH INTERMEDIARY RISK CAPITAL LEADER)" if receipt.is_high_intermediary_risk_leader else "NO"
    haven_str = "YES (INTERMEDIARY DISTRESS SAFE HAVEN)" if receipt.is_intermediary_distress_safe_haven else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: He-Kelly-Manela (2017) Intermediary Pricing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Primary Dealer Balance Sheet & Capital Inputs:")
    print(f"    Dealer Capital Ratio (eta_t):   {receipt.dealer_capital_ratio:>10.2%}")
    print(f"    Baseline Capital (eta_0):       {receipt.baseline_capital_ratio:>10.2%}")
    print(f"    Capital Shock ((eta-eta0)/eta0):{receipt.capital_ratio_shock_pct:>+10.2%}")
    print(f"    Dealer Leverage (Assets/Equity):{receipt.dealer_leverage_ratio:>10.2f}x")
    print(f"    Baseline Leverage (Lev_0):      {receipt.baseline_leverage_ratio:>10.2f}x")
    print(f"    Intermediary Capital Beta:      {receipt.intermediary_capital_beta:>+10.2f}")
    print("-" * 80)
    print("  He-Kelly-Manela Intermediary Risk Inferences:")
    print(f"    Balance Sheet Capacity (C_dlr): {receipt.balance_sheet_capacity_index:>10.2f}")
    print(f"    Intermediary Risk Premium:      {receipt.intermediary_risk_premium_pct:>+10.2f}% / yr")
    print(f"    Pricing Impulse (IAPI_t):       {receipt.intermediary_pricing_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Intermediary Regime & Alpha Projections:")
    print(f"    Intermediary Pricing Regime:    {receipt.regime}")
    print(f"    High Intermediary Risk Leader:  {lead_str}")
    print(f"    Intermediary Safe Haven:        {haven_str}")
    print(f"    Expected Intermediary Alpha:    {receipt.expected_intermediary_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def margin_pricing_command(args: argparse.Namespace) -> None:
    """CLI handler for Gârleanu & Pedersen (2011) Margin-Based Asset Pricing Engine."""
    import json
    from src.analytics.margin_asset_pricing import (
        evaluate_margin_pricing_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    m_i = getattr(args, "margin_requirement", None)
    if m_i is not None:
        m_i = float(m_i)
    m_0 = getattr(args, "baseline_margin", None)
    if m_0 is not None:
        m_0 = float(m_0)
    psi = getattr(args, "shadow_cost", None)
    if psi is not None:
        psi = float(psi)
    beta = getattr(args, "market_beta", None)
    if beta is not None:
        beta = float(beta)
    mrp = getattr(args, "market_risk_premium", None)
    if mrp is not None:
        mrp = float(mrp)
    rf = getattr(args, "risk_free_rate", None)
    if rf is not None:
        rf = float(rf)

    try:
        receipt = evaluate_margin_pricing_profile(
            ticker=ticker,
            margin_requirement=m_i,
            baseline_margin_requirement=m_0,
            margin_shadow_cost_psi=psi,
            market_beta=beta,
            market_risk_premium_pct=mrp,
            risk_free_rate_pct=rf,
        )
    except Exception as ex:
        print(f"\nError: Margin Asset Pricing evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    disc_str = "YES (MARGIN CONSTRAINED BASIS DISCOUNT)" if receipt.is_margin_constrained_discount else "NO"
    trap_str = "YES (HIGH-BETA MARGIN OVERPRICING TRAP)" if receipt.is_margin_overpricing_trap else "NO"
    bab_str = "YES (BETTING AGAINST BETA ALPHA)" if receipt.is_bab_alpha_opportunity else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Gârleanu & Pedersen (2011) Margin-Based Asset Pricing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Margin Requirements & Leverage Constraints:")
    print(f"    Margin Haircut (m_i):           {receipt.margin_requirement:>10.2%}")
    print(f"    Baseline Haircut (m_0):         {receipt.baseline_margin_requirement:>10.2%}")
    print(f"    Margin Shadow Cost (psi):       {receipt.margin_shadow_cost_psi:>10.2%}")
    print(f"    Systematic Market Beta (beta):  {receipt.market_beta:>10.2f}")
    print(f"    Market Risk Premium (lambda_m): {receipt.market_risk_premium_pct:>10.2f}%")
    print(f"    Risk-Free Rate (R_f):           {receipt.risk_free_rate_pct:>10.2f}%")
    print("-" * 80)
    print("  Margin-CAPM Expected Return & Basis Spread:")
    print(f"    Standard CAPM Return:           {receipt.standard_capm_expected_return_pct:>10.2f}% / yr")
    print(f"    Margin Premium (m_i * psi):     {receipt.margin_premium_contribution_pct:>+10.2f}% / yr")
    print(f"    Margin-CAPM Expected Return:    {receipt.margin_capm_expected_return_pct:>10.2f}% / yr")
    print(f"    Basis Spread Discount (D_i):    {receipt.basis_spread_discount_pct:>+10.2f}%")
    print(f"    BAB Compression Score:          {receipt.bab_leverage_compression_score:>+10.2f}")
    print(f"    Pricing Impulse:                {receipt.margin_pricing_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Margin Pricing Regime & Alpha Projections:")
    print(f"    Margin Pricing Regime:          {receipt.regime}")
    print(f"    Margin Constrained Discount:    {disc_str}")
    print(f"    High-Beta Overpricing Trap:     {trap_str}")
    print(f"    BAB Alpha Opportunity:          {bab_str}")
    print(f"    Expected Margin Alpha:          {receipt.expected_margin_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def ses_command(args: argparse.Namespace) -> None:
    """CLI handler for Acharya et al. (2017) Systemic Expected Shortfall Engine."""
    import json
    from src.analytics.systemic_expected_shortfall import (
        evaluate_ses_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    c_tail = getattr(args, "market_tail_threshold", None)
    if c_tail is not None:
        c_tail = float(c_tail)
    mes = getattr(args, "marginal_shortfall", None)
    if mes is not None:
        mes = float(mes)
    lev = getattr(args, "leverage", None)
    if lev is not None:
        lev = float(lev)
    lev0 = getattr(args, "baseline_leverage", None)
    if lev0 is not None:
        lev0 = float(lev0)

    try:
        receipt = evaluate_ses_profile(
            ticker=ticker,
            market_tail_threshold_pct=c_tail,
            marginal_expected_shortfall_pct=mes,
            institution_leverage=lev,
            baseline_leverage=lev0,
        )
    except Exception as ex:
        print(f"\nError: Systemic Expected Shortfall evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    crit_str = "YES (CRITICAL SYSTEMIC EXTERNALITY THREAT)" if receipt.is_critical_systemic_threat else "NO"
    def_str = "YES (DEFENSIVE SYSTEMIC RESILIENCE)" if receipt.is_defensive_systemic_resilience else "NO"
    elev_str = "YES (ELEVATED MARGINAL TAIL SENSITIVITY)" if receipt.is_elevated_tail_sensitivity else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Acharya et al. (2017) Systemic Expected Shortfall (SES) Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Systemic Tail Conditioning & Leverage Inputs:")
    print(f"    Crisis Conditioning Drop (C):   {receipt.market_tail_threshold_pct:>10.2f}%")
    print(f"    Marginal Expected Shortfall:    {receipt.marginal_expected_shortfall_pct:>10.2f}% (MES_i)")
    print(f"    Institution Leverage:           {receipt.institution_leverage:>10.2f}x (Assets/Equity)")
    print(f"    Baseline Leverage:              {receipt.baseline_leverage:>10.2f}x")
    print(f"    Tail Dependence Ratio (MES/C):  {receipt.tail_dependence_ratio:>10.2f}x")
    print("-" * 80)
    print("  Acharya et al. Systemic Risk Inferences:")
    print(f"    Systemic Expected Shortfall:    {receipt.systemic_expected_shortfall_pct:>10.2f}% (SES_i)")
    print(f"    Pigouvian Capital Surcharge:    {receipt.systemic_capital_surcharge_pct:>10.2f}% (tau_i)")
    print(f"    Systemic Fragility Impulse:     {receipt.systemic_fragility_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Systemic Regime & Alpha Projections:")
    print(f"    Systemic Risk Regime:           {receipt.regime}")
    print(f"    Critical Systemic Threat:       {crit_str}")
    print(f"    Defensive Systemic Resilience:  {def_str}")
    print(f"    Elevated Tail Sensitivity:      {elev_str}")
    print(f"    Expected Systemic Alpha:        {receipt.expected_systemic_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def covar_command(args: argparse.Namespace) -> None:
    """CLI handler for Adrian & Brunnermeier (2016) CoVaR Tail Spillover Engine."""
    import json
    from src.analytics.covar_systemic_spillover import (
        evaluate_covar_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    q = getattr(args, "quantile", None)
    if q is not None:
        q = float(q)
    var_i = getattr(args, "standalone_var", None)
    if var_i is not None:
        var_i = float(var_i)
    var_m = getattr(args, "system_var", None)
    if var_m is not None:
        var_m = float(var_m)
    beta = getattr(args, "tail_beta", None)
    if beta is not None:
        beta = float(beta)

    try:
        receipt = evaluate_covar_profile(
            ticker=ticker,
            tail_quantile=q,
            standalone_var_pct=var_i,
            system_unconditional_var_pct=var_m,
            tail_beta=beta,
        )
    except Exception as ex:
        print(f"\nError: CoVaR Tail Spillover evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    crit_str = "YES (HIGH SYSTEMIC SPILLOVER CONTAGION)" if receipt.is_critical_spillover_threat else "NO"
    abs_str = "YES (SYSTEMIC DISTRESS ABSORBER)" if receipt.is_systemic_distress_absorber else "NO"
    elev_str = "YES (ELEVATED TAIL SPILLOVER)" if receipt.is_elevated_spillover else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Adrian & Brunnermeier (2016) CoVaR Spillover Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Tail & Quantile Conditioning Inputs:")
    print(f"    Tail Confidence Quantile (q):   {receipt.tail_quantile:>10.2%}")
    print(f"    Standalone Stock VaR (VaR_i):   {receipt.standalone_var_pct:>10.2f}%")
    print(f"    Unconditional Market VaR (VaR_m):{receipt.system_unconditional_var_pct:>10.2f}%")
    print(f"    Tail Quantile Regression Beta:  {receipt.tail_beta:>10.2f}")
    print("-" * 80)
    print("  Adrian-Brunnermeier CoVaR & Spillover Inferences:")
    print(f"    Conditional System CoVaR:       {receipt.conditional_system_covar_pct:>10.2f}% (CoVaR_{{m|i}})")
    print(f"    Marginal Contribution (ΔCoVaR): {receipt.delta_covar_pct:>10.2f}%")
    print(f"    Systemic Exposure (ΔCoVaR_{{i|m}}):{receipt.systemic_exposure_covar_pct:>10.2f}%")
    print(f"    Spillover Multiplier:           {receipt.systemic_spillover_multiplier:>10.2f}x (ΔCoVaR / VaR_m)")
    print(f"    Fragility Impulse:              {receipt.covar_fragility_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  CoVaR Regime & Alpha Projections:")
    print(f"    CoVaR Spillover Regime:         {receipt.regime}")
    print(f"    High Spillover Threat:          {crit_str}")
    print(f"    Systemic Distress Absorber:     {abs_str}")
    print(f"    Elevated Tail Spillover:        {elev_str}")
    print(f"    Expected CoVaR Alpha:           {receipt.expected_covar_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def amh_command(args: argparse.Namespace) -> None:
    """CLI handler for Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis Engine."""
    import json
    from src.analytics.adaptive_markets import (
        evaluate_amh_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    vr = getattr(args, "variance_ratio", None)
    if vr is not None:
        vr = float(vr)
    rho = getattr(args, "autocorrelation", None)
    if rho is not None:
        rho = float(rho)
    dd = getattr(args, "drawdown", None)
    if dd is not None:
        dd = float(dd)
    vol = getattr(args, "volatility", None)
    if vol is not None:
        vol = float(vol)
    vol0 = getattr(args, "baseline_volatility", None)
    if vol0 is not None:
        vol0 = float(vol0)

    try:
        receipt = evaluate_amh_profile(
            ticker=ticker,
            lo_mackinlay_variance_ratio=vr,
            serial_autocorrelation_rho1=rho,
            ecological_stress_drawdown=dd,
            realized_volatility=vol,
            baseline_volatility=vol0,
        )
    except Exception as ex:
        print(f"\nError: Adaptive Markets Hypothesis evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sat_str = "YES (HIGH COMPETITIVE SATURATION)" if receipt.is_competitive_saturation else "NO"
    punc_str = "YES (PUNCTUATED EQUILIBRIUM BEHAVIORAL REGIME)" if receipt.is_punctuated_equilibrium else "NO"
    rec_str = "YES (ADAPTIVE ARBITRAGE RECOVERY)" if receipt.is_adaptive_recovery else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Andrew W. Lo (2004, 2005) Adaptive Markets Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Evolutionary & Microstructure Dynamics:")
    print(f"    Variance Ratio (Lo-MacKinlay):  {receipt.lo_mackinlay_variance_ratio:>10.4f} (VR=1.0 under Random Walk)")
    print(f"    Serial Autocorrelation (rho_1): {receipt.serial_autocorrelation_rho1:>+10.4f}")
    print(f"    Ecological Stress Drawdown:     {receipt.ecological_stress_drawdown:>10.2%}")
    print(f"    Realized Volatility:            {receipt.realized_volatility:>10.2%}")
    print(f"    Baseline Volatility:            {receipt.baseline_volatility:>10.2%}")
    print("-" * 80)
    print("  Andrew W. Lo Adaptive Markets Metrics:")
    print(f"    Adaptive Efficiency Index:      {receipt.adaptive_market_efficiency_index:>10.4f} (E_AMH in [0..1])")
    print(f"    Arbitrage Carrying Capacity:    {receipt.arbitrage_carrying_capacity:>10.2f}x (K_arb)")
    print(f"    Adaptation Selection Pressure:  {receipt.evolutionary_adaptation_pressure:>+10.4f} (Pi_evol)")
    print(f"    Arbitrage Impulse:              {receipt.amh_arbitrage_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  AMH Regime & Alpha Projections:")
    print(f"    Adaptive Markets Regime:        {receipt.regime}")
    print(f"    Competitive Saturation:         {sat_str}")
    print(f"    Punctuated Equilibrium:         {punc_str}")
    print(f"    Adaptive Arbitrage Recovery:    {rec_str}")
    print(f"    Expected Adaptive Alpha:        {receipt.expected_adaptive_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def search_bargaining_command(args: argparse.Namespace) -> None:
    """CLI handler for Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Engine."""
    import json
    from src.analytics.search_bargaining_illiquidity import (
        evaluate_search_bargaining_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    lam = getattr(args, "search_intensity", None)
    if lam is not None:
        lam = float(lam)
    q = getattr(args, "bargaining_power", None)
    if q is not None:
        q = float(q)
    chi = getattr(args, "inventory_cost", None)
    if chi is not None:
        chi = float(chi)
    r = getattr(args, "discount_rate", None)
    if r is not None:
        r = float(r)
    d = getattr(args, "cash_flow_rate", None)
    if d is not None:
        d = float(d)

    try:
        receipt = evaluate_search_bargaining_profile(
            ticker=ticker,
            search_intensity_lambda=lam,
            investor_bargaining_power=q,
            dealer_inventory_cost=chi,
            opportunity_cost_rate=r,
            asset_cash_flow_rate=d,
        )
    except Exception as ex:
        print(f"\nError: Search-and-Bargaining Illiquidity evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    frz_str = "YES (SEVERE OTC SEARCH FREEZE)" if receipt.is_severe_search_freeze else "NO"
    cen_str = "YES (HIGH-LIQUIDITY CENTRALIZED MATCHING)" if receipt.is_centralized_high_liquidity else "NO"
    elev_str = "YES (ELEVATED SEARCH FRICTION)" if receipt.is_elevated_search_friction else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Search & Bargaining Parameters:")
    print(f"    Search Arrival Intensity (λ):   {receipt.search_intensity_lambda:>10.2f} contacts / yr")
    print(f"    Investor Bargaining Power (q):  {receipt.investor_bargaining_power:>10.2%}")
    print(f"    Dealer Inventory Cost (χ):      {receipt.dealer_inventory_cost:>10.2%}")
    print(f"    Opportunity Discount Rate (r):  {receipt.opportunity_cost_rate:>10.2%}")
    print(f"    Cash Flow / Yield Rate (D):     {receipt.asset_cash_flow_rate:>10.2%}")
    print("-" * 80)
    print("  DGP Search Valuation & Spread Inferences:")
    print(f"    Search Illiquidity Discount:    {receipt.search_illiquidity_discount_pct:>10.2f}% (S_DGP)")
    print(f"    Endogenous Bid-Ask Spread:      {receipt.endogenous_bid_ask_spread_pct:>10.2f}% (ΔP)")
    print(f"    Frictionless Walrasian Price:  ${receipt.walrasian_frictionless_price:>10.2f} (P*)")
    print(f"    OTC Effective Market Price:    ${receipt.otc_effective_price:>10.2f} (P_OTC)")
    print(f"    Search Fragility Impulse:       {receipt.dgp_illiquidity_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Search Regime & Alpha Projections:")
    print(f"    Search-and-Bargaining Regime:   {receipt.regime}")
    print(f"    Severe Search Freeze:           {frz_str}")
    print(f"    Centralized Matching:           {cen_str}")
    print(f"    Elevated Search Friction:       {elev_str}")
    print(f"    Expected Search Alpha:          {receipt.expected_search_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def chs_distress_command(args: argparse.Namespace) -> None:
    """CLI handler for Campbell-Hilscher-Szilagyi (2008) Financial Distress Puzzle Engine."""
    import json
    from src.analytics.campbell_distress_puzzle import (
        evaluate_campbell_distress_profile,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    nimta = getattr(args, "nimta", None)
    if nimta is not None:
        nimta = float(nimta)
    tlmta = getattr(args, "tlmta", None)
    if tlmta is not None:
        tlmta = float(tlmta)
    cashmta = getattr(args, "cashmta", None)
    if cashmta is not None:
        cashmta = float(cashmta)
    exret = getattr(args, "excess_return", None)
    if exret is not None:
        exret = float(exret)
    sigma = getattr(args, "sigma", None)
    if sigma is not None:
        sigma = float(sigma)
    rsize = getattr(args, "relative_size", None)
    if rsize is not None:
        rsize = float(rsize)
    mb = getattr(args, "market_to_book", None)
    if mb is not None:
        mb = float(mb)

    try:
        receipt = evaluate_campbell_distress_profile(
            ticker=ticker,
            net_income_to_market_assets=nimta,
            total_liabilities_to_market_assets=tlmta,
            cash_to_market_assets=cashmta,
            trailing_excess_return=exret,
            idiosyncratic_volatility_sigma=sigma,
            relative_firm_size=rsize,
            market_to_book_ratio=mb,
        )
    except Exception as ex:
        print(f"\nError: Campbell Distress evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    trap_str = "YES (DISTRESS ANOMALY OVERPRICED TRAP)" if receipt.is_distress_anomaly_trap else "NO"
    fort_str = "YES (SOLVENT FINANCIAL FORTRESS)" if receipt.is_solvent_financial_fortress else "NO"
    watch_str = "YES (ELEVATED DISTRESS WATCH)" if receipt.is_elevated_distress_watch else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell-Hilscher-Szilagyi (2008) Distress Puzzle Engine")
    print("=" * 80)
    print(f"  Security / Ticker:            {receipt.ticker}")
    print(f"  Latest Price (P_t):           ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  CHS Accounting & Market Balance-Sheet Ratios:")
    print(f"    Net Income to Market Assets:    {receipt.net_income_to_market_assets:>10.2%} (NIMTA)")
    print(f"    Total Liabilities to Mkt Assets:{receipt.total_liabilities_to_market_assets:>10.2%} (TLMTA)")
    print(f"    Cash to Market Assets:          {receipt.cash_to_market_assets:>10.2%} (CASHMTA)")
    print(f"    Trailing Excess Return:         {receipt.trailing_excess_return:>+10.2%} (EXRET vs S&P 500)")
    print(f"    Idiosyncratic Volatility:       {receipt.idiosyncratic_volatility_sigma:>10.2%} (SIGMA)")
    print(f"    Relative Market Size (log):     {receipt.relative_firm_size:>10.2f} (RSIZE)")
    print(f"    Market-to-Book Ratio:           {receipt.market_to_book_ratio:>10.2f}x (MB)")
    print("-" * 80)
    print("  Campbell-Hilscher-Szilagyi Distress Inferences:")
    print(f"    CHS Logit Distress Score:      {receipt.chs_logit_score:>10.4f}")
    print(f"    12-Month Failure Probability:   {receipt.failure_probability_12m_pct:>10.4f}% (P(Failure))")
    print(f"    Distress Anomaly Alpha Drag:    {receipt.distress_anomaly_alpha_drag_pct:>+10.2f}% / year")
    print(f"    Fragility Impulse:              {receipt.chs_fragility_impulse:>+10.4f} [-2.5..+2.5]")
    print("-" * 80)
    print("  Distress Regime & Alpha Projections:")
    print(f"    Solvency Distress Regime:       {receipt.regime}")
    print(f"    Distress Anomaly Trap:          {trap_str}")
    print(f"    Solvent Financial Fortress:     {fort_str}")
    print(f"    Elevated Distress Watch:        {watch_str}")
    print(f"    Expected Distress Alpha:        {receipt.expected_distress_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):          {receipt.signature[:16]}... (Verified: {receipt.verify_integrity()})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def grossman_stiglitz_command(args: argparse.Namespace) -> None:
    """CLI handler for Grossman & Stiglitz (1980) Informational Equilibrium Engine."""
    import json
    from src.analytics.grossman_stiglitz_equilibrium import (
        evaluate_grossman_stiglitz_profile,
        calculate_grossman_stiglitz_equilibrium,
        verify_grossman_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    cost = getattr(args, "cost", None)
    if cost is None:
        cost = 0.0020
    else:
        cost = float(cost)

    risk_aversion = getattr(args, "risk_aversion", None)
    if risk_aversion is None:
        risk_aversion = 2.50
    else:
        risk_aversion = float(risk_aversion)

    fund_var = getattr(args, "fundamental_var", None)
    if fund_var is not None:
        fund_var = float(fund_var)

    noise_var = getattr(args, "noise_var", None)
    if noise_var is not None:
        noise_var = float(noise_var)

    lam_override = getattr(args, "informed_fraction", None)
    if lam_override is not None:
        lam_override = float(lam_override)

    try:
        if fund_var is not None and noise_var is not None:
            receipt = calculate_grossman_stiglitz_equilibrium(
                ticker=ticker,
                latest_price=100.0,
                fundamental_variance=fund_var,
                signal_noise_variance=0.30 * fund_var,
                noise_trader_variance=noise_var,
                risk_aversion=risk_aversion,
                information_cost=cost,
                observed_lambda_override=lam_override,
            )
        else:
            receipt = evaluate_grossman_stiglitz_profile(
                ticker=ticker,
                risk_aversion=risk_aversion,
                information_cost=cost,
            )
    except Exception as ex:
        print(f"\nError: Grossman-Stiglitz evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    eff_str = "YES (COMPETITIVE SATURATION)" if receipt.is_efficient_saturated else "NO"
    opp_str = "YES (INFORMATIONAL RENT OPPORTUNITY)" if receipt.is_noisy_opportunity else "NO"
    herd_str = "YES (UNINFORMED HERD DOMINATED)" if receipt.is_uninformed_herd else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Grossman & Stiglitz (1980) Informational Equilibrium Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Price (P_t):             ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Microstructure & Information Parameters:")
    print(f"    Fundamental Variance (sigma_u^2):   {receipt.fundamental_variance:>10.6f}")
    print(f"    Signal Noise Var (sigma_eps^2):    {receipt.signal_noise_variance:>10.6f}")
    print(f"    Noise Trader Var (sigma_x^2):      {receipt.noise_trader_variance:>10.6f}")
    print(f"    Noise-to-Signal Ratio:             {receipt.noise_to_signal_ratio:>10.4f}")
    print(f"    Investor CARA Risk Aversion:       {receipt.risk_aversion:>10.2f}")
    print(f"    Info Acquisition Cost (c):         {receipt.information_cost * 10000.0:>10.1f} bps ({receipt.information_cost:.4%})")
    print("-" * 80)
    print("  Grossman-Stiglitz Equilibrium Inferences:")
    print(f"    Equilibrium Informed Fraction:     {receipt.equilibrium_informed_fraction:>10.2%} (Lambda*)")
    print(f"    Price Informativeness Index:       {receipt.price_informativeness:>10.2%} (I_GS)")
    print(f"    Uninformed Posterior Var (Var_P):  {receipt.uninformed_posterior_variance:>10.6f}")
    print(f"    Informed Posterior Var (Var_th):   {receipt.informed_posterior_variance:>10.6f}")
    print(f"    Gross Informational Value:         {receipt.gross_informational_value * 10000.0:>10.1f} bps/session")
    print(f"    Net Informational Alpha:           {receipt.net_informational_alpha * 10000.0:>+10.1f} bps/session")
    print("-" * 80)
    print("  Informational Regime & Alpha Projections:")
    print(f"    Informational Regime:              {receipt.informational_regime}")
    print(f"    Informationally Efficient:         {eff_str}")
    print(f"    Noisy Equilibrium Opportunity:     {opp_str}")
    print(f"    Uninformed Herd Dominated:         {herd_str}")
    print(f"    Expected Annual Alpha Boost:       {receipt.expected_alpha_boost_bps:>+10.1f} bps / year")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_grossman_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def information_risk_command(args: argparse.Namespace) -> None:
    """CLI handler for Easley, Hvidkjaer & O'Hara (2002, 2004) Information Risk Engine."""
    import json
    from src.analytics.information_risk_capital import (
        evaluate_information_risk_profile,
        calculate_information_risk_premium,
        verify_information_risk_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    pin = getattr(args, "pin", None)
    if pin is not None:
        pin = float(pin)

    beta = getattr(args, "beta", None)
    if beta is not None:
        beta = float(beta)

    rf = getattr(args, "rf", None)
    if rf is None:
        rf = 0.0425
    else:
        rf = float(rf)

    erp = getattr(args, "erp", None)
    if erp is None:
        erp = 0.0550
    else:
        erp = float(erp)

    lambda_info = getattr(args, "lambda_info", None)
    if lambda_info is None:
        lambda_info = 0.2500
    else:
        lambda_info = float(lambda_info)

    try:
        receipt = evaluate_information_risk_profile(
            ticker=ticker,
            pin_override=pin,
            market_beta_override=beta,
            risk_free_rate=rf,
            equity_risk_premium=erp,
            lambda_info=lambda_info,
        )
    except Exception as ex:
        print(f"\nError: Information Risk evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    severe_str = "YES (SEVERE ADVERSE SELECTION)" if receipt.is_severe_surcharge else "NO"
    elev_str = "YES (ELEVATED INFORMATION RISK)" if receipt.is_elevated_risk else "NO"
    benign_str = "YES (BENIGN TRANSPARENT LIQUIDITY)" if receipt.is_benign_transparent else "NO"
    disc_str = "YES (DISCLOSURE EFFICIENCY DISCOUNT)" if receipt.is_minimal_discount else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Easley-Hvidkjaer-O'Hara (2002, 2004) Information Risk Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Price (P_t):             ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Microstructure & Asset Pricing Parameters:")
    print(f"    Probability of Informed Trading:   {receipt.pin:>10.2%} (PIN)")
    print(f"    Baseline Market PIN Benchmark:     {receipt.baseline_market_pin:>10.2%} (PIN_bar)")
    print(f"    Market Equity Beta (CAPM):         {receipt.market_equity_beta:>10.2f}x")
    print(f"    Standardized Information Beta:     {receipt.info_risk_beta:>+10.2f} (Beta_INFO)")
    print(f"    Info Factor Risk Premium:          {receipt.lambda_info_factor_premium:>10.2%} (lambda_INFO)")
    print("-" * 80)
    print("  Easley-Hvidkjaer-O'Hara Cost of Capital Inferences:")
    print(f"    CAPM Required Return (r_CAPM):     {receipt.capm_required_return_pct:>10.2f}% / year")
    print(f"    Information Risk Surcharge:        {receipt.cost_of_capital_surcharge_pct:>+10.2f}% / year")
    print(f"    Information Risk Surcharge (bps):  {receipt.cost_of_capital_surcharge_bps:>+10.1f} bps / year")
    print(f"    Total Required Hurdle Rate:        {receipt.total_required_hurdle_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Information Risk Regime & Surcharge Flags:")
    print(f"    Information Risk Regime:           {receipt.regime}")
    print(f"    Severe Adverse Selection:          {severe_str}")
    print(f"    Elevated Information Risk:         {elev_str}")
    print(f"    Benign Transparent Liquidity:      {benign_str}")
    print(f"    Minimal Disclosure Discount:       {disc_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_information_risk_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def admati_pfleiderer_command(args: argparse.Namespace) -> None:
    """CLI handler for Admati & Pfleiderer (1988) Volume Clustering & Execution Timing Engine."""
    import json
    from src.analytics.admati_pfleiderer_clustering import (
        evaluate_admati_pfleiderer_profile,
        calculate_admati_pfleiderer_clustering,
        verify_admati_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    disc_ratio = getattr(args, "discretionary_ratio", None)
    if disc_ratio is not None:
        disc_ratio = float(disc_ratio)

    informed_count = getattr(args, "informed_count", None)
    if informed_count is None:
        informed_count = 3
    else:
        informed_count = int(informed_count)

    spread_bps = getattr(args, "spread_bps", None)
    if spread_bps is not None:
        spread_bps = float(spread_bps)

    try:
        if disc_ratio is not None and spread_bps is not None:
            receipt = calculate_admati_pfleiderer_clustering(
                ticker=ticker,
                discretionary_liquidity_ratio=disc_ratio,
                informed_trader_count=informed_count,
                baseline_spread_bps=spread_bps,
            )
        else:
            receipt = evaluate_admati_pfleiderer_profile(
                ticker=ticker,
                discretionary_ratio_override=disc_ratio,
            )
    except Exception as ex:
        print(f"\nError: Admati-Pfleiderer evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    conc_str = "YES (OPTIMAL HIGH-DEPTH EXECUTION)" if receipt.is_concentrated_cluster else "NO"
    bal_str = "YES (BALANCED CONTINUOUS LIQUIDITY)" if receipt.is_balanced_equilibrium else "NO"
    drought_str = "YES (AVOID MIDDAY EXECUTION DROUGHT)" if receipt.is_midday_drought else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Admati-Pfleiderer (1988) Volume Clustering Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Price (P_t):             ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Microstructure & Coordination Parameters:")
    print(f"    Discretionary Liquidity Ratio:     {receipt.discretionary_liquidity_ratio:>10.2%}")
    print(f"    Non-Discretionary Variance:        {receipt.non_discretionary_variance:>10.4f}")
    print(f"    Informed Trader Count (k):         {receipt.informed_trader_count:>10d}")
    print(f"    Admati-Pfleiderer Clumping Index:  {receipt.clumping_index:>10.2%} (C_AP)")
    print("-" * 80)
    print("  Concentrated Trading Equilibrium Inferences:")
    print(f"    Market Depth Multiplier:           {receipt.market_depth_multiplier:>10.2f}x (D_AP)")
    print(f"    Kyle Lambda Impact Reduction:      {receipt.kyle_lambda_reduction_pct:>10.2f}%")
    print(f"    Execution Slippage Savings:        {receipt.execution_slippage_discount_bps:>10.1f} bps")
    print(f"    Recommended Execution Window:      {receipt.recommended_execution_window}")
    print("-" * 80)
    print("  Intraday Liquidity Regime & Execution Flags:")
    print(f"    Intraday Liquidity Regime:         {receipt.regime}")
    print(f"    Concentrated Liquidity Cluster:    {conc_str}")
    print(f"    Balanced Continuous Liquidity:     {bal_str}")
    print(f"    Midday High-Slippage Drought:      {drought_str}")
    print(f"    Optimal Execution Strategy:        {receipt.optimal_execution_timing}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_admati_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def noise_bands_command(args: argparse.Namespace) -> None:
    """CLI handler for Black (1986) Noise Valuation Bounds & Mean-Reversion Engine."""
    import json
    from src.analytics.noise_valuation_bands import (
        evaluate_noise_valuation_profile,
        calculate_noise_valuation_bands,
        verify_noise_valuation_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    fund_val = getattr(args, "fundamental_value", None)
    if fund_val is not None:
        fund_val = float(fund_val)

    noise_ratio = getattr(args, "noise_ratio", None)
    if noise_ratio is not None:
        noise_ratio = float(noise_ratio)

    kappa = getattr(args, "kappa", None)
    if kappa is None:
        kappa = 0.35
    else:
        kappa = float(kappa)

    try:
        if fund_val is not None and noise_ratio is not None:
            receipt = calculate_noise_valuation_bands(
                ticker=ticker,
                latest_price=fund_val * (1.0 + 0.15),  # representative test point
                fundamental_value=fund_val,
                noise_to_information_ratio=noise_ratio,
                mean_reversion_speed_kappa=kappa,
            )
        else:
            receipt = evaluate_noise_valuation_profile(
                ticker=ticker,
                fundamental_value_override=fund_val,
                noise_ratio_override=noise_ratio,
            )
    except Exception as ex:
        print(f"\nError: Noise Valuation evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    disloc_str = "YES (OUTSIDE FACTOR-OF-TWO ENVELOPE)" if receipt.is_extreme_dislocation else "NO"
    over_str = "YES (NOISE OVERVALUATION DRAG)" if receipt.is_overvalued else "NO"
    under_str = "YES (DEEP VALUE MEAN-REVERSION TAILWIND)" if receipt.is_undervalued else "NO"
    eq_str = "YES (WELL-ANCHORED WITHIN EQUILIBRIUM CORRIDOR)" if receipt.is_equilibrium_band else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black (1986) Noise Valuation Bounds Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print(f"  Fundamental Value (P*):         ${receipt.fundamental_value:.2f}")
    print("-" * 80)
    print("  Black (1986) Factor-of-Two Efficiency Envelopes:")
    print(f"    Lower Factor-of-Two Bound:         ${receipt.lower_factor2_bound:>10.2f} (0.50x P*)")
    print(f"    Inner Lower Corridor:              ${receipt.inner_lower_bound:>10.2f} (0.80x P*)")
    print(f"    Inner Upper Corridor:              ${receipt.inner_upper_bound:>10.2f} (1.25x P*)")
    print(f"    Upper Factor-of-Two Bound:         ${receipt.upper_factor2_bound:>10.2f} (2.00x P*)")
    print(f"    Inside Factor-of-Two Envelope:     {'YES' if receipt.is_within_factor2_bounds else 'NO'}")
    print("-" * 80)
    print("  Noise Dynamics & Variance Decomposition:")
    print(f"    Pricing Deviation from P*:         {receipt.noise_deviation_pct:>+10.2f}%")
    print(f"    Noise-to-Information Ratio:        {receipt.noise_to_information_ratio:>10.4f} (Var_noise / Var_info)")
    print(f"    Information Share:                 {receipt.information_share_pct:>10.2f}%")
    print(f"    Annual Convergence Rate (kappa):   {receipt.mean_reversion_speed_kappa:>10.2%}")
    print(f"    Expected Mean-Reversion Alpha:     {receipt.expected_reversion_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print("  Valuation Regime & Mean-Reversion Inferences:")
    print(f"    Noise Valuation Regime:            {receipt.regime}")
    print(f"    Extreme Noise Dislocation:         {disloc_str}")
    print(f"    Noise-Driven Overvaluation:        {over_str}")
    print(f"    Noise-Driven Undervaluation:       {under_str}")
    print(f"    Equilibrium Corridor:              {eq_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_noise_valuation_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bad_beta_command(args: argparse.Namespace) -> None:
    """CLI handler for Campbell & Vuolteenaho (2004) Two-Beta ("Bad Beta, Good Beta") Engine."""
    import json
    from src.analytics.bad_beta_good_beta import (
        evaluate_two_beta_profile,
        calculate_two_beta_decomposition,
        verify_two_beta_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta_cf = getattr(args, "beta_cf", None)
    if beta_cf is not None:
        beta_cf = float(beta_cf)

    beta_dr = getattr(args, "beta_dr", None)
    if beta_dr is not None:
        beta_dr = float(beta_dr)

    gamma = getattr(args, "gamma", None)
    if gamma is None:
        gamma = 6.0
    else:
        gamma = float(gamma)

    try:
        if beta_cf is not None and beta_dr is not None:
            receipt = calculate_two_beta_decomposition(
                ticker=ticker,
                cash_flow_bad_beta=beta_cf,
                discount_rate_good_beta=beta_dr,
                risk_aversion_gamma=gamma,
            )
        else:
            receipt = evaluate_two_beta_profile(
                ticker=ticker,
                cash_flow_beta_override=beta_cf,
                discount_rate_beta_override=beta_dr,
                risk_aversion_gamma=gamma,
            )
    except Exception as ex:
        print(f"\nError: Two-Beta evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    cf_str = "YES (HIGH PERMANENT CASH-FLOW RISK)" if receipt.is_cash_flow_vulnerable else "NO"
    dr_str = "YES (GROWTH PROFILE / DISCOUNT-RATE DOMINATED)" if receipt.is_discount_rate_growth else "NO"
    def_str = "YES (DEFENSIVE LOW-BETA RESILIENCE)" if receipt.is_defensive_insulated else "NO"
    bal_str = "YES (BALANCED INTERTEMPORAL RISK)" if receipt.is_balanced_exposure else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell-Vuolteenaho (2004) Two-Beta Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Two-Beta Systematic Market Decomposition:")
    print(f"    Total Market Beta (beta_M):        {receipt.total_market_beta:>10.2f}x (beta_CF + beta_DR)")
    print(f"    Cash-Flow Beta ('Bad Beta'):       {receipt.cash_flow_bad_beta:>10.2f}x (beta_CF)")
    print(f"    Discount-Rate Beta ('Good Beta'):  {receipt.discount_rate_good_beta:>10.2f}x (beta_DR)")
    print(f"    Bad Beta Ratio (beta_CF / beta_M): {receipt.bad_beta_ratio:>10.2%}")
    print("-" * 80)
    print("  ICAPM Risk Prices & Cost of Capital Inferences:")
    print(f"    Relative Risk Aversion (gamma):    {receipt.relative_risk_aversion_gamma:>10.2f}")
    print(f"    Cash-Flow Risk Price (lambda_CF):  {receipt.cash_flow_risk_price_pct:>10.2f}% / unit beta")
    print(f"    Discount-Rate Price (lambda_DR):   {receipt.discount_rate_risk_price_pct:>10.2f}% / unit beta")
    print(f"    Standard CAPM Required Return:     {receipt.standard_capm_return_pct:>10.2f}% / year")
    print(f"    Two-Beta ICAPM Required Return:    {receipt.two_beta_icapm_return_pct:>10.2f}% / year")
    print(f"    Bad Beta Hurdle Surcharge:         {receipt.bad_beta_hurdle_surcharge_bps:>+10.1f} bps / year")
    print("-" * 80)
    print("  Two-Beta Risk Regime & Classification:")
    print(f"    Intertemporal Risk Regime:         {receipt.regime}")
    print(f"    Cash-Flow Vulnerable (Bad Beta):   {cf_str}")
    print(f"    Discount-Rate Growth (Good Beta):  {dr_str}")
    print(f"    Defensive Insulated:               {def_str}")
    print(f"    Balanced Intertemporal Exposure:   {bal_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_two_beta_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def merton_recognition_command(args: argparse.Namespace) -> None:
    """CLI handler for Merton (1987) Incomplete Information & Investor Recognition Engine."""
    import json
    from src.analytics.merton_incomplete_information import (
        evaluate_merton_incomplete_info_profile,
        calculate_merton_incomplete_information,
        verify_merton_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    q_val = getattr(args, "recognition_q", None)
    if q_val is not None:
        q_val = float(q_val)

    lam_val = getattr(args, "shadow_lambda", None)
    if lam_val is not None:
        lam_val = float(lam_val)

    try:
        if q_val is not None and lam_val is not None:
            receipt = calculate_merton_incomplete_information(
                ticker=ticker,
                investor_recognition_ratio_q=q_val,
                shadow_cost_lambda=lam_val,
            )
        else:
            receipt = evaluate_merton_incomplete_info_profile(
                ticker=ticker,
                recognition_q_override=q_val,
                shadow_lambda_override=lam_val,
            )
    except Exception as ex:
        print(f"\nError: Merton Incomplete Information evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    negl_str = "YES (NEGLECTED FIRM HIGH SURCHARGE)" if receipt.is_neglected_firm else "NO"
    elev_str = "YES (ELEVATED INCOMPLETE INFORMATION)" if receipt.is_elevated_incomplete else "NO"
    broad_str = "YES (BROAD RECOGNITION EQUILIBRIUM)" if receipt.is_broad_recognition else "NO"
    univ_str = "YES (UNIVERSAL SATURATED COVERAGE)" if receipt.is_universal_saturated else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Merton (1987) Incomplete Information Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Investor Recognition & Idiosyncratic Risk:")
    print(f"    Investor Recognition Ratio (q):    {receipt.investor_recognition_ratio_q:>10.2%}")
    print(f"    Idiosyncratic Variance:            {receipt.idiosyncratic_variance:>10.6f} (sigma_eps^2)")
    print(f"    Market Equity Beta (beta_M):       {receipt.market_equity_beta:>10.2f}x")
    print(f"    Shadow Cost Multiplier (lambda):   {receipt.shadow_cost_lambda:>10.2f}")
    print("-" * 80)
    print("  Merton (1987) Cost of Capital & Surcharge Inferences:")
    print(f"    CAPM Required Return:              {receipt.capm_required_return_pct:>10.2f}% / year")
    print(f"    Merton Shadow Surcharge:           {receipt.merton_shadow_surcharge_pct:>+10.2f}% / year")
    print(f"    Merton Shadow Surcharge (bps):     {receipt.merton_shadow_surcharge_bps:>+10.1f} bps / year")
    print(f"    Total Required Hurdle Return:      {receipt.total_required_hurdle_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Recognition Regime & Neglected Firm Classification:")
    print(f"    Incomplete Information Regime:     {receipt.regime}")
    print(f"    Neglected Firm High Surcharge:     {negl_str}")
    print(f"    Elevated Incomplete Information:   {elev_str}")
    print(f"    Broad Recognition Equilibrium:     {broad_str}")
    print(f"    Universal Saturated Coverage:      {univ_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_merton_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def rare_disasters_command(args: argparse.Namespace) -> None:
    """CLI handler for Xavier Gabaix (2012) Variable Rare Disasters & Macro Tail Risk Engine."""
    import json
    from src.analytics.variable_rare_disasters import (
        evaluate_rare_disasters_profile,
        calculate_variable_rare_disasters,
        verify_disaster_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    lam_val = getattr(args, "disaster_lambda", None)
    if lam_val is not None:
        lam_val = float(lam_val)

    rec_val = getattr(args, "recovery_f", None)
    if rec_val is not None:
        rec_val = float(rec_val)

    gamma_val = getattr(args, "gamma", None)
    if gamma_val is not None:
        gamma_val = float(gamma_val)

    try:
        if lam_val is not None and rec_val is not None:
            kwargs = {
                "ticker": ticker,
                "disaster_arrival_rate_lambda": lam_val,
                "firm_dividend_recovery_factor_f": rec_val,
            }
            if gamma_val is not None:
                kwargs["risk_aversion_gamma"] = gamma_val
            receipt = calculate_variable_rare_disasters(**kwargs)
        else:
            receipt = evaluate_rare_disasters_profile(
                ticker=ticker,
                disaster_lambda_override=lam_val,
                recovery_f_override=rec_val,
            )
    except Exception as ex:
        print(f"\nError: Variable Rare Disasters evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    vuln_str = "YES (DISASTER VULNERABLE HIGH PREMIUM)" if receipt.is_disaster_vulnerable else "NO"
    cycl_str = "YES (CYCLICAL TAIL RISK EXPOSURE)" if receipt.is_cyclical_exposure else "NO"
    stab_str = "YES (BENIGN TAIL STABILITY)" if receipt.is_benign_stability else "NO"
    fort_str = "YES (DISASTER RESILIENT FORTRESS)" if receipt.is_resilient_fortress else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Gabaix (2012) Variable Rare Disasters Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Macro Disasters & Tail Resilience Parameters:")
    print(f"    Disaster Arrival Rate (lambda):    {receipt.disaster_arrival_rate_lambda:>10.2%} / year")
    print(f"    Consumption Contraction (B):       {receipt.consumption_factor_b:>10.2%}")
    print(f"    Pricing Kernel Spike (B^-gamma):   {receipt.pricing_kernel_disaster_spike:>10.2f}x")
    print(f"    Firm Recovery Factor (F):          {receipt.firm_dividend_recovery_factor_f:>10.2%}")
    print("-" * 80)
    print("  Gabaix (2012) Disaster Resilience & Hurdle Inferences:")
    print(f"    CAPM Required Return:              {receipt.capm_required_return_pct:>10.2f}% / year")
    print(f"    Disaster Resilience Premium (H):   {receipt.disaster_resilience_premium_pct:>+10.2f}% / year")
    print(f"    Disaster Resilience Premium (bps): {receipt.disaster_resilience_premium_bps:>+10.1f} bps / year")
    print(f"    Total Disaster Hurdle Return:      {receipt.total_disaster_hurdle_pct:>10.2f}% / year")
    print(f"    Implied Price/Dividend Multiple:   {receipt.implied_price_dividend_multiple:>10.2f}x")
    print("-" * 80)
    print("  Disaster Resilience Regime Classification:")
    print(f"    Macro Tail Regime:                 {receipt.regime}")
    print(f"    Disaster Vulnerable High Premium:  {vuln_str}")
    print(f"    Cyclical Tail Risk Exposure:       {cycl_str}")
    print(f"    Benign Tail Stability:             {stab_str}")
    print(f"    Disaster Resilient Fortress:       {fort_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_disaster_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def kyle_xiong_command(args: argparse.Namespace) -> None:
    """CLI handler for Albert S. Kyle & Wei Xiong (2001) Contagion as a Wealth Effect Engine."""
    import json
    from src.analytics.kyle_xiong_contagion import (
        evaluate_kyle_xiong_profile,
        calculate_kyle_xiong_contagion,
        verify_kyle_xiong_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    wealth_dd = getattr(args, "wealth_drawdown", None)
    if wealth_dd is not None:
        wealth_dd = float(wealth_dd)

    gamma_val = getattr(args, "gamma", None)
    if gamma_val is not None:
        gamma_val = float(gamma_val)

    vol_val = getattr(args, "volatility", None)
    if vol_val is not None:
        vol_val = float(vol_val)

    try:
        if wealth_dd is not None and vol_val is not None:
            kwargs = {
                "ticker": ticker,
                "arbitrageur_wealth_ratio": max(0.05, 1.0 - (wealth_dd / 100.0)),
                "fundamental_volatility": vol_val,
            }
            if gamma_val is not None:
                kwargs["risk_aversion_gamma"] = gamma_val
            receipt = calculate_kyle_xiong_contagion(**kwargs)
        else:
            receipt = evaluate_kyle_xiong_profile(
                ticker=ticker,
                wealth_drawdown_override=wealth_dd,
                gamma_override=gamma_val,
            )
    except Exception as ex:
        print(f"\nError: Kyle-Xiong Contagion evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    sys_str = "YES (SYSTEMIC LIQUIDATION CONTAGION)" if receipt.is_systemic_contagion else "NO"
    mod_str = "YES (MODERATE WEALTH SPILLOVER)" if receipt.is_moderate_spillover else "NO"
    ben_str = "YES (BENIGN DECOUPLED)" if receipt.is_benign_decoupled else "NO"
    sur_str = "YES (ARBITRAGE CAPITAL SURPLUS)" if receipt.is_capital_surplus else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Kyle & Xiong (2001) Wealth Contagion Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Arbitrageur Wealth & Capital Capacity Parameters:")
    print(f"    Arbitrageur Wealth Ratio:          {receipt.arbitrageur_wealth_ratio:>10.2f}x (W_t / W_0)")
    print(f"    Arbitrageur Wealth Drawdown:       {receipt.wealth_drawdown_pct:>10.2f}%")
    print(f"    Risk-Bearing Capacity (kappa):     {receipt.risk_bearing_capacity:>10.2f}")
    print("-" * 80)
    print("  Kyle & Xiong (2001) Spillover & Forced Liquidation Inferences:")
    print(f"    Contagion Spillover Beta:          {receipt.contagion_spillover_beta:>10.2f}x")
    print(f"    Non-Fundamental Excess Corr:       {receipt.non_fundamental_excess_corr:>10.2f} (Delta rho)")
    print(f"    Forced Liquidation Pressure:       {receipt.forced_liquidation_pressure:>10.1f} / 100")
    print(f"    Fire-Sale Liquidation Discount:    {receipt.liquidation_discount_pct:>10.2f}%")
    print(f"    Volatility Amplification Factor:   {receipt.volatility_amplification_factor:>10.2f}x")
    print("-" * 80)
    print("  Contagion Regime Classification:")
    print(f"    Market Contagion Regime:           {receipt.regime}")
    print(f"    Systemic Liquidation Contagion:    {sys_str}")
    print(f"    Moderate Wealth Spillover:         {mod_str}")
    print(f"    Benign Decoupled:                  {ben_str}")
    print(f"    Arbitrage Capital Surplus:         {sur_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_kyle_xiong_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def predatory_trading_command(args: argparse.Namespace) -> None:
    """CLI handler for Markus K. Brunnermeier & Lasse Heje Pedersen (2005) Predatory Trading Engine."""
    import json
    from src.analytics.predatory_trading import (
        evaluate_predatory_trading_profile,
        calculate_predatory_trading,
        verify_predatory_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    dist_size = getattr(args, "distressed_size", None)
    if dist_size is not None:
        dist_size = float(dist_size)

    pred_ratio = getattr(args, "predator_concentration", None)
    if pred_ratio is not None:
        pred_ratio = float(pred_ratio)

    depth_val = getattr(args, "market_depth", None)
    if depth_val is not None:
        depth_val = float(depth_val)

    try:
        if dist_size is not None and pred_ratio is not None:
            kwargs = {
                "ticker": ticker,
                "distressed_liquidation_size_pct_adv": dist_size,
                "predator_concentration_ratio": pred_ratio,
            }
            if depth_val is not None:
                kwargs["market_depth_multiplier"] = depth_val
            receipt = calculate_predatory_trading(**kwargs)
        else:
            receipt = evaluate_predatory_trading_profile(
                ticker=ticker,
                distressed_size_override=dist_size,
                predator_ratio_override=pred_ratio,
            )
    except Exception as ex:
        print(f"\nError: Predatory Trading evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    agg_str = "YES (AGGRESSIVE PREDATORY RUN)" if receipt.is_aggressive_predatory_run else "NO"
    coord_str = "YES (COORDINATED FRONT RUNNING)" if receipt.is_coordinated_front_running else "NO"
    comp_str = "YES (COMPETITIVE ABSORPTION)" if receipt.is_competitive_absorption else "NO"
    ben_str = "YES (BENIGN LIQUIDITY PROVISION)" if receipt.is_benign_liquidity else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Brunnermeier & Pedersen (2005) Predatory Trading Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Distressed Liquidation & Predator Order Flow Parameters:")
    print(f"    Distressed Liquidation Size:       {receipt.distressed_liquidation_size_pct_adv:>10.2f}% of ADV")
    print(f"    Predator Concentration Share:      {receipt.predator_concentration_ratio:>10.2%}")
    print(f"    Market Depth Multiplier:           {receipt.market_depth_multiplier:>10.2f}x")
    print(f"    Predatory Shorting Volume:         {receipt.predatory_shorting_volume_pct_adv:>10.2f}% of ADV")
    print("-" * 80)
    print("  Brunnermeier & Pedersen (2005) Price Overshoot & Slippage Inferences:")
    print(f"    Predatory Price Overshoot Ratio:   {receipt.price_overshoot_ratio:>10.2f}x")
    print(f"    Predatory Liquidity Drain Rate:    {receipt.predatory_liquidity_drain_pct:>10.2f}%")
    print(f"    Distressed Slippage Inflation:     {receipt.distressed_slippage_inflation_pct:>+10.2f}%")
    print(f"    Expected Post-Trough Rebound:      {receipt.expected_rebound_potential_pct:>10.2f}%")
    print("-" * 80)
    print("  Predatory Trading Regime Classification:")
    print(f"    Microstructure Regime:             {receipt.regime}")
    print(f"    Aggressive Predatory Run:          {agg_str}")
    print(f"    Coordinated Front-Running:         {coord_str}")
    print(f"    Competitive Absorption:            {comp_str}")
    print(f"    Benign Liquidity Provision:        {ben_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_predatory_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def hong_stein_crash_command(args: argparse.Namespace) -> None:
    """CLI handler for Harrison Hong & Jeremy C. Stein (2003) Crash Asymmetry & Overhang Engine."""
    import json
    from src.analytics.hong_stein_crash import (
        evaluate_hong_stein_crash_profile,
        calculate_hong_stein_crash,
        verify_hong_stein_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    disp_val = getattr(args, "opinion_dispersion", None)
    if disp_val is not None:
        disp_val = float(disp_val)

    const_val = getattr(args, "short_constraint", None)
    if const_val is not None:
        const_val = float(const_val)

    turn_val = getattr(args, "turnover", None)
    if turn_val is not None:
        turn_val = float(turn_val)

    try:
        if disp_val is not None and const_val is not None:
            kwargs = {
                "ticker": ticker,
                "opinion_dispersion_ratio": disp_val,
                "short_sale_constraint_tightness": const_val,
            }
            if turn_val is not None:
                kwargs["turnover_ratio"] = turn_val
            receipt = calculate_hong_stein_crash(**kwargs)
        else:
            receipt = evaluate_hong_stein_crash_profile(
                ticker=ticker,
                opinion_dispersion_override=disp_val,
                short_constraint_override=const_val,
            )
    except Exception as ex:
        print(f"\nError: Hong-Stein Crash evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    acute_str = "YES (ACUTE CRASH VULNERABILITY)" if receipt.is_acute_crash_vulnerability else "NO"
    elev_str = "YES (ELEVATED LATENT OVERHANG)" if receipt.is_elevated_latent_overhang else "NO"
    symm_str = "YES (SYMMETRIC UNCONSTRAINED)" if receipt.is_symmetric_unconstrained else "NO"
    cons_str = "YES (CONSENSUS STABILITY)" if receipt.is_consensus_stability else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hong & Stein (2003) Crash Asymmetry Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Differences of Opinion & Short-Sale Constraint Parameters:")
    print(f"    Opinion Dispersion Ratio:          {receipt.opinion_dispersion_ratio:>10.2%}")
    print(f"    Short-Sale Constraint Friction:    {receipt.short_sale_constraint_tightness:>10.2%}")
    print(f"    Trading Turnover Multiplier:       {receipt.turnover_ratio:>10.2f}x")
    print("-" * 80)
    print("  Hong & Stein (2003) Latent Overhang & Crash Asymmetry Inferences:")
    print(f"    Latent Information Overhang:       {receipt.latent_information_overhang:>10.2f} (I_latent)")
    print(f"    Conditional Crash Skewness:        {receipt.conditional_crash_skewness:>10.2f} (S_crash)")
    print(f"    Crash Vulnerability Index:         {receipt.crash_vulnerability_index:>10.1f} / 100")
    print(f"    Downside Volatility Surcharge:     {receipt.downside_volatility_surcharge_pct:>+10.2f}%")
    print(f"    Expected Cascade Drawdown:         {receipt.expected_crash_drawdown_pct:>10.2f}%")
    print("-" * 80)
    print("  Crash Vulnerability Regime Classification:")
    print(f"    Market Asymmetry Regime:           {receipt.regime}")
    print(f"    Acute Crash Vulnerability:         {acute_str}")
    print(f"    Elevated Latent Overhang:          {elev_str}")
    print(f"    Symmetric Unconstrained:           {symm_str}")
    print(f"    Consensus Stability:               {cons_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_hong_stein_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def breeden_ccapm_command(args: argparse.Namespace) -> None:
    """CLI handler for Douglas T. Breeden (1979) Consumption CAPM Engine."""
    import json
    from src.analytics.breeden_ccapm import (
        evaluate_breeden_ccapm_profile,
        calculate_breeden_ccapm,
        verify_breeden_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta_c = getattr(args, "consumption_beta", None)
    if beta_c is not None:
        beta_c = float(beta_c)

    lambda_c = getattr(args, "consumption_premium", None)
    if lambda_c is not None:
        lambda_c = float(lambda_c)

    c_vol = getattr(args, "consumption_vol", None)
    if c_vol is not None:
        c_vol = float(c_vol)

    try:
        if beta_c is not None and lambda_c is not None:
            kwargs = {
                "ticker": ticker,
                "consumption_beta": beta_c,
                "consumption_risk_premium": lambda_c,
            }
            if c_vol is not None:
                kwargs["consumption_growth_std"] = c_vol
            receipt = calculate_breeden_ccapm(**kwargs)
        else:
            receipt = evaluate_breeden_ccapm_profile(
                ticker=ticker,
                consumption_beta_override=beta_c,
                lambda_c_override=lambda_c,
            )
    except Exception as ex:
        print(f"\nError: Breeden CCAPM evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    high_str = "YES (HIGH CONSUMPTION EXPOSURE)" if receipt.is_high_consumption_exposure else "NO"
    mod_str = "YES (MODERATE PROCYCLICAL)" if receipt.is_moderate_procyclical else "NO"
    def_str = "YES (DEFENSIVE CONSUMPTION HEDGE)" if receipt.is_defensive_hedge else "NO"
    count_str = "YES (COUNTERCYCLICAL ASSET)" if receipt.is_countercyclical_asset else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Breeden (1979) Consumption CAPM Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Intertemporal Macro Consumption Parameters:")
    print(f"    Consumption Beta (beta_C):         {receipt.consumption_beta:>10.2f}x")
    print(f"    Consumption Correlation:           {receipt.consumption_correlation:>10.2f}")
    print(f"    Consumption Growth Volatility:     {receipt.consumption_growth_volatility_pct:>10.2f}% / year")
    print(f"    Unit Price of Consumption Risk:    {receipt.consumption_risk_premium_pct:>10.2f}% (lambda_C)")
    print("-" * 80)
    print("  Breeden (1979) CCAPM Required Return & Hedging Inferences:")
    print(f"    CCAPM Required Hurdle Return:      {receipt.ccapm_required_return_pct:>10.2f}% / year")
    print(f"    CAPM Market Benchmark Return:      {receipt.capm_benchmark_return_pct:>10.2f}% / year")
    print(f"    Consumption Pricing Wedge:         {receipt.equity_premium_puzzle_wedge_pct:>+10.2f}% (CAPM - CCAPM)")
    print(f"    Intertemporal Hedge Elasticity:    {receipt.intertemporal_hedge_elasticity:>10.2f}%")
    print(f"    Macro Hurdle Discount Rate:        {receipt.macro_hurdle_rate_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Consumption Risk Regime Classification:")
    print(f"    Macroeconomic Risk Regime:         {receipt.regime}")
    print(f"    High Consumption Exposure:         {high_str}")
    print(f"    Moderate Procyclical Consumption:  {mod_str}")
    print(f"    Defensive Consumption Hedge:       {def_str}")
    print(f"    Countercyclical Asset:             {count_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_breeden_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def bhs_prospect_command(args: argparse.Namespace) -> None:
    """CLI handler for Barberis, Huang & Santos (2001) Prospect Theory Engine."""
    import json
    from src.analytics.barberis_huang_santos import (
        evaluate_bhs_prospect_profile,
        calculate_bhs_prospect,
        verify_bhs_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    ref_price = getattr(args, "reference_price", None)
    if ref_price is not None:
        ref_price = float(ref_price)

    lam_0 = getattr(args, "lambda_zero", None)
    if lam_0 is not None:
        lam_0 = float(lam_0)

    try:
        if ref_price is not None:
            kwargs = {
                "ticker": ticker,
                "reference_price": ref_price,
            }
            if lam_0 is not None:
                kwargs["baseline_loss_aversion"] = lam_0
            receipt = calculate_bhs_prospect(**kwargs)
        else:
            receipt = evaluate_bhs_prospect_profile(
                ticker=ticker,
                reference_price_override=ref_price,
                lambda_zero_override=lam_0,
            )
    except Exception as ex:
        print(f"\nError: BHS Prospect Theory evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    acute_str = "YES (ACUTE LOSS AVERSION PANIC)" if receipt.is_acute_loss_aversion else "NO"
    mod_str = "YES (MODERATE RECESSION SENSITIVITY)" if receipt.is_moderate_sensitivity else "NO"
    ben_str = "YES (BENIGN NEUTRAL PROSPECT)" if receipt.is_benign_neutral else "NO"
    cush_str = "YES (HOUSE MONEY CUSHION)" if receipt.is_house_money_cushion else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Barberis, Huang & Santos (2001) Prospect Theory Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print(f"  Reference / Cost-Basis Price:   ${receipt.reference_price:.2f}")
    print("-" * 80)
    print("  Prior Return & Cushion State Parameters:")
    print(f"    Prior Return Relative to Ref:      {receipt.prior_return_pct:>+10.2f}%")
    print(f"    Prior Cushion State (z_t):         {receipt.prior_cushion_state_z:>10.2f} (P_ref / P_t)")
    print(f"    House Money Cushion:               {receipt.house_money_cushion_pct:>10.2f}%")
    print(f"    Dynamic Loss Aversion (lambda):    {receipt.dynamic_loss_aversion_lambda:>10.2f}x")
    print("-" * 80)
    print("  Prospect Theory Required Return & Surcharge Inferences:")
    print(f"    CAPM Market Benchmark Return:      {receipt.capm_benchmark_return_pct:>10.2f}% / year")
    print(f"    Prospective Loss Surcharge:        {receipt.prospective_loss_surcharge_pct:>+10.2f}% / year")
    print(f"    Prospective Loss Surcharge (bps):  {receipt.prospective_loss_surcharge_bps:>+10.1f} bps / year")
    print(f"    Total Prospect Hurdle Return:      {receipt.total_prospect_hurdle_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Prospect Theory Regime Classification:")
    print(f"    Behavioral Risk Regime:            {receipt.regime}")
    print(f"    Acute Loss Aversion Panic:         {acute_str}")
    print(f"    Moderate Recession Sensitivity:    {mod_str}")
    print(f"    Benign Neutral Prospect:           {ben_str}")
    print(f"    House Money Cushion:               {cush_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_bhs_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def allen_gale_command(args: argparse.Namespace) -> None:
    """CLI handler for Franklin Allen & Douglas Gale (2000) Network Contagion Engine."""
    import json
    from src.analytics.allen_gale_network import (
        evaluate_allen_gale_network_profile,
        calculate_allen_gale_network,
        verify_allen_gale_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    comp_val = getattr(args, "network_completeness", None)
    if comp_val is not None:
        comp_val = float(comp_val)

    shock_val = getattr(args, "liquidity_shock", None)
    if shock_val is not None:
        shock_val = float(shock_val)

    agg_l = getattr(args, "aggregate_liquidity", None)
    if agg_l is not None:
        agg_l = float(agg_l)

    try:
        if comp_val is not None and shock_val is not None:
            kwargs = {
                "ticker": ticker,
                "network_completeness_ratio": comp_val,
                "local_liquidity_shock_ratio": shock_val,
            }
            if agg_l is not None:
                kwargs["aggregate_liquidity_buffer"] = agg_l
            receipt = calculate_allen_gale_network(**kwargs)
        else:
            receipt = evaluate_allen_gale_network_profile(
                ticker=ticker,
                network_completeness_override=comp_val,
                liquidity_shock_override=shock_val,
            )
    except Exception as ex:
        print(f"\nError: Allen & Gale Network Contagion evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    casc_str = "YES (SYSTEMIC CASCADE COLLAPSE)" if receipt.is_systemic_cascade else "NO"
    frag_str = "YES (INCOMPLETE FRAGILE NETWORK)" if receipt.is_incomplete_fragile else "NO"
    resil_str = "YES (RESILIENT PARTIAL SHARING)" if receipt.is_resilient_sharing else "NO"
    fort_str = "YES (COMPLETE NETWORK FORTRESS)" if receipt.is_complete_fortress else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Allen & Gale (2000) Financial Network Contagion Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Financial Network Topology & Liquidity Parameters:")
    print(f"    Network Completeness Ratio:        {receipt.network_completeness_ratio:>10.2%}")
    print(f"    Localized Liquidity Shock:         {receipt.local_liquidity_shock_ratio:>10.2%}")
    print(f"    Cross-Holding Buffer Capacity:     {receipt.cross_holding_absorption_capacity:>10.2f}")
    print(f"    Unabsorbed Liquidity Deficit:      {receipt.unabsorbed_liquidity_deficit:>10.2f}")
    print("-" * 80)
    print("  Allen & Gale (2000) Network Cascade & Contagion Inferences:")
    print(f"    Contagion Cascade Multiplier:      {receipt.contagion_cascade_multiplier:>10.2f}x")
    print(f"    Systemic Contagion Index:          {receipt.systemic_contagion_index:>10.1f} / 100")
    print(f"    Network Fire-Sale Price Markdown:  {receipt.network_fire_sale_markdown_pct:>10.2f}%")
    print(f"    Systemic Hurdle Risk Surcharge:    {receipt.systemic_hurdle_surcharge_pct:>+10.2f}% / year")
    print("-" * 80)
    print("  Network Contagion Regime Classification:")
    print(f"    Architecture Topology Regime:      {receipt.regime}")
    print(f"    Systemic Cascade Collapse:         {casc_str}")
    print(f"    Incomplete Fragile Network:        {frag_str}")
    print(f"    Resilient Partial Sharing:         {resil_str}")
    print(f"    Complete Network Fortress:         {fort_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_allen_gale_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def leverage_cycle_command(args: argparse.Namespace) -> None:
    """CLI handler for John Geanakoplos (2010) Leverage Cycle & Collateral Haircut Engine."""
    import json
    from src.analytics.geanakoplos_leverage_cycle import (
        evaluate_geanakoplos_leverage_profile,
        calculate_geanakoplos_leverage,
        verify_leverage_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    m_val = getattr(args, "haircut", None)
    if m_val is not None:
        m_val = float(m_val)

    m0_val = getattr(args, "baseline_haircut", None)
    if m0_val is not None:
        m0_val = float(m0_val)

    vol_val = getattr(args, "volatility", None)
    if vol_val is not None:
        vol_val = float(vol_val)

    try:
        if m_val is not None:
            kwargs = {
                "ticker": ticker,
                "collateral_haircut_m": m_val,
            }
            if m0_val is not None:
                kwargs["baseline_haircut_m0"] = m0_val
            if vol_val is not None:
                kwargs["asset_volatility"] = vol_val
            receipt = calculate_geanakoplos_leverage(**kwargs)
        else:
            receipt = evaluate_geanakoplos_leverage_profile(
                ticker=ticker,
                haircut_override=m_val,
                baseline_haircut_override=m0_val,
            )
    except Exception as ex:
        print(f"\nError: Geanakoplos Leverage Cycle evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    acute_str = "YES (ACUTE MARGIN SQUEEZE PANIC)" if receipt.is_acute_margin_squeeze else "NO"
    elev_str = "YES (ELEVATED DELEVERAGING PHASE)" if receipt.is_elevated_deleveraging else "NO"
    euph_str = "YES (EUPHORIC HIGH LEVERAGE EXPANSION)" if receipt.is_euphoric_expansion else "NO"
    prud_str = "YES (PRUDENT LOW LEVERAGE STABILITY)" if receipt.is_prudent_stability else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Geanakoplos (2010) Leverage Cycle Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print("-" * 80)
    print("  Collateral Haircut & Leverage Parameters:")
    print(f"    Collateral Margin Haircut (m):     {receipt.collateral_haircut_m:>10.2%}")
    print(f"    Baseline Normal Haircut (m_0):     {receipt.baseline_haircut_m0:>10.2%}")
    print(f"    Effective Leverage Ratio (L):      {receipt.effective_leverage_ratio:>10.2f}x (1 / m)")
    print(f"    Haircut Tightening Ratio:          {receipt.haircut_tightening_ratio:>10.2f}x (m / m_0)")
    print("-" * 80)
    print("  Geanakoplos (2010) De-leveraging & Fire-Sale Inferences:")
    print(f"    Forced De-leveraging Pressure:     {receipt.forced_deleveraging_pressure:>10.1f} / 100")
    print(f"    Collateral Squeeze Markdown:       {receipt.collateral_squeeze_markdown_pct:>10.2f}%")
    print(f"    Volatility Amplification Factor:   {receipt.volatility_amplification_multiple:>10.2f}x")
    print(f"    Leverage Risk Hurdle Surcharge:    {receipt.leverage_hurdle_surcharge_pct:>+10.2f}% / year")
    print(f"    Total Leverage Hurdle Return:      {receipt.total_leverage_hurdle_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Leverage Cycle Regime Classification:")
    print(f"    Collateral Cycle Regime:           {receipt.regime}")
    print(f"    Acute Margin Squeeze Panic:        {acute_str}")
    print(f"    Elevated De-leveraging Phase:      {elev_str}")
    print(f"    Euphoric High Leverage Expansion:  {euph_str}")
    print(f"    Prudent Low Leverage Stability:    {prud_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_leverage_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def diamond_dybvig_command(args: argparse.Namespace) -> None:
    """CLI handler for Douglas W. Diamond & Philip H. Dybvig (1983) Bank Runs, Liquidity Transformation & Sequential Service Engine."""
    import json
    from src.analytics.diamond_dybvig_run import (
        evaluate_diamond_dybvig_profile,
        calculate_diamond_dybvig_run,
        verify_diamond_dybvig_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    reserve_val = getattr(args, "reserve", None)
    if reserve_val is not None:
        reserve_val = float(reserve_val)

    withdrawals_val = getattr(args, "withdrawals", None)
    if withdrawals_val is not None:
        withdrawals_val = float(withdrawals_val)

    r1_val = getattr(args, "r1", None)
    if r1_val is not None:
        r1_val = float(r1_val)

    r2_val = getattr(args, "r2", None)
    if r2_val is not None:
        r2_val = float(r2_val)

    salvage_val = getattr(args, "salvage", None)
    if salvage_val is not None:
        salvage_val = float(salvage_val)

    insured = bool(getattr(args, "insured", False))

    try:
        kwargs = {"ticker": ticker, "deposit_insurance": insured}
        if reserve_val is not None:
            kwargs["liquid_reserve_ratio"] = reserve_val
        if withdrawals_val is not None:
            kwargs["early_withdrawal_ratio"] = withdrawals_val
        if r1_val is not None:
            kwargs["r1_payout"] = r1_val
        if r2_val is not None:
            kwargs["R2_yield"] = r2_val
        if salvage_val is not None:
            kwargs["salvage_L"] = salvage_val

        receipt = evaluate_diamond_dybvig_profile(**kwargs)
    except Exception as ex:
        print(f"\nError: Diamond-Dybvig evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    run_str = "YES (SYSTEMIC COORDINATION RUN)" if receipt.is_systemic_coordination_run else "NO"
    elev_str = "YES (ELEVATED RUN HAZARD)" if receipt.is_elevated_run_hazard else "NO"
    norm_str = "YES (NORMAL STABILITY)" if receipt.is_normal_stability else "NO"
    ins_str = "YES (INSURED SAFETY)" if receipt.is_insured_safety else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Diamond & Dybvig (1983) Bank Runs & Liquidity Transformation")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print(f"  Deposit Insurance Active:       {receipt.deposit_insurance_active}")
    print("-" * 80)
    print("  Liquidity Transformation & Asset Allocation:")
    print(f"    Liquid Reserve Buffer (c):         {receipt.liquid_reserve_ratio:>10.2%}")
    print(f"    Illiquid Productive Pool (1-c):    {receipt.illiquid_asset_ratio:>10.2%}")
    print(f"    Total Liquidation Capacity:        {receipt.total_liquidation_capacity:>10.4f}x")
    print(f"    Premature Salvage Recovery (L):    {receipt.salvage_liquidation_L:>10.2%}")
    print("-" * 80)
    print("  Sequential Service & Coordination Run Dynamics:")
    print(f"    Observed Early Withdrawals (f_1):  {receipt.early_withdrawal_ratio:>10.2%}")
    print(f"    Asset Exhaustion Cutoff (f_bar):   {receipt.exhaustion_threshold_fbar:>10.2%}")
    print(f"    Short-Term Promised Return (r_1):  {receipt.r1_payout:>10.3f}x")
    print(f"    Long-Term Productive Yield (R_2):  {receipt.R2_yield:>10.3f}x")
    print(f"    Patient Depositor Payoff (v_2):    {receipt.patient_t2_expected_payoff:>10.3f}x")
    print(f"    Liquidity Mismatch Ratio (LMR):    {receipt.liquidity_mismatch_ratio:>10.2%}")
    print(f"    Fire-Sale Loss Destroyed:          {receipt.fire_sale_loss_fraction:>10.2%}")
    print(f"    Run Coordination Probability:      {receipt.run_hazard_probability:>10.1%}")
    print(f"    Distressed Asset Price:            ${receipt.distressed_asset_price:>9.2f}")
    print("-" * 80)
    print("  Diamond-Dybvig Equilibrium Regime:")
    print(f"    Active Regime:                     {receipt.regime}")
    print(f"    Systemic Coordination Run:         {run_str}")
    print(f"    Elevated Run Hazard:               {elev_str}")
    print(f"    Normal Stability:                  {norm_str}")
    print(f"    Insured Safety:                    {ins_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_diamond_dybvig_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def merton_icapm_command(args: argparse.Namespace) -> None:
    """CLI handler for Robert C. Merton (1973) Intertemporal CAPM & State Variable Hedging Engine."""
    import json
    from src.analytics.merton_icapm import (
        evaluate_merton_icapm_profile,
        verify_merton_icapm_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    mkt_prem = getattr(args, "market_premium", None)
    if mkt_prem is not None:
        mkt_prem = float(mkt_prem)

    rate_prem = getattr(args, "rate_premium", None)
    if rate_prem is not None:
        rate_prem = float(rate_prem)

    vol_prem = getattr(args, "vol_premium", None)
    if vol_prem is not None:
        vol_prem = float(vol_prem)

    rf_val = getattr(args, "rf", None)
    if rf_val is not None:
        rf_val = float(rf_val)

    try:
        kwargs = {"ticker": ticker}
        if mkt_prem is not None:
            kwargs["lambda_market"] = mkt_prem
        if rate_prem is not None:
            kwargs["lambda_rate"] = rate_prem
        if vol_prem is not None:
            kwargs["lambda_vol"] = vol_prem
        if rf_val is not None:
            kwargs["risk_free_rate"] = rf_val

        receipt = evaluate_merton_icapm_profile(**kwargs)
    except Exception as ex:
        print(f"\nError: Merton ICAPM evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    vol_str = "YES (VOLATILITY HEDGE DEFENSE)" if receipt.is_volatility_hedge_defense else "NO"
    dur_str = "YES (DURATION / INTEREST RATE STRESS)" if receipt.is_duration_stress else "NO"
    exp_str = "YES (AGGRESSIVE OPPORTUNITY EXPANSION)" if receipt.is_aggressive_expansion else "NO"
    bal_str = "YES (STEADY STATE BALANCED)" if receipt.is_steady_state else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Merton (1973) Intertemporal CAPM & State Variable Hedging")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print(f"  Historical Realized Return:     {receipt.annualized_asset_return:>+10.2f}% / year")
    print(f"  Risk-Free Rate (R_f):           {receipt.risk_free_rate_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Intertemporal Multi-Beta Factor Exposures:")
    print(f"    Market Beta (beta_M):              {receipt.market_beta:>10.4f} (Risk Price: {receipt.market_premium_pct:>+.2f}%)")
    print(f"    Term Structure Beta (beta_r):      {receipt.interest_rate_beta:>10.4f} (Risk Price: {receipt.interest_rate_premium_pct:>+.2f}%)")
    print(f"    Volatility Hedge Beta (beta_vol):  {receipt.volatility_beta:>10.4f} (Risk Price: {receipt.volatility_premium_pct:>+.2f}%)")
    print(f"    Multi-Factor Regression R^2:       {receipt.r_squared:>10.2%}")
    print("-" * 80)
    print("  Intertemporal Hedging & Hurdle Decomposition:")
    print(f"    Market Risk Contribution:          {receipt.market_risk_contribution_pct:>+10.2f}% / year")
    print(f"    Interest Rate Hedging Premium:     {receipt.rate_hedging_contribution_pct:>+10.2f}% / year")
    print(f"    Volatility Hedging Premium:        {receipt.vol_hedging_contribution_pct:>+10.2f}% / year")
    print(f"    Total State Hedging Premium:       {receipt.total_state_hedging_premium_pct:>+10.2f}% / year")
    print(f"    ICAPM Expected Excess Return:      {receipt.icapm_expected_excess_return_pct:>+10.2f}% / year")
    print(f"    ICAPM Required Hurdle Rate:        {receipt.icapm_hurdle_rate_pct:>10.2f}% / year")
    print(f"    ICAPM Intertemporal Alpha:         {receipt.icapm_alpha_pct:>+10.2f}% / year")
    print("-" * 80)
    print("  Hedging Properties & Regime Classification:")
    print(f"    Volatility Hedge Asset:            {receipt.is_volatility_hedge}")
    print(f"    Interest Rate Hedge Asset:         {receipt.is_interest_rate_hedge}")
    print(f"    Active ICAPM Regime:               {receipt.regime}")
    print(f"    Volatility Hedge Defense:          {vol_str}")
    print(f"    Duration Stress:                   {dur_str}")
    print(f"    Aggressive Expansion:              {exp_str}")
    print(f"    Steady State Balanced:             {bal_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_merton_icapm_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def ross_apt_command(args: argparse.Namespace) -> None:
    """CLI handler for Stephen A. Ross (1976) Arbitrage Pricing Theory & Macro Multi-Factor Engine."""
    import json
    from src.analytics.ross_apt import (
        evaluate_ross_apt_profile,
        verify_ross_apt_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    growth_prem = getattr(args, "growth_premium", None)
    if growth_prem is not None:
        growth_prem = float(growth_prem)

    inf_prem = getattr(args, "inflation_premium", None)
    if inf_prem is not None:
        inf_prem = float(inf_prem)

    def_prem = getattr(args, "default_premium", None)
    if def_prem is not None:
        def_prem = float(def_prem)

    term_prem = getattr(args, "term_premium", None)
    if term_prem is not None:
        term_prem = float(term_prem)

    rf_val = getattr(args, "rf", None)
    if rf_val is not None:
        rf_val = float(rf_val)

    try:
        kwargs = {"ticker": ticker}
        if growth_prem is not None:
            kwargs["lambda_growth"] = growth_prem
        if inf_prem is not None:
            kwargs["lambda_inflation"] = inf_prem
        if def_prem is not None:
            kwargs["lambda_default"] = def_prem
        if term_prem is not None:
            kwargs["lambda_term"] = term_prem
        if rf_val is not None:
            kwargs["risk_free_rate"] = rf_val

        receipt = evaluate_ross_apt_profile(**kwargs)
    except Exception as ex:
        print(f"\nError: Ross APT evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    buy_str = "YES (STATISTICAL UNDERPRICING ALPHA)" if receipt.is_arbitrage_buy else "NO"
    sell_str = "YES (STATISTICAL OVERPRICING DEFICIT)" if receipt.is_arbitrage_sell else "NO"
    macro_str = "YES (SYSTEMIC FACTOR RISK DOMINANCE)" if receipt.is_macro_dominated else "NO"
    idle_str = "YES (FAIRLY PRICED NO-ARBITRAGE EQUILIBRIUM)" if receipt.is_idiosyncratic_idle else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Stephen A. Ross (1976) Arbitrage Pricing Theory (APT)")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Latest Market Price:            ${receipt.latest_price:.2f}")
    print(f"  Historical Realized Return:     {receipt.annualized_asset_return_pct:>+10.2f}% / year")
    print(f"  Risk-Free Benchmark (R_f):      {receipt.risk_free_rate_pct:>10.2f}% / year")
    print("-" * 80)
    print("  Chen-Roll-Ross (1986) Systematic Macroeconomic Loadings:")
    print(f"    Growth Factor Loading (b_growth):  {receipt.growth_factor_beta:>10.4f} (Risk Price: {receipt.growth_premium_pct:>+.2f}%)")
    print(f"    Inflation Shock Beta (b_inf):      {receipt.inflation_factor_beta:>10.4f} (Risk Price: {receipt.inflation_premium_pct:>+.2f}%)")
    print(f"    Credit Default Beta (b_def):       {receipt.default_factor_beta:>10.4f} (Risk Price: {receipt.default_premium_pct:>+.2f}%)")
    print(f"    Term Structure Beta (b_term):      {receipt.term_factor_beta:>10.4f} (Risk Price: {receipt.term_premium_pct:>+.2f}%)")
    print("-" * 80)
    print("  Variance & Factor Risk Premium Decomposition:")
    print(f"    Growth Risk Contribution:          {receipt.growth_risk_contribution_pct:>+10.2f}% / year")
    print(f"    Inflation Risk Contribution:       {receipt.inflation_risk_contribution_pct:>+10.2f}% / year")
    print(f"    Default Risk Contribution:         {receipt.default_risk_contribution_pct:>+10.2f}% / year")
    print(f"    Term Structure Contribution:       {receipt.term_risk_contribution_pct:>+10.2f}% / year")
    print(f"    Total Systematic Risk Premium:     {receipt.total_systematic_risk_premium_pct:>+10.2f}% / year")
    print(f"    APT Expected No-Arbitrage Return:  {receipt.apt_expected_return_pct:>10.2f}% / year")
    print(f"    Arbitrage Spread (Delta_APT):      {receipt.apt_arbitrage_spread_pct:>+10.2f}% / year")
    print(f"    Systematic Variance Ratio:         {receipt.systematic_variance_ratio_pct:>10.2f}%")
    print(f"    Idiosyncratic Variance Ratio:      {receipt.idiosyncratic_variance_ratio_pct:>10.2f}%")
    print("-" * 80)
    print("  APT Equilibrium Regime Classification:")
    print(f"    Active APT Regime:                 {receipt.regime}")
    print(f"    Arbitrage Buy Opportunity:         {buy_str}")
    print(f"    Arbitrage Sell Overpricing:        {sell_str}")
    print(f"    Macro Factor Dominated:            {macro_str}")
    print(f"    Idiosyncratic Idle Equilibrium:    {idle_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_ross_apt_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def dssw_command(args: argparse.Namespace) -> None:
    """CLI handler for De Long, Shleifer, Summers & Waldmann (1990) Noise Trader Risk Engine."""
    import json
    from src.analytics.dssw_noise_trader_risk import (
        evaluate_dssw_profile,
        verify_dssw_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    mu_val = getattr(args, "mean_sentiment", None)
    if mu_val is not None:
        mu_val = float(mu_val)

    st_val = getattr(args, "shock", None)
    if st_val is not None:
        st_val = float(st_val)

    var_val = getattr(args, "variance", None)
    if var_val is not None:
        var_val = float(var_val)

    gamma_val = getattr(args, "risk_aversion", None)
    if gamma_val is not None:
        gamma_val = float(gamma_val)

    rf_val = getattr(args, "rf", None)
    if rf_val is not None:
        rf_val = float(rf_val)

    try:
        kwargs = {"ticker": ticker}
        if mu_val is not None:
            kwargs["mean_sentiment_override"] = mu_val
        if st_val is not None:
            kwargs["sentiment_shock_override"] = st_val
        if var_val is not None:
            kwargs["sentiment_variance_override"] = var_val
        if gamma_val is not None:
            kwargs["risk_aversion"] = gamma_val
        if rf_val is not None:
            kwargs["risk_free_rate"] = rf_val

        receipt = evaluate_dssw_profile(**kwargs)
    except Exception as ex:
        print(f"\nError: DSSW Noise Trader Risk evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    euph_str = "YES (SENTIMENT EUPHORIA)" if receipt.is_sentiment_euphoria else "NO"
    panic_str = "YES (PANIC DEPRESSION WEDGE)" if receipt.is_panic_depression else "NO"
    risk_str = "YES (CREATE-SPACE ARBITRAGE RISK DOMINATED)" if receipt.is_arbitrage_risk_dominated else "NO"
    fund_str = "YES (RATIONAL FUNDAMENTAL EQUILIBRIUM)" if receipt.is_fundamental_equilibrium else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: De Long, Shleifer, Summers & Waldmann (1990) Noise Trader Risk")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Fundamental Intrinsic Value:    ${receipt.fundamental_price:.2f}")
    print(f"  DSSW Equilibrium Market Price:  ${receipt.dssw_market_price:.2f}")
    print(f"  Mispricing Dollar Wedge:        ${receipt.mispricing_dollar_wedge:>+9.2f} ({receipt.mispricing_percentage_pct:>+6.2f}%)")
    print("-" * 80)
    print("  Noise Trader Sentiment & Arbitrage Risk Parameters:")
    print(f"    Average Sentiment Bias (mu):       {receipt.mean_sentiment_mu:>+10.4f}")
    print(f"    Active Sentiment Shock (S_t):      {receipt.active_sentiment_shock:>+10.4f}")
    print(f"    Sentiment Variance (sigma_S^2):    {receipt.sentiment_variance_sigma_sq:>10.4f}")
    print(f"    Arbitrageur Risk Aversion (gamma): {receipt.risk_aversion_gamma:>10.2f}")
    print(f"    Discount / Risk-Free Rate (r):     {receipt.risk_free_rate_pct:>10.2f}% / year")
    print("-" * 80)
    print("  DSSW Equilibrium Component Price Breakdown:")
    print(f"    Persistent Sentiment Premium:      ${receipt.persistent_bias_premium_dollar:>+9.2f}")
    print(f"    Transitory Sentiment Wedge:        ${receipt.transitory_shock_wedge_dollar:>+9.2f}")
    print(f"    Create-Space Volatility Discount: -${receipt.create_space_discount_dollar:>9.2f}")
    print("-" * 80)
    print("  Noise Trader Performance & Limits of Arbitrage:")
    print(f"    Return Differential (E[R_N] - E[R_A]): {receipt.expected_return_differential_pct:>+8.2f}% / year")
    print(f"    Noise Traders Outperforming Arbs:      {receipt.is_noise_trader_outperforming}")
    print("-" * 80)
    print("  DSSW Equilibrium Regime:")
    print(f"    Active Regime:                     {receipt.regime}")
    print(f"    Sentiment Euphoria:                {euph_str}")
    print(f"    Panic Depression:                  {panic_str}")
    print(f"    Arbitrage Risk Dominated:          {risk_str}")
    print(f"    Fundamental Equilibrium:           {fund_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_dssw_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def black_leverage_command(args: argparse.Namespace) -> None:
    """CLI handler for Fischer Black (1976) & Campbell-Hentschel (1992) Leverage Effect Engine."""
    import json
    from src.analytics.black_leverage_effect import (
        evaluate_black_leverage_profile,
        verify_black_leverage_audit_digest,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    debt_val = getattr(args, "debt", None)
    if debt_val is not None:
        debt_val = float(debt_val)

    vol_val = getattr(args, "asset_vol", None)
    if vol_val is not None:
        vol_val = float(vol_val)

    feedback_val = getattr(args, "feedback", None)
    if feedback_val is not None:
        feedback_val = float(feedback_val)

    try:
        kwargs = {"ticker": ticker}
        if debt_val is not None:
            kwargs["debt_per_share_override"] = debt_val
        if vol_val is not None:
            kwargs["asset_volatility_override"] = vol_val
        if feedback_val is not None:
            kwargs["feedback_elasticity"] = feedback_val

        receipt = evaluate_black_leverage_profile(**kwargs)
    except Exception as ex:
        print(f"\nError: Black Leverage Effect evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    acute_str = "YES (ACUTE LEVERAGE AMPLIFICATION)" if receipt.is_acute_leverage_amplification else "NO"
    feed_str = "YES (VOLATILITY FEEDBACK CASCADE)" if receipt.is_volatility_feedback_cascade else "NO"
    unlev_str = "YES (UNLEVERED INSULATION)" if receipt.is_unlevered_insulation else "NO"
    mild_str = "YES (MILD ASYMMETRY STABILITY)" if receipt.is_mild_asymmetry_stability else "NO"

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black (1976) & Campbell-Hentschel (1992) Leverage Effect")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Current Stock Price:            ${receipt.market_price:.2f}")
    print(f"  Debt per Share:                 ${receipt.debt_per_share:.2f}")
    print(f"  Debt-to-Equity Ratio (D/E):      {receipt.debt_to_equity_ratio:>10.2f}x")
    print("-" * 80)
    print("  Capital Structure & Volatility Amplification:")
    print(f"    Intrinsic Asset Volatility:        {receipt.asset_volatility_pct:>10.2f}% / year")
    print(f"    Equity Volatility \\sigma_E(P):      {receipt.equity_volatility_pct:>10.2f}% / year")
    print(f"    Leverage Volatility Multiplier:    {receipt.leverage_volatility_multiplier:>10.2f}x (1 + D/P)")
    print(f"    Price Sensitivity (d\\sigma_E/dP):   {receipt.volatility_price_sensitivity:>10.6f}")
    print("-" * 80)
    print("  Empirical Asymmetry & Volatility Feedback:")
    print(f"    Upside Return Volatility:          {receipt.realized_upside_volatility_pct:>10.2f}% / year")
    print(f"    Downside Return Volatility:        {receipt.realized_downside_volatility_pct:>10.2f}% / year")
    print(f"    Volatility Asymmetry Ratio (VAR):  {receipt.volatility_asymmetry_ratio:>10.2f}x")
    print(f"    Volatility Feedback Markdown:     -${receipt.volatility_feedback_discount_dollar:>9.2f} (-{receipt.volatility_feedback_discount_pct:.1f}%)")
    print(f"    Feedback Adjusted Price:          ${receipt.feedback_adjusted_price:>9.2f}")
    print("-" * 80)
    print("  Leverage & Asymmetry Regime:")
    print(f"    Active Regime:                     {receipt.regime}")
    print(f"    Acute Leverage Amplification:      {acute_str}")
    print(f"    Volatility Feedback Cascade:       {feed_str}")
    print(f"    Unlevered Insulation:              {unlev_str}")
    print(f"    Mild Asymmetry Stability:          {mild_str}")
    print("-" * 80)
    print(f"  Audit Digest (HMAC):            {receipt.audit_digest[:16]}... (Verified: {verify_black_leverage_audit_digest(receipt)})")
    print(f"  Summary: {receipt.summary}")
    print("=" * 80 + "\n")


def lucas_tree_command(args: argparse.Namespace) -> None:
    """CLI handler for Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium Engine."""
    import json
    from src.analytics.lucas_tree_equilibrium import (
        calculate_lucas_tree_equilibrium,
        evaluate_lucas_tree_profile,
        persist_lucas_tree_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta = getattr(args, "beta", None)
    beta_val = float(beta) if beta is not None else 0.96

    gamma = getattr(args, "gamma", None)
    gamma_val = float(gamma) if gamma is not None else 2.50

    growth_mean = getattr(args, "growth_mean", None)
    growth_vol = getattr(args, "growth_vol", None)
    dividend = getattr(args, "current_dividend", None)

    try:
        if growth_mean is not None and growth_vol is not None:
            div_val = float(dividend) if dividend is not None else 2.0
            receipt = calculate_lucas_tree_equilibrium(
                current_dividend=div_val,
                subjective_discount_factor=beta_val,
                relative_risk_aversion=gamma_val,
                dividend_growth_mean=float(growth_mean),
                dividend_growth_vol=float(growth_vol),
                ticker=ticker,
            )
        else:
            receipt = evaluate_lucas_tree_profile(
                prices=[],
                ticker=ticker,
                beta=beta_val,
                gamma=gamma_val,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_lucas_tree_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Lucas Tree Equilibrium evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Current Dividend Payout (D_t):  ${receipt.current_dividend:.2f}")
    print(f"  Subjective Discount Factor (beta): {receipt.subjective_discount_factor:.4f}")
    print(f"  Relative Risk Aversion (gamma): {receipt.relative_risk_aversion:.2f}")
    print("-" * 80)
    print("  Dividend Growth Process ln(g) ~ N(mu, sigma^2):")
    print(f"    Expected Dividend Growth (mu):  {receipt.dividend_growth_mean * 100:.2f}% / period")
    print(f"    Dividend Growth Volatility (sigma): {receipt.dividend_growth_vol * 100:.2f}%")
    print(f"    Expected Gross Growth E[g]:    {receipt.expected_dividend_growth:.4f}x")
    print("-" * 80)
    print("  Equilibrium Asset Pricing & Transversality:")
    print(f"    Transversality Ratio (kappa):   {receipt.transversality_ratio:.4f} (< 1.0 required)")
    print(f"    Transversality Satisfied:       {'YES' if receipt.transversality_satisfied else 'NO (BUBBLE)'}")
    print(f"    Price-Dividend Ratio (P/D):     {receipt.price_dividend_ratio:.2f}x")
    print(f"    Theoretical Equilibrium Price:  ${receipt.equilibrium_price:.2f}")
    print("-" * 80)
    print("  Return Dynamics & Hansen-Jagannathan Bounds:")
    print(f"    Risk-Free Gross Rate (R_f):     {receipt.risk_free_gross_rate:.4f} ({(receipt.risk_free_gross_rate - 1.0) * 100:.2f}%)")
    print(f"    Expected Equity Return E[R]:    {receipt.expected_equity_gross_return:.4f} ({(receipt.expected_equity_gross_return - 1.0) * 100:.2f}%)")
    print(f"    Theoretical Equity Premium:     {receipt.equity_premium * 100:.2f}%")
    print(f"    Hansen-Jagannathan Bound:       {receipt.hansen_jagannathan_bound:.4f} (Max Sharpe)")
    print("-" * 80)
    print(f"  Lucas Regime:                   {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def harrison_kreps_command(args: argparse.Namespace) -> None:
    """CLI handler for J. Michael Harrison & David M. Kreps (1978) Speculative Disagreement & Resale Engine."""
    import json
    from src.analytics.harrison_kreps_resale import (
        calculate_harrison_kreps_equilibrium,
        evaluate_harrison_kreps_profile,
        persist_harrison_kreps_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta = getattr(args, "beta", None)
    beta_val = float(beta) if beta is not None else 0.95

    spread = getattr(args, "belief_spread", None)
    spread_val = float(spread) if spread is not None else 0.20

    try:
        receipt = evaluate_harrison_kreps_profile(
            prices=[],
            ticker=ticker,
            discount_factor=beta_val,
            belief_spread_override=spread_val,
        )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_harrison_kreps_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Harrison-Kreps evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Harrison & Kreps (1978) Speculative Disagreement & Resale Option")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Discount Factor (beta):         {receipt.discount_factor:.4f}")
    print(f"  Belief Divergence Spread:       {receipt.belief_divergence * 100:.1f}%")
    print(f"  Number of Dividend States:      {receipt.states_count}")
    print("-" * 80)
    print("  State Pricing & Resale Option Premium Decomposition:")
    for s in range(receipt.states_count):
        d_val = receipt.dividend_states[s]
        p_val = receipt.equilibrium_prices[s]
        va_val = receipt.fundamental_val_A[s]
        vb_val = receipt.fundamental_val_B[s]
        vmax_val = receipt.max_fundamental_val[s]
        resale_val = receipt.resale_option_premia[s]
        bubble_val = receipt.speculative_bubble_ratios[s] * 100.0
        owner = receipt.active_owner_states[s]

        print(f"    State {s} [Dividend D={d_val:.2f}]:")
        print(f"      Equilibrium Price P*(s):     ${p_val:>8.2f} (Active Owner: {owner})")
        print(f"      Fundamental Buy-Hold V_A:    ${va_val:>8.2f} | V_B: ${vb_val:>8.2f} (V_max: ${vmax_val:>8.2f})")
        print(f"      Resale Option Premium:      +${resale_val:>8.2f} ({bubble_val:.1f}% of Price)")
    print("-" * 80)
    print("  Aggregate Speculative Dynamics:")
    print(f"    Mean Resale Option Premium:   ${receipt.mean_resale_premium:.2f}")
    print(f"    Mean Speculative Bubble Ratio: {receipt.mean_bubble_ratio * 100:.2f}%")
    print(f"    Active Harrison-Kreps Regime:  {receipt.regime}")
    print("-" * 80)
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def brennan_agency_command(args: argparse.Namespace) -> None:
    """CLI handler for Michael J. Brennan (1998) Two-Beta Agency Asset Pricing Engine."""
    import json
    from src.analytics.brennan_agency_pricing import (
        calculate_brennan_agency_pricing,
        evaluate_brennan_agency_profile,
        persist_brennan_agency_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta_m = getattr(args, "beta_m", None)
    beta_b = getattr(args, "beta_b", None)
    inst_share = getattr(args, "inst_share", None)
    tracking_aversion = getattr(args, "tracking_aversion", None)

    try:
        if beta_m is not None and beta_b is not None:
            receipt = calculate_brennan_agency_pricing(
                beta_market=float(beta_m),
                beta_benchmark=float(beta_b),
                institutional_share=float(inst_share) if inst_share is not None else 0.70,
                tracking_aversion_rel=float(tracking_aversion) if tracking_aversion is not None else 5.0,
                ticker=ticker,
            )
        else:
            receipt = evaluate_brennan_agency_profile(
                asset_prices=[],
                ticker=ticker,
                institutional_share=float(inst_share) if inst_share is not None else 0.70,
                tracking_aversion=float(tracking_aversion) if tracking_aversion is not None else 5.0,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_brennan_agency_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Brennan Agency Pricing evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Michael J. Brennan (1998) Two-Beta Agency Asset Pricing")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Market Beta (beta_M):           {receipt.beta_market:.2f}")
    print(f"  Benchmark Beta (beta_B):        {receipt.beta_benchmark:.2f}")
    print(f"  Benchmark Overweight Ratio:     {receipt.benchmark_overweight_ratio:.2f}x (beta_B / beta_M)")
    print("-" * 80)
    print("  Institutional Delegation & Tracking Penalties:")
    print(f"    Institutional Wealth Share:    {receipt.institutional_share * 100:.1f}%")
    print(f"    Absolute Risk Aversion (gamma): {receipt.risk_aversion_abs:.2f}")
    print(f"    Tracking Error Penalty (gamma): {receipt.tracking_aversion_rel:.2f}")
    print(f"    Agency Risk Intensity (lambda): {receipt.agency_risk_intensity * 100:.4f}%")
    print("-" * 80)
    print("  Asset Pricing & Agency Alpha Decomposition:")
    print(f"    Standard CAPM Expected Return: {receipt.expected_return_capm * 100:.2f}%")
    print(f"    Brennan Agency Expected Return: {receipt.expected_return_agency * 100:.2f}%")
    print(f"    Agency Distortion Spread:      {receipt.agency_distortion_spread * 100:+.2f}%")
    print(f"    Agency Alpha Premium:          {receipt.agency_alpha * 100:+.2f}%")
    print("-" * 80)
    print(f"  Brennan Agency Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def shleifer_vishny_command(args: argparse.Namespace) -> None:
    """CLI handler for Andrei Shleifer & Robert W. Vishny (1997) Limits of Arbitrage Engine."""
    import json
    from src.analytics.shleifer_vishny_arbitrage import (
        calculate_shleifer_vishny_equilibrium,
        evaluate_shleifer_vishny_profile,
        persist_shleifer_vishny_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    fund_val = getattr(args, "fundamental_val", None)
    shock_sev = getattr(args, "shock_severity", None)
    flow_sens = getattr(args, "flow_sensitivity", None)
    capital = getattr(args, "capital", None)
    cap_val = float(capital) if capital is not None else 1000.0
    flow_val = float(flow_sens) if flow_sens is not None else 1.20

    try:
        if fund_val is not None and shock_sev is not None:
            v_val = float(fund_val)
            p0_val = v_val * 0.90
            p1_val = p0_val * (1.0 - float(shock_sev))
            receipt = calculate_shleifer_vishny_equilibrium(
                fundamental_value=v_val,
                initial_price=p0_val,
                adverse_shock_price=p1_val,
                flow_sensitivity=flow_val,
                initial_capital=cap_val,
                ticker=ticker,
            )
        else:
            receipt = evaluate_shleifer_vishny_profile(
                prices=[],
                ticker=ticker,
                flow_sensitivity=flow_val,
                initial_capital=cap_val,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_shleifer_vishny_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Shleifer-Vishny evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Shleifer & Vishny (1997) Limits of Arbitrage & PBA Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Fundamental Liquidation Value:  ${receipt.fundamental_value:.2f}")
    print(f"  Initial Market Price (P_0):     ${receipt.initial_price:.2f}")
    print(f"  Adverse Shock Price (P_1):      ${receipt.adverse_shock_price:.2f}")
    print(f"  Initial Mispricing:             ${receipt.initial_mispricing_dollar:.2f} ({receipt.initial_mispricing_pct * 100:.2f}%)")
    print("-" * 80)
    print("  Performance-Based Arbitrage & Capital Structure:")
    print(f"    Initial Arbitrage Capital:     ${receipt.initial_capital:.2f}")
    print(f"    Optimal Position (x_0*):       {receipt.optimal_initial_position:.2f} units (${receipt.capital_deployed:.2f} deployed)")
    print(f"    Precautionary Cash Buffer:     ${receipt.cash_buffer:.2f}")
    print(f"    Investor Flow Sensitivity (a): {receipt.flow_sensitivity:.2f}x")
    print("-" * 80)
    print("  Interim Widening & Distress Liquidation Risk:")
    print(f"    Interim Mark-to-Market Loss:   {receipt.interim_loss_pct * 100:.2f}%")
    print(f"    Redemption Cash Call:          ${receipt.redemption_amount:.2f}")
    print(f"    Forced Fire-Sale Liquidation:  {receipt.forced_liquidation_units:.2f} units ({receipt.forced_liquidation_pct * 100:.2f}% of holding)")
    print(f"    Residual Mispricing Ratio:     {receipt.residual_mispricing_ratio * 100:.2f}%")
    print("-" * 80)
    print(f"  Limits-of-Arbitrage Regime:     {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def myers_majluf_command(args: argparse.Namespace) -> None:
    """CLI handler for Stewart C. Myers & Nicholas S. Majluf (1984) Pecking Order Engine."""
    import json
    from src.analytics.myers_majluf_pecking_order import (
        calculate_myers_majluf_equilibrium,
        evaluate_myers_majluf_profile,
        persist_myers_majluf_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    assets = getattr(args, "assets", None)
    outlay = getattr(args, "outlay", None)
    payoff = getattr(args, "payoff", None)
    slack = getattr(args, "cash_slack", None)
    spread = getattr(args, "info_spread", None)
    slack_val = float(slack) if slack is not None else 10.0
    spread_val = float(spread) if spread is not None else 0.15

    try:
        if assets is not None and outlay is not None and payoff is not None:
            a_val = float(assets)
            p_val = a_val / (1.0 + spread_val)
            receipt = calculate_myers_majluf_equilibrium(
                assets_in_place_true=a_val,
                assets_market_price=p_val,
                project_outlay=float(outlay),
                project_payoff=float(payoff),
                cash_slack=slack_val,
                information_asymmetry_spread=spread_val,
                ticker=ticker,
            )
        else:
            receipt = evaluate_myers_majluf_profile(
                prices=[],
                ticker=ticker,
                cash_slack=slack_val,
                capex_outlay=float(outlay) if outlay is not None else None,
                info_spread=spread_val,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_myers_majluf_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Myers-Majluf evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Myers & Majluf (1984) Pecking Order & Dilution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Assets in Place (True Value A): ${receipt.assets_in_place_true:.2f}")
    print(f"  Market Price (P):               ${receipt.assets_market_price:.2f}")
    print(f"  Information Asymmetry Spread:   {receipt.information_asymmetry_spread * 100:.1f}% (Undervaluation: ${receipt.undervaluation_wedge:.2f})")
    print("-" * 80)
    print("  Project Investment & Financial Slack:")
    print(f"    Project Investment Outlay (I): ${receipt.project_outlay:.2f}")
    print(f"    Expected Terminal Payoff (B):  ${receipt.project_payoff:.2f}")
    print(f"    Project Net Present Value:     ${receipt.project_npv:+.2f}")
    print(f"    Internal Cash Slack (C):       ${receipt.cash_slack:.2f}")
    print(f"    External Financing Needed:     ${receipt.external_financing_needed:.2f} ({receipt.financing_source})")
    print("-" * 80)
    print("  Equity Offering Dilution & Underinvestment Status:")
    print(f"    Dilution Transfer to New Shs:  ${receipt.dilution_transfer:.2f}")
    print(f"    Net Old Shareholder Gain:      ${receipt.net_old_shareholder_gain:+.2f}")
    print(f"    Underinvestment Trap:          {'YES (REJECTED POSITIVE NPV)' if receipt.underinvestment_triggered else 'NO (INVESTMENT PROCEEDS)'}")
    print(f"    Offering Lemons Markdown:      -${receipt.equity_lemons_discount_dollar:.2f} (-{receipt.equity_lemons_discount_pct * 100:.2f}%)")
    print("-" * 80)
    print(f"  Myers-Majluf Regime:            {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def breeden_litzenberger_command(args: argparse.Namespace) -> None:
    """CLI handler for Breeden & Litzenberger (1978) Arrow-Debreu State Price & RND Engine."""
    import json
    from src.analytics.breeden_litzenberger_rnd import (
        calculate_breeden_litzenberger_rnd,
        evaluate_breeden_litzenberger_profile,
        generate_skewed_call_surface,
        persist_breeden_litzenberger_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    spot = getattr(args, "spot", None)
    vol_skew = float(getattr(args, "vol_skew", 0.22) if getattr(args, "vol_skew", None) is not None else 0.22)
    vol_smile = float(getattr(args, "vol_smile", 0.04) if getattr(args, "vol_smile", None) is not None else 0.04)
    rate = float(getattr(args, "rate", 0.04) if getattr(args, "rate", None) is not None else 0.04)
    expiry = float(getattr(args, "expiry", 0.25) if getattr(args, "expiry", None) is not None else 0.25)
    num_strikes = int(getattr(args, "num_strikes", 49) if getattr(args, "num_strikes", None) is not None else 49)

    try:
        if spot is not None:
            s0 = float(spot)
            strikes, calls = generate_skewed_call_surface(
                spot_price=s0,
                atm_vol=0.25,
                vol_skew=vol_skew,
                vol_smile=vol_smile,
                risk_free_rate=rate,
                time_to_expiry=expiry,
                num_strikes=num_strikes,
            )
            receipt = calculate_breeden_litzenberger_rnd(
                spot_price=s0,
                strikes=strikes,
                call_prices=calls,
                risk_free_rate=rate,
                time_to_expiry=expiry,
                ticker=ticker,
            )
        else:
            receipt = evaluate_breeden_litzenberger_profile(
                prices=[],
                ticker=ticker,
                vol_skew=vol_skew,
                vol_smile=vol_smile,
                risk_free_rate=rate,
                time_to_expiry=expiry,
                num_strikes=num_strikes,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_breeden_litzenberger_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Breeden-Litzenberger evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Breeden & Litzenberger (1978) Arrow-Debreu State Prices & RND Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Spot Price (S_0):               ${receipt.spot_price:.2f} (Strikes: {receipt.num_strikes})")
    print(f"  Risk-Free Rate / Expiry (r, T): {receipt.risk_free_rate * 100:.2f}% | {receipt.time_to_expiry:.2f} yrs")
    print(f"  Theoretical Forward (F_0):      ${receipt.theoretical_forward:.2f}")
    print("-" * 80)
    print("  Recovered Risk-Neutral Density (RND) Moments:")
    print(f"    Risk-Neutral Mean E^Q[S_T]:   ${receipt.rnd_mean:.2f}")
    print(f"    Risk-Neutral Std Dev:         ${receipt.rnd_std:.2f} (Implied Vol: {receipt.rnd_implied_vol * 100:.2f}%)")
    print(f"    Risk-Neutral Skewness:        {receipt.rnd_skewness:+.3f}")
    print(f"    Risk-Neutral Kurtosis:        {receipt.rnd_kurtosis:.3f}")
    print("-" * 80)
    print("  Arrow-Debreu State Prices & Tail Crash Risk:")
    print(f"    Left Tail Mass (< -10%):      {receipt.left_tail_prob_10pct * 100:.2f}%")
    print(f"    Right Tail Mass (> +10%):     {receipt.right_tail_prob_10pct * 100:.2f}%")
    print(f"    Peak State Price p(K*):       ${receipt.peak_state_price:.5f} at Strike ${receipt.peak_state_strike:.2f}")
    print(f"    Raw Mass / Integral Check:    {receipt.raw_probability_mass:.4f} / {receipt.normalized_integral:.4f}")
    print(f"    Butterfly Arbitrage Check:    {'DETECTED (' + str(receipt.butterfly_violation_count) + ' violations)' if receipt.butterfly_arbitrage_detected else 'CLEAN (Convex)'}")
    print("-" * 80)
    print(f"  Breeden-Litzenberger Regime:    {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def aghion_bolton_command(args: argparse.Namespace) -> None:
    """CLI handler for Aghion & Bolton (1992) / Hart & Moore (1994) Control Rights Engine."""
    import json
    from src.analytics.aghion_bolton_control_rights import (
        calculate_aghion_bolton_equilibrium,
        evaluate_aghion_bolton_profile,
        persist_aghion_bolton_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    cash_flow = getattr(args, "cash_flow", None)
    debt = getattr(args, "debt", None)
    liquidation = getattr(args, "liquidation", None)
    private_benefit = getattr(args, "private_benefit", None)
    creditor_power = getattr(args, "creditor_power", None)

    pb_val = float(private_benefit) if private_benefit is not None else 15.0
    cp_val = float(creditor_power) if creditor_power is not None else 0.50

    try:
        if cash_flow is not None and debt is not None:
            liq_val = float(liquidation) if liquidation is not None else 50.0
            receipt = calculate_aghion_bolton_equilibrium(
                cash_flow=float(cash_flow),
                debt_face_value=float(debt),
                liquidation_value=liq_val,
                private_benefit=pb_val,
                creditor_bargaining_power=cp_val,
                ticker=ticker,
            )
        else:
            receipt = evaluate_aghion_bolton_profile(
                prices=[],
                ticker=ticker,
                creditor_power=cp_val,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_aghion_bolton_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Aghion-Bolton evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Aghion & Bolton (1992) / Hart & Moore (1994) Control Rights Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Realized Cash Flow (y):         ${receipt.realized_cash_flow:.2f}")
    print(f"  Contractual Debt Face Value (D):${receipt.debt_face_value:.2f}")
    print(f"  Liquidation Threat Value (L):   ${receipt.liquidation_value:.2f} | Private Benefit (B): ${receipt.private_benefit:.2f}")
    print(f"  Creditor Bargaining Power (a):  {receipt.creditor_bargaining_power * 100:.1f}%")
    print("-" * 80)
    print("  State-Contingent Control & Debt Renegotiation:")
    print(f"    Residual Control Holder:      {receipt.contractual_control_holder}")
    print(f"    Strategic Default Triggered:  {'YES (HARTS-MOORE RENEGOTIATION)' if receipt.strategic_default_triggered else 'NO'}")
    print(f"    Efficient Liquidation:        {'YES (FORECLOSURE)' if receipt.liquidation_triggered else 'NO (CONTINUATION)'}")
    print(f"    Renegotiated Debt Service D*: ${receipt.renegotiated_debt_service:.2f} (Haircut: {receipt.debt_haircut_pct * 100:.2f}%)")
    print("-" * 80)
    print("  Surplus Allocation:")
    print(f"    Creditor Recovery Payoff:     ${receipt.creditor_payoff:.2f}")
    print(f"    Entrepreneur / Equity Payoff: ${receipt.entrepreneur_payoff:.2f}")
    print(f"    Total Realized Surplus:       ${receipt.total_surplus_realized:.2f} (First-Best: ${receipt.first_best_surplus:.2f})")
    print("-" * 80)
    print(f"  Governance Regime:              {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def leland_command(args: argparse.Namespace) -> None:
    """CLI handler for Hayne E. Leland (1994) Endogenous Default & Optimal Capital Structure Engine."""
    import json
    from src.analytics.leland_endogenous_default import (
        calculate_leland_equilibrium,
        evaluate_leland_profile,
        persist_leland_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    asset_value = getattr(args, "asset_value", None)
    vol = float(getattr(args, "volatility", 0.25) if getattr(args, "volatility", None) is not None else 0.25)
    rate = float(getattr(args, "rate", 0.05) if getattr(args, "rate", None) is not None else 0.05)
    tax_rate = float(getattr(args, "tax_rate", 0.25) if getattr(args, "tax_rate", None) is not None else 0.25)
    b_cost = float(getattr(args, "bankruptcy_cost", 0.35) if getattr(args, "bankruptcy_cost", None) is not None else 0.35)
    coupon = getattr(args, "coupon", None)

    try:
        if asset_value is not None:
            coupon_val = float(coupon) if coupon is not None else 4.0
            receipt = calculate_leland_equilibrium(
                asset_value=float(asset_value),
                asset_volatility=vol,
                risk_free_rate=rate,
                corporate_tax_rate=tax_rate,
                bankruptcy_cost_alpha=b_cost,
                coupon=coupon_val,
                ticker=ticker,
            )
        else:
            receipt = evaluate_leland_profile(
                prices=[],
                ticker=ticker,
                corporate_tax_rate=tax_rate,
                bankruptcy_cost_alpha=b_cost,
                risk_free_rate=rate,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_leland_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Leland evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Leland (1994) Endogenous Default & Optimal Capital Structure Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Unlevered Asset Value (V):      ${receipt.asset_value:.2f} (sigma: {receipt.asset_volatility * 100:.1f}%, r: {receipt.risk_free_rate * 100:.2f}%)")
    print(f"  Tax Rate (tau) / Bankr Cost (a):{receipt.corporate_tax_rate * 100:.1f}% | {receipt.bankruptcy_cost_alpha * 100:.1f}%")
    print(f"  Perpetual Coupon Rate (C):      ${receipt.contractual_coupon:.2f}")
    print("-" * 80)
    print("  Endogenous Default & Structural Claims:")
    print(f"    Endogenous Default Barrier V_B:${receipt.endogenous_default_barrier:.2f} ({'DEFAULT TRIGGERED' if receipt.default_triggered else 'SOLVENT'})")
    print(f"    Default Claim PV (V/V_B)^(-X):{receipt.default_probability_pv:.4f}")
    print(f"    Risky Debt Value D(V):        ${receipt.debt_value:.2f} (Credit Spread: {receipt.credit_spread_bps:.1f} bps)")
    print(f"    Equity Value E(V):            ${receipt.equity_value:.2f}")
    print(f"    Tax Shield / Bankruptcy Loss: +${receipt.tax_shield_value:.2f} / -${receipt.bankruptcy_cost_value:.2f}")
    print("-" * 80)
    print("  Optimal Capital Structure Tradeoff:")
    print(f"    Levered Firm Value v(V):      ${receipt.levered_firm_value:.2f} (Leverage: {receipt.leverage_ratio * 100:.2f}%)")
    print(f"    Optimal Coupon (C*):          ${receipt.optimal_coupon:.2f} (Optimal Firm Value v*: ${receipt.optimal_firm_value:.2f})")
    print(f"    Optimal Target Leverage (L*): {receipt.optimal_leverage_ratio * 100:.2f}%")
    print("-" * 80)
    print(f"  Leland Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def hansen_jagannathan_command(args: argparse.Namespace) -> None:
    """CLI handler for Hansen & Jagannathan (1991, 1997) SDF Volatility Bounds Engine."""
    import json
    from src.analytics.hansen_jagannathan_bounds import (
        evaluate_hansen_jagannathan_profile,
        persist_hansen_jagannathan_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    risk_aversion = float(getattr(args, "risk_aversion", 5.0) if getattr(args, "risk_aversion", None) is not None else 5.0)
    beta = float(getattr(args, "beta", 0.99) if getattr(args, "beta", None) is not None else 0.99)
    rf = float(getattr(args, "rate", 0.04) if getattr(args, "rate", None) is not None else 0.04)

    try:
        receipt = evaluate_hansen_jagannathan_profile(
            prices=[],
            ticker=ticker,
            risk_aversion=risk_aversion,
            discount_beta=beta,
            risk_free_annual=rf,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_hansen_jagannathan_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Hansen-Jagannathan evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hansen & Jagannathan (1991, 1997) SDF Volatility Bounds Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Test Assets / Sessions:         {receipt.num_assets} basis assets | {receipt.num_observations} observations")
    print(f"  Max Tangency Sharpe Ratio (Ann):{receipt.max_sharpe_annualized:.3f} (Period: {receipt.max_sharpe_period:.4f})")
    print("-" * 80)
    print("  Stochastic Discount Factor (SDF) Admissibility:")
    print(f"    Candidate SDF Mean E[m]:      {receipt.candidate_sdf_mean:.6f}")
    print(f"    Candidate SDF Std sigma(m):   {receipt.candidate_sdf_std:.6f}")
    print(f"    HJ Volatility Lower Bound:    {receipt.hj_volatility_bound:.6f} (sigma(m)/E[m]: {receipt.hj_dispersion_ratio:.4f})")
    print(f"    HJ Bound Satisfied:           {'PASS (ADMISSIBLE)' if receipt.hj_bound_satisfied else 'FAIL (EQUITY PREMIUM PUZZLE)'}")
    print("-" * 80)
    print("  Pricing Kernel Misspecification (HJ-Distance):")
    print(f"    HJ-Distance (delta_HJ):       {receipt.hj_distance:.6f}")
    print(f"    Mean Abs Pricing Error:       {receipt.mean_abs_pricing_error:.6f}")
    print(f"    Implied Min CRRA Gamma:       {receipt.implied_min_risk_aversion:.2f}")
    print("-" * 80)
    print(f"  Hansen-Jagannathan Regime:      {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def jensen_meckling_command(args: argparse.Namespace) -> None:
    """CLI handler for Jensen & Meckling (1976) Agency Costs & Asset Substitution Engine."""
    import json
    from src.analytics.jensen_meckling_agency import (
        calculate_jensen_meckling_equilibrium,
        evaluate_jensen_meckling_profile,
        persist_jensen_meckling_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    firm_value = getattr(args, "firm_value", None)
    alpha = float(getattr(args, "alpha", 0.40) if getattr(args, "alpha", None) is not None else 0.40)
    debt = getattr(args, "debt", None)
    safe_vol = float(getattr(args, "safe_vol", 0.20) if getattr(args, "safe_vol", None) is not None else 0.20)
    risky_vol = float(getattr(args, "risky_vol", 0.55) if getattr(args, "risky_vol", None) is not None else 0.55)
    npv_disc = float(getattr(args, "npv_discount", 0.06) if getattr(args, "npv_discount", None) is not None else 0.06)

    try:
        if firm_value is not None or debt is not None:
            v0 = float(firm_value) if firm_value is not None else 100.0
            d_val = float(debt) if debt is not None else 75.0
            receipt = calculate_jensen_meckling_equilibrium(
                firm_potential_value=v0,
                insider_ownership_alpha=alpha,
                debt_face_value=d_val,
                safe_volatility=safe_vol,
                risky_volatility=risky_vol,
                risky_npv_discount_pct=npv_disc,
                ticker=ticker,
            )
        else:
            receipt = evaluate_jensen_meckling_profile(
                prices=[],
                ticker=ticker,
                insider_alpha=alpha,
                risky_volatility=risky_vol,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_jensen_meckling_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Jensen-Meckling evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Jensen & Meckling (1976) Agency Costs & Asset Substitution Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Firm Value Potential (V_0):     ${receipt.firm_potential_value:.2f} | Debt Face (D): ${receipt.debt_face_value:.2f}")
    print(f"  Insider Equity Stake (alpha):   {receipt.insider_ownership_alpha * 100:.1f}%")
    print(f"  Safe vs Risky Volatility:       {receipt.safe_volatility * 100:.1f}% vs {receipt.risky_volatility * 100:.1f}%")
    print("-" * 80)
    print("  Agency Costs of Outside Equity (Perk Consumption):")
    print(f"    First-Best Perks F*(1):       ${receipt.optimal_perk_consumption:.2f}")
    print(f"    Induced Perks F*(alpha):      ${receipt.actual_perk_consumption:.2f}")
    print(f"    Deadweight Equity Agency Loss:${receipt.equity_agency_loss:.2f}")
    print("-" * 80)
    print("  Agency Costs of Debt (Asset Substitution / Risk-Shifting):")
    print(f"    Safe Equity vs Risky Equity:  ${receipt.safe_equity_value:.2f} vs ${receipt.risky_equity_value:.2f}")
    print(f"    Risk-Shifting Triggered:      {'YES (CREDITOR EXPROPRIATION)' if receipt.risk_shifting_triggered else 'NO (PRUDENT OPERATING CHOICE)'}")
    print(f"    Creditor Wealth Expropriated: ${receipt.creditor_wealth_expropriation:.2f}")
    print(f"    Asset Substitution Loss:      ${receipt.asset_substitution_loss:.2f}")
    print(f"    Total Agency Cost:            ${receipt.total_agency_cost:.2f} ({receipt.agency_cost_pct_of_value * 100:.2f}% of V_0)")
    print("-" * 80)
    print(f"  Jensen-Meckling Regime:         {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def heston_vol_command(args: argparse.Namespace) -> None:
    """CLI handler for Steven L. Heston (1993) Stochastic Volatility Option Pricing Engine."""
    import json
    from src.analytics.heston_stochastic_vol import (
        calculate_heston_pricing,
        evaluate_heston_profile,
        persist_heston_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    spot = getattr(args, "spot", None)
    strike = getattr(args, "strike", None)
    xi = float(getattr(args, "xi", 0.35) if getattr(args, "xi", None) is not None else 0.35)
    rho = float(getattr(args, "rho", -0.65) if getattr(args, "rho", None) is not None else -0.65)
    kappa = float(getattr(args, "kappa", 2.0) if getattr(args, "kappa", None) is not None else 2.0)
    theta = float(getattr(args, "theta", 0.04) if getattr(args, "theta", None) is not None else 0.04)
    v0 = float(getattr(args, "v0", 0.04) if getattr(args, "v0", None) is not None else 0.04)
    rate = float(getattr(args, "rate", 0.04) if getattr(args, "rate", None) is not None else 0.04)
    expiry = float(getattr(args, "expiry", 0.50) if getattr(args, "expiry", None) is not None else 0.50)

    try:
        if spot is not None:
            s0 = float(spot)
            k_val = float(strike) if strike is not None else s0
            receipt = calculate_heston_pricing(
                spot_price=s0,
                strike_price=k_val,
                risk_free_rate=rate,
                time_to_expiry=expiry,
                initial_variance=v0,
                long_run_variance=theta,
                mean_reversion_kappa=kappa,
                vol_of_vol_xi=xi,
                spot_vol_corr_rho=rho,
                ticker=ticker,
            )
        else:
            receipt = evaluate_heston_profile(
                prices=[],
                ticker=ticker,
                vol_of_vol_xi=xi,
                spot_vol_corr_rho=rho,
                time_to_expiry=expiry,
                risk_free_rate=rate,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_heston_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Heston stochastic vol evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Heston (1993) Stochastic Volatility Option Pricing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Spot / Strike / Expiry:         ${receipt.spot_price:.2f} / ${receipt.strike_price:.2f} / {receipt.time_to_expiry:.2f} yrs")
    print(f"  Variance Dynamics (v0, th, kap):{receipt.initial_variance:.4f} | {receipt.long_run_variance:.4f} | {receipt.mean_reversion_kappa:.2f}")
    print(f"  Vol-of-Vol (xi) / Corr (rho):   {receipt.vol_of_vol_xi:.2f} | {receipt.spot_vol_corr_rho:+.2f}")
    print("-" * 80)
    print("  Semi-Closed-Form Option Valuations & Surface:")
    print(f"    Heston Call / Put Price:      ${receipt.heston_call_price:.4f} / ${receipt.heston_put_price:.4f}")
    print(f"    Black-Scholes Benchmark Call: ${receipt.bs_benchmark_call_price:.4f}")
    print(f"    Implied Vol (85%/ATM/115%):   {receipt.otm_put_implied_vol * 100:.2f}% | {receipt.atm_implied_vol * 100:.2f}% | {receipt.otm_call_implied_vol * 100:.2f}%")
    print(f"    Put-Call Skew Spread:         {receipt.volatility_skew_spread * 100:+.2f}% (Wings: {receipt.butterfly_wing_spread * 100:+.2f}%)")
    print(f"    Feller Stability (2k*th>xi2): {receipt.feller_ratio:.2f} ({'SATISFIED' if receipt.feller_condition_satisfied else 'VIOLATED'})")
    print("-" * 80)
    print(f"  Heston Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def myers_overhang_command(args: argparse.Namespace) -> None:
    """CLI handler for Stewart C. Myers (1977) Debt Overhang & Real Growth Option Engine."""
    import json
    from src.analytics.myers_debt_overhang import (
        calculate_myers_debt_overhang,
        evaluate_myers_overhang_profile,
        persist_myers_overhang_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    assets = getattr(args, "assets", None)
    growth_pv = float(getattr(args, "growth_pv", 60.0) if getattr(args, "growth_pv", None) is not None else 60.0)
    outlay = float(getattr(args, "outlay", 40.0) if getattr(args, "outlay", None) is not None else 40.0)
    debt = getattr(args, "debt", None)

    try:
        if assets is not None or debt is not None:
            a0 = float(assets) if assets is not None else 100.0
            d_val = float(debt) if debt is not None else 95.0
            receipt = calculate_myers_debt_overhang(
                assets_in_place_pv=a0,
                growth_option_gross_pv=growth_pv,
                investment_outlay=outlay,
                debt_face_value=d_val,
                ticker=ticker,
            )
        else:
            receipt = evaluate_myers_overhang_profile(
                prices=[],
                ticker=ticker,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_myers_overhang_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Myers (1977) Debt Overhang evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Myers (1977) Debt Overhang & Real Growth Option Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Assets in Place (A_0):          ${receipt.assets_in_place_pv:.2f} | Debt Face (D): ${receipt.debt_face_value:.2f}")
    print(f"  Growth Option Gross PV (G_0):   ${receipt.growth_option_gross_pv:.2f} | Capex Outlay (I): ${receipt.investment_outlay:.2f}")
    print("-" * 80)
    print("  Growth Option Forfeiture & Underinvestment:")
    print(f"    First-Best Growth Value:      ${receipt.first_best_growth_value:.2f}")
    print(f"    Levered Growth Captured:      ${receipt.levered_growth_captured:.2f}")
    print(f"    Overhang Agency Loss:         ${receipt.overhang_agency_loss:.2f} ({receipt.overhang_severity_pct * 100:.1f}% destroyed)")
    print(f"    Overhang State Probability:   {receipt.overhang_probability * 100:.1f}% ({'TRIGGERED' if receipt.debt_overhang_triggered else 'OVERHANG-FREE'})")
    print("-" * 80)
    print("  Enterprise Value Conservation:")
    print(f"    Levered Equity Value E_0:     ${receipt.levered_equity_value:.2f}")
    print(f"    Levered Debt Value D_0:       ${receipt.levered_debt_value:.2f}")
    print(f"    Total Levered Firm Value:     ${receipt.levered_firm_value:.2f} (First-Best: ${receipt.first_best_firm_value:.2f})")
    print("-" * 80)
    print(f"  Debt Overhang Regime:           {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def holmstrom_tirole_command(args: argparse.Namespace) -> None:
    """CLI handler for Holmstrom & Tirole (1997) Pledgeable Income & Credit Rationing Engine."""
    import json
    from src.analytics.holmstrom_tirole_pledgeable import (
        calculate_holmstrom_tirole_equilibrium,
        evaluate_holmstrom_tirole_profile,
        persist_holmstrom_tirole_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    outlay = getattr(args, "outlay", None)
    net_worth = getattr(args, "net_worth", None)
    payoff = float(getattr(args, "payoff", 160.0) if getattr(args, "payoff", None) is not None else 160.0)
    b_priv = float(getattr(args, "private_benefit", 28.0) if getattr(args, "private_benefit", None) is not None else 28.0)

    try:
        if outlay is not None or net_worth is not None:
            i_val = float(outlay) if outlay is not None else 100.0
            a_val = float(net_worth) if net_worth is not None else 35.0
            receipt = calculate_holmstrom_tirole_equilibrium(
                investment_outlay=i_val,
                corporate_net_worth=a_val,
                success_payoff=payoff,
                private_benefit_unmonitored=b_priv,
                ticker=ticker,
            )
        else:
            receipt = evaluate_holmstrom_tirole_profile(
                prices=[],
                ticker=ticker,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_holmstrom_tirole_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Holmstrom-Tirole evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Holmstrom & Tirole (1997) Pledgeable Income & Credit Rationing Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Investment Outlay (I) / Cash (A):${receipt.investment_outlay:.2f} | ${receipt.corporate_net_worth:.2f}")
    print(f"  Success Payoff (R) / NPV:       ${receipt.success_payoff:.2f} | ${receipt.expected_project_npv:+.2f}")
    print(f"  Private Benefit (B vs b):       ${receipt.private_benefit_unmonitored:.2f} vs ${receipt.private_benefit_monitored:.2f} (Cost c=${receipt.monitoring_cost:.2f})")
    print("-" * 80)
    print("  Moral Hazard & Pledgeable Income Thresholds:")
    print(f"    Direct Pledgeable Income:     ${receipt.direct_pledgeable_income:.2f} ({receipt.pledgeable_share_of_payoff * 100:.1f}% pledgeable)")
    print(f"    Monitored Pledgeable Income:  ${receipt.monitored_pledgeable_income:.2f}")
    print(f"    Min Net Worth (Direct Market):${receipt.min_net_worth_direct:.2f}")
    print(f"    Min Net Worth (Bank Monitored):${receipt.min_net_worth_bank:.2f}")
    print("-" * 80)
    print("  Capital Access & Credit Crunch Diagnostics:")
    print(f"    Financing Tier:               {receipt.financing_tier}")
    print(f"    Credit Rationed:              {'YES (FUNDING SHORTFALL: $' + f'{receipt.funding_gap:.2f})' if receipt.credit_rationed else 'NO (FUNDABLE)'}")
    print(f"    Balance-Sheet Credit Mult:    {receipt.credit_multiplier:.2f}x")
    print("-" * 80)
    print(f"  Holmstrom-Tirole Regime:        {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def grossman_hart_command(args: argparse.Namespace) -> None:
    """CLI handler for Grossman & Hart (1980) Takeover Bids & Free-Rider Engine."""
    import json
    from src.analytics.grossman_hart_takeover import (
        calculate_grossman_hart_takeover,
        evaluate_grossman_hart_profile,
        persist_grossman_hart_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    status_quo = getattr(args, "status_quo", None)
    synergy = float(getattr(args, "synergy", 30.0) if getattr(args, "synergy", None) is not None else 30.0)
    cost = float(getattr(args, "bidding_cost", 10.0) if getattr(args, "bidding_cost", None) is not None else 10.0)
    dilution = float(getattr(args, "dilution", 14.0) if getattr(args, "dilution", None) is not None else 14.0)
    toehold = float(getattr(args, "toehold", 0.05) if getattr(args, "toehold", None) is not None else 0.05)

    try:
        if status_quo is not None:
            receipt = calculate_grossman_hart_takeover(
                status_quo_value=float(status_quo),
                restructuring_synergy=synergy,
                bidding_cost=cost,
                dilution_cap_phi=dilution,
                pre_bid_toehold_alpha=toehold,
                ticker=ticker,
            )
        else:
            receipt = evaluate_grossman_hart_profile(
                prices=[],
                ticker=ticker,
                dilution_pct=dilution / 100.0 if dilution > 1.0 else dilution,
                toehold_alpha=toehold,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_grossman_hart_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Grossman-Hart takeover evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Grossman & Hart (1980) Takeover Bids & Free-Rider Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Status Quo (q) / Post-Bid (v):  ${receipt.status_quo_value:.2f} -> ${receipt.post_takeover_value:.2f} (+${receipt.restructuring_synergy:.2f} synergy)")
    print(f"  Dilution Cap (phi) / Toehold:   ${receipt.dilution_cap_phi:.2f} | {receipt.pre_bid_toehold_alpha * 100:.1f}%")
    print(f"  Bidding Execution Cost (c):     ${receipt.bidding_cost:.2f}")
    print("-" * 80)
    print("  Free-Rider Tender Equilibrium:")
    print(f"    Free-Rider Equilibrium Bid:   ${receipt.equilibrium_tender_price:.2f} (Premium: +${receipt.target_bid_premium:.2f}, +{receipt.target_bid_premium_pct * 100:.1f}%)")
    print(f"    Bidder Net Profit:            ${receipt.bidder_net_profit:+.2f} ({'TAKEOVER SUCCEEDS' if receipt.takeover_succeeds else 'FAILED (FREE-RIDER PARALYSIS)'})")
    print(f"    Optimal Min Dilution Floor:   ${receipt.optimal_min_dilution:.2f} (0-Dilution Toehold: {receipt.breakeven_toehold_zero_phi * 100:.1f}%)")
    print(f"    Missed Social Synergy Loss:   ${receipt.missed_synergy_loss:.2f}")
    print("-" * 80)
    print(f"  Takeover Control Regime:        {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def glosten_milgrom_command(args: argparse.Namespace) -> None:
    """CLI handler for Glosten & Milgrom (1985) Sequential Trade Adverse Selection Spread Engine."""
    import json
    from src.analytics.glosten_milgrom_spread import (
        calculate_glosten_milgrom_spread,
        evaluate_glosten_milgrom_profile,
        persist_glosten_milgrom_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    v_low = getattr(args, "value_low", None)
    v_high = getattr(args, "value_high", None)
    pi = float(getattr(args, "pi", 0.25) if getattr(args, "pi", None) is not None else 0.25)
    prior = float(getattr(args, "prior", 0.50) if getattr(args, "prior", None) is not None else 0.50)

    try:
        if v_low is not None and v_high is not None:
            receipt = calculate_glosten_milgrom_spread(
                value_low=float(v_low),
                value_high=float(v_high),
                informed_fraction_pi=pi,
                prior_prob_high=prior,
                order_flow=[1, 1, -1],
                ticker=ticker,
            )
        else:
            receipt = evaluate_glosten_milgrom_profile(
                prices=[],
                ticker=ticker,
                informed_fraction_pi=pi,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_glosten_milgrom_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Glosten-Milgrom evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Glosten & Milgrom (1985) Adverse Selection Spread Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  True States (V_L / V_H):        ${receipt.value_low:.2f} / ${receipt.value_high:.2f} (Band: ${receipt.value_uncertainty_band:.2f})")
    print(f"  Informed Flow Fraction (pi):    {receipt.informed_fraction_pi * 100:.1f}% | Prior High Prob: {receipt.prior_prob_high * 100:.1f}%")
    print("-" * 80)
    print("  Specialist Quote Setting & Bayesian Discovery:")
    print(f"    Initial Bid / Ask Quote:      ${receipt.initial_bid:.2f} / ${receipt.initial_ask:.2f} (Mid: ${receipt.initial_midprice:.2f})")
    print(f"    Initial Adverse Spread:       ${receipt.initial_spread:.2f} ({receipt.initial_spread_bps:.1f} bps)")
    print(f"    Posterior High Prob ({receipt.order_flow_steps} trades):{receipt.posterior_prob_high * 100:.1f}%")
    print(f"    Final Bid / Ask Quote:        ${receipt.final_bid:.2f} / ${receipt.final_ask:.2f} (Spread: ${receipt.final_spread:.2f})")
    print(f"    Spread Narrowing / Discovery: {receipt.spread_narrowing_pct * 100:.1f}% | Std Reduction: ${receipt.price_discovery_gain:.2f}")
    print("-" * 80)
    print(f"  Microstructure Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def leland_pyle_command(args: argparse.Namespace) -> None:
    """CLI handler for Leland & Pyle (1977) Insider Ownership Signaling Engine."""
    import json
    from src.analytics.leland_pyle_signaling import (
        calculate_leland_pyle_equilibrium,
        evaluate_leland_pyle_profile,
        persist_leland_pyle_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    floor_val = getattr(args, "floor_value", None)
    alpha = float(getattr(args, "alpha", 0.55) if getattr(args, "alpha", None) is not None else 0.55)
    vol = float(getattr(args, "volatility", 0.40) if getattr(args, "volatility", None) is not None else 0.40)
    gamma = float(getattr(args, "gamma", 3.0) if getattr(args, "gamma", None) is not None else 3.0)
    true_val = getattr(args, "true_value", None)

    try:
        if floor_val is not None:
            fv = float(floor_val)
            tv = float(true_val) if true_val is not None else fv * 1.15
            receipt = calculate_leland_pyle_equilibrium(
                floor_value_mu_min=fv,
                true_project_value=tv,
                annual_volatility=vol,
                relative_risk_aversion=gamma,
                insider_retention_alpha=alpha,
                ticker=ticker,
            )
        else:
            receipt = evaluate_leland_pyle_profile(
                prices=[],
                ticker=ticker,
                insider_retention_alpha=alpha,
                relative_risk_aversion=gamma,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_leland_pyle_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Leland-Pyle signaling evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Leland & Pyle (1977) Insider Ownership Signaling Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Lemon Floor (mu_min) / True Val:${receipt.floor_value_mu_min:.2f} / ${receipt.true_project_value:.2f}")
    print(f"  Insider Equity Stake (alpha):   {receipt.insider_retention_alpha * 100:.1f}% (Volatility: {receipt.annual_volatility * 100:.1f}%)")
    print("-" * 80)
    print("  Separating Signaling Equilibrium:")
    print(f"    Signaled Market Valuation:    ${receipt.signaled_market_value:.2f} (+${receipt.signaling_premium_dollar:.2f}, +{receipt.signaling_premium_pct * 100:.1f}%)")
    print(f"    Marginal Sensitivity dV/da:   ${receipt.marginal_value_sensitivity:.2f} (Scale K: ${receipt.signaling_scale_k:.2f})")
    print(f"    Diversification Risk Cost:    ${receipt.diversification_risk_cost:.2f} (Intermediary Threshold)")
    print(f"    Quality-Implied Alpha*:       {receipt.quality_implied_alpha * 100:.1f}% (Mispricing: ${receipt.market_mispricing_wedge:+.2f})")
    print("-" * 80)
    print(f"  Leland-Pyle Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def ross_signaling_command(args: argparse.Namespace) -> None:
    """CLI handler for Stephen A. Ross (1977) Incentive Debt-Signaling Engine."""
    import json
    from src.analytics.ross_debt_signaling import (
        calculate_ross_debt_signaling,
        evaluate_ross_signaling_profile,
        persist_ross_signaling_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    floor_t = getattr(args, "floor_t", None)
    true_t = getattr(args, "true_t", None)
    debt = getattr(args, "debt", None)
    penalty = float(getattr(args, "penalty", 60.0) if getattr(args, "penalty", None) is not None else 60.0)

    try:
        if floor_t is not None or debt is not None:
            t0 = float(floor_t) if floor_t is not None else 100.0
            tt = float(true_t) if true_t is not None else t0 * 1.60
            f_val = float(debt) if debt is not None else 65.0
            receipt = calculate_ross_debt_signaling(
                floor_quality_t_min=t0,
                true_quality_t=tt,
                debt_face_value=f_val,
                bankruptcy_penalty_l=penalty,
                ticker=ticker,
            )
        else:
            receipt = evaluate_ross_signaling_profile(
                prices=[],
                ticker=ticker,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_ross_signaling_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Ross debt-signaling evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Ross (1977) Incentive Debt-Signaling Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Quality Ceilings (t_min/t_true):{receipt.floor_quality_t_min:.2f} / {receipt.true_quality_t:.2f}")
    print(f"  Debt Signal (F) / Penalty (L):  ${receipt.debt_face_value:.2f} | ${receipt.bankruptcy_penalty_l:.2f} (kappa: {receipt.kappa_incentive_coeff:.2f})")
    print("-" * 80)
    print("  Separating Debt-Signaling Equilibrium:")
    print(f"    Signaled Market Valuation:    ${receipt.signaled_market_value:.2f} (+{receipt.signaling_valuation_lift_pct * 100:.1f}% above floor ${receipt.floor_market_value:.2f})")
    print(f"    Optimal Separating Debt F*:   ${receipt.optimal_separating_debt_f_star:.2f} (dV_0/dF: {receipt.marginal_value_dv_df:.4f})")
    print(f"    Default Prob (True/Signaled): {receipt.true_default_probability * 100:.1f}% / {receipt.signaled_default_probability * 100:.1f}%")
    print(f"    Mimicry Deterrence Gap:       ${receipt.mimicry_deterrence_gap:.2f}")
    print("-" * 80)
    print(f"  Ross Signaling Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def diamond_monitoring_command(args: argparse.Namespace) -> None:
    """CLI handler for Douglas W. Diamond (1984) Financial Intermediation & Delegated Monitoring Engine."""
    import json
    from src.analytics.diamond_delegated_monitoring import (
        calculate_diamond_delegated_monitoring,
        evaluate_diamond_monitoring_profile,
        persist_diamond_monitoring_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    borrowers = getattr(args, "borrowers", None)
    corr = float(getattr(args, "correlation", 0.05) if getattr(args, "correlation", None) is not None else 0.05)
    depositors = int(getattr(args, "depositors", 10) if getattr(args, "depositors", None) is not None else 10)
    k_cost = float(getattr(args, "monitoring_cost", 2.0) if getattr(args, "monitoring_cost", None) is not None else 2.0)

    try:
        if borrowers is not None:
            receipt = calculate_diamond_delegated_monitoring(
                num_borrowers_n=int(borrowers),
                depositors_per_firm_m=depositors,
                monitoring_cost_k=k_cost,
                loan_correlation_rho=corr,
                ticker=ticker,
            )
        else:
            receipt = evaluate_diamond_monitoring_profile(
                prices=[],
                ticker=ticker,
                loan_correlation_rho=corr,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_diamond_monitoring_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Diamond delegated monitoring evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Diamond (1984) Financial Intermediation & Delegated Monitoring Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Pooled Borrowers (N) / Lenders: {receipt.num_borrowers_n} firms | {receipt.depositors_per_firm_m} depositors/firm")
    print(f"  Monitoring Cost (K) / Loan Corr:{receipt.monitoring_cost_k:.2f} | rho={receipt.loan_correlation_rho:.2f} (Div Eff: {receipt.diversification_efficiency_pct * 100:.1f}%)")
    print("-" * 80)
    print("  Delegation vs Direct Lending Benchmarks:")
    print(f"    Duplicative Direct Cost (mK): ${receipt.direct_duplicative_cost:.2f}")
    print(f"    Unmonitored Debt Loss (D_dir):${receipt.direct_unmonitored_cost:.2f}")
    print(f"    Delegation Cost d_N:          ${receipt.delegation_cost_d_n:.2f} (Promised Deposit H*: ${receipt.optimal_deposit_face_value:.2f})")
    print(f"    Delegated Bank Total (K+d_N): ${receipt.delegated_total_cost:.2f}")
    print(f"    Net Intermediation Surplus:   ${receipt.net_intermediation_surplus:+.2f} ({'VIABLE' if receipt.delegated_intermediation_viable else 'UNVIABLE'})")
    print("-" * 80)
    print(f"  Intermediation Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def miller_tax_command(args: argparse.Namespace) -> None:
    """CLI handler for Merton H. Miller (1977) Debt and Taxes Personal Clientele Engine."""
    import json
    from src.analytics.miller_debt_taxes import (
        calculate_miller_debt_taxes,
        evaluate_miller_tax_profile,
        persist_miller_tax_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    firm_val = getattr(args, "firm_value", None)
    debt = getattr(args, "debt", None)
    corp_tax = float(getattr(args, "corp_tax", 0.25) if getattr(args, "corp_tax", None) is not None else 0.25)
    eq_tax = float(getattr(args, "equity_tax", 0.12) if getattr(args, "equity_tax", None) is not None else 0.12)
    debt_tax = float(getattr(args, "debt_tax", 0.30) if getattr(args, "debt_tax", None) is not None else 0.30)

    try:
        if firm_val is not None or debt is not None:
            vu = float(firm_val) if firm_val is not None else 100.0
            bd = float(debt) if debt is not None else 40.0
            receipt = calculate_miller_debt_taxes(
                unlevered_firm_value=vu,
                debt_market_value=bd,
                corporate_tax_rate=corp_tax,
                personal_equity_tax_rate=eq_tax,
                personal_debt_tax_rate=debt_tax,
                ticker=ticker,
            )
        else:
            receipt = evaluate_miller_tax_profile(
                prices=[],
                ticker=ticker,
                corporate_tax_rate=corp_tax,
                personal_equity_tax_rate=eq_tax,
                personal_debt_tax_rate=debt_tax,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_miller_tax_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Miller Debt & Taxes evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Miller (1977) Debt & Taxes Personal Clientele Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Unlevered Value (V_U) / Debt:   ${receipt.unlevered_firm_value:.2f} | ${receipt.debt_market_value:.2f}")
    print(f"  Tax Rates (tau_c / ps / pd):    {receipt.corporate_tax_rate * 100:.1f}% | {receipt.personal_equity_tax_rate * 100:.1f}% | {receipt.personal_debt_tax_rate * 100:.1f}%")
    print("-" * 80)
    print("  Miller Personal Clientele Tax Arbitrage:")
    print(f"    Miller Leverage Gain Factor:  {receipt.miller_gain_factor * 100:+.2f}%")
    print(f"    Net Miller Tax Gain (G_L):    ${receipt.miller_leverage_gain:+.2f} (MM63 Shield: ${receipt.mm63_corporate_shield:.2f})")
    print(f"    Personal Tax Erosion:         ${receipt.personal_tax_erosion:+.2f}")
    print(f"    Break-Even Bond Bracket t_pd*:{receipt.breakeven_debt_tax_rate * 100:.2f}% (Clientele: {receipt.investor_clientele})")
    print(f"    Market Clearing Bond Yield:   {receipt.market_clearing_bond_yield * 100:.2f}% (+{receipt.tax_clientele_spread_bps:.1f} bps vs tax-free)")
    print("-" * 80)
    print(f"  Miller Tax Regime:              {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def campbell_cochrane_command(args: argparse.Namespace) -> None:
    """CLI handler for Campbell & Cochrane (1999) External Habit Formation Engine."""
    import json
    from src.analytics.campbell_cochrane_habit import (
        calculate_campbell_cochrane_habit,
        evaluate_campbell_cochrane_profile,
        persist_campbell_cochrane_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    gamma = float(getattr(args, "gamma", 2.0) if getattr(args, "gamma", None) is not None else 2.0)
    phi = float(getattr(args, "phi", 0.87) if getattr(args, "phi", None) is not None else 0.87)
    surplus = getattr(args, "surplus_ratio", None)

    try:
        if surplus is not None or getattr(args, "gamma", None) is not None:
            receipt = calculate_campbell_cochrane_habit(
                utility_curvature_gamma=gamma,
                habit_persistence_phi=phi,
                current_surplus_ratio=float(surplus) if surplus is not None else None,
                ticker=ticker,
            )
        else:
            receipt = evaluate_campbell_cochrane_profile(
                prices=[],
                ticker=ticker,
                utility_curvature_gamma=gamma,
            )

        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_campbell_cochrane_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Campbell-Cochrane evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Campbell & Cochrane (1999) External Habit Formation Engine")
    print("=" * 80)
    print(f"  Security / Ticker:              {receipt.ticker}")
    print(f"  Utility Curvature (gamma) / phi:{receipt.utility_curvature_gamma:.2f} | {receipt.habit_persistence_phi:.2f} (Half-Life: {receipt.habit_half_life_years:.2f} yrs)")
    print(f"  Surplus Ratio S_t (vs S_bar):   {receipt.current_surplus_ratio * 100:.2f}% (Steady: {receipt.steady_state_surplus_s_bar * 100:.2f}%, Max: {receipt.max_surplus_ratio_s_max * 100:.2f}%)")
    print("-" * 80)
    print("  Counter-Cyclical Risk Aversion & Asset Pricing:")
    print(f"    Local Relative Risk Aversion: {receipt.local_risk_aversion_rra:.2f}x (lambda: {receipt.sensitivity_lambda:.3f})")
    print(f"    Conditional Max Sharpe (SR):  {receipt.conditional_max_sharpe:.4f}")
    print(f"    Expected Equity Premium:      {receipt.expected_equity_premium * 100:.2f}% / yr")
    print(f"    Constant Real Risk-Free Rate: {receipt.implied_risk_free_rate * 100:.2f}% / yr")
    print("-" * 80)
    print(f"  Habit Cycle Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def term_structure_command(args: argparse.Namespace) -> None:
    """CLI handler for CIR (1985) & Vasicek (1977) Affine Term Structure Engine."""
    import json
    from src.analytics.cir_vasicek_term_structure import (
        calculate_affine_term_structure,
        persist_term_structure_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    r0 = float(getattr(args, "r0", 0.035) if getattr(args, "r0", None) is not None else 0.035)
    theta = float(getattr(args, "theta", 0.050) if getattr(args, "theta", None) is not None else 0.050)
    kappa = float(getattr(args, "kappa", 0.35) if getattr(args, "kappa", None) is not None else 0.35)
    sigma = float(getattr(args, "sigma", 0.04) if getattr(args, "sigma", None) is not None else 0.04)
    model_type = str(getattr(args, "model", "CIR") or "CIR").strip().upper()

    try:
        receipt = calculate_affine_term_structure(
            short_rate_r0=r0,
            long_run_mean_theta=theta,
            mean_reversion_kappa=kappa,
            rate_volatility_sigma=sigma,
            model_type=model_type,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_term_structure_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Term structure evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: CIR (1985) & Vasicek (1977) Affine Term Structure Engine")
    print("=" * 80)
    print(f"  Instrument / Ticker:            {receipt.ticker} ({receipt.model_type})")
    print(f"  Short Rate r0 / Long Mean theta:{receipt.short_rate_r0 * 100:.2f}% | {receipt.long_run_mean_theta * 100:.2f}% (kappa={receipt.mean_reversion_kappa:.2f}, sigma={receipt.rate_volatility_sigma * 100:.2f}%)")
    print(f"  Feller Ratio (2*k*theta/sig^2): {receipt.feller_ratio:.2f}x (Positivity Guaranteed: {receipt.feller_condition_satisfied})")
    print("-" * 80)
    print("  Zero-Coupon Yields & Discount Bond Prices:")
    print(f"    3M / 2Y / 5Y / 10Y / 30Y (%): {receipt.yield_3m * 100:.2f}% | {receipt.yield_2y * 100:.2f}% | {receipt.yield_5y * 100:.2f}% | {receipt.yield_10y * 100:.2f}% | {receipt.yield_30y * 100:.2f}%")
    print(f"    1Y / 2Y / 5Y / 10Y / 30Y P($):{receipt.bond_price_1y:.4f} | {receipt.bond_price_2y:.4f} | {receipt.bond_price_5y:.4f} | {receipt.bond_price_10y:.4f} | {receipt.bond_price_30y:.4f}")
    print(f"    10Y-2Y Curve Slope:           {receipt.term_slope_10y_2y_bps:+.2f} bps")
    print(f"    10Y Factor Duration B(10):    {receipt.duration_10y:.4f} | Jensen Convexity: -{receipt.convexity_adjustment_bps:.2f} bps")
    print("-" * 80)
    print(f"  Term Structure Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def binomial_option_command(args: argparse.Namespace) -> None:
    """CLI handler for Cox, Ross & Rubinstein (1979) Binomial Option Pricing Engine."""
    import json
    from src.analytics.crr_binomial_options import (
        calculate_crr_binomial_option,
        persist_binomial_option_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    spot = float(getattr(args, "spot", 100.0) if getattr(args, "spot", None) is not None else 100.0)
    strike = float(getattr(args, "strike", 100.0) if getattr(args, "strike", None) is not None else 100.0)
    maturity = float(getattr(args, "maturity", 1.0) if getattr(args, "maturity", None) is not None else 1.0)
    rate = float(getattr(args, "rate", 0.05) if getattr(args, "rate", None) is not None else 0.05)
    div_yield = float(getattr(args, "div_yield", 0.0) if getattr(args, "div_yield", None) is not None else 0.0)
    vol = float(getattr(args, "vol", 0.25) if getattr(args, "vol", None) is not None else 0.25)
    steps = int(getattr(args, "steps", 150) if getattr(args, "steps", None) is not None else 150)
    opt_type = str(getattr(args, "type", "PUT") or "PUT").strip().upper()
    style = str(getattr(args, "style", "AMERICAN") or "AMERICAN").strip().upper()

    try:
        receipt = calculate_crr_binomial_option(
            spot_price=spot,
            strike_price=strike,
            maturity_years=maturity,
            risk_free_rate=rate,
            dividend_yield=div_yield,
            volatility=vol,
            steps=steps,
            option_type=opt_type,
            exercise_style=style,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_binomial_option_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Binomial option evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Cox, Ross & Rubinstein (1979) Binomial Option Pricing Engine")
    print("=" * 80)
    print(f"  Contract / Ticker:              {receipt.ticker} ({receipt.exercise_style} {receipt.option_type})")
    print(f"  Spot S0 / Strike K / T (yrs):   ${receipt.spot_price:.2f} | ${receipt.strike_price:.2f} | {receipt.maturity_years:.2f}y")
    print(f"  Vol / Rate r / Div q / Steps N: {receipt.volatility * 100:.2f}% | {receipt.risk_free_rate * 100:.2f}% | {receipt.dividend_yield * 100:.2f}% | {receipt.steps}")
    print(f"  Lattice Factors (u / d / p):    {receipt.up_factor_u:.4f} | {receipt.down_factor_d:.4f} | p={receipt.risk_neutral_prob_p:.4f}")
    print("-" * 80)
    print("  Valuation, Early-Exercise Decomposition & Lattice Greeks:")
    print(f"    Fair Option Value:            ${receipt.option_price:.4f} (BSM Euro Ref: ${receipt.black_scholes_benchmark:.4f}, Err: ${receipt.discretization_error_vs_bsm:.4f})")
    print(f"    American vs European Value:   ${receipt.american_price:.4f} vs ${receipt.european_price:.4f}")
    print(f"    Early-Exercise Premium:       ${receipt.early_exercise_premium:.4f} ({receipt.early_exercise_nodes} exercise nodes)")
    print(f"    Greeks (Delta/Gamma/Theta):   {receipt.delta:+.4f} | {receipt.gamma:.4f} | {receipt.theta_annual:+.4f}/yr")
    print("-" * 80)
    print(f"  Exercise Regime:                {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def margrabe_exchange_command(args: argparse.Namespace) -> None:
    """CLI handler for William Margrabe (1978) Exchange Option & Relative-Performance Engine."""
    import json
    from src.analytics.margrabe_exchange_option import (
        calculate_margrabe_exchange_option,
        persist_margrabe_exchange_record,
    )

    try:
        t1 = validate_ticker(args.asset1)
        t2 = validate_ticker(args.asset2)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    s1 = float(getattr(args, "s1", 100.0) if getattr(args, "s1", None) is not None else 100.0)
    s2 = float(getattr(args, "s2", 100.0) if getattr(args, "s2", None) is not None else 100.0)
    v1 = float(getattr(args, "vol1", 0.25) if getattr(args, "vol1", None) is not None else 0.25)
    v2 = float(getattr(args, "vol2", 0.20) if getattr(args, "vol2", None) is not None else 0.20)
    q1 = float(getattr(args, "div1", 0.01) if getattr(args, "div1", None) is not None else 0.01)
    q2 = float(getattr(args, "div2", 0.015) if getattr(args, "div2", None) is not None else 0.015)
    rho = float(getattr(args, "rho", 0.50) if getattr(args, "rho", None) is not None else 0.50)
    maturity = float(getattr(args, "maturity", 1.0) if getattr(args, "maturity", None) is not None else 1.0)

    try:
        receipt = calculate_margrabe_exchange_option(
            spot_asset1=s1,
            spot_asset2=s2,
            vol_asset1=v1,
            vol_asset2=v2,
            div_yield_asset1=q1,
            div_yield_asset2=q2,
            correlation_rho=rho,
            maturity_years=maturity,
            asset1_ticker=t1,
            asset2_ticker=t2,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_margrabe_exchange_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Margrabe exchange option evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: William Margrabe (1978) Exchange Option & Relative-Performance Engine")
    print("=" * 80)
    print(f"  Asset Exchange Pair:            {receipt.asset1_ticker} vs {receipt.asset2_ticker} (Maturity: {receipt.maturity_years:.2f}y)")
    print(f"  Spots (S1 / S2) & Vols:         ${receipt.spot_asset1:.2f} / ${receipt.spot_asset2:.2f} | {receipt.vol_asset1 * 100:.2f}% / {receipt.vol_asset2 * 100:.2f}% (rho={receipt.correlation_rho:+.2f})")
    print(f"  Composite Spread Vol (rel):     {receipt.composite_volatility * 100:.2f}%")
    print("-" * 80)
    print("  Relative Valuation & Hedge Sensitivities:")
    print(f"    Exchange Option Value:        ${receipt.exchange_option_value:.4f} ({receipt.normalized_switch_premium_pct:.2f}% switch premium)")
    print(f"    Intrinsic Forward / Time Val: ${receipt.intrinsic_forward_bound:.4f} | ${receipt.time_switching_value:.4f}")
    print(f"    Hedge Deltas (D1 / D2):       {receipt.delta_asset1:+.4f} {receipt.asset1_ticker} | {receipt.delta_asset2:+.4f} {receipt.asset2_ticker}")
    print(f"    Rel Vega / Corr Sensitivity:  ${receipt.relative_vega:.4f} | ${receipt.correlation_sensitivity:+.4f} (Outperf P={receipt.outperformance_prob * 100:.2f}%)")
    print("-" * 80)
    print(f"  Pair Exchange Regime:           {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def black_cox_command(args: argparse.Namespace) -> None:
    """CLI handler for Black & Cox (1976) First-Passage Structural Credit & Safety Covenant Engine."""
    import json
    from src.analytics.black_cox_covenants import (
        calculate_black_cox_covenants,
        persist_black_cox_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    v0 = float(getattr(args, "v0", 150.0) if getattr(args, "v0", None) is not None else 150.0)
    f_debt = float(getattr(args, "debt", 100.0) if getattr(args, "debt", None) is not None else 100.0)
    k_cov = float(getattr(args, "covenant_k", 80.0) if getattr(args, "covenant_k", None) is not None else 80.0)
    gamma = float(getattr(args, "gamma", 0.03) if getattr(args, "gamma", None) is not None else 0.03)
    maturity = float(getattr(args, "maturity", 5.0) if getattr(args, "maturity", None) is not None else 5.0)
    vol = float(getattr(args, "vol", 0.28) if getattr(args, "vol", None) is not None else 0.28)
    rate = float(getattr(args, "rate", 0.045) if getattr(args, "rate", None) is not None else 0.045)

    try:
        receipt = calculate_black_cox_covenants(
            asset_value_v0=v0,
            debt_face_f=f_debt,
            covenant_level_k=k_cov,
            covenant_growth_gamma=gamma,
            maturity_years=maturity,
            asset_volatility=vol,
            risk_free_rate=rate,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_black_cox_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Black-Cox covenant evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black & Cox (1976) First-Passage Structural Credit & Safety Covenant Engine")
    print("=" * 80)
    print(f"  Issuer / Ticker:                {receipt.ticker} (Breached: {receipt.covenant_breached_at_inception})")
    print(f"  Assets V0 / Debt F / Cov K:     ${receipt.asset_value_v0:.2f} | ${receipt.debt_face_f:.2f} | ${receipt.covenant_level_k:.2f} (H0=${receipt.initial_barrier_h0:.2f})")
    print(f"  Maturity T / Asset Vol / gamma: {receipt.maturity_years:.2f}y | {receipt.asset_volatility * 100:.2f}% | {receipt.covenant_growth_gamma * 100:.2f}%")
    print("-" * 80)
    print("  Safety Covenant Valuation & First-Passage Default Risk:")
    print(f"    Down-and-Out Equity E_BC:     ${receipt.black_cox_equity_value:.2f} (vs Merton: ${receipt.merton_equity_value:.2f}, Transfer: ${receipt.covenant_equity_transfer:.2f})")
    print(f"    Covenant Debt Value D_BC:     ${receipt.black_cox_debt_value:.2f} (vs Merton: ${receipt.merton_debt_value:.2f})")
    print(f"    Credit Spread (BC vs Merton): {receipt.black_cox_credit_spread_bps:.2f} bps vs {receipt.merton_credit_spread_bps:.2f} bps")
    print(f"    Covenant Spread Savings:      {receipt.covenant_spread_savings_bps:.2f} bps")
    print(f"    First-Passage / Total PD:     {receipt.first_passage_barrier_pd * 100:.2f}% | {receipt.total_default_prob * 100:.2f}% (Merton Term PD: {receipt.merton_terminal_pd * 100:.2f}%)")
    print(f"    Distance to Barrier (sigma):  {receipt.distance_to_covenant_barrier:.2f}x (Pre-Mat Share: {receipt.pre_maturity_default_share * 100:.1f}%)")
    print("-" * 80)
    print(f"  Credit Covenant Regime:         {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def kim_verrecchia_command(args: argparse.Namespace) -> None:
    """CLI handler for Kim & Verrecchia (1991, 1994) Differential Precision & Announcement Liquidity Engine."""
    import json
    from src.analytics.kim_verrecchia_liquidity import (
        calculate_kim_verrecchia_liquidity,
        persist_kim_verrecchia_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    prior_h = float(getattr(args, "prior_h", 4.0) if getattr(args, "prior_h", None) is not None else 4.0)
    pub_s = float(getattr(args, "pub_s", 2.0) if getattr(args, "pub_s", None) is not None else 2.0)
    priv_p = float(getattr(args, "priv_p", 3.0) if getattr(args, "priv_p", None) is not None else 3.0)
    dispersion = float(getattr(args, "dispersion", 1.5) if getattr(args, "dispersion", None) is not None else 1.5)
    abs_dp = float(getattr(args, "abs_dp", 0.04) if getattr(args, "abs_dp", None) is not None else 0.04)
    noise_u = float(getattr(args, "noise_u", 0.50) if getattr(args, "noise_u", None) is not None else 0.50)

    try:
        receipt = calculate_kim_verrecchia_liquidity(
            prior_precision_h=prior_h,
            public_signal_precision_s=pub_s,
            mean_private_precision_p=priv_p,
            precision_dispersion_sigma_p=dispersion,
            announcement_price_change_abs=abs_dp,
            noise_liquidity_sigma_u=noise_u,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_kim_verrecchia_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Kim-Verrecchia evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Kim & Verrecchia (1991, 1994) Differential Precision & Announcement Liquidity Engine")
    print("=" * 80)
    print(f"  Issuer / Ticker:                {receipt.ticker}")
    print(f"  Precisions (Prior h / Pub s / p): {receipt.prior_precision_h:.2f} | {receipt.public_signal_precision_s:.2f} | {receipt.mean_private_precision_p:.2f} (sigma_p={receipt.precision_dispersion_sigma_p:.2f})")
    print(f"  Surprise |Delta P| / Total Prec: {receipt.announcement_price_change_abs * 100:.2f}% | {receipt.total_posterior_precision:.2f} (alpha={receipt.price_response_alpha:.3f})")
    print("-" * 80)
    print("  Announcement Volume & Event-Window Adverse Selection:")
    print(f"    Expected Disclosure Volume:   {receipt.expected_announcement_volume:.6f} (Slope V/|dP|={receipt.volume_to_price_change_ratio:.4f})")
    print(f"    Differential Belief Ratio:    {receipt.belief_dispersion_ratio * 100:.2f}%")
    print(f"    Event Illiquidity lambda_KV:  {receipt.announcement_illiquidity_lambda:.4f} (Baseline: {receipt.baseline_illiquidity_lambda:.4f})")
    print(f"    Illiquidity Surge (KV 1994):  +{receipt.illiquidity_surge_bps:.2f} bps")
    print("-" * 80)
    print(f"  Disclosure Liquidity Regime:    {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def myers_rajan_command(args: argparse.Namespace) -> None:
    """CLI handler for Myers & Rajan (1998) Paradox of Liquidity Engine."""
    import json
    from src.analytics.myers_rajan_liquidity_paradox import (
        calculate_myers_rajan_paradox,
        persist_myers_rajan_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    c1 = float(getattr(args, "assets", 100.0) if getattr(args, "assets", None) is not None else 100.0)
    d2 = float(getattr(args, "continuation", 125.0) if getattr(args, "continuation", None) is not None else 125.0)
    alpha = float(getattr(args, "alpha", 0.75) if getattr(args, "alpha", None) is not None else 0.75)
    lam = float(getattr(args, "lambda_trans", 0.25) if getattr(args, "lambda_trans", None) is not None else 0.25)
    theta = float(getattr(args, "theta", 0.0) if getattr(args, "theta", None) is not None else 0.0)

    try:
        receipt = calculate_myers_rajan_paradox(
            invested_assets_c1=c1,
            continuation_value_d2=d2,
            asset_liquidity_alpha=alpha,
            transformation_friction_lambda=lam,
            insider_rent_share_theta=theta,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_myers_rajan_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Myers & Rajan evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Myers & Rajan (1998) Paradox of Liquidity Engine")
    print("=" * 80)
    print(f"  Firm / Ticker:                  {receipt.ticker} (Paradox Zone: {receipt.paradox_zone_active})")
    print(f"  Assets c1 / Continuation d2:    ${receipt.invested_assets_c1:.2f} | ${receipt.continuation_value_d2:.2f} (Rent theta={receipt.insider_rent_share_theta * 100:.1f}%)")
    print(f"  Liquidity alpha / Friction lam: {receipt.asset_liquidity_alpha * 100:.1f}% | {receipt.transformation_friction_lambda * 100:.1f}% (Optimal alpha*={receipt.optimal_liquidity_alpha_star * 100:.1f}%)")
    print("-" * 80)
    print("  Liquidation Collateral vs. Transformation Incentive Ceiling:")
    print(f"    Creditor Liquidation Bound:   ${receipt.liquidation_debt_capacity:.2f}")
    print(f"    Transformation Ceiling:       ${receipt.transformation_debt_ceiling:.2f}")
    print(f"    Binding Debt Capacity:        ${receipt.binding_debt_capacity:.2f} ({receipt.leverage_ratio_pct:.1f}% of c1, Peak: ${receipt.peak_debt_capacity:.2f})")
    print(f"    Paradox Capacity Loss:        ${receipt.paradox_capacity_loss:.2f} (Bank Short-Debt Req: {receipt.bank_monitoring_requirement_pct:.1f}%)")
    print("-" * 80)
    print(f"  Liquidity Governance Regime:    {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def hull_white_command(args: argparse.Namespace) -> None:
    """CLI handler for Hull & White (1990) No-Arbitrage Term Structure & Bond Option Engine."""
    import json
    from src.analytics.hull_white_bond_options import (
        calculate_hull_white_bond_option,
        persist_hull_white_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    r0 = float(getattr(args, "r0", 0.035) if getattr(args, "r0", None) is not None else 0.035)
    y_inf = float(getattr(args, "y_inf", 0.050) if getattr(args, "y_inf", None) is not None else 0.050)
    a_param = float(getattr(args, "a", 0.15) if getattr(args, "a", None) is not None else 0.15)
    sigma = float(getattr(args, "sigma", 0.015) if getattr(args, "sigma", None) is not None else 0.015)
    t_opt = float(getattr(args, "t_opt", 2.0) if getattr(args, "t_opt", None) is not None else 2.0)
    s_bond = float(getattr(args, "s_bond", 10.0) if getattr(args, "s_bond", None) is not None else 10.0)
    strike = float(getattr(args, "strike", 0.65) if getattr(args, "strike", None) is not None else 0.65)
    opt_type = str(getattr(args, "type", "CALL") or "CALL").strip().upper()

    try:
        receipt = calculate_hull_white_bond_option(
            short_rate_r0=r0,
            long_yield_y_inf=y_inf,
            mean_reversion_a=a_param,
            rate_volatility_sigma=sigma,
            option_maturity_t=t_opt,
            bond_maturity_s=s_bond,
            strike_k=strike,
            option_type=opt_type,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_hull_white_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Hull-White bond option evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Hull & White (1990) No-Arbitrage Term Structure & Bond Option Engine")
    print("=" * 80)
    print(f"  Instrument / Ticker:            {receipt.ticker} (Zero-Coupon Bond {receipt.option_type})")
    print(f"  Rates (r0 / y_inf) & a / sig:   {receipt.short_rate_r0 * 100:.2f}% / {receipt.long_yield_y_inf * 100:.2f}% | a={receipt.mean_reversion_a:.2f} | sig={receipt.rate_volatility_sigma * 100:.2f}%")
    print(f"  Option Expiry T / Bond S / K:   {receipt.option_maturity_t:.2f}y | {receipt.bond_maturity_s:.2f}y | Strike K=${receipt.strike_k:.4f}")
    print("-" * 80)
    print("  No-Arbitrage Curve Fit & Closed-Form Bond Option Valuation:")
    print(f"    Market Discounts P(0,T / S):  ${receipt.discount_p_0_t:.4f} | ${receipt.discount_p_0_s:.4f} (Fit Err: {receipt.curve_fit_error_bps:.2f} bps)")
    print(f"    Forward Bond Price / Yield:   ${receipt.forward_bond_price:.4f} ({receipt.forward_yield_rate * 100:.2f}% | vol_p={receipt.bond_price_vol_sigma_p * 100:.2f}%)")
    print(f"    Selected Option Fair Value:   ${receipt.option_price:.4f} (Intrinsic: ${receipt.forward_intrinsic_value:.4f}, Time: ${receipt.time_value_premium:.4f})")
    print(f"    Bond Call / Put & Delta:      ${receipt.bond_call_price:.4f} | ${receipt.bond_put_price:.4f} (Spot Delta={receipt.spot_delta:+.4f})")
    print("-" * 80)
    print(f"  Rate Option Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def modigliani_miller_command(args: argparse.Namespace) -> None:
    """CLI handler for Modigliani & Miller (1958, 1961, 1963) Capital Structure & Dividend Engine."""
    import json
    from src.analytics.modigliani_miller_corporate import (
        calculate_modigliani_miller_corporate,
        persist_modigliani_miller_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    ebit = float(getattr(args, "ebit", 20.0) if getattr(args, "ebit", None) is not None else 20.0)
    debt = float(getattr(args, "debt", 50.0) if getattr(args, "debt", None) is not None else 50.0)
    r0 = float(getattr(args, "r0", 0.10) if getattr(args, "r0", None) is not None else 0.10)
    rd = float(getattr(args, "rd", 0.05) if getattr(args, "rd", None) is not None else 0.05)
    tc = float(getattr(args, "tax_rate", 0.21) if getattr(args, "tax_rate", None) is not None else 0.21)
    k_inv = float(getattr(args, "reinvest", 0.40) if getattr(args, "reinvest", None) is not None else 0.40)
    roic = float(getattr(args, "roic", 0.14) if getattr(args, "roic", None) is not None else 0.14)
    payout = float(getattr(args, "payout", 0.60) if getattr(args, "payout", None) is not None else 0.60)

    try:
        receipt = calculate_modigliani_miller_corporate(
            ebit=ebit,
            debt_d=debt,
            unlevered_r0=r0,
            cost_of_debt_rd=rd,
            corporate_tax_rate_tc=tc,
            reinvestment_rate_k=k_inv,
            incremental_roic=roic,
            payout_ratio=payout,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_modigliani_miller_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Modigliani & Miller evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Modigliani & Miller (1958, 1961, 1963) Capital Structure & Dividend Engine")
    print("=" * 80)
    print(f"  Corporation / Ticker:           {receipt.ticker} (EBIT=${receipt.ebit:.2f}, NOPAT=${receipt.nopat:.2f})")
    print(f"  Debt D / Tax tc / Cost r0, rD:  ${receipt.debt_d:.2f} | {receipt.corporate_tax_rate_tc * 100:.1f}% | {receipt.unlevered_r0 * 100:.2f}%, {receipt.cost_of_debt_rd * 100:.2f}%")
    print(f"  Reinvest k / ROIC / Payout:     {receipt.reinvestment_rate_k * 100:.1f}% | {receipt.incremental_roic * 100:.2f}% | {receipt.payout_ratio * 100:.1f}%")
    print("-" * 80)
    print("  MM Propositions I/II, Tax Shield & PVGO Decomposition:")
    print(f"    Unlevered V_U + Tax Shield:   ${receipt.unlevered_value_vu:.2f} + ${receipt.tax_shield_pv:.2f} = V_L ${receipt.levered_value_vl:.2f}")
    print(f"    Levered Equity E_L / D-to-E:  ${receipt.levered_equity_el:.2f} ({receipt.debt_to_equity_ratio:.2f}x)")
    print(f"    Levered Cost of Equity (rE):  {receipt.levered_equity_cost_re * 100:.2f}%")
    print(f"    After-Tax WACC:               {receipt.wacc * 100:.2f}%")
    print(f"    Growth PVGO / MM Equity:      ${receipt.pvgo:+.2f} ({receipt.pvgo_share_pct:+.1f}% of V_total) -> ${receipt.mm_growth_equity_value:.2f}")
    print("-" * 80)
    print(f"  Corporate Valuation Regime:     {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def mcdonald_siegel_command(args: argparse.Namespace) -> None:
    """CLI handler for McDonald & Siegel (1986) Value of Waiting & Irreversible Investment Engine."""
    import json
    from src.analytics.mcdonald_siegel_waiting import (
        calculate_mcdonald_siegel_waiting,
        persist_mcdonald_siegel_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    v_val = float(getattr(args, "v", 130.0) if getattr(args, "v", None) is not None else 130.0)
    i_cost = float(getattr(args, "cost", 100.0) if getattr(args, "cost", None) is not None else 100.0)
    r_rate = float(getattr(args, "rate", 0.05) if getattr(args, "rate", None) is not None else 0.05)
    delta_y = float(getattr(args, "delta", 0.04) if getattr(args, "delta", None) is not None else 0.04)
    sigma = float(getattr(args, "vol", 0.28) if getattr(args, "vol", None) is not None else 0.28)

    try:
        receipt = calculate_mcdonald_siegel_waiting(
            project_value_v=v_val,
            investment_cost_i=i_cost,
            risk_free_rate_r=r_rate,
            dividend_shortfall_delta=delta_y,
            cashflow_volatility_sigma=sigma,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_mcdonald_siegel_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: McDonald & Siegel evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: McDonald & Siegel (1986) Value of Waiting & Irreversible Investment Engine")
    print("=" * 80)
    print(f"  Project / Ticker:               {receipt.ticker} (Invest Immediately: {receipt.invest_immediately})")
    print(f"  Value V / Cost I / Vol sigma:   ${receipt.project_value_v:.2f} | ${receipt.investment_cost_i:.2f} | {receipt.cashflow_volatility_sigma * 100:.2f}%")
    print(f"  Rate r / Shortfall delta / b1:  {receipt.risk_free_rate_r * 100:.2f}% | {receipt.dividend_shortfall_delta * 100:.2f}% | beta_1={receipt.characteristic_root_beta1:.3f}")
    print("-" * 80)
    print("  Hysteresis Hurdle & Optimal Investment Timing:")
    print(f"    Optimal Hurdle Trigger V*:    ${receipt.optimal_trigger_v_star:.2f} ({receipt.hysteresis_hurdle_multiple:.2f}x I, Cov={receipt.trigger_coverage_ratio * 100:.1f}%)")
    print(f"    Real Option Value F(V):       ${receipt.real_option_value_f:.2f} (Delta={receipt.waiting_option_delta:.3f})")
    print(f"    Naive Static NPV (V - I):     ${receipt.naive_static_npv:+.2f}")
    print(f"    Pure Value of Waiting:        ${receipt.pure_value_of_waiting:.2f}")
    print("-" * 80)
    print(f"  Investment Timing Regime:       {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def shleifer_vishny_liquidation_command(args: argparse.Namespace) -> None:
    """CLI handler for Shleifer & Vishny (1992) Liquidation Values & Debt Capacity Engine."""
    import json
    from src.analytics.shleifer_vishny_liquidation import (
        calculate_shleifer_vishny_liquidation,
        persist_shleifer_vishny_liquidation_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    v1 = float(getattr(args, "v1", 100.0) if getattr(args, "v1", None) is not None else 100.0)
    spec = float(getattr(args, "specificity", 0.40) if getattr(args, "specificity", None) is not None else 0.40)
    slack = float(getattr(args, "peer_slack", 0.35) if getattr(args, "peer_slack", None) is not None else 0.35)
    rho = float(getattr(args, "rho", 0.65) if getattr(args, "rho", None) is not None else 0.65)
    q_prob = float(getattr(args, "downturn_prob", 0.30) if getattr(args, "downturn_prob", None) is not None else 0.30)

    try:
        receipt = calculate_shleifer_vishny_liquidation(
            specialist_asset_value_v1=v1,
            asset_specificity_discount=spec,
            peer_debt_capacity_slack=slack,
            industry_correlation_rho=rho,
            downturn_prob_q=q_prob,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_shleifer_vishny_liquidation_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Shleifer & Vishny liquidation evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Shleifer & Vishny (1992) Liquidation Values & Debt Capacity Engine")
    print("=" * 80)
    print(f"  Issuer / Ticker:                {receipt.ticker} (Specialist Value V1=${receipt.specialist_asset_value_v1:.2f}, Outsider V2=${receipt.outside_buyer_value_v2:.2f})")
    print(f"  Specificity delta / Peer Slack: {receipt.asset_specificity_discount * 100:.1f}% | {receipt.peer_debt_capacity_slack * 100:.1f}% (Industry rho={receipt.industry_correlation_rho:.2f}, q={receipt.downturn_prob_q:.2f})")
    print("-" * 80)
    print("  Equilibrium Liquidation & Ex-Ante Debt Capacity:")
    print(f"    Idiosyncratic / Downturn Liq: ${receipt.idiosyncratic_liquidation_price:.2f} | ${receipt.downturn_liquidation_price:.2f}")
    print(f"    Expected Distress Liquidation:${receipt.expected_distress_liquidation_price:.2f}")
    print(f"    Ex-Ante Risky Debt Capacity:  ${receipt.risky_debt_capacity:.2f} (Haircut: -{receipt.debt_capacity_haircut_pct:.2f}%)")
    print(f"    Fire-Sale Haircut / Loss:     {receipt.fire_sale_haircut_pct:.2f}% (${receipt.collateral_capacity_loss:.2f})")
    print("-" * 80)
    print(f"  Collateral Equilibrium Regime:  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def dixit_hysteresis_command(args: argparse.Namespace) -> None:
    """CLI handler for Dixit (1989) Entry-Exit Hysteresis Band & Switching Engine."""
    import json
    from src.analytics.dixit_hysteresis_entry_exit import (
        calculate_dixit_hysteresis,
        persist_dixit_hysteresis_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    price = float(getattr(args, "price", 1.10) if getattr(args, "price", None) is not None else 1.10)
    cost = float(getattr(args, "cost", 1.00) if getattr(args, "cost", None) is not None else 1.00)
    entry_k = float(getattr(args, "entry_k", 2.00) if getattr(args, "entry_k", None) is not None else 2.00)
    exit_l = float(getattr(args, "exit_l", 1.50) if getattr(args, "exit_l", None) is not None else 1.50)
    vol = float(getattr(args, "vol", 0.25) if getattr(args, "vol", None) is not None else 0.25)
    state = str(getattr(args, "state", "ACTIVE") or "ACTIVE").strip().upper()

    try:
        receipt = calculate_dixit_hysteresis(
            current_price_p=price,
            operating_cost_c=cost,
            entry_cost_k=entry_k,
            exit_cost_l=exit_l,
            price_volatility_sigma=vol,
            current_state=state,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_dixit_hysteresis_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Dixit hysteresis evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Dixit (1989) Entry-Exit Hysteresis Band & Switching Engine")
    print("=" * 80)
    print(f"  Operator / Ticker:              {receipt.ticker} (State: {receipt.current_state} -> Action: {receipt.recommended_action})")
    print(f"  Price P / Op Cost C / Vol sig:  ${receipt.current_price_p:.4f} | ${receipt.operating_cost_c:.4f} | {receipt.price_volatility_sigma * 100:.2f}%")
    print(f"  Sunk Costs (Entry k / Exit l):  ${receipt.entry_cost_k:.2f} | ${receipt.exit_cost_l:.2f} (beta1={receipt.root_beta1:.3f}, alpha={receipt.root_alpha:.3f})")
    print("-" * 80)
    print("  Coupled Switching Boundaries & Real-Option Valuations:")
    print(f"    Entry Trigger P_H (Marsh):    ${receipt.entry_trigger_ph:.4f} (Marsh=${receipt.marshallian_entry_trigger:.4f})")
    print(f"    Exit Trigger P_L (Marsh):     ${receipt.exit_trigger_pl:.4f} (Marsh=${receipt.marshallian_exit_trigger:.4f})")
    print(f"    Hysteresis Band Width:        ${receipt.hysteresis_band_width:.4f} (Option Prem: +${receipt.option_hysteresis_premium:.4f})")
    print(f"    Idle V0(P) / Active V1(P):    ${receipt.idle_value_v0:.2f} | ${receipt.active_value_v1:.2f} (Res: {receipt.value_matching_residual:.2e})")
    print("-" * 80)
    print(f"  Hysteresis Switching Regime:    {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def holmstrom_tirole_1998_command(args: argparse.Namespace) -> None:
    """CLI handler for Holmstrom & Tirole (1998) Private & Public Supply of Liquidity Engine."""
    import json
    from src.analytics.holmstrom_tirole_public_liquidity import (
        calculate_public_liquidity_supply,
        persist_public_liquidity_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    equity_a = float(getattr(args, "equity", 20.0) if getattr(args, "equity", None) is not None else 20.0)
    rho0 = float(getattr(args, "rho0", 0.70) if getattr(args, "rho0", None) is not None else 0.70)
    rho1 = float(getattr(args, "rho1", 1.45) if getattr(args, "rho1", None) is not None else 1.45)
    shock = float(getattr(args, "shock", 0.90) if getattr(args, "shock", None) is not None else 0.90)
    theta = float(getattr(args, "theta", 0.35) if getattr(args, "theta", None) is not None else 0.35)
    lam = float(getattr(args, "lambda_agg", 0.60) if getattr(args, "lambda_agg", None) is not None else 0.60)

    try:
        receipt = calculate_public_liquidity_supply(
            entrepreneur_equity_a=equity_a,
            pledgeable_return_rho0=rho0,
            total_return_rho1=rho1,
            interim_shock_rho=shock,
            shock_prob_theta=theta,
            aggregate_corr_lambda=lam,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_public_liquidity_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Public liquidity evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Holmstrom & Tirole (1998) Private & Public Supply of Liquidity Engine")
    print("=" * 80)
    print(f"  Sector / Ticker:                {receipt.ticker} (Net Worth A=${receipt.entrepreneur_equity_a:.2f}, Cont Efficient: {receipt.continuation_efficient})")
    print(f"  Pledgeable rho0 / Return rho1:  {receipt.pledgeable_return_rho0:.2f} | {receipt.total_return_rho1:.2f} (Interim Shock={receipt.interim_shock_rho:.2f}, Gap={receipt.pledgeable_liquidity_gap:.2f})")
    print(f"  Shock Prob theta / Agg lambda:  {receipt.shock_prob_theta * 100:.1f}% | {receipt.aggregate_corr_lambda * 100:.1f}% (Res Req={receipt.optimal_reserve_ratio_pct:.1f}%)")
    print("-" * 80)
    print("  Investment Capacity & Contingent Treasury Liquidity Equilibrium:")
    print(f"    Pooled / Agg / Hoard Scale:   ${receipt.pooled_investment_scale:.2f} | ${receipt.aggregate_investment_scale:.2f} | ${receipt.hoarding_investment_scale:.2f}")
    print(f"    Public Liquidity Scale Gain:  +${receipt.public_liquidity_scale_gain:.2f} (Social Welfare Gain: +${receipt.welfare_gain_public_supply:.2f})")
    print(f"    Treasury Liquidity q_L:       {receipt.treasury_liquidity_price_ql:.4f} (+{receipt.liquidity_shortage_premium_bps:.2f} bps liquidity premium)")
    print("-" * 80)
    print(f"  Liquidity Supply Regime:        {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def cppi_insurance_command(args: argparse.Namespace) -> None:
    """CLI handler for Black & Jones (1987) CPPI & TIPP Portfolio Insurance Engine."""
    import json
    from src.analytics.cppi_portfolio_insurance import (
        calculate_cppi_allocation,
        persist_cppi_insurance_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    wealth = float(getattr(args, "wealth", 100.0) if getattr(args, "wealth", None) is not None else 100.0)
    floor_v = float(getattr(args, "floor", 85.0) if getattr(args, "floor", None) is not None else 85.0)
    mult = float(getattr(args, "multiplier", 4.0) if getattr(args, "multiplier", None) is not None else 4.0)
    lev_cap = float(getattr(args, "max_lev", 1.0) if getattr(args, "max_lev", None) is not None else 1.0)
    strat = str(getattr(args, "strategy", "CPPI") or "CPPI").strip().upper()

    try:
        receipt = calculate_cppi_allocation(
            portfolio_value_w=wealth,
            floor_value_f=floor_v,
            multiplier_m=mult,
            max_leverage_cap=lev_cap,
            strategy=strat,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_cppi_insurance_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: CPPI portfolio insurance evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Black & Jones (1987) CPPI & TIPP Portfolio Insurance Engine")
    print("=" * 80)
    print(f"  Portfolio / Ticker:             {receipt.ticker} ({receipt.strategy})")
    print(f"  Wealth W / Floor F / Cushion C: ${receipt.portfolio_value_w:.2f} | ${receipt.floor_value_f:.2f} | ${receipt.cushion_c:.2f}")
    print(f"  Multiplier m / Lev Cap / Tol:   {receipt.multiplier_m:.2f}x | {receipt.max_leverage_cap:.2f}x | -{receipt.gap_crash_tolerance_pct:.2f}% max 1-step crash")
    print("-" * 80)
    print("  Dynamic Cushion Risk Allocation:")
    print(f"    Risky Asset Exposure E_t:     ${receipt.risky_exposure_e:.2f} ({receipt.risky_weight_pct:.2f}% weight)")
    print(f"    Safe Treasury Reserve B_t:    ${receipt.safe_cash_reserve_b:.2f} ({100.0 - receipt.risky_weight_pct:.2f}% weight)")
    print(f"    Cash Locked / Gap Breach:     {receipt.cash_locked} | {receipt.gap_risk_breach}")
    print("-" * 80)
    print(f"  Insurance Protection Regime:    {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def harris_raviv_debt_command(args: argparse.Namespace) -> None:
    """CLI handler for Harris & Raviv (1990) Capital Structure & Informational Debt Engine."""
    import json
    from src.analytics.harris_raviv_informational_debt import (
        calculate_harris_raviv_debt,
        persist_harris_raviv_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    y_min = float(getattr(args, "y_min", 40.0) if getattr(args, "y_min", None) is not None else 40.0)
    y_max = float(getattr(args, "y_max", 160.0) if getattr(args, "y_max", None) is not None else 160.0)
    liq_val = float(getattr(args, "liq_val", 95.0) if getattr(args, "liq_val", None) is not None else 95.0)
    audit_k = float(getattr(args, "audit_cost", 8.0) if getattr(args, "audit_cost", None) is not None else 8.0)
    debt_d = getattr(args, "debt", None)

    try:
        receipt = calculate_harris_raviv_debt(
            y_min=y_min,
            y_max=y_max,
            liquidation_value_l=liq_val,
            investigation_cost_k=audit_k,
            promised_debt_d=float(debt_d) if debt_d is not None else None,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_harris_raviv_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Harris & Raviv evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Harris & Raviv (1990) Capital Structure & Informational Debt Engine")
    print("=" * 80)
    print(f"  Corporation / Ticker:           {receipt.ticker} (Cash-Flow Support: [${receipt.y_min:.2f}, ${receipt.y_max:.2f}])")
    print(f"  Liquidation L / Audit Cost k:   ${receipt.liquidation_value_l:.2f} | ${receipt.investigation_cost_k:.2f}")
    print(f"  Promised Debt D (Chosen):       ${receipt.promised_debt_d:.2f} (Default Audit Prob: {receipt.default_audit_prob * 100:.1f}%, Liq Prob: {receipt.liquidation_prob * 100:.1f}%)")
    print("-" * 80)
    print("  Informational Investigation & Liquidation Discipline Equilibrium:")
    print(f"    Optimal Debt Face D*:         ${receipt.optimal_debt_d_star:.2f} (Max Firm Value: ${receipt.optimal_firm_value_v_star:.2f})")
    print(f"    Unlevered vs. V(D):           ${receipt.unlevered_value:.2f} -> ${receipt.firm_value_v_d:.2f} (Net Gain: +${receipt.net_discipline_value:.2f})")
    print(f"    Gross Liq / Expected Audit:   +${receipt.gross_liquidation_gain:.2f} | -${receipt.expected_investigation_cost:.2f} (Over-Lev Waste: ${receipt.over_leverage_waste:.2f})")
    print("-" * 80)
    print(f"  Debt Discipline Regime:         {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def berk_green_command(args: argparse.Namespace) -> None:
    """CLI handler for Berk & Green (2004) Rational Fund Flows & Alpha Capacity Engine."""
    import json
    from src.analytics.berk_green_fund_flows import (
        calculate_berk_green_equilibrium,
        persist_berk_green_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    prior = float(getattr(args, "prior", 0.03) if getattr(args, "prior", None) is not None else 0.03)
    realized = float(getattr(args, "realized", 0.06) if getattr(args, "realized", None) is not None else 0.06)
    fee = float(getattr(args, "fee", 0.010) if getattr(args, "fee", None) is not None else 0.010)
    cost_a = float(getattr(args, "cost_a", 0.005) if getattr(args, "cost_a", None) is not None else 0.005)
    aum = float(getattr(args, "aum", 3.0) if getattr(args, "aum", None) is not None else 3.0)

    try:
        receipt = calculate_berk_green_equilibrium(
            prior_skill_phi0=prior,
            realized_alpha_r=realized,
            management_fee_f=fee,
            cost_convexity_a=cost_a,
            current_aum_q=aum,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_berk_green_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Berk & Green evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Berk & Green (2004) Rational Fund Flows & Alpha Capacity Engine")
    print("=" * 80)
    print(f"  Fund / Strategy Ticker:         {receipt.ticker} (Current AUM: ${receipt.current_aum_q:.2f}B | Fee: {receipt.management_fee_f * 100:.2f}%)")
    print(f"  Prior Skill / Realized Alpha:   {receipt.prior_skill_phi0 * 100:.2f}% | {receipt.realized_alpha_r * 100:.2f}% (Signal Weight={receipt.signal_weight:.3f})")
    print(f"  Bayesian Posterior Skill phi:   {receipt.posterior_skill_phi_hat * 100:.2f}% (Scale Cost a={receipt.cost_convexity_a:.4f})")
    print("-" * 80)
    print("  Decreasing-Returns Equilibrium & Fee Rent Capture:")
    print(f"    Equilibrium AUM q*:           ${receipt.equilibrium_aum_q_star:.2f}B (Utilization: {receipt.capacity_utilization_pct:.1f}%)")
    print(f"    Predicted Rational Flow d_q:  {receipt.expected_flow_delta_q:+.2f}B ({receipt.flow_pct_of_aum:+.1f}%)")
    print(f"    Current vs Eq Net Alpha:      {receipt.current_net_alpha * 100:+.2f}% vs {receipt.equilibrium_net_alpha * 100:.2f}%")
    print(f"    Equilibrium Fee Rent (f*q*):  {receipt.equilibrium_fee_revenue * 1000:.2f}M (Opt Fee f*={receipt.optimal_monopoly_fee_f_star * 100:.2f}%, Max Rent={receipt.max_manager_fee_rent * 1000:.2f}M)")
    print("-" * 80)
    print(f"  Capacity & Flow Regime:         {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def diamond_maturity_command(args: argparse.Namespace) -> None:
    """CLI handler for Douglas W. Diamond (1991) Debt Maturity Structure & Liquidity Risk Engine."""
    import json
    from src.analytics.diamond_debt_maturity import (
        calculate_diamond_debt_maturity,
        persist_diamond_maturity_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    f_prob = float(getattr(args, "rating_f", 0.72) if getattr(args, "rating_f", None) is not None else 0.72)
    liq_l = float(getattr(args, "liq_l", 0.70) if getattr(args, "liq_l", None) is not None else 0.70)
    rent_c = float(getattr(args, "rent_c", 0.50) if getattr(args, "rent_c", None) is not None else 0.50)
    qg = float(getattr(args, "qg", 0.25) if getattr(args, "qg", None) is not None else 0.25)
    qb = float(getattr(args, "qb", 0.80) if getattr(args, "qb", None) is not None else 0.80)

    try:
        receipt = calculate_diamond_debt_maturity(
            prior_good_prob_f=f_prob,
            liquidation_value_l=liq_l,
            control_rent_c=rent_c,
            good_bad_news_prob_qg=qg,
            bad_bad_news_prob_qb=qb,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_diamond_maturity_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Diamond debt maturity evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Douglas W. Diamond (1991) Debt Maturity Structure & Rollover Liquidity Risk Engine")
    print("=" * 80)
    print(f"  Borrower / Ticker:              {receipt.ticker} (Prior Rating f={receipt.prior_good_prob_f:.2f}, Downgrade f_d={receipt.downgrade_posterior_fd:.2f})")
    print(f"  Payoff X / Liq L / Control C:   {receipt.continuation_payoff_x:.2f} | {receipt.liquidation_value_l:.2f} | {receipt.control_rent_c:.2f} (q_G={receipt.good_bad_news_prob_qg:.2f}, q_B={receipt.bad_bad_news_prob_qb:.2f})")
    print(f"  Feasibility (ST / LT) & Panic:  {receipt.short_term_feasible} / {receipt.long_term_feasible} | Rollover Panic: {receipt.rollover_panic_triggered}")
    print("-" * 80)
    print("  Maturity Tradeoff (Screening Capture vs. Rollover Liquidation):")
    print(f"    Recommended Maturity:         {receipt.recommended_maturity} (Net ST-LT Advantage: {receipt.short_vs_long_net_advantage:+.4f})")
    print(f"    Short vs Long Payoffs U_ST/LT:{receipt.short_term_utility_ust:.4f} | {receipt.long_term_utility_ult:.4f}")
    print(f"    Upgrade Gain / Rollover Cost: +{receipt.upgrade_screening_gain:.4f} | -{receipt.rollover_liquidation_cost:.4f}")
    print(f"    Contract Faces (R_ST / R_LT): {receipt.short_term_face_rst:.4f} | {receipt.long_term_face_rlt:.4f}")
    print("-" * 80)
    print(f"  Debt Maturity Regime:           {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def blanchard_watson_command(args: argparse.Namespace) -> None:
    """CLI handler for Blanchard & Watson (1982) Rational Speculative Bubbles Engine."""
    import json
    from src.analytics.blanchard_watson_bubble import (
        calculate_blanchard_watson_bubble,
        persist_blanchard_watson_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    price = float(getattr(args, "price", 150.0) if getattr(args, "price", None) is not None else 150.0)
    fund = float(getattr(args, "fundamental", 100.0) if getattr(args, "fundamental", None) is not None else 100.0)
    rate_r = float(getattr(args, "rate", 0.05) if getattr(args, "rate", None) is not None else 0.05)
    pi_surv = float(getattr(args, "pi", 0.85) if getattr(args, "pi", None) is not None else 0.85)

    try:
        receipt = calculate_blanchard_watson_bubble(
            observed_price_p=price,
            fundamental_value_f=fund,
            discount_rate_r=rate_r,
            survival_prob_pi=pi_surv,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_blanchard_watson_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Blanchard & Watson evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Blanchard & Watson (1982) Rational Speculative Bubbles Engine")
    print("=" * 80)
    print(f"  Asset / Ticker:                 {receipt.ticker} (Discount r={receipt.discount_rate_r * 100:.2f}%)")
    print(f"  Price P_t / Fundamental F_t:    ${receipt.observed_price_p:.2f} | ${receipt.fundamental_value_f:.2f}")
    print(f"  Speculative Bubble B_t (Share): ${receipt.bubble_component_b:.2f} ({receipt.bubble_share_pct:.1f}% of Price)")
    print("-" * 80)
    print("  Euler Martingale Dynamics & Crash Hazard:")
    print(f"    Survival pi / Crash Hazard:   {receipt.survival_prob_pi * 100:.1f}% | {receipt.crash_prob * 100:.1f}% (Exp Horizon: {receipt.expected_remaining_periods:.1f} periods)")
    print(f"    Required Bubble Growth (1+r)/pi-1: {receipt.required_bubble_growth * 100:.2f}% (Surviving Asset Ret: {receipt.surviving_asset_return * 100:.2f}%)")
    print(f"    Euler Exp E[B_t+1] / Surv:    ${receipt.expected_next_bubble:.2f} | ${receipt.next_surviving_bubble:.2f}")
    print(f"    Instantaneous Burst Loss:     -{receipt.burst_drawdown_loss_pct:.1f}%")
    print("-" * 80)
    print(f"  Speculative Bubble Regime:      {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def black_zero_beta_command(args: argparse.Namespace) -> None:
    """CLI handler for Fischer Black (1972) Zero-Beta CAPM & SML Flattening Engine."""
    import json
    from src.analytics.black_zero_beta_capm import (
        calculate_black_zero_beta_capm,
        persist_black_zero_beta_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    beta = float(getattr(args, "beta", 0.70) if getattr(args, "beta", None) is not None else 0.70)
    rf = float(getattr(args, "rf", 0.035) if getattr(args, "rf", None) is not None else 0.035)
    rz = float(getattr(args, "rz", 0.055) if getattr(args, "rz", None) is not None else 0.055)
    rm = float(getattr(args, "rm", 0.095) if getattr(args, "rm", None) is not None else 0.095)

    try:
        receipt = calculate_black_zero_beta_capm(
            asset_beta=beta,
            risk_free_rate_rf=rf,
            zero_beta_rate_rz=rz,
            market_return_rm=rm,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_black_zero_beta_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Black Zero-Beta CAPM evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fischer Black (1972) Zero-Beta CAPM & SML Flattening Engine")
    print("=" * 80)
    print(f"  Asset / Ticker:                 {receipt.ticker} (Beta: {receipt.asset_beta:.2f})")
    print(f"  Rates (rf / R_z / R_m):         {receipt.risk_free_rate_rf * 100:.2f}% | {receipt.zero_beta_rate_rz * 100:.2f}% | {receipt.market_return_rm * 100:.2f}% (ZB Spread: +{receipt.zero_beta_spread * 100:.2f}%)")
    print("-" * 80)
    print("  Security Market Line Flattening & CAPM Comparison:")
    print(f"    Black Expected Return:        {receipt.black_expected_return * 100:.2f}% / yr (Slope: {receipt.black_sml_slope * 100:.2f}%)")
    print(f"    Sharpe-Lintner Expected Ret:  {receipt.sharpe_lintner_return * 100:.2f}% / yr (Slope: {receipt.sharpe_lintner_slope * 100:.2f}%)")
    print(f"    SML Alpha Wedge (Black - SL): {receipt.sml_alpha_wedge_pct:+.2f}% | SML Flattening: {receipt.sml_flattening_pct:.1f}%")
    print("-" * 80)
    print(f"  Zero-Beta Pricing Regime:       {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def stein_internal_capital_command(args: argparse.Namespace) -> None:
    """CLI handler for Jeremy C. Stein (1997) Internal Capital Markets & Winner-Picking Engine."""
    import json
    from src.analytics.stein_internal_capital import (
        calculate_stein_internal_capital,
        persist_stein_internal_capital_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    th1 = float(getattr(args, "theta1", 8.0) if getattr(args, "theta1", None) is not None else 8.0)
    th2 = float(getattr(args, "theta2", 4.0) if getattr(args, "theta2", None) is not None else 4.0)
    c1 = float(getattr(args, "c1", 5.0) if getattr(args, "c1", None) is not None else 5.0)
    c2 = float(getattr(args, "c2", 5.0) if getattr(args, "c2", None) is not None else 5.0)
    lam = float(getattr(args, "agency_lambda", 0.20) if getattr(args, "agency_lambda", None) is not None else 0.20)

    try:
        receipt = calculate_stein_internal_capital(
            div1_productivity_theta1=th1,
            div2_productivity_theta2=th2,
            div1_standalone_budget_c1=c1,
            div2_standalone_budget_c2=c2,
            agency_subsidy_lambda=lam,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_stein_internal_capital_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Stein internal capital evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Stein (1997) Internal Capital Markets & Winner-Picking Engine")
    print("=" * 80)
    print(f"  Conglomerate / Ticker:          {receipt.ticker} (Budget Constrained: {receipt.budget_constrained})")
    print(f"  Productivities (th1 / th2):     {receipt.div1_productivity_theta1:.2f} | {receipt.div2_productivity_theta2:.2f} (Standalone C: ${receipt.div1_standalone_budget_c1:.2f} + ${receipt.div2_standalone_budget_c2:.2f})")
    print(f"  Socialism Friction lambda:      {receipt.agency_subsidy_lambda * 100:.1f}% (Total Capital: ${receipt.total_available_capital:.2f} vs FB ${receipt.first_best_capacity_sum:.2f})")
    print("-" * 80)
    print("  Internal Capital Allocation & Conglomerate Value:")
    print(f"    Winner-Pick (K1* / K2*):      ${receipt.winner_pick_k1:.2f} | ${receipt.winner_pick_k2:.2f} (Equal MP={receipt.marginal_product_div1:.3f})")
    print(f"    Actual HQ Alloc (K1 / K2):    ${receipt.actual_alloc_k1:.2f} | ${receipt.actual_alloc_k2:.2f}")
    print(f"    Standalone vs Conglom Value:  ${receipt.standalone_total_value:.2f} -> ${receipt.actual_conglomerate_value:.2f}")
    print(f"    Conglomerate Premium (Net):   ${receipt.net_conglomerate_premium:+.2f} ({receipt.conglomerate_premium_pct:+.2f}%)")
    print(f"    Winner Surplus / Agency Loss: +${receipt.gross_winner_picking_surplus:.2f} | -${receipt.agency_socialism_loss:.2f}")
    print("-" * 80)
    print(f"  Internal Capital Regime:        {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def stambaugh_bias_command(args: argparse.Namespace) -> None:
    """CLI handler for Stambaugh (1999) Predictive Regressions & Finite-Sample Bias Engine."""
    import json
    from src.analytics.stambaugh_predictive_bias import (
        calculate_stambaugh_bias,
        persist_stambaugh_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    ols_b = float(getattr(args, "ols_beta", 0.08) if getattr(args, "ols_beta", None) is not None else 0.08)
    rho = float(getattr(args, "rho", 0.96) if getattr(args, "rho", None) is not None else 0.96)
    gamma = float(getattr(args, "gamma", -0.80) if getattr(args, "gamma", None) is not None else -0.80)
    sample_t = int(getattr(args, "sample_t", 40) if getattr(args, "sample_t", None) is not None else 40)
    se_b = float(getattr(args, "std_err", 0.035) if getattr(args, "std_err", None) is not None else 0.035)

    try:
        receipt = calculate_stambaugh_bias(
            ols_beta=ols_b,
            ar1_rho=rho,
            gamma_cov_ratio=gamma,
            sample_size_t=sample_t,
            ols_std_err=se_b,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_stambaugh_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Stambaugh predictive bias evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Stambaugh (1999) Predictive Regressions & Finite-Sample Bias Engine")
    print("=" * 80)
    print(f"  Asset / Signal Ticker:          {receipt.ticker} (Sample Horizon T: {receipt.sample_size_t})")
    print(f"  Predictor Persistence rho (Adj):{receipt.ar1_rho:.4f} (Kendall Adj: {receipt.kendall_adjusted_rho:.4f})")
    print(f"  Shock Cov Ratio gamma = Cuv/Vv: {receipt.gamma_cov_ratio:+.4f} (OLS SE: {receipt.ols_std_err:.4f})")
    print("-" * 80)
    print("  Finite-Sample Predictive Bias Decomposition:")
    print(f"    Raw OLS Predictive Slope:     {receipt.ols_beta:+.6f} (OLS t-stat: {receipt.ols_t_stat:+.2f})")
    print(f"    Stambaugh Finite-Sample Bias: {receipt.stambaugh_bias:+.6f} ({receipt.bias_share_pct:.1f}% of OLS slope)")
    print(f"    Stambaugh Bias-Adjusted Slope:{receipt.bias_adjusted_beta:+.6f} (Adj t-stat: {receipt.adjusted_t_stat:+.2f})")
    print(f"    True Signal Retention Ratio:  {receipt.signal_retention_ratio * 100:.1f}%")
    print("-" * 80)
    print(f"  Predictability Durability:      {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def amihud_mendelson_command(args: argparse.Namespace) -> None:
    """CLI handler for Amihud & Mendelson (1986) Bid-Ask Spread & Clientele Engine."""
    import json
    from src.analytics.amihud_mendelson_clientele import (
        calculate_amihud_mendelson_clientele,
        persist_amihud_mendelson_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    spread = float(getattr(args, "spread", 0.015) if getattr(args, "spread", None) is not None else 0.015)
    base_r0 = float(getattr(args, "base_r0", 0.06) if getattr(args, "base_r0", None) is not None else 0.06)
    turnover = float(getattr(args, "turnover", 1.0) if getattr(args, "turnover", None) is not None else 1.0)

    try:
        receipt = calculate_amihud_mendelson_clientele(
            proportional_spread_s=spread,
            base_liquid_return_r0=base_r0,
            investor_turnover_mu=turnover,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_amihud_mendelson_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Amihud & Mendelson evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Amihud & Mendelson (1986) Bid-Ask Spread & Clientele Engine")
    print("=" * 80)
    print(f"  Asset / Ticker:                 {receipt.ticker} (Spread: {receipt.spread_bps:.1f} bps = {receipt.proportional_spread_s * 100:.2f}%)")
    print(f"  Base Return R_0 / Turnover mu:  {receipt.base_liquid_return_r0 * 100:.2f}% | {receipt.investor_turnover_mu:.2f}/yr ({receipt.investor_holding_years:.2f}y hold)")
    print(f"  Equilibrium Clientele Cohort:   {receipt.equilibrium_clientele_tier} (Opt Hold: {receipt.optimal_clientele_holding_years:.2f}y)")
    print("-" * 80)
    print("  Concave Spread-Return Pricing & Clientele Net Yield:")
    print(f"    Gross Expected Return R(S):   {receipt.gross_expected_return * 100:.2f}% / yr")
    print(f"    Illiquidity Return Premium:   +{receipt.illiquidity_premium_bps:.2f} bps")
    print(f"    Investor Spread-Net Return:   {receipt.investor_net_return * 100:.2f}% (Alpha vs R0: {receipt.clientele_net_alpha_bps:+.1f} bps)")
    print(f"    Spread-Compression Uplift:    +{receipt.spread_compression_upside_pct:.2f}% Valuation Re-rating")
    print("-" * 80)
    print(f"  Clientele Alignment Regime:     {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def black76_command(args: argparse.Namespace) -> None:
    """CLI handler for Fischer Black (1976) Futures & Commodity Option Pricing Engine."""
    import json
    from src.analytics.black76_futures_options import (
        calculate_black76_option,
        persist_black76_record,
    )

    try:
        ticker = validate_ticker(args.ticker)
    except ValueError as e:
        print(f"\nError: {e}\n")
        return

    fut_f = float(getattr(args, "futures", 100.0) if getattr(args, "futures", None) is not None else 100.0)
    strike = float(getattr(args, "strike", 100.0) if getattr(args, "strike", None) is not None else 100.0)
    maturity = float(getattr(args, "maturity", 1.0) if getattr(args, "maturity", None) is not None else 1.0)
    rate_r = float(getattr(args, "rate", 0.05) if getattr(args, "rate", None) is not None else 0.05)
    vol = float(getattr(args, "vol", 0.25) if getattr(args, "vol", None) is not None else 0.25)
    opt_type = str(getattr(args, "type", "CALL") or "CALL").strip().upper()

    try:
        receipt = calculate_black76_option(
            futures_price_f=fut_f,
            strike_k=strike,
            maturity_years=maturity,
            risk_free_rate_r=rate_r,
            volatility=vol,
            option_type=opt_type,
            ticker=ticker,
        )
        db_path = getattr(args, "db_path", "data/nextomni_stock.db")
        persist_black76_record(receipt, db_path=db_path)
    except Exception as ex:
        print(f"\nError: Black-76 futures option evaluation failed: {ex}\n")
        return

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print("  NextOmni Stock: Fischer Black (1976) Futures & Commodity Option Pricing Engine")
    print("=" * 80)
    print(f"  Contract / Ticker:              {receipt.ticker} (Futures {receipt.option_type})")
    print(f"  Futures F / Strike K / Expiry:  ${receipt.futures_price_f:.2f} | ${receipt.strike_k:.2f} | {receipt.maturity_years:.2f}y")
    print(f"  Vol sigma / Rate r / Discount:  {receipt.volatility * 100:.2f}% | {receipt.risk_free_rate_r * 100:.2f}% | df={receipt.discount_factor:.4f}")
    print("-" * 80)
    print("  Black-76 Forward Option Valuation & Futures Greeks:")
    print(f"    Option Value ({receipt.option_type}):        ${receipt.option_price:.4f} (Intrinsic: ${receipt.discounted_intrinsic:.4f}, Time: ${receipt.time_value_premium:.4f})")
    print(f"    Call / Put Parity Pair:       ${receipt.call_price:.4f} | ${receipt.put_price:.4f}")
    print(f"    Futures Delta / Gamma:        {receipt.futures_delta:+.4f} | {receipt.futures_gamma:.5f}")
    print(f"    Vega / Theta / Disc Rho:      {receipt.vega:.4f} | {receipt.theta_annual:+.4f}/yr | {receipt.discount_rho:+.4f}")
    print("-" * 80)
    print(f"  Forward Option Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def merton_perold_command(args: argparse.Namespace) -> None:
    """CLI handler for Merton & Perold (1993) Risk Capital & Allocation Engine."""
    import json
    from src.analytics.merton_perold_risk_capital import (
        calculate_merton_perold_risk_capital,
        persist_merton_perold_record,
    )

    receipt = calculate_merton_perold_risk_capital(
        unit1_asset_v1=args.v1,
        unit1_volatility=args.vol1,
        unit2_asset_v2=args.v2,
        unit2_volatility=args.vol2,
        cross_unit_correlation=args.corr,
        horizon_years=args.horizon,
        risk_free_rate_r=args.rate,
        expected_excess_return=args.excess_return,
        ticker=args.ticker,
    )
    persist_merton_perold_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"MERTON & PEROLD (1993) RISK CAPITAL AUDIT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Unit 1 Exposure / Vol:          ${receipt.unit1_asset_v1:.2f} / {receipt.unit1_volatility * 100:.2f}%")
    print(f"  Unit 2 Exposure / Vol:          ${receipt.unit2_asset_v2:.2f} / {receipt.unit2_volatility * 100:.2f}%")
    print(f"  Correlation (rho) / Horizon:    {receipt.cross_unit_correlation:+.2f} / {receipt.horizon_years:.2f}y")
    print(f"  Standalone RC_1 / RC_2:         ${receipt.standalone_rc_1:.4f} / ${receipt.standalone_rc_2:.4f}")
    print(f"  Firm Portfolio RC_P:            ${receipt.portfolio_rc:.4f} (Vol: {receipt.portfolio_volatility * 100:.2f}%)")
    print(f"  Marginal RC_1 / Unalloc Gap:    ${receipt.marginal_rc_1:.4f} / ${receipt.unallocated_capital_gap:.4f}")
    print(f"  Euler RC_1 / RC_2 (100%):       ${receipt.euler_rc_1:.4f} / ${receipt.euler_rc_2:.4f}")
    print(f"  Diversification Benefit:        ${receipt.diversification_benefit:.4f} ({receipt.diversification_benefit_pct * 100:.2f}%)")
    print(f"  Standalone / Euler RAROC:       {receipt.standalone_raroc * 100:.2f}% / {receipt.euler_raroc * 100:.2f}%")
    print(f"  Institutional Regime:           {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def fama_bliss_command(args: argparse.Namespace) -> None:
    """CLI handler for Fama & Bliss (1987) Forward-Spot Spread & Bond Excess Return Engine."""
    import json
    from src.analytics.fama_bliss_forward_spread import (
        calculate_fama_bliss_forward_spread,
        persist_fama_bliss_record,
    )

    receipt = calculate_fama_bliss_forward_spread(
        spot_y1=args.y1,
        long_yn=args.yn,
        maturity_n=args.maturity,
        prev_maturity_y_nm1=args.y_nm1,
        next_y_nm1=args.next_y1,
        ticker=args.ticker,
    )
    persist_fama_bliss_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"FAMA & BLISS (1987) FORWARD-SPOT AUDIT RECEIPT — {receipt.ticker} ({receipt.maturity_n}y)")
    print("=" * 80)
    print(f"  1y Spot Rate / n-Year Spot:     {receipt.spot_y1 * 100:.2f}% / {receipt.long_yn * 100:.2f}%")
    print(f"  Implied Forward Rate f(n):      {receipt.implied_forward_fn * 100:.2f}%")
    print(f"  Forward-Spot Spread s(n):       {receipt.forward_spot_spread * 100:+.2f}% ({receipt.forward_spread_bps:+.1f} bps)")
    print(f"  1y Holding / Excess Return rx:  {receipt.holding_return_hn * 100:+.2f}% / {receipt.excess_bond_return_rx * 100:+.2f}%")
    print(f"  Scaled Future Yield Shift:      {receipt.scaled_yield_change * 100:+.2f}% (Res: {receipt.identity_residual:.2e})")
    print(f"  EH Breakeven Yield Shift:       {receipt.eh_breakeven_yield_shift * 100:+.2f}%/yr")
    print(f"  Term Premium Regime:            {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def duffie_singleton_command(args: argparse.Namespace) -> None:
    """CLI handler for Duffie & Singleton (1999) Reduced-Form Credit Spread Engine."""
    import json
    from src.analytics.duffie_singleton_credit import (
        calculate_duffie_singleton_credit,
        persist_duffie_singleton_record,
    )

    receipt = calculate_duffie_singleton_credit(
        hazard_rate_h=args.hazard,
        recovery_rate_omega=args.recovery,
        risk_free_rate_r=args.rate,
        maturity_years=args.maturity,
        notional=args.notional,
        ticker=args.ticker,
    )
    persist_duffie_singleton_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"DUFFIE & SINGLETON (1999) REDUCED-FORM CREDIT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Hazard h / Recovery omega:      {receipt.hazard_rate_h * 100:.2f}% / {receipt.recovery_rate_omega * 100:.1f}%")
    print(f"  Credit Spread (h * LGD):        {receipt.credit_spread_s * 100:.2f}% ({receipt.credit_spread_bps:.1f} bps)")
    print(f"  Risk-Free / Adjusted Rate R:    {receipt.risk_free_rate_r * 100:.2f}% / {receipt.adjusted_short_rate * 100:.2f}%")
    print(f"  RMV Price / RT Price:           ${receipt.rmv_bond_price:.4f} / ${receipt.rt_bond_price:.4f}")
    print(f"  Risk-Free Value / CVA:          ${receipt.risk_free_bond_price:.4f} / ${receipt.cva_discount:.4f}")
    print(f"  Survival / Default Prob ({receipt.maturity_years:.1f}y): {receipt.survival_probability * 100:.2f}% / {receipt.default_probability * 100:.2f}%")
    print(f"  Credit Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def fundamental_law_command(args: argparse.Namespace) -> None:
    """CLI handler for Grinold (1989) & Clarke-de Silva-Thorley (2002) Fundamental Law."""
    import json
    from src.analytics.fundamental_law_active import (
        calculate_fundamental_law,
        persist_fundamental_law_record,
    )

    receipt = calculate_fundamental_law(
        information_coefficient_ic=args.ic,
        breadth_br=args.breadth,
        transfer_coefficient_tc=args.tc,
        tracking_error_te=args.te,
        ticker=args.ticker,
    )
    persist_fundamental_law_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"FUNDAMENTAL LAW OF ACTIVE MANAGEMENT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  IC / Breadth (BR):              {receipt.information_coefficient_ic:+.4f} / {receipt.breadth_br:.1f}")
    print(f"  Transfer Coefficient (TC):      {receipt.transfer_coefficient_tc:.2f} (Eff BR: {receipt.effective_breadth:.1f}, -{receipt.breadth_loss_pct * 100:.1f}%)")
    print(f"  Unconstrained / Constrained IR: {receipt.unconstrained_ir:.3f} / {receipt.constrained_ir:.3f}")
    print(f"  Tracking Error (TE):            {receipt.tracking_error_te * 100:.2f}%")
    print(f"  Constrained Alpha / Drag:       {receipt.constrained_alpha * 100:+.2f}% / {receipt.constraint_alpha_drag * 100:+.2f}%")
    print(f"  Active Skill Regime:            {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def jorion_bayes_stein_command(args: argparse.Namespace) -> None:
    """CLI handler for Philippe Jorion (1986) Bayes-Stein Shrinkage Engine."""
    import json
    from src.analytics.jorion_bayes_stein import (
        build_synthetic_asset_moments,
        calculate_jorion_bayes_stein,
        persist_jorion_bayes_stein_record,
    )

    means, cov = build_synthetic_asset_moments(
        n_assets=args.n_assets,
        mean_dispersion=args.disp,
        base_vol=args.vol,
    )
    receipt = calculate_jorion_bayes_stein(
        sample_means=means,
        sample_cov=cov,
        sample_size_t=args.sample_t,
        risk_free_rate_r=args.rf,
        ticker=args.ticker,
    )
    persist_jorion_bayes_stein_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"JORION (1986) BAYES-STEIN SHRINKAGE AUDIT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Assets N / Sample Window T:     {receipt.n_assets} / {receipt.sample_size_t}")
    print(f"  GMV Grand Mean (mu_0) / d^2:    {receipt.gmv_grand_mean_mu0 * 100:.2f}% / {receipt.mahalanobis_d2:.4f}")
    print(f"  Shrinkage Intensity (w_hat):    {receipt.shrinkage_intensity_w * 100:.2f}% (lambda: {receipt.lambda_hyperparam:.2f})")
    print(f"  Mean Dispersion (Sample -> BS): {receipt.mean_dispersion_sample * 100:.2f}% -> {receipt.mean_dispersion_bs * 100:.2f}%")
    print(f"  MLE -> BS Tangency Leverage:    {receipt.mle_gross_leverage:.3f}x -> {receipt.bs_gross_leverage:.3f}x (-{receipt.leverage_reduction_pct * 100:.1f}%)")
    print(f"  Predictive Cov Inflation:       {receipt.cov_inflation_ratio:.4f}x")
    print(f"  Estimation Regime:              {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def modigliani_m2_command(args: argparse.Namespace) -> None:
    """CLI handler for Modigliani & Modigliani (1997) M^2 Risk-Adjusted Performance."""
    import json
    from src.analytics.modigliani_m2_rap import (
        calculate_modigliani_m2,
        persist_modigliani_m2_record,
    )

    receipt = calculate_modigliani_m2(
        portfolio_return_rp=args.rp,
        portfolio_vol_p=args.vol_p,
        benchmark_return_rm=args.rm,
        benchmark_vol_m=args.vol_m,
        risk_free_rate_r=args.rf,
        portfolio_beta=args.beta,
        ticker=args.ticker,
    )
    persist_modigliani_m2_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"MODIGLIANI & MODIGLIANI (1997) M^2 RAP AUDIT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Portfolio Return / Vol:         {receipt.portfolio_return_rp * 100:.2f}% / {receipt.portfolio_vol_p * 100:.2f}% (SR: {receipt.portfolio_sharpe:+.3f})")
    print(f"  Benchmark Return / Vol:         {receipt.benchmark_return_rm * 100:.2f}% / {receipt.benchmark_vol_m * 100:.2f}% (SR: {receipt.benchmark_sharpe:+.3f})")
    print(f"  Risk-Match Leverage / Cash:     {receipt.leverage_ratio:.3f}x / {receipt.cash_or_margin_weight * 100:+.2f}%")
    print(f"  M^2 Return / Alpha vs Rm:       {receipt.m2_return * 100:.2f}% / {receipt.m2_alpha_bps:+.1f} bps")
    print(f"  Jensen Alpha / Appraisal AR:    {receipt.jensen_alpha * 100:+.2f}% / {receipt.appraisal_ratio:+.3f}")
    print(f"  Treynor-Black Combined Sharpe:  {receipt.treynor_black_sharpe:.3f}")
    print(f"  Performance Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def malkiel_duration_command(args: argparse.Namespace) -> None:
    """CLI handler for Macaulay (1938) & Malkiel (1962) Bond Duration & Convexity Engine."""
    import json
    from src.analytics.malkiel_bond_duration import (
        calculate_malkiel_bond_duration,
        persist_malkiel_duration_record,
    )

    receipt = calculate_malkiel_bond_duration(
        coupon_rate=args.coupon,
        yield_to_maturity=args.ytm,
        maturity_years=args.maturity,
        compounding_freq=args.freq,
        shock_bps=args.shock_bps,
        par_value=args.par,
        ticker=args.ticker,
    )
    persist_malkiel_duration_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"MACAULAY & MALKIEL (1962) BOND DURATION AUDIT RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Par / Coupon / YTM / Maturity:  ${receipt.par_value:.2f} / {receipt.coupon_rate * 100:.2f}% / {receipt.yield_to_maturity * 100:.2f}% / {receipt.maturity_years:.1f}y ({receipt.compounding_freq}x)")
    print(f"  Bond Clean Price / Status:      ${receipt.bond_price:.4f} ({receipt.par_status})")
    print(f"  Macaulay / Modified Duration:   {receipt.macaulay_duration:.4f}y / {receipt.modified_duration:.4f}y (DV01: ${receipt.dv01:.4f})")
    print(f"  Exact Analytical Convexity:     {receipt.convexity:.4f}")
    print(f"  Shock (+/-{receipt.shock_bps:.0f} bps) Cut/Hike:   +${receipt.exact_gain_on_cut:.4f} / -${receipt.exact_loss_on_hike:.4f}")
    print(f"  Malkiel Convexity Asymmetry:    +${receipt.malkiel_convexity_asymmetry:.4f}")
    print(f"  Duration Regime:                {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def brown_goetzmann_command(args: argparse.Namespace) -> None:
    """CLI handler for BGIR (1992) Survivorship Bias & Attrition Engine."""
    import json
    from src.analytics.brown_goetzmann_survivorship import (
        calculate_survivorship_bias,
        persist_brown_goetzmann_record,
    )

    receipt = calculate_survivorship_bias(
        true_universe_mean=args.mu,
        true_universe_vol=args.vol,
        liquidation_cutoff_c=args.cutoff,
        horizon_years=args.horizon,
        risk_free_rate_r=args.rf,
        ticker=args.ticker,
    )
    persist_brown_goetzmann_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"BROWN, GOETZMANN, IBBOTSON & ROSS (1992) SURVIVORSHIP RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  True Universe Mean / Vol:       {receipt.true_universe_mean * 100:.2f}% / {receipt.true_universe_vol * 100:.2f}% (SR: {receipt.true_universe_sharpe:+.3f})")
    print(f"  Cutoff c / Annual Surv Rate:    {receipt.liquidation_cutoff_c * 100:+.2f}% / {receipt.annual_survival_prob * 100:.2f}% ({receipt.horizon_years:.0f}y Surv: {receipt.multiyear_survival_prob * 100:.2f}%)")
    print(f"  Survivorship Return Bias:       +{receipt.survivorship_return_bias * 100:.2f}% (+{receipt.survivorship_bias_bps:.1f} bps)")
    print(f"  Observed Survivor Mean / Vol:   {receipt.observed_survivor_mean * 100:.2f}% / {receipt.observed_survivor_vol * 100:.2f}%")
    print(f"  Apparent Sharpe / Inflation:    {receipt.apparent_survivor_sharpe:+.3f} (+{receipt.sharpe_bias_delta:.3f})")
    print(f"  Attrition Regime:               {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def sharpe_style_command(args: argparse.Namespace) -> None:
    """CLI handler for William F. Sharpe (1992) Returns-Based Style Analysis."""
    import json
    from src.analytics.sharpe_style_analysis import (
        build_synthetic_style_returns,
        calculate_sharpe_style_analysis,
        persist_sharpe_style_record,
    )

    raw_w = [args.w_value, args.w_growth, args.w_small, args.w_bond]
    total_w = sum(raw_w)
    if total_w <= 0.0:
        raise ValueError("Sum of style weights must be strictly positive.")
    norm_w = [w / total_w for w in raw_w]

    rp, factors, names = build_synthetic_style_returns(
        weights=norm_w,
        n_obs=args.n_obs,
        alpha_ann=args.alpha_ann,
        selection_vol_ann=args.selection_vol,
    )
    receipt = calculate_sharpe_style_analysis(
        portfolio_returns=rp,
        style_returns=factors,
        factor_names=names,
        ticker=args.ticker,
    )
    persist_sharpe_style_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"SHARPE (1992) RETURNS-BASED STYLE ANALYSIS RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Observations / Style Factors:   {receipt.n_obs} / {receipt.n_factors}")
    print(f"  Style R^2 / Selection Share:    {receipt.style_r2 * 100:.2f}% / {receipt.selection_var_share * 100:.2f}%")
    print(f"  Dominant Style Exposure:        {receipt.dominant_style} ({receipt.dominant_weight * 100:.1f}%)")
    print(f"  Style Allocation Weights:       " + ", ".join(f"{k}={v*100:.1f}%" for k, v in receipt.style_weights.items()))
    print(f"  Selection Alpha / TE / IR:      {receipt.selection_alpha_ann * 100:+.2f}% / {receipt.selection_te_ann * 100:.2f}% / {receipt.selection_ir:+.3f}")
    print(f"  Style Regime:                   {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def market_timing_command(args: argparse.Namespace) -> None:
    """CLI handler for Treynor-Mazuy (1966) & Henriksson-Merton (1981) Market Timing."""
    import json
    from src.analytics.market_timing_convexity import (
        build_synthetic_timing_returns,
        calculate_market_timing,
        persist_market_timing_record,
    )

    rp, rm = build_synthetic_timing_returns(
        beta_down=args.beta_down,
        gamma_hm=args.gamma_hm,
        alpha_ann=args.alpha_ann,
        n_obs=args.n_obs,
        noise_vol_ann=args.noise_vol,
    )
    receipt = calculate_market_timing(
        portfolio_returns=rp,
        market_returns=rm,
        risk_free_annual=args.rf,
        ticker=args.ticker,
    )
    persist_market_timing_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"TREYNOR-MAZUY & HENRIKSSON-MERTON MARKET TIMING RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Observations / Risk-Free Rate:  {receipt.n_obs} / {receipt.risk_free_annual * 100:.2f}%")
    print(f"  HM Bear / Bull Beta:            {receipt.hm_beta_down:.3f} / {receipt.hm_beta_up:.3f} (Spread: {receipt.bull_bear_beta_spread:+.3f})")
    print(f"  HM Option Timing gamma / t:     {receipt.hm_gamma:+.4f} / {receipt.hm_gamma_tstat:+.2f} (R^2: {receipt.hm_r2 * 100:.1f}%)")
    print(f"  TM Quadratic gamma / t:         {receipt.tm_gamma:+.3f} / {receipt.tm_gamma_tstat:+.2f} (R^2: {receipt.tm_r2 * 100:.1f}%)")
    print(f"  Annualized Selection Alpha:     TM {receipt.tm_alpha_ann * 100:+.2f}% / HM {receipt.hm_alpha_ann * 100:+.2f}%")
    print(f"  Timing Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def kelly_growth_command(args: argparse.Namespace) -> None:
    """CLI handler for Kelly (1956) & Thorp (1997) Fractional Kelly Log-Wealth Growth."""
    import json
    from src.analytics.kelly_growth_criterion import (
        calculate_kelly_growth,
        persist_kelly_growth_record,
    )

    receipt = calculate_kelly_growth(
        expected_return_annual=args.mu,
        volatility_annual=args.vol,
        risk_free_annual=args.rf,
        kelly_fraction=args.kelly_fraction,
        drawdown_threshold=args.drawdown_threshold,
        ticker=args.ticker,
    )
    persist_kelly_growth_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"KELLY (1956) & THORP (1997) FRACTIONAL KELLY GROWTH RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Drift mu / Vol sigma / Rf:      {receipt.expected_return_annual * 100:.2f}% / {receipt.volatility_annual * 100:.2f}% / {receipt.risk_free_annual * 100:.2f}% (SR {receipt.sharpe_ratio:+.2f})")
    print(f"  Full Kelly f* / Applied f:      {receipt.full_kelly_fraction * 100:.2f}% / {receipt.applied_fraction * 100:.2f}% (k={receipt.kelly_multiplier:.2f}x)")
    print(f"  Half-Kelly / Ruin Boundary:     {receipt.half_kelly_fraction * 100:.2f}% / {receipt.double_kelly_ruin_fraction * 100:.2f}%")
    print(f"  Applied / Optimal Log-Growth:   {receipt.applied_growth_ann * 100:+.2f}% / {receipt.full_kelly_growth_ann * 100:+.2f}% (Eff {receipt.growth_efficiency_ratio * 100:.1f}%)")
    print(f"  Variance Reduction / P(DD):     {receipt.variance_reduction_ratio * 100:.1f}% / {receipt.drawdown_exceedance_prob * 100:.2f}% (at {receipt.drawdown_threshold * 100:.0f}% DD)")
    print(f"  Growth Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def lo_mackinlay_vr_command(args: argparse.Namespace) -> None:
    """CLI handler for Lo & MacKinlay (1988) Heteroskedasticity-Robust Variance Ratio."""
    import json
    from src.analytics.lo_mackinlay_variance_ratio import (
        build_synthetic_vr_prices,
        calculate_lo_mackinlay_variance_ratio,
        persist_variance_ratio_record,
    )

    prices = build_synthetic_vr_prices(
        rho=args.rho,
        n_obs=args.n_obs,
        vol_daily=args.vol_daily,
        heteroskedastic=args.heteroskedastic,
    )
    receipt = calculate_lo_mackinlay_variance_ratio(
        prices=prices,
        lag_q=args.lag,
        ticker=args.ticker,
    )
    persist_variance_ratio_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"LO & MACKINLAY (1988) VARIANCE RATIO RECEIPT — {receipt.ticker} (q={receipt.lag_q})")
    print("=" * 80)
    print(f"  Returns N / Lag Horizon q:      {receipt.n_returns} / {receipt.lag_q}")
    print(f"  Variance Ratio VR(q) / Excess:  {receipt.variance_ratio:.4f} / {receipt.excess_variance_ratio:+.4f}")
    print(f"  Homoskedastic z(q) / p-val:     {receipt.z_stat_homoskedastic:+.3f} / {receipt.p_value_homoskedastic:.4f}")
    print(f"  Robust z*(q) / p-val (ARCH):    {receipt.z_stat_heteroskedastic:+.3f} / {receipt.p_value_heteroskedastic:.4f}")
    print(f"  Phi(q) / Phi*(q):               {receipt.phi_homoskedastic:.6f} / {receipt.phi_heteroskedastic:.6f}")
    print(f"  Random-Walk Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def grs_test_command(args: argparse.Namespace) -> None:
    """CLI handler for Gibbons, Ross & Shanken (1989) Multivariate GRS F-Test."""
    import json
    from src.analytics.gibbons_ross_shanken import (
        build_synthetic_grs_returns,
        calculate_grs_test,
        persist_grs_record,
    )

    assets, factors = build_synthetic_grs_returns(
        n_obs=args.n_obs,
        n_assets=args.n_assets,
        n_factors=args.n_factors,
        alpha_ann_level=args.alpha_ann,
        idio_vol_ann=args.idio_vol,
    )
    receipt = calculate_grs_test(assets, factors, ticker=args.ticker)
    persist_grs_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"GIBBONS, ROSS & SHANKEN (1989) GRS MULTIVARIATE EFFICIENCY RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Panel Shape (T / N / K):        {receipt.n_obs} / {receipt.n_assets} / {receipt.n_factors} (df: {receipt.df_num}, {receipt.df_den})")
    print(f"  GRS F-Statistic / p-value:      {receipt.grs_f_stat:.4f} / {receipt.grs_p_value:.6f}")
    print(f"  Factor / Combined Ann. Sharpe:  {receipt.factor_sharpe_ann:.3f} / {receipt.combined_sharpe_ann:.3f}")
    print(f"  Daily Sharpe^2 Spanning Gap:    {receipt.sharpe_spanning_gap_sq:.6f} (Mean |alpha| ann: {receipt.mean_abs_alpha_ann * 100:.2f}%)")
    print(f"  Efficiency Regime:              {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def jobson_korkie_command(args: argparse.Namespace) -> None:
    """CLI handler for Jobson & Korkie (1981) & Memmel (2003) Paired Sharpe Equality Test."""
    import json
    from src.analytics.jobson_korkie_memmel import (
        build_synthetic_paired_returns,
        calculate_jobson_korkie_memmel,
        persist_sharpe_comparison_record,
    )

    ra, rb = build_synthetic_paired_returns(
        n_obs=args.n_obs,
        sharpe_a_ann=args.sr_a,
        sharpe_b_ann=args.sr_b,
        correlation=args.rho,
        risk_free_annual=args.rf,
    )
    receipt = calculate_jobson_korkie_memmel(
        returns_a=ra,
        returns_b=rb,
        risk_free_annual=args.rf,
        ticker_a=args.ticker_a,
        ticker_b=args.ticker_b,
    )
    persist_sharpe_comparison_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"JOBSON & KORKIE (1981) / MEMMEL (2003) SHARPE EQUALITY — {receipt.ticker_a} vs {receipt.ticker_b}")
    print("=" * 80)
    print(f"  Sessions N / Risk-Free Rate:    {receipt.n_obs} / {receipt.risk_free_annual * 100:.2f}%")
    print(f"  Ann. Sharpe A / B / Diff:       {receipt.sharpe_a_ann:+.3f} / {receipt.sharpe_b_ann:+.3f} ({receipt.sharpe_diff_ann:+.3f})")
    print(f"  Correlation rho:                {receipt.correlation:.4f}")
    print(f"  Memmel z_JKM / p-value:         {receipt.z_stat:+.4f} / {receipt.p_value:.6f}")
    print(f"  95% CI Ann. Sharpe Diff:        [{receipt.ci_95_low_ann:+.3f}, {receipt.ci_95_high_ann:+.3f}]")
    print(f"  Equality Verdict:               {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def hurst_exponent_command(args: argparse.Namespace) -> None:
    """CLI handler for Hurst (1951) & Peters (1994) Rescaled Range (R/S) Engine."""
    import json
    from src.analytics.hurst_fractal_persistence import (
        build_synthetic_fractional_prices,
        calculate_hurst_exponent,
        persist_hurst_record,
    )

    prices = build_synthetic_fractional_prices(
        hurst_target=args.hurst_target,
        n_obs=args.n_obs,
        vol_daily=args.vol_daily,
    )
    receipt = calculate_hurst_exponent(
        prices=prices,
        min_window=args.min_window,
        ticker=args.ticker,
    )
    persist_hurst_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"HURST (1951) & PETERS (1994) R/S FRACTAL PERSISTENCE RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Returns N / Scaling Windows:    {receipt.n_returns} / {receipt.n_windows}")
    print(f"  Classical R/S H / RW Bias:      {receipt.hurst_classical:.4f} / {receipt.hurst_random_walk_bias:.4f}")
    print(f"  Anis-Lloyd Corrected H_adj:     {receipt.hurst_corrected:.4f} (R^2: {receipt.scaling_r2 * 100:.2f}%)")
    print(f"  Fractal Dimension D / Corr C:   {receipt.fractal_dimension:.4f} / {receipt.implied_correlation:+.4f}")
    print(f"  Memory Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def max_diversification_command(args: argparse.Namespace) -> None:
    """CLI handler for Choueifaty & Coignard (2008) Maximum Diversification Engine."""
    import json
    from src.analytics.choueifaty_diversification import (
        build_synthetic_correlation_cov,
        calculate_maximum_diversification,
        persist_diversification_record,
    )

    cov = build_synthetic_correlation_cov(
        n_assets=args.n_assets,
        base_corr=args.corr,
    )
    receipt = calculate_maximum_diversification(cov, ticker=args.ticker)
    persist_diversification_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"CHOUEIFATY & COIGNARD (2008) MAXIMUM DIVERSIFICATION RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Assets / Active Holdings:       {receipt.n_assets} / {receipt.n_active_holdings} (Max Wt {receipt.max_weight * 100:.1f}%)")
    print(f"  Diversification Ratio DR(MDP):  {receipt.dr_mdp:.4f} (EW: {receipt.dr_equal_weight:.4f}, GMV: {receipt.dr_min_variance:.4f})")
    print(f"  Effective Independent Bets:     {receipt.effective_independent_bets:.3f} (Benefit: {receipt.diversification_benefit_pct * 100:.1f}%)")
    print(f"  MDP Vol / EW Vol / Spread:      {receipt.mdp_volatility * 100:.2f}% / {receipt.equal_weight_volatility * 100:.2f}% / {receipt.active_correlation_spread:.6f}")
    print(f"  Diversification Regime:         {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def brinson_attribution_command(args: argparse.Namespace) -> None:
    """CLI handler for Brinson, Hood & Beebower (1986) / Fachler Attribution."""
    import json
    from src.analytics.brinson_attribution import (
        calculate_brinson_attribution,
        persist_brinson_record,
    )

    wb = [0.35, 0.30, 0.20, 0.15]
    wp = [0.45, 0.25, 0.20, 0.10]
    rb = [0.12, 0.08, 0.04, 0.09]
    rp = [rb[0] + args.active_alpha, rb[1] + 0.005, rb[2] - 0.005, rb[3] + 0.010]
    sectors = ["Tech_Growth", "Core_Equity", "Defensive_Val", "Quality_Tilt"]

    receipt = calculate_brinson_attribution(wp, wb, rp, rb, sector_names=sectors, ticker=args.ticker)
    persist_brinson_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"BRINSON, HOOD & BEEBOWER (1986) RETURN ATTRIBUTION RECEIPT — {receipt.ticker}")
    print("=" * 80)
    print(f"  Portfolio / Benchmark Return:   {receipt.portfolio_return * 100:+.2f}% / {receipt.benchmark_return * 100:+.2f}% (Active: {receipt.active_return * 100:+.2f}%)")
    print(f"  Allocation / Selection / Inter: {receipt.allocation_effect * 100:+.2f}% / {receipt.selection_effect * 100:+.2f}% / {receipt.interaction_effect * 100:+.2f}%")
    print(f"  Total Selection / Top Sector:   {receipt.total_selection_effect * 100:+.2f}% / {receipt.top_driver_sector}")
    print(f"  Attribution Regime:             {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def copeland_galai_command(args: argparse.Namespace) -> None:
    """CLI handler for Copeland & Galai (1983) Option-Theoretic Bid-Ask Spread."""
    import json
    from src.analytics.copeland_galai_spread import (
        calculate_copeland_galai_spread,
        persist_copeland_galai_record,
    )

    receipt = calculate_copeland_galai_spread(
        spot_price=args.spot,
        volatility_annual=args.vol,
        informed_prob=args.informed_prob,
        quote_life_years=args.quote_days / 252.0,
        ticker=args.ticker,
    )
    persist_copeland_galai_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"COPELAND & GALAI (1983) BID-ASK SPREAD EQUILIBRIUM — {receipt.ticker}")
    print("=" * 80)
    print(f"  Spot / Ann Vol / Informed pi_I: ${receipt.spot_price:.2f} / {receipt.volatility_annual * 100:.1f}% / {receipt.informed_prob * 100:.1f}%")
    print(f"  Equilibrium Bid / Ask Quotes:   ${receipt.bid_price:.4f} / ${receipt.ask_price:.4f}")
    print(f"  Equilibrium Spread / bps:       ${receipt.equilibrium_spread:.4f} ({receipt.spread_bps:.2f} bps)")
    print(f"  Informed Call / Put Value:      ${receipt.informed_call_value:.4f} / ${receipt.informed_put_value:.4f} (Residual: {receipt.equilibrium_residual:.2e})")
    print(f"  Sensitivity per +1% pi_I:       +{receipt.spread_sensitivity_per_1pct_pi:.2f} bps")
    print(f"  Microstructure Regime:          {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def erc_portfolio_command(args: argparse.Namespace) -> None:
    """CLI handler for Maillard, Roncalli & Teiletche (2010) / Spinu (2013) ERC."""
    import json
    from src.analytics.maillard_erc_portfolio import (
        build_synthetic_erc_cov,
        calculate_erc_portfolio,
        persist_erc_record,
    )

    vols = [0.10 + 0.05 * i for i in range(args.n_assets)]
    cov = build_synthetic_erc_cov(vols=vols, base_corr=args.corr)
    receipt = calculate_erc_portfolio(cov, ticker=args.ticker)
    persist_erc_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(receipt.to_dict(), indent=2))
        return

    print("\n" + "=" * 80)
    print(f"MAILLARD, RONCALLI & TEILETCHE (2010) ERC RISK PARITY — {receipt.ticker}")
    print("=" * 80)
    print(f"  Assets / Effective Risk Bets:   {receipt.n_assets} / {receipt.effective_risk_bets:.4f} (DR: {receipt.diversification_ratio:.3f})")
    print(f"  ERC / EW / InvVol Volatility:   {receipt.erc_volatility * 100:.2f}% / {receipt.equal_weight_volatility * 100:.2f}% / {receipt.inverse_vol_volatility * 100:.2f}%")
    print(f"  Max Risk-Budget Error / HHI:    {receipt.max_risk_budget_error:.2e} / {receipt.weight_herfindahl:.4f}")
    print(f"  Parity Regime:                  {receipt.regime}")
    print(f"  HMAC Signature:                 {receipt.hmac_signature[:16]}... (Sealed)")
    print(f"  Timestamp:                      {receipt.timestamp}")
    print("=" * 80 + "\n")


def var_backtest_command(args: argparse.Namespace) -> None:
    """Run Kupiec (1995) POF & Christoffersen (1998) Conditional Coverage VaR Backtest."""
    import json
    from dataclasses import asdict
    from src.analytics.kupiec_christoffersen_var import (
        build_synthetic_var_series,
        calculate_var_backtest,
        persist_var_backtest_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    alpha = float(getattr(args, "alpha", 0.05))
    n_obs = int(getattr(args, "n_obs", 250))
    mode = str(getattr(args, "mode", "calibrated"))

    returns, var_forecasts = build_synthetic_var_series(
        n_obs=n_obs,
        alpha=alpha,
        mode=mode,
    )
    receipt = calculate_var_backtest(
        returns=returns,
        var_forecasts=var_forecasts,
        alpha=alpha,
        ticker=ticker,
    )
    persist_var_backtest_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🛡️ Kupiec (1995) & Christoffersen (1998) VaR Backtest [{receipt.ticker}]"
        f"\n  N={receipt.n_obs} | α={receipt.alpha_target:.4f} | "
        f"Exceptions={receipt.n_exceptions} (Exp={receipt.expected_exceptions:.2f}) | "
        f"FailRate={receipt.empirical_failure_rate:.4f}"
        f"\n  π₀₁={receipt.pi_01_after_ok:.4f} | π₁₁={receipt.pi_11_after_breach:.4f}"
        f"\n  LR_uc={receipt.lr_uc:.4f} (p={receipt.p_value_uc:.4f}) | "
        f"LR_ind={receipt.lr_ind:.4f} (p={receipt.p_value_ind:.4f}) | "
        f"LR_cc={receipt.lr_cc:.4f} (p={receipt.p_value_cc:.4f})"
        f"\n  Basel Zone={receipt.basel_zone} | Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def bdt_lattice_command(args: argparse.Namespace) -> None:
    """Run Black-Derman-Toy (1990) Lognormal Short-Rate Binomial Lattice Calibration."""
    import json
    from dataclasses import asdict
    from src.analytics.black_derman_toy_lattice import (
        build_synthetic_yield_curve,
        calibrate_bdt_lattice,
        persist_bdt_lattice_record,
    )

    ticker = str(getattr(args, "ticker", "UST")).strip().upper()
    n_steps = int(getattr(args, "steps", 8))
    base_rate = float(getattr(args, "base_rate", 0.04))
    slope_bps = float(getattr(args, "slope_bps", 120.0))
    vol = float(getattr(args, "vol", 0.16))
    call_price = float(getattr(args, "call_price", 100.0))

    curve = build_synthetic_yield_curve(
        n_steps=n_steps,
        base_rate=base_rate,
        slope_bps=slope_bps,
    )
    receipt = calibrate_bdt_lattice(
        spot_rates=curve,
        volatilities=vol,
        dt=1.0,
        coupon_rate=base_rate + 0.01,
        call_price=call_price,
        ticker=ticker,
    )
    persist_bdt_lattice_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🌲 Black-Derman-Toy (1990) Short-Rate Lattice [{receipt.ticker}]"
        f"\n  Steps={receipt.n_steps} (dt={receipt.dt:.2f}y) | r0={receipt.short_rate_initial*100:.2f}% | "
        f"Slope={receipt.curve_slope_bps:+.1f} bps | Vol={receipt.volatility*100:.1f}%"
        f"\n  Terminal Rate Fan: [{receipt.terminal_rate_low*100:.2f}%, "
        f"{receipt.terminal_rate_median*100:.2f}%, {receipt.terminal_rate_high*100:.2f}%]"
        f"\n  Straight Px={receipt.straight_bond_price:.4f} | Callable Px={receipt.callable_bond_price:.4f} | "
        f"Embedded Call={receipt.embedded_call_value:.4f} | MaxCalibErr={receipt.max_calibration_error:.2e}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def nss_curve_command(args: argparse.Namespace) -> None:
    """Run Nelson-Siegel-Svensson (1987/1994) Yield Curve Factor Fitting."""
    import json
    from dataclasses import asdict
    from src.analytics.nelson_siegel_svensson import (
        build_synthetic_nss_curve,
        fit_nelson_siegel_svensson,
        persist_nss_record,
    )

    ticker = str(getattr(args, "ticker", "UST")).strip().upper()
    mode = str(getattr(args, "mode", "normal"))

    maturities, yields = build_synthetic_nss_curve(mode=mode)
    receipt = fit_nelson_siegel_svensson(
        maturities=maturities,
        yields=yields,
        ticker=ticker,
    )
    persist_nss_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n📐 Nelson-Siegel-Svensson (1987/1994) Yield Curve [{receipt.ticker}]"
        f"\n  β₀(Level)={receipt.beta0_level*100:.3f}% | β₁(Slope)={receipt.beta1_slope*100:+.3f}% | "
        f"β₂(Curv1)={receipt.beta2_curvature*100:+.3f}% | β₃(Curv2)={receipt.beta3_second_hump*100:+.3f}%"
        f"\n  τ₁={receipt.tau1:.3f}y | τ₂={receipt.tau2:.3f}y | "
        f"10Y-2Y={receipt.spread_10y_2y_bps:+.2f} bps | RMSE={receipt.rmse_bps:.4f} bps | R²={receipt.r_squared:.6f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def gp_dynamic_command(args: argparse.Namespace) -> None:
    """Run Gârleanu & Pedersen (2013) Dynamic Trading Policy Optimization."""
    import json
    from dataclasses import asdict
    from src.analytics.garleanu_pedersen_trading import (
        build_synthetic_gp_inputs,
        persist_gp_record,
        solve_garleanu_pedersen,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    risk_aversion = float(getattr(args, "risk_aversion", 2.5))
    cost_bps = float(getattr(args, "cost_bps", 15.0))
    fast_rho = float(getattr(args, "fast_rho", 0.35))
    slow_rho = float(getattr(args, "slow_rho", 0.03))

    sigma, b_mat, phi, f_t, x_prev = build_synthetic_gp_inputs(
        n_assets=3,
        fast_rho=fast_rho,
        slow_rho=slow_rho,
    )
    receipt = solve_garleanu_pedersen(
        sigma=sigma,
        b_loadings=b_mat,
        phi_decay=phi,
        f_t=f_t,
        x_prev=x_prev,
        risk_aversion=risk_aversion,
        cost_bps=cost_bps,
        ticker=ticker,
    )
    persist_gp_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🎯 Gârleanu & Pedersen (2013) Dynamic Trading [{receipt.ticker}]"
        f"\n  γ={receipt.risk_aversion:.2f} | Cost={receipt.cost_bps:.1f} bps | "
        f"TradeRate={receipt.mean_trade_rate:.4f} | TurnoverReduction={receipt.turnover_reduction_pct:.1f}%"
        f"\n  AimRatio(Slow/Fast)={receipt.slow_alpha_weight_ratio:.3f} / {receipt.fast_alpha_weight_ratio:.3f} | "
        f"RiccatiRes={receipt.riccati_residual:.2e}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def dcc_garch_command(args: argparse.Namespace) -> None:
    """Run Engle (2002) Dynamic Conditional Correlation (DCC-GARCH(1,1)) Estimation."""
    import json
    from dataclasses import asdict
    from src.analytics.dcc_garch import (
        build_synthetic_dcc_returns,
        fit_dcc_garch,
        persist_dcc_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    n_obs = int(getattr(args, "n_obs", 220))
    regime = str(getattr(args, "regime", "crisis_spike"))

    rets = build_synthetic_dcc_returns(n_obs=n_obs, regime=regime)
    receipt = fit_dcc_garch(returns=rets, ticker=ticker)
    persist_dcc_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🔗 Engle (2002) DCC-GARCH(1,1) [{receipt.ticker}]"
        f"\n  N={receipt.n_obs} | Assets={receipt.n_assets} | "
        f"a={receipt.a_dcc:.4f}, b={receipt.b_dcc:.4f} (a+b={receipt.dcc_persistence:.4f})"
        f"\n  Latest ρ={receipt.latest_corr_mean:.4f} vs Uncond ρ̄={receipt.unconditional_corr_mean:.4f} "
        f"[{receipt.corr_min:.3f}, {receipt.corr_max:.3f}]"
        f"\n  LogLik(DCC)={receipt.dcc_loglik:.2f} vs CCC={receipt.ccc_loglik:.2f} (+{receipt.loglik_gain:.2f})"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def haircut_sharpe_command(args: argparse.Namespace) -> None:
    """Run Harvey & Liu (2015) Multiple-Testing Haircut Sharpe Ratio Analysis."""
    import json
    from dataclasses import asdict
    from src.analytics.harvey_liu_haircut_sharpe import (
        calculate_haircut_sharpe,
        persist_haircut_sharpe_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    sharpe = float(getattr(args, "sharpe", 1.85))
    n_obs = int(getattr(args, "n_obs", 1260))
    n_trials = int(getattr(args, "n_trials", 100))

    receipt = calculate_haircut_sharpe(
        sharpe_observed=sharpe,
        n_obs=n_obs,
        n_trials=n_trials,
        ticker=ticker,
    )
    persist_haircut_sharpe_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n✂️ Harvey & Liu (2015) Haircut Sharpe [{receipt.ticker}]"
        f"\n  N={receipt.n_obs} | M={receipt.n_trials} trials | "
        f"SR_obs={receipt.sharpe_observed:.3f} (t={receipt.t_stat_raw:.2f}, p_raw={receipt.p_value_raw:.4e})"
        f"\n  SR_BH={receipt.sharpe_bh:.3f} | SR_Holm={receipt.sharpe_holm:.3f} | "
        f"SR_Bonf={receipt.sharpe_bonferroni:.3f} | SR_BHY={receipt.sharpe_bhy:.3f} "
        f"(Haircut={receipt.haircut_pct_bhy:.1f}%)"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def lo_sharpe_command(args: argparse.Namespace) -> None:
    """Run Andrew Lo (2002) Autocorrelation-Adjusted Sharpe Ratio Analysis."""
    import json
    from dataclasses import asdict
    from src.analytics.lo_autocorr_sharpe import (
        build_synthetic_ar1_returns,
        calculate_lo_autocorr_sharpe,
        persist_lo_sharpe_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    q_horizon = int(getattr(args, "q", 12))
    rho1 = float(getattr(args, "rho1", 0.35))
    n_obs = int(getattr(args, "n_obs", 300))

    rets = build_synthetic_ar1_returns(n_obs=n_obs, rho1=rho1)
    receipt = calculate_lo_autocorr_sharpe(
        returns=rets,
        q_horizon=q_horizon,
        ticker=ticker,
    )
    persist_lo_sharpe_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n📉 Andrew Lo (2002) Autocorrelation-Adjusted Sharpe [{receipt.ticker}]"
        f"\n  N={receipt.n_obs} | q={receipt.q_horizon} | ρ₁={receipt.lag1_autocorr:+.4f} | "
        f"η(q)={receipt.eta_scaling_factor:.3f}"
        f"\n  SR_naive(q)={receipt.sharpe_naive_q:.3f} vs SR_Lo(q)={receipt.sharpe_lo_q:.3f} "
        f"(Inflation={receipt.inflation_pct:+.1f}%)"
        f"\n  95% HAC CI: [{receipt.ci95_low:.3f}, {receipt.ci95_high:.3f}] (SE={receipt.se_lo_q:.3f})"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def key_rate_immunization_command(args: argparse.Namespace) -> None:
    """Run Ho (1992) Key-Rate Duration & Fisher-Weil (1971) Liability Immunization."""
    import json
    from dataclasses import asdict
    from src.analytics.key_rate_immunization import (
        persist_krd_record,
        solve_key_rate_immunization,
    )

    ticker = str(getattr(args, "ticker", "ALM")).strip().upper()
    liab_mat = float(getattr(args, "liab_maturity", 10.0))
    mode = str(getattr(args, "mode", "ladder"))

    receipt = solve_key_rate_immunization(
        liability_maturity=liab_mat,
        mode=mode,
        ticker=ticker,
    )
    persist_krd_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🛡️ Ho (1992) Key-Rate Duration & Fisher-Weil Immunization [{receipt.ticker}]"
        f"\n  Liab T={receipt.liability_maturity:.1f}y | PV={receipt.liability_pv:.2f} | "
        f"Dur(A/L)={receipt.asset_duration:.3f}/{receipt.liability_duration:.3f} | "
        f"ConvSurplus={receipt.convexity_surplus:+.2f}"
        f"\n  KRD[1Y={receipt.krd_1y:.2f}, 2Y={receipt.krd_2y:.2f}, 5Y={receipt.krd_5y:.2f}, "
        f"10Y={receipt.krd_10y:.2f}, 30Y={receipt.krd_30y:.2f}] | TwistGap={receipt.krd_twist_l2_gap:.3f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def safety_first_command(args: argparse.Namespace) -> None:
    """Run Roy (1952), Kataoka (1963) & Telser (1955) Safety-First Portfolio Selection."""
    import json
    from dataclasses import asdict
    from src.analytics.roy_safety_first import (
        build_synthetic_safety_first_inputs,
        persist_safety_first_record,
        solve_safety_first_portfolio,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    threshold = float(getattr(args, "threshold", -0.10))
    alpha = float(getattr(args, "alpha", 0.05))

    mu, sigma = build_synthetic_safety_first_inputs(n_assets=4)
    receipt = solve_safety_first_portfolio(
        expected_returns=mu,
        cov_matrix=sigma,
        threshold_return=threshold,
        alpha_shortfall=alpha,
        ticker=ticker,
    )
    persist_safety_first_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🛡️ Roy (1952) / Kataoka / Telser Safety-First Portfolio [{receipt.ticker}]"
        f"\n  Threshold R_L={receipt.threshold_return*100:+.2f}% | α={receipt.alpha_shortfall:.3f} | "
        f"Roy SFR={receipt.roy_sfr:.3f} (P_shortfall={receipt.roy_shortfall_prob*100:.2f}%, Cheb={receipt.chebyshev_bound:.3f})"
        f"\n  Kataoka Floor={receipt.kataoka_floor*100:+.2f}% | "
        f"Telser Feasible={receipt.telser_feasible} (μ={receipt.telser_mean*100:+.2f}%, σ={receipt.telser_vol*100:.2f}%)"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def merton_crra_command(args: argparse.Namespace) -> None:
    """Run Samuelson (1969) & Merton (1969/1971) CRRA Lifetime Portfolio & Consumption."""
    import json
    from dataclasses import asdict
    from src.analytics.merton_crra_allocation import (
        calculate_merton_crra_allocation,
        persist_merton_crra_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    mu = float(getattr(args, "mu", 0.09))
    vol = float(getattr(args, "vol", 0.16))
    rf = float(getattr(args, "rf", 0.04))
    gamma = float(getattr(args, "gamma", 3.0))
    rho = float(getattr(args, "rho", 0.03))
    rho_state = float(getattr(args, "rho_state", -0.40))

    receipt = calculate_merton_crra_allocation(
        expected_return=mu,
        volatility=vol,
        risk_free_rate=rf,
        gamma_risk_aversion=gamma,
        time_discount_rate=rho,
        state_correlation=rho_state,
        ticker=ticker,
    )
    persist_merton_crra_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⏳ Samuelson (1969) & Merton (1969/1971) CRRA Portfolio [{receipt.ticker}]"
        f"\n  μ={receipt.expected_return*100:.2f}% | σ={receipt.volatility*100:.2f}% | rf={receipt.risk_free_rate*100:.2f}% | γ={receipt.gamma_risk_aversion:.2f}"
        f"\n  Myopic Share={receipt.merton_share_myopic*100:.2f}% | Hedging Demand={receipt.hedging_demand*100:+.2f}% | Total Weight={receipt.total_risky_weight*100:.2f}%"
        f"\n  Optimal c/W={receipt.consumption_wealth_ratio*100:.2f}% | Growth Rate={receipt.expected_growth_rate*100:+.2f}% | r_CE={receipt.certainty_equivalent_return*100:.2f}%"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def variance_swap_command(args: argparse.Namespace) -> None:
    """Run Carr & Madan (1999) Variance Swap Fair Strike Pricing & VRP Analysis."""
    import json
    from dataclasses import asdict
    from src.analytics.carr_madan_variance_swap import (
        calculate_carr_madan_variance_swap,
        persist_variance_swap_record,
    )

    ticker = str(getattr(args, "ticker", "SPY")).strip().upper()
    spot = float(getattr(args, "spot", 100.0))
    ttm = float(getattr(args, "ttm", 0.25))
    rf = float(getattr(args, "rf", 0.04))
    vol = float(getattr(args, "vol", 0.20))
    skew = float(getattr(args, "skew", -0.15))
    real_vol = float(getattr(args, "realized_vol", 0.17))
    n_strikes = int(getattr(args, "n_strikes", 200))

    receipt = calculate_carr_madan_variance_swap(
        spot_price=spot,
        time_to_maturity=ttm,
        risk_free_rate=rf,
        atm_implied_vol=vol,
        vol_skew_slope=skew,
        realized_vol=real_vol,
        n_strikes=n_strikes,
        ticker=ticker,
    )
    persist_variance_swap_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Carr & Madan (1999) Variance Swap Fair Strike [{receipt.ticker}]"
        f"\n  Spot={receipt.spot_price:.2f} | T={receipt.time_to_maturity:.2f}y | rf={receipt.risk_free_rate*100:.2f}%"
        f"\n  Fair Strike Vol σ_swap={receipt.fair_vol_strike*100:.2f}% | Fair K_var={receipt.fair_variance_strike:.6f}"
        f"\n  Realized Vol={receipt.realized_vol*100:.2f}% | VRP={receipt.vol_risk_premium*100:+.2f}% | Put/Call Ratio={receipt.put_call_variance_ratio:.3f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def kou_jump_command(args: argparse.Namespace) -> None:
    """Run Steven G. Kou (2002) Double Exponential Jump-Diffusion Option Pricing."""
    import json
    from dataclasses import asdict
    from src.analytics.kou_double_exponential_jump import (
        calculate_kou_option_price,
        persist_kou_record,
    )

    receipt = calculate_kou_option_price(
        spot_price=args.spot,
        strike_price=args.strike,
        time_to_maturity=args.ttm,
        risk_free_rate=args.rf,
        diffusion_vol=args.vol,
        jump_intensity=args.lambda_jump,
        prob_up_jump=args.p_up,
        eta1_up=args.eta1,
        eta2_down=args.eta2,
        ticker=args.ticker,
    )
    persist_kou_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Steven G. Kou (2002) Jump-Diffusion Option Pricing [{receipt.ticker}]"
        f"\n  Spot=${receipt.spot_price:.2f} | Strike=${receipt.strike_price:.2f} | T={receipt.time_to_maturity:.2f}y | rf={receipt.risk_free_rate*100:.2f}%"
        f"\n  Kou Call=${receipt.kou_call_price:.4f} | Kou Put=${receipt.kou_put_price:.4f} | BS Call=${receipt.bs_benchmark_call:.4f}"
        f"\n  Jump Premium=${receipt.jump_premium:+.4f} | ATM Skew={receipt.atm_skew*100:+.2f}%"
        f"\n  Jump λ={receipt.jump_intensity:.2f} | p_up={receipt.prob_up_jump*100:.1f}% | η_up={receipt.eta1_up:.1f} | η_down={receipt.eta2_down:.1f}"
        f"\n  Compensated Drift ζ={receipt.compensated_drift:+.4f} | Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def sabr_vol_command(args: argparse.Namespace) -> None:
    """Run Patrick S. Hagan et al. (2002) SABR Stochastic Volatility Smile Evaluation."""
    import json
    from dataclasses import asdict
    from src.analytics.sabr_stochastic_volatility import (
        calculate_sabr_profile,
        persist_sabr_record,
    )

    receipt = calculate_sabr_profile(
        forward_price=args.forward,
        time_to_maturity=args.ttm,
        alpha=args.alpha,
        beta=args.beta,
        rho=args.rho,
        nu=args.nu,
        ticker=args.ticker,
    )
    persist_sabr_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Patrick S. Hagan et al. (2002) SABR Volatility Smile [{receipt.ticker}]"
        f"\n  Forward F={receipt.forward_price:.2f} | T={receipt.time_to_maturity:.2f}y | α={receipt.alpha:.4f} | β={receipt.beta:.2f} | ρ={receipt.rho:+.2f} | ν={receipt.nu:.2f}"
        f"\n  ATM Vol σ_ATM={receipt.atm_vol*100:.2f}% | 10% OTM Put Vol={receipt.otm_put_vol*100:.2f}% | 10% OTM Call Vol={receipt.otm_call_vol*100:.2f}%"
        f"\n  Skew Slope={receipt.skew_slope*100:+.2f}% | Smile Convexity={receipt.smile_convexity*100:+.2f}%"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def dupire_vol_command(args: argparse.Namespace) -> None:
    """Run Bruno Dupire (1994) Non-Parametric Local Volatility Surface Engine."""
    import json
    from dataclasses import asdict
    from src.analytics.dupire_local_volatility import (
        calculate_dupire_local_volatility,
        persist_dupire_record,
    )

    receipt = calculate_dupire_local_volatility(
        spot_price=args.spot,
        strike_price=args.strike,
        time_to_maturity=args.ttm,
        risk_free_rate=args.rf,
        dividend_yield=args.div,
        atm_vol=args.vol,
        skew_slope=args.skew,
        vol_of_vol=args.vol_of_vol,
        ticker=args.ticker,
    )
    persist_dupire_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Bruno Dupire (1994) Local Volatility Surface [{receipt.ticker}]"
        f"\n  Spot=${receipt.spot_price:.2f} | Strike=${receipt.strike_price:.2f} | T={receipt.time_to_maturity:.2f}y | rf={receipt.risk_free_rate*100:.2f}% | q={receipt.dividend_yield*100:.2f}%"
        f"\n  Local Vol σ_loc={receipt.local_vol*100:.2f}% | Implied Vol σ_imp={receipt.implied_vol*100:.2f}% | Ratio={receipt.local_implied_ratio:.3f}x"
        f"\n  dC/dT={receipt.dC_dT:.6f} | d²C/dK²={receipt.d2C_dK2:.8f} | RND Density ϕ(K)={receipt.risk_neutral_density:.5f}"
        f"\n  Arbitrage Checks: Calendar={receipt.calendar_arbitrage_free} | Butterfly={receipt.butterfly_arbitrage_free}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def bates_opt_command(args: argparse.Namespace) -> None:
    """Run David S. Bates (1996) Stochastic Volatility Jump-Diffusion Option Pricing."""
    import json
    from dataclasses import asdict
    from src.analytics.bates_jump_diffusion import (
        calculate_bates_option_price,
        persist_bates_record,
    )

    receipt = calculate_bates_option_price(
        spot_price=args.spot,
        strike_price=args.strike,
        time_to_maturity=args.ttm,
        risk_free_rate=args.rf,
        dividend_yield=args.div,
        v0=args.v0,
        kappa=args.kappa,
        theta=args.theta,
        sigma_v=args.sigma_v,
        rho=args.rho,
        lambda_jump=args.lambda_jump,
        mu_jump=args.mu_jump,
        sigma_jump=args.sigma_jump,
        ticker=args.ticker,
    )
    persist_bates_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ David S. Bates (1996) SVJ Option Pricing [{receipt.ticker}]"
        f"\n  Spot=${receipt.spot_price:.2f} | Strike=${receipt.strike_price:.2f} | T={receipt.time_to_maturity:.2f}y | rf={receipt.risk_free_rate*100:.2f}% | q={receipt.dividend_yield*100:.2f}%"
        f"\n  Bates Call=${receipt.bates_call_price:.4f} | Bates Put=${receipt.bates_put_price:.4f} | Heston Call=${receipt.heston_benchmark_call:.4f}"
        f"\n  Jump Contribution=${receipt.jump_contribution:+.4f} | Feller Ratio={receipt.feller_ratio:.2f}x ({'Satisfied' if receipt.feller_satisfied else 'Violated'})"
        f"\n  SV: V0={receipt.v0:.4f}, κ={receipt.kappa:.2f}, θ={receipt.theta:.4f}, σ_v={receipt.sigma_v:.2f}, ρ={receipt.rho:+.2f}"
        f"\n  Jumps: λ={receipt.lambda_jump:.2f}, μ_J={receipt.mu_jump:+.2f}, σ_J={receipt.sigma_jump:.2f} | Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def cgarch_command(args: argparse.Namespace) -> None:
    """Run Robert F. Engle & Gary G.J. Lee (1999) Component GARCH Two-Scale Volatility Engine."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.component_garch import (
        calculate_component_garch,
        evaluate_component_garch_profile,
        persist_component_garch_record,
    )
    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 25:
        prices = df["close"].values
        receipt = evaluate_component_garch_profile(
            prices,
            ticker=args.ticker,
            omega=args.omega,
            rho=args.rho,
            phi=args.phi,
            alpha=args.alpha,
            beta=args.beta,
        )
    else:
        np.random.seed(42)
        sim_returns = np.random.normal(0.0004, 0.012, size=150)
        receipt = calculate_component_garch(
            sim_returns,
            omega=args.omega or 0.0001,
            rho=args.rho,
            phi=args.phi,
            alpha=args.alpha,
            beta=args.beta,
            ticker=args.ticker,
        )
    persist_component_garch_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Engle & Lee (1999) Component GARCH Two-Scale Volatility [{receipt.ticker}]"
        f"\n  Sample N={receipt.sample_size} | Total Vol={receipt.latest_total_vol*100:.2f}% | Permanent Vol={receipt.latest_permanent_vol*100:.2f}%"
        f"\n  Transitory Half-Life={receipt.transitory_half_life:.1f}d | Permanent Half-Life={receipt.permanent_half_life:.1f}d"
        f"\n  Parameters: ω={receipt.omega:.6f}, ρ={receipt.rho:.4f}, φ={receipt.phi:.4f}, α={receipt.alpha:.4f}, β={receipt.beta:.4f}"
        f"\n  Log-Likelihood={receipt.log_likelihood:.2f} | AIC={receipt.aic:.2f} | Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def huang_stoll_command(args: argparse.Namespace) -> None:
    """Run Roger D. Huang & Hans R. Stoll (1997) Three-Way Spread Decomposition."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.huang_stoll_spread import (
        calculate_huang_stoll_decomposition,
        evaluate_huang_stoll_from_prices,
        persist_huang_stoll_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 32:
        receipt = evaluate_huang_stoll_from_prices(df["close"].values, ticker=args.ticker)
    else:
        np.random.seed(42)
        n = 150
        q = np.ones(n)
        for t in range(1, n):
            q[t] = q[t - 1] if np.random.rand() < 0.60 else -q[t - 1]
        s_half = (args.spread * 100.0 / 10000.0) / 2.0
        alpha_val = args.alpha
        beta_val = args.beta
        gamma_val = max(0.0, 1.0 - alpha_val - beta_val)
        dp = np.zeros(n)
        dp[2:] = s_half * q[2:] - gamma_val * s_half * q[1:-1] - beta_val * s_half * q[:-2] + np.random.normal(0, 0.001, size=n - 2)
        receipt = calculate_huang_stoll_decomposition(dp, q, ticker=args.ticker, reference_price=100.0)
    persist_huang_stoll_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Huang & Stoll (1997) Three-Way Spread Decomposition [{receipt.ticker}]"
        f"\n  Effective Spread={receipt.effective_spread_bps:.2f} bps | Sample N={receipt.sample_size}"
        f"\n  Adverse Selection (α)={receipt.adverse_selection_pct*100:.1f}% ({receipt.adverse_spread_bps:.2f} bps)"
        f"\n  Inventory Holding (β)={receipt.inventory_holding_pct*100:.1f}% ({receipt.inventory_spread_bps:.2f} bps)"
        f"\n  Order Processing (γ)={receipt.order_processing_pct*100:.1f}% ({receipt.processing_spread_bps:.2f} bps)"
        f"\n  Trade Continuation Prob π={receipt.trade_continuation_prob*100:.1f}% | R²={receipt.r_squared:.3f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def ekop_pin_command(args: argparse.Namespace) -> None:
    """Run David Easley et al. (1996) EKOP Probability of Informed Trading (PIN) Engine."""
    import json
    from dataclasses import asdict
    from src.analytics.ekop_pin import (
        calculate_ekop_pin,
        evaluate_ekop_pin_from_trades,
        persist_ekop_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 20:
        receipt = evaluate_ekop_pin_from_trades(df["close"].values, ticker=args.ticker)
    else:
        receipt = calculate_ekop_pin(
            alpha=args.alpha,
            delta=args.delta,
            mu=args.mu,
            epsilon=args.epsilon,
            ticker=args.ticker,
        )
    persist_ekop_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ David Easley et al. (1996) EKOP PIN Microstructure [{receipt.ticker}]"
        f"\n  Probability of Informed Trading (PIN)={receipt.pin*100:.2f}% | Sample N={receipt.sample_size}"
        f"\n  Info Event Prob (α)={receipt.alpha*100:.1f}% | Bad News Prob (δ)={receipt.delta*100:.1f}%"
        f"\n  Informed Intensity μ={receipt.mu:.1f} | Uninformed Intensity ε={receipt.epsilon:.1f}"
        f"\n  Informed Share={receipt.informed_order_share*100:.1f}% | Daily Turnover={receipt.daily_order_turnover:.1f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def mrr_command(args: argparse.Namespace) -> None:
    """Run Madhavan, Richardson & Roomans (1997) MRR Microstructure Engine."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.mrr_microstructure import (
        calculate_mrr_microstructure,
        evaluate_mrr_from_prices,
        persist_mrr_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 32:
        receipt = evaluate_mrr_from_prices(df["close"].values, ticker=args.ticker)
    else:
        np.random.seed(42)
        n = 150
        rho_val = (args.autocorr + 1.0) / 2.0
        x = np.ones(n)
        for t in range(1, n):
            x[t] = x[t - 1] if np.random.rand() < rho_val else -x[t - 1]
        s_half = (args.spread * 100.0 / 10000.0) / 2.0
        theta_val = s_half * args.theta_ratio
        phi0_val = max(0.0, s_half - theta_val)
        dp = np.zeros(n)
        dp[1:] = (theta_val + phi0_val) * x[1:] - (args.autocorr * theta_val + phi0_val) * x[:-1] + np.random.normal(0, 0.001, size=n - 1)
        receipt = calculate_mrr_microstructure(dp, x, ticker=args.ticker, reference_price=100.0)
    persist_mrr_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚡ Madhavan, Richardson & Roomans (1997) MRR Microstructure [{receipt.ticker}]"
        f"\n  Effective Spread={receipt.effective_spread_bps:.2f} bps | Sample N={receipt.sample_size}"
        f"\n  Information Asymmetry Impact θ=${receipt.theta_dollar:.4f} ({receipt.adverse_selection_ratio*100:.1f}%)"
        f"\n  Order Processing Friction φ0=${receipt.phi0_dollar:.4f} ({receipt.order_processing_ratio*100:.1f}%)"
        f"\n  Order Autocorrelation φ={receipt.order_autocorrelation_phi:+.3f} (Continuation ρ={receipt.continuation_prob_rho*100:.1f}%) | R²={receipt.r_squared:.3f}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def hasbrouck_gibbs_command(args: argparse.Namespace) -> None:
    """Run Joel Hasbrouck (2009) Bayesian Gibbs Sampler Effective Spread Engine."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.hasbrouck_gibbs_spread import (
        run_hasbrouck_gibbs,
        persist_hasbrouck_gibbs_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 20:
        prices = df["close"].values
    else:
        np.random.seed(args.seed)
        n = 150
        u = np.random.normal(0, 0.10, size=n)
        m = 100.0 + np.cumsum(u)
        q = np.random.choice([-1.0, 1.0], size=n)
        c_true = (args.spread * 100.0 / 10000.0) / 2.0
        prices = m + c_true * q

    receipt = run_hasbrouck_gibbs(
        prices=prices,
        iterations=args.iterations,
        burn_in=args.burn_in,
        seed=args.seed,
        ticker=args.ticker,
    )
    persist_hasbrouck_gibbs_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    roll_str = "⚠️ Positive Cov (Roll Failed)" if receipt.positive_autocovariance_flag else f"${receipt.roll_spread_dollar:.4f}"
    print(
        f"\n🎲 Joel Hasbrouck (2009) Bayesian Gibbs Sampler Effective Spread [{receipt.ticker}]"
        f"\n  Effective Spread=${receipt.effective_spread_dollar:.4f} ({receipt.effective_spread_bps:.2f} bps)"
        f"\n  Posterior Half-Spread c=${receipt.posterior_c_mean:.4f} ± {receipt.posterior_c_std:.4f}"
        f"\n  Benchmark Roll (1984) Spread={roll_str}"
        f"\n  Fundamental Volatility σ_u=${receipt.posterior_sigma_u:.4f} | Sample N={receipt.sample_size}"
        f"\n  MCMC Iterations={receipt.iterations} (Burn-in={receipt.burn_in}) | Regime={receipt.regime}"
        f"\n  HMAC={receipt.hmac_signature[:16]}..."
    )


def gkn_command(args: argparse.Namespace) -> None:
    """Run Thomas J. George, Gautam Kaul & M. Nimalendran (1991) GKN Spread Decomposition."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.gkn_spread_decomposition import (
        calculate_gkn_spread,
        persist_gkn_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 15:
        prices = df["close"].values
        receipt = calculate_gkn_spread(prices, ticker=args.ticker)
    else:
        np.random.seed(args.seed)
        n = 150
        u = np.random.normal(0, 0.001, size=n)
        m = 100.0 * np.exp(np.cumsum(u))
        q = np.random.choice([-1.0, 1.0], size=n)
        s0_prop = (args.spread * 100.0 / 10000.0) / 100.0
        p = m * (1.0 + 0.5 * s0_prop * q)
        receipt = calculate_gkn_spread(p, quote_midpoints=m, ticker=args.ticker)

    persist_gkn_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n⚖️ George, Kaul & Nimalendran (1991) GKN Spread Decomposition [{receipt.ticker}]"
        f"\n  Effective Spread={receipt.total_effective_spread_bps:.2f} bps | Sample N={receipt.sample_size}"
        f"\n  Order Processing Friction s₀={receipt.order_processing_spread_bps:.2f} bps ({receipt.order_processing_ratio*100:.1f}%)"
        f"\n  Adverse Selection Impact sᵢ={receipt.adverse_selection_spread_bps:.2f} bps ({receipt.adverse_selection_ratio*100:.1f}%)"
        f"\n  Naive Roll Benchmark={receipt.roll_spread_bps:.2f} bps | Roll Bias Corrected={receipt.roll_bias_bps:+.2f} bps"
        f"\n  Expected Return Autocovariance={receipt.expected_return_autocovariance:.6e} | Derivation Mode={receipt.derivation_mode}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def glosten_harris_command(args: argparse.Namespace) -> None:
    """Run Lawrence R. Glosten & Lawrence E. Harris (1988) Trade-Size Dependent Spread Decomposition."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.glosten_harris_spread import (
        calculate_glosten_harris_spread,
        evaluate_glosten_harris_from_prices,
        persist_glosten_harris_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 20:
        prices = df["close"].values
        volumes = df["volume"].values if "volume" in df.columns else None
        receipt = evaluate_glosten_harris_from_prices(
            prices=prices,
            volumes=volumes,
            ticker=args.ticker,
            benchmark_volume=args.benchmark_volume,
        )
    else:
        np.random.seed(args.seed)
        n = 150
        q = np.random.choice([-1.0, 1.0], size=n)
        v = np.random.uniform(0.5, 3.0, size=n)
        dq = q[1:] - q[:-1]
        dqv = (q[1:] * v[1:]) - (q[:-1] * v[:-1])
        c0_val = 0.05
        c1_val = 0.02
        z0_val = 0.04
        z1_val = 0.03
        dp = c0_val * dq + c1_val * dqv + z0_val * q[1:] + z1_val * (q[1:] * v[1:]) + np.random.normal(0, 0.005, size=n - 1)
        receipt = calculate_glosten_harris_spread(
            price_changes=dp,
            trade_indicators=q,
            trade_volumes=v,
            ticker=args.ticker,
            reference_price=100.0,
            benchmark_volume=args.benchmark_volume,
        )

    persist_glosten_harris_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n📦 Glosten & Harris (1988) Trade-Size Dependent Spread Decomposition [{receipt.ticker}]"
        f"\n  Effective Spread=${receipt.effective_spread_dollar:.4f} ({receipt.effective_spread_bps:.2f} bps) at V={receipt.benchmark_volume:.1f}"
        f"\n  Transitory Friction Spread=${receipt.transitory_spread_dollar:.4f} ({receipt.order_processing_ratio*100:.1f}%)"
        f"\n  Adverse Selection Spread=${receipt.adverse_spread_dollar:.4f} ({receipt.adverse_selection_ratio*100:.1f}%)"
        f"\n  Fixed Components: c₀=${receipt.c0_fixed_transitory:.4f} | z₀=${receipt.z0_fixed_adverse:.4f}"
        f"\n  Variable Components: c₁=${receipt.c1_variable_transitory:.4f} | z₁=${receipt.z1_variable_adverse:.4f}"
        f"\n  Model Fit R²={receipt.r_squared:.3f} | Sample N={receipt.sample_size} | Mode={receipt.derivation_mode}"
        f"\n  Regime={receipt.regime} | HMAC={receipt.hmac_signature[:16]}..."
    )


def lot_command(args: argparse.Namespace) -> None:
    """Run David A. Lesmond, Joseph P. Ogden & Charles A. Trzcinka (1999) LOT Transaction Cost Engine."""
    import json
    from dataclasses import asdict
    import numpy as np
    from src.analytics.lesmond_ogden_trzcinka_spread import (
        calculate_lot_spread,
        evaluate_lot_from_prices,
        persist_lot_record,
    )

    df = get_observations(args.ticker)
    if df is not None and not df.empty and "close" in df.columns and len(df) >= 26:
        prices = df["close"].values
        receipt = evaluate_lot_from_prices(stock_prices=prices, ticker=args.ticker)
    else:
        np.random.seed(args.seed)
        n = 150
        r_m = np.random.normal(0.0004, 0.010, size=n)
        e = np.random.normal(0.0, 0.012, size=n)
        r_star = 1.10 * r_m + e
        a1_val = -0.0050
        a2_val = 0.0060
        r_obs = np.zeros(n)
        for t in range(n):
            if r_star[t] < a1_val:
                r_obs[t] = r_star[t] - a1_val
            elif r_star[t] > a2_val:
                r_obs[t] = r_star[t] - a2_val
            else:
                r_obs[t] = 0.0
        receipt = calculate_lot_spread(
            stock_returns=r_obs,
            market_returns=r_m,
            ticker=args.ticker,
            derivation_mode="SYNTHETIC_FALLBACK",
        )

    persist_lot_record(receipt)

    if getattr(args, "json", False):
        print(json.dumps(asdict(receipt), indent=2))
        return

    print(
        f"\n🎯 Lesmond, Ogden & Trzcinka (1999) LOT Transaction Costs [{receipt.ticker}]"
        f"\n  Round-Trip Spread={receipt.round_trip_spread_bps:.2f} bps ({receipt.round_trip_spread_pct*100:.3f}%) | Sample N={receipt.sample_size}"
        f"\n  Buy Cost Threshold α₂=+{receipt.buy_cost_bps:.2f} bps | Sell Cost Threshold |α₁|=-{receipt.sell_cost_bps:.2f} bps"
        f"\n  Zero-Return Days={receipt.zero_return_count} ({receipt.zero_return_frequency*100:.1f}%)"
        f"\n  Market Beta={receipt.market_beta:.2f} | Idiosyncratic Risk σ={receipt.idiosyncratic_sigma*100:.2f}% | Log-Likelihood={receipt.log_likelihood:.2f}"
        f"\n  Regime={receipt.regime} | Mode={receipt.derivation_mode} | HMAC={receipt.hmac_signature[:16]}..."
    )


def main() -> None:




    parser = argparse.ArgumentParser(
        prog="nextomni-stock",
        description="NextOmni Stock Intelligence & Forecasting CLI",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Sync parser
    sync_parser = subparsers.add_parser("sync", help="Synchronize historical market data")
    sync_parser.add_argument("--ticker", type=str, help="Stock ticker (e.g. JPM, 1155.KL)")
    sync_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        help="Exchange MIC (XNYS, XKLS)",
    )
    sync_parser.add_argument("--name", type=str, default=None, help="Company name")
    default_end = datetime.now().strftime("%Y-%m-%d")
    default_start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    sync_parser.add_argument(
        "--start", type=str, default=default_start, help="Start date YYYY-MM-DD"
    )
    sync_parser.add_argument(
        "--end", type=str, default=default_end, help="End date YYYY-MM-DD"
    )
    sync_parser.add_argument(
        "--all", action="store_true", help="Sync all default initial tickers"
    )

    # Status parser
    subparsers.add_parser("status", help="Show database status and tracked stocks")

    # Analyze parser
    analyze_parser = subparsers.add_parser(
        "analyze", help="Run exploratory data analysis and export charts"
    )
    analyze_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker to analyze (e.g. JPM, 1155.KL)"
    )

    # Forecast Baseline parser
    baseline_parser = subparsers.add_parser(
        "forecast-baseline", help="Generate next-session persistence forecast & evaluate benchmark"
    )
    baseline_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    baseline_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )

    # Train Linear parser
    linear_parser = subparsers.add_parser(
        "train-linear", help="Train Linear Regression model with walk-forward validation"
    )
    linear_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    linear_parser.add_argument(
        "--min-train", type=int, default=120, help="Minimum initial training sessions"
    )
    linear_parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength alpha"
    )

    # Train All Horizons parser
    multi_parser = subparsers.add_parser(
        "train-all-horizons", help="Train and validate dedicated models for 1, 5, and 21 sessions"
    )
    multi_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    multi_parser.add_argument(
        "--model",
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble", "stacking"],
        default="linear",
        help="Forecasting model family (linear, gradient_boosting, ensemble, adaptive_ensemble, stacking)",
    )
    multi_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    multi_parser.add_argument(
        "--min-train", type=int, default=120, help="Minimum initial training sessions"
    )
    multi_parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength alpha (linear only)"
    )

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict", help="Generate and store a forecast prediction"
    )
    predict_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    predict_parser.add_argument(
        "--model",
        choices=["linear", "gradient_boosting", "ensemble", "adaptive_ensemble", "stacking", "baseline"],
        default="linear",
        help="Forecasting model family to use",
    )

    predict_parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 5, 21], help="Forecast horizon sessions"
    )
    predict_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    predict_parser.add_argument(
        "--explain", action="store_true", help="Print local feature signal attribution breakdown"
    )

    # Evaluate pending parser
    subparsers.add_parser(
        "evaluate-pending", help="Reconcile and evaluate pending predictions with actual prices"
    )

    # List predictions parser
    list_parser = subparsers.add_parser(
        "list-predictions", help="Display predictions audit log"
    )
    list_parser.add_argument("--ticker", type=str, default=None, help="Filter by ticker")
    list_parser.add_argument(
        "--status",
        choices=["PENDING", "AWAITING_ACTUAL", "EVALUATED", "FAILED"],
        default=None,
        help="Filter by status",
    )
    list_parser.add_argument(
        "--export",
        type=str,
        default=None,
        help="Export predictions to signed CSV or JSON file with SHA-256 manifest",
    )

    # Monitor parser
    monitor_parser = subparsers.add_parser(
        "monitor", help="Monitor real-time quotes, session status, and feed latency"
    )
    monitor_parser.add_argument("--ticker", type=str, default=None, help="Stock ticker (e.g. JPM, 1155.KL)")
    monitor_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    monitor_parser.add_argument(
        "--refresh", action="store_true", help="Bypass cache and force live fetch"
    )

    # Session status parser
    session_parser = subparsers.add_parser(
        "session-status", help="Display current exchange operational state and calendar countdown"
    )
    session_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )

    # Daemon reconcile parser
    daemon_parser = subparsers.add_parser(
        "daemon-reconcile", help="Run automated session-aware forecast evaluation daemon"
    )
    daemon_parser.add_argument(
        "--once", action="store_true", help="Run a single reconciliation pass and exit"
    )
    daemon_parser.add_argument(
        "--interval", type=int, default=300, help="Polling interval in seconds (default: 300)"
    )
    daemon_parser.add_argument(
        "--max-iterations", type=int, default=None, help="Maximum daemon cycles before stopping"
    )

    # Verify audit parser
    subparsers.add_parser(
        "verify-audit", help="Verify cryptographic SHA-256 integrity of predictions audit trail"
    )

    # Backup DB parser
    backup_parser = subparsers.add_parser(
        "backup-db", help="Create an atomic verified online SQLite database backup"
    )
    backup_parser.add_argument(
        "--dest", type=str, default=None, help="Target destination file path for backup"
    )

    # Risk report parser
    risk_parser = subparsers.add_parser(
        "risk-report", help="Compute quantitative tail-risk metrics (VaR & CVaR)"
    )
    risk_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    risk_parser.add_argument(
        "--stress-test", action="store_true", help="Simulate historical macroeconomic stress scenarios"
    )
    risk_parser.add_argument(
        "--json", action="store_true", help="Output risk report and path profile as JSON"
    )

    # Prune DB parser
    prune_parser = subparsers.add_parser(
        "prune-db", help="Safely prune expired raw provider payloads older than N days"
    )
    prune_parser.add_argument(
        "--days", type=int, default=90, help="Retention period in days (default: 90)"
    )

    # Vacuum DB parser
    subparsers.add_parser(
        "vacuum-db", help="Verify B-tree integrity and execute VACUUM storage defragmentation"
    )

    # Backtest parser
    backtest_parser = subparsers.add_parser(
        "backtest", help="Simulate out-of-sample quantitative strategy backtest"
    )
    backtest_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    backtest_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
        default="ensemble",
        help="Forecasting model family (default: ensemble)",
    )
    backtest_parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 5, 21], help="Forecast horizon (default: 1)"
    )
    backtest_parser.add_argument(
        "--fee-bps", type=float, default=5.0, help="Transaction fee in basis points (default: 5.0)"
    )
    backtest_parser.add_argument(
        "--min-train", type=int, default=120, help="Initial burn-in training sessions (default: 120)"
    )
    backtest_parser.add_argument(
        "--sizing",
        choices=["fixed", "vol_target", "half_kelly"],
        default="fixed",
        help="Position sizing mechanism (default: fixed)",
    )
    backtest_parser.add_argument(
        "--target-vol",
        type=float,
        default=0.15,
        help="Annualized volatility target for vol_target sizing (default: 0.15)",
    )
    backtest_parser.add_argument(
        "--max-leverage",
        type=float,
        default=1.0,
        help="Maximum portfolio leverage constraint (default: 1.0)",
    )

    # System health parser
    health_parser = subparsers.add_parser(
        "system-health", help="Execute unified security and health pre-flight audit"
    )
    health_parser.add_argument(
        "--json", action="store_true", help="Output audit receipt as JSON"
    )

    # Screener parser
    screen_parser = subparsers.add_parser(
        "screen", help="Scan and rank asset universe cross-sectionally"
    )
    screen_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    screen_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=5,
        help="Forecast horizon in sessions (default: 5)",
    )
    screen_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
        default="ensemble",
        help="Forecasting model family (default: ensemble)",
    )

    screen_parser.add_argument(
        "--top",
        type=int,
        default=None,
        help="Limit output to top N ranked assets",
    )
    screen_parser.add_argument(
        "--json",
        action="store_true",
        help="Output screener rankings as JSON",
    )

    # IC Drift parser
    ic_drift_parser = subparsers.add_parser(
        "ic-drift", help="Monitor Information Coefficient drift and alpha decay"
    )
    ic_drift_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional ticker symbol to filter evaluated predictions",
    )
    ic_drift_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    ic_drift_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling window size in sessions (default: 20)",
    )
    ic_drift_parser.add_argument(
        "--json",
        action="store_true",
        help="Output IC drift receipt as JSON",
    )

    # Portfolio parser
    portfolio_parser = subparsers.add_parser(
        "portfolio", help="Optimize multi-asset alpha portfolio using Ledoit-Wolf shrinkage"
    )
    portfolio_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    portfolio_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=5,
        help="Forecast horizon in sessions (default: 5)",
    )
    portfolio_parser.add_argument(
        "--mode",
        choices=["max_sharpe", "min_variance", "risk_parity", "equal_weight", "hrp", "cvar_min", "cvar_parity"],
        default="max_sharpe",
        help="Optimization objective mode (default: max_sharpe)",
    )

    portfolio_parser.add_argument(
        "--max-weight",
        type=float,
        default=0.40,
        help="Maximum individual asset allocation weight (default: 0.40)",
    )
    portfolio_parser.add_argument(
        "--json",
        action="store_true",
        help="Output portfolio allocation as JSON",
    )

    # Calibration report parser
    calib_parser = subparsers.add_parser(
        "calibration-report", help="Audit prediction interval empirical coverage and Winkler sharpness"
    )
    calib_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional ticker symbol to filter evaluated predictions",
    )
    calib_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    calib_parser.add_argument(
        "--confidence",
        type=float,
        default=0.80,
        help="Nominal target confidence level (default: 0.80)",
    )
    calib_parser.add_argument(
        "--json",
        action="store_true",
        help="Output calibration report as JSON",
    )

    # Lead-lag Granger causality parser
    lead_lag_parser = subparsers.add_parser(
        "lead-lag", help="Evaluate cross-market lead-lag Granger causality"
    )
    lead_lag_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Source leading ticker symbol (e.g. JPM)",
    )
    lead_lag_parser.add_argument(
        "--target",
        type=str,
        required=True,
        help="Target lagging ticker symbol (e.g. 1155.KL)",
    )
    lead_lag_parser.add_argument(
        "--max-lag",
        type=int,
        default=5,
        help="Maximum lag orders to test in sessions (default: 5)",
    )
    lead_lag_parser.add_argument(
        "--json",
        action="store_true",
        help="Output lead-lag receipt as JSON",
    )

    # Cointegration parser
    coint_parser = subparsers.add_parser(
        "cointegrate", help="Evaluate Engle-Granger cointegration and mean-reverting spread"
    )
    coint_parser.add_argument(
        "--ticker1",
        type=str,
        required=True,
        help="First ticker symbol (dependent P1)",
    )
    coint_parser.add_argument(
        "--ticker2",
        type=str,
        required=True,
        help="Second ticker symbol (independent P2)",
    )
    coint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output cointegration receipt as JSON",
    )

    # Liquidity report parser
    liq_parser = subparsers.add_parser(
        "liquidity-report", help="Audit market microstructure liquidity and price impact"
    )
    liq_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    liq_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Rolling observation window in sessions (default: 21)",
    )
    liq_parser.add_argument(
        "--json",
        action="store_true",
        help="Output liquidity receipt as JSON",
    )

    # Volatility report parser
    vol_parser = subparsers.add_parser(
        "volatility-report", help="Multi-estimator realized volatility diagnostics (Yang-Zhang, GK, Parkinson, EWMA)"
    )
    vol_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    vol_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling observation window in sessions (default: 20)",
    )
    vol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output volatility diagnostics receipt as JSON",
    )

    # CRPS evaluation parser
    crps_parser = subparsers.add_parser(
        "crps-eval", help="Evaluate probabilistic forecasting quality (CRPS, Pinball Loss, Winkler Score)"
    )
    crps_parser.add_argument(
        "--ticker",
        type=str,
        default=None,
        help="Optional stock ticker filter (e.g. JPM, 1155.KL)",
    )
    crps_parser.add_argument(
        "--horizon",
        type=int,
        choices=[1, 5, 21],
        default=None,
        help="Optional forecast horizon filter (1, 5, 21)",
    )
    crps_parser.add_argument(
        "--model",
        choices=["ensemble", "adaptive_ensemble", "stacking", "linear", "gradient_boosting"],
        default=None,
        help="Optional model family filter",
    )

    # Changepoint detection parser
    changepoint_parser = subparsers.add_parser(
        "changepoint-detect", help="Detect structural breaks and regime shifts using Bayesian Online Change-Point Detection"
    )
    changepoint_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    changepoint_parser.add_argument(
        "--hazard",
        type=int,
        default=100,
        help="Expected timescale between regime shifts in trading sessions (default: 100)",
    )
    changepoint_parser.add_argument(
        "--json",
        action="store_true",
        help="Output change-point receipt as JSON",
    )

    # Toxicity report parser
    toxicity_parser = subparsers.add_parser(
        "toxicity-report", help="Audit order flow toxicity and Volume-Synchronized Probability of Toxicity (VPIN)"
    )
    toxicity_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol to audit (e.g. JPM, 1155.KL)",
    )
    toxicity_parser.add_argument(
        "--window",
        type=int,
        default=20,
        help="Rolling observation window in sessions (default: 20)",
    )
    toxicity_parser.add_argument(
        "--json",
        action="store_true",
        help="Output toxicity receipt as JSON",
    )

    # Tail dependence parser
    tail_parser = subparsers.add_parser(
        "tail-dependence", help="Evaluate multi-asset lower-tail dependence and joint crash fragility"
    )
    tail_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default=None,
        help="Optional exchange MIC filter (XNYS, XKLS)",
    )
    tail_parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Lower-tail quantile cutoff q (default: 0.05 for 5% extreme crash tail)",
    )
    tail_parser.add_argument(
        "--json",
        action="store_true",
        help="Output tail dependence receipt as JSON",
    )

    # Factor attribution parser
    factor_parser = subparsers.add_parser(
        "factor-attribution", help="Evaluate multi-factor risk attribution and style exposures"
    )
    factor_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. JPM, MSFT)",
    )
    factor_parser.add_argument(
        "--exchange",
        type=str,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        default="XNYS",
        help="Exchange MIC code (default: XNYS)",
    )
    factor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output factor attribution receipt as JSON",
    )

    # Wavelet denoise parser
    wavelet_parser = subparsers.add_parser(
        "wavelet-denoise", help="Evaluate multiscale wavelet decomposition and adaptive denoising"
    )
    wavelet_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, JPM)",
    )
    wavelet_parser.add_argument(
        "--levels",
        type=int,
        default=3,
        help="Multiresolution decomposition levels (default: 3)",
    )
    wavelet_parser.add_argument(
        "--json",
        action="store_true",
        help="Output wavelet receipt as JSON",
    )

    # Momentum crash report parser
    momentum_parser = subparsers.add_parser(
        "momentum-crash-report", help="Evaluate risk-managed momentum and crash hazard diagnostics"
    )
    momentum_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, JPM)",
    )
    momentum_parser.add_argument(
        "--target-vol",
        type=float,
        default=0.12,
        help="Target annualized volatility sigma* (default: 0.12)",
    )
    momentum_parser.add_argument(
        "--max-leverage",
        type=float,
        default=2.0,
        help="Maximum allowable leverage cap (default: 2.0)",
    )
    momentum_parser.add_argument(
        "--json",
        action="store_true",
        help="Output momentum crash receipt as JSON",
    )

    # Subparser: fracdiff
    fracdiff_parser = subparsers.add_parser(
        "fracdiff",
        help="Evaluate memory-preserving fractional differentiation and optimal d*",
    )
    fracdiff_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    fracdiff_parser.add_argument(
        "--threshold",
        type=float,
        default=1e-4,
        help="FFD weight truncation threshold (default: 1e-4)",
    )
    fracdiff_parser.add_argument(
        "--d",
        type=float,
        default=None,
        help="Specific fractional differentiation order to evaluate (optional)",
    )
    fracdiff_parser.add_argument(
        "--auto-d",
        action="store_true",
        default=True,
        help="Automatically search for minimal stationary order d* (default: True)",
    )
    fracdiff_parser.add_argument(
        "--json",
        action="store_true",
        help="Output fractional differentiation receipt as JSON",
    )

    # Subparser: kalman-spread
    kalman_parser = subparsers.add_parser(
        "kalman-spread",
        help="Evaluate online Kalman Filter dynamic hedge ratio and adaptive spread",
    )
    kalman_parser.add_argument(
        "--y-ticker",
        type=str,
        required=True,
        help="Dependent asset ticker Y (e.g. JPM, MSFT)",
    )
    kalman_parser.add_argument(
        "--x-ticker",
        type=str,
        required=True,
        help="Independent asset ticker X (e.g. BAC, AAPL)",
    )
    kalman_parser.add_argument(
        "--delta",
        type=float,
        default=1e-4,
        help="Process noise factor controlling rate of parameter adaptation (default: 1e-4)",
    )
    kalman_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kalman spread receipt as JSON",
    )

    # Subparser: evt-report
    evt_parser = subparsers.add_parser(
        "evt-report",
        help="Evaluate Extreme Value Theory (EVT) Peaks-Over-Threshold tail risk and GPD",
    )
    evt_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    evt_parser.add_argument(
        "--threshold-quantile",
        type=float,
        default=0.95,
        help="Loss threshold quantile u (default: 0.95)",
    )
    evt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output EVT tail receipt as JSON",
    )

    # Subparser: entropy-report
    entropy_parser = subparsers.add_parser(
        "entropy-report",
        help="Evaluate market information entropy, permutation complexity, and predictability",
    )
    entropy_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    entropy_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling analysis window in trading sessions (default: 60)",
    )
    entropy_parser.add_argument(
        "--embedding-dim",
        type=int,
        default=3,
        help="Permutation embedding dimension m (default: 3)",
    )
    entropy_parser.add_argument(
        "--json",
        action="store_true",
        help="Output market entropy receipt as JSON",
    )

    # Subparser: garch-volatility
    garch_parser = subparsers.add_parser(
        "garch-volatility",
        help="Evaluate asymmetric GJR-GARCH(1, 1) volatility forecasting and leverage effect",
    )
    garch_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    garch_parser.add_argument(
        "--model",
        type=str,
        default="gjr",
        choices=["gjr", "standard"],
        help="GARCH model specification: 'gjr' (asymmetric) or 'standard' (default: gjr)",
    )
    garch_parser.add_argument(
        "--json",
        action="store_true",
        help="Output GARCH receipt as JSON",
    )

    # Subparser: resampled-frontier
    rf_parser = subparsers.add_parser(
        "resampled-frontier",
        help="Evaluate Michaud (1998) Resampled Efficiency and compare with classical Markowitz frontier",
    )
    rf_parser.add_argument(
        "--tickers",
        type=str,
        required=True,
        help="Comma-separated ticker symbols (e.g. AAPL,MSFT,NVDA,GOOGL)",
    )
    rf_parser.add_argument(
        "--simulations",
        type=int,
        default=100,
        help="Number of Monte Carlo resampling paths (default: 100)",
    )
    rf_parser.add_argument(
        "--points",
        type=int,
        default=15,
        help="Number of discrete efficient frontier points (default: 15)",
    )
    rf_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free rate benchmark (default: 0.04)",
    )
    rf_parser.add_argument(
        "--json",
        action="store_true",
        help="Output resampled frontier receipt as JSON",
    )

    # Subparser: mrmr-select
    mrmr_parser = subparsers.add_parser(
        "mrmr-select",
        help="Perform Minimum Redundancy Maximum Relevance (mRMR) quantitative feature selection",
    )
    mrmr_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    mrmr_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 5, 21],
        help="Forecast horizon in sessions: 1, 5, or 21 (default: 1)",
    )
    mrmr_parser.add_argument(
        "--n-features",
        type=int,
        default=5,
        help="Target number of non-redundant features to select (default: 5)",
    )
    mrmr_parser.add_argument(
        "--criterion",
        type=str,
        default="MID",
        choices=["MID", "MIQ"],
        help="mRMR objective criterion: 'MID' (Difference) or 'MIQ' (Quotient) (default: MID)",
    )
    mrmr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output mRMR selection receipt as JSON",
    )

    # Subparser: price-impact
    impact_parser = subparsers.add_parser(
        "price-impact",
        help="Evaluate Kyle's (1985) Informed Trading Lambda and market price impact",
    )
    impact_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    impact_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Rolling regression window in sessions (default: 60)",
    )
    impact_parser.add_argument(
        "--json",
        action="store_true",
        help="Output price impact receipt as JSON",
    )

    # Subparser: tail-shape
    tail_parser = subparsers.add_parser(
        "tail-shape",
        help="Evaluate robust quantile tail shape, Moors kurtosis, and crash asymmetry",
    )
    tail_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    tail_parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Rolling observation window in sessions (default: 120)",
    )
    tail_parser.add_argument(
        "--json",
        action="store_true",
        help="Output tail shape receipt as JSON",
    )

    # Subparser: pain-index
    pain_parser = subparsers.add_parser(
        "pain-index",
        help="Evaluate Burke Ratio, Becker Pain Index, and Ulcer Performance Index",
    )
    pain_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    pain_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free benchmark rate (default: 0.04)",
    )
    pain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output drawdown analytics receipt as JSON",
    )

    # Subparser: markov-regime
    markov_parser = subparsers.add_parser(
        "markov-regime",
        help="Evaluate Hamilton (1989) two-state Markov-switching dynamic regime model",
    )
    markov_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    markov_parser.add_argument(
        "--window",
        type=int,
        default=120,
        help="Observation rolling window in sessions (default: 120)",
    )
    markov_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Markov regime receipt as JSON",
    )

    # Subparser: rmt-denoise
    rmt_parser = subparsers.add_parser(
        "rmt-denoise",
        help="Denoise empirical covariance matrix via Marchenko-Pastur (1967) Random Matrix Theory",
    )
    rmt_parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="Comma-separated stock tickers (e.g. AAPL,MSFT,NVDA,JPM)",
    )
    rmt_parser.add_argument(
        "--exchange",
        type=str,
        default=None,
        choices=list(SUPPORTED_EXCHANGES.keys()),
        help="Exchange MIC code for universe selection (default: XNYS)",
    )
    rmt_parser.add_argument(
        "--lookback",
        type=int,
        default=252,
        help="Lookback window in trading sessions (default: 252)",
    )
    rmt_parser.add_argument(
        "--method",
        type=str,
        default="constant_residual",
        choices=["constant_residual", "targeted_shrinkage"],
        help="RMT eigenvalue replacement algorithm (default: constant_residual)",
    )
    rmt_parser.add_argument(
        "--detrend",
        action="store_true",
        help="Remove dominant market eigenvalue mode (lambda_1)",
    )
    rmt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output RMT denoising receipt as JSON",
    )

    # Subparser: jump-test
    jump_parser = subparsers.add_parser(
        "jump-test",
        help="Evaluate Barndorff-Nielsen & Shephard (2004, 2006) Bipower Variation and Jump Decomposition",
    )
    jump_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    jump_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Observation rolling window in sessions (default: 21)",
    )
    jump_parser.add_argument(
        "--json",
        action="store_true",
        help="Output jump diffusion receipt as JSON",
    )

    # Subparser: deflated-sharpe
    dsr_parser = subparsers.add_parser(
        "deflated-sharpe",
        help="Evaluate Bailey & López de Prado (2014) Probabilistic & Deflated Sharpe Ratio",
    )
    dsr_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dsr_parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of strategy/model trials tested (default: 10)",
    )
    dsr_parser.add_argument(
        "--benchmark-sr",
        type=float,
        default=0.0,
        help="Hurdle annualized benchmark Sharpe ratio (default: 0.0)",
    )
    dsr_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    dsr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Deflated Sharpe receipt as JSON",
    )

    # Subparser: effective-spread
    spread_parser = subparsers.add_parser(
        "effective-spread",
        help="Evaluate Richard Roll (1984) Effective Bid-Ask Spread and Composite Execution Friction",
    )
    spread_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    spread_parser.add_argument(
        "--window",
        type=int,
        default=21,
        help="Observation rolling window in sessions (default: 21)",
    )
    spread_parser.add_argument(
        "--json",
        action="store_true",
        help="Output effective spread receipt as JSON",
    )

    # Subparser: cpcv-validate
    cpcv_parser = subparsers.add_parser(
        "cpcv-validate",
        help="Evaluate Marcos López de Prado (2018) Combinatorial Purged & Embargoed Cross-Validation (CPCV)",
    )
    cpcv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    cpcv_parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon in sessions (default: 1)",
    )
    cpcv_parser.add_argument(
        "--groups",
        type=int,
        default=6,
        help="Total contiguous groups N (default: 6)",
    )
    cpcv_parser.add_argument(
        "--test-groups",
        type=int,
        default=2,
        help="Test groups per combination k (default: 2)",
    )
    cpcv_parser.add_argument(
        "--embargo",
        type=float,
        default=0.01,
        help="Post-test embargo proportion (default: 0.01)",
    )
    cpcv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CPCV receipt as JSON",
    )

    # Subparser: downside-beta
    downside_parser = subparsers.add_parser(
        "downside-beta",
        help="Evaluate Ang, Chen & Xing (2006) Downside Beta and Realized Semivariance",
    )
    downside_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    downside_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    downside_parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    downside_parser.add_argument(
        "--json",
        action="store_true",
        help="Output downside risk receipt as JSON",
    )

    # Subparser: optimal-execution
    execution_parser = subparsers.add_parser(
        "optimal-execution",
        help="Evaluate Almgren & Chriss (2000) optimal execution trajectory and market impact",
    )
    execution_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    execution_parser.add_argument(
        "--shares",
        type=float,
        required=True,
        help="Target shares to execute (positive for sell/liquidation, negative for buy)",
    )
    execution_parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Execution horizon in trading days/sessions (default: 5)",
    )
    execution_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=1e-6,
        help="Investor risk aversion parameter lambda >= 0 (default: 1e-6, 0 for pure TWAP)",
    )
    execution_parser.add_argument(
        "--json",
        action="store_true",
        help="Output execution trajectory receipt as JSON",
    )

    # Subparser: fama-french
    fama_parser = subparsers.add_parser(
        "fama-french",
        help="Evaluate Fama-French (1993) & Carhart (1997) multi-factor risk and alpha attribution",
    )
    fama_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    fama_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    fama_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    fama_parser.add_argument(
        "--json",
        action="store_true",
        help="Output factor attribution receipt as JSON",
    )

    # Subparser: black-litterman
    bl_parser = subparsers.add_parser(
        "black-litterman",
        help="Execute Black-Litterman (1992) Bayesian portfolio optimization & active view blending",
    )
    bl_parser.add_argument(
        "--universe",
        type=str,
        required=True,
        help="Comma-separated stock universe (e.g. AAPL,MSFT,NVDA,GOOGL)",
    )
    bl_parser.add_argument(
        "--views",
        type=str,
        default=None,
        help="Active views as key:value pairs or JSON (e.g. NVDA:0.25,AAPL:0.12)",
    )
    bl_parser.add_argument(
        "--tau",
        type=float,
        default=0.05,
        help="Prior uncertainty scalar (default: 0.05)",
    )
    bl_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.5,
        help="Market risk aversion parameter delta (default: 2.5)",
    )
    bl_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    bl_parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Disable Marchenko-Pastur RMT covariance denoising",
    )
    bl_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black-Litterman receipt as JSON",
    )

    # Subparser: idiosyncratic-vol
    ivol_parser = subparsers.add_parser(
        "idiosyncratic-vol",
        help="Evaluate Ang, Hodrick, Xing & Zhang (2006, 2009) idiosyncratic volatility & higher moments",
    )
    ivol_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    ivol_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    ivol_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    ivol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output idiosyncratic volatility receipt as JSON",
    )

    # Subparser: liquidity-beta
    liq_parser = subparsers.add_parser(
        "liquidity-beta",
        help="Evaluate Pástor & Stambaugh (2003) systematic liquidity risk factor & liquidity beta",
    )
    liq_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    liq_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    liq_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    liq_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Pástor-Stambaugh liquidity receipt as JSON",
    )

    # Subparser: fama-macbeth
    fm_parser = subparsers.add_parser(
        "fama-macbeth",
        help="Execute Fama & MacBeth (1973) two-pass cross-sectional regression with Shanken (1992) corrections",
    )
    fm_parser.add_argument(
        "--universe",
        type=str,
        required=True,
        help="Comma-separated stock universe (e.g. AAPL,MSFT,NVDA,GOOGL,AMZN)",
    )
    fm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    fm_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Observation lookback window in sessions (default: 120)",
    )
    fm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Fama-MacBeth receipt as JSON",
    )

    # Subparser: higher-moments
    hm_parser = subparsers.add_parser(
        "higher-moments",
        help="Evaluate Harvey-Siddique (2000) co-skewness & Dittmar (2002) co-kurtosis higher-moment asset pricing",
    )
    hm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    hm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Market benchmark ticker symbol (default: SPY)",
    )
    hm_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Observation lookback window in sessions (default: 60)",
    )
    hm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output higher moments receipt as JSON",
    )

    # Subparser: merton-credit
    mc_parser = subparsers.add_parser(
        "merton-credit",
        help="Evaluate Robert C. Merton (1974) structural credit risk & distance-to-default",
    )
    mc_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    mc_parser.add_argument(
        "--debt",
        type=float,
        required=True,
        help="Face value of total debt liabilities / default point in dollars",
    )
    mc_parser.add_argument(
        "--shares",
        type=float,
        default=None,
        help="Common shares outstanding (optional; defaults to 1B shares or market cap)",
    )
    mc_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Explicit total equity market capitalization in dollars (optional)",
    )
    mc_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Explicit annualized equity return volatility sigma_E (optional)",
    )
    mc_parser.add_argument(
        "--risk-free",
        type=float,
        default=0.045,
        help="Annualized risk-free interest rate (default: 0.045)",
    )
    mc_parser.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="Credit maturity / forecast horizon in years (default: 1.0)",
    )
    mc_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback window in sessions to calculate equity volatility (default: 60)",
    )
    mc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton credit receipt as JSON",
    )

    # Subparser: hasbrouck-var
    hv_parser = subparsers.add_parser(
        "hasbrouck-var",
        help="Evaluate Joel Hasbrouck (1991) VAR price discovery and trade information share",
    )
    hv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    hv_parser.add_argument(
        "--lags",
        type=int,
        default=5,
        help="Number of autoregressive lags p in VAR (default: 5)",
    )
    hv_parser.add_argument(
        "--horizon",
        type=int,
        default=20,
        help="Impulse response function simulation horizon H (default: 20)",
    )
    hv_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    hv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hasbrouck receipt as JSON",
    )

    # Subparser: vrp
    vrp_parser = subparsers.add_parser(
        "vrp",
        help="Evaluate Bollerslev, Tauchen & Zhou (2009) variance risk premium & volatility-of-volatility",
    )
    vrp_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    vrp_parser.add_argument(
        "--iv",
        type=float,
        default=None,
        help="Annualized implied volatility (e.g. 0.28 for 28%; optional)",
    )
    vrp_parser.add_argument(
        "--horizon",
        type=int,
        default=21,
        help="Forward forecast horizon in sessions (default: 21)",
    )
    vrp_parser.add_argument(
        "--lookback",
        type=int,
        default=120,
        help="Lookback window in sessions (default: 120)",
    )
    vrp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output VRP receipt as JSON",
    )

    # Subparser: max-lottery
    ml_parser = subparsers.add_parser(
        "max-lottery",
        help="Evaluate Bali, Cakici & Whitelaw (2011) extreme daily return MAX effect & lottery demand",
    )
    ml_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    ml_parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of extreme upside days N to average for MAX_N (default: 5)",
    )
    ml_parser.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback observation window in sessions (default: 60)",
    )
    ml_parser.add_argument(
        "--json",
        action="store_true",
        help="Output MAX lottery receipt as JSON",
    )

    # Subparser: profitability
    gp_parser = subparsers.add_parser(
        "profitability",
        help="Evaluate Robert Novy-Marx (2013) gross profitability & Asness et al. (2019) QMJ quality",
    )
    gp_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    gp_parser.add_argument(
        "--revenue",
        type=float,
        default=None,
        help="Total annualized company revenue in dollars (optional)",
    )
    gp_parser.add_argument(
        "--cogs",
        type=float,
        default=None,
        help="Total Cost of Goods Sold in dollars (optional)",
    )
    gp_parser.add_argument(
        "--assets",
        type=float,
        default=None,
        help="Total Book Assets in dollars (optional)",
    )
    gp_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Total Debt Obligations in dollars (optional)",
    )
    gp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output profitability receipt as JSON",
    )

    # Subparser: dual-momentum
    dm_parser = subparsers.add_parser(
        "dual-momentum",
        help="Evaluate Jegadeesh & Titman (1993) / Antonacci Dual Momentum & Industry Relative Strength",
    )
    dm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dm_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    dm_parser.add_argument(
        "--rf-rate",
        type=float,
        default=4.0,
        help="Annualized risk-free hurdle rate percentage (default: 4.0)",
    )
    dm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output dual momentum receipt as JSON",
    )

    # Subparser: bab
    bab_parser = subparsers.add_parser(
        "bab",
        help="Evaluate Andrea Frazzini & Lasse Heje Pedersen (2014) Betting Against Beta & Leverage Multipliers",
    )
    bab_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    bab_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    bab_parser.add_argument(
        "--rf-rate",
        type=float,
        default=4.0,
        help="Annualized risk-free hurdle rate percentage (default: 4.0)",
    )
    bab_parser.add_argument(
        "--market-premium",
        type=float,
        default=6.0,
        help="Annualized expected equity risk premium ERP = E[Rm] - Rf (default: 6.0)",
    )
    bab_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BAB receipt as JSON",
    )

    # Subparser: reversal
    rev_parser = subparsers.add_parser(
        "reversal",
        help="Evaluate Narasimhan Jegadeesh (1990) & Bruce Lehmann (1990) Short-Term Reversal & Contrarian Alpha",
    )
    rev_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    rev_parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Reference benchmark ticker (default: SPY)",
    )
    rev_parser.add_argument(
        "--json",
        action="store_true",
        help="Output reversal receipt as JSON",
    )

    # Subparser: distress
    distress_parser = subparsers.add_parser(
        "distress",
        help="Evaluate Campbell, Hilscher & Szilagyi (2008) Financial Distress Risk Hazard & Insolvency Alpha",
    )
    distress_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    distress_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in dollars (optional)",
    )
    distress_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Total debt liabilities in dollars (optional)",
    )
    distress_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Annual net income in dollars (optional)",
    )
    distress_parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Cash and liquid reserves in dollars (optional)",
    )
    distress_parser.add_argument(
        "--json",
        action="store_true",
        help="Output distress receipt as JSON",
    )

    # Subparser: sentiment
    sent_parser = subparsers.add_parser(
        "sentiment",
        help="Evaluate Malcolm Baker & Jeffrey Wurgler (2006) Cross-Sectional Investor Sentiment & Limits to Arbitrage",
    )
    sent_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    sent_parser.add_argument(
        "--market-sentiment",
        type=float,
        default=0.0,
        help="Standardized macro investor sentiment index I_sent in [-3.0, +3.0] (default: 0.0)",
    )
    sent_parser.add_argument(
        "--net-margin",
        type=float,
        default=None,
        help="Net profit margin percentage (optional)",
    )
    sent_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in dollars (optional)",
    )
    sent_parser.add_argument(
        "--json",
        action="store_true",
        help="Output sentiment receipt as JSON",
    )

    # Subparser: arbitrage-asymmetry
    asym_parser = subparsers.add_parser(
        "arbitrage-asymmetry",
        aliases=["asymmetry", "short-impediment"],
        help="Evaluate Robert F. Stambaugh, Jianfeng Yu & Yu Yuan (2012, 2015) Short-Sale Impediments & Arbitrage Asymmetry",
    )
    asym_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. TSLA, GME, AAPL)",
    )
    asym_parser.add_argument(
        "--market-sentiment",
        type=float,
        default=0.0,
        help="Standardized macro investor sentiment index I_sent in [-3.0, +3.0] (default: 0.0)",
    )
    asym_parser.add_argument(
        "--borrow-fee",
        type=float,
        default=None,
        help="Annualized cost to borrow equity shares (%/yr) (optional)",
    )
    asym_parser.add_argument(
        "--short-interest",
        type=float,
        default=None,
        help="Short interest as percentage of free float (%) (optional)",
    )
    asym_parser.add_argument(
        "--days-to-cover",
        type=float,
        default=None,
        help="Short ratio / days to cover in sessions (optional)",
    )
    asym_parser.add_argument(
        "--json",
        action="store_true",
        help="Output arbitrage asymmetry receipt as JSON",
    )

    # Subparser: q-factor
    q_parser = subparsers.add_parser(
        "q-factor",
        aliases=["qfactor", "hou-xue-zhang"],
        help="Evaluate Kewei Hou, Chen Xue & Lu Zhang (2015, 2020) q-Factor Investment & Profitability Profile",
    )
    q_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT, TSLA)",
    )
    q_parser.add_argument(
        "--roe",
        type=float,
        default=None,
        help="Return on Equity percentage (Net Income / Book Equity) (optional)",
    )
    q_parser.add_argument(
        "--investment-rate",
        type=float,
        default=None,
        help="Annual Investment-to-Assets I/A percentage (Total Asset Growth %) (optional)",
    )
    q_parser.add_argument(
        "--asset-growth",
        type=float,
        default=None,
        help="Alias for --investment-rate",
    )
    q_parser.add_argument(
        "--tobin-q",
        type=float,
        default=None,
        help="Approximate Tobin's q ratio (optional)",
    )
    q_parser.add_argument(
        "--json",
        action="store_true",
        help="Output q-factor receipt as JSON",
    )

    # Subparser: accruals
    accrual_parser = subparsers.add_parser(
        "accruals",
        aliases=["accrual", "earnings-quality", "sloan"],
        help="Evaluate Richard G. Sloan (1996) Accrual Anomaly & Earnings Quality Decomposition",
    )
    accrual_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, BRK.B, TSLA)",
    )
    accrual_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Reported headline net income ($) (optional)",
    )
    accrual_parser.add_argument(
        "--operating-cash-flow",
        type=float,
        default=None,
        help="Operating cash flow from operations ($) (optional)",
    )
    accrual_parser.add_argument(
        "--cash-flow",
        type=float,
        default=None,
        help="Alias for --operating-cash-flow",
    )
    accrual_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Average total assets ($) (optional)",
    )
    accrual_parser.add_argument(
        "--json",
        action="store_true",
        help="Output accrual anomaly receipt as JSON",
    )

    # Subparser: beneish
    beneish_parser = subparsers.add_parser(
        "beneish",
        aliases=["mscore", "forensic-audit", "fraud-risk"],
        help="Evaluate Messod D. Beneish (1999) Forensic Accounting M-Score & Manipulation Risk",
    )
    beneish_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. ENRN, TSLA, AAPL)",
    )
    beneish_parser.add_argument(
        "--dsri",
        type=float,
        default=None,
        help="Days' Sales in Receivables Index (optional)",
    )
    beneish_parser.add_argument(
        "--gmi",
        type=float,
        default=None,
        help="Gross Margin Index (optional)",
    )
    beneish_parser.add_argument(
        "--aqi",
        type=float,
        default=None,
        help="Asset Quality Index (optional)",
    )
    beneish_parser.add_argument(
        "--sgi",
        type=float,
        default=None,
        help="Sales Growth Index (optional)",
    )
    beneish_parser.add_argument(
        "--depi",
        type=float,
        default=None,
        help="Depreciation Index (optional)",
    )
    beneish_parser.add_argument(
        "--sgai",
        type=float,
        default=None,
        help="SG&A Expense Index (optional)",
    )
    beneish_parser.add_argument(
        "--lvgi",
        type=float,
        default=None,
        help="Leverage Index (optional)",
    )
    beneish_parser.add_argument(
        "--tata",
        type=float,
        default=None,
        help="Total Accruals to Total Assets (optional)",
    )
    beneish_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Beneish M-Score receipt as JSON",
    )

    # Subparser: fscore
    fscore_parser = subparsers.add_parser(
        "fscore",
        aliases=["piotroski", "fundamental-health", "value-quality"],
        help="Evaluate Joseph D. Piotroski (2000) 9-Point Fundamental Accounting F-Score",
    )
    fscore_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. BRK.B, JNJ, TSLA)",
    )
    fscore_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Current net income ($) (optional)",
    )
    fscore_parser.add_argument(
        "--operating-cash-flow",
        type=float,
        default=None,
        help="Operating cash flow ($) (optional)",
    )
    fscore_parser.add_argument(
        "--cash-flow",
        type=float,
        default=None,
        help="Alias for --operating-cash-flow",
    )
    fscore_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets ($) (optional)",
    )
    fscore_parser.add_argument(
        "--roa",
        type=float,
        default=None,
        help="Return on assets % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-roa",
        type=float,
        default=None,
        help="Change in return on assets % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-debt",
        type=float,
        default=None,
        help="Change in long-term debt ($) (negative = debt repaid) (optional)",
    )
    fscore_parser.add_argument(
        "--delta-current-ratio",
        type=float,
        default=None,
        help="Change in current ratio (positive = liquidity expanded) (optional)",
    )
    fscore_parser.add_argument(
        "--shares-dilution",
        type=float,
        default=None,
        help="Share dilution % (negative = repurchased) (optional)",
    )
    fscore_parser.add_argument(
        "--delta-gross-margin",
        type=float,
        default=None,
        help="Change in gross margin % (optional)",
    )
    fscore_parser.add_argument(
        "--delta-asset-turnover",
        type=float,
        default=None,
        help="Change in asset turnover (optional)",
    )
    fscore_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Piotroski F-Score receipt as JSON",
    )

    # Hong & Stein (1999) Gradual Information Diffusion Parser
    diffusion_parser = subparsers.add_parser(
        "diffusion",
        aliases=["hong-stein", "information-diffusion", "momentum-regime"],
        help="Evaluate Hong & Stein (1999) gradual information diffusion and trend maturity",
    )
    diffusion_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    diffusion_parser.add_argument(
        "--analyst-coverage",
        type=int,
        default=None,
        help="Number of equity research analysts covering the firm (optional)",
    )
    diffusion_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hong-Stein diffusion receipt as JSON",
    )

    # Lakonishok, Shleifer & Vishny (1994) Contrarian Investment Parser
    contrarian_parser = subparsers.add_parser(
        "contrarian",
        aliases=["lsv", "extrapolation-bias", "glamour-overpricing"],
        help="Evaluate Lakonishok, Shleifer & Vishny (1994) contrarian investment and extrapolation bias",
    )
    contrarian_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    contrarian_parser.add_argument(
        "--sales-growth",
        type=float,
        default=None,
        help="Annualized 3-year sales growth rate % (optional)",
    )
    contrarian_parser.add_argument(
        "--earnings-growth",
        type=float,
        default=None,
        help="Annualized 3-year net income growth rate % (optional)",
    )
    contrarian_parser.add_argument(
        "--pe",
        type=float,
        default=None,
        help="Price-to-earnings (P/E) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--pcf",
        type=float,
        default=None,
        help="Price-to-cash-flow (P/CF) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--pb",
        type=float,
        default=None,
        help="Price-to-book (P/B) ratio (optional)",
    )
    contrarian_parser.add_argument(
        "--json",
        action="store_true",
        help="Output LSV Contrarian receipt as JSON",
    )

    # Daniel, Hirshleifer & Subrahmanyam (1998) Overconfidence & Self-Attribution Parser
    dhs_parser = subparsers.add_parser(
        "dhs",
        aliases=["overconfidence", "self-attribution", "behavioral-lifecycle"],
        help="Evaluate Daniel, Hirshleifer & Subrahmanyam (1998) overconfidence and self-attribution",
    )
    dhs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA)",
    )
    dhs_parser.add_argument(
        "--earnings-surprise",
        type=float,
        default=None,
        help="Public earnings surprise % (optional)",
    )
    dhs_parser.add_argument(
        "--dispersion",
        type=float,
        default=None,
        help="Analyst EPS forecast dispersion % (optional)",
    )
    dhs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output DHS Behavioral receipt as JSON",
    )

    # Barberis, Shleifer & Vishny (1998) Dual-Heuristic Sentiment Parser
    bsv_parser = subparsers.add_parser(
        "bsv",
        aliases=["barberis", "conservatism-representativeness", "dual-heuristic"],
        help="Evaluate Barberis, Shleifer & Vishny (1998) conservatism and representativeness sentiment",
    )
    bsv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, MSFT)",
    )
    bsv_parser.add_argument(
        "--earnings-surprise",
        type=float,
        default=None,
        help="Latest quarterly earnings surprise % (optional)",
    )
    bsv_parser.add_argument(
        "--streak",
        type=int,
        default=None,
        help="Consecutive quarters with earnings surprise in same direction (optional)",
    )
    bsv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BSV Sentiment receipt as JSON",
    )

    # Eberhart et al. (2004) / Cohen et al. (2013) Innovation Capital Parser
    innovation_parser = subparsers.add_parser(
        "innovation",
        aliases=["rd-anomaly", "knowledge-capital", "rnd-intensity"],
        help="Evaluate Eberhart et al. (2004) / Cohen et al. (2013) innovation capital and R&D misvaluation",
    )
    innovation_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, BMY)",
    )
    innovation_parser.add_argument(
        "--rd-expense",
        type=float,
        default=None,
        help="Annual R&D expenditure in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--revenues",
        type=float,
        default=None,
        help="Annual revenues in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets in dollars (optional)",
    )
    innovation_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Innovation Capital receipt as JSON",
    )

    # Charles M. C. Lee & Bhaskaran Swaminathan (2000) Momentum Life Cycle Parser
    lifecycle_parser = subparsers.add_parser(
        "lifecycle",
        aliases=["lee-swaminathan", "momentum-volume", "mlc-stage"],
        help="Evaluate Lee & Swaminathan (2000) price momentum and trading volume life-cycle",
    )
    lifecycle_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    lifecycle_parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        help="Daily turnover ratio % override (optional)",
    )
    lifecycle_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Momentum Life Cycle receipt as JSON",
    )

    # Edward I. Altman (1968, 2000) Z-Score & James A. Ohlson (1980) O-Score Parser
    altman_parser = subparsers.add_parser(
        "altman",
        aliases=["zscore", "ohlson", "distress-risk"],
        help="Evaluate Altman (1968, 2000) Z-Score and Ohlson (1980) O-Score structural solvency",
    )
    altman_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    altman_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Total assets in $M (optional)",
    )
    altman_parser.add_argument(
        "--total-liabilities",
        type=float,
        default=None,
        help="Total liabilities in $M (optional)",
    )
    altman_parser.add_argument(
        "--working-capital",
        type=float,
        default=None,
        help="Working capital in $M (optional)",
    )
    altman_parser.add_argument(
        "--retained-earnings",
        type=float,
        default=None,
        help="Retained earnings in $M (optional)",
    )
    altman_parser.add_argument(
        "--ebit",
        type=float,
        default=None,
        help="EBIT operating profit in $M (optional)",
    )
    altman_parser.add_argument(
        "--sales",
        type=float,
        default=None,
        help="Total sales / revenues in $M (optional)",
    )
    altman_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Market capitalization in $M (optional)",
    )
    altman_parser.add_argument(
        "--current-assets",
        type=float,
        default=None,
        help="Current assets in $M (optional)",
    )
    altman_parser.add_argument(
        "--current-liabilities",
        type=float,
        default=None,
        help="Current liabilities in $M (optional)",
    )
    altman_parser.add_argument(
        "--net-income",
        type=float,
        default=None,
        help="Net income in $M (optional)",
    )
    altman_parser.add_argument(
        "--funds-from-operations",
        type=float,
        default=None,
        help="Funds from operations / CFO in $M (optional)",
    )
    altman_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Altman Z-Score & Ohlson O-Score receipt as JSON",
    )

    # Clifford S. Asness, Tobias J. Moskowitz & Lasse Heje Pedersen (2013) VME Parser
    vme_parser = subparsers.add_parser(
        "vme",
        aliases=["value-momentum", "asness-moskowitz", "vme-everywhere"],
        help="Evaluate Asness, Moskowitz & Pedersen (2013) 'Value and Momentum Everywhere' (VME)",
    )
    vme_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    vme_parser.add_argument(
        "--book-to-market",
        type=float,
        default=None,
        help="Book-to-Market equity ratio B/M (optional)",
    )
    vme_parser.add_argument(
        "--earnings-yield",
        type=float,
        default=None,
        help="Earnings-to-Price yield E/P % (optional)",
    )
    vme_parser.add_argument(
        "--correlation",
        type=float,
        default=None,
        help="Value-Momentum correlation rho override in [-0.95, 0.95] (optional)",
    )
    vme_parser.add_argument(
        "--json",
        action="store_true",
        help="Output VME receipt as JSON",
    )

    # Victor L. Bernard & Jacob K. Thomas (1989, 1990) PEAD & SUE Parser
    pead_parser = subparsers.add_parser(
        "pead",
        aliases=["sue", "bernard-thomas", "earnings-drift"],
        help="Evaluate Bernard & Thomas (1989, 1990) Post-Earnings-Announcement Drift (PEAD) & SUE",
    )
    pead_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    pead_parser.add_argument(
        "--current-eps",
        type=float,
        default=None,
        help="Current quarterly diluted EPS (optional)",
    )
    pead_parser.add_argument(
        "--year-ago-eps",
        type=float,
        default=None,
        help="Same quarter prior-year diluted EPS (optional)",
    )
    pead_parser.add_argument(
        "--surprise-std",
        type=float,
        default=None,
        help="Standard deviation of seasonal EPS changes (optional)",
    )
    pead_parser.add_argument(
        "--days-since",
        type=int,
        default=None,
        help="Trading sessions elapsed since earnings release (optional)",
    )
    pead_parser.add_argument(
        "--json",
        action="store_true",
        help="Output PEAD & SUE receipt as JSON",
    )

    # Cooper, Gulen & Schill (2008) / Titman, Wei & Xie (2004) Asset Growth & Empire-Building Parser
    empire_parser = subparsers.add_parser(
        "empire",
        aliases=["asset-growth", "cooper-gulen", "titman-capex"],
        help="Evaluate Cooper et al. (2008) & Titman et al. (2004) Asset Growth & Empire-Building",
    )
    empire_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    empire_parser.add_argument(
        "--current-assets",
        type=float,
        default=None,
        help="Current total assets in $M (optional)",
    )
    empire_parser.add_argument(
        "--prior-assets",
        type=float,
        default=None,
        help="Prior total assets in $M (optional)",
    )
    empire_parser.add_argument(
        "--current-capex",
        type=float,
        default=None,
        help="Current CapEx-to-Sales ratio % (optional)",
    )
    empire_parser.add_argument(
        "--historical-capex",
        type=float,
        default=None,
        help="Trailing 3-year average CapEx-to-Sales ratio % (optional)",
    )
    empire_parser.add_argument(
        "--fcf-to-assets",
        type=float,
        default=None,
        help="Free cash flow to total assets ratio % (optional)",
    )
    empire_parser.add_argument(
        "--debt-to-assets",
        type=float,
        default=None,
        help="Total debt to total assets ratio % (optional)",
    )
    empire_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Asset Growth & Empire-Building receipt as JSON",
    )

    # Boudoukh et al. (2007) / Pontiff & Woodgate (2008) Net Payout Yield & Share Issuance Parser
    payout_parser = subparsers.add_parser(
        "payout",
        aliases=["net-payout", "boudoukh", "share-issuance"],
        help="Evaluate Boudoukh et al. (2007) & Pontiff-Woodgate (2008) Net Payout Yield & Share Issuance",
    )
    payout_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    payout_parser.add_argument(
        "--market-cap",
        type=float,
        default=None,
        help="Total market capitalization in $M (optional)",
    )
    payout_parser.add_argument(
        "--dividends",
        type=float,
        default=None,
        help="Annual cash dividends in $M (optional)",
    )
    payout_parser.add_argument(
        "--repurchases",
        type=float,
        default=None,
        help="Annual gross share repurchases in $M (optional)",
    )
    payout_parser.add_argument(
        "--issuance",
        type=float,
        default=None,
        help="Annual gross equity issuance in $M (optional)",
    )
    payout_parser.add_argument(
        "--current-shares",
        type=float,
        default=None,
        help="Current split-adjusted diluted shares in M (optional)",
    )
    payout_parser.add_argument(
        "--prior-shares",
        type=float,
        default=None,
        help="Prior split-adjusted diluted shares in M (optional)",
    )
    payout_parser.add_argument(
        "--free-cash-flow",
        type=float,
        default=None,
        help="Annual free cash flow in $M (optional)",
    )
    payout_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Net Payout Yield receipt as JSON",
    )

    # Hirshleifer, Hou, Teoh & Zhang (2004) Net Operating Assets (NOA) & Balance Sheet Bloat Parser
    noa_parser = subparsers.add_parser(
        "noa",
        aliases=["bloat", "hirshleifer-teoh", "balance-sheet-bloat"],
        help="Evaluate Hirshleifer, Hou, Teoh & Zhang (2004) Net Operating Assets (NOA) & Balance Sheet Bloat",
    )
    noa_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g. AAPL, NVDA, TSLA)",
    )
    noa_parser.add_argument(
        "--total-assets",
        type=float,
        default=None,
        help="Current total assets in $M (optional)",
    )
    noa_parser.add_argument(
        "--prior-assets",
        type=float,
        default=None,
        help="Prior total assets in $M (optional)",
    )
    noa_parser.add_argument(
        "--cash",
        type=float,
        default=None,
        help="Cash and short-term investments in $M (optional)",
    )
    noa_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Total debt in $M (optional)",
    )
    noa_parser.add_argument(
        "--equity",
        type=float,
        default=None,
        help="Shareholders' book equity in $M (optional)",
    )
    noa_parser.add_argument(
        "--prior-snoa",
        type=float,
        default=None,
        help="Prior-year Scaled NOA ratio (optional)",
    )
    noa_parser.add_argument(
        "--json",
        action="store_true",
        help="Output NOA Bloat receipt as JSON",
    )

    dispersion_parser = subparsers.add_parser(
        "dispersion",
        aliases=["diether-malloy", "miller-opinion", "forecast-dispersion"],
        help="Run Diether, Malloy & Scherbina (2002) & Miller (1977) Analyst Forecast Dispersion Engine",
    )
    dispersion_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    dispersion_parser.add_argument(
        "--mean-eps",
        type=float,
        default=None,
        help="Consensus mean EPS forecast in $ (optional; defaults to return-proxy estimation)",
    )
    dispersion_parser.add_argument(
        "--std-eps",
        type=float,
        default=None,
        help="Standard deviation across analyst EPS forecasts in $ (optional)",
    )
    dispersion_parser.add_argument(
        "--analysts",
        type=int,
        default=None,
        help="Number of analysts covering the firm (optional)",
    )
    dispersion_parser.add_argument(
        "--borrow-fee",
        type=float,
        default=None,
        help="Annualized stock loan borrow fee in % (optional)",
    )
    dispersion_parser.add_argument(
        "--inst-ownership",
        type=float,
        default=None,
        help="Institutional ownership fraction in [0, 1] (optional)",
    )
    dispersion_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Belief Dispersion receipt as JSON",
    )

    catering_parser = subparsers.add_parser(
        "catering",
        aliases=["polk-sapienza", "catering-capex", "investment-catering"],
        help="Run Polk & Sapienza (2009) Catering Theory of Investment & Valuation Mispricing Engine",
    )
    catering_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    catering_parser.add_argument(
        "--capex",
        type=float,
        default=None,
        help="Annual capital expenditures in $M (optional)",
    )
    catering_parser.add_argument(
        "--ppent-lag",
        type=float,
        default=None,
        help="Lagged net Property, Plant & Equipment K_{t-1} in $M (optional)",
    )
    catering_parser.add_argument(
        "--tobins-q",
        type=float,
        default=None,
        help="Tobin's Q ratio (optional)",
    )
    catering_parser.add_argument(
        "--cash-flow-ratio",
        type=float,
        default=None,
        help="Operating cash flow scaled by lagged net PPE (optional)",
    )
    catering_parser.add_argument(
        "--mispricing-score",
        type=float,
        default=None,
        help="Discretionary accrual / valuation mispricing score in [-1, 1] (optional)",
    )
    catering_parser.add_argument(
        "--turnover-pct",
        type=float,
        default=None,
        help="Annualized share turnover in % (optional)",
    )
    catering_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Catering Investment receipt as JSON",
    )

    commonality_parser = subparsers.add_parser(
        "commonality",
        aliases=["chordia-roll", "liquidity-commonality", "order-imbalance"],
        help="Run Chordia, Roll & Subrahmanyam (2000, 2001) Commonality in Liquidity & Order Imbalance Engine",
    )
    commonality_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    commonality_parser.add_argument(
        "--spread-beta",
        type=float,
        default=None,
        help="Spread commonality beta (optional)",
    )
    commonality_parser.add_argument(
        "--depth-beta",
        type=float,
        default=None,
        help="Depth commonality beta (optional)",
    )
    commonality_parser.add_argument(
        "--liquidity-r2",
        type=float,
        default=None,
        help="Market liquidity model R^2 in [0, 1] (optional)",
    )
    commonality_parser.add_argument(
        "--order-imbalance",
        type=float,
        default=None,
        help="Signed volume order imbalance OIB_t in [-1, 1] (optional)",
    )
    commonality_parser.add_argument(
        "--oib-autocorr",
        type=float,
        default=None,
        help="First-order OIB autocorrelation in [-1, 1] (optional)",
    )
    commonality_parser.add_argument(
        "--down-market-surge",
        type=float,
        default=None,
        help="Asymmetric down-market illiquidity surge multiplier >= 0.5 (optional)",
    )
    commonality_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Liquidity Commonality receipt as JSON",
    )

    supply_chain_parser = subparsers.add_parser(
        "supply-chain",
        aliases=["cohen-frazzini", "customer-momentum", "economic-links"],
        help="Run Cohen & Frazzini (2008) & Ahern (2013) Economic Links & Supply-Chain Customer Momentum Engine",
    )
    supply_chain_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    supply_chain_parser.add_argument(
        "--customer-return",
        type=float,
        default=None,
        help="Principal customer return over lookback window in decimal (e.g., 0.08 for +8%) (optional)",
    )
    supply_chain_parser.add_argument(
        "--supplier-return",
        type=float,
        default=None,
        help="Supplier own return over lookback window in decimal (optional)",
    )
    supply_chain_parser.add_argument(
        "--revenue-share",
        type=float,
        default=None,
        help="Customer revenue dependency share in [0, 1] (optional)",
    )
    supply_chain_parser.add_argument(
        "--shared-coverage",
        type=float,
        default=None,
        help="Shared analyst / fund co-coverage ratio in [0, 1] (optional)",
    )
    supply_chain_parser.add_argument(
        "--network-centrality",
        type=float,
        default=None,
        help="Ahern (2013) production network eigenvector centrality in [0, 1] (optional)",
    )
    supply_chain_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Supply-Chain Momentum receipt as JSON",
    )

    governance_parser = subparsers.add_parser(
        "governance",
        aliases=["gompers-ishii", "bebchuk-entrenchment", "g-index", "e-index"],
        help="Run Gompers, Ishii & Metrick (2003) & Bebchuk et al. (2009) Corporate Governance Engine",
    )
    governance_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    governance_parser.add_argument(
        "--e-index",
        type=int,
        default=None,
        help="Bebchuk Entrenchment Index in [0, 6] (optional)",
    )
    governance_parser.add_argument(
        "--g-index",
        type=int,
        default=None,
        help="Gompers Governance Index in [0, 24] (>= e-index) (optional)",
    )
    governance_parser.add_argument(
        "--dual-class-wedge",
        type=float,
        default=None,
        help="Dual-class voting control wedge in [0, 0.95] (optional)",
    )
    governance_parser.add_argument(
        "--board-independence",
        type=float,
        default=None,
        help="Independent board director fraction in [0, 1] (optional)",
    )
    governance_parser.add_argument(
        "--fcf-retention",
        type=float,
        default=None,
        help="Unreturned free cash flow retention fraction in [0, 1] (optional)",
    )
    governance_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Corporate Governance receipt as JSON",
    )

    fip_parser = subparsers.add_parser(
        "fip",
        aliases=["frog-in-the-pan", "da-gurun", "information-discreteness"],
        help="Run Da, Gurun & Warachka (2014) 'Frog in the Pan' Information Discreteness Engine",
    )
    fip_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    fip_parser.add_argument(
        "--pret",
        type=float,
        default=None,
        help="Past cumulative return in decimal (e.g., 0.18 for +18%) (optional)",
    )
    fip_parser.add_argument(
        "--pct-positive",
        type=float,
        default=None,
        help="Fraction of positive return days in [0, 1] (optional)",
    )
    fip_parser.add_argument(
        "--pct-negative",
        type=float,
        default=None,
        help="Fraction of negative return days in [0, 1] (optional)",
    )
    fip_parser.add_argument(
        "--max-jump-share",
        type=float,
        default=None,
        help="Fraction of total absolute daily movement in top 3 days in [0, 1] (optional)",
    )
    fip_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Frog in the Pan receipt as JSON",
    )

    attention_parser = subparsers.add_parser(
        "attention",
        aliases=["barber-odean", "da-engelberg", "svi-attention", "attention-spike"],
        help="Run Barber & Odean (2008) & Da, Engelberg & Gao (2011) Retail Attention Salience Engine",
    )
    attention_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    attention_parser.add_argument(
        "--asvi",
        type=float,
        default=None,
        help="Abnormal Search Volume Index log change in [-3.0, 5.0] (optional)",
    )
    attention_parser.add_argument(
        "--abnormal-volume",
        type=float,
        default=None,
        help="Abnormal trading volume ratio > 0 (optional)",
    )
    attention_parser.add_argument(
        "--extreme-return",
        type=float,
        default=None,
        help="Most salient recent 1-day return in decimal (optional)",
    )
    attention_parser.add_argument(
        "--retail-buy-imbalance",
        type=float,
        default=None,
        help="Net retail buy order imbalance in [-1, 1] (optional)",
    )
    attention_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Attention Salience receipt as JSON",
    )

    bm_parser = subparsers.add_parser(
        "benchmark-mandate",
        aliases=["baker-bradley", "tracking-error-limit", "low-vol-mandate"],
        help="Run Baker, Bradley & Wurgler (2011) Institutional Benchmark Mandate & Tracking-Error Engine",
    )
    bm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    bm_parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Stock CAPM beta in [-1.0, 4.0] (optional)",
    )
    bm_parser.add_argument(
        "--market-vol",
        type=float,
        default=None,
        help="Annualized market volatility in (0, 2.0] (optional)",
    )
    bm_parser.add_argument(
        "--idio-vol",
        type=float,
        default=None,
        help="Annualized idiosyncratic volatility in (0, 3.0] (optional)",
    )
    bm_parser.add_argument(
        "--market-premium",
        type=float,
        default=None,
        help="Expected market risk premium in % (optional)",
    )
    bm_parser.add_argument(
        "--inst-benchmarked-share",
        type=float,
        default=None,
        help="Benchmarked institutional ownership share in [0, 1] (optional)",
    )
    bm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Benchmark Mandate receipt as JSON",
    )

    cgo_parser = subparsers.add_parser(
        "cgo",
        aliases=["grinblatt-han", "frazzini-disposition", "disposition-effect", "capital-gains-overhang"],
        help="Run Grinblatt & Han (2005) & Frazzini (2006) Capital Gains Overhang & Disposition Effect Engine",
    )
    cgo_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    cgo_parser.add_argument(
        "--reference-price",
        type=float,
        default=None,
        help="Turnover-weighted aggregate reference cost basis R_t in $ (> 0) (optional)",
    )
    cgo_parser.add_argument(
        "--daily-turnover",
        type=float,
        default=None,
        help="Average daily share turnover fraction in (0, 0.50] (optional)",
    )
    cgo_parser.add_argument(
        "--news-surprise",
        type=float,
        default=None,
        help="Standardized fundamental news surprise score in [-5, 5] (optional)",
    )
    cgo_parser.add_argument(
        "--institutional-disposition-shield",
        type=float,
        default=None,
        help="Institutional rationality shield against disposition effect in [0, 1] (optional)",
    )
    cgo_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Capital Gains Overhang receipt as JSON",
    )

    high52_parser = subparsers.add_parser(
        "high52",
        aliases=["george-hwang", "52-week-high", "anchoring-momentum"],
        help="Run George & Hwang (2004) 52-Week High Psychological Anchoring & Nearness-to-High Momentum Engine",
    )
    high52_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    high52_parser.add_argument(
        "--high-52w",
        type=float,
        default=None,
        help="Rolling 252-day (52-week) maximum price H_{52,t} in $ (>= P_t) (optional)",
    )
    high52_parser.add_argument(
        "--low-52w",
        type=float,
        default=None,
        help="Rolling 252-day (52-week) minimum price L_{52,t} in $ (<= P_t) (optional)",
    )
    high52_parser.add_argument(
        "--past-return",
        type=float,
        default=None,
        help="Standard intermediate past return PRET_t in (-0.95, 5.0] (optional)",
    )
    high52_parser.add_argument(
        "--breakout-volume-ratio",
        type=float,
        default=None,
        help="Ratio of recent volume to 52-week average volume in [0.1, 10.0] (optional)",
    )
    high52_parser.add_argument(
        "--json",
        action="store_true",
        help="Output 52-Week High Anchoring receipt as JSON",
    )

    pv_parser = subparsers.add_parser(
        "pastor-veronesi",
        aliases=["learning-uncertainty", "profitability-learning", "uncertainty-convexity"],
        help="Run Pástor & Veronesi (2003, 2006) Bayesian Learning about Profitability & Uncertainty Convexity Engine",
    )
    pv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    pv_parser.add_argument(
        "--expected-excess-profitability",
        type=float,
        default=None,
        help="Expected excess ROE over cost of equity psi_hat in [-0.50, 0.80] (optional)",
    )
    pv_parser.add_argument(
        "--prior-uncertainty",
        type=float,
        default=None,
        help="Prior profitability uncertainty sigma_{psi,0} in [0.005, 0.50] (optional)",
    )
    pv_parser.add_argument(
        "--firm-age-years",
        type=float,
        default=None,
        help="Effective firm learning age T_age in [0.5, 100.0] years (optional)",
    )
    pv_parser.add_argument(
        "--retention-ratio",
        type=float,
        default=None,
        help="Earnings retention ratio b_ret in [0.0, 1.0] (optional)",
    )
    pv_parser.add_argument(
        "--observed-mb",
        type=float,
        default=None,
        help="Observed Market-to-Book ratio (M/B)_obs in [0.20, 50.0] (optional)",
    )
    pv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Pástor-Veronesi Uncertainty receipt as JSON",
    )

    lrr_parser = subparsers.add_parser(
        "lrr",
        aliases=["bansal-yaron", "long-run-risks", "epstein-zin"],
        help="Run Bansal & Yaron (2004) Long-Run Risks (LRR), Persistent Cash-Flow Growth & Uncertainty Pricing Engine",
    )
    lrr_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    lrr_parser.add_argument(
        "--cashflow-growth-beta",
        type=float,
        default=None,
        help="Persistent cash-flow growth beta beta_{x,i} in [-2.0, 4.0] (optional)",
    )
    lrr_parser.add_argument(
        "--uncertainty-shock-beta",
        type=float,
        default=None,
        help="Macroeconomic volatility shock beta beta_{sigma,i} in [-3.0, 3.0] (optional)",
    )
    lrr_parser.add_argument(
        "--growth-persistence",
        type=float,
        default=None,
        help="Long-run growth persistence rho_x in [0.50, 0.995] (optional)",
    )
    lrr_parser.add_argument(
        "--ies",
        type=float,
        default=None,
        help="Epstein-Zin intertemporal elasticity of substitution psi_IES in [0.50, 3.0] (optional)",
    )
    lrr_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=None,
        help="Epstein-Zin relative risk aversion gamma_RRA in [1.5, 20.0] (optional)",
    )
    lrr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Long-Run Risks receipt as JSON",
    )

    rev_parser = subparsers.add_parser(
        "reversibility",
        aliases=["lu-zhang", "costly-reversibility", "asymmetric-capital"],
        help="Run Lu Zhang (2005) Costly Reversibility of Capital, Asymmetric Adjustment Costs & Value Beta Engine",
    )
    rev_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    rev_parser.add_argument(
        "--book-to-market",
        type=float,
        default=None,
        help="Book-to-Market equity ratio (B/M)_t in [0.05, 5.0] (optional)",
    )
    rev_parser.add_argument(
        "--cost-asymmetry",
        type=float,
        default=None,
        help="Downward-to-upward capital adjustment cost ratio theta^-/theta^+ in [1.0, 25.0] (optional)",
    )
    rev_parser.add_argument(
        "--fixed-capital-intensity",
        type=float,
        default=None,
        help="Operating fixed capital overhead f_K in [0.0, 1.0] (optional)",
    )
    rev_parser.add_argument(
        "--macro-state-zscore",
        type=float,
        default=None,
        help="Standardized macroeconomic state S_macro in [-3.0, 3.0] (< 0 recession) (optional)",
    )
    rev_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Costly Reversibility receipt as JSON",
    )

    cochrane_parser = subparsers.add_parser(
        "cochrane",
        aliases=["campbell-shiller", "discount-rate-decomposition", "present-value-identity"],
        help="Run Campbell-Shiller (1988) & Cochrane (2008, 2011) Present-Value & Discount-Rate Decomposition Engine",
    )
    cochrane_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    cochrane_parser.add_argument(
        "--log-dividend-price",
        type=float,
        default=None,
        help="Current log dividend/payout-to-price ratio dp_t in [-6.0, 0.0] (optional)",
    )
    cochrane_parser.add_argument(
        "--mean-log-dividend-price",
        type=float,
        default=None,
        help="Long-run mean log yield bar{dp} in [-6.0, 0.0] (optional)",
    )
    cochrane_parser.add_argument(
        "--valuation-persistence",
        type=float,
        default=None,
        help="AR(1) persistence of valuation ratio phi_{dp} in [0.50, 0.995] (optional)",
    )
    cochrane_parser.add_argument(
        "--return-slope",
        type=float,
        default=None,
        help="1-period return forecasting slope b_r^{(1)} in [-0.50, 1.0] (optional)",
    )
    cochrane_parser.add_argument(
        "--cashflow-slope",
        type=float,
        default=None,
        help="1-period cashflow growth forecasting slope b_d^{(1)} (optional)",
    )
    cochrane_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Campbell-Shiller Cochrane receipt as JSON",
    )

    stub_parser = subparsers.add_parser(
        "stub",
        aliases=["lamont-thaler", "stub-arbitrage", "equity-carveout", "negative-stub"],
        help="Run Lamont & Thaler (2003) & Mitchell-Pulvino-Stafford (2002) Stub Value Arbitrage Engine",
    )
    stub_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target parent stock ticker symbol (e.g., COMS)",
    )
    stub_parser.add_argument(
        "--subsidiary-ticker",
        type=str,
        default=None,
        help="Carved-out subsidiary stock ticker symbol (e.g., PALM) (optional)",
    )
    stub_parser.add_argument(
        "--subsidiary-price",
        type=float,
        default=None,
        help="Carved subsidiary market price P_{sub,t} in $ (> 0) (optional)",
    )
    stub_parser.add_argument(
        "--shares-per-parent",
        type=float,
        default=None,
        help="Subsidiary shares owned per parent share alpha in [0.01, 10.0] (optional)",
    )
    stub_parser.add_argument(
        "--borrow-fee",
        type=float,
        default=None,
        help="Annualized subsidiary short borrowing fee c_{borrow} in [0.0, 100.0]% (optional)",
    )
    stub_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Stub Value Arbitrage receipt as JSON",
    )

    spiral_parser = subparsers.add_parser(
        "liquidity-spiral",
        aliases=["brunnermeier-pedersen", "margin-spiral", "funding-liquidity", "fire-sale"],
        help="Run Brunnermeier & Pedersen (2009) Market Liquidity & Funding Liquidity Spirals Engine",
    )
    spiral_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    spiral_parser.add_argument(
        "--margin-haircut",
        type=float,
        default=None,
        help="Current margin haircut requirement m_t in [0.02, 0.85] (optional)",
    )
    spiral_parser.add_argument(
        "--baseline-margin",
        type=float,
        default=None,
        help="Baseline normal-times margin haircut m_0 in [0.02, 0.50] (optional)",
    )
    spiral_parser.add_argument(
        "--funding-buffer",
        type=float,
        default=None,
        help="Trader equity margin buffer B_fund in [-1.0, 1.0] (< 0 margin call) (optional)",
    )
    spiral_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Current annualized return volatility sigma_t in [0.05, 1.50] (optional)",
    )
    spiral_parser.add_argument(
        "--baseline-volatility",
        type=float,
        default=None,
        help="Baseline annualized return volatility sigma_0 in [0.05, 1.00] (optional)",
    )
    spiral_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Liquidity Spiral receipt as JSON",
    )

    intermediary_parser = subparsers.add_parser(
        "intermediary",
        aliases=["he-kelly-manela", "dealer-capital", "intermediary-capital", "adrian-shin"],
        help="Run He, Kelly & Manela (2017) & Adrian-Etula-Muir (2014) Intermediary Asset Pricing Engine",
    )
    intermediary_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    intermediary_parser.add_argument(
        "--dealer-capital-ratio",
        type=float,
        default=None,
        help="Current primary dealer equity capital ratio eta_t in [0.01, 0.40] (optional)",
    )
    intermediary_parser.add_argument(
        "--baseline-capital-ratio",
        type=float,
        default=None,
        help="Baseline normal-times dealer capital ratio eta_0 in [0.03, 0.35] (optional)",
    )
    intermediary_parser.add_argument(
        "--dealer-leverage",
        type=float,
        default=None,
        help="Current intermediary assets-to-equity leverage in [2.5, 40.0] (optional)",
    )
    intermediary_parser.add_argument(
        "--baseline-leverage",
        type=float,
        default=None,
        help="Baseline normal-times dealer leverage in [5.0, 30.0] (optional)",
    )
    intermediary_parser.add_argument(
        "--capital-beta",
        type=float,
        default=None,
        help="Sensitivity to dealer capital ratio shocks beta_{eta,i} in [-2.0, 4.0] (optional)",
    )
    intermediary_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Intermediary Asset Pricing receipt as JSON",
    )

    margin_parser = subparsers.add_parser(
        "margin-pricing",
        aliases=["garleanu-pedersen", "margin-capm", "basis-spread", "margin-constraints"],
        help="Run Gârleanu & Pedersen (2011) Margin-Based Asset Pricing & Leverage Constraint Engine",
    )
    margin_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    margin_parser.add_argument(
        "--margin-requirement",
        type=float,
        default=None,
        help="Asset haircut / margin requirement m_i in [0.05, 0.90] (optional)",
    )
    margin_parser.add_argument(
        "--baseline-margin",
        type=float,
        default=None,
        help="Normal-times baseline margin requirement in [0.05, 0.50] (optional)",
    )
    margin_parser.add_argument(
        "--shadow-cost",
        type=float,
        default=None,
        help="Margin constraint Lagrange multiplier shadow cost psi in [0.0, 0.25] (optional)",
    )
    margin_parser.add_argument(
        "--market-beta",
        type=float,
        default=None,
        help="Systematic market beta beta_i in [-1.0, 3.5] (optional)",
    )
    margin_parser.add_argument(
        "--market-risk-premium",
        type=float,
        default=None,
        help="Market risk premium lambda_m (%) in [2.0, 12.0] (optional)",
    )
    margin_parser.add_argument(
        "--risk-free-rate",
        type=float,
        default=None,
        help="Risk-free rate R_f (%) in [0.0, 10.0] (optional)",
    )
    margin_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Margin Asset Pricing receipt as JSON",
    )

    ses_parser = subparsers.add_parser(
        "ses",
        aliases=["acharya-pedersen", "systemic-risk", "mes", "capital-surcharge"],
        help="Run Acharya, Pedersen, Philippon & Richardson (2017) Systemic Expected Shortfall Engine",
    )
    ses_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    ses_parser.add_argument(
        "--market-tail-threshold",
        type=float,
        default=None,
        help="Conditioning market crisis drop C (%) in [1.0, 15.0] (optional)",
    )
    ses_parser.add_argument(
        "--marginal-shortfall",
        type=float,
        default=None,
        help="Marginal Expected Shortfall MES_i (%) in [-5.0, 30.0] (optional)",
    )
    ses_parser.add_argument(
        "--leverage",
        type=float,
        default=None,
        help="Institution leverage Assets / Equity in [1.0, 45.0] (optional)",
    )
    ses_parser.add_argument(
        "--baseline-leverage",
        type=float,
        default=None,
        help="Normal-times baseline leverage in [1.0, 25.0] (optional)",
    )
    ses_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Systemic Expected Shortfall receipt as JSON",
    )

    covar_parser = subparsers.add_parser(
        "covar",
        aliases=["adrian-brunnermeier", "delta-covar", "systemic-spillover", "tail-contagion"],
        help="Run Adrian & Brunnermeier (2016) CoVaR & Systemic Tail Spillover Engine",
    )
    covar_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    covar_parser.add_argument(
        "--quantile",
        type=float,
        default=None,
        help="Tail conditioning quantile q in [0.01, 0.15] (optional)",
    )
    covar_parser.add_argument(
        "--standalone-var",
        type=float,
        default=None,
        help="Standalone stock Value-at-Risk VaR_i (%) in [1.0, 30.0] (optional)",
    )
    covar_parser.add_argument(
        "--system-var",
        type=float,
        default=None,
        help="Unconditional system market Value-at-Risk VaR_m (%) in [1.0, 20.0] (optional)",
    )
    covar_parser.add_argument(
        "--tail-beta",
        type=float,
        default=None,
        help="Quantile regression slope beta_{m|i}^q in [0.0, 3.5] (optional)",
    )
    covar_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CoVaR spillover receipt as JSON",
    )

    amh_parser = subparsers.add_parser(
        "amh",
        aliases=["adaptive-markets", "andrew-lo", "market-efficiency", "punctuated-equilibrium"],
        help="Run Andrew W. Lo (2004, 2005) Adaptive Markets Hypothesis Engine",
    )
    amh_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    amh_parser.add_argument(
        "--variance-ratio",
        type=float,
        default=None,
        help="Lo-MacKinlay Variance Ratio VR(k) in [0.30, 2.50] (optional)",
    )
    amh_parser.add_argument(
        "--autocorrelation",
        type=float,
        default=None,
        help="First-order serial autocorrelation rho_1 in [-0.80, 0.80] (optional)",
    )
    amh_parser.add_argument(
        "--drawdown",
        type=float,
        default=None,
        help="Ecological stress drawdown in [0.0, 0.85] (optional)",
    )
    amh_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Realized annualized volatility in [0.05, 1.50] (optional)",
    )
    amh_parser.add_argument(
        "--baseline-volatility",
        type=float,
        default=None,
        help="Normal-times baseline volatility in [0.05, 0.80] (optional)",
    )
    amh_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Adaptive Markets receipt as JSON",
    )

    search_bargaining_parser = subparsers.add_parser(
        "search-bargaining",
        aliases=["duffie-garleanu-pedersen", "dgp", "otc-illiquidity", "bargaining-spread"],
        help="Run Duffie-Gârleanu-Pedersen (2005) OTC Search & Bargaining Illiquidity Engine",
    )
    search_bargaining_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    search_bargaining_parser.add_argument(
        "--search-intensity",
        type=float,
        default=None,
        help="Counterparty Poisson contact rate lambda in [0.10, 50.0] (optional)",
    )
    search_bargaining_parser.add_argument(
        "--bargaining-power",
        type=float,
        default=None,
        help="Investor Nash bargaining power q in [0.05, 0.95] (optional)",
    )
    search_bargaining_parser.add_argument(
        "--inventory-cost",
        type=float,
        default=None,
        help="Dealer inventory holding cost chi in [0.00, 0.25] (optional)",
    )
    search_bargaining_parser.add_argument(
        "--discount-rate",
        type=float,
        default=None,
        help="Risk-free opportunity discount rate r in [0.01, 0.15] (optional)",
    )
    search_bargaining_parser.add_argument(
        "--cash-flow-rate",
        type=float,
        default=None,
        help="Asset cash flow / dividend yield rate D in [0.005, 0.20] (optional)",
    )
    search_bargaining_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Search & Bargaining receipt as JSON",
    )

    chs_parser = subparsers.add_parser(
        "chs-distress",
        aliases=["campbell-hilscher-szilagyi", "distress-puzzle", "failure-probability", "chs"],
        help="Run Campbell, Hilscher & Szilagyi (2008) Financial Distress Puzzle Engine",
    )
    chs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    chs_parser.add_argument(
        "--nimta",
        type=float,
        default=None,
        help="Net Income to Market Total Assets in [-1.0, 0.50] (optional)",
    )
    chs_parser.add_argument(
        "--tlmta",
        type=float,
        default=None,
        help="Total Liabilities to Market Total Assets in [0.0, 0.98] (optional)",
    )
    chs_parser.add_argument(
        "--cashmta",
        type=float,
        default=None,
        help="Cash & Short-Term Investments to Market Total Assets in [0.0, 0.80] (optional)",
    )
    chs_parser.add_argument(
        "--excess-return",
        type=float,
        default=None,
        help="Trailing excess log return over S&P 500 in [-1.50, 1.50] (optional)",
    )
    chs_parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        help="Annualized idiosyncratic volatility in [0.05, 1.50] (optional)",
    )
    chs_parser.add_argument(
        "--relative-size",
        type=float,
        default=None,
        help="Relative log equity size in [-12.0, 5.0] (optional)",
    )
    chs_parser.add_argument(
        "--market-to-book",
        type=float,
        default=None,
        help="Market-to-book equity ratio in [0.10, 25.0] (optional)",
    )
    chs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Campbell Distress receipt as JSON",
    )

    gs_parser = subparsers.add_parser(
        "grossman-stiglitz",
        aliases=["gs-equilibrium", "informational-efficiency", "info-rent", "grossman"],
        help="Run Grossman & Stiglitz (1980) Informational Equilibrium & Rent Engine",
    )
    gs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    gs_parser.add_argument(
        "--cost",
        type=float,
        default=0.0020,
        help="Information acquisition / analysis cost c in return terms (default: 0.0020 = 20 bps)",
    )
    gs_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.50,
        help="Investor CARA absolute risk aversion gamma (default: 2.50)",
    )
    gs_parser.add_argument(
        "--fundamental-var",
        type=float,
        default=None,
        help="Fundamental payoff variance sigma_u^2 (optional)",
    )
    gs_parser.add_argument(
        "--noise-var",
        type=float,
        default=None,
        help="Noise trader / supply shock variance sigma_x^2 (optional)",
    )
    gs_parser.add_argument(
        "--informed-fraction",
        type=float,
        default=None,
        help="Observed or hypothetical informed fraction lambda in [0.01, 0.99] (optional)",
    )
    gs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Grossman-Stiglitz receipt as JSON",
    )

    info_risk_parser = subparsers.add_parser(
        "information-risk",
        aliases=["eho-risk", "cost-of-capital", "info-beta", "easley-ohara"],
        help="Run Easley, Hvidkjaer & O'Hara (2002, 2004) Information Risk & Cost of Capital Engine",
    )
    info_risk_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    info_risk_parser.add_argument(
        "--pin",
        type=float,
        default=None,
        help="Probability of Informed Trading in [0.01, 0.85] (optional)",
    )
    info_risk_parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Systematic market equity beta (optional)",
    )
    info_risk_parser.add_argument(
        "--rf",
        type=float,
        default=0.0425,
        help="Annual risk-free rate Rf (default: 0.0425 = 4.25%)",
    )
    info_risk_parser.add_argument(
        "--erp",
        type=float,
        default=0.0550,
        help="Market Equity Risk Premium ERP (default: 0.0550 = 5.50%)",
    )
    info_risk_parser.add_argument(
        "--lambda-info",
        type=float,
        default=0.2500,
        help="Factor risk price per unit PIN difference (default: 0.2500 = 25.0% / unit PIN)",
    )
    info_risk_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Information Risk receipt as JSON",
    )

    admati_parser = subparsers.add_parser(
        "admati-pfleiderer",
        aliases=["volume-clustering", "concentrated-trading", "intraday-liquidity", "ap-clustering"],
        help="Run Admati & Pfleiderer (1988) Volume Clustering & Execution Timing Engine",
    )
    admati_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    admati_parser.add_argument(
        "--discretionary-ratio",
        type=float,
        default=None,
        help="Ratio of discretionary liquidity trading in [0.02, 0.98] (optional)",
    )
    admati_parser.add_argument(
        "--informed-count",
        type=int,
        default=3,
        help="Number of informed traders k entering the cluster (default: 3)",
    )
    admati_parser.add_argument(
        "--spread-bps",
        type=float,
        default=None,
        help="Baseline execution spread in basis points (optional)",
    )
    admati_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Admati-Pfleiderer receipt as JSON",
    )

    noise_parser = subparsers.add_parser(
        "noise-bands",
        aliases=["valuation-bounds", "noise-trading", "black-bounds"],
        help="Run Black (1986) Noise Valuation Bounds & Mean-Reversion Engine",
    )
    noise_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    noise_parser.add_argument(
        "--fundamental-value",
        type=float,
        default=None,
        help="Intrinsic fundamental value P* ($) (optional)",
    )
    noise_parser.add_argument(
        "--noise-ratio",
        type=float,
        default=None,
        help="Noise-to-information variance ratio in [0.01, 30.0] (optional)",
    )
    noise_parser.add_argument(
        "--kappa",
        type=float,
        default=0.35,
        help="Annual speed of mean-reversion kappa (default: 0.35 = 35%/yr)",
    )
    noise_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Noise Valuation receipt as JSON",
    )

    bad_beta_parser = subparsers.add_parser(
        "bad-beta",
        aliases=["two-beta", "campbell-vuolteenaho", "icapm-two-beta"],
        help="Run Campbell & Vuolteenaho (2004) Two-Beta ('Bad Beta, Good Beta') Engine",
    )
    bad_beta_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    bad_beta_parser.add_argument(
        "--beta-cf",
        type=float,
        default=None,
        help="Cash-flow beta ('Bad Beta') beta_CF (optional)",
    )
    bad_beta_parser.add_argument(
        "--beta-dr",
        type=float,
        default=None,
        help="Discount-rate beta ('Good Beta') beta_DR (optional)",
    )
    bad_beta_parser.add_argument(
        "--gamma",
        type=float,
        default=6.0,
        help="Investor relative risk aversion gamma (default: 6.0)",
    )
    bad_beta_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Two-Beta receipt as JSON",
    )

    merton_parser = subparsers.add_parser(
        "merton-recognition",
        aliases=["incomplete-information", "neglected-firm", "investor-recognition", "merton-1987"],
        help="Run Merton (1987) Incomplete Information & Investor Recognition Engine",
    )
    merton_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target stock ticker symbol (e.g., AAPL)",
    )
    merton_parser.add_argument(
        "--recognition-q",
        type=float,
        default=None,
        help="Investor recognition / attention ratio q in [0.02, 1.00] (optional)",
    )
    merton_parser.add_argument(
        "--shadow-lambda",
        type=float,
        default=None,
        help="Shadow cost parameter lambda_shadow in [0.01, 5.0] (optional)",
    )
    merton_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton Incomplete Information receipt as JSON",
    )

    # Subparser: rare-disasters (Xavier Gabaix 2012 Variable Rare Disasters Engine)
    disaster_parser = subparsers.add_parser(
        "rare-disasters",
        aliases=["gabaix-disasters", "tail-resilience", "variable-disasters", "disaster-risk"],
        help="Xavier Gabaix (2012) Variable Rare Disasters, Macro Tail Risk & Firm Resilience Engine",
    )
    disaster_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    disaster_parser.add_argument(
        "--disaster-lambda",
        type=float,
        default=None,
        help="Macro disaster arrival intensity lambda in [0.005, 0.20] (optional)",
    )
    disaster_parser.add_argument(
        "--recovery-f",
        type=float,
        default=None,
        help="Firm dividend recovery factor F in [0.05, 1.50] (optional)",
    )
    disaster_parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Relative risk aversion gamma in [1.0, 20.0] (optional)",
    )
    disaster_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Gabaix Variable Rare Disasters receipt as JSON",
    )

    # Subparser: kyle-xiong (Albert S. Kyle & Wei Xiong 2001 Contagion as a Wealth Effect Engine)
    kyle_parser = subparsers.add_parser(
        "kyle-xiong",
        aliases=["wealth-contagion", "forced-liquidation", "spillover-beta", "kyle-2001"],
        help="Albert S. Kyle & Wei Xiong (2001) Contagion as a Wealth Effect & Forced Liquidation Engine",
    )
    kyle_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    kyle_parser.add_argument(
        "--wealth-drawdown",
        type=float,
        default=None,
        help="Arbitrageur equity wealth drawdown percentage in [0.0, 95.0] (optional)",
    )
    kyle_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Asset annualized fundamental volatility sigma in [0.02, 2.00] (optional)",
    )
    kyle_parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Arbitrageur risk aversion gamma in [0.5, 20.0] (optional)",
    )
    kyle_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kyle & Xiong Wealth Contagion receipt as JSON",
    )

    # Subparser: predatory-trading (Markus K. Brunnermeier & Lasse Heje Pedersen 2005 Predatory Trading Engine)
    predatory_parser = subparsers.add_parser(
        "predatory-trading",
        aliases=["predatory", "strategic-frontrun", "liquidation-overshoot", "brunnermeier-2005"],
        help="Markus K. Brunnermeier & Lasse Heje Pedersen (2005) Predatory Trading & Price Overshoot Engine",
    )
    predatory_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    predatory_parser.add_argument(
        "--distressed-size",
        type=float,
        default=None,
        help="Distressed liquidation order flow as % of ADV in [0.1, 25.0] (optional)",
    )
    predatory_parser.add_argument(
        "--predator-concentration",
        type=float,
        default=None,
        help="Strategic predator concentration share in [0.05, 0.95] (optional)",
    )
    predatory_parser.add_argument(
        "--market-depth",
        type=float,
        default=None,
        help="Market depth multiplier lambda_inv in [0.20, 5.00] (optional)",
    )
    predatory_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Predatory Trading receipt as JSON",
    )

    # Subparser: hong-stein-crash (Harrison Hong & Jeremy C. Stein 2003 Differences of Opinion Engine)
    hong_stein_parser = subparsers.add_parser(
        "hong-stein-crash",
        aliases=["crash-asymmetry", "latent-overhang", "short-constraint-crash", "hong-stein-2003"],
        help="Harrison Hong & Jeremy C. Stein (2003) Differences of Opinion, Short Constraints & Crashes Engine",
    )
    hong_stein_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    hong_stein_parser.add_argument(
        "--opinion-dispersion",
        type=float,
        default=None,
        help="Investor opinion dispersion ratio sigma_theta in [0.01, 1.00] (optional)",
    )
    hong_stein_parser.add_argument(
        "--short-constraint",
        type=float,
        default=None,
        help="Short-sale constraint tightness C_short in [0.00, 1.00] (optional)",
    )
    hong_stein_parser.add_argument(
        "--turnover",
        type=float,
        default=None,
        help="Trading volume turnover multiplier in [0.10, 10.0] (optional)",
    )
    hong_stein_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hong & Stein Crash Asymmetry receipt as JSON",
    )

    # Subparser: breeden-ccapm (Douglas T. Breeden 1979 Consumption CAPM Engine)
    breeden_parser = subparsers.add_parser(
        "breeden-ccapm",
        aliases=["ccapm", "consumption-beta", "macro-consumption", "breeden-1979"],
        help="Douglas T. Breeden (1979) Consumption-Based Capital Asset Pricing Model (CCAPM) Engine",
    )
    breeden_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    breeden_parser.add_argument(
        "--consumption-beta",
        type=float,
        default=None,
        help="Consumption beta beta_{i, C} in [-3.0, 6.0] (optional)",
    )
    breeden_parser.add_argument(
        "--consumption-premium",
        type=float,
        default=None,
        help="Unit price of consumption risk lambda_C in [0.005, 0.20] (optional)",
    )
    breeden_parser.add_argument(
        "--consumption-vol",
        type=float,
        default=None,
        help="Real consumption growth volatility in [0.005, 0.15] (optional)",
    )
    breeden_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Breeden CCAPM receipt as JSON",
    )

    # Subparser: bhs-prospect (Nicholas Barberis, Ming Huang & Tano Santos 2001 Prospect Theory Engine)
    bhs_parser = subparsers.add_parser(
        "bhs-prospect",
        aliases=["prospect-theory", "loss-aversion", "house-money", "barberis-2001"],
        help="Barberis, Huang & Santos (2001) Prospect Theory, Dynamic Loss Aversion & Cushion Engine",
    )
    bhs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    bhs_parser.add_argument(
        "--reference-price",
        type=float,
        default=None,
        help="Purchase/reference cost-basis price P_ref (optional)",
    )
    bhs_parser.add_argument(
        "--lambda-zero",
        type=float,
        default=None,
        help="Benchmark baseline loss aversion lambda_0 in [1.5, 3.5] (optional)",
    )
    bhs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BHS Prospect Theory receipt as JSON",
    )

    # Subparser: allen-gale (Franklin Allen & Douglas Gale 2000 Financial Network Contagion Engine)
    allen_gale_parser = subparsers.add_parser(
        "allen-gale",
        aliases=["network-contagion", "incomplete-network", "liquidity-cascade", "allen-2000"],
        help="Franklin Allen & Douglas Gale (2000) Financial Network Contagion & Cascades Engine",
    )
    allen_gale_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    allen_gale_parser.add_argument(
        "--network-completeness",
        type=float,
        default=None,
        help="Network completeness degree gamma_net in [0.05, 1.00] (optional)",
    )
    allen_gale_parser.add_argument(
        "--liquidity-shock",
        type=float,
        default=None,
        help="Localized liquidity demand shock theta in [0.01, 1.00] (optional)",
    )
    allen_gale_parser.add_argument(
        "--aggregate-liquidity",
        type=float,
        default=None,
        help="Aggregate liquidity buffer L_agg in [0.10, 5.00] (optional)",
    )
    allen_gale_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Allen & Gale Network Contagion receipt as JSON",
    )

    # Subparser: leverage-cycle (John Geanakoplos 2010 The Leverage Cycle Engine)
    leverage_parser = subparsers.add_parser(
        "leverage-cycle",
        aliases=["geanakoplos", "collateral-haircut", "deleveraging-crisis", "margin-squeeze"],
        help="John Geanakoplos (2010) Leverage Cycle, Collateral Margins & De-leveraging Engine",
    )
    leverage_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    leverage_parser.add_argument(
        "--haircut",
        type=float,
        default=None,
        help="Collateral margin requirement haircut m_t in [0.05, 0.85] (optional)",
    )
    leverage_parser.add_argument(
        "--baseline-haircut",
        type=float,
        default=None,
        help="Baseline normal haircut m_0 in [0.05, 0.60] (optional)",
    )
    leverage_parser.add_argument(
        "--volatility",
        type=float,
        default=None,
        help="Asset annualized volatility in [0.02, 2.00] (optional)",
    )
    leverage_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Geanakoplos Leverage Cycle receipt as JSON",
    )

    # Subparser: diamond-dybvig (Douglas W. Diamond & Philip H. Dybvig 1983 Bank Runs Engine)
    diamond_parser = subparsers.add_parser(
        "diamond-dybvig",
        aliases=["bank-run", "liquidity-transformation", "sequential-service", "redemption-panic"],
        help="Douglas W. Diamond & Philip H. Dybvig (1983) Bank Runs, Liquidity Transformation & Sequential Service Engine",
    )
    diamond_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    diamond_parser.add_argument(
        "--reserve",
        type=float,
        default=None,
        help="Liquid cash reserve buffer ratio c in [0.01, 0.90] (optional)",
    )
    diamond_parser.add_argument(
        "--withdrawals",
        type=float,
        default=None,
        help="Observed or stress early withdrawal ratio f_1 in [0.0, 1.0] (optional)",
    )
    diamond_parser.add_argument(
        "--r1",
        type=float,
        default=None,
        help="Promised gross short-term payout r_1 >= 1.0 (optional)",
    )
    diamond_parser.add_argument(
        "--r2",
        type=float,
        default=None,
        help="Long-term productive investment yield R_2 > r_1 (optional)",
    )
    diamond_parser.add_argument(
        "--salvage",
        type=float,
        default=None,
        help="Premature liquidation salvage recovery rate L in (0.0, 1.0) (optional)",
    )
    diamond_parser.add_argument(
        "--insured",
        action="store_true",
        help="Enable full deposit insurance / central bank backstop",
    )
    diamond_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Diamond & Dybvig Run receipt as JSON",
    )

    # Subparser: merton-icapm (Robert C. Merton 1973 Intertemporal CAPM Engine)
    merton_parser = subparsers.add_parser(
        "merton-icapm",
        aliases=["icapm", "intertemporal-capm", "state-hedging", "merton-1973"],
        help="Robert C. Merton (1973) Intertemporal Capital Asset Pricing Model & State Hedging Engine",
    )
    merton_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    merton_parser.add_argument(
        "--market-premium",
        type=float,
        default=None,
        help="Expected annual market risk premium lambda_M (e.g. 0.055)",
    )
    merton_parser.add_argument(
        "--rate-premium",
        type=float,
        default=None,
        help="Price of interest rate shock risk lambda_r (e.g. -0.020)",
    )
    merton_parser.add_argument(
        "--vol-premium",
        type=float,
        default=None,
        help="Price of market volatility innovation risk lambda_sigma (e.g. -0.035)",
    )
    merton_parser.add_argument(
        "--rf",
        type=float,
        default=None,
        help="Annualized benchmark risk-free rate (e.g. 0.0425)",
    )
    merton_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton ICAPM receipt as JSON",
    )

    # Subparser: ross-apt (Stephen A. Ross 1976 Arbitrage Pricing Theory Engine)
    apt_parser = subparsers.add_parser(
        "ross-apt",
        aliases=["apt", "arbitrage-pricing", "multi-factor-apt", "ross-1976"],
        help="Stephen A. Ross (1976) Arbitrage Pricing Theory & Macro Multi-Factor Engine",
    )
    apt_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    apt_parser.add_argument(
        "--growth-premium",
        type=float,
        default=None,
        help="Risk price for economic growth lambda_growth (e.g. 0.045)",
    )
    apt_parser.add_argument(
        "--inflation-premium",
        type=float,
        default=None,
        help="Risk price for unexpected inflation lambda_inf (e.g. -0.025)",
    )
    apt_parser.add_argument(
        "--default-premium",
        type=float,
        default=None,
        help="Risk price for credit default spread lambda_def (e.g. -0.030)",
    )
    apt_parser.add_argument(
        "--term-premium",
        type=float,
        default=None,
        help="Risk price for term structure slope lambda_term (e.g. 0.020)",
    )
    apt_parser.add_argument(
        "--rf",
        type=float,
        default=None,
        help="Annualized benchmark risk-free rate (e.g. 0.0425)",
    )
    apt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Ross APT receipt as JSON",
    )

    # Subparser: dssw-noise (De Long, Shleifer, Summers & Waldmann 1990 Noise Trader Risk Engine)
    dssw_parser = subparsers.add_parser(
        "dssw-noise",
        aliases=["noise-trader-risk", "limits-of-arbitrage", "resale-price-risk", "dssw-1990"],
        help="De Long, Shleifer, Summers & Waldmann (1990) Noise Trader Risk & Limits of Arbitrage Engine",
    )
    dssw_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    dssw_parser.add_argument(
        "--mean-sentiment",
        type=float,
        default=None,
        help="Average bullish sentiment bias mu (e.g. 0.05)",
    )
    dssw_parser.add_argument(
        "--shock",
        type=float,
        default=None,
        help="Current period sentiment realization shock S_t (e.g. 0.08)",
    )
    dssw_parser.add_argument(
        "--variance",
        type=float,
        default=None,
        help="Sentiment unpredictability variance sigma_S^2 (e.g. 0.015)",
    )
    dssw_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=None,
        help="Arbitrageur absolute risk aversion gamma (e.g. 1.25)",
    )
    dssw_parser.add_argument(
        "--rf",
        type=float,
        default=None,
        help="Risk-free / dividend discount rate r (e.g. 0.0425)",
    )
    dssw_parser.add_argument(
        "--json",
        action="store_true",
        help="Output DSSW Noise Trader Risk receipt as JSON",
    )

    # Subparser: black-leverage (Fischer Black 1976 Leverage Effect & Volatility Asymmetry Engine)
    black_parser = subparsers.add_parser(
        "black-leverage",
        aliases=["leverage-effect", "asymmetric-volatility", "volatility-feedback", "black-1976"],
        help="Fischer Black (1976) & Campbell-Hentschel (1992) Leverage Effect & Volatility Asymmetry Engine",
    )
    black_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, NVDA, MSFT)",
    )
    black_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Financial debt per share D in dollars (optional)",
    )
    black_parser.add_argument(
        "--asset-vol",
        type=float,
        default=None,
        help="Unlevered enterprise asset volatility sigma_A (optional, e.g. 0.18)",
    )
    black_parser.add_argument(
        "--feedback",
        type=float,
        default=None,
        help="Volatility feedback elasticity theta (optional, e.g. 0.25)",
    )
    black_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black Leverage Effect receipt as JSON",
    )

    # Subparser: lucas-tree (Robert E. Lucas Jr. 1978 Pure Exchange Tree Equilibrium Engine)
    lucas_parser = subparsers.add_parser(
        "lucas-tree",
        aliases=["lucas-1978", "exchange-economy", "pure-exchange", "lucas-sdf"],
        help="Robert E. Lucas Jr. (1978) Pure Exchange Tree Equilibrium & Stochastic Discount Factor Engine",
    )
    lucas_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, SPY, LUCAS)",
    )
    lucas_parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Subjective discount factor beta in (0, 1) (default: 0.96)",
    )
    lucas_parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Relative risk aversion coefficient gamma (default: 2.50)",
    )
    lucas_parser.add_argument(
        "--growth-mean",
        type=float,
        default=None,
        help="Log dividend growth mean mu (e.g. 0.02)",
    )
    lucas_parser.add_argument(
        "--growth-vol",
        type=float,
        default=None,
        help="Log dividend growth volatility sigma (e.g. 0.04)",
    )
    lucas_parser.add_argument(
        "--current-dividend",
        type=float,
        default=None,
        help="Current period dividend payout D_t in dollars (e.g. 2.0)",
    )
    lucas_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Lucas Tree Equilibrium receipt as JSON",
    )

    # Subparser: harrison-kreps (Harrison & Kreps 1978 Speculative Disagreement & Resale Option Engine)
    hk_parser = subparsers.add_parser(
        "harrison-kreps",
        aliases=["resale-option", "speculative-bubble", "hk-1979", "investor-disagreement"],
        help="Harrison & Kreps (1978) Speculative Disagreement & Resale Option Premium Engine",
    )
    hk_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. TSLA, NVDA, GME)",
    )
    hk_parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="Subjective discount factor beta in (0, 1) (default: 0.95)",
    )
    hk_parser.add_argument(
        "--belief-spread",
        type=float,
        default=None,
        help="Belief divergence spread delta in [0.0, 0.45] (default: 0.20)",
    )
    hk_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Harrison-Kreps Resale Option receipt as JSON",
    )

    # Subparser: brennan-agency (Michael J. Brennan 1998 Two-Beta Agency Asset Pricing Engine)
    brennan_parser = subparsers.add_parser(
        "brennan-agency",
        aliases=["agency-pricing", "brennan-1998", "two-beta-agency", "benchmark-distortion"],
        help="Michael J. Brennan (1998) Two-Beta Agency & Institutional Benchmark Pricing Engine",
    )
    brennan_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. AAPL, MSFT, IWM)",
    )
    brennan_parser.add_argument(
        "--beta-m",
        type=float,
        default=None,
        help="Market portfolio beta beta_{i, M} (optional, e.g. 1.15)",
    )
    brennan_parser.add_argument(
        "--beta-b",
        type=float,
        default=None,
        help="Institutional benchmark beta beta_{i, B} (optional, e.g. 1.35)",
    )
    brennan_parser.add_argument(
        "--inst-share",
        type=float,
        default=None,
        help="Institutional wealth share omega_inst in [0, 1] (default: 0.70)",
    )
    brennan_parser.add_argument(
        "--tracking-aversion",
        type=float,
        default=None,
        help="Institutional tracking error penalty gamma_rel (default: 5.0)",
    )
    brennan_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Brennan Agency Pricing receipt as JSON",
    )

    # Subparser: shleifer-vishny (Shleifer & Vishny 1997 Limits of Arbitrage & PBA Engine)
    sv_parser = subparsers.add_parser(
        "shleifer-vishny",
        aliases=["shleifer-vishny-1997", "pba-arbitrage", "sv-1997", "performance-arbitrage"],
        help="Andrei Shleifer & Robert W. Vishny (1997) Performance-Based Arbitrage & Limits of Arbitrage Engine",
    )
    sv_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. ARB, AAPL, GME)",
    )
    sv_parser.add_argument(
        "--fundamental-val",
        type=float,
        default=None,
        help="Terminal fundamental liquidation value V (e.g. 100.0)",
    )
    sv_parser.add_argument(
        "--shock-severity",
        type=float,
        default=None,
        help="Adverse interim noise shock severity fraction (e.g. 0.15 for 15% further drop)",
    )
    sv_parser.add_argument(
        "--flow-sensitivity",
        type=float,
        default=None,
        help="Investor redemption flow sensitivity a (default: 1.20)",
    )
    sv_parser.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Arbitrageur initial equity capital F_0 (default: 1000.0)",
    )
    sv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Shleifer-Vishny limits of arbitrage receipt as JSON",
    )

    # Subparser: myers-majluf (Stewart C. Myers & Nicholas S. Majluf 1984 Pecking Order Engine)
    mm_parser = subparsers.add_parser(
        "myers-majluf",
        aliases=["pecking-order", "equity-issuance", "lemons-discount", "myers-1984"],
        help="Myers & Majluf (1984) Pecking Order, Dilution & Adverse Selection Lemons Engine",
    )
    mm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. CORP, AAPL, TSLA)",
    )
    mm_parser.add_argument(
        "--assets",
        type=float,
        default=None,
        help="True intrinsic value of assets in place A (e.g. 120.0)",
    )
    mm_parser.add_argument(
        "--outlay",
        type=float,
        default=None,
        help="Project investment capital expenditure I (e.g. 30.0)",
    )
    mm_parser.add_argument(
        "--payoff",
        type=float,
        default=None,
        help="Expected terminal project cash flow payoff B (e.g. 36.0)",
    )
    mm_parser.add_argument(
        "--cash-slack",
        type=float,
        default=None,
        help="Internal cash reserves / safe debt slack C (default: 10.0)",
    )
    mm_parser.add_argument(
        "--info-spread",
        type=float,
        default=None,
        help="Information asymmetry spread between insiders and market (default: 0.15)",
    )
    mm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Myers-Majluf Pecking Order receipt as JSON",
    )

    # Subparser: breeden-litzenberger (Douglas T. Breeden & Robert H. Litzenberger 1978 Arrow-Debreu State Prices & RND Engine)
    bl_parser = subparsers.add_parser(
        "breeden-litzenberger",
        aliases=["rnd-density", "arrow-debreu", "state-prices", "bl-1978"],
        help="Breeden & Litzenberger (1978) Arrow-Debreu State Price Recovery & Risk-Neutral Density Engine",
    )
    bl_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. RND, SPY, AAPL)",
    )
    bl_parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Underlying spot price S_0 (e.g. 100.0)",
    )
    bl_parser.add_argument(
        "--vol-skew",
        type=float,
        default=0.22,
        help="Implied volatility skew slope coefficient (default: 0.22)",
    )
    bl_parser.add_argument(
        "--vol-smile",
        type=float,
        default=0.04,
        help="Implied volatility smile curvature coefficient (default: 0.04)",
    )
    bl_parser.add_argument(
        "--rate",
        type=float,
        default=0.04,
        help="Continuously compounded risk-free interest rate r (default: 0.04)",
    )
    bl_parser.add_argument(
        "--expiry",
        type=float,
        default=0.25,
        help="Time to option expiry T in years (default: 0.25)",
    )
    bl_parser.add_argument(
        "--num-strikes",
        type=int,
        default=49,
        help="Number of discrete strike intervals across surface (default: 49)",
    )
    bl_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Breeden-Litzenberger state-price receipt as JSON",
    )

    # Subparser: aghion-bolton (Aghion & Bolton 1992 / Hart & Moore 1994 Control Rights & Strategic Default Engine)
    ab_parser = subparsers.add_parser(
        "aghion-bolton",
        aliases=["control-rights", "incomplete-contracts", "hart-moore", "strategic-default", "ab-1992"],
        help="Aghion & Bolton (1992) / Hart & Moore (1994) Control Rights & Strategic Default Engine",
    )
    ab_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. GOV, AAPL, JPM)",
    )
    ab_parser.add_argument(
        "--cash-flow",
        type=float,
        default=None,
        help="Realized verifiable cash flow y (e.g. 100.0)",
    )
    ab_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Contractual debt face value D (e.g. 90.0)",
    )
    ab_parser.add_argument(
        "--liquidation",
        type=float,
        default=None,
        help="Asset liquidation threat value L (default: 50.0)",
    )
    ab_parser.add_argument(
        "--private-benefit",
        type=float,
        default=None,
        help="Entrepreneur non-monetary private benefit B (default: 15.0)",
    )
    ab_parser.add_argument(
        "--creditor-power",
        type=float,
        default=None,
        help="Creditor Nash bargaining power alpha in [0, 1] (default: 0.50)",
    )
    ab_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Aghion-Bolton control rights receipt as JSON",
    )

    # Subparser: leland-default (Hayne E. Leland 1994 Endogenous Default & Optimal Capital Structure Engine)
    lel_parser = subparsers.add_parser(
        "leland-default",
        aliases=["leland-1994", "endogenous-default", "optimal-leverage", "tax-shield"],
        help="Leland (1994) Endogenous Default Barrier & Optimal Capital Structure Engine",
    )
    lel_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. DEBT, F, IBM)",
    )
    lel_parser.add_argument(
        "--asset-value",
        type=float,
        default=None,
        help="Unlevered firm asset value V (e.g. 100.0)",
    )
    lel_parser.add_argument(
        "--volatility",
        type=float,
        default=0.25,
        help="Annual asset return volatility sigma (default: 0.25)",
    )
    lel_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Risk-free interest rate r (default: 0.05)",
    )
    lel_parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.25,
        help="Effective corporate tax rate tau (default: 0.25)",
    )
    lel_parser.add_argument(
        "--bankruptcy-cost",
        type=float,
        default=0.35,
        help="Proportional bankruptcy loss fraction alpha (default: 0.35)",
    )
    lel_parser.add_argument(
        "--coupon",
        type=float,
        default=None,
        help="Annual perpetual debt coupon C (default: 4.0)",
    )
    lel_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Leland (1994) capital structure receipt as JSON",
    )

    # Subparser: hansen-jagannathan (Hansen & Jagannathan 1991, 1997 SDF Volatility Bounds Engine)
    hj_parser = subparsers.add_parser(
        "hansen-jagannathan",
        aliases=["hj-bound", "hj-distance", "sdf-bound", "hj-1991"],
        help="Hansen & Jagannathan (1991, 1997) SDF Volatility Bounds & HJ-Distance Engine",
    )
    hj_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. SDF, SPY, QQQ)",
    )
    hj_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=5.0,
        help="CRRA Relative Risk Aversion gamma (default: 5.0)",
    )
    hj_parser.add_argument(
        "--beta",
        type=float,
        default=0.99,
        help="Subjective annual time discount factor beta (default: 0.99)",
    )
    hj_parser.add_argument(
        "--rate",
        type=float,
        default=0.04,
        help="Annual risk-free interest rate r (default: 0.04)",
    )
    hj_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hansen-Jagannathan SDF diagnostics receipt as JSON",
    )

    # Subparser: jensen-meckling (Jensen & Meckling 1976 Agency Costs & Asset Substitution Engine)
    jm_parser = subparsers.add_parser(
        "jensen-meckling",
        aliases=["asset-substitution", "risk-shifting", "agency-cost", "jm-1976"],
        help="Jensen & Meckling (1976) Agency Costs of Equity & Debt Asset Substitution Engine",
    )
    jm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. FIRM, ORCL, META)",
    )
    jm_parser.add_argument(
        "--firm-value",
        type=float,
        default=None,
        help="Unconstrained firm asset potential V_0 (e.g. 100.0)",
    )
    jm_parser.add_argument(
        "--alpha",
        type=float,
        default=0.40,
        help="Insider managerial equity stake alpha in (0.05, 1.0] (default: 0.40)",
    )
    jm_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Debt obligation face value D (e.g. 80.0)",
    )
    jm_parser.add_argument(
        "--safe-vol",
        type=float,
        default=0.20,
        help="Safe core project asset volatility sigma_L (default: 0.20)",
    )
    jm_parser.add_argument(
        "--risky-vol",
        type=float,
        default=0.55,
        help="Risky gamble project volatility sigma_H (default: 0.55)",
    )
    jm_parser.add_argument(
        "--npv-discount",
        type=float,
        default=0.06,
        help="Fractional asset value lost by undertaking risky gamble (default: 0.06)",
    )
    jm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Jensen-Meckling agency cost receipt as JSON",
    )

    # Subparser: heston-vol (Steven L. Heston 1993 Stochastic Volatility Option Pricing Engine)
    hes_parser = subparsers.add_parser(
        "heston-vol",
        aliases=["heston-1993", "stochastic-vol", "heston-pricing", "feller-vol"],
        help="Heston (1993) Stochastic Volatility Option Pricing & Characteristic Function Engine",
    )
    hes_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. VOL, SPY, NVDA)",
    )
    hes_parser.add_argument(
        "--spot",
        type=float,
        default=None,
        help="Current underlying spot price S_0 (e.g. 100.0)",
    )
    hes_parser.add_argument(
        "--strike",
        type=float,
        default=None,
        help="Option strike price K (default: ATM = spot)",
    )
    hes_parser.add_argument(
        "--v0",
        type=float,
        default=0.04,
        help="Initial instantaneous variance v_0 (default: 0.04)",
    )
    hes_parser.add_argument(
        "--theta",
        type=float,
        default=0.04,
        help="Long-run variance attractor theta (default: 0.04)",
    )
    hes_parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Mean-reversion speed kappa (default: 2.0)",
    )
    hes_parser.add_argument(
        "--xi",
        type=float,
        default=0.35,
        help="Volatility of variance xi (default: 0.35)",
    )
    hes_parser.add_argument(
        "--rho",
        type=float,
        default=-0.65,
        help="Spot-variance correlation rho in [-0.99, 0.99] (default: -0.65)",
    )
    hes_parser.add_argument(
        "--rate",
        type=float,
        default=0.04,
        help="Risk-free interest rate r (default: 0.04)",
    )
    hes_parser.add_argument(
        "--expiry",
        type=float,
        default=0.50,
        help="Time to option expiry T in years (default: 0.50)",
    )
    hes_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Heston stochastic vol pricing receipt as JSON",
    )

    # Subparser: debt-overhang (Stewart C. Myers 1977 Debt Overhang & Real Growth Option Engine)
    mo_parser = subparsers.add_parser(
        "debt-overhang",
        aliases=["myers-1977", "growth-option", "real-option", "overhang-loss"],
        help="Myers (1977) Debt Overhang & Real Growth Option Underinvestment Engine",
    )
    mo_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. GROWTH, BA, AMZN)",
    )
    mo_parser.add_argument(
        "--assets",
        type=float,
        default=None,
        help="Current PV of existing assets in place A_0 (e.g. 100.0)",
    )
    mo_parser.add_argument(
        "--growth-pv",
        type=float,
        default=60.0,
        help="Expected gross PV of discretionary growth option G_0 (default: 60.0)",
    )
    mo_parser.add_argument(
        "--outlay",
        type=float,
        default=40.0,
        help="Capex outlay I required to exercise growth option (default: 40.0)",
    )
    mo_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Senior debt face value D due at maturity (default: 95.0)",
    )
    mo_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Myers (1977) debt overhang receipt as JSON",
    )

    # Subparser: holmstrom-tirole (Holmstrom & Tirole 1997 Pledgeable Income & Credit Rationing Engine)
    ht_parser = subparsers.add_parser(
        "holmstrom-tirole",
        aliases=["pledgeable-income", "credit-rationing", "ht-1997", "moral-hazard"],
        help="Holmstrom & Tirole (1997) Pledgeable Income & Credit Rationing Engine",
    )
    ht_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. BANK, GS, JPM)",
    )
    ht_parser.add_argument(
        "--outlay",
        type=float,
        default=None,
        help="Required capital outlay I (e.g. 100.0)",
    )
    ht_parser.add_argument(
        "--net-worth",
        type=float,
        default=None,
        help="Corporate net worth / collateral A (e.g. 20.0)",
    )
    ht_parser.add_argument(
        "--payoff",
        type=float,
        default=160.0,
        help="Project gross payoff R on success (default: 160.0)",
    )
    ht_parser.add_argument(
        "--private-benefit",
        type=float,
        default=28.0,
        help="Unmonitored shirking private benefit B (default: 28.0)",
    )
    ht_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Holmstrom-Tirole credit rationing receipt as JSON",
    )

    # Subparser: takeover-bid (Grossman & Hart 1980 Takeover Bids & Free-Rider Engine)
    ght_parser = subparsers.add_parser(
        "takeover-bid",
        aliases=["free-rider", "grossman-hart", "gh-1980", "tender-offer"],
        help="Grossman & Hart (1980) Takeover Bids & Free-Rider Engine",
    )
    ght_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Target security ticker symbol (e.g. TARGET, SAVE, VMW)",
    )
    ght_parser.add_argument(
        "--status-quo",
        type=float,
        default=None,
        help="Target firm status-quo enterprise/share value q (e.g. 100.0)",
    )
    ght_parser.add_argument(
        "--synergy",
        type=float,
        default=30.0,
        help="Post-takeover restructuring synergy Delta_v (default: 30.0)",
    )
    ght_parser.add_argument(
        "--bidding-cost",
        type=float,
        default=10.0,
        help="Takeover research & execution cost c (default: 10.0)",
    )
    ght_parser.add_argument(
        "--dilution",
        type=float,
        default=14.0,
        help="Minority freeze-out dilution allowance phi (default: 14.0)",
    )
    ght_parser.add_argument(
        "--toehold",
        type=float,
        default=0.05,
        help="Pre-bid activist toehold equity stake alpha in [0, 0.49] (default: 0.05)",
    )
    ght_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Grossman-Hart takeover equilibrium receipt as JSON",
    )

    # Subparser: glosten-milgrom (Glosten & Milgrom 1985 Adverse Selection Spread Engine)
    gm_parser = subparsers.add_parser(
        "glosten-milgrom",
        aliases=["adverse-spread", "gm-1985", "specialist-spread", "bayesian-spread"],
        help="Glosten & Milgrom (1985) Sequential Trade Adverse Selection Spread Engine",
    )
    gm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. BOOK, SPY, NVDA)",
    )
    gm_parser.add_argument(
        "--value-low",
        type=float,
        default=None,
        help="Low asset fundamental state V_L (e.g. 90.0)",
    )
    gm_parser.add_argument(
        "--value-high",
        type=float,
        default=None,
        help="High asset fundamental state V_H (e.g. 110.0)",
    )
    gm_parser.add_argument(
        "--pi",
        type=float,
        default=0.25,
        help="Informed trader arrival fraction pi in [0, 1] (default: 0.25)",
    )
    gm_parser.add_argument(
        "--prior",
        type=float,
        default=0.50,
        help="Initial specialist prior probability of high state V_H (default: 0.50)",
    )
    gm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Glosten-Milgrom adverse selection receipt as JSON",
    )

    # Subparser: leland-pyle (Leland & Pyle 1977 Insider Equity Retention Signaling Engine)
    lp_parser = subparsers.add_parser(
        "leland-pyle",
        aliases=["insider-signal", "ownership-signal", "lp-1977"],
        help="Leland & Pyle (1977) Insider Ownership Signaling & Costly Diversification Engine",
    )
    lp_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. SIGNAL, IPO, PLTR)",
    )
    lp_parser.add_argument(
        "--floor-value",
        type=float,
        default=None,
        help="Baseline lemon floor project valuation mu_min (e.g. 100.0)",
    )
    lp_parser.add_argument(
        "--true-value",
        type=float,
        default=None,
        help="True intrinsic expected valuation mu_true (default: 1.15 * floor)",
    )
    lp_parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Insider retained equity fraction alpha in [0, 0.98] (default: 0.55)",
    )
    lp_parser.add_argument(
        "--volatility",
        type=float,
        default=0.40,
        help="Annual idiosyncratic/project return volatility sigma (default: 0.40)",
    )
    lp_parser.add_argument(
        "--gamma",
        type=float,
        default=3.0,
        help="Founder relative risk aversion coefficient (default: 3.0)",
    )
    lp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Leland-Pyle signaling receipt as JSON",
    )

    # Subparser: debt-signaling (Stephen A. Ross 1977 Incentive Debt-Signaling Engine)
    rs_parser = subparsers.add_parser(
        "debt-signaling",
        aliases=["ross-1977", "ross-signal", "bankruptcy-penalty"],
        help="Ross (1977) Incentive Signaling via Debt Structure & Bankruptcy Penalty Engine",
    )
    rs_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. ROSS, BA, IBM)",
    )
    rs_parser.add_argument(
        "--floor-t",
        type=float,
        default=None,
        help="Minimum lemon floor quality cash-flow ceiling t_min (e.g. 100.0)",
    )
    rs_parser.add_argument(
        "--true-t",
        type=float,
        default=None,
        help="True private quality cash-flow ceiling t_true (default: 1.6 * floor)",
    )
    rs_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Contractual debt face value F (default: 65.0)",
    )
    rs_parser.add_argument(
        "--penalty",
        type=float,
        default=60.0,
        help="Personal managerial bankruptcy penalty L (default: 60.0)",
    )
    rs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Ross debt-signaling receipt as JSON",
    )

    # Subparser: delegated-monitoring (Douglas W. Diamond 1984 Delegated Monitoring Engine)
    dm_parser = subparsers.add_parser(
        "delegated-monitoring",
        aliases=["diamond-1984", "bank-monitoring", "delegation-cost"],
        help="Diamond (1984) Financial Intermediation & Delegated Monitoring Engine",
    )
    dm_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Intermediary ticker symbol (e.g. BANK, JPM, C)",
    )
    dm_parser.add_argument(
        "--borrowers",
        type=int,
        default=None,
        help="Number of firms N pooled in intermediary portfolio (e.g. 50)",
    )
    dm_parser.add_argument(
        "--depositors",
        type=int,
        default=10,
        help="Atomistic depositors m per borrower under direct lending (default: 10)",
    )
    dm_parser.add_argument(
        "--monitoring-cost",
        type=float,
        default=2.0,
        help="Verification cost K per monitored borrower (default: 2.0)",
    )
    dm_parser.add_argument(
        "--correlation",
        type=float,
        default=0.05,
        help="Systemic pairwise loan correlation rho in [0, 1] (default: 0.05)",
    )
    dm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Diamond delegated monitoring receipt as JSON",
    )

    # Subparser: debt-taxes (Merton H. Miller 1977 Debt & Taxes Personal Clientele Engine)
    mil_parser = subparsers.add_parser(
        "debt-taxes",
        aliases=["miller-1977", "tax-clientele", "miller-equilibrium"],
        help="Miller (1977) Debt & Taxes Personal Clientele Equilibrium Engine",
    )
    mil_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. TAX, KO, XOM)",
    )
    mil_parser.add_argument(
        "--firm-value",
        type=float,
        default=None,
        help="Unlevered firm enterprise value V_U (e.g. 100.0)",
    )
    mil_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Corporate debt market value B_D (e.g. 40.0)",
    )
    mil_parser.add_argument(
        "--corp-tax",
        type=float,
        default=0.25,
        help="Statutory corporate tax rate tau_c (default: 0.25)",
    )
    mil_parser.add_argument(
        "--equity-tax",
        type=float,
        default=0.12,
        help="Effective personal equity tax rate tau_ps (default: 0.12)",
    )
    mil_parser.add_argument(
        "--debt-tax",
        type=float,
        default=0.30,
        help="Personal ordinary income tax rate on debt interest tau_pd (default: 0.30)",
    )
    mil_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Miller (1977) tax equilibrium receipt as JSON",
    )

    # Subparser: habit-formation (Campbell & Cochrane 1999 External Habit Formation Engine)
    cc_parser = subparsers.add_parser(
        "habit-formation",
        aliases=["campbell-cochrane", "cc-1999", "surplus-consumption"],
        help="Campbell & Cochrane (1999) External Habit Formation & Surplus Consumption Engine",
    )
    cc_parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Security ticker symbol (e.g. HABIT, SPY, QQQ)",
    )
    cc_parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="Utility curvature parameter gamma (default: 2.0)",
    )
    cc_parser.add_argument(
        "--phi",
        type=float,
        default=0.87,
        help="Annual habit persistence parameter phi (default: 0.87)",
    )
    cc_parser.add_argument(
        "--surplus-ratio",
        type=float,
        default=None,
        help="Current surplus consumption ratio S_t in (0.002, 0.50) (default: steady state S_bar)",
    )
    cc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Campbell-Cochrane habit receipt as JSON",
    )

    # CIR (1985) & Vasicek (1977) Affine Term Structure parser
    ts_parser = subparsers.add_parser(
        "term-structure",
        aliases=["cir-1985", "vasicek", "yield-curve"],
        help="Evaluate CIR (1985) & Vasicek (1977) affine yield curve & zero-coupon bond prices",
    )
    ts_parser.add_argument(
        "--ticker",
        default="BOND",
        help="Instrument / benchmark ticker symbol (default: BOND)",
    )
    ts_parser.add_argument(
        "--r0",
        type=float,
        default=0.035,
        help="Instantaneous short rate r_0 (default: 0.035)",
    )
    ts_parser.add_argument(
        "--theta",
        type=float,
        default=0.050,
        help="Long-run mean short rate theta (default: 0.050)",
    )
    ts_parser.add_argument(
        "--kappa",
        type=float,
        default=0.35,
        help="Speed of mean reversion kappa (default: 0.35)",
    )
    ts_parser.add_argument(
        "--sigma",
        type=float,
        default=0.04,
        help="Short-rate diffusion volatility sigma (default: 0.04)",
    )
    ts_parser.add_argument(
        "--model",
        default="CIR",
        choices=["CIR", "VASICEK"],
        help="Affine short-rate specification: CIR or VASICEK (default: CIR)",
    )
    ts_parser.add_argument(
        "--json",
        action="store_true",
        help="Output affine term structure receipt as JSON",
    )

    # CRR (1979) Binomial Option Pricing parser
    bo_parser = subparsers.add_parser(
        "binomial-option",
        aliases=["crr-1979", "american-option", "option-tree"],
        help="Value American & European options via Cox-Ross-Rubinstein (1979) binomial lattice",
    )
    bo_parser.add_argument(
        "--ticker",
        default="OPT",
        help="Underlying ticker symbol (default: OPT)",
    )
    bo_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Underlying spot price S_0 (default: 100.0)",
    )
    bo_parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Option strike price K (default: 100.0)",
    )
    bo_parser.add_argument(
        "--maturity",
        type=float,
        default=1.0,
        help="Time to expiration T in years (default: 1.0)",
    )
    bo_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Continuously compounded risk-free rate r (default: 0.05)",
    )
    bo_parser.add_argument(
        "--div-yield",
        type=float,
        default=0.0,
        help="Continuous dividend yield q (default: 0.0)",
    )
    bo_parser.add_argument(
        "--vol",
        type=float,
        default=0.25,
        help="Annualized volatility sigma (default: 0.25)",
    )
    bo_parser.add_argument(
        "--steps",
        type=int,
        default=150,
        help="Number of recombining binomial steps N in [10, 1000] (default: 150)",
    )
    bo_parser.add_argument(
        "--type",
        default="PUT",
        choices=["CALL", "PUT"],
        help="Option payoff type: CALL or PUT (default: PUT)",
    )
    bo_parser.add_argument(
        "--style",
        default="AMERICAN",
        choices=["AMERICAN", "EUROPEAN"],
        help="Exercise style: AMERICAN or EUROPEAN (default: AMERICAN)",
    )
    bo_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CRR binomial option receipt as JSON",
    )

    # Margrabe (1978) Exchange Option parser
    mx_parser = subparsers.add_parser(
        "exchange-option",
        aliases=["margrabe", "margrabe-1978", "switch-option"],
        help="Price Margrabe (1978) two-asset relative-performance exchange option",
    )
    mx_parser.add_argument(
        "--asset1",
        default="ASSET1",
        help="Target outperforming asset ticker S_1 (default: ASSET1)",
    )
    mx_parser.add_argument(
        "--asset2",
        default="SPY",
        help="Surrendered benchmark asset ticker S_2 (default: SPY)",
    )
    mx_parser.add_argument(
        "--s1",
        type=float,
        default=100.0,
        help="Spot price of Asset 1 S_1 (default: 100.0)",
    )
    mx_parser.add_argument(
        "--s2",
        type=float,
        default=100.0,
        help="Spot price of Asset 2 S_2 (default: 100.0)",
    )
    mx_parser.add_argument(
        "--vol1",
        type=float,
        default=0.25,
        help="Volatility sigma_1 of Asset 1 (default: 0.25)",
    )
    mx_parser.add_argument(
        "--vol2",
        type=float,
        default=0.20,
        help="Volatility sigma_2 of Asset 2 (default: 0.20)",
    )
    mx_parser.add_argument(
        "--div1",
        type=float,
        default=0.01,
        help="Dividend yield q_1 of Asset 1 (default: 0.01)",
    )
    mx_parser.add_argument(
        "--div2",
        type=float,
        default=0.015,
        help="Dividend yield q_2 of Asset 2 (default: 0.015)",
    )
    mx_parser.add_argument(
        "--rho",
        type=float,
        default=0.50,
        help="Return correlation rho between Asset 1 and Asset 2 in [-1, 1] (default: 0.50)",
    )
    mx_parser.add_argument(
        "--maturity",
        type=float,
        default=1.0,
        help="Maturity horizon T in years (default: 1.0)",
    )
    mx_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Margrabe exchange option receipt as JSON",
    )

    # Black & Cox (1976) First-Passage Covenant Credit parser
    bc_parser = subparsers.add_parser(
        "black-cox",
        aliases=["first-passage", "safety-covenant", "bc-1976"],
        help="Evaluate Black & Cox (1976) first-passage structural credit & safety covenants",
    )
    bc_parser.add_argument(
        "--ticker",
        default="CORP",
        help="Corporate issuer ticker symbol (default: CORP)",
    )
    bc_parser.add_argument(
        "--v0",
        type=float,
        default=150.0,
        help="Current firm asset value V_0 (default: 150.0)",
    )
    bc_parser.add_argument(
        "--debt",
        type=float,
        default=100.0,
        help="Debt face value F at maturity (default: 100.0)",
    )
    bc_parser.add_argument(
        "--covenant-k",
        type=float,
        default=80.0,
        help="Terminal safety covenant level K <= F (default: 80.0)",
    )
    bc_parser.add_argument(
        "--gamma",
        type=float,
        default=0.03,
        help="Exponential covenant growth rate gamma (default: 0.03)",
    )
    bc_parser.add_argument(
        "--maturity",
        type=float,
        default=5.0,
        help="Debt maturity T in years (default: 5.0)",
    )
    bc_parser.add_argument(
        "--vol",
        type=float,
        default=0.28,
        help="Annualized firm asset volatility sigma (default: 0.28)",
    )
    bc_parser.add_argument(
        "--rate",
        type=float,
        default=0.045,
        help="Risk-free interest rate r (default: 0.045)",
    )
    bc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black & Cox covenant credit receipt as JSON",
    )

    # Kim & Verrecchia (1991, 1994) Announcement Liquidity parser
    kv_parser = subparsers.add_parser(
        "announcement-liquidity",
        aliases=["kim-verrecchia", "kv-1991", "differential-precision"],
        help="Evaluate Kim & Verrecchia (1991, 1994) disclosure volume & illiquidity surge",
    )
    kv_parser.add_argument(
        "--ticker",
        default="KV91",
        help="Security ticker symbol (default: KV91)",
    )
    kv_parser.add_argument(
        "--prior-h",
        type=float,
        default=4.0,
        help="Pre-announcement prior precision h (default: 4.0)",
    )
    kv_parser.add_argument(
        "--pub-s",
        type=float,
        default=2.0,
        help="Public disclosure signal precision s (default: 2.0)",
    )
    kv_parser.add_argument(
        "--priv-p",
        type=float,
        default=3.0,
        help="Average private information precision p_bar (default: 3.0)",
    )
    kv_parser.add_argument(
        "--dispersion",
        type=float,
        default=1.5,
        help="Differential private precision dispersion sigma_p (default: 1.5)",
    )
    kv_parser.add_argument(
        "--abs-dp",
        type=float,
        default=0.04,
        help="Absolute announcement price revision |Delta P| (default: 0.04)",
    )
    kv_parser.add_argument(
        "--noise-u",
        type=float,
        default=0.50,
        help="Noise liquidity depth std dev sigma_u (default: 0.50)",
    )
    kv_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kim & Verrecchia receipt as JSON",
    )

    # Myers & Rajan (1998) Paradox of Liquidity parser
    mr_parser = subparsers.add_parser(
        "liquidity-paradox",
        aliases=["myers-rajan", "mr-1998"],
        help="Evaluate Myers & Rajan (1998) paradox of liquidity & debt capacity",
    )
    mr_parser.add_argument(
        "--ticker",
        default="MR98",
        help="Firm / corporate issuer ticker symbol (default: MR98)",
    )
    mr_parser.add_argument(
        "--assets",
        type=float,
        default=100.0,
        help="Invested assets base c_1 (default: 100.0)",
    )
    mr_parser.add_argument(
        "--continuation",
        type=float,
        default=125.0,
        help="Going-concern continuation cash flow d_2 (default: 125.0)",
    )
    mr_parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Asset secondary market liquidity alpha in (0, 1] (default: 0.75)",
    )
    mr_parser.add_argument(
        "--lambda-trans",
        type=float,
        default=0.25,
        help="Insider asset transformation friction lambda in [0, 1] (default: 0.25)",
    )
    mr_parser.add_argument(
        "--theta",
        type=float,
        default=0.0,
        help="Insider going-concern control rent share theta in [0, 1) (default: 0.0)",
    )
    mr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Myers & Rajan receipt as JSON",
    )

    # Hull & White (1990) No-Arbitrage Bond Option parser
    hw_parser = subparsers.add_parser(
        "hull-white",
        aliases=["hw-1990", "bond-option"],
        help="Price zero-coupon bond options under Hull & White (1990) exact term structure fit",
    )
    hw_parser.add_argument(
        "--ticker",
        default="HW90",
        help="Fixed-income benchmark ticker symbol (default: HW90)",
    )
    hw_parser.add_argument(
        "--r0",
        type=float,
        default=0.035,
        help="Initial instantaneous short rate r_0 (default: 0.035)",
    )
    hw_parser.add_argument(
        "--y-inf",
        type=float,
        default=0.050,
        help="Long-horizon asymptotic yield y_inf (default: 0.050)",
    )
    hw_parser.add_argument(
        "--a",
        type=float,
        default=0.15,
        help="Short-rate mean reversion parameter a (default: 0.15)",
    )
    hw_parser.add_argument(
        "--sigma",
        type=float,
        default=0.015,
        help="Normal short-rate diffusion volatility sigma (default: 0.015)",
    )
    hw_parser.add_argument(
        "--t-opt",
        type=float,
        default=2.0,
        help="Option expiration T in years (default: 2.0)",
    )
    hw_parser.add_argument(
        "--s-bond",
        type=float,
        default=10.0,
        help="Underlying zero-coupon bond maturity S > T in years (default: 10.0)",
    )
    hw_parser.add_argument(
        "--strike",
        type=float,
        default=0.65,
        help="Zero-coupon bond option strike K (default: 0.65)",
    )
    hw_parser.add_argument(
        "--type",
        default="CALL",
        choices=["CALL", "PUT"],
        help="Bond option type: CALL or PUT (default: CALL)",
    )
    hw_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hull & White receipt as JSON",
    )

    # Modigliani & Miller (1958, 1961, 1963) Corporate Finance parser
    mm_parser = subparsers.add_parser(
        "modigliani-miller",
        aliases=["dividend-irrelevance", "mm-1961"],
        help="Evaluate Modigliani & Miller (1958, 1961, 1963) Propositions I/II, WACC & PVGO",
    )
    mm_parser.add_argument(
        "--ticker",
        default="MM61",
        help="Corporate issuer ticker symbol (default: MM61)",
    )
    mm_parser.add_argument(
        "--ebit",
        type=float,
        default=20.0,
        help="Annual operating income EBIT (default: 20.0)",
    )
    mm_parser.add_argument(
        "--debt",
        type=float,
        default=50.0,
        help="Market debt D (default: 50.0)",
    )
    mm_parser.add_argument(
        "--r0",
        type=float,
        default=0.10,
        help="Unlevered asset cost of capital r_0 (default: 0.10)",
    )
    mm_parser.add_argument(
        "--rd",
        type=float,
        default=0.05,
        help="Pre-tax cost of debt r_D (default: 0.05)",
    )
    mm_parser.add_argument(
        "--tax-rate",
        type=float,
        default=0.21,
        help="Corporate income tax rate tau_c in [0, 1) (default: 0.21)",
    )
    mm_parser.add_argument(
        "--reinvest",
        type=float,
        default=0.40,
        help="Real capital reinvestment rate k in [0, 1] (default: 0.40)",
    )
    mm_parser.add_argument(
        "--roic",
        type=float,
        default=0.14,
        help="Incremental return on invested capital ROIC (default: 0.14)",
    )
    mm_parser.add_argument(
        "--payout",
        type=float,
        default=0.60,
        help="Dividend payout ratio in [0, 1] (default: 0.60)",
    )
    mm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Modigliani & Miller receipt as JSON",
    )

    # McDonald & Siegel (1986) Value of Waiting Real Option parser
    ms_parser = subparsers.add_parser(
        "waiting-to-invest",
        aliases=["mcdonald-siegel", "ms-1986", "irreversible-investment"],
        help="Evaluate McDonald & Siegel (1986) irreversible capex timing & waiting value",
    )
    ms_parser.add_argument(
        "--ticker",
        default="MS86",
        help="Project / issuer ticker symbol (default: MS86)",
    )
    ms_parser.add_argument(
        "--v",
        type=float,
        default=130.0,
        help="Present value of completed project cash flows V (default: 130.0)",
    )
    ms_parser.add_argument(
        "--cost",
        type=float,
        default=100.0,
        help="Irreversible sunk investment cost I (default: 100.0)",
    )
    ms_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Risk-free interest rate r (default: 0.05)",
    )
    ms_parser.add_argument(
        "--delta",
        type=float,
        default=0.04,
        help="Dividend / convenience yield shortfall delta > 0 (default: 0.04)",
    )
    ms_parser.add_argument(
        "--vol",
        type=float,
        default=0.28,
        help="Project cash-flow diffusion volatility sigma (default: 0.28)",
    )
    ms_parser.add_argument(
        "--json",
        action="store_true",
        help="Output McDonald & Siegel real-option receipt as JSON",
    )

    # Shleifer & Vishny (1992) Industry Liquidation parser
    svl_parser = subparsers.add_parser(
        "industry-liquidation",
        aliases=["sv-1992"],
        help="Evaluate Shleifer & Vishny (1992) liquidation values & industry debt capacity",
    )
    svl_parser.add_argument(
        "--ticker",
        default="SV92",
        help="Issuer ticker symbol (default: SV92)",
    )
    svl_parser.add_argument(
        "--v1",
        type=float,
        default=100.0,
        help="First-best specialist asset value V_1 (default: 100.0)",
    )
    svl_parser.add_argument(
        "--specificity",
        type=float,
        default=0.40,
        help="Outside generalist liquidation discount delta_spec in [0, 1) (default: 0.40)",
    )
    svl_parser.add_argument(
        "--peer-slack",
        type=float,
        default=0.35,
        help="Healthy peer balance-sheet absorption slack in [0, 1] (default: 0.35)",
    )
    svl_parser.add_argument(
        "--rho",
        type=float,
        default=0.65,
        help="Correlated industry downturn distress probability in [0, 1] (default: 0.65)",
    )
    svl_parser.add_argument(
        "--downturn-prob",
        type=float,
        default=0.30,
        help="Ex-ante state distress likelihood q in (0, 1) (default: 0.30)",
    )
    svl_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Shleifer & Vishny liquidation receipt as JSON",
    )

    # Dixit (1989) Entry-Exit Hysteresis Band parser
    dx_parser = subparsers.add_parser(
        "entry-exit",
        aliases=["hysteresis-exit", "dixit-1989"],
        help="Evaluate Dixit (1989) two-sided entry/exit hysteresis triggers & switching options",
    )
    dx_parser.add_argument(
        "--ticker",
        default="DX89",
        help="Operator / asset ticker symbol (default: DX89)",
    )
    dx_parser.add_argument(
        "--price",
        type=float,
        default=1.10,
        help="Current output price P (default: 1.10)",
    )
    dx_parser.add_argument(
        "--cost",
        type=float,
        default=1.00,
        help="Flow variable operating cost C (default: 1.00)",
    )
    dx_parser.add_argument(
        "--entry-k",
        type=float,
        default=2.00,
        help="Sunk entry capex k (default: 2.00)",
    )
    dx_parser.add_argument(
        "--exit-l",
        type=float,
        default=1.50,
        help="Sunk abandonment / exit cost l (default: 1.50)",
    )
    dx_parser.add_argument(
        "--vol",
        type=float,
        default=0.25,
        help="Output price diffusion volatility sigma (default: 0.25)",
    )
    dx_parser.add_argument(
        "--state",
        default="ACTIVE",
        choices=["IDLE", "ACTIVE"],
        help="Current production status: IDLE or ACTIVE (default: ACTIVE)",
    )
    dx_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Dixit hysteresis receipt as JSON",
    )

    # Holmstrom & Tirole (1998) Public Supply of Liquidity parser
    ht98_parser = subparsers.add_parser(
        "public-liquidity",
        aliases=["ht-1998", "liquidity-line"],
        help="Evaluate Holmstrom & Tirole (1998) private pooling vs. public liquidity supply",
    )
    ht98_parser.add_argument(
        "--ticker",
        default="HT98",
        help="Corporate / banking benchmark ticker symbol (default: HT98)",
    )
    ht98_parser.add_argument(
        "--equity",
        type=float,
        default=20.0,
        help="Entrepreneur initial net worth A (default: 20.0)",
    )
    ht98_parser.add_argument(
        "--rho0",
        type=float,
        default=0.70,
        help="Pledgeable return per investment unit rho_0 (default: 0.70)",
    )
    ht98_parser.add_argument(
        "--rho1",
        type=float,
        default=1.45,
        help="Total continuation return rho_1 > rho_0 (default: 1.45)",
    )
    ht98_parser.add_argument(
        "--shock",
        type=float,
        default=0.90,
        help="Interim liquidity reinvestment shock rho (default: 0.90)",
    )
    ht98_parser.add_argument(
        "--theta",
        type=float,
        default=0.35,
        help="Probability theta in (0, 1) of interim liquidity shock (default: 0.35)",
    )
    ht98_parser.add_argument(
        "--lambda-agg",
        type=float,
        default=0.60,
        help="Fraction lambda in [0, 1] of aggregate correlated liquidity risk (default: 0.60)",
    )
    ht98_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Holmstrom & Tirole (1998) receipt as JSON",
    )

    # Black & Jones (1987) CPPI & TIPP Portfolio Insurance parser
    cppi_parser = subparsers.add_parser(
        "cppi",
        aliases=["portfolio-insurance", "black-jones"],
        help="Evaluate Black & Jones (1987) CPPI / TIPP dynamic floor portfolio insurance",
    )
    cppi_parser.add_argument(
        "--ticker",
        default="CPPI",
        help="Portfolio / underlying asset ticker symbol (default: CPPI)",
    )
    cppi_parser.add_argument(
        "--wealth",
        type=float,
        default=100.0,
        help="Current portfolio wealth W (default: 100.0)",
    )
    cppi_parser.add_argument(
        "--floor",
        type=float,
        default=85.0,
        help="Protected capital floor F (default: 85.0)",
    )
    cppi_parser.add_argument(
        "--multiplier",
        type=float,
        default=4.0,
        help="CPPI cushion multiplier m >= 1.0 (default: 4.0)",
    )
    cppi_parser.add_argument(
        "--max-lev",
        type=float,
        default=1.0,
        help="Maximum risky weight cap in [1.0, 2.0] (default: 1.0)",
    )
    cppi_parser.add_argument(
        "--strategy",
        default="CPPI",
        choices=["CPPI", "TIPP"],
        help="Floor insurance variant: CPPI or TIPP (default: CPPI)",
    )
    cppi_parser.add_argument(
        "--json",
        action="store_true",
        help="Output CPPI portfolio insurance receipt as JSON",
    )

    # Harris & Raviv (1990) Informational Role of Debt parser
    hr90_parser = subparsers.add_parser(
        "informational-debt",
        aliases=["hr-1990", "default-investigation"],
        help="Evaluate Harris & Raviv (1990) debt auditing & liquidation discipline equilibrium",
    )
    hr90_parser.add_argument(
        "--ticker",
        default="HR90",
        help="Corporate issuer ticker symbol (default: HR90)",
    )
    hr90_parser.add_argument(
        "--y-min",
        type=float,
        default=40.0,
        help="Lower bound of going-concern cash flow support y_min (default: 40.0)",
    )
    hr90_parser.add_argument(
        "--y-max",
        type=float,
        default=160.0,
        help="Upper bound of going-concern cash flow support y_max (default: 160.0)",
    )
    hr90_parser.add_argument(
        "--liq-val",
        type=float,
        default=95.0,
        help="Tangible asset liquidation value L (default: 95.0)",
    )
    hr90_parser.add_argument(
        "--audit-cost",
        type=float,
        default=8.0,
        help="Bankruptcy default investigation cost k (default: 8.0)",
    )
    hr90_parser.add_argument(
        "--debt",
        type=float,
        default=None,
        help="Optional promised debt level D (defaults to optimal D*)",
    )
    hr90_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Harris & Raviv informational debt receipt as JSON",
    )

    # Berk & Green (2004) Rational Fund Flows & Alpha Capacity parser
    bg_parser = subparsers.add_parser(
        "fund-flows",
        aliases=["berk-green", "bg-2004", "capacity-alpha"],
        help="Evaluate Berk & Green (2004) Bayesian skill, rational AUM flows & alpha capacity",
    )
    bg_parser.add_argument(
        "--ticker",
        default="BG04",
        help="Fund / strategy ticker symbol (default: BG04)",
    )
    bg_parser.add_argument(
        "--prior",
        type=float,
        default=0.03,
        help="Prior expected gross annual skill phi_0 (default: 0.03)",
    )
    bg_parser.add_argument(
        "--realized",
        type=float,
        default=0.06,
        help="Realized gross active alpha R_t (default: 0.06)",
    )
    bg_parser.add_argument(
        "--fee",
        type=float,
        default=0.010,
        help="Annual management fee f (default: 0.010)",
    )
    bg_parser.add_argument(
        "--cost-a",
        type=float,
        default=0.005,
        help="Quadratic scale cost slope a > 0 (default: 0.005)",
    )
    bg_parser.add_argument(
        "--aum",
        type=float,
        default=3.0,
        help="Current assets under management q in $B (default: 3.0)",
    )
    bg_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Berk & Green receipt as JSON",
    )

    # Diamond (1991) Debt Maturity & Rollover Risk parser
    dm_parser = subparsers.add_parser(
        "debt-maturity",
        aliases=["diamond-1991", "rollover-risk"],
        help="Evaluate Diamond (1991) short-term vs. long-term debt maturity & rollover risk",
    )
    dm_parser.add_argument(
        "--ticker",
        default="DM91",
        help="Borrower / issuer ticker symbol (default: DM91)",
    )
    dm_parser.add_argument(
        "--rating-f",
        type=float,
        default=0.72,
        help="Prior good-borrower credit rating probability f in (0, 1) (default: 0.72)",
    )
    dm_parser.add_argument(
        "--liq-l",
        type=float,
        default=0.70,
        help="Interim date-1 forced liquidation value L < X (default: 0.70)",
    )
    dm_parser.add_argument(
        "--rent-c",
        type=float,
        default=0.50,
        help="Manager private non-pecuniary control rent C >= 0 (default: 0.50)",
    )
    dm_parser.add_argument(
        "--qg",
        type=float,
        default=0.25,
        help="Good-type interim bad-news probability q_G (default: 0.25)",
    )
    dm_parser.add_argument(
        "--qb",
        type=float,
        default=0.80,
        help="Bad-type interim bad-news probability q_B >= q_G (default: 0.80)",
    )
    dm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Diamond (1991) debt maturity receipt as JSON",
    )

    # Blanchard & Watson (1982) Rational Speculative Bubbles parser
    bw_parser = subparsers.add_parser(
        "rational-bubble",
        aliases=["blanchard-watson", "bw-1982"],
        help="Evaluate Blanchard & Watson (1982) rational bubbles & explosive crash hazard",
    )
    bw_parser.add_argument(
        "--ticker",
        default="BW82",
        help="Asset ticker symbol (default: BW82)",
    )
    bw_parser.add_argument(
        "--price",
        type=float,
        default=150.0,
        help="Observed market asset price P_t (default: 150.0)",
    )
    bw_parser.add_argument(
        "--fundamental",
        type=float,
        default=100.0,
        help="Fundamental discounted cash-flow anchor F_t (default: 100.0)",
    )
    bw_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Required periodic discount rate r > 0 (default: 0.05)",
    )
    bw_parser.add_argument(
        "--pi",
        type=float,
        default=0.85,
        help="Bubble per-period survival probability pi in [0.10, 1.0] (default: 0.85)",
    )
    bw_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Blanchard & Watson receipt as JSON",
    )

    # Fischer Black (1972) Zero-Beta CAPM parser
    bz_parser = subparsers.add_parser(
        "zero-beta",
        aliases=["black-1972"],
        help="Evaluate Fischer Black (1972) restricted-borrowing Zero-Beta CAPM & SML flattening",
    )
    bz_parser.add_argument(
        "--ticker",
        default="BZ72",
        help="Asset ticker symbol (default: BZ72)",
    )
    bz_parser.add_argument(
        "--beta",
        type=float,
        default=0.70,
        help="Asset systematic market beta (default: 0.70)",
    )
    bz_parser.add_argument(
        "--rf",
        type=float,
        default=0.035,
        help="Unconstrained risk-free lending rate r_f (default: 0.035)",
    )
    bz_parser.add_argument(
        "--rz",
        type=float,
        default=0.055,
        help="Orthogonal minimum-variance Zero-Beta return R_z >= r_f (default: 0.055)",
    )
    bz_parser.add_argument(
        "--rm",
        type=float,
        default=0.095,
        help="Expected market portfolio return R_m > R_z (default: 0.095)",
    )
    bz_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black Zero-Beta CAPM receipt as JSON",
    )

    # Jeremy C. Stein (1997) Internal Capital Markets parser
    st97_parser = subparsers.add_parser(
        "internal-capital",
        aliases=["winner-picking", "stein-1997"],
        help="Evaluate Stein (1997) internal capital markets, winner-picking & conglomerate discount",
    )
    st97_parser.add_argument(
        "--ticker",
        default="ST97",
        help="Conglomerate ticker symbol (default: ST97)",
    )
    st97_parser.add_argument(
        "--theta1",
        type=float,
        default=8.0,
        help="Division 1 productivity parameter theta_1 > 0 (default: 8.0)",
    )
    st97_parser.add_argument(
        "--theta2",
        type=float,
        default=4.0,
        help="Division 2 productivity parameter theta_2 > 0 (default: 4.0)",
    )
    st97_parser.add_argument(
        "--c1",
        type=float,
        default=5.0,
        help="Division 1 standalone credit budget c_1 (default: 5.0)",
    )
    st97_parser.add_argument(
        "--c2",
        type=float,
        default=5.0,
        help="Division 2 standalone credit budget c_2 (default: 5.0)",
    )
    st97_parser.add_argument(
        "--agency-lambda",
        type=float,
        default=0.20,
        help="Cross-subsidization socialism friction lambda in [0, 1] (default: 0.20)",
    )
    st97_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Stein (1997) internal capital receipt as JSON",
    )

    # Robert F. Stambaugh (1999) Predictive Regressions Bias parser
    rs99_parser = subparsers.add_parser(
        "predictive-regression",
        aliases=["stambaugh-bias", "rs-1999"],
        help="Evaluate Stambaugh (1999) finite-sample predictive regression bias & true t-stat",
    )
    rs99_parser.add_argument(
        "--ticker",
        default="RS99",
        help="Security ticker symbol (default: RS99)",
    )
    rs99_parser.add_argument(
        "--ols-beta",
        type=float,
        default=0.08,
        help="Unadjusted OLS predictive regression slope beta_hat (default: 0.08)",
    )
    rs99_parser.add_argument(
        "--rho",
        type=float,
        default=0.96,
        help="Predictor AR(1) autocorrelation rho in (-0.999, 1.0) (default: 0.96)",
    )
    rs99_parser.add_argument(
        "--gamma",
        type=float,
        default=-0.80,
        help="Return-predictor shock covariance ratio Cov(u,v)/Var(v) (default: -0.80)",
    )
    rs99_parser.add_argument(
        "--sample-t",
        type=int,
        default=40,
        help="Effective finite sample observations T >= 12 (default: 40)",
    )
    rs99_parser.add_argument(
        "--std-err",
        type=float,
        default=0.035,
        help="OLS standard error on predictive slope > 0 (default: 0.035)",
    )
    rs99_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Stambaugh predictive bias receipt as JSON",
    )

    # Amihud & Mendelson (1986) Spread-Return Clientele parser
    am86_parser = subparsers.add_parser(
        "amihud-mendelson",
        aliases=["am-1986", "clientele-spread"],
        help="Evaluate Amihud & Mendelson (1986) bid-ask spread pricing & clientele equilibrium",
    )
    am86_parser.add_argument(
        "--ticker",
        default="AM86",
        help="Security ticker symbol (default: AM86)",
    )
    am86_parser.add_argument(
        "--spread",
        type=float,
        default=0.015,
        help="Proportional bid-ask spread S in [0, 0.25] (default: 0.015 = 150 bps)",
    )
    am86_parser.add_argument(
        "--base-r0",
        type=float,
        default=0.06,
        help="Frictionless zero-spread expected return R_0 (default: 0.06)",
    )
    am86_parser.add_argument(
        "--turnover",
        type=float,
        default=1.0,
        help="Investor portfolio turnover rate mu = 1/holding_years (default: 1.0)",
    )
    am86_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Amihud & Mendelson receipt as JSON",
    )

    # Fischer Black (1976) Futures & Commodity Option parser
    b76_parser = subparsers.add_parser(
        "futures-option",
        aliases=["black-76", "commodity-option"],
        help="Price forward, futures, and commodity options via Fischer Black (1976)",
    )
    b76_parser.add_argument(
        "--ticker",
        default="CL",
        help="Futures / commodity ticker symbol (default: CL)",
    )
    b76_parser.add_argument(
        "--futures",
        type=float,
        default=100.0,
        help="Current futures / forward price F (default: 100.0)",
    )
    b76_parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Option strike price K (default: 100.0)",
    )
    b76_parser.add_argument(
        "--maturity",
        type=float,
        default=1.0,
        help="Time to option expiration T in years (default: 1.0)",
    )
    b76_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Risk-free discounting rate r (default: 0.05)",
    )
    b76_parser.add_argument(
        "--vol",
        type=float,
        default=0.25,
        help="Annualized futures volatility sigma (default: 0.25)",
    )
    b76_parser.add_argument(
        "--type",
        default="CALL",
        choices=["CALL", "PUT"],
        help="Option payoff type: CALL or PUT (default: CALL)",
    )
    b76_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black-76 futures option receipt as JSON",
    )

    # Merton & Perold (1993) Risk Capital & Allocation parser
    mp_parser = subparsers.add_parser(
        "risk-capital",
        aliases=["merton-perold", "mp-1993"],
        help="Compute Merton & Perold (1993) risk capital, diversification & Euler RAROC",
    )
    mp_parser.add_argument(
        "--ticker",
        default="JPM",
        help="Primary firm or desk ticker symbol (default: JPM)",
    )
    mp_parser.add_argument(
        "--v1",
        type=float,
        default=100.0,
        help="Primary unit asset value V1 (default: 100.0)",
    )
    mp_parser.add_argument(
        "--vol1",
        type=float,
        default=0.30,
        help="Primary unit annualized asset volatility (default: 0.30)",
    )
    mp_parser.add_argument(
        "--v2",
        type=float,
        default=100.0,
        help="Secondary diversification unit asset value V2 (default: 100.0)",
    )
    mp_parser.add_argument(
        "--vol2",
        type=float,
        default=0.20,
        help="Secondary unit annualized asset volatility (default: 0.20)",
    )
    mp_parser.add_argument(
        "--corr",
        type=float,
        default=0.25,
        help="Cross-unit asset correlation rho in [-0.9999, 1.0] (default: 0.25)",
    )
    mp_parser.add_argument(
        "--horizon",
        type=float,
        default=1.0,
        help="Risk capital guarantee horizon T in years (default: 1.0)",
    )
    mp_parser.add_argument(
        "--rate",
        type=float,
        default=0.05,
        help="Risk-free benchmark rate r (default: 0.05)",
    )
    mp_parser.add_argument(
        "--excess-return",
        type=float,
        default=0.06,
        help="Expected unit excess return over risk-free rate (default: 0.06)",
    )
    mp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton & Perold risk capital receipt as JSON",
    )

    # Fama & Bliss (1987) Forward-Spot Term Spread parser
    fb_parser = subparsers.add_parser(
        "fama-bliss",
        aliases=["forward-spread", "fb-1987"],
        help="Analyze Fama & Bliss (1987) forward-spot term spreads & excess bond returns",
    )
    fb_parser.add_argument(
        "--ticker",
        default="TLT",
        help="Fixed-income / treasury ETF or bond benchmark ticker (default: TLT)",
    )
    fb_parser.add_argument(
        "--y1",
        type=float,
        default=0.040,
        help="1-year zero-coupon spot yield (default: 0.040)",
    )
    fb_parser.add_argument(
        "--yn",
        type=float,
        default=0.052,
        help="n-year zero-coupon spot yield (default: 0.052)",
    )
    fb_parser.add_argument(
        "--y-nm1",
        type=float,
        default=None,
        dest="y_nm1",
        help="Optional (n-1)-year spot yield (interpolated linearly if omitted)",
    )
    fb_parser.add_argument(
        "--maturity",
        type=int,
        default=2,
        help="Bond maturity n in [2, 30] years (default: 2)",
    )
    fb_parser.add_argument(
        "--next-y1",
        type=float,
        default=None,
        dest="next_y1",
        help="Realized (n-1)-year yield one year later (default: unchanged curve)",
    )
    fb_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Fama & Bliss forward-spot receipt as JSON",
    )

    # Duffie & Singleton (1999) Reduced-Form Credit Spread parser
    ds_parser = subparsers.add_parser(
        "reduced-credit",
        aliases=["duffie-singleton", "ds-1999"],
        help="Price defaultable bonds & hazard spreads via Duffie & Singleton (1999)",
    )
    ds_parser.add_argument(
        "--ticker",
        default="HYG",
        help="Credit ETF or corporate bond ticker symbol (default: HYG)",
    )
    ds_parser.add_argument(
        "--hazard",
        type=float,
        default=0.04,
        help="Poisson default hazard intensity h (default: 0.04)",
    )
    ds_parser.add_argument(
        "--recovery",
        type=float,
        default=0.40,
        help="Recovery of market value omega in [0, 0.99] (default: 0.40)",
    )
    ds_parser.add_argument(
        "--rate",
        type=float,
        default=0.045,
        help="Risk-free benchmark short rate r (default: 0.045)",
    )
    ds_parser.add_argument(
        "--maturity",
        type=float,
        default=5.0,
        help="Bond maturity T in years (default: 5.0)",
    )
    ds_parser.add_argument(
        "--notional",
        type=float,
        default=100.0,
        help="Bond face notional value (default: 100.0)",
    )
    ds_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Duffie & Singleton credit receipt as JSON",
    )

    # Grinold (1989) & Clarke-de Silva-Thorley (2002) Fundamental Law parser
    fl_parser = subparsers.add_parser(
        "fundamental-law",
        aliases=["grinold-kahn", "gk-1989"],
        help="Evaluate active IR, transfer coefficient & effective breadth via Grinold-Kahn",
    )
    fl_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Benchmark or active strategy ticker symbol (default: SPY)",
    )
    fl_parser.add_argument(
        "--ic",
        type=float,
        default=0.06,
        help="Information Coefficient IC in [-1.0, 1.0] (default: 0.06)",
    )
    fl_parser.add_argument(
        "--breadth",
        type=float,
        default=250.0,
        help="Independent annual decision breadth BR >= 1 (default: 250.0)",
    )
    fl_parser.add_argument(
        "--tc",
        type=float,
        default=0.70,
        help="Transfer Coefficient TC in [0.0, 1.0] (default: 0.70)",
    )
    fl_parser.add_argument(
        "--te",
        type=float,
        default=0.04,
        help="Annualized active tracking error TE (default: 0.04)",
    )
    fl_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Fundamental Law active management receipt as JSON",
    )

    # Philippe Jorion (1986) Bayes-Stein Shrinkage Portfolio parser
    jbs_parser = subparsers.add_parser(
        "bayes-stein",
        aliases=["jorion-1986", "shrinkage-weights"],
        help="Compute Philippe Jorion (1986) Bayes-Stein return shrinkage & predictive covariance",
    )
    jbs_parser.add_argument(
        "--ticker",
        default="PORT",
        help="Portfolio or benchmark ticker identifier (default: PORT)",
    )
    jbs_parser.add_argument(
        "--n-assets",
        type=int,
        default=4,
        dest="n_assets",
        help="Number of portfolio assets N in [2, 50] (default: 4)",
    )
    jbs_parser.add_argument(
        "--sample-t",
        type=int,
        default=60,
        dest="sample_t",
        help="Historical sample window length T > N + 2 (default: 60)",
    )
    jbs_parser.add_argument(
        "--disp",
        type=float,
        default=0.06,
        help="Cross-asset sample mean dispersion (default: 0.06)",
    )
    jbs_parser.add_argument(
        "--vol",
        type=float,
        default=0.20,
        help="Annualized base asset volatility (default: 0.20)",
    )
    jbs_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Risk-free benchmark rate (default: 0.04)",
    )
    jbs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Jorion Bayes-Stein shrinkage receipt as JSON",
    )

    # Modigliani & Modigliani (1997) M^2 Risk-Adjusted Performance parser
    m2_parser = subparsers.add_parser(
        "m2-modigliani",
        aliases=["risk-adjusted-m2", "mm-rap"],
        help="Evaluate volatility-matched Modigliani M^2 & Treynor-Black Sharpe decomposition",
    )
    m2_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Fund, strategy, or equity ticker symbol (default: SPY)",
    )
    m2_parser.add_argument(
        "--rp",
        type=float,
        default=0.14,
        help="Portfolio annualized return r_p (default: 0.14)",
    )
    m2_parser.add_argument(
        "--vol-p",
        type=float,
        default=0.20,
        dest="vol_p",
        help="Portfolio annualized volatility sigma_p (default: 0.20)",
    )
    m2_parser.add_argument(
        "--rm",
        type=float,
        default=0.10,
        help="Benchmark market return r_m (default: 0.10)",
    )
    m2_parser.add_argument(
        "--vol-m",
        type=float,
        default=0.16,
        dest="vol_m",
        help="Benchmark market volatility sigma_m (default: 0.16)",
    )
    m2_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Risk-free benchmark rate r_f (default: 0.04)",
    )
    m2_parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Portfolio beta relative to benchmark (default: 1.0)",
    )
    m2_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Modigliani M^2 RAP receipt as JSON",
    )

    # Macaulay (1938) & Malkiel (1962) Bond Duration & Convexity parser
    mbd_parser = subparsers.add_parser(
        "bond-duration",
        aliases=["malkiel-duration", "convexity-theorems"],
        help="Compute Macaulay/Modified duration, DV01 & Malkiel convexity asymmetry",
    )
    mbd_parser.add_argument(
        "--ticker",
        default="TLT",
        help="Bond ETF or fixed-income ticker symbol (default: TLT)",
    )
    mbd_parser.add_argument(
        "--coupon",
        type=float,
        default=0.04,
        help="Annual bond coupon rate (default: 0.04)",
    )
    mbd_parser.add_argument(
        "--ytm",
        type=float,
        default=0.05,
        help="Annual yield to maturity (default: 0.05)",
    )
    mbd_parser.add_argument(
        "--maturity",
        type=float,
        default=10.0,
        help="Time to maturity in years (default: 10.0)",
    )
    mbd_parser.add_argument(
        "--freq",
        type=int,
        default=2,
        choices=[1, 2, 4, 12],
        help="Coupon compounding frequency per year (default: 2)",
    )
    mbd_parser.add_argument(
        "--shock-bps",
        type=float,
        default=100.0,
        dest="shock_bps",
        help="Symmetric yield shock magnitude in bps (default: 100.0)",
    )
    mbd_parser.add_argument(
        "--par",
        type=float,
        default=100.0,
        help="Bond face par value (default: 100.0)",
    )
    mbd_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Malkiel bond duration receipt as JSON",
    )

    # Brown, Goetzmann, Ibbotson & Ross (1992) Survivorship Bias parser
    bg_parser = subparsers.add_parser(
        "survivorship-bias",
        aliases=["bgir-1992", "truncation-bias"],
        help="Quantify truncated-normal survivorship return bias & Sharpe inflation (BGIR 1992)",
    )
    bg_parser.add_argument(
        "--ticker",
        default="HFRI",
        help="Fund index or strategy universe ticker (default: HFRI)",
    )
    bg_parser.add_argument(
        "--mu",
        type=float,
        default=0.08,
        help="True unconditional universe expected return (default: 0.08)",
    )
    bg_parser.add_argument(
        "--vol",
        type=float,
        default=0.20,
        help="True unconditional universe volatility (default: 0.20)",
    )
    bg_parser.add_argument(
        "--cutoff",
        type=float,
        default=-0.10,
        help="Annual liquidation attrition cutoff return (default: -0.10)",
    )
    bg_parser.add_argument(
        "--horizon",
        type=float,
        default=5.0,
        help="Cumulative survival attrition horizon in years (default: 5.0)",
    )
    bg_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Risk-free benchmark rate (default: 0.04)",
    )
    bg_parser.add_argument(
        "--json",
        action="store_true",
        help="Output BGIR survivorship bias receipt as JSON",
    )

    # William F. Sharpe (1992) Returns-Based Style Analysis parser
    rbsa_parser = subparsers.add_parser(
        "style-analysis",
        aliases=["sharpe-rbsa", "rbsa-1992"],
        help="Decompose portfolio style weights & selection risk via Sharpe (1992) RBSA",
    )
    rbsa_parser.add_argument(
        "--ticker",
        default="FUND",
        help="Fund or portfolio ticker symbol (default: FUND)",
    )
    rbsa_parser.add_argument(
        "--w-value",
        type=float,
        default=0.40,
        dest="w_value",
        help="Value Equity style weight proxy (default: 0.40)",
    )
    rbsa_parser.add_argument(
        "--w-growth",
        type=float,
        default=0.35,
        dest="w_growth",
        help="Growth Equity style weight proxy (default: 0.35)",
    )
    rbsa_parser.add_argument(
        "--w-small",
        type=float,
        default=0.15,
        dest="w_small",
        help="Small Cap style weight proxy (default: 0.15)",
    )
    rbsa_parser.add_argument(
        "--w-bond",
        type=float,
        default=0.10,
        dest="w_bond",
        help="Govt Bonds style weight proxy (default: 0.10)",
    )
    rbsa_parser.add_argument(
        "--alpha-ann",
        type=float,
        default=0.02,
        dest="alpha_ann",
        help="Annualized manager selection alpha (default: 0.02)",
    )
    rbsa_parser.add_argument(
        "--selection-vol",
        type=float,
        default=0.03,
        dest="selection_vol",
        help="Annualized active selection residual volatility (default: 0.03)",
    )
    rbsa_parser.add_argument(
        "--n-obs",
        type=int,
        default=120,
        dest="n_obs",
        help="Number of daily return observations (default: 120)",
    )
    rbsa_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Sharpe RBSA receipt as JSON",
    )

    # Treynor & Mazuy (1966) & Henriksson & Merton (1981) Market Timing parser
    mt_parser = subparsers.add_parser(
        "market-timing",
        aliases=["henriksson-merton", "treynor-mazuy", "hm-1981"],
        help="Evaluate parametric market-timing convexity & Bull/Bear beta asymmetry",
    )
    mt_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Fund or equity strategy ticker symbol (default: SPY)",
    )
    mt_parser.add_argument(
        "--beta-down",
        type=float,
        default=0.70,
        dest="beta_down",
        help="Downside bear-market beta exposure (default: 0.70)",
    )
    mt_parser.add_argument(
        "--gamma-hm",
        type=float,
        default=0.50,
        dest="gamma_hm",
        help="Henriksson-Merton option-like timing delta beta (default: 0.50)",
    )
    mt_parser.add_argument(
        "--alpha-ann",
        type=float,
        default=0.01,
        dest="alpha_ann",
        help="Annualized stock-selection alpha component (default: 0.01)",
    )
    mt_parser.add_argument(
        "--noise-vol",
        type=float,
        default=0.02,
        dest="noise_vol",
        help="Annualized idiosyncratic residual volatility (default: 0.02)",
    )
    mt_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annual risk-free benchmark rate (default: 0.04)",
    )
    mt_parser.add_argument(
        "--n-obs",
        type=int,
        default=160,
        dest="n_obs",
        help="Number of paired return sessions (default: 160)",
    )
    mt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Market Timing convexity receipt as JSON",
    )

    # Kelly (1956) & Thorp (1997) Fractional Kelly Log-Wealth Growth parser
    kelly_parser = subparsers.add_parser(
        "kelly-growth",
        aliases=["fractional-kelly", "kt-1956"],
        help="Evaluate Fractional Kelly optimal compounding growth & overbetting ruin risk",
    )
    kelly_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Asset or strategy ticker symbol (default: SPY)",
    )
    kelly_parser.add_argument(
        "--mu",
        type=float,
        default=0.12,
        help="Annualized arithmetic expected return mu (default: 0.12)",
    )
    kelly_parser.add_argument(
        "--vol",
        type=float,
        default=0.20,
        help="Annualized return volatility sigma (default: 0.20)",
    )
    kelly_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annual risk-free benchmark rate (default: 0.04)",
    )
    kelly_parser.add_argument(
        "--kelly-fraction",
        type=float,
        default=0.50,
        dest="kelly_fraction",
        help="Fractional Kelly multiplier k = f / f* (default: 0.50 Half-Kelly)",
    )
    kelly_parser.add_argument(
        "--drawdown-threshold",
        type=float,
        default=0.50,
        dest="drawdown_threshold",
        help="Drawdown level d in (0.05, 0.95) for P(DD >= d) ruin metric (default: 0.50)",
    )
    kelly_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Fractional Kelly growth receipt as JSON",
    )

    # Lo & MacKinlay (1988) Variance Ratio parser
    vr_parser = subparsers.add_parser(
        "variance-ratio",
        aliases=["lo-mackinlay", "lm-1988"],
        help="Run Lo & MacKinlay heteroskedasticity-robust Variance Ratio test",
    )
    vr_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Asset or index ticker symbol (default: SPY)",
    )
    vr_parser.add_argument(
        "--lag",
        type=int,
        default=5,
        help="Overlapping lag horizon q in [2, 126] (default: 5)",
    )
    vr_parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Synthetic AR(1) serial correlation coefficient (default: 0.0 random walk)",
    )
    vr_parser.add_argument(
        "--n-obs",
        type=int,
        default=260,
        dest="n_obs",
        help="Number of synthetic 1-period log returns (default: 260)",
    )
    vr_parser.add_argument(
        "--vol-daily",
        type=float,
        default=0.012,
        dest="vol_daily",
        help="Daily return volatility scale (default: 0.012)",
    )
    vr_parser.add_argument(
        "--heteroskedastic",
        action="store_true",
        help="Simulate volatility clustering regimes",
    )
    vr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Lo & MacKinlay Variance Ratio receipt as JSON",
    )

    # Gibbons, Ross & Shanken (1989) Multivariate GRS F-Test parser
    grs_parser = subparsers.add_parser(
        "grs-test",
        aliases=["gibbons-ross-shanken", "grs-1989"],
        help="Run exact finite-sample Gibbons-Ross-Shanken multivariate alpha efficiency F-test",
    )
    grs_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Benchmark factor proxy ticker symbol (default: SPY)",
    )
    grs_parser.add_argument(
        "--n-obs",
        type=int,
        default=180,
        dest="n_obs",
        help="Time-series session count T (default: 180)",
    )
    grs_parser.add_argument(
        "--n-assets",
        type=int,
        default=4,
        dest="n_assets",
        help="Number of test portfolios N (default: 4)",
    )
    grs_parser.add_argument(
        "--n-factors",
        type=int,
        default=1,
        dest="n_factors",
        help="Number of benchmark factors K (default: 1)",
    )
    grs_parser.add_argument(
        "--alpha-ann",
        type=float,
        default=0.0,
        dest="alpha_ann",
        help="Annualized joint pricing alpha injected into synthetic test assets (default: 0.0)",
    )
    grs_parser.add_argument(
        "--idio-vol",
        type=float,
        default=0.10,
        dest="idio_vol",
        help="Annualized idiosyncratic residual volatility (default: 0.10)",
    )
    grs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output GRS multivariate efficiency receipt as JSON",
    )

    # Jobson & Korkie (1981) / Memmel (2003) Paired Sharpe Equality Test parser
    jkm_parser = subparsers.add_parser(
        "sharpe-test",
        aliases=["jobson-korkie", "memmel", "jkm-2003"],
        help="Run correlation-adjusted Jobson-Korkie-Memmel paired Sharpe equality test",
    )
    jkm_parser.add_argument(
        "--ticker-a",
        default="STRAT",
        dest="ticker_a",
        help="Strategy A ticker label (default: STRAT)",
    )
    jkm_parser.add_argument(
        "--ticker-b",
        default="SPY",
        dest="ticker_b",
        help="Benchmark B ticker label (default: SPY)",
    )
    jkm_parser.add_argument(
        "--sr-a",
        type=float,
        default=1.40,
        dest="sr_a",
        help="Annualized Sharpe ratio of Stream A (default: 1.40)",
    )
    jkm_parser.add_argument(
        "--sr-b",
        type=float,
        default=0.70,
        dest="sr_b",
        help="Annualized Sharpe ratio of Stream B (default: 0.70)",
    )
    jkm_parser.add_argument(
        "--rho",
        type=float,
        default=0.85,
        help="Cross-correlation between streams in [-0.99, 0.99] (default: 0.85)",
    )
    jkm_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annual risk-free benchmark rate (default: 0.04)",
    )
    jkm_parser.add_argument(
        "--n-obs",
        type=int,
        default=252,
        dest="n_obs",
        help="Number of paired return sessions T (default: 252)",
    )
    jkm_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Jobson-Korkie-Memmel Sharpe equality receipt as JSON",
    )

    # Hurst (1951) & Peters (1994) Rescaled Range (R/S) parser
    hurst_parser = subparsers.add_parser(
        "hurst-exponent",
        aliases=["fractal-market", "rs-1951"],
        help="Compute Anis-Lloyd bias-corrected R/S Hurst exponent & fractal dimension",
    )
    hurst_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Asset or index ticker symbol (default: SPY)",
    )
    hurst_parser.add_argument(
        "--hurst-target",
        type=float,
        default=0.50,
        dest="hurst_target",
        help="Synthetic fBm target Hurst exponent in (0.05, 0.95) (default: 0.50)",
    )
    hurst_parser.add_argument(
        "--min-window",
        type=int,
        default=8,
        dest="min_window",
        help="Minimum non-overlapping R/S block size in [4, 64] (default: 8)",
    )
    hurst_parser.add_argument(
        "--n-obs",
        type=int,
        default=320,
        dest="n_obs",
        help="Number of synthetic fBm sessions (default: 320)",
    )
    hurst_parser.add_argument(
        "--vol-daily",
        type=float,
        default=0.012,
        dest="vol_daily",
        help="Daily return volatility scale (default: 0.012)",
    )
    hurst_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hurst R/S fractal persistence receipt as JSON",
    )

    # Choueifaty & Coignard (2008) Maximum Diversification Portfolio parser
    dr_parser = subparsers.add_parser(
        "max-diversification",
        aliases=["choueifaty-dr", "cc-2008"],
        help="Solve Choueifaty & Coignard Maximum Diversification Ratio DR(w) portfolio",
    )
    dr_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Portfolio or benchmark ticker symbol (default: SPY)",
    )
    dr_parser.add_argument(
        "--n-assets",
        type=int,
        default=5,
        dest="n_assets",
        help="Number of sleeve assets N in [2, 50] (default: 5)",
    )
    dr_parser.add_argument(
        "--corr",
        type=float,
        default=0.30,
        help="Synthetic uniform pairwise asset correlation (default: 0.30)",
    )
    dr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Maximum Diversification receipt as JSON",
    )

    # Brinson, Hood & Beebower (1986) / Brinson & Fachler (1985) Attribution parser
    bhb_parser = subparsers.add_parser(
        "brinson-attribution",
        aliases=["bhb-1986", "brinson-fachler"],
        help="Decompose active return into Allocation, Selection, and Interaction effects",
    )
    bhb_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Benchmark or portfolio sleeve ticker label (default: SPY)",
    )
    bhb_parser.add_argument(
        "--active-alpha",
        type=float,
        default=0.02,
        dest="active_alpha",
        help="Lead sleeve active selection spread (default: 0.02)",
    )
    bhb_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Brinson attribution receipt as JSON",
    )

    # Copeland & Galai (1983) Option-Theoretic Bid-Ask Spread parser
    cg_parser = subparsers.add_parser(
        "copeland-galai",
        aliases=["option-spread", "cg-1983"],
        help="Solve Copeland & Galai option-theoretic dealer bid-ask spread equilibrium",
    )
    cg_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Asset ticker symbol (default: SPY)",
    )
    cg_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Current spot asset price P_0 (default: 100.0)",
    )
    cg_parser.add_argument(
        "--vol",
        type=float,
        default=0.25,
        help="Annualized fundamental volatility sigma (default: 0.25)",
    )
    cg_parser.add_argument(
        "--informed-prob",
        type=float,
        default=0.15,
        dest="informed_prob",
        help="Probability of informed trader arrival pi_I in [0, 0.85) (default: 0.15)",
    )
    cg_parser.add_argument(
        "--quote-days",
        type=float,
        default=1.0,
        dest="quote_days",
        help="Dealer quote-life horizon in trading days (default: 1.0)",
    )
    cg_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Copeland & Galai spread equilibrium receipt as JSON",
    )

    # Maillard, Roncalli & Teiletche (2010) / Spinu (2013) ERC Risk Parity parser
    erc_parser = subparsers.add_parser(
        "erc-portfolio",
        aliases=["risk-budget", "mrt-2010"],
        help="Solve convex Equal Risk Contribution (ERC) / Risk-Budgeting portfolio",
    )
    erc_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Benchmark or portfolio sleeve ticker symbol (default: SPY)",
    )
    erc_parser.add_argument(
        "--n-assets",
        type=int,
        default=4,
        dest="n_assets",
        help="Number of sleeve assets N >= 2 (default: 4)",
    )
    erc_parser.add_argument(
        "--corr",
        type=float,
        default=0.25,
        help="Synthetic uniform pairwise asset correlation (default: 0.25)",
    )
    erc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output ERC Risk Parity receipt as JSON",
    )

    var_bt_parser = subparsers.add_parser(
        "var-backtest",
        aliases=["kupiec-pof", "christoffersen-cc"],
        help="Run Kupiec (1995) & Christoffersen (1998) Conditional Coverage VaR Backtest",
    )
    var_bt_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Ticker symbol to tag VaR backtest receipt (default: SPY)",
    )
    var_bt_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Target VaR tail significance level in (0, 0.5) (default: 0.05)",
    )
    var_bt_parser.add_argument(
        "--n-obs",
        type=int,
        default=250,
        help="Number of synthetic daily observations when running CLI test series (default: 250)",
    )
    var_bt_parser.add_argument(
        "--mode",
        choices=["calibrated", "excessive", "clustered"],
        default="calibrated",
        help="Synthetic backtest regime mode (default: calibrated)",
    )
    var_bt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kupiec-Christoffersen VaR backtest receipt as JSON",
    )

    bdt_parser = subparsers.add_parser(
        "bdt-lattice",
        aliases=["black-derman-toy", "bdt-1990", "rate-lattice"],
        help="Calibrate Black-Derman-Toy (1990) Lognormal Short-Rate Binomial Lattice",
    )
    bdt_parser.add_argument(
        "--ticker",
        default="UST",
        help="Curve/instrument ticker symbol (default: UST)",
    )
    bdt_parser.add_argument(
        "--steps",
        type=int,
        default=8,
        help="Number of binomial maturity steps (default: 8)",
    )
    bdt_parser.add_argument(
        "--base-rate",
        type=float,
        default=0.04,
        help="Front-end annualized spot rate (default: 0.04)",
    )
    bdt_parser.add_argument(
        "--slope-bps",
        type=float,
        default=120.0,
        help="Yield curve slope from short to long end in bps (default: 120.0)",
    )
    bdt_parser.add_argument(
        "--vol",
        type=float,
        default=0.16,
        help="Annualized lognormal short-rate volatility (default: 0.16)",
    )
    bdt_parser.add_argument(
        "--call-price",
        type=float,
        default=100.0,
        help="Bermudan call strike price (default: 100.0)",
    )
    bdt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Black-Derman-Toy lattice receipt as JSON",
    )

    nss_parser = subparsers.add_parser(
        "nss-curve",
        aliases=["nelson-siegel", "svensson", "nss-1994", "yield-curve-fit"],
        help="Fit Nelson-Siegel-Svensson (1987/1994) Parametric Yield Curve Factors",
    )
    nss_parser.add_argument(
        "--ticker",
        default="UST",
        help="Curve ticker symbol (default: UST)",
    )
    nss_parser.add_argument(
        "--mode",
        choices=["normal", "humped", "inverted", "flat"],
        default="normal",
        help="Synthetic yield curve shape mode (default: normal)",
    )
    nss_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Nelson-Siegel-Svensson receipt as JSON",
    )

    gp_parser = subparsers.add_parser(
        "gp-trading",
        aliases=["gp-dynamic", "gp-2013", "dynamic-aim"],
        help="Solve Gârleanu & Pedersen (2013) Dynamic Trading Aim Portfolio Policy",
    )
    gp_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Portfolio/benchmark ticker symbol (default: SPY)",
    )
    gp_parser.add_argument(
        "--risk-aversion",
        type=float,
        default=2.5,
        help="Mean-variance risk aversion parameter gamma (default: 2.5)",
    )
    gp_parser.add_argument(
        "--cost-bps",
        type=float,
        default=15.0,
        help="Quadratic transaction cost scaling in bps (default: 15.0)",
    )
    gp_parser.add_argument(
        "--fast-rho",
        type=float,
        default=0.35,
        help="Fast alpha mean-reversion decay speed in (0, 1) (default: 0.35)",
    )
    gp_parser.add_argument(
        "--slow-rho",
        type=float,
        default=0.03,
        help="Slow alpha mean-reversion decay speed in (0, 1) (default: 0.03)",
    )
    gp_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Gârleanu-Pedersen dynamic trading receipt as JSON",
    )

    dcc_parser = subparsers.add_parser(
        "dcc-garch",
        aliases=["dcc-2002", "dynamic-correlation", "engle-dcc"],
        help="Estimate Engle (2002) Dynamic Conditional Correlation (DCC-GARCH(1,1))",
    )
    dcc_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Portfolio/pair ticker symbol (default: SPY)",
    )
    dcc_parser.add_argument(
        "--n-obs",
        type=int,
        default=220,
        help="Number of synthetic return observations (default: 220)",
    )
    dcc_parser.add_argument(
        "--regime",
        choices=["crisis_spike", "calm", "low_corr"],
        default="crisis_spike",
        help="Synthetic correlation regime mode (default: crisis_spike)",
    )
    dcc_parser.add_argument(
        "--json",
        action="store_true",
        help="Output DCC-GARCH(1,1) receipt as JSON",
    )

    hlz_parser = subparsers.add_parser(
        "haircut-sharpe",
        aliases=["harvey-liu", "hlz-2016", "bh-fdr-sharpe"],
        help="Compute Harvey & Liu (2015) Multiple-Testing Haircut Sharpe Ratio",
    )
    hlz_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Strategy/asset ticker symbol (default: SPY)",
    )
    hlz_parser.add_argument(
        "--sharpe",
        type=float,
        default=1.85,
        help="Observed annualized Sharpe ratio (default: 1.85)",
    )
    hlz_parser.add_argument(
        "--n-obs",
        type=int,
        default=1260,
        help="Number of daily observations (default: 1260)",
    )
    hlz_parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of backtested strategy trials M (default: 100)",
    )
    hlz_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Harvey-Liu Haircut Sharpe receipt as JSON",
    )

    lo_sr_parser = subparsers.add_parser(
        "lo-sharpe",
        aliases=["lo-2002", "autocorr-sharpe", "lo-adjusted-sr"],
        help="Compute Andrew Lo (2002) Autocorrelation-Adjusted & Asymptotic Sharpe Ratio",
    )
    lo_sr_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Strategy/asset ticker symbol (default: SPY)",
    )
    lo_sr_parser.add_argument(
        "--q",
        type=int,
        default=12,
        help="Time-aggregation horizon q (default: 12)",
    )
    lo_sr_parser.add_argument(
        "--rho1",
        type=float,
        default=0.35,
        help="Synthetic AR(1) lag-1 autocorrelation in (-0.95, 0.95) (default: 0.35)",
    )
    lo_sr_parser.add_argument(
        "--n-obs",
        type=int,
        default=300,
        help="Number of synthetic observations (default: 300)",
    )
    lo_sr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Andrew Lo (2002) Sharpe receipt as JSON",
    )

    krd_parser = subparsers.add_parser(
        "key-rate-duration",
        aliases=["krd-1992", "fisher-weil", "immunization"],
        help="Compute Ho (1992) Key-Rate Duration & Fisher-Weil (1971) Liability Immunization",
    )
    krd_parser.add_argument(
        "--ticker",
        default="ALM",
        help="ALM fund/portfolio ticker symbol (default: ALM)",
    )
    krd_parser.add_argument(
        "--liab-maturity",
        type=float,
        default=10.0,
        help="Target liability bullet maturity in years (default: 10.0)",
    )
    krd_parser.add_argument(
        "--mode",
        choices=["barbell", "ladder", "bullet"],
        default="ladder",
        help="Hedging bond ladder structure (default: ladder)",
    )
    krd_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Ho (1992) KRD immunization receipt as JSON",
    )

    roy_parser = subparsers.add_parser(
        "safety-first",
        aliases=["roy-1952", "kataoka", "telser-1955"],
        help="Solve Roy (1952), Kataoka (1963) & Telser (1955) Safety-First Portfolios",
    )
    roy_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Portfolio/endowment ticker symbol (default: SPY)",
    )
    roy_parser.add_argument(
        "--threshold",
        type=float,
        default=-0.10,
        help="Minimum disaster threshold return R_L (default: -0.10)",
    )
    roy_parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Maximum shortfall probability alpha in (0, 0.5) (default: 0.05)",
    )
    roy_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Roy Safety-First receipt as JSON",
    )

    merton_crra_parser = subparsers.add_parser(
        "merton-share",
        aliases=["merton-1969", "samuelson-merton", "crra-allocation"],
        help="Compute Samuelson (1969) & Merton (1969/1971) CRRA Lifetime Portfolio & Consumption",
    )
    merton_crra_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Strategy/portfolio ticker symbol (default: SPY)",
    )
    merton_crra_parser.add_argument(
        "--mu",
        type=float,
        default=0.09,
        help="Annualized expected return (default: 0.09)",
    )
    merton_crra_parser.add_argument(
        "--vol",
        type=float,
        default=0.16,
        help="Annualized volatility (default: 0.16)",
    )
    merton_crra_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    merton_crra_parser.add_argument(
        "--gamma",
        type=float,
        default=3.0,
        help="CRRA relative risk aversion parameter (default: 3.0)",
    )
    merton_crra_parser.add_argument(
        "--rho",
        type=float,
        default=0.03,
        help="Subjective time discount rate (default: 0.03)",
    )
    merton_crra_parser.add_argument(
        "--rho-state",
        type=float,
        default=-0.40,
        help="Correlation between returns and state variable (default: -0.40)",
    )
    merton_crra_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Merton CRRA allocation receipt as JSON",
    )

    var_swap_parser = subparsers.add_parser(
        "variance-swap",
        aliases=["carr-madan", "cm-1999", "var-swap"],
        help="Compute Carr & Madan (1999) Static Replication Variance Swap Fair Strike",
    )
    var_swap_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Option/asset ticker symbol (default: SPY)",
    )
    var_swap_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Underlying spot price (default: 100.0)",
    )
    var_swap_parser.add_argument(
        "--vol",
        type=float,
        default=0.20,
        help="ATM implied volatility (default: 0.20)",
    )
    var_swap_parser.add_argument(
        "--skew",
        type=float,
        default=-0.15,
        help="Implied volatility skew slope (default: -0.15)",
    )
    var_swap_parser.add_argument(
        "--realized-vol",
        type=float,
        default=0.17,
        help="Expected or historical realized volatility (default: 0.17)",
    )
    var_swap_parser.add_argument(
        "--ttm",
        type=float,
        default=0.25,
        help="Time to swap maturity in years (default: 0.25)",
    )
    var_swap_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    var_swap_parser.add_argument(
        "--n-strikes",
        type=int,
        default=200,
        help="Number of discrete numerical quadrature strikes (default: 200)",
    )
    var_swap_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Carr-Madan variance swap receipt as JSON",
    )

    kou_jump_parser = subparsers.add_parser(
        "kou-jump",
        aliases=["kou-2002", "double-exponential", "kou-model"],
        help="Price options under Steven G. Kou (2002) Double Exponential Jump-Diffusion",
    )
    kou_jump_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Option/asset ticker symbol (default: SPY)",
    )
    kou_jump_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Underlying spot price (default: 100.0)",
    )
    kou_jump_parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Option strike price (default: 100.0)",
    )
    kou_jump_parser.add_argument(
        "--vol",
        type=float,
        default=0.16,
        help="Continuous diffusion volatility sigma (default: 0.16)",
    )
    kou_jump_parser.add_argument(
        "--lambda-jump",
        type=float,
        default=0.75,
        help="Annual Poisson jump arrival intensity lambda (default: 0.75)",
    )
    kou_jump_parser.add_argument(
        "--p-up",
        type=float,
        default=0.35,
        help="Probability of upward jump p in [0, 1] (default: 0.35)",
    )
    kou_jump_parser.add_argument(
        "--eta1",
        type=float,
        default=10.0,
        help="Upward jump exponential decay rate eta1 > 1 (default: 10.0)",
    )
    kou_jump_parser.add_argument(
        "--eta2",
        type=float,
        default=5.0,
        help="Downward jump exponential decay rate eta2 > 0 (default: 5.0)",
    )
    kou_jump_parser.add_argument(
        "--ttm",
        type=float,
        default=0.50,
        help="Time to maturity in years (default: 0.50)",
    )
    kou_jump_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    kou_jump_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Kou jump-diffusion receipt as JSON",
    )

    sabr_vol_parser = subparsers.add_parser(
        "sabr-vol",
        aliases=["sabr", "sabr-2002", "hagan-sabr"],
        help="Evaluate Patrick S. Hagan et al. (2002) SABR Stochastic Volatility Smile",
    )
    sabr_vol_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    sabr_vol_parser.add_argument(
        "--forward",
        type=float,
        default=100.0,
        help="Forward price F_0 (default: 100.0)",
    )
    sabr_vol_parser.add_argument(
        "--alpha",
        type=float,
        default=0.20,
        help="Initial volatility level alpha > 0 (default: 0.20)",
    )
    sabr_vol_parser.add_argument(
        "--beta",
        type=float,
        default=0.50,
        help="CEV elasticity parameter beta in [0, 1] (default: 0.50)",
    )
    sabr_vol_parser.add_argument(
        "--rho",
        type=float,
        default=-0.30,
        help="Correlation between asset and volatility rho in (-1, 1) (default: -0.30)",
    )
    sabr_vol_parser.add_argument(
        "--nu",
        type=float,
        default=0.40,
        help="Volatility of volatility nu >= 0 (default: 0.40)",
    )
    sabr_vol_parser.add_argument(
        "--ttm",
        type=float,
        default=0.50,
        help="Time to option maturity in years (default: 0.50)",
    )
    sabr_vol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output SABR volatility smile receipt as JSON",
    )

    dupire_vol_parser = subparsers.add_parser(
        "dupire-vol",
        aliases=["dupire", "dupire-1994", "local-vol"],
        help="Compute Bruno Dupire (1994) Non-Parametric Local Volatility Surface",
    )
    dupire_vol_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    dupire_vol_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Underlying spot price (default: 100.0)",
    )
    dupire_vol_parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Evaluation strike price (default: 100.0)",
    )
    dupire_vol_parser.add_argument(
        "--vol",
        type=float,
        default=0.20,
        help="ATM implied volatility level (default: 0.20)",
    )
    dupire_vol_parser.add_argument(
        "--skew",
        type=float,
        default=-0.15,
        help="Implied volatility skew slope (default: -0.15)",
    )
    dupire_vol_parser.add_argument(
        "--vol-of-vol",
        type=float,
        default=0.25,
        help="Implied volatility smile curvature (default: 0.25)",
    )
    dupire_vol_parser.add_argument(
        "--ttm",
        type=float,
        default=0.50,
        help="Time to expiration in years (default: 0.50)",
    )
    dupire_vol_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    dupire_vol_parser.add_argument(
        "--div",
        type=float,
        default=0.0,
        help="Continuous dividend yield (default: 0.0)",
    )
    dupire_vol_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Dupire local volatility receipt as JSON",
    )

    bates_opt_parser = subparsers.add_parser(
        "bates-opt",
        aliases=["bates", "bates-1996", "svj-model"],
        help="Price options under David S. Bates (1996) Stochastic Volatility Jump-Diffusion",
    )
    bates_opt_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    bates_opt_parser.add_argument(
        "--spot",
        type=float,
        default=100.0,
        help="Underlying spot price (default: 100.0)",
    )
    bates_opt_parser.add_argument(
        "--strike",
        type=float,
        default=100.0,
        help="Evaluation strike price (default: 100.0)",
    )
    bates_opt_parser.add_argument(
        "--v0",
        type=float,
        default=0.04,
        help="Initial stochastic variance V0 (default: 0.04)",
    )
    bates_opt_parser.add_argument(
        "--kappa",
        type=float,
        default=2.0,
        help="Mean-reversion rate kappa (default: 2.0)",
    )
    bates_opt_parser.add_argument(
        "--theta",
        type=float,
        default=0.04,
        help="Long-term variance theta (default: 0.04)",
    )
    bates_opt_parser.add_argument(
        "--sigma-v",
        type=float,
        default=0.30,
        help="Volatility of variance sigma_v (default: 0.30)",
    )
    bates_opt_parser.add_argument(
        "--rho",
        type=float,
        default=-0.50,
        help="Correlation between asset and variance rho in (-1, 1) (default: -0.50)",
    )
    bates_opt_parser.add_argument(
        "--lambda-jump",
        type=float,
        default=0.50,
        help="Annual Poisson jump arrival rate lambda (default: 0.50)",
    )
    bates_opt_parser.add_argument(
        "--mu-jump",
        type=float,
        default=-0.05,
        help="Mean log-jump size mu_J (default: -0.05)",
    )
    bates_opt_parser.add_argument(
        "--sigma-jump",
        type=float,
        default=0.15,
        help="Standard deviation of log-jump size sigma_J (default: 0.15)",
    )
    bates_opt_parser.add_argument(
        "--ttm",
        type=float,
        default=0.50,
        help="Time to expiration in years (default: 0.50)",
    )
    bates_opt_parser.add_argument(
        "--rf",
        type=float,
        default=0.04,
        help="Annualized risk-free rate (default: 0.04)",
    )
    bates_opt_parser.add_argument(
        "--div",
        type=float,
        default=0.0,
        help="Continuous dividend yield (default: 0.0)",
    )
    bates_opt_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Bates SVJ receipt as JSON",
    )

    cgarch_parser = subparsers.add_parser(
        "cgarch",
        aliases=["c-garch", "component-garch", "engle-lee"],
        help="Evaluate Robert F. Engle & Gary G.J. Lee (1999) Component GARCH Two-Scale Volatility",
    )
    cgarch_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    cgarch_parser.add_argument(
        "--omega",
        type=float,
        default=0.0001,
        help="Baseline permanent baseline variance omega (default: 0.0001)",
    )
    cgarch_parser.add_argument(
        "--rho",
        type=float,
        default=0.98,
        help="Permanent component persistence rho in (0, 1) (default: 0.98)",
    )
    cgarch_parser.add_argument(
        "--phi",
        type=float,
        default=0.04,
        help="Permanent component shock sensitivity phi >= 0 (default: 0.04)",
    )
    cgarch_parser.add_argument(
        "--alpha",
        type=float,
        default=0.06,
        help="Transitory component shock reaction alpha >= 0 (default: 0.06)",
    )
    cgarch_parser.add_argument(
        "--beta",
        type=float,
        default=0.70,
        help="Transitory component decay persistence beta (default: 0.70)",
    )
    cgarch_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Component GARCH receipt as JSON",
    )

    huang_stoll_parser = subparsers.add_parser(
        "huang-stoll",
        aliases=["hs-1997", "spread-decomp", "three-way-spread"],
        help="Estimate Roger D. Huang & Hans R. Stoll (1997) Three-Way Bid-Ask Spread Decomposition",
    )
    huang_stoll_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    huang_stoll_parser.add_argument(
        "--spread",
        type=float,
        default=10.0,
        help="Synthetic effective spread in basis points (default: 10.0 bps)",
    )
    huang_stoll_parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Target adverse selection proportion alpha (default: 0.35)",
    )
    huang_stoll_parser.add_argument(
        "--beta",
        type=float,
        default=0.20,
        help="Target inventory holding proportion beta (default: 0.20)",
    )
    huang_stoll_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Huang-Stoll spread receipt as JSON",
    )

    ekop_pin_parser = subparsers.add_parser(
        "ekop-pin",
        aliases=["ekop", "pin-model", "informed-trading"],
        help="Estimate David Easley et al. (1996) Probability of Informed Trading (PIN)",
    )
    ekop_pin_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    ekop_pin_parser.add_argument(
        "--alpha",
        type=float,
        default=0.30,
        help="Information event probability alpha in [0, 1] (default: 0.30)",
    )
    ekop_pin_parser.add_argument(
        "--delta",
        type=float,
        default=0.40,
        help="Bad news conditional probability delta in [0, 1] (default: 0.40)",
    )
    ekop_pin_parser.add_argument(
        "--mu",
        type=float,
        default=120.0,
        help="Informed trader Poisson arrival intensity mu > 0 (default: 120.0)",
    )
    ekop_pin_parser.add_argument(
        "--epsilon",
        type=float,
        default=80.0,
        help="Uninformed baseline Poisson arrival intensity epsilon > 0 (default: 80.0)",
    )
    ekop_pin_parser.add_argument(
        "--json",
        action="store_true",
        help="Output EKOP PIN receipt as JSON",
    )

    mrr_parser = subparsers.add_parser(
        "mrr-model",
        aliases=["mrr", "mrr-1997", "order-flow-impact"],
        help="Estimate Madhavan, Richardson & Roomans (1997) MRR Microstructure Model",
    )
    mrr_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    mrr_parser.add_argument(
        "--spread",
        type=float,
        default=12.0,
        help="Synthetic effective spread in basis points (default: 12.0 bps)",
    )
    mrr_parser.add_argument(
        "--theta-ratio",
        type=float,
        default=0.40,
        help="Target adverse selection fraction theta / (theta + phi0) (default: 0.40)",
    )
    mrr_parser.add_argument(
        "--autocorr",
        type=float,
        default=0.25,
        help="Target order flow autocorrelation phi in (-1, 1) (default: 0.25)",
    )
    mrr_parser.add_argument(
        "--json",
        action="store_true",
        help="Output MRR microstructure receipt as JSON",
    )

    gibbs_parser = subparsers.add_parser(
        "hasbrouck-gibbs",
        aliases=["gibbs-spread", "hasbrouck-2009", "mcmc-spread"],
        help="Estimate Joel Hasbrouck (2009) Bayesian Gibbs Sampler Effective Spread",
    )
    gibbs_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    gibbs_parser.add_argument(
        "--spread",
        type=float,
        default=15.0,
        help="Synthetic effective spread in basis points when fallback needed (default: 15.0 bps)",
    )
    gibbs_parser.add_argument(
        "--iterations",
        type=int,
        default=500,
        help="Total MCMC iterations (default: 500)",
    )
    gibbs_parser.add_argument(
        "--burn-in",
        type=int,
        default=120,
        help="Number of burn-in iterations to discard (default: 120)",
    )
    gibbs_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for MCMC reproducibility (default: 42)",
    )
    gibbs_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Hasbrouck Gibbs receipt as JSON",
    )

    gkn_parser = subparsers.add_parser(
        "gkn-spread",
        aliases=["gkn", "gkn-1991", "spread-purged"],
        help="Estimate George, Kaul & Nimalendran (1991) GKN Spread Decomposition",
    )
    gkn_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    gkn_parser.add_argument(
        "--spread",
        type=float,
        default=18.0,
        help="Synthetic effective spread in basis points when fallback needed (default: 18.0 bps)",
    )
    gkn_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic fallback generation (default: 42)",
    )
    gkn_parser.add_argument(
        "--json",
        action="store_true",
        help="Output GKN spread receipt as JSON",
    )

    gh_parser = subparsers.add_parser(
        "glosten-harris",
        aliases=["gh-1988", "size-spread", "volume-impact-spread"],
        help="Estimate Glosten & Harris (1988) Trade-Size Dependent Spread Decomposition",
    )
    gh_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    gh_parser.add_argument(
        "--benchmark-volume",
        type=float,
        default=1.0,
        help="Benchmark order size/volume for effective spread evaluation (default: 1.0)",
    )
    gh_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fallback generation (default: 42)",
    )
    gh_parser.add_argument(
        "--json",
        action="store_true",
        help="Output Glosten-Harris spread receipt as JSON",
    )

    lot_parser = subparsers.add_parser(
        "lot-spread",
        aliases=["lot", "lot-1999", "zero-return-spread", "ldv-spread"],
        help="Estimate Lesmond, Ogden & Trzcinka (1999) LOT Zero-Return Transaction Costs",
    )
    lot_parser.add_argument(
        "--ticker",
        default="SPY",
        help="Underlying asset ticker symbol (default: SPY)",
    )
    lot_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for synthetic fallback generation (default: 42)",
    )
    lot_parser.add_argument(
        "--json",
        action="store_true",
        help="Output LOT spread receipt as JSON",
    )

    args = parser.parse_args()

    if args.command == "sync":
        sync_command(args)
    elif args.command == "status":
        status_command(args)
    elif args.command == "analyze":
        analyze_command(args)
    elif args.command == "forecast-baseline":
        forecast_baseline_command(args)
    elif args.command == "train-linear":
        train_linear_command(args)
    elif args.command == "train-all-horizons":
        train_all_horizons_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "evaluate-pending":
        evaluate_pending_command(args)
    elif args.command == "list-predictions":
        list_predictions_command(args)
    elif args.command == "monitor":
        monitor_command(args)
    elif args.command == "session-status":
        session_status_command(args)
    elif args.command == "daemon-reconcile":
        daemon_reconcile_command(args)
    elif args.command == "verify-audit":
        verify_audit_command(args)
    elif args.command == "backup-db":
        backup_db_command(args)
    elif args.command == "risk-report":
        risk_report_command(args)
    elif args.command == "prune-db":
        prune_db_command(args)
    elif args.command == "vacuum-db":
        vacuum_db_command(args)
    elif args.command == "backtest":
        backtest_command(args)
    elif args.command == "system-health":
        system_health_command(args)
    elif args.command == "screen":
        screen_command(args)
    elif args.command == "ic-drift":
        ic_drift_command(args)
    elif args.command == "portfolio":
        portfolio_command(args)
    elif args.command == "calibration-report":
        calibration_report_command(args)
    elif args.command == "lead-lag":
        lead_lag_command(args)
    elif args.command == "cointegrate":
        cointegrate_command(args)
    elif args.command == "liquidity-report":
        liquidity_report_command(args)
    elif args.command == "volatility-report":
        volatility_report_command(args)
    elif args.command == "crps-eval":
        crps_eval_command(args)
    elif args.command == "changepoint-detect":
        changepoint_detect_command(args)
    elif args.command == "toxicity-report":
        toxicity_report_command(args)
    elif args.command == "tail-dependence":
        tail_dependence_command(args)
    elif args.command == "factor-attribution":
        factor_attribution_command(args)
    elif args.command == "wavelet-denoise":
        wavelet_denoise_command(args)
    elif args.command == "momentum-crash-report":
        momentum_crash_report_command(args)
    elif args.command == "fracdiff":
        fracdiff_command(args)
    elif args.command == "kalman-spread":
        kalman_spread_command(args)
    elif args.command == "evt-report":
        evt_report_command(args)
    elif args.command == "entropy-report":
        entropy_report_command(args)
    elif args.command == "garch-volatility":
        garch_volatility_command(args)
    elif args.command == "resampled-frontier":
        resampled_frontier_command(args)
    elif args.command == "mrmr-select":
        mrmr_select_command(args)
    elif args.command == "price-impact":
        price_impact_command(args)
    elif args.command == "tail-shape":
        tail_shape_command(args)
    elif args.command == "pain-index":
        pain_index_command(args)
    elif args.command == "markov-regime":
        markov_regime_command(args)
    elif args.command == "rmt-denoise":
        rmt_denoise_command(args)
    elif args.command == "jump-test":
        jump_test_command(args)
    elif args.command == "deflated-sharpe":
        deflated_sharpe_command(args)
    elif args.command == "effective-spread":
        effective_spread_command(args)
    elif args.command == "cpcv-validate":
        cpcv_validate_command(args)
    elif args.command == "downside-beta":
        downside_beta_command(args)
    elif args.command == "optimal-execution":
        optimal_execution_command(args)
    elif args.command == "fama-french":
        fama_french_command(args)
    elif args.command == "black-litterman":
        black_litterman_command(args)
    elif args.command == "idiosyncratic-vol":
        idiosyncratic_vol_command(args)
    elif args.command == "liquidity-beta":
        liquidity_beta_command(args)
    elif args.command == "fama-macbeth":
        fama_macbeth_command(args)
    elif args.command == "higher-moments":
        higher_moments_command(args)
    elif args.command == "merton-credit":
        merton_credit_command(args)
    elif args.command == "hasbrouck-var":
        hasbrouck_var_command(args)
    elif args.command == "vrp":
        vrp_command(args)
    elif args.command == "max-lottery":
        max_lottery_command(args)
    elif args.command == "profitability":
        profitability_command(args)
    elif args.command == "dual-momentum":
        dual_momentum_command(args)
    elif args.command == "bab":
        bab_command(args)
    elif args.command == "reversal":
        reversal_command(args)
    elif args.command == "distress":
        distress_command(args)
    elif args.command == "sentiment":
        sentiment_command(args)
    elif args.command in ("arbitrage-asymmetry", "asymmetry", "short-impediment"):
        arbitrage_asymmetry_command(args)
    elif args.command in ("q-factor", "qfactor", "hou-xue-zhang"):
        q_factor_command(args)
    elif args.command in ("accruals", "accrual", "earnings-quality", "sloan"):
        accruals_command(args)
    elif args.command in ("beneish", "mscore", "forensic-audit", "fraud-risk"):
        beneish_command(args)
    elif args.command in ("fscore", "piotroski", "fundamental-health", "value-quality"):
        fscore_command(args)
    elif args.command in ("diffusion", "hong-stein", "information-diffusion", "momentum-regime"):
        diffusion_command(args)
    elif args.command in ("contrarian", "lsv", "extrapolation-bias", "glamour-overpricing"):
        contrarian_command(args)
    elif args.command in ("dhs", "overconfidence", "self-attribution", "behavioral-lifecycle"):
        dhs_command(args)
    elif args.command in ("bsv", "barberis", "conservatism-representativeness", "dual-heuristic"):
        bsv_command(args)
    elif args.command in ("innovation", "rd-anomaly", "knowledge-capital", "rnd-intensity"):
        innovation_command(args)
    elif args.command in ("lifecycle", "lee-swaminathan", "momentum-volume", "mlc-stage"):
        lifecycle_command(args)
    elif args.command in ("altman", "zscore", "ohlson", "distress-risk"):
        altman_command(args)
    elif args.command in ("vme", "value-momentum", "asness-moskowitz", "vme-everywhere"):
        vme_command(args)
    elif args.command in ("pead", "sue", "bernard-thomas", "earnings-drift"):
        pead_command(args)
    elif args.command in ("empire", "asset-growth", "cooper-gulen", "titman-capex"):
        empire_command(args)
    elif args.command in ("payout", "net-payout", "boudoukh", "share-issuance"):
        payout_command(args)
    elif args.command in ("noa", "bloat", "hirshleifer-teoh", "balance-sheet-bloat"):
        noa_command(args)
    elif args.command in ("dispersion", "diether-malloy", "miller-opinion", "forecast-dispersion"):
        dispersion_command(args)
    elif args.command in ("catering", "polk-sapienza", "catering-capex", "investment-catering"):
        catering_command(args)
    elif args.command in ("commonality", "chordia-roll", "liquidity-commonality", "order-imbalance"):
        commonality_command(args)
    elif args.command in ("supply-chain", "cohen-frazzini", "customer-momentum", "economic-links"):
        supply_chain_command(args)
    elif args.command in ("governance", "gompers-ishii", "bebchuk-entrenchment", "g-index", "e-index"):
        governance_command(args)
    elif args.command in ("fip", "frog-in-the-pan", "da-gurun", "information-discreteness"):
        fip_command(args)
    elif args.command in ("attention", "barber-odean", "da-engelberg", "svi-attention", "attention-spike"):
        attention_command(args)
    elif args.command in ("benchmark-mandate", "baker-bradley", "tracking-error-limit", "low-vol-mandate"):
        benchmark_mandate_command(args)
    elif args.command in ("cgo", "grinblatt-han", "frazzini-disposition", "disposition-effect", "capital-gains-overhang"):
        cgo_command(args)
    elif args.command in ("high52", "george-hwang", "52-week-high", "anchoring-momentum"):
        high52_command(args)
    elif args.command in ("pastor-veronesi", "learning-uncertainty", "profitability-learning", "uncertainty-convexity"):
        pastor_veronesi_command(args)
    elif args.command in ("lrr", "bansal-yaron", "long-run-risks", "epstein-zin"):
        lrr_command(args)
    elif args.command in ("reversibility", "lu-zhang", "costly-reversibility", "asymmetric-capital"):
        reversibility_command(args)
    elif args.command in ("cochrane", "campbell-shiller", "discount-rate-decomposition", "present-value-identity"):
        cochrane_command(args)
    elif args.command in ("stub", "lamont-thaler", "stub-arbitrage", "equity-carveout", "negative-stub"):
        stub_command(args)
    elif args.command in ("liquidity-spiral", "brunnermeier-pedersen", "margin-spiral", "funding-liquidity", "fire-sale"):
        liquidity_spiral_command(args)
    elif args.command in ("intermediary", "he-kelly-manela", "dealer-capital", "intermediary-capital", "adrian-shin"):
        intermediary_command(args)
    elif args.command in ("margin-pricing", "garleanu-pedersen", "margin-capm", "basis-spread", "margin-constraints"):
        margin_pricing_command(args)
    elif args.command in ("ses", "acharya-pedersen", "systemic-risk", "mes", "capital-surcharge"):
        ses_command(args)
    elif args.command in ("covar", "adrian-brunnermeier", "delta-covar", "systemic-spillover", "tail-contagion"):
        covar_command(args)
    elif args.command in ("amh", "adaptive-markets", "andrew-lo", "market-efficiency", "punctuated-equilibrium"):
        amh_command(args)
    elif args.command in ("search-bargaining", "duffie-garleanu-pedersen", "dgp", "otc-illiquidity", "bargaining-spread"):
        search_bargaining_command(args)
    elif args.command in ("chs-distress", "campbell-hilscher-szilagyi", "distress-puzzle", "failure-probability", "chs"):
        chs_distress_command(args)
    elif args.command in ("grossman-stiglitz", "gs-equilibrium", "informational-efficiency", "info-rent", "grossman"):
        grossman_stiglitz_command(args)
    elif args.command in ("information-risk", "eho-risk", "cost-of-capital", "info-beta", "easley-ohara"):
        information_risk_command(args)
    elif args.command in ("admati-pfleiderer", "volume-clustering", "concentrated-trading", "intraday-liquidity", "ap-clustering"):
        admati_pfleiderer_command(args)
    elif args.command in ("noise-bands", "valuation-bounds", "noise-trading", "black-bounds"):
        noise_bands_command(args)
    elif args.command in ("bad-beta", "two-beta", "campbell-vuolteenaho", "icapm-two-beta"):
        bad_beta_command(args)
    elif args.command in ("merton-recognition", "incomplete-information", "neglected-firm", "investor-recognition", "merton-1987"):
        merton_recognition_command(args)
    elif args.command in ("rare-disasters", "gabaix-disasters", "tail-resilience", "variable-disasters", "disaster-risk"):
        rare_disasters_command(args)
    elif args.command in ("kyle-xiong", "wealth-contagion", "forced-liquidation", "spillover-beta", "kyle-2001"):
        kyle_xiong_command(args)
    elif args.command in ("predatory-trading", "predatory", "strategic-frontrun", "liquidation-overshoot", "brunnermeier-2005"):
        predatory_trading_command(args)
    elif args.command in ("hong-stein-crash", "crash-asymmetry", "latent-overhang", "short-constraint-crash", "hong-stein-2003"):
        hong_stein_crash_command(args)
    elif args.command in ("breeden-ccapm", "ccapm", "consumption-beta", "macro-consumption", "breeden-1979"):
        breeden_ccapm_command(args)
    elif args.command in ("bhs-prospect", "prospect-theory", "loss-aversion", "house-money", "barberis-2001"):
        bhs_prospect_command(args)
    elif args.command in ("allen-gale", "network-contagion", "incomplete-network", "liquidity-cascade", "allen-2000"):
        allen_gale_command(args)
    elif args.command in ("leverage-cycle", "geanakoplos", "collateral-haircut", "deleveraging-crisis", "margin-squeeze"):
        leverage_cycle_command(args)
    elif args.command in ("diamond-dybvig", "bank-run", "liquidity-transformation", "sequential-service", "redemption-panic"):
        diamond_dybvig_command(args)
    elif args.command in ("merton-icapm", "icapm", "intertemporal-capm", "state-hedging", "merton-1973"):
        merton_icapm_command(args)
    elif args.command in ("ross-apt", "apt", "arbitrage-pricing", "multi-factor-apt", "ross-1976"):
        ross_apt_command(args)
    elif args.command in ("dssw-noise", "noise-trader-risk", "limits-of-arbitrage", "resale-price-risk", "dssw-1990"):
        dssw_command(args)
    elif args.command in ("black-leverage", "leverage-effect", "asymmetric-volatility", "volatility-feedback", "black-1976"):
        black_leverage_command(args)
    elif args.command in ("lucas-tree", "lucas-1978", "exchange-economy", "pure-exchange", "lucas-sdf"):
        lucas_tree_command(args)
    elif args.command in ("harrison-kreps", "resale-option", "speculative-bubble", "hk-1979", "investor-disagreement"):
        harrison_kreps_command(args)
    elif args.command in ("brennan-agency", "agency-pricing", "brennan-1998", "two-beta-agency", "benchmark-distortion"):
        brennan_agency_command(args)
    elif args.command in ("shleifer-vishny", "shleifer-vishny-1997", "pba-arbitrage", "sv-1997", "performance-arbitrage"):
        shleifer_vishny_command(args)
    elif args.command in ("myers-majluf", "pecking-order", "equity-issuance", "lemons-discount", "myers-1984"):
        myers_majluf_command(args)
    elif args.command in ("breeden-litzenberger", "rnd-density", "arrow-debreu", "state-prices", "bl-1978"):
        breeden_litzenberger_command(args)
    elif args.command in ("aghion-bolton", "control-rights", "incomplete-contracts", "hart-moore", "strategic-default", "ab-1992"):
        aghion_bolton_command(args)
    elif args.command in ("leland-default", "leland-1994", "endogenous-default", "optimal-leverage", "tax-shield"):
        leland_command(args)
    elif args.command in ("hansen-jagannathan", "hj-bound", "hj-distance", "sdf-bound", "hj-1991"):
        hansen_jagannathan_command(args)
    elif args.command in ("jensen-meckling", "asset-substitution", "risk-shifting", "agency-cost", "jm-1976"):
        jensen_meckling_command(args)
    elif args.command in ("heston-vol", "heston-1993", "stochastic-vol", "heston-pricing", "feller-vol"):
        heston_vol_command(args)
    elif args.command in ("debt-overhang", "myers-1977", "growth-option", "real-option", "overhang-loss"):
        myers_overhang_command(args)
    elif args.command in ("holmstrom-tirole", "pledgeable-income", "credit-rationing", "ht-1997", "moral-hazard"):
        holmstrom_tirole_command(args)
    elif args.command in ("takeover-bid", "free-rider", "grossman-hart", "gh-1980", "tender-offer"):
        grossman_hart_command(args)
    elif args.command in ("glosten-milgrom", "adverse-spread", "gm-1985", "specialist-spread", "bayesian-spread"):
        glosten_milgrom_command(args)
    elif args.command in ("leland-pyle", "insider-signal", "ownership-signal", "lp-1977"):
        leland_pyle_command(args)
    elif args.command in ("debt-signaling", "ross-1977", "ross-signal", "bankruptcy-penalty"):
        ross_signaling_command(args)
    elif args.command in ("delegated-monitoring", "diamond-1984", "bank-monitoring", "delegation-cost"):
        diamond_monitoring_command(args)
    elif args.command in ("debt-taxes", "miller-1977", "tax-clientele", "miller-equilibrium"):
        miller_tax_command(args)
    elif args.command in ("habit-formation", "campbell-cochrane", "cc-1999", "surplus-consumption"):
        campbell_cochrane_command(args)
    elif args.command in ("term-structure", "cir-1985", "vasicek", "yield-curve"):
        term_structure_command(args)
    elif args.command in ("binomial-option", "crr-1979", "american-option", "option-tree"):
        binomial_option_command(args)
    elif args.command in ("exchange-option", "margrabe", "margrabe-1978", "switch-option"):
        margrabe_exchange_command(args)
    elif args.command in ("black-cox", "first-passage", "safety-covenant", "bc-1976"):
        black_cox_command(args)
    elif args.command in ("announcement-liquidity", "kim-verrecchia", "kv-1991", "differential-precision"):
        kim_verrecchia_command(args)
    elif args.command in ("liquidity-paradox", "myers-rajan", "mr-1998"):
        myers_rajan_command(args)
    elif args.command in ("hull-white", "hw-1990", "bond-option"):
        hull_white_command(args)
    elif args.command in ("modigliani-miller", "dividend-irrelevance", "mm-1961"):
        modigliani_miller_command(args)
    elif args.command in ("waiting-to-invest", "mcdonald-siegel", "ms-1986", "irreversible-investment"):
        mcdonald_siegel_command(args)
    elif args.command in ("industry-liquidation", "sv-1992"):
        shleifer_vishny_liquidation_command(args)
    elif args.command in ("entry-exit", "hysteresis-exit", "dixit-1989"):
        dixit_hysteresis_command(args)
    elif args.command in ("public-liquidity", "ht-1998", "liquidity-line"):
        holmstrom_tirole_1998_command(args)
    elif args.command in ("cppi", "portfolio-insurance", "black-jones"):
        cppi_insurance_command(args)
    elif args.command in ("informational-debt", "hr-1990", "default-investigation"):
        harris_raviv_debt_command(args)
    elif args.command in ("fund-flows", "berk-green", "bg-2004", "capacity-alpha"):
        berk_green_command(args)
    elif args.command in ("debt-maturity", "diamond-1991", "rollover-risk"):
        diamond_maturity_command(args)
    elif args.command in ("rational-bubble", "blanchard-watson", "bw-1982"):
        blanchard_watson_command(args)
    elif args.command in ("zero-beta", "black-1972"):
        black_zero_beta_command(args)
    elif args.command in ("internal-capital", "winner-picking", "stein-1997"):
        stein_internal_capital_command(args)
    elif args.command in ("predictive-regression", "stambaugh-bias", "rs-1999"):
        stambaugh_bias_command(args)
    elif args.command in ("amihud-mendelson", "am-1986", "clientele-spread"):
        amihud_mendelson_command(args)
    elif args.command in ("futures-option", "black-76", "commodity-option"):
        black76_command(args)
    elif args.command in ("risk-capital", "merton-perold", "mp-1993"):
        merton_perold_command(args)
    elif args.command in ("fama-bliss", "forward-spread", "fb-1987"):
        fama_bliss_command(args)
    elif args.command in ("reduced-credit", "duffie-singleton", "ds-1999"):
        duffie_singleton_command(args)
    elif args.command in ("fundamental-law", "grinold-kahn", "gk-1989"):
        fundamental_law_command(args)
    elif args.command in ("bayes-stein", "jorion-1986", "shrinkage-weights"):
        jorion_bayes_stein_command(args)
    elif args.command in ("m2-modigliani", "risk-adjusted-m2", "mm-rap"):
        modigliani_m2_command(args)
    elif args.command in ("bond-duration", "malkiel-duration", "convexity-theorems"):
        malkiel_duration_command(args)
    elif args.command in ("survivorship-bias", "bgir-1992", "truncation-bias"):
        brown_goetzmann_command(args)
    elif args.command in ("style-analysis", "sharpe-rbsa", "rbsa-1992"):
        sharpe_style_command(args)
    elif args.command in ("market-timing", "henriksson-merton", "treynor-mazuy", "hm-1981"):
        market_timing_command(args)
    elif args.command in ("kelly-growth", "fractional-kelly", "kt-1956"):
        kelly_growth_command(args)
    elif args.command in ("variance-ratio", "lo-mackinlay", "lm-1988"):
        lo_mackinlay_vr_command(args)
    elif args.command in ("grs-test", "gibbons-ross-shanken", "grs-1989"):
        grs_test_command(args)
    elif args.command in ("sharpe-test", "jobson-korkie", "memmel", "jkm-2003"):
        jobson_korkie_command(args)
    elif args.command in ("hurst-exponent", "fractal-market", "rs-1951"):
        hurst_exponent_command(args)
    elif args.command in ("max-diversification", "choueifaty-dr", "cc-2008"):
        max_diversification_command(args)
    elif args.command in ("brinson-attribution", "bhb-1986", "brinson-fachler"):
        brinson_attribution_command(args)
    elif args.command in ("copeland-galai", "option-spread", "cg-1983"):
        copeland_galai_command(args)
    elif args.command in ("erc-portfolio", "risk-budget", "mrt-2010"):
        erc_portfolio_command(args)
    elif args.command in ("var-backtest", "kupiec-pof", "christoffersen-cc"):
        var_backtest_command(args)
    elif args.command in ("bdt-lattice", "black-derman-toy", "bdt-1990", "rate-lattice"):
        bdt_lattice_command(args)
    elif args.command in ("nss-curve", "nelson-siegel", "svensson", "nss-1994", "yield-curve-fit"):
        nss_curve_command(args)
    elif args.command in ("gp-trading", "gp-dynamic", "gp-2013", "dynamic-aim"):
        gp_dynamic_command(args)
    elif args.command in ("dcc-garch", "dcc-2002", "dynamic-correlation", "engle-dcc"):
        dcc_garch_command(args)
    elif args.command in ("haircut-sharpe", "harvey-liu", "hlz-2016", "bh-fdr-sharpe"):
        haircut_sharpe_command(args)
    elif args.command in ("lo-sharpe", "lo-2002", "autocorr-sharpe", "lo-adjusted-sr"):
        lo_sharpe_command(args)
    elif args.command in ("key-rate-duration", "krd-1992", "fisher-weil", "immunization"):
        key_rate_immunization_command(args)
    elif args.command in ("safety-first", "roy-1952", "kataoka", "telser-1955"):
        safety_first_command(args)
    elif args.command in ("merton-share", "merton-1969", "samuelson-merton", "crra-allocation"):
        merton_crra_command(args)
    elif args.command in ("variance-swap", "carr-madan", "cm-1999", "var-swap"):
        variance_swap_command(args)
    elif args.command in ("kou-jump", "kou-2002", "double-exponential", "kou-model"):
        kou_jump_command(args)
    elif args.command in ("sabr-vol", "sabr", "sabr-2002", "hagan-sabr"):
        sabr_vol_command(args)
    elif args.command in ("dupire-vol", "dupire", "dupire-1994", "local-vol"):
        dupire_vol_command(args)
    elif args.command in ("bates-opt", "bates", "bates-1996", "svj-model"):
        bates_opt_command(args)
    elif args.command in ("cgarch", "c-garch", "component-garch", "engle-lee"):
        cgarch_command(args)
    elif args.command in ("huang-stoll", "hs-1997", "spread-decomp", "three-way-spread"):
        huang_stoll_command(args)
    elif args.command in ("ekop-pin", "ekop", "pin-model", "informed-trading"):
        ekop_pin_command(args)
    elif args.command in ("mrr-model", "mrr", "mrr-1997", "order-flow-impact"):
        mrr_command(args)
    elif args.command in ("hasbrouck-gibbs", "gibbs-spread", "hasbrouck-2009", "mcmc-spread"):
        hasbrouck_gibbs_command(args)
    elif args.command in ("gkn-spread", "gkn", "gkn-1991", "spread-purged"):
        gkn_command(args)
    elif args.command in ("glosten-harris", "gh-1988", "size-spread", "volume-impact-spread"):
        glosten_harris_command(args)
    elif args.command in ("lot-spread", "lot", "lot-1999", "zero-return-spread", "ldv-spread"):
        lot_command(args)


if __name__ == "__main__":
    main()



