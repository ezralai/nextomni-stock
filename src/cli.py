"""Command-line interface for NextOmni Stock (nextomni-stock).

Allows syncing historical stock data, inspecting database state,
and testing data providers directly from the terminal.
"""

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys
import pandas as pd



from src.config import INITIAL_TICKERS, SUPPORTED_EXCHANGES
from src.data.db import get_connection, get_observations, init_db
from src.data.pipeline import DataPipeline


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

    print(f"Syncing {args.ticker} on {args.exchange} from {args.start} to {args.end}...")
    result = pipeline.sync_stock(
        ticker=args.ticker,
        exchange=args.exchange,
        start_date=args.start,
        end_date=args.end,
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

def analyze_command(args: argparse.Namespace) -> None:
    """Run exploratory data analysis and export interactive visual report."""
    from src.analytics.eda import export_eda_report

    print(f"\nRunning Exploratory Data Analysis for {args.ticker}...")
    try:
        stats, report_path = export_eda_report(ticker=args.ticker)
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
    print(f"\nInteractive visual report generated at: {report_path.resolve()}\n")


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

def forecast_baseline_command(args: argparse.Namespace) -> None:
    """Generate persistence baseline forecast for next trading session and evaluate test performance."""
    from src.data.db import get_observations
    from src.models.baseline import PersistenceBaselineModel

    df = get_observations(args.ticker)
    if df.empty:
        print(f"Error: No observations found for {args.ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = args.exchange or df["exchange"].iloc[0]
    model = PersistenceBaselineModel()

    # 1. Evaluate historical performance
    eval_result = model.evaluate(df, horizon_sessions=1, train_ratio=0.8)
    metrics = eval_result["metrics"]

    # 2. Generate future forecast
    future = model.generate_future_forecast(args.ticker, exchange, horizon_sessions=1)

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

def train_linear_command(args: argparse.Namespace) -> None:
    """Execute walk-forward validation and train linear regression model."""
    from src.data.db import get_observations
    from src.evaluation.validation import WalkForwardValidator
    from src.features.builder import build_feature_matrix
    from src.models.linear_model import LinearRegressionForecaster

    df = get_observations(args.ticker)
    if df.empty:
        print(f"Error: No observations found for {args.ticker}. Run 'sync' first.")
        sys.exit(1)

    currency = "USD" if df["exchange"].iloc[0] == "XNYS" else "MYR"
    validator = WalkForwardValidator(min_train_sessions=args.min_train, step_size=1)

    print(f"\nRunning Chronological Walk-Forward Validation for {args.ticker}...")
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
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / f"linear_{args.ticker.replace('.', '_')}.joblib"

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

def predict_command(args: argparse.Namespace) -> None:
    """Generate and record a live forecast prediction into SQLite."""
    import exchange_calendars as xcals
    from src.data.db import get_observations
    from src.evaluation.tracker import PredictionTracker
    from src.features.builder import build_feature_matrix
    from src.models.baseline import PersistenceBaselineModel
    from src.models.linear_model import LinearRegressionForecaster

    df = get_observations(args.ticker)
    if df.empty:
        print(f"Error: No observations found for {args.ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = args.exchange or df["exchange"].iloc[0]
    horizon = args.horizon
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

    if args.model == "baseline":
        model_name = "PersistenceBaseline"
        model_version = "1.0.0"
        pred_price = origin_price
        repro_ref = "persistence_t"
    else:
        model_path = Path("models") / f"linear_{args.ticker.replace('.', '_')}.joblib"
        if not model_path.exists():
            print(f"Model artifact not found for {args.ticker}. Run 'train-linear --ticker {args.ticker}' first.")
            sys.exit(1)
        forecaster = LinearRegressionForecaster.load(model_path)
        feature_df, feat_cols = build_feature_matrix(df, include_target=False)
        latest_features = feature_df.iloc[-1][feat_cols].to_frame().T
        pred_price = float(forecaster.predict(latest_features, p_origin=origin_price)[0])
        model_name = forecaster.model_name
        model_version = forecaster.model_version
        repro_ref = str(model_path)

    record = tracker.record_prediction(
        ticker=args.ticker,
        exchange=exchange,
        forecast_origin_session=origin_session,
        origin_price=origin_price,
        horizon_sessions=horizon,
        target_session_date=target_date_str,
        predicted_price=pred_price,
        model_name=model_name,
        model_version=model_version,
        reproducibility_ref=repro_ref,
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
    print(f"  Model:                  {record['model_name']} (v{record['model_version']})")
    print(f"  Status:                 {record['status']}")
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

    df = get_predictions(status=args.status, ticker=args.ticker)
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


def monitor_command(args: argparse.Namespace) -> None:
    """Monitor real-time quotes with session status and latency badges."""
    from src.data.monitor import MarketMonitorService
    service = MarketMonitorService()
    quotes = service.get_multi_quotes(
        tickers=[args.ticker] if args.ticker else None,
        exchange=args.exchange,
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
    exchanges = [args.exchange] if args.exchange else list(SUPPORTED_EXCHANGES.keys())

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
def train_all_horizons_command(args: argparse.Namespace) -> None:
    """Train direct models for 1-session, 5-session, and 21-session horizons with walk-forward validation."""
    from src.data.db import get_observations
    from src.models.multi_horizon import MultiHorizonForecaster

    df = get_observations(args.ticker)
    if df.empty:
        print(f"Error: No observations found for {args.ticker}. Run 'sync' first.")
        sys.exit(1)

    exchange = args.exchange or df["exchange"].iloc[0]
    currency = "USD" if exchange == "XNYS" else "MYR"

    print(f"\n==========================================================================")
    print(f"  NextOmni Stock: Multi-Horizon Direct Model Training ({args.ticker})")
    print(f"==========================================================================")
    print("Training dedicated direct models for:")
    print("  • 1 Session  (P_{t+1})  - Next Trading Day")
    print("  • 5 Sessions (P_{t+5})  - Next Trading Week (1 Calendar Week)")
    print("  • 21 Sessions (P_{t+21}) - Next Trading Month (~1 Calendar Month)")
    print("\nRunning Walk-Forward Out-of-Sample Validation across all horizons...")

    forecaster = MultiHorizonForecaster(alpha=args.alpha)
    validation_results = forecaster.evaluate_walk_forward_all(df, min_train_sessions=args.min_train)

    print("\n" + "=" * 90)
    print(f"  Multi-Horizon Out-of-Sample Performance Comparison ({args.ticker})")
    print("=" * 90)
    print(f"{'Horizon':<15} {'Steps':<8} {'Model MAE':<14} {'Baseline MAE':<15} {'Improvement':<13} {'Dir Acc (%)':<12}")
    print("-" * 90)

    for h_key, res in validation_results.items():
        m_m = res["model_metrics"]
        b_m = res["baseline_metrics"]
        mae_imp = f"{res['mae_improvement_pct']:>+6.2f}%"
        da = f"{m_m['directional_accuracy_pct']:.2f}%"
        print(
            f"{h_key:<15} {res['walk_forward_steps']:<8} "
            f"{m_m['mae']:<14.4f} {b_m['mae']:<15.4f} {mae_imp:<13} {da:<12}"
        )
    print("-" * 90)

    # Train final models on full feature sets and save
    forecaster.fit(df)
    models_dir = Path("models")
    save_path = forecaster.save_models(models_dir, args.ticker)
    print(f"\nMulti-horizon models saved to: {save_path.resolve()}")

    # Generate current future predictions
    future_forecasts = forecaster.predict_all_horizons(df, args.ticker, exchange)
    print("\nCurrent Multi-Horizon Forecasts (from latest observed session):")
    print(f"{'Horizon':<15} {'Target Date':<14} {'Origin Price':<14} {'Predicted Price':<16} {'Exp Return':<12}")
    print("-" * 75)
    for h_key, fc in future_forecasts.items():
        print(
            f"{h_key:<15} {fc['target_session_date']:<14} "
            f"{fc['origin_price']:<14.4f} {fc['predicted_price']:<16.4f} {fc['expected_return_pct']:>+6.2f}%"
        )
    print("-" * 75 + "\n")


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
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
    )
    multi_parser.add_argument(
        "--min-train", type=int, default=120, help="Minimum initial training sessions"
    )
    multi_parser.add_argument(
        "--alpha", type=float, default=1.0, help="Ridge regularization strength alpha"
    )

    # Predict parser
    predict_parser = subparsers.add_parser(
        "predict", help="Generate and store a forecast prediction"
    )
    predict_parser.add_argument(
        "--ticker", type=str, required=True, help="Stock ticker (e.g. JPM, 1155.KL)"
    )
    predict_parser.add_argument(
        "--model", choices=["linear", "baseline"], default="linear", help="Forecasting model to use"
    )
    predict_parser.add_argument(
        "--horizon", type=int, default=1, choices=[1, 5, 21], help="Forecast horizon sessions"
    )
    predict_parser.add_argument(
        "--exchange", choices=list(SUPPORTED_EXCHANGES.keys()), default=None, help="Exchange MIC"
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



if __name__ == "__main__":
    main()




