"""Database management and SQLite storage layer for NextOmni Stock.

Implements idempotent schema creation, stock metadata tracking,
time-series observation storage, and raw payload audit records.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

from src.config import DATABASE_PATH


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database with foreign keys enabled."""
    path = Path(db_path) if db_path else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: Optional[Path] = None) -> None:
    """Initialize the NextOmni Stock relational schema if it does not exist."""
    conn = get_connection(db_path)
    with conn:
        # 1. Stocks metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                exchange TEXT NOT NULL,
                company_name TEXT,
                currency TEXT NOT NULL,
                timezone TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # 2. Market observations table (Chronological historical & recent prices)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_observations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                exchange TEXT NOT NULL,
                session_date TEXT NOT NULL,
                open REAL NOT NULL,
                high REAL NOT NULL,
                low REAL NOT NULL,
                close REAL NOT NULL,
                adj_close REAL NOT NULL,
                volume INTEGER NOT NULL,
                data_status TEXT NOT NULL,
                provider TEXT NOT NULL,
                retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks (ticker) ON DELETE CASCADE,
                UNIQUE (ticker, session_date)
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_obs_ticker_date ON market_observations(ticker, session_date);"
        )

        # 3. Raw provider payloads table (Preserves unadjusted raw responses separately)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS raw_provider_payloads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                provider TEXT NOT NULL,
                period_start TEXT,
                period_end TEXT,
                payload_json TEXT NOT NULL,
                retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # 4. Corporate actions table (Dividends and Stock Splits)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS corporate_actions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                action_type TEXT NOT NULL,
                action_date TEXT NOT NULL,
                value REAL NOT NULL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks (ticker) ON DELETE CASCADE,
                UNIQUE (ticker, action_type, action_date)
            );
            """
        )

        # 5. Immutable predictions table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                prediction_id TEXT PRIMARY KEY,
                ticker TEXT NOT NULL,
                exchange TEXT NOT NULL,
                currency TEXT NOT NULL,
                forecast_origin_timestamp TIMESTAMP NOT NULL,
                forecast_origin_session TEXT NOT NULL,
                origin_price REAL NOT NULL,
                horizon_type TEXT NOT NULL,
                horizon_sessions INTEGER NOT NULL,
                target_session_date TEXT NOT NULL,
                predicted_price REAL NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                adjustment_convention TEXT NOT NULL,
                reproducibility_ref TEXT,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks (ticker) ON DELETE CASCADE
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_status_target ON predictions(status, target_session_date);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_ticker_horizon ON predictions(ticker, horizon_type);"
        )

        # 6. Prediction evaluations table (Stores reconciliation with actual market prices)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_evaluations (
                evaluation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id TEXT NOT NULL UNIQUE,
                actual_price REAL NOT NULL,
                actual_source TEXT NOT NULL,
                actual_observed_at TIMESTAMP NOT NULL,
                absolute_error REAL NOT NULL,
                percentage_error REAL NOT NULL,
                actual_direction INTEGER NOT NULL,
                predicted_direction INTEGER NOT NULL,
                direction_correct BOOLEAN NOT NULL,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id) ON DELETE CASCADE
            );
            """
        )
    conn.close()



def upsert_stock(
    ticker: str,
    exchange: str,
    company_name: str,
    currency: str,
    timezone: str,
    db_path: Optional[Path] = None,
) -> None:
    """Insert or update stock metadata."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO stocks (ticker, exchange, company_name, currency, timezone)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                exchange=excluded.exchange,
                company_name=excluded.company_name,
                currency=excluded.currency,
                timezone=excluded.timezone;
            """,
            (ticker, exchange, company_name, currency, timezone),
        )
    conn.close()


def save_observations(
    df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """Idempotently insert or update market observations in SQLite.

    Returns the number of rows inserted/updated.
    """
    if df.empty:
        return 0

    required_cols = [
        "ticker",
        "exchange",
        "session_date",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "data_status",
        "provider",
    ]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column in observations DataFrame: '{col}'")

    conn = get_connection(db_path)
    rows_affected = 0
    with conn:
        for _, row in df.iterrows():
            conn.execute(
                """
                INSERT INTO market_observations (
                    ticker, exchange, session_date, open, high, low, close, adj_close, volume, data_status, provider
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, session_date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    adj_close=excluded.adj_close,
                    volume=excluded.volume,
                    data_status=excluded.data_status,
                    provider=excluded.provider,
                    retrieved_at=CURRENT_TIMESTAMP;
                """,
                (
                    row["ticker"],
                    row["exchange"],
                    str(row["session_date"]),
                    float(row["open"]),
                    float(row["high"]),
                    float(row["low"]),
                    float(row["close"]),
                    float(row["adj_close"]),
                    int(row["volume"]),
                    str(row["data_status"]),
                    str(row["provider"]),
                ),
            )
            rows_affected += 1
    conn.close()
    return rows_affected


def get_observations(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Retrieve chronologically ordered market observations for a ticker."""
    conn = get_connection(db_path)
    query = "SELECT * FROM market_observations WHERE ticker = ?"
    params: List[Any] = [ticker]

    if start_date:
        query += " AND session_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND session_date <= ?"
        params.append(end_date)

    query += " ORDER BY session_date ASC;"

    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert dictionary keys and non-primitives to serializable types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
    elif hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj


def save_raw_payload(
    ticker: str,
    provider: str,
    payload: Dict[str, Any],
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> None:
    """Store raw provider payload as JSON for auditing and reproducibility."""
    conn = get_connection(db_path)
    sanitized = _sanitize_for_json(payload)
    with conn:
        conn.execute(
            """
            INSERT INTO raw_provider_payloads (
                ticker, provider, period_start, period_end, payload_json
            ) VALUES (?, ?, ?, ?, ?);
            """,
            (
                ticker,
                provider,
                period_start,
                period_end,
                json.dumps(sanitized, default=str),
            ),
        )
    conn.close()



def save_corporate_actions(
    ticker: str,
    actions_df: pd.DataFrame,
    db_path: Optional[Path] = None,
) -> int:
    """Store dividend and stock split corporate actions."""
    if actions_df.empty:
        return 0

    conn = get_connection(db_path)
    count = 0
    with conn:
        for _, row in actions_df.iterrows():
            conn.execute(
                """
                INSERT INTO corporate_actions (ticker, action_type, action_date, value)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(ticker, action_type, action_date) DO UPDATE SET
                    value=excluded.value;
                """,
                (
                    ticker,
                    str(row["action_type"]).upper(),
                    str(row["action_date"]),
                    float(row["value"]),
                ),
            )
            count += 1
    conn.close()
    return count


def save_prediction(
    pred: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> str:
    """Store an immutable forecast prediction record."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, ticker, exchange, currency,
                forecast_origin_timestamp, forecast_origin_session, origin_price,
                horizon_type, horizon_sessions, target_session_date,
                predicted_price, model_name, model_version,
                adjustment_convention, reproducibility_ref, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(pred["prediction_id"]),
                str(pred["ticker"]),
                str(pred["exchange"]),
                str(pred["currency"]),
                str(pred["forecast_origin_timestamp"]),
                str(pred["forecast_origin_session"]),
                float(pred["origin_price"]),
                str(pred["horizon_type"]),
                int(pred["horizon_sessions"]),
                str(pred["target_session_date"]),
                float(pred["predicted_price"]),
                str(pred["model_name"]),
                str(pred["model_version"]),
                str(pred.get("adjustment_convention", "ADJUSTED_CLOSE")),
                str(pred.get("reproducibility_ref", "")),
                str(pred.get("status", "PENDING")),
            ),
        )
    conn.close()
    return str(pred["prediction_id"])


def get_predictions(
    status: Optional[str] = None,
    ticker: Optional[str] = None,
    horizon_type: Optional[str] = None,
    db_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Retrieve predictions with optional status, ticker, and horizon filters."""
    conn = get_connection(db_path)
    query = """
        SELECT p.*, e.actual_price, e.absolute_error, e.percentage_error, e.direction_correct, e.evaluated_at
        FROM predictions p
        LEFT JOIN prediction_evaluations e ON p.prediction_id = e.prediction_id
        WHERE 1=1
    """
    params: List[Any] = []
    if status:
        query += " AND p.status = ?"
        params.append(status)
    if ticker:
        query += " AND p.ticker = ?"
        params.append(ticker)
    if horizon_type:
        query += " AND p.horizon_type = ?"
        params.append(horizon_type)

    query += " ORDER BY p.created_at DESC;"
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df


def update_prediction_status(
    prediction_id: str,
    new_status: str,
    db_path: Optional[Path] = None,
) -> None:
    """Update prediction lifecycle status."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE predictions SET status = ? WHERE prediction_id = ?;",
            (new_status, prediction_id),
        )
    conn.close()


def save_prediction_evaluation(
    eval_record: Dict[str, Any],
    db_path: Optional[Path] = None,
) -> int:
    """Store reconciled evaluation and update prediction status to EVALUATED."""
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO prediction_evaluations (
                prediction_id, actual_price, actual_source,
                actual_observed_at, absolute_error, percentage_error,
                actual_direction, predicted_direction, direction_correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                actual_price=excluded.actual_price,
                absolute_error=excluded.absolute_error,
                percentage_error=excluded.percentage_error,
                direction_correct=excluded.direction_correct,
                evaluated_at=CURRENT_TIMESTAMP;
            """,
            (
                str(eval_record["prediction_id"]),
                float(eval_record["actual_price"]),
                str(eval_record["actual_source"]),
                str(eval_record["actual_observed_at"]),
                float(eval_record["absolute_error"]),
                float(eval_record["percentage_error"]),
                int(eval_record["actual_direction"]),
                int(eval_record["predicted_direction"]),
                bool(eval_record["direction_correct"]),
            ),
        )
        # Update status in predictions table
        conn.execute(
            "UPDATE predictions SET status = 'EVALUATED' WHERE prediction_id = ?;",
            (str(eval_record["prediction_id"]),),
        )
    conn.close()
    return 1

