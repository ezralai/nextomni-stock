"""Database management and SQLite storage layer for NextOmni Stock.

Implements idempotent schema creation, stock metadata tracking,
time-series observation storage, and raw payload audit records.
"""

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.config import DATABASE_PATH
from src.security.audit import (
    compute_evaluation_audit_hash,
    compute_prediction_audit_hash,
    verify_evaluation_hash,
    verify_prediction_hash,
)
from src.security.validators import assert_finite_number


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a connection to the SQLite database with foreign keys enabled."""
    path = Path(db_path) if db_path else DATABASE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA busy_timeout = 5000;")
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
                lower_bound REAL,
                upper_bound REAL,
                confidence_level REAL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                adjustment_convention TEXT NOT NULL,
                reproducibility_ref TEXT,
                audit_hash TEXT,
                status TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (ticker) REFERENCES stocks (ticker) ON DELETE CASCADE
            );
            """
        )
        for col in ["lower_bound", "upper_bound", "confidence_level"]:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {col} REAL;")
            except sqlite3.OperationalError:
                pass

        try:
            conn.execute("ALTER TABLE predictions ADD COLUMN audit_hash TEXT;")
        except sqlite3.OperationalError:
            pass

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_status_target ON predictions(status, target_session_date);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_ticker_horizon ON predictions(ticker, horizon_type);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_status_created ON predictions(status, created_at);"
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
                audit_hash TEXT,
                evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (prediction_id) REFERENCES predictions (prediction_id) ON DELETE CASCADE
            );
            """
        )
        try:
            conn.execute("ALTER TABLE prediction_evaluations ADD COLUMN audit_hash TEXT;")
        except sqlite3.OperationalError:
            pass
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


def prune_raw_payloads(
    retention_days: int = 90,
    db_path: Optional[Path] = None,
) -> int:
    """Safely prune raw provider payload JSON blobs older than retention_days.

    Strictly preserves market observations, predictions, and evaluation audit records.
    Uses parameterized SQL to guard against SQL injection.

    Args:
        retention_days: Maximum age of payload records in days (must be >= 1).
        db_path: Optional custom path to SQLite database.

    Returns:
        Number of raw payload records deleted.
    """
    if retention_days < 1:
        raise ValueError(f"retention_days must be >= 1, got {retention_days}")

    conn = get_connection(db_path)
    with conn:
        cursor = conn.execute(
            """
            DELETE FROM raw_provider_payloads
            WHERE retrieved_at < datetime('now', ?);
            """,
            (f"-{int(retention_days)} days",),
        )
        deleted_count = cursor.rowcount
    conn.close()
    return int(deleted_count)




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
    origin_price = assert_finite_number(pred["origin_price"], "origin_price")
    predicted_price = assert_finite_number(pred["predicted_price"], "predicted_price")
    lower_bound = (
        assert_finite_number(pred["lower_bound"], "lower_bound")
        if pred.get("lower_bound") is not None
        else None
    )
    upper_bound = (
        assert_finite_number(pred["upper_bound"], "upper_bound")
        if pred.get("upper_bound") is not None
        else None
    )
    confidence_level = (
        float(pred["confidence_level"])
        if pred.get("confidence_level") is not None
        else None
    )

    audit_hash = pred.get("audit_hash") or compute_prediction_audit_hash(pred)

    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO predictions (
                prediction_id, ticker, exchange, currency,
                forecast_origin_timestamp, forecast_origin_session, origin_price,
                horizon_type, horizon_sessions, target_session_date,
                predicted_price, lower_bound, upper_bound, confidence_level,
                model_name, model_version,
                adjustment_convention, reproducibility_ref, audit_hash, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                str(pred["prediction_id"]),
                str(pred["ticker"]),
                str(pred["exchange"]),
                str(pred["currency"]),
                str(pred["forecast_origin_timestamp"]),
                str(pred["forecast_origin_session"]),
                origin_price,
                str(pred["horizon_type"]),
                int(pred["horizon_sessions"]),
                str(pred["target_session_date"]),
                predicted_price,
                lower_bound,
                upper_bound,
                confidence_level,
                str(pred["model_name"]),
                str(pred["model_version"]),
                str(pred.get("adjustment_convention", "ADJUSTED_CLOSE")),
                str(pred.get("reproducibility_ref", "")),
                str(audit_hash),
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
    actual_price = assert_finite_number(eval_record["actual_price"], "actual_price")
    absolute_error = assert_finite_number(eval_record["absolute_error"], "absolute_error")
    percentage_error = assert_finite_number(eval_record["percentage_error"], "percentage_error")

    audit_hash = eval_record.get("audit_hash") or compute_evaluation_audit_hash(eval_record)

    conn = get_connection(db_path)
    with conn:
        conn.execute(
            """
            INSERT INTO prediction_evaluations (
                prediction_id, actual_price, actual_source,
                actual_observed_at, absolute_error, percentage_error,
                actual_direction, predicted_direction, direction_correct,
                audit_hash
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prediction_id) DO UPDATE SET
                actual_price=excluded.actual_price,
                absolute_error=excluded.absolute_error,
                percentage_error=excluded.percentage_error,
                direction_correct=excluded.direction_correct,
                audit_hash=excluded.audit_hash,
                evaluated_at=CURRENT_TIMESTAMP;
            """,
            (
                str(eval_record["prediction_id"]),
                actual_price,
                str(eval_record["actual_source"]),
                str(eval_record["actual_observed_at"]),
                absolute_error,
                percentage_error,
                int(eval_record["actual_direction"]),
                int(eval_record["predicted_direction"]),
                bool(eval_record["direction_correct"]),
                str(audit_hash),
            ),
        )
        # Update status in predictions table
        conn.execute(
            "UPDATE predictions SET status = 'EVALUATED' WHERE prediction_id = ?;",
            (str(eval_record["prediction_id"]),),
        )
    conn.close()
    return 1


def verify_database_integrity(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Cryptographically verify the SHA-256 integrity of all predictions and evaluations at rest.

    Returns summary metrics and lists of any tampered record identifiers.
    """
    init_db(db_path)
    conn = get_connection(db_path)
    cur = conn.cursor()

    cur.execute("SELECT * FROM predictions WHERE audit_hash IS NOT NULL;")
    pred_rows = [dict(row) for row in cur.fetchall()]

    cur.execute("SELECT * FROM prediction_evaluations WHERE audit_hash IS NOT NULL;")
    eval_rows = [dict(row) for row in cur.fetchall()]
    conn.close()

    tampered_predictions = []
    for row in pred_rows:
        expected = row.get("audit_hash", "")
        if not verify_prediction_hash(row, expected):
            tampered_predictions.append(row["prediction_id"])

    tampered_evaluations = []
    for row in eval_rows:
        expected = row.get("audit_hash", "")
        if not verify_evaluation_hash(row, expected):
            tampered_evaluations.append(row["prediction_id"])

    return {
        "predictions_audited": len(pred_rows),
        "predictions_valid": len(pred_rows) - len(tampered_predictions),
        "tampered_predictions": tampered_predictions,
        "evaluations_audited": len(eval_rows),
        "evaluations_valid": len(eval_rows) - len(tampered_evaluations),
        "tampered_evaluations": tampered_evaluations,
        "is_intact": (len(tampered_predictions) == 0 and len(tampered_evaluations) == 0),
    }


def backup_database(
    destination_path: Optional[Union[str, Path]] = None,
    db_path: Optional[Path] = None,
) -> Path:
    """Create an atomic, non-blocking online SQLite backup snapshot and verify integrity.

    Uses SQLite's online backup API to ensure 100% consistency even under active WAL writes,
    then cryptographically validates the snapshot against tamper detection.

    Args:
        destination_path: Path to target backup file. If None, saves to data/backups/stock_system_YYYYMMDD_HHMMSS.db.
        db_path: Source database path.

    Returns:
        Path to the verified backup file.
    """
    from datetime import datetime

    source_path = Path(db_path) if db_path else DATABASE_PATH
    init_db(source_path)

    if destination_path is None:
        backup_dir = source_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_path = backup_dir / f"stock_system_backup_{timestamp_str}.db"
    else:
        target_path = Path(destination_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)

    src_conn = get_connection(source_path)
    dst_conn = sqlite3.connect(str(target_path))

    with dst_conn:
        src_conn.backup(dst_conn, pages=250)

    dst_conn.close()
    src_conn.close()

    # Cryptographically verify the backup snapshot
    audit_report = verify_database_integrity(target_path)
    if not audit_report["is_intact"]:
        raise RuntimeError(
            f"Backup verification failed: snapshot at {target_path} failed cryptographic integrity checks."
        )

    return target_path


def vacuum_database(db_path: Optional[Path] = None) -> Dict[str, Any]:
    """Execute SQLite integrity checks and VACUUM to defragment B-trees and reclaim disk pages.

    1. Checks initial database file size.
    2. Runs PRAGMA foreign_key_check to ensure referential integrity.
    3. Runs PRAGMA integrity_check to verify page allocation and B-tree consistency.
    4. Executes VACUUM to reclaim deleted page space and compact storage.
    5. Returns a structured report with bytes reclaimed and verification status.

    Args:
        db_path: Path to target database file.

    Returns:
        Dict[str, Any]: Status, initial size, final size, bytes reclaimed, and integrity reports.
    """
    path = Path(db_path) if db_path else DATABASE_PATH
    if not path.exists():
        init_db(path)

    size_before = path.stat().st_size

    conn = get_connection(path)
    # Check foreign keys
    fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
    if fk_violations:
        conn.close()
        raise RuntimeError(f"Database foreign key check failed with {len(fk_violations)} violation(s).")

    # Check structural integrity
    integrity_rows = conn.execute("PRAGMA integrity_check;").fetchall()
    integrity_status = [row[0] for row in integrity_rows]
    if "ok" not in integrity_status:
        conn.close()
        raise RuntimeError(f"Database integrity check failed: {integrity_status}")

    # Reclaim free pages and defragment
    conn.execute("VACUUM;")
    conn.close()

    size_after = path.stat().st_size
    reclaimed = max(0, size_before - size_after)

    return {
        "status": "success",
        "db_path": str(path),
        "size_before_bytes": size_before,
        "size_after_bytes": size_after,
        "bytes_reclaimed": reclaimed,
        "integrity": "ok",
        "foreign_keys": "ok",
    }



