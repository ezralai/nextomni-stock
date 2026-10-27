"""Automated session-aware reconciliation daemon for NextOmni Stock.

Monitors active exchange schedules and triggers idempotent evaluation
and price reconciliation as soon as market trading sessions close.
"""

from datetime import datetime
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

from src.config import DATABASE_PATH, SUPPORTED_EXCHANGES
from src.data.base_provider import BaseMarketDataProvider
from src.data.session_status import ExchangeSessionEngine
from src.evaluation.tracker import PredictionTracker

logger = logging.getLogger(__name__)


class EvaluationDaemon:
    """Session-aware background daemon for automated prediction reconciliation."""

    def __init__(
        self,
        provider: Optional[BaseMarketDataProvider] = None,
        db_path: Optional[Path] = None,
        poll_interval_seconds: int = 300,
    ):
        self.db_path = db_path or DATABASE_PATH
        self.tracker = PredictionTracker(provider=provider, db_path=self.db_path)
        self.poll_interval = poll_interval_seconds

    def check_exchange_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Query real-time operational status for all supported exchanges."""
        status_map = {}
        for mic in SUPPORTED_EXCHANGES:
            info = ExchangeSessionEngine.get_status(mic)
            status_map[mic] = {
                "name": info.name,
                "status": info.status,
                "is_open": info.is_open,
                "session_date": info.session_date,
                "local_time": info.local_time,
            }
        return status_map

    def run_once(self) -> Dict[str, Any]:
        """Execute a single reconciliation evaluation cycle across closed sessions and audit integrity.

        Returns:
            Dictionary containing session states, evaluation counts, and cryptographic integrity report.
        """
        from src.data.db import verify_database_integrity

        session_states = self.check_exchange_sessions()
        eval_result = self.tracker.evaluate_pending_predictions()
        audit_report = verify_database_integrity(db_path=self.db_path)

        if not audit_report["is_intact"]:
            logger.critical(
                f"SECURITY TAMPER ALERT: Database at {self.db_path} failed cryptographic integrity checks! "
                f"Valid predictions: {audit_report['predictions_valid']}/{audit_report['predictions_audited']}, "
                f"Valid evaluations: {audit_report['evaluations_valid']}/{audit_report['evaluations_audited']}"
            )

        # Check Information Coefficient drift and alpha decay
        from src.analytics.drift import calculate_information_coefficient_drift
        from src.data.db import get_predictions

        evaluated_df = get_predictions(status="EVALUATED", db_path=self.db_path)
        drift_receipt = calculate_information_coefficient_drift(evaluated_df)
        if drift_receipt.is_decaying and drift_receipt.decay_warning:
            logger.warning(f"MODEL ALPHA DECAY ALERT: {drift_receipt.decay_warning}")

        return {
            "timestamp": datetime.now().isoformat(),
            "exchange_sessions": session_states,
            "reconciliation_metrics": eval_result,
            "integrity_report": audit_report,
            "alpha_drift": drift_receipt.to_dict(),
        }

    def run_loop(
        self,
        interval_seconds: Optional[int] = None,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Run continuous monitoring loop until interrupted or max iterations reached."""
        interval = self.poll_interval if interval_seconds is None else interval_seconds
        iterations = 0

        logger.info(
            f"Starting NextOmni Evaluation Daemon with {interval}s poll interval..."
        )
        try:
            while max_iterations is None or iterations < max_iterations:
                res = self.run_once()
                eval_metrics = res["reconciliation_metrics"]
                logger.info(
                    f"Daemon cycle {iterations + 1}: Evaluated={eval_metrics.get('evaluated', 0)}, "
                    f"Awaiting={eval_metrics.get('awaiting', 0)}, Pending={eval_metrics.get('pending', 0)}"
                )
                iterations += 1
                if max_iterations is not None and iterations >= max_iterations:
                    break
                time.sleep(interval)
        except KeyboardInterrupt:
            logger.info("Evaluation Daemon interrupted by user. Shutting down gracefully.")
