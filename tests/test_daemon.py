"""Unit tests for EvaluationDaemon."""

from pathlib import Path
import pytest
from unittest.mock import MagicMock

from src.data.db import init_db, upsert_stock
from src.evaluation.daemon import EvaluationDaemon


@pytest.fixture
def temp_daemon_db(tmp_path: Path) -> Path:
    db_file = tmp_path / "test_daemon.db"
    init_db(db_file)
    upsert_stock("JPM", "XNYS", "JPMorgan Chase & Co.", "USD", "America/New_York", db_path=db_file)
    return db_file


def test_daemon_check_exchange_sessions(temp_daemon_db: Path):
    """Verify daemon retrieves structured operational session statuses for supported exchanges."""
    daemon = EvaluationDaemon(db_path=temp_daemon_db)
    statuses = daemon.check_exchange_sessions()

    assert "XNYS" in statuses
    assert "XKLS" in statuses
    assert "status" in statuses["XNYS"]
    assert "is_open" in statuses["XNYS"]
    assert isinstance(statuses["XNYS"]["is_open"], bool)


def test_daemon_run_once(temp_daemon_db: Path):
    """Verify daemon executes a single cycle and returns reconciliation metrics."""
    mock_provider = MagicMock()
    daemon = EvaluationDaemon(provider=mock_provider, db_path=temp_daemon_db)

    result = daemon.run_once()
    assert "timestamp" in result
    assert "exchange_sessions" in result
    assert "reconciliation_metrics" in result
    assert "evaluated" in result["reconciliation_metrics"]


def test_daemon_run_loop_bounded(temp_daemon_db: Path):
    """Verify daemon loop terminates cleanly when max_iterations is supplied."""
    mock_provider = MagicMock()
    daemon = EvaluationDaemon(provider=mock_provider, db_path=temp_daemon_db)

    # Run exactly 2 iterations with 0 delay
    daemon.run_loop(interval_seconds=0, max_iterations=2)
