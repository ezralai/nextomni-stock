"""Security and input validation contracts for NextOmni Stock.

Implements strict allowlist verification, regex checks, and path traversal
guards across data ingestion, model persistence, and CLI commands.
"""

from datetime import datetime
from pathlib import Path
import re
from typing import Set

from src.config import SUPPORTED_EXCHANGES

SUPPORTED_MODEL_FAMILIES = {"linear", "gradient_boosting", "baseline"}

# Ticker pattern: 1 to 6 alphanumeric characters, optional exchange suffix (e.g. .KL)
TICKER_REGEX = re.compile(r"^[A-Z0-9]{1,6}(\.[A-Z]{2})?$")
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class SecurityValidationError(ValueError):
    """Raised when an input fails security or contract validation."""
    pass


def validate_ticker(ticker: str) -> str:
    """Validate and normalize a stock ticker against strict security patterns.

    Args:
        ticker: Stock symbol (e.g., 'JPM', '1155.KL').

    Returns:
        Cleaned, uppercase ticker string.

    Raises:
        SecurityValidationError: If the ticker contains illegal characters or path elements.
    """
    if not ticker or not isinstance(ticker, str):
        raise SecurityValidationError("Ticker must be a non-empty string.")

    cleaned = ticker.strip().upper()
    if not TICKER_REGEX.match(cleaned):
        raise SecurityValidationError(
            f"Invalid ticker format '{ticker}'. Must match pattern: ^[A-Z0-9]{{1,6}}(\\.[A-Z]{{2}})?$"
        )

    # Explicitly reject directory traversal sequences
    if ".." in cleaned or "/" in cleaned or "\\" in cleaned:
        raise SecurityValidationError(f"Path traversal sequence detected in ticker: {ticker}")

    return cleaned


def validate_exchange(exchange: str) -> str:
    """Validate exchange Market Identifier Code (MIC) against supported allowlist.

    Args:
        exchange: Exchange MIC ('XNYS', 'XKLS').

    Returns:
        Uppercase exchange code.

    Raises:
        SecurityValidationError: If exchange is not in SUPPORTED_EXCHANGES.
    """
    if not exchange or not isinstance(exchange, str):
        raise SecurityValidationError("Exchange must be a non-empty string.")

    cleaned = exchange.strip().upper()
    if cleaned not in SUPPORTED_EXCHANGES:
        raise SecurityValidationError(
            f"Unsupported exchange '{exchange}'. Must be one of {list(SUPPORTED_EXCHANGES.keys())}"
        )

    return cleaned


def validate_session_date(date_str: str) -> str:
    """Validate that a date string matches strict ISO 8601 YYYY-MM-DD format and is a valid calendar date.

    Args:
        date_str: Date string to validate.

    Returns:
        Standardized date string.

    Raises:
        SecurityValidationError: If date is malformed or invalid.
    """
    if not date_str or not isinstance(date_str, str):
        raise SecurityValidationError("Date string must be non-empty.")

    cleaned = date_str.strip()
    if not ISO_DATE_REGEX.match(cleaned):
        raise SecurityValidationError(f"Invalid date format '{date_str}'. Expected format YYYY-MM-DD.")

    try:
        dt = datetime.strptime(cleaned, "%Y-%m-%d")
        return dt.strftime("%Y-%m-%d")
    except ValueError as e:
        raise SecurityValidationError(f"Invalid calendar date '{date_str}': {str(e)}")


def sanitize_path_component(name: str) -> str:
    """Sanitize a string for safe use in file system directory or filename creation.

    Replaces non-alphanumeric characters with underscores and strips path traversal.
    """
    if not name or not isinstance(name, str):
        raise SecurityValidationError("Path component must be a non-empty string.")

    # Remove any directory separators
    clean = re.sub(r"[/\\]", "", name.strip())
    # Remove leading dots or directory traversal
    clean = clean.replace("..", "").replace(".", "_")
    # Replace non-alphanumeric characters with underscores
    clean = re.sub(r"[^A-Za-z0-9_]", "_", clean)
    if not clean:
        raise SecurityValidationError(f"Component '{name}' reduced to empty string after sanitization.")
    return clean


def assert_safe_model_path(target_path: Path, base_dir: Path) -> Path:
    """Assert that target_path resolves strictly within base_dir, guarding against path traversal.

    Args:
        target_path: Proposed destination path.
        base_dir: Approved parent directory boundary.

    Returns:
        Resolved target Path.

    Raises:
        SecurityValidationError: If target_path escapes base_dir.
    """
    resolved_target = target_path.resolve()
    resolved_base = base_dir.resolve()

    try:
        resolved_target.relative_to(resolved_base)
    except ValueError:
        raise SecurityValidationError(
            f"Path traversal security violation: Target path '{resolved_target}' escapes base directory '{resolved_base}'."
        )

    return resolved_target


def validate_model_family(model: str) -> str:
    """Validate model family choice against approved allowlist."""
    if not model or not isinstance(model, str):
        raise SecurityValidationError("Model family must be a non-empty string.")

    cleaned = model.strip().lower()
    if cleaned not in SUPPORTED_MODEL_FAMILIES:
        raise SecurityValidationError(
            f"Unsupported model family '{model}'. Must be one of {sorted(list(SUPPORTED_MODEL_FAMILIES))}"
        )
    return cleaned


def assert_finite_number(val: float, field_name: str = "Value") -> float:
    """Validate that a numeric value is a valid finite float (rejecting NaN, Inf, -Inf)."""
    import math
    try:
        f_val = float(val)
    except (TypeError, ValueError):
        raise SecurityValidationError(f"{field_name} must be a valid float, got {type(val)}: {val}")

    if not math.isfinite(f_val):
        raise SecurityValidationError(f"{field_name} must be finite (got {val}).")

    return f_val
