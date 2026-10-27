"""Unit tests for security validators and path traversal defenses."""

from pathlib import Path
import pytest

from src.security.validators import (
    SecurityValidationError,
    assert_safe_model_path,
    sanitize_path_component,
    validate_exchange,
    validate_session_date,
    validate_ticker,
)


def test_validate_ticker_valid():
    """Verify accepted ticker patterns for NYSE and Bursa Malaysia."""
    assert validate_ticker("JPM") == "JPM"
    assert validate_ticker("ko") == "KO"
    assert validate_ticker("1155.KL") == "1155.KL"
    assert validate_ticker("5347.kl") == "5347.KL"
    assert validate_ticker("  WMT  ") == "WMT"


def test_validate_ticker_invalid_and_traversal():
    """Verify rejection of malicious, malformed, or path traversal tickers."""
    with pytest.raises(SecurityValidationError):
        validate_ticker("")

    with pytest.raises(SecurityValidationError):
        validate_ticker("../../../etc/passwd")

    with pytest.raises(SecurityValidationError):
        validate_ticker("INVALID_TICKER_TOOLONG")

    with pytest.raises(SecurityValidationError):
        validate_ticker("JPM; DROP TABLE stocks;")

    with pytest.raises(SecurityValidationError):
        validate_ticker("JPM/test")


def test_validate_exchange_valid_and_invalid():
    """Verify exchange allowlist verification."""
    assert validate_exchange("XNYS") == "XNYS"
    assert validate_exchange("xkls") == "XKLS"

    with pytest.raises(SecurityValidationError):
        validate_exchange("NASDAQ")

    with pytest.raises(SecurityValidationError):
        validate_exchange("UNKNOWN")


def test_validate_session_date_valid_and_invalid():
    """Verify calendar date validation."""
    assert validate_session_date("2024-10-01") == "2024-10-01"

    with pytest.raises(SecurityValidationError):
        validate_session_date("2024-02-30")  # Invalid date

    with pytest.raises(SecurityValidationError):
        validate_session_date("01-10-2024")  # Wrong format

    with pytest.raises(SecurityValidationError):
        validate_session_date("2024/10/01")


def test_sanitize_path_component():
    """Verify stripping of traversal dots and slashes."""
    assert sanitize_path_component("1155.KL") == "1155_KL"
    assert sanitize_path_component("../evil/path") == "evilpath"
    assert sanitize_path_component("JPM") == "JPM"


def test_assert_safe_model_path_containment(tmp_path):
    """Verify assert_safe_model_path strictly confines writes within the base directory."""
    base_dir = tmp_path / "models"
    base_dir.mkdir()

    safe_target = base_dir / "safe_model.joblib"
    assert assert_safe_model_path(safe_target, base_dir) == safe_target.resolve()

    # Traversal escape attempt
    unsafe_target = base_dir / ".." / "evil.joblib"
    with pytest.raises(SecurityValidationError):
        assert_safe_model_path(unsafe_target, base_dir)


def test_validate_model_family_valid_and_invalid():
    """Verify validation against supported model families."""
    from src.security.validators import validate_model_family

    assert validate_model_family("linear") == "linear"
    assert validate_model_family("LINEAR") == "linear"
    assert validate_model_family("gradient_boosting") == "gradient_boosting"
    assert validate_model_family("  GRADIENT_BOOSTING  ") == "gradient_boosting"

    with pytest.raises(SecurityValidationError):
        validate_model_family("arima")

    with pytest.raises(SecurityValidationError):
        validate_model_family("random_forest")

    with pytest.raises(SecurityValidationError):
        validate_model_family("")


def test_assert_finite_number_valid_and_invalid():
    """Verify numeric finite assertion prevents NaN and infinite values."""
    from src.security.validators import assert_finite_number

    assert assert_finite_number(100.5, "price") == 100.5
    assert assert_finite_number(0.0, "return") == 0.0
    assert assert_finite_number(-12.34, "error") == -12.34

    with pytest.raises(ValueError):
        assert_finite_number(float("nan"), "nan_price")

    with pytest.raises(ValueError):
        assert_finite_number(float("inf"), "inf_price")

    with pytest.raises(ValueError):
        assert_finite_number(float("-inf"), "neg_inf_price")

    with pytest.raises(ValueError):
        assert_finite_number("not_a_number", "invalid_type")

