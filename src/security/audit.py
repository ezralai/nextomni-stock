"""Cryptographic audit hashing and tamper-detection for NextOmni Stock.

Implements deterministic SHA-256 integrity digests across immutable predictions
and evaluation records to guarantee audit trail integrity at rest.
"""

import hashlib
import json
from typing import Any, Dict, List, Optional


PREDICTION_CORE_FIELDS = [
    "prediction_id",
    "ticker",
    "exchange",
    "forecast_origin_session",
    "origin_price",
    "horizon_type",
    "horizon_sessions",
    "target_session_date",
    "predicted_price",
    "model_name",
    "model_version",
]

EVALUATION_CORE_FIELDS = [
    "prediction_id",
    "actual_price",
    "actual_source",
    "absolute_error",
    "percentage_error",
    "direction_correct",
]


def _canonicalize_value(val: Any) -> Any:
    """Normalize values for deterministic hashing across platforms."""
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (int, )):
        return str(val)
    if isinstance(val, float):
        # Round to 6 decimal places to prevent platform-dependent floating point drift
        return f"{float(val):.6f}"
    if val is None:
        return ""
    return str(val).strip()


def compute_prediction_audit_hash(pred: Dict[str, Any]) -> str:
    """Compute a canonical SHA-256 audit digest for an immutable prediction record.

    Args:
        pred: Dictionary containing prediction fields.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    canonical_items = [
        f"{field}:{_canonicalize_value(pred.get(field))}"
        for field in PREDICTION_CORE_FIELDS
    ]
    canonical_payload = "|".join(canonical_items).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def compute_evaluation_audit_hash(eval_entry: Dict[str, Any]) -> str:
    """Compute a canonical SHA-256 audit digest for an evaluation record.

    Args:
        eval_entry: Dictionary containing evaluation fields.

    Returns:
        Hex-encoded SHA-256 digest.
    """
    canonical_items = [
        f"{field}:{_canonicalize_value(eval_entry.get(field))}"
        for field in EVALUATION_CORE_FIELDS
    ]
    canonical_payload = "|".join(canonical_items).encode("utf-8")
    return hashlib.sha256(canonical_payload).hexdigest()


def verify_prediction_hash(pred: Dict[str, Any], expected_hash: str) -> bool:
    """Verify if a prediction's fields match its stored audit hash."""
    if not expected_hash:
        return False
    computed = compute_prediction_audit_hash(pred)
    return computed.lower() == expected_hash.strip().lower()


def verify_evaluation_hash(eval_entry: Dict[str, Any], expected_hash: str) -> bool:
    """Verify if an evaluation record matches its stored audit hash."""
    if not expected_hash:
        return False
    computed = compute_evaluation_audit_hash(eval_entry)
    return computed.lower() == expected_hash.strip().lower()
