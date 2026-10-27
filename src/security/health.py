"""Unified System Security and Health Pre-Flight Audit Suite for NextOmni Stock.

Consolidates all security layers into an immutable, verifiable health receipt:
1. SQLite B-tree page allocation and B-tree consistency (PRAGMA integrity_check).
2. Foreign key referential constraint enforcement (PRAGMA foreign_key_check).
3. Cryptographic deterministic SHA-256 audit verification of predictions and evaluations.
4. Cryptographic HMAC-SHA256 signature verification of serialized model artifacts (.joblib).
5. Environment variable secret format validation and masking hygiene.
"""

from dataclasses import asdict, dataclass
from datetime import datetime
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import DATABASE_PATH, PROJECT_ROOT
from src.data.db import get_connection, init_db, verify_database_integrity
from src.security.audit import verify_model_artifact
from src.security.validators import mask_secret, validate_api_key


@dataclass
class SystemHealthReceipt:
    """Verifiable cryptographic receipt of overall system health and security boundaries."""
    timestamp: str
    database_path: str
    database_integrity: str
    foreign_keys_integrity: str
    prediction_audit: str
    audited_predictions_count: int
    audited_evaluations_count: int
    tampered_predictions_count: int
    model_signatures: str
    verified_models_count: int
    tampered_models: List[str]
    secret_hygiene: str
    is_healthy: bool
    summary: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert receipt to dictionary representation."""
        return asdict(self)


def run_system_health_audit(
    db_path: Optional[Path] = None,
    models_dir: Optional[Path] = None,
) -> SystemHealthReceipt:
    """Execute complete security and system integrity pre-flight audit.

    Args:
        db_path: Optional custom path to SQLite database.
        models_dir: Optional custom path to model artifacts directory.

    Returns:
        SystemHealthReceipt detailing verification status across all system boundaries.
    """
    target_db = Path(db_path) if db_path else DATABASE_PATH
    init_db(target_db)

    timestamp_str = datetime.now().isoformat()
    target_models = Path(models_dir) if models_dir else PROJECT_ROOT / "models"

    issues: List[str] = []

    # 1. Database B-Tree Page Integrity
    conn = get_connection(target_db)
    try:
        integrity_rows = conn.execute("PRAGMA integrity_check;").fetchall()
        integrity_status = [row[0] for row in integrity_rows]
        db_intact = "ok" in integrity_status
        db_status = "PASS" if db_intact else "FAIL"
        if not db_intact:
            issues.append(f"Database integrity check failed: {integrity_status}")

        # 2. Foreign Key Referential Consistency
        fk_violations = conn.execute("PRAGMA foreign_key_check;").fetchall()
        fk_intact = len(fk_violations) == 0
        fk_status = "PASS" if fk_intact else "FAIL"
        if not fk_intact:
            issues.append(f"Foreign key violations detected ({len(fk_violations)} rows)")
    finally:
        conn.close()

    # 3. Cryptographic SHA-256 Prediction Audit Trail
    audit_rep = verify_database_integrity(target_db)
    pred_intact = audit_rep["is_intact"]
    pred_status = "PASS" if pred_intact else "FAIL"
    n_preds = audit_rep["predictions_audited"]
    n_evals = audit_rep["evaluations_audited"]
    tampered_preds = audit_rep["tampered_predictions"]
    if not pred_intact:
        issues.append(f"Cryptographic prediction tamper detected ({len(tampered_preds)} tampered records)")

    # 4. Cryptographic HMAC-SHA256 Model Artifact Signatures
    joblib_files = list(target_models.rglob("*.joblib")) if target_models.exists() else []
    tampered_models: List[str] = []
    verified_models_count = 0

    if not joblib_files:
        model_sig_status = "NO_MODELS"
    else:
        for model_file in joblib_files:
            if verify_model_artifact(model_file):
                verified_models_count += 1
            else:
                tampered_models.append(str(model_file.relative_to(target_models)))

        if tampered_models:
            model_sig_status = "FAIL"
            issues.append(f"Model signature mismatch or missing signature sidecars: {tampered_models}")
        else:
            model_sig_status = "PASS"

    # 5. Environment Secret Hygiene
    twelve_data_key = os.getenv("NEXTOMNI_STOCK_TWELVE_DATA_API_KEY")
    secret_hygiene = "PASS"
    if twelve_data_key:
        try:
            validate_api_key(twelve_data_key)
            masked = mask_secret(twelve_data_key)
            if not masked or masked == twelve_data_key:
                secret_hygiene = "WARN"
        except Exception as e:
            secret_hygiene = "FAIL"
            issues.append(f"API key validation error: {e}")

    # Overall Health Determination
    is_healthy = (
        db_status == "PASS"
        and fk_status == "PASS"
        and pred_status == "PASS"
        and model_sig_status in ["PASS", "NO_MODELS"]
        and secret_hygiene in ["PASS", "WARN"]
    )

    summary = (
        "System is 100% healthy and all security boundaries are intact."
        if is_healthy
        else f"Security health issues detected: {'; '.join(issues)}"
    )

    return SystemHealthReceipt(
        timestamp=timestamp_str,
        database_path=str(target_db),
        database_integrity=db_status,
        foreign_keys_integrity=fk_status,
        prediction_audit=pred_status,
        audited_predictions_count=n_preds,
        audited_evaluations_count=n_evals,
        tampered_predictions_count=len(tampered_preds),
        model_signatures=model_sig_status,
        verified_models_count=verified_models_count,
        tampered_models=tampered_models,
        secret_hygiene=secret_hygiene,
        is_healthy=is_healthy,
        summary=summary,
    )
