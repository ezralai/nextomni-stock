"""Provider rate limiting and quota telemetry tracking for NextOmni Stock.

Implements thread-safe sliding window rate limiting and telemetry tracking
to prevent API quota exhaustion and log automated provider failovers.
"""

from dataclasses import asdict, dataclass
import threading
import time
from typing import Dict, List, Optional


@dataclass
class QuotaTelemetryReceipt:
    """Diagnostic receipt documenting API quota consumption and health."""
    provider: str
    total_requests: int
    requests_last_minute: int
    max_requests_per_minute: int
    failover_count: int
    quota_remaining: int
    is_throttled: bool
    last_request_time: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class ProviderQuotaTracker:
    """Thread-safe sliding-window token tracker for external market data APIs."""

    def __init__(self, provider: str, max_requests_per_minute: int = 8):
        self.provider = provider
        self.max_requests_per_minute = max_requests_per_minute
        self._timestamps: List[float] = []
        self._total_requests: int = 0
        self._failover_count: int = 0
        self._last_request_time: Optional[float] = None
        self._lock = threading.Lock()

    def _purge_stale(self, now: float) -> None:
        """Purge timestamps older than 60 seconds (must be called with lock held)."""
        cutoff = now - 60.0
        self._timestamps = [t for t in self._timestamps if t >= cutoff]

    def acquire(self) -> bool:
        """Attempt to acquire a request token within the 60-second sliding window.

        Returns:
            True if permitted and token recorded; False if rate limit reached.
        """
        with self._lock:
            now = time.time()
            self._purge_stale(now)
            if len(self._timestamps) >= self.max_requests_per_minute:
                return False

            self._timestamps.append(now)
            self._total_requests += 1
            self._last_request_time = now
            return True

    def record_failover(self) -> None:
        """Record an automated provider failover event."""
        with self._lock:
            self._failover_count += 1

    def get_telemetry(self) -> QuotaTelemetryReceipt:
        """Retrieve point-in-time quota utilization and health telemetry."""
        with self._lock:
            now = time.time()
            self._purge_stale(now)
            current_window = len(self._timestamps)
            remaining = max(0, self.max_requests_per_minute - current_window)
            throttled = current_window >= self.max_requests_per_minute

            return QuotaTelemetryReceipt(
                provider=self.provider,
                total_requests=self._total_requests,
                requests_last_minute=current_window,
                max_requests_per_minute=self.max_requests_per_minute,
                failover_count=self._failover_count,
                quota_remaining=remaining,
                is_throttled=throttled,
                last_request_time=self._last_request_time,
            )

    def reset(self) -> None:
        """Reset internal metrics (useful in unit testing)."""
        with self._lock:
            self._timestamps.clear()
            self._total_requests = 0
            self._failover_count = 0
            self._last_request_time = None


_REGISTRY: Dict[str, ProviderQuotaTracker] = {}
_REGISTRY_LOCK = threading.Lock()


def get_quota_tracker(provider: str, max_requests_per_minute: int = 8) -> ProviderQuotaTracker:
    """Retrieve or create singleton quota tracker for a specific provider."""
    with _REGISTRY_LOCK:
        if provider not in _REGISTRY:
            _REGISTRY[provider] = ProviderQuotaTracker(
                provider=provider, max_requests_per_minute=max_requests_per_minute
            )
        return _REGISTRY[provider]
