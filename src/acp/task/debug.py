"""Debugging extensions for ACP task system."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from acp.task.state import InMemoryMessageStateStore


if TYPE_CHECKING:
    import asyncio

    from acp.task.state import IncomingMessage


__all__ = ["DebugEntry", "DebuggingMessageStateStore"]


@dataclass
class DebugEntry:
    """Structured debug entry for ACP message tracking."""

    timestamp: str
    direction: Literal["outgoing", "incoming"]
    event: Literal["register", "resolve", "reject", "begin", "complete", "fail"]
    request_id: int | None
    method: str
    params: Any = None
    result: Any = None
    error: Any = None
    status: str | None = None
    duration_ms: float | None = None


class DebuggingMessageStateStore(InMemoryMessageStateStore):
    """Enhanced message state store with structured debugging output.

    Provides much richer debugging information than raw JSON logging:
    - Request/response correlation
    - Timing information
    - Status tracking
    - Structured data output
    - Error details
    """

    def __init__(self, debug_file: str | Path | None = None) -> None:
        super().__init__()
        self._debug_file = Path(debug_file) if debug_file else None
        self._request_start_times: dict[int, datetime] = {}

    def register_outgoing(self, request_id: int, method: str) -> asyncio.Future[Any]:
        """Register outgoing request with debug logging."""
        future = super().register_outgoing(request_id, method)

        # Track start time for duration calculation
        self._request_start_times[request_id] = datetime.now()
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="outgoing",
            event="register",
            request_id=request_id,
            method=method,
        )
        self._log_debug(entry)

        return future

    def resolve_outgoing(self, request_id: int, result: Any) -> None:
        """Resolve outgoing request with debug logging."""
        duration = self._calculate_duration(request_id)
        super().resolve_outgoing(request_id, result)
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="outgoing",
            event="resolve",
            request_id=request_id,
            method=self._get_method_for_request(request_id),
            result=result,
            duration_ms=duration,
        )
        self._log_debug(entry)
        self._cleanup_request_timing(request_id)

    def reject_outgoing(self, request_id: int, error: Any) -> None:
        """Reject outgoing request with debug logging."""
        duration = self._calculate_duration(request_id)
        super().reject_outgoing(request_id, error)
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="outgoing",
            event="reject",
            request_id=request_id,
            method=self._get_method_for_request(request_id),
            error=str(error),
            duration_ms=duration,
        )
        self._log_debug(entry)
        self._cleanup_request_timing(request_id)

    def reject_all_outgoing(self, error: Any) -> None:
        """Reject all outgoing requests with debug logging."""
        # Log for each pending request
        for request_id in list(self._outgoing.keys()):
            duration = self._calculate_duration(request_id)
            entry = DebugEntry(
                timestamp=datetime.now().isoformat(),
                direction="outgoing",
                event="reject",
                request_id=request_id,
                method=self._get_method_for_request(request_id),
                error=f"Connection error: {error}",
                duration_ms=duration,
            )
            self._log_debug(entry)

        super().reject_all_outgoing(error)
        self._request_start_times.clear()

    def begin_incoming(self, method: str, params: Any) -> IncomingMessage:
        """Begin processing incoming request with debug logging."""
        record = super().begin_incoming(method, params)
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="incoming",
            event="begin",
            request_id=None,
            method=method,
            params=params,
            status="pending",
        )
        self._log_debug(entry)
        return record

    def complete_incoming(self, record: IncomingMessage, result: Any) -> None:
        """Complete incoming request with debug logging."""
        super().complete_incoming(record, result)
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="incoming",
            event="complete",
            request_id=None,
            method=record.method,
            result=result,
            status="completed",
        )
        self._log_debug(entry)

    def fail_incoming(self, record: IncomingMessage, error: Any) -> None:
        """Fail incoming request with debug logging."""
        super().fail_incoming(record, error)
        entry = DebugEntry(
            timestamp=datetime.now().isoformat(),
            direction="incoming",
            event="fail",
            request_id=None,
            method=record.method,
            error=str(error),
            status="failed",
        )
        self._log_debug(entry)

    def _log_debug(self, entry: DebugEntry) -> None:
        """Write debug entry to file if configured."""
        if not self._debug_file:
            return

        try:
            # Convert to dict and filter out None values for cleaner output
            data = {k: v for k, v in asdict(entry).items() if v is not None}
            # Write as JSONL (one JSON object per line)
            with self._debug_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(data, separators=(",", ":")) + "\n")

        except Exception:
            # Don't let debug logging break the connection
            logging.exception("Failed to write debug entry")

    def _calculate_duration(self, request_id: int) -> float | None:
        """Calculate request duration in milliseconds."""
        start_time = self._request_start_times.get(request_id)
        if start_time:
            return (datetime.now() - start_time).total_seconds() * 1000
        return None

    def _cleanup_request_timing(self, request_id: int) -> None:
        """Clean up timing tracking for completed request."""
        self._request_start_times.pop(request_id, None)

    def _get_method_for_request(self, request_id: int) -> str:
        """Get method name for outgoing request."""
        record = self._outgoing.get(request_id)
        return record.method if record else "unknown"
