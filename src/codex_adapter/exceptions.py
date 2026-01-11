"""Codex adapter exceptions."""

from __future__ import annotations

from typing import Any


class CodexError(Exception):
    """Base exception for Codex adapter errors."""


class CodexProcessError(CodexError):
    """Error starting or communicating with the Codex app-server process."""


class CodexRequestError(CodexError):
    """Error from a Codex app-server request (JSON-RPC error response)."""

    def __init__(self, code: int, message: str, data: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data or {}

    def __str__(self) -> str:
        if self.data:
            return f"[{self.code}] {self.message}: {self.data}"
        return f"[{self.code}] {self.message}"
