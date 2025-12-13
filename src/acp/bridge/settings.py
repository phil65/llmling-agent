"""Configuration settings for ACP Bridge."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BridgeSettings:
    """Settings for the ACP Bridge server."""

    host: str = "127.0.0.1"
    port: int = 8080
    log_level: str = "INFO"
    allow_origins: list[str] | None = None
