"""Observability package."""

from __future__ import annotations

from agentpool.observability.observability_registry import ObservabilityRegistry

registry = ObservabilityRegistry()

__all__ = ["ObservabilityRegistry", "registry"]
