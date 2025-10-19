"""Simplified observability registry using Logfire with single backend."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import logfire

from llmling_agent.log import get_logger


if TYPE_CHECKING:
    from llmling_agent_config.observability import (
        BaseObservabilityConfig,
        ObservabilityConfig,
    )

logger = get_logger(__name__)


class ObservabilityRegistry:
    """Simplified registry that configures Logfire for single backend export."""

    def __init__(self):
        self._configured = False

    def configure_observability(self, observability_config: ObservabilityConfig):
        """Configure Logfire for single backend export.

        Args:
            observability_config: Configuration for observability
        """
        if not observability_config.enabled or not observability_config.provider:
            logger.debug("Observability disabled or no provider configured")
            return

        if self._configured:
            logger.warning("Observability already configured, skipping")
            return

        config = observability_config.provider
        if not config.enabled:
            logger.debug("Provider %s is disabled", config.type)
            return

        _setup_otel_environment(config)  # Configure OTEL env variables based on provider
        logfire.configure(
            service_name=config.service_name,
            environment=config.environment,
            send_to_logfire=(config.type == "logfire"),
        )
        logfire.instrument_pydantic_ai()
        logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])

        self._configured = True
        logger.info("Configured observability with %s backend", config.type)


def _setup_otel_environment(config: BaseObservabilityConfig):
    """Set up OTEL environment variables for the configured backend."""
    # Get endpoint and headers from config
    endpoint = getattr(config, "_endpoint", getattr(config, "endpoint", None))
    headers = getattr(config, "_headers", getattr(config, "headers", {}))

    if not endpoint:
        logger.warning("No endpoint found for provider %s", config.type)
        return

    # Set standard OTEL environment variables
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = config.protocol

    # Set headers if available
    if headers:
        header_str = ",".join(f"{k}={v}" for k, v in headers.items())
        os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = header_str

    # Set resource attributes
    resource_attrs = []
    if config.service_name:
        resource_attrs.append(f"service.name={config.service_name}")
    if config.environment:
        resource_attrs.append(f"deployment.environment.name={config.environment}")

    if resource_attrs:
        os.environ["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)

    msg = "Set OTEL environment: endpoint=%s, protocol=%s, headers=%s"
    logger.debug(msg, endpoint, config.protocol, bool(headers))


# Global registry instance
registry = ObservabilityRegistry()
