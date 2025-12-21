"""Core data models for AgentPool."""

from __future__ import annotations

from agentpool_config.resources import ResourceInfo
from agentpool_config.forward_targets import ForwardingTarget
from agentpool_config.session import SessionQuery
from agentpool_config.teams import TeamConfig
from agentpool_config.mcp_server import (
    BaseMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
    MCPServerConfig,
    SSEMCPServerConfig,
)
from agentpool_config.event_handlers import (
    BaseEventHandlerConfig,
    StdoutEventHandlerConfig,
    CallbackEventHandlerConfig,
    EventHandlerConfig,
    resolve_handler_configs,
)
from agentpool_config.hooks import (
    BaseHookConfig,
    CallableHookConfig,
    CommandHookConfig,
    HookConfig,
    HooksConfig,
    PromptHookConfig,
)

__all__ = [
    "BaseEventHandlerConfig",
    "BaseHookConfig",
    "BaseMCPServerConfig",
    "CallableHookConfig",
    "CallbackEventHandlerConfig",
    "CommandHookConfig",
    "EventHandlerConfig",
    "ForwardingTarget",
    "HookConfig",
    "HooksConfig",
    "MCPServerConfig",
    "PromptHookConfig",
    "ResourceInfo",
    "SSEMCPServerConfig",
    "SessionQuery",
    "StdioMCPServerConfig",
    "StdoutEventHandlerConfig",
    "StreamableHTTPMCPServerConfig",
    "TeamConfig",
    "resolve_handler_configs",
]
