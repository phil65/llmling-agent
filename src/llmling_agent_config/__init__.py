"""Core data models for LLMling agent."""

from __future__ import annotations

from llmling_agent_config.resources import ResourceInfo
from llmling_agent_config.forward_targets import ForwardingTarget
from llmling_agent_config.session import SessionQuery
from llmling_agent_config.teams import TeamConfig
from llmling_agent_config.mcp_server import (
    BaseMCPServerConfig,
    StdioMCPServerConfig,
    StreamableHTTPMCPServerConfig,
    MCPServerConfig,
    SSEMCPServerConfig,
)
from llmling_agent_config.event_handlers import (
    BaseEventHandlerConfig,
    StdoutEventHandlerConfig,
    CallbackEventHandlerConfig,
    EventHandlerConfig,
    resolve_handler_configs,
)
from llmling_agent_config.hooks import (
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
