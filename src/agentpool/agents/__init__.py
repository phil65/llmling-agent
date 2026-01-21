"""CLI commands for agentpool."""

from __future__ import annotations

from agentpool.agents.base_agent import BaseAgentKwargs, get_agent_class
from agentpool.agents.native_agent import Agent
from agentpool.agents.agui_agent import AGUIAgent
from agentpool.agents.acp_agent import ACPAgent
from agentpool.agents.claude_code_agent import ClaudeCodeAgent
from agentpool.agents.codex_agent import CodexAgent
from agentpool.agents.events import (
    detailed_print_handler,
    resolve_event_handlers,
    simple_print_handler,
)
from agentpool.agents.context import AgentContext
from agentpool.agents.interactions import Interactions
from agentpool.agents.sys_prompts import SystemPrompts


__all__ = [
    "ACPAgent",
    "AGUIAgent",
    "Agent",
    "AgentContext",
    "BaseAgentKwargs",
    "ClaudeCodeAgent",
    "CodexAgent",
    "Interactions",
    "SystemPrompts",
    "detailed_print_handler",
    "get_agent_class",
    "resolve_event_handlers",
    "simple_print_handler",
]
