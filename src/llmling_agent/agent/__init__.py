"""CLI commands for llmling-agent."""

from __future__ import annotations

from llmling_agent.agent.agent import Agent
from llmling_agent.agent.agui_agent import AGUIAgent
from llmling_agent.agent.events import (
    detailed_print_handler,
    resolve_event_handlers,
    simple_print_handler,
)
from llmling_agent.agent.context import AgentContext
from llmling_agent.agent.interactions import Interactions
from llmling_agent.agent.slashed_agent import SlashedAgent
from llmling_agent.agent.sys_prompts import SystemPrompts


__all__ = [
    "AGUIAgent",
    "Agent",
    "AgentContext",
    "Interactions",
    "SlashedAgent",
    "SystemPrompts",
    "detailed_print_handler",
    "resolve_event_handlers",
    "simple_print_handler",
]
