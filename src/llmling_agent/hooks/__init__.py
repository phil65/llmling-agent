"""Runtime hook classes for agent lifecycle events."""

from __future__ import annotations

from llmling_agent.hooks.agent_hooks import AgentHooks
from llmling_agent.hooks.base import Hook, HookEvent, HookInput, HookResult
from llmling_agent.hooks.callable import CallableHook
from llmling_agent.hooks.command import CommandHook
from llmling_agent.hooks.prompt import PromptHook

__all__ = [
    "AgentHooks",
    "CallableHook",
    "CommandHook",
    "Hook",
    "HookEvent",
    "HookInput",
    "HookResult",
    "PromptHook",
]
