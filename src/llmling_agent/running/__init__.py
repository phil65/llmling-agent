"""Decorator auto-injection / run package."""

from __future__ import annotations

from llmling_agent.running.decorators import with_nodes
from llmling_agent.running.discovery import NodeFunction, node_function
from llmling_agent.running.executor import execute_single, execute_functions
from llmling_agent.running.run_nodes import run_nodes_async, run_nodes
from llmling_agent.running.injection import NodeInjectionError

__all__ = [
    "NodeFunction",
    "NodeInjectionError",
    "execute_functions",
    "execute_single",
    "node_function",
    "run_nodes",
    "run_nodes_async",
    "with_nodes",
]
