"""LLMling-Agent: main package.

Pydantic-AI based Multi-Agent Framework with YAML-based Agents, Teams, Workflows &
Extended ACP / AGUI integration.
"""

from __future__ import annotations

from importlib.metadata import version

from llmling_agent.models.agents import NativeAgentConfig
from llmling_agent.models.manifest import AgentsManifest

# Builtin toolsets imports removed to avoid circular dependency
# Import them directly from llmling_agent_toolsets.builtin when needed
from llmling_agent.agents import Agent, AgentContext
from llmling_agent.delegation import AgentPool, Team, TeamRun, BaseTeam
from dotenv import load_dotenv
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools import Tool, ToolCallInfo
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.testing import acp_test_session
from pydantic_ai import (
    AudioUrl,
    BinaryContent,
    BinaryImage,
    DocumentUrl,
    ImageUrl,
    VideoUrl,
)

__version__ = version("llmling-agent")
__title__ = "LLMling-Agent"
__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2025 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/llmling-agent"

load_dotenv()

__all__ = [
    "Agent",
    "AgentContext",
    "AgentPool",
    "AgentsManifest",
    "AudioUrl",
    "BaseTeam",
    "BinaryContent",
    "BinaryImage",
    "ChatMessage",
    "DocumentUrl",
    "ImageUrl",
    "MessageNode",
    "NativeAgentConfig",
    "Team",
    "TeamRun",
    "Tool",
    "ToolCallInfo",
    "VideoUrl",
    "__version__",
    "acp_test_session",
]
