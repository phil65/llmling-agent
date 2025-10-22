"""LLMling-Agent: main package.

A pydantic-ai based Agent with LLMling backend.
"""

from __future__ import annotations

from importlib.metadata import version

from llmling_agent.models import AgentsManifest, AgentConfig
from llmling_agent_toolsets.builtin import (
    AgentManagementTools,
    FileAccessTools,
    CodeExecutionTools,
    ProcessManagementTools,
    ResourceAccessTools,
    ToolManagementTools,
    UserInteractionTools,
    HistoryTools,
    IntegrationTools,
)
from llmling_agent.agent import Agent, StructuredAgent, AnyAgent, AgentContext
from llmling_agent.delegation import AgentPool, Team, TeamRun, BaseTeam
from dotenv import load_dotenv
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools import Tool, ToolCallInfo
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.models.content import (
    PDFURLContent,
    PDFBase64Content,
    ImageBase64Content,
    ImageURLContent,
    AudioURLContent,
    AudioBase64Content,
    VideoURLContent,
)

__version__ = version("llmling-agent")
__title__ = "LLMling-Agent"
__author__ = "Philipp Temminghoff"
__author_email__ = "philipptemminghoff@googlemail.com"
__copyright__ = "Copyright (c) 2024 Philipp Temminghoff"
__license__ = "MIT"
__url__ = "https://github.com/phil65/llmling-agent"

load_dotenv()

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentContext",
    "AgentManagementTools",
    "AgentPool",
    "AgentsManifest",
    "AnyAgent",
    "AudioBase64Content",
    "AudioURLContent",
    "BaseTeam",
    "ChatMessage",
    "CodeExecutionTools",
    "FileAccessTools",
    "HistoryTools",
    "ImageBase64Content",
    "ImageURLContent",
    "IntegrationTools",
    "MessageNode",
    "PDFBase64Content",
    "PDFURLContent",
    "ProcessManagementTools",
    "ResourceAccessTools",
    "StructuredAgent",
    "Team",
    "TeamRun",
    "Tool",
    "ToolCallInfo",
    "ToolManagementTools",
    "UserInteractionTools",
    "VideoURLContent",
    "__version__",
]
