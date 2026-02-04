"""Codex app-server Python adapter.

Provides programmatic control over Codex via the app-server JSON-RPC protocol.

Example:
    async with CodexClient() as client:
        response = await client.thread_start(cwd="/path/to/project")
        async for event in client.turn_stream(response.thread.id, "Help me refactor"):
            if event.event_type == "item/agentMessage/delta":
                print(event.data.text, end="", flush=True)
"""

from codex_adapter.client import CodexClient
from codex_adapter.codex_types import (
    ApprovalPolicy,
    CodexTurn,
    HttpMcpServer,
    ItemStatus,
    ItemType,
    McpServerConfig,
    ModelProvider,
    ReasoningEffort,
    SandboxMode,
    StdioMcpServer,
    TurnStatus,
)
from codex_adapter.events import (
    AgentMessageDeltaEvent,
    CodexEvent,
    CommandExecutionOutputDeltaEvent,
    EventType,
    FileChangeOutputDeltaEvent,
    ItemCompletedEvent,
    ItemStartedEvent,
    ReasoningTextDeltaEvent,
    ThreadCompactedEvent,
    ThreadStartedEvent,
    TurnCompletedEvent,
    TurnErrorEvent,
    TurnPlanUpdatedEvent,
    TurnStartedEvent,
    get_text_delta,
    is_completed_event,
    is_delta_event,
    is_error_event,
    parse_codex_event,
)
from codex_adapter.exceptions import CodexError, CodexProcessError, CodexRequestError
from codex_adapter.models import (
    AgentMessageDeltaData,
    CommandExecResponse,
    CommandExecutionOutputDeltaData,
    EventData,
    ImageInputItem,
    LocalImageInputItem,
    ModelData,
    ReasoningTextDeltaData,
    SkillData,
    SkillInputItem,
    TextInputItem,
    ThreadData,
    ThreadListResponse,
    ThreadResponse,
    ThreadRollbackResponse,
    ThreadStartedData,
    TurnCompletedData,
    TurnErrorData,
    TurnInputItem,
    TurnStartedData,
    Usage,
)

__all__ = [
    # Event classes
    "AgentMessageDeltaEvent",
    "CodexEvent",
    "CommandExecutionOutputDeltaEvent",
    "EventType",
    "FileChangeOutputDeltaEvent",
    "ItemCompletedEvent",
    "ItemStartedEvent",
    "ReasoningTextDeltaEvent",
    "ThreadCompactedEvent",
    "ThreadStartedEvent",
    "TurnCompletedEvent",
    "TurnErrorEvent",
    "TurnPlanUpdatedEvent",
    "TurnStartedEvent",
    # Event helper functions
    "get_text_delta",
    "is_completed_event",
    "is_delta_event",
    "is_error_event",
    "parse_codex_event",
    # Data models
    "AgentMessageDeltaData",
    "CommandExecutionOutputDeltaData",
    "EventData",
    "ImageInputItem",
    "LocalImageInputItem",
    "ModelData",
    "ReasoningTextDeltaData",
    "SkillData",
    "SkillInputItem",
    "TextInputItem",
    "ThreadData",
    "ThreadListResponse",
    "ThreadResponse",
    "ThreadRollbackResponse",
    "ThreadStartedData",
    "TurnCompletedData",
    "TurnErrorData",
    "TurnInputItem",
    "TurnStartedData",
    "Usage",
    # Client and exceptions
    "CodexClient",
    "CodexError",
    "CodexProcessError",
    "CodexRequestError",
    # Response models
    "CommandExecResponse",
    # Types
    "ApprovalPolicy",
    "CodexTurn",
    "HttpMcpServer",
    "ItemStatus",
    "ItemType",
    "McpServerConfig",
    "ModelProvider",
    "ReasoningEffort",
    "SandboxMode",
    "StdioMcpServer",
    "TurnStatus",
]
