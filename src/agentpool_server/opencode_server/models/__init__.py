"""OpenCode API models.

All models inherit from OpenCodeBaseModel which provides:
- populate_by_name=True for camelCase alias support
- by_alias=True serialization by default
"""

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import (
    TimeCreated,
    TimeCreatedUpdated,
    TimeStartEnd,
)
from agentpool_server.opencode_server.models.app import (
    App,
    AppTimeInfo,
    HealthResponse,
    PathInfo,
    Project,
    VcsInfo,
)
from agentpool_server.opencode_server.models.provider import (
    Model,
    ModelCost,
    ModelLimit,
    Mode,
    ModeModel,
    Provider,
    ProviderListResponse,
    ProvidersResponse,
)
from agentpool_server.opencode_server.models.session import (
    Session,
    SessionCreateRequest,
    SessionRevert,
    SessionShare,
    SessionStatus,
    SessionUpdateRequest,
    Todo,
)
from agentpool_server.opencode_server.models.message import (
    AssistantMessage,
    FilePartInput,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    PartInput,
    TextPartInput,
    Tokens,
    TokensCache,
    UserMessage,
)
from agentpool_server.opencode_server.models.parts import (
    FilePart,
    Part,
    StepFinishPart,
    StepStartPart,
    TextPart,
    ToolPart,
    ToolState,
    ToolStateCompleted,
    ToolStateError,
    ToolStatePending,
    ToolStateRunning,
)
from agentpool_server.opencode_server.models.file import (
    FileContent,
    FileNode,
    FileStatus,
    FindMatch,
    Symbol,
)
from agentpool_server.opencode_server.models.agent import (
    Agent,
    Command,
)
from agentpool_server.opencode_server.models.events import (
    Event,
    MessageCreatedEvent,
    MessageUpdatedEvent,
    PartUpdatedEvent,
    ServerConnectedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionUpdatedEvent,
)
from agentpool_server.opencode_server.models.mcp import (
    LogRequest,
    MCPStatus,
)
from agentpool_server.opencode_server.models.config import (
    Config,
)

__all__ = [
    # Agent
    "Agent",
    # App
    "App",
    "AppTimeInfo",
    # Message
    "AssistantMessage",
    "Command",
    # Config
    "Config",
    # Events
    "Event",
    # File
    "FileContent",
    "FileNode",
    # Parts
    "FilePart",
    "FilePartInput",
    "FileStatus",
    "FindMatch",
    "HealthResponse",
    # MCP
    "LogRequest",
    "MCPStatus",
    "MessageCreatedEvent",
    "MessagePath",
    "MessageRequest",
    "MessageTime",
    "MessageUpdatedEvent",
    "MessageWithParts",
    "Mode",
    "ModeModel",
    # Provider
    "Model",
    "ModelCost",
    "ModelLimit",
    # Base
    "OpenCodeBaseModel",
    "Part",
    "PartInput",
    "PartUpdatedEvent",
    "PathInfo",
    "Project",
    "Provider",
    "ProviderListResponse",
    "ProvidersResponse",
    "ServerConnectedEvent",
    # Session
    "Session",
    "SessionCreateRequest",
    "SessionCreatedEvent",
    "SessionDeletedEvent",
    "SessionRevert",
    "SessionShare",
    "SessionStatus",
    "SessionUpdateRequest",
    "SessionUpdatedEvent",
    "StepFinishPart",
    "StepStartPart",
    "Symbol",
    "TextPart",
    "TextPartInput",
    # Common
    "TimeCreated",
    "TimeCreatedUpdated",
    "TimeStartEnd",
    "Todo",
    "Tokens",
    "TokensCache",
    "ToolPart",
    "ToolState",
    "ToolStateCompleted",
    "ToolStateError",
    "ToolStatePending",
    "ToolStateRunning",
    "UserMessage",
    "VcsInfo",
]
