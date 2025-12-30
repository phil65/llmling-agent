"""OpenCode API models.

All models inherit from OpenCodeBaseModel which provides:
- populate_by_name=True for camelCase alias support
- by_alias=True serialization by default
"""

from agentpool_server.opencode_server.models.base import OpenCodeBaseModel
from agentpool_server.opencode_server.models.common import (
    TimeCreated,
    TimeCreatedUpdated,
)
from agentpool_server.opencode_server.models.app import (
    App,
    AppTimeInfo,
    HealthResponse,
    PathInfo,
    Project,
    ProjectTime,
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
    SessionForkRequest,
    SessionInitRequest,
    SessionRevert,
    SessionShare,
    SessionStatus,
    SessionUpdateRequest,
    Todo,
)
from agentpool_server.opencode_server.models.message import (
    AssistantMessage,
    CommandRequest,
    FilePartInput,
    MessagePath,
    MessageRequest,
    MessageTime,
    MessageWithParts,
    PartInput,
    ShellRequest,
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
    TimeStart,
    TimeStartEnd,
    TimeStartEndCompacted,
    TimeStartEndOptional,
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
from agentpool_server.opencode_server.models.pty import (
    PtyCreateRequest,
    PtyInfo,
    PtySize,
    PtyUpdateRequest,
)
from agentpool_server.opencode_server.models.events import (
    Event,
    MessageUpdatedEvent,
    MessageUpdatedEventProperties,
    PartUpdatedEvent,
    PartUpdatedEventProperties,
    ServerConnectedEvent,
    SessionCreatedEvent,
    SessionDeletedEvent,
    SessionDeletedProperties,
    SessionIdleEvent,
    SessionIdleProperties,
    SessionInfoProperties,
    SessionStatusEvent,
    SessionStatusProperties,
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
    "CommandRequest",
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
    "MessagePath",
    "MessageRequest",
    "MessageTime",
    "MessageUpdatedEvent",
    "MessageUpdatedEventProperties",
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
    "PartUpdatedEventProperties",
    "PathInfo",
    "Project",
    "ProjectTime",
    "Provider",
    "ProviderListResponse",
    "ProvidersResponse",
    # PTY
    "PtyCreateRequest",
    "PtyInfo",
    "PtySize",
    "PtyUpdateRequest",
    "ServerConnectedEvent",
    # Session
    "Session",
    "SessionCreateRequest",
    "SessionCreatedEvent",
    "SessionDeletedEvent",
    "SessionDeletedProperties",
    "SessionForkRequest",
    "SessionIdleEvent",
    "SessionIdleProperties",
    "SessionInfoProperties",
    "SessionInitRequest",
    "SessionRevert",
    "SessionShare",
    "SessionStatus",
    "SessionStatusEvent",
    "SessionStatusProperties",
    "SessionUpdateRequest",
    "SessionUpdatedEvent",
    "ShellRequest",
    "StepFinishPart",
    "StepStartPart",
    "Symbol",
    "TextPart",
    "TextPartInput",
    # Common
    "TimeCreated",
    "TimeCreatedUpdated",
    # Time types (from parts.py)
    "TimeStart",
    "TimeStartEnd",
    "TimeStartEndCompacted",
    "TimeStartEndOptional",
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
