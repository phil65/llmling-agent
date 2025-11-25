"""Schema definitions for the ACP protocol."""

from acp.schema.agent_plan import PlanEntry, PlanEntryPriority, PlanEntryStatus
from acp.schema.agent_requests import (
    AgentRequest,
    CreateTerminalRequest,
    KillTerminalCommandRequest,
    ReadTextFileRequest,
    ReleaseTerminalRequest,
    RequestPermissionRequest,
    TerminalOutputRequest,
    WaitForTerminalExitRequest,
    WriteTextFileRequest,
)
from acp.schema.agent_responses import (
    AgentResponse,
    AuthenticateResponse,
    CustomResponse,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    ListSessionsResponse,
    PromptResponse,
    SetSessionModeResponse,
    SetSessionModelResponse,
    StopReason,
)
from acp.schema.capabilities import (
    AgentCapabilities,
    ClientCapabilities,
    FileSystemCapability,
    McpCapabilities,
    PromptCapabilities,
    SessionCapabilities,
    SessionListCapabilities,
)
from acp.schema.client_requests import (
    AuthenticateRequest,
    ClientRequest,
    CustomRequest,
    InitializeRequest,
    ListSessionsRequest,
    LoadSessionRequest,
    NewSessionRequest,
    PromptRequest,
    SetSessionModeRequest,
    SetSessionModelRequest,
)
from acp.schema.client_responses import (
    ClientResponse,
    CreateTerminalResponse,
    KillTerminalCommandResponse,
    ReadTextFileResponse,
    ReleaseTerminalResponse,
    RequestPermissionResponse,
    TerminalOutputResponse,
    WaitForTerminalExitResponse,
    WriteTextFileResponse,
)
from acp.schema.common import AuthMethod, EnvVariable, Implementation
from acp.schema.content_blocks import (
    Annotations,
    AudioContentBlock,
    Audience,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResourceContentBlock,
    ImageContentBlock,
    ResourceContentBlock,
    TextContentBlock,
    TextResourceContents,
)
from acp.schema.mcp import (
    HttpHeader,
    HttpMcpServer,
    McpServer,
    SseMcpServer,
    StdioMcpServer,
)
from acp.schema.messages import AgentMethod, ClientMethod
from acp.schema.notifications import (
    CancelNotification,
    ExtNotification,
    SessionNotification,
)
from acp.schema.session_state import (
    ModelInfo,
    SessionInfo,
    SessionMode,
    SessionModeState,
    SessionModelState,
)
from acp.schema.slash_commands import (
    AvailableCommand,
    AvailableCommandInput,
    CommandInputHint,
)
from acp.schema.terminal import TerminalExitStatus
from acp.schema.tool_call import (
    AllowedOutcome,
    ContentToolCallContent,
    DeniedOutcome,
    FileEditToolCallContent,
    PermissionKind,
    PermissionOption,
    TerminalToolCallContent,
    ToolCall,
    ToolCallContent,
    ToolCallKind,
    ToolCallLocation,
    ToolCallStatus,
)
from acp.schema.session_updates import (
    AgentMessageChunk,
    AgentPlanUpdate,
    AgentThoughtChunk,
    AvailableCommandsUpdate,
    CurrentModeUpdate,
    SessionUpdate,
    ToolCallProgress,
    ToolCallStart,
    UserMessageChunk,
)

PROTOCOL_VERSION = 1

__all__ = [
    # Protocol version
    "PROTOCOL_VERSION",
    # Agent capabilities
    "AgentCapabilities",
    # Session updates
    "AgentMessageChunk",
    # Messages/Methods
    "AgentMethod",
    "AgentPlanUpdate",
    # Agent requests (agent -> client)
    "AgentRequest",
    # Agent responses
    "AgentResponse",
    "AgentThoughtChunk",
    # Tool calls
    "AllowedOutcome",
    # Content blocks
    "Annotations",
    "Audience",
    "AudioContentBlock",
    # Common types
    "AuthMethod",
    # Client requests (client -> agent)
    "AuthenticateRequest",
    "AuthenticateResponse",
    # Slash commands
    "AvailableCommand",
    "AvailableCommandInput",
    "AvailableCommandsUpdate",
    "BlobResourceContents",
    # Notifications
    "CancelNotification",
    "ClientCapabilities",
    "ClientMethod",
    "ClientRequest",
    # Client responses
    "ClientResponse",
    "CommandInputHint",
    "ContentBlock",
    "ContentToolCallContent",
    "CreateTerminalRequest",
    "CreateTerminalResponse",
    "CurrentModeUpdate",
    "CustomRequest",
    "CustomResponse",
    "DeniedOutcome",
    "EmbeddedResourceContentBlock",
    "EnvVariable",
    "ExtNotification",
    "FileEditToolCallContent",
    "FileSystemCapability",
    # MCP servers
    "HttpHeader",
    "HttpMcpServer",
    "ImageContentBlock",
    "Implementation",
    "InitializeRequest",
    "InitializeResponse",
    "KillTerminalCommandRequest",
    "KillTerminalCommandResponse",
    "ListSessionsRequest",
    "ListSessionsResponse",
    "LoadSessionRequest",
    "LoadSessionResponse",
    "McpCapabilities",
    "McpServer",
    # Session state
    "ModelInfo",
    "NewSessionRequest",
    "NewSessionResponse",
    "PermissionKind",
    "PermissionOption",
    # Plan
    "PlanEntry",
    "PlanEntryPriority",
    "PlanEntryStatus",
    "PromptCapabilities",
    "PromptRequest",
    "PromptResponse",
    "ReadTextFileRequest",
    "ReadTextFileResponse",
    "ReleaseTerminalRequest",
    "ReleaseTerminalResponse",
    "RequestPermissionRequest",
    "RequestPermissionResponse",
    "ResourceContentBlock",
    "SessionCapabilities",
    "SessionInfo",
    "SessionListCapabilities",
    "SessionMode",
    "SessionModeState",
    "SessionModelState",
    "SessionNotification",
    "SessionUpdate",
    "SetSessionModeRequest",
    "SetSessionModeResponse",
    "SetSessionModelRequest",
    "SetSessionModelResponse",
    "SseMcpServer",
    "StdioMcpServer",
    "StopReason",
    # Terminal
    "TerminalExitStatus",
    "TerminalOutputRequest",
    "TerminalOutputResponse",
    "TerminalToolCallContent",
    "TextContentBlock",
    "TextResourceContents",
    "ToolCall",
    "ToolCallContent",
    "ToolCallKind",
    "ToolCallLocation",
    "ToolCallProgress",
    "ToolCallStart",
    "ToolCallStatus",
    "UserMessageChunk",
    "WaitForTerminalExitRequest",
    "WaitForTerminalExitResponse",
    "WriteTextFileRequest",
    "WriteTextFileResponse",
]
