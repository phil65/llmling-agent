"""Agent Client protocol (ACP) implementation."""

from acp.client import DefaultACPClient, ClientSideConnection
from acp.agent import AgentSideConnection
from acp import cli
from acp.core import create_session_model_state
from acp.filesystem import ACPFileSystem, ACPFileSystemSync, ACPPath
from acp.agent.protocol import (
    Agent,
    AgentExtensibilityCapability,
    BaseAgent,
    SessionModeCapability,
    SessionModelCapability,
    SessionPersistenceCapability,
)
from acp.client.protocol import (
    BaseClient,
    Client,
    ExtensibilityCapability,
    FileSystemCapability,
    TerminalCapability,
)
from acp.terminal_handle import TerminalHandle
from acp.schema import (
    AuthenticateRequest,
    AuthenticateResponse,
    CancelNotification,
    CreateTerminalRequest,
    CreateTerminalResponse,
    InitializeRequest,
    InitializeResponse,
    KillTerminalCommandRequest,
    KillTerminalCommandResponse,
    LoadSessionRequest,
    LoadSessionResponse,
    ModelInfo,
    NewSessionRequest,
    NewSessionResponse,
    PromptRequest,
    PromptResponse,
    ReadTextFileRequest,
    ReadTextFileResponse,
    ReleaseTerminalRequest,
    ReleaseTerminalResponse,
    RequestPermissionRequest,
    RequestPermissionResponse,
    SessionModelState,
    SessionNotification,
    SetSessionModelRequest,
    SetSessionModelResponse,
    SetSessionModeRequest,
    SetSessionModeResponse,
    TerminalOutputRequest,
    TerminalOutputResponse,
    WaitForTerminalExitRequest,
    WaitForTerminalExitResponse,
    WriteTextFileRequest,
    WriteTextFileResponse,
)
from acp.stdio import stdio_streams
from acp.exceptions import RequestError
from acp.meta import PROTOCOL_VERSION, AgentMethod, ClientMethod

__version__ = "0.0.1"

__all__ = [  # noqa: RUF022
    # constants
    "PROTOCOL_VERSION",
    # literal types
    "AgentMethod",
    "ClientMethod",
    # types
    "InitializeRequest",
    "InitializeResponse",
    "NewSessionRequest",
    "NewSessionResponse",
    "LoadSessionRequest",
    "LoadSessionResponse",
    "AuthenticateRequest",
    "AuthenticateResponse",
    "PromptRequest",
    "PromptResponse",
    "WriteTextFileRequest",
    "WriteTextFileResponse",
    "ReadTextFileRequest",
    "ReadTextFileResponse",
    "RequestPermissionRequest",
    "RequestPermissionResponse",
    "CancelNotification",
    "SessionNotification",
    "SetSessionModeRequest",
    "SetSessionModeResponse",
    # model types
    "ModelInfo",
    "SessionModelState",
    "SetSessionModelRequest",
    "SetSessionModelResponse",
    # terminal types
    "CreateTerminalRequest",
    "CreateTerminalResponse",
    "TerminalOutputRequest",
    "TerminalOutputResponse",
    "WaitForTerminalExitRequest",
    "WaitForTerminalExitResponse",
    "KillTerminalCommandRequest",
    "KillTerminalCommandResponse",
    "ReleaseTerminalRequest",
    "ReleaseTerminalResponse",
    # core
    "AgentSideConnection",
    "ClientSideConnection",
    "RequestError",
    "Agent",
    "Client",
    "DefaultACPClient",
    "TerminalHandle",
    # split protocols
    "BaseAgent",
    "SessionPersistenceCapability",
    "SessionModeCapability",
    "SessionModelCapability",
    "AgentExtensibilityCapability",
    "BaseClient",
    "FileSystemCapability",
    "TerminalCapability",
    "ExtensibilityCapability",
    # utilities
    "create_session_model_state",
    # stdio helper
    "stdio_streams",
    # filesystem
    "ACPFileSystem",
    "ACPFileSystemSync",
    "ACPPath",
    # cli
    "cli",
]
