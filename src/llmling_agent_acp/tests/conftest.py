"""Test fixtures for ACP tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, Mock

import pytest

from acp import (
    Agent,
    DefaultACPClient,
    InitializeResponse,
    LoadSessionResponse,
    NewSessionResponse,
    PromptResponse,
)
from acp.schema import ClientCapabilities, FileSystemCapability, Implementation
from llmling_agent_acp.acp_agent import LLMlingACPAgent


if TYPE_CHECKING:
    from acp import (
        InitializeRequest,
        LoadSessionRequest,
        NewSessionRequest,
        PromptRequest,
    )
    from acp.schema import CancelNotification


class TestAgent(Agent):
    """Test agent implementation for ACP testing."""

    def __init__(self) -> None:
        self.prompts: list[PromptRequest] = []
        self.cancellations: list[str] = []
        self.ext_calls: list[tuple[str, dict]] = []
        self.ext_notes: list[tuple[str, dict]] = []

    async def initialize(self, params: InitializeRequest) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=params.protocol_version,
            agent_capabilities=None,
            agent_info=Implementation(name="test-agent", version="1.0"),
        )

    async def new_session(self, params: NewSessionRequest) -> NewSessionResponse:
        return NewSessionResponse(session_id="test-session-123")

    async def load_session(self, params: LoadSessionRequest) -> LoadSessionResponse:
        return LoadSessionResponse()

    async def authenticate(self, params) -> None:
        return None

    async def prompt(self, params: PromptRequest) -> PromptResponse:
        self.prompts.append(params)
        return PromptResponse(stop_reason="end_turn")

    async def cancel(self, params: CancelNotification) -> None:
        self.cancellations.append(params.session_id)

    async def set_session_mode(self, params):
        return {}

    async def set_session_model(self, params):
        return {}

    async def ext_method(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.ext_calls.append((method, params))
        return {"ok": True, "method": method}

    async def ext_notification(self, method: str, params: dict[str, Any]) -> None:
        self.ext_notes.append((method, params))


@pytest.fixture
def test_client() -> DefaultACPClient:
    """Create a fresh test client for each test."""
    return DefaultACPClient(allow_file_operations=True, use_real_files=False)


@pytest.fixture
def test_agent() -> TestAgent:
    """Create a fresh test agent for each test."""
    return TestAgent()


@pytest.fixture
def mock_connection():
    """Create a mock ACP connection."""
    return Mock()


@pytest.fixture
def mock_agent_pool():
    """Create a mock agent pool with a test agent."""
    from llmling_agent import Agent
    from llmling_agent.delegation import AgentPool

    # Create a simple test agent
    def simple_callback(message: str) -> str:
        return f"Test response: {message}"

    agent = Agent(name="test_agent", provider=simple_callback)
    pool = AgentPool[None]()
    pool.register("test_agent", agent)
    return pool


@pytest.fixture
def client_capabilities():
    """Create client capabilities for testing."""
    return ClientCapabilities(
        fs=FileSystemCapability(read_text_file=True, write_text_file=True),
        terminal=True,
    )


@pytest.fixture
def mock_acp_agent(mock_connection, mock_agent_pool, client_capabilities):
    """Create a mock ACP agent for testing."""
    return LLMlingACPAgent(
        connection=mock_connection,
        agent_pool=mock_agent_pool,
        available_models=[],
        session_support=True,
        file_access=True,
        terminal_access=True,
    )


@pytest.fixture
def mock_client():
    """Create mock ACP client."""
    client = AsyncMock()
    client.request_permission = AsyncMock()
    client.session_update = AsyncMock()
    client.read_text_file = AsyncMock()
    client.write_text_file = AsyncMock()
    return client
