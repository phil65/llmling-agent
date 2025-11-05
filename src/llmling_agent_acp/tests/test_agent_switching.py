"""Test agent pool mode switching functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from acp.schema import SessionMode, SessionModeState
from llmling_agent.models.manifest import AgentConfig, AgentsManifest
from llmling_agent_acp import ACPServer, ACPSession


if TYPE_CHECKING:
    from unittest.mock import AsyncMock

    from llmling_agent.delegation.pool import AgentPool


@pytest.fixture
def test_manifest():
    """Create test manifest with multiple agents."""
    agents = {
        "coding-agent": AgentConfig(
            name="Coding Assistant",
            model="test",
            system_prompts=["You are a coding expert."],
        ),
        "research-agent": AgentConfig(
            name="Research Assistant",
            model="test",
            system_prompts=["You are a research expert."],
        ),
        "writing-agent": AgentConfig(
            name="Writing Assistant",
            model="test",
            system_prompts=["You are a writing expert."],
        ),
    }
    return AgentsManifest(agents=agents)


@pytest.fixture
async def agent_pool(test_manifest: AgentsManifest):
    """Create test agent pool."""
    async with test_manifest.pool as pool:
        yield pool


def test_server_agent_pool_setup(agent_pool: AgentPool[Any]):
    """Test that server correctly stores agent pool."""
    server = ACPServer(agent_pool=agent_pool)

    assert server.agent_pool is agent_pool
    agent_names = list(server.agent_pool.agents.keys())
    assert len(agent_names) == 3  # noqa: PLR2004
    assert "coding-agent" in agent_names
    assert "research-agent" in agent_names
    assert "writing-agent" in agent_names


def test_session_agent_property(
    agent_pool: AgentPool[Any],
    mock_client: AsyncMock,
    mock_acp_agent,
    client_capabilities,
):
    """Test session agent property returns current agent."""
    session = ACPSession(
        session_id="test-session",
        agent_pool=agent_pool,
        current_agent_name="coding-agent",
        cwd="/test",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=client_capabilities,
    )

    # Should return coding agent initially
    current_agent = session.agent
    assert current_agent.name == "coding-agent"


async def test_switch_active_agent(
    agent_pool: AgentPool[Any],
    mock_client: AsyncMock,
    mock_acp_agent,
    client_capabilities,
):
    """Test switching active agent in session."""
    session = ACPSession(
        session_id="test-session",
        agent_pool=agent_pool,
        current_agent_name="coding-agent",
        cwd="/test",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=client_capabilities,
    )

    # Initially should be coding agent
    assert session.current_agent_name == "coding-agent"
    assert session.agent.name == "coding-agent"

    # Switch to research agent
    await session.switch_active_agent("research-agent")
    assert session.current_agent_name == "research-agent"
    assert session.agent.name == "research-agent"


async def test_switch_to_invalid_agent(
    agent_pool: AgentPool[Any],
    mock_client: AsyncMock,
    mock_acp_agent,
    client_capabilities,
):
    """Test switching to non-existent agent raises error."""
    session = ACPSession(
        session_id="test-session",
        agent_pool=agent_pool,
        current_agent_name="coding-agent",
        cwd="/test",
        client=mock_client,
        acp_agent=mock_acp_agent,
        client_capabilities=client_capabilities,
    )

    with pytest.raises(ValueError, match="Agent 'invalid-agent' not found"):
        await session.switch_active_agent("invalid-agent")


def test_session_modes_from_agent_pool(agent_pool: AgentPool[Any]):
    """Test that session modes are correctly generated from agent pool."""
    # Simulate what newSession does
    available_modes = [
        SessionMode(id=name, name=name, description=f"Switch to {name} agent")
        for name in list(agent_pool.agents.keys())
    ]

    modes = SessionModeState(
        current_mode_id="coding-agent", available_modes=available_modes
    )

    assert len(modes.available_modes) == 3  # noqa: PLR2004
    assert modes.current_mode_id == "coding-agent"

    # Check mode details
    mode_ids = [mode.id for mode in modes.available_modes]
    mode_names = [mode.name for mode in modes.available_modes]

    assert "coding-agent" in mode_ids
    assert "research-agent" in mode_ids
    assert "writing-agent" in mode_ids

    assert "coding-agent" in mode_names
    assert "research-agent" in mode_names
    assert "writing-agent" in mode_names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
