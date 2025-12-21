"""Tests for agent injection and decorators."""

from __future__ import annotations

from typing import Any

import pytest

from agentpool.agents import Agent
from agentpool.running import NodeInjectionError, with_nodes


async def test_basic_injection(pool):
    """Test basic agent injection."""

    @with_nodes(pool)
    async def test_func(
        agent1: Agent[None] | None = None,
        agent2: Agent[None] | None = None,
        arg: str = "test",
    ) -> str:
        assert isinstance(agent1, Agent)
        assert isinstance(agent2, Agent)
        assert agent1.name == "agent1"
        assert agent2.name == "agent2"
        return arg

    result = await test_func()
    assert result == "test"


async def test_missing_agent(pool):
    """Test error when agent not in pool."""

    @with_nodes(pool)
    async def test_func(missing_agent: Agent[None] | None = None) -> str:
        return "unreachable"

    with pytest.raises(NodeInjectionError) as exc:
        await test_func()
    assert "No node named 'missing_agent'" in str(exc.value)
    assert "Available nodes: agent1, agent2" in str(exc.value)


async def test_duplicate_parameter(pool):
    """Test error when agent parameter provided explicitly."""

    @with_nodes(pool)
    async def test_func(agent1: Agent[None] | None = None) -> str:
        return "unreachable"

    dummy_agent = pool.get_agent("agent1")
    with pytest.raises(NodeInjectionError) as exc:
        await test_func(agent1=dummy_agent)
    assert "Parameter already provided" in str(exc.value)


async def test_non_agent_parameter(pool):
    """Test regular parameters are ignored."""

    @with_nodes(pool)
    async def test_func(
        agent1: Agent[None] | None = None,
        normal: str = "test",
        no_hint: Any = 123,
        *,
        kwonly: int = 456,
    ) -> str:
        assert isinstance(agent1, Agent)
        assert normal == "test"
        assert no_hint == 123  # noqa: PLR2004
        assert kwonly == 456  # noqa: PLR2004
        return "ok"

    result = await test_func()
    assert result == "ok"


async def test_wrapper_usage(pool):
    """Test using as wrapper instead of decorator."""

    async def test_func(agent1: Agent[None] | None = None, arg: str = "test") -> str:
        assert isinstance(agent1, Agent)
        return arg

    wrapped = with_nodes(pool)(test_func)
    result = await wrapped()
    assert result == "test"


async def test_agent_functionality(pool):
    """Test injected agents are fully functional."""

    @with_nodes(pool)
    async def test_func(
        agent1: Agent[None] | None = None,
        agent2: Agent[None] | None = None,
    ) -> list[str]:
        assert agent1 is not None
        assert agent2 is not None
        result1 = await agent1.run("test1")
        result2 = await agent2.run("test2")
        return [result1.content, result2.content]

    results = await test_func()
    assert len(results) == 2  # noqa: PLR2004
    assert all(isinstance(r, str) for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
