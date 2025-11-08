from __future__ import annotations

from pydantic_ai import RunContext
import pytest

from llmling_agent import Agent, AgentContext, AgentPool
from llmling_agent_config.toolsets import AgentManagementToolsetConfig


async def run_ctx_tool(ctx: RunContext, arg: str) -> str:
    """Tool expecting RunContext."""
    assert isinstance(ctx, RunContext)
    return f"RunContext tool got: {arg}"


async def agent_ctx_tool(ctx: AgentContext, arg: str) -> str:
    """Tool expecting AgentContext."""
    assert isinstance(ctx, AgentContext)
    return f"AgentContext tool got: {arg}"


async def data_with_run_ctx(ctx: RunContext) -> str:
    """Tool accessing data through RunContext."""
    return f"Data from RunContext: {ctx.deps}"


async def data_with_agent_ctx(ctx: AgentContext) -> str:
    """Tool accessing data through AgentContext."""
    return f"Data from AgentContext: {ctx.data}"


async def no_ctx_tool(arg: str) -> str:
    """Tool without any context."""
    return f"No context tool got: {arg}"


async def dual_ctx_tool(run_ctx: RunContext, agent_ctx: AgentContext, arg: str) -> str:
    """Tool expecting both RunContext and AgentContext."""
    assert isinstance(run_ctx, RunContext)
    assert isinstance(agent_ctx, AgentContext)
    return f"Dual context tool got: {arg} (agent: {agent_ctx.node_name})"


async def test_tool_context_injection(default_model: str):
    """Test that tools receive correct context."""
    context_received = None
    deps_received = None

    async def test_tool(ctx: RunContext[AgentContext], arg: str) -> str:
        """Test tool that captures its context."""
        nonlocal context_received, deps_received
        context_received = ctx
        deps_received = ctx.deps
        return f"Got arg: {arg}"

    async with Agent(model=default_model, deps_type=bool) as agent:
        # Register our test tool
        agent.tools.register_tool(test_tool, enabled=True)
        # Run agent which should trigger tool
        await agent.run("Use the test_tool with arg='test'", deps=True)

        # Verify context
        assert context_received is not None, "Tool did not receive context"
        assert isinstance(context_received, RunContext), "Wrong context type"

        # Verify dependencies
        assert deps_received is not None, "Tool did not receive dependencies"


async def test_plain_tool_no_context(default_model: str):
    """Test that plain tools work without context."""
    count = 0

    async def plain_tool(arg: str) -> str:
        """Tool without context parameter."""
        assert "test" in arg
        nonlocal count
        count += 1
        return f"Got arg: {arg}"

    async with Agent(model=default_model) as agent:
        agent.tools.register_tool(plain_tool, enabled=True)
        # Should work without error
        await agent.run("Use the plain_tool with arg='test'")
        assert count == 1


@pytest.mark.integration
async def test_capability_tools(default_model: str):
    """Test that capability tools work with AgentContext."""
    async with AgentPool() as pool:
        toolsets = [AgentManagementToolsetConfig()]
        toolset_providers = [config.get_provider() for config in toolsets]
        agent = await pool.add_agent(
            name="test", model=default_model, toolsets=toolset_providers
        )
        prompt = "Get available agents using the list_agents tool and return all names."
        result = await agent.run(prompt)
        assert agent.name in str(result.content)
        agent_2 = await pool.add_agent(
            name="test_2", model=default_model, toolsets=toolset_providers
        )
        await pool.add_agent(
            "helper", system_prompt="You help with tasks", model=default_model
        )
        result = await agent_2.run("Delegate 'say hello' to agent with name `helper`")
        assert result.get_tool_calls()
        assert result.get_tool_calls()[0].tool_name == "delegate_to"


@pytest.mark.flaky(reruns=3)
async def test_team_creation(default_model: str):
    """Test that an agent can create other agents and form them into a team."""
    async with AgentPool() as pool:
        # Create creator agent with needed toolsets
        toolsets = [AgentManagementToolsetConfig()]
        toolset_providers = [config.get_provider() for config in toolsets]
        creator = await pool.add_agent(
            name="creator", model=default_model, toolsets=toolset_providers
        )
        # Ask it to create a content team
        result = await creator.run("""
            Create two agents:
            1. A researcher who finds information named "alice"
            2. A writer who creates content named "bob"
            Then create a sequential team named "crew" with these agents.
        """)

        # Verify agents were created
        assert "alice" in pool.agents
        assert "bob" in pool.agents
        assert "crew" in pool.teams
        # Verify team creation message
        assert "alice" in str(result.content.lower())
        assert "bob" in str(result.content.lower())


async def test_context_compatibility(default_model: str):
    """Test that both context types work in tools."""
    async with Agent(model=default_model) as agent:
        agent.tools.register_tool(run_ctx_tool, name_override="run_ctx_tool")
        agent.tools.register_tool(agent_ctx_tool, name_override="agent_ctx_tool")
        agent.tools.register_tool(no_ctx_tool, name_override="no_ctx_tool")

        # All should work
        result1 = await agent.run("Use run_ctx_tool with argument 'test'")
        assert any(
            call.result == "RunContext tool got: test"
            for call in result1.get_tool_calls()
        )

        result2 = await agent.run("Use agent_ctx_tool with argument 'test'")
        assert any(
            call.result == "AgentContext tool got: test"
            for call in result2.get_tool_calls()
        )

        result3 = await agent.run("Use no_ctx_tool with argument 'test'")
        assert any(
            call.result == "No context tool got: test"
            for call in result3.get_tool_calls()
        )


async def test_context_sharing(default_model: str):
    """Test that both context types access same data."""
    shared_data = {"key": "value"}

    agent = Agent[dict](name="test", model=default_model, deps_type=dict)
    agent.context.data = shared_data

    agent.tools.register_tool(data_with_run_ctx)
    agent.tools.register_tool(data_with_agent_ctx)

    async with agent:
        result1 = await agent.run("Use data_with_run_ctx tool", deps=shared_data)
        result2 = await agent.run("Use data_with_agent_ctx tool", deps=shared_data)

        assert any(
            call.result == "Data from RunContext: {'key': 'value'}"
            for call in result1.get_tool_calls()
        )
        assert any(
            call.result == "Data from AgentContext: {'key': 'value'}"
            for call in result2.get_tool_calls()
        )


async def test_dual_context_tool(default_model: str):
    """Test tool that requires both RunContext and AgentContext."""
    async with Agent(model=default_model, name="dual-agent") as agent:
        agent.tools.register_tool(dual_ctx_tool)

        # This should work if dual context injection is implemented
        result = await agent.run("Use dual_ctx_tool with argument 'test'")

        # Should successfully call the tool with both contexts
        tool_calls = result.get_tool_calls()
        assert len(tool_calls) > 0

        expected_result = "Dual context tool got: test (agent: dual-agent)"
        assert any(call.result == expected_result for call in tool_calls)


if __name__ == "__main__":
    pytest.main([__file__, "-vv", "--log-level", "debug"])
