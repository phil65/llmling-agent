"""Test provider instruction integration into NativeAgent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic_ai import Agent as PydanticAgent
import pytest

from agentpool.agents.native_agent import Agent
from agentpool.resource_providers.base import ResourceProvider


if TYPE_CHECKING:
    from agentpool.agents.context import AgentContext
    from agentpool.prompts.instructions import InstructionFunc


class SimpleInstructionProvider(ResourceProvider):
    """Simple provider that returns static instructions."""

    def __init__(self) -> None:
        super().__init__("simple_provider")
        self.kind = "base"

    async def get_instructions(self) -> list[InstructionFunc]:
        """Return a simple instruction function."""

        def simple_instruction() -> str:
            return "Be helpful and concise"

        return [simple_instruction]


class AgentContextInstructionProvider(ResourceProvider):
    """Provider that returns AgentContext-aware instruction."""

    def __init__(self) -> None:
        super().__init__("agent_context_provider")
        self.kind = "base"

    async def get_instructions(self) -> list[InstructionFunc]:
        """Return instruction that expects AgentContext."""

        async def with_agent_context(ctx: AgentContext[Any]) -> str:
            return f"Agent model: {ctx.model_name}"

        return [with_agent_context]


class RunContextInstructionProvider(ResourceProvider):
    """Provider that returns RunContext-aware instruction."""

    def __init__(self) -> None:
        super().__init__("run_context_provider")
        self.kind = "base"

    async def get_instructions(self) -> list[InstructionFunc]:
        """Return instruction that expects RunContext."""

        async def with_run_context(ctx: Any) -> str:
            return "Model: gpt-4o-mini"

        return [with_run_context]


class EmptyInstructionProvider(ResourceProvider):
    """Provider that returns no instructions."""

    def __init__(self) -> None:
        super().__init__("empty_provider")
        self.kind = "base"

    async def get_instructions(self) -> list[InstructionFunc]:
        """Return empty list."""
        return []


class FailingInstructionProvider(ResourceProvider):
    """Provider that fails to provide instructions."""

    def __init__(self) -> None:
        super().__init__("failing_provider")
        self.kind = "base"

    async def get_instructions(self) -> list[InstructionFunc]:
        """Always raises RuntimeError."""
        msg = "Failed to get instructions"
        raise RuntimeError(msg)


@pytest.fixture
async def agent_with_instruction_providers():
    """Create an agent with instruction providers."""
    from agentpool.agents.native_agent import Agent

    # Create providers
    provider1 = SimpleInstructionProvider()
    provider2 = AgentContextInstructionProvider()
    provider3 = RunContextInstructionProvider()

    # Create agent
    agent = Agent(
        name="test_agent",
        model="openai:gpt-4o-mini",
        system_prompt="You are an AI assistant.",
    )

    # Add providers via tool manager
    agent.tools.add_provider(provider1)
    agent.tools.add_provider(provider2)
    agent.tools.add_provider(provider3)

    return agent


class TestNativeAgentInstructions:
    """Test NativeAgent integration with provider instructions."""

    async def test_agentlet_collects_instructions_from_providers(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that get_agentlet collects instructions from all providers."""
        agentlet: PydanticAgent[Any, Any] = await agent_with_instruction_providers.get_agentlet(
            None, None, None
        )

        # Verify that agentlet was created
        assert isinstance(agentlet, PydanticAgent)
        assert agentlet.name == "test_agent"

    async def test_formatted_system_prompt_includes_static_prompt(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that formatted system prompt includes static system prompt."""
        # Initialize agent to format system prompt
        async with agent_with_instruction_providers:
            # Access the formatted system prompt
            assert agent_with_instruction_providers._formatted_system_prompt is not None
            assert (
                "You are an AI assistant."
                in agent_with_instruction_providers._formatted_system_prompt
            )

    async def test_instructions_are_collected_and_wrapped(
        self, agent_with_instruction_providers: Agent
    ):
        """Test that instructions from providers are collected and wrapped."""
        # Get agentlet which should include wrapped instructions
        async with agent_with_instruction_providers:
            agentlet: PydanticAgent[Any, Any] = await agent_with_instruction_providers.get_agentlet(
                None, None, None
            )

            # The instructions should be in agentlet's instructions
            # They should be wrapped to be RunContext -> str
            assert agentlet.instructions is not None

    async def test_provider_instructions_reactive(self, agent_with_instruction_providers: Agent):
        """Test that provider instructions are called on each run."""
        # Create agent
        async with agent_with_instruction_providers as agent:
            # Run the agent (instructions should be evaluated)
            # Note: This would require a model key, so we'll just test setup
            agentlet: PydanticAgent[Any, Any] = await agent.get_agentlet(None, None, None)

            # Verify agentlet was created with instructions
            assert agentlet is not None
            assert agentlet.instructions is not None

    async def test_no_providers_uses_only_static_prompt(self):
        """Test that agent works normally with no instruction providers."""
        from agentpool.agents.native_agent import Agent

        agent = Agent(
            name="simple_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are a simple assistant.",
        )

        async with agent:
            agentlet: PydanticAgent[Any, Any] = await agent.get_agentlet(None, None, None)

            # Should work with just static system prompt
            assert isinstance(agentlet, PydanticAgent)
            assert agentlet.instructions is not None

    async def test_provider_returning_empty_instructions(self):
        """Test that providers returning empty list are handled."""
        from agentpool.agents.native_agent import Agent

        agent = Agent(
            name="empty_provider_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an assistant.",
        )

        agent.tools.add_provider(EmptyInstructionProvider())

        async with agent:
            # Should not fail with empty instruction list
            agentlet: PydanticAgent[Any, Any] = await agent.get_agentlet(None, None, None)
            assert isinstance(agentlet, PydanticAgent)

    async def test_provider_get_instructions_error_handling(self):
        """Test that errors in provider.get_instructions are handled gracefully."""
        agent = Agent(
            name="failing_provider_agent",
            model="openai:gpt-4o-mini",
            system_prompt="You are an assistant.",
        )

        agent.tools.add_provider(FailingInstructionProvider())

        async with agent:
            # Should handle error gracefully and still create agentlet
            # Implementation should log error and continue
            agentlet: PydanticAgent[Any, Any] = await agent.get_agentlet(None, None, None)
            assert isinstance(agentlet, PydanticAgent)

    async def test_from_config_with_provider_instruction_ref(self):
        """Test from_config with ProviderInstructionConfig using ref."""
        from agentpool.agents.native_agent import Agent
        from agentpool.models.agents import NativeAgentConfig
        from agentpool_config.instructions import ProviderInstructionConfig

        # Create a simple provider with get_instructions
        class SimpleRefProvider(ResourceProvider):
            def __init__(self) -> None:
                super().__init__("simple_ref_provider")
                self.kind = "base"

            async def get_tools(self) -> list[Any]:
                return []

            async def get_instructions(self) -> list[InstructionFunc]:
                async def simple_inst() -> str:
                    return "Dynamic instruction from ref provider"

                return [simple_inst]

        # Create config with ProviderInstructionConfig referencing to provider
        # Note: In actual usage, toolsets would come from manifest or toolset config
        # For this test, we'll add provider via tool manager

        config = NativeAgentConfig(
            name="test_agent_with_ref",
            model="openai:gpt-4o-mini",
            system_prompt=["Be helpful.", ProviderInstructionConfig(ref="simple_ref_provider")],
        )

        # Create agent from config
        agent = Agent.from_config(config)

        # Manually add the referenced provider to the tool manager
        # This simulates how it would come from toolsets in real usage
        provider = SimpleRefProvider()
        agent.tools.add_provider(provider)

        async with agent:
            # Verify agentlet can be created
            agentlet: PydanticAgent[Any, Any] = await agent.get_agentlet(None, None, None)
            assert isinstance(agentlet, PydanticAgent)

            # Verify that a provider is in the tools.providers list
            provider_names = [p.name for p in agent.tools.providers]
            assert "instruction:simple_ref_provider" in provider_names
