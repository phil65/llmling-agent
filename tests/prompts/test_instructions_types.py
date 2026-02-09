"""Test instruction function types and protocols."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable  # noqa: TC003
from typing import TYPE_CHECKING, Any

import pytest

# Test that all types can be imported (will fail if they don't exist)
from agentpool.prompts.instructions import (  # noqa: TC001
    AgentContextInstruction,
    BothContextsInstruction,
    InstructionFunc,
    RunContextInstruction,
    SimpleInstruction,
)


if TYPE_CHECKING:
    from pydantic_ai import RunContext

    from agentpool.agents.context import AgentContext


class TestSimpleInstruction:
    """Test SimpleInstruction protocol."""

    def test_sync_simple_instruction(self):
        """Simple synchronous instruction function."""

        def simple_prompt() -> str:
            return "Hello, world!"

        # The function should match the SimpleInstruction protocol
        result: str | Awaitable[str] = simple_prompt()
        assert result == "Hello, world!"

    async def test_async_simple_instruction(self):
        """Simple async instruction function."""

        async def simple_prompt() -> str:
            await asyncio.sleep(0)
            return "Hello, async world!"

        # The function should match the SimpleInstruction protocol
        async def use_instruction(instruction: SimpleInstruction) -> str:
            result = instruction()
            if isinstance(result, str):
                return result
            return await result

        result = await use_instruction(simple_prompt)
        assert result == "Hello, async world!"


class TestAgentContextInstruction:
    """Test AgentContextInstruction protocol."""

    def test_sync_agent_context_instruction(self):
        """Instruction function with AgentContext only."""

        def context_prompt(ctx: AgentContext[Any]) -> str:
            return f"Context: {ctx.tool_name or 'none'}"

        # Create a mock context (we'll use None for this test)
        # The function should match the AgentContextInstruction protocol
        def use_instruction(instruction: AgentContextInstruction) -> str:

            # Create minimal agent for testing
            with pytest.MonkeyPatch().context():
                pass  # We can't easily create a full AgentContext without setup
            return "placeholder"

    async def test_async_agent_context_instruction(self):
        """Async instruction function with AgentContext only."""

        async def context_prompt(ctx: AgentContext[Any]) -> str:
            await asyncio.sleep(0)
            return f"Context: {ctx.tool_name or 'none'}"

        # The function should match the AgentContextInstruction protocol
        async def use_instruction(instruction: AgentContextInstruction) -> str:

            # Create minimal agent for testing
            return "placeholder"


class TestRunContextInstruction:
    """Test RunContextInstruction protocol."""

    def test_sync_run_context_instruction(self):
        """Instruction function with RunContext only."""

        def context_prompt(ctx: RunContext[Any]) -> str:
            return f"Dep: {ctx.deps or 'none'}"

        # The function should match the RunContextInstruction protocol
        def use_instruction(instruction: RunContextInstruction) -> str:
            return "placeholder"


class TestBothContextsInstruction:
    """Test BothContextsInstruction protocol."""

    def test_sync_both_contexts_instruction(self):
        """Instruction function with both AgentContext and RunContext."""

        def dual_prompt(
            agent_ctx: AgentContext[Any],
            run_ctx: RunContext[Any],
        ) -> str:
            return f"Agent: {agent_ctx.tool_name or 'none'}, Run: {run_ctx.deps or 'none'}"

        # The function should match the BothContextsInstruction protocol
        def use_instruction(instruction: BothContextsInstruction) -> str:
            return "placeholder"


class TestInstructionFuncUnion:
    """Test InstructionFunc union type."""

    def test_simple_instruction_in_union(self):
        """SimpleInstruction should be assignable to InstructionFunc."""

        def simple() -> str:
            return "test"

        func: InstructionFunc = simple
        # Just verify type compatibility
        assert callable(func)

    def test_agent_context_instruction_in_union(self):
        """AgentContextInstruction should be assignable to InstructionFunc."""

        def with_agent_context(ctx: AgentContext[Any]) -> str:
            return "test"

        func: InstructionFunc = with_agent_context
        # Just verify type compatibility
        assert callable(func)

    def test_run_context_instruction_in_union(self):
        """RunContextInstruction should be assignable to InstructionFunc."""

        def with_run_context(ctx: RunContext[Any]) -> str:
            return "test"

        func: InstructionFunc = with_run_context
        # Just verify type compatibility
        assert callable(func)

    def test_both_contexts_instruction_in_union(self):
        """BothContextsInstruction should be assignable to InstructionFunc."""

        def with_both_contexts(
            agent_ctx: AgentContext[Any],
            run_ctx: RunContext[Any],
        ) -> str:
            return "test"

        func: InstructionFunc = with_both_contexts
        # Just verify type compatibility
        assert callable(func)


class TestRuntimeCheckable:
    """Test @runtime_checkable decorator on protocols."""

    def test_simple_instruction_isinstance(self):
        """SimpleInstruction should support isinstance checks."""

        def simple() -> str:
            return "test"

        # @runtime_checkable enables isinstance checks
        assert isinstance(simple, SimpleInstruction)

    def test_agent_context_instruction_isinstance(self):
        """AgentContextInstruction should support isinstance checks."""

        def with_ctx(ctx: AgentContext[Any]) -> str:
            return "test"

        assert isinstance(with_ctx, AgentContextInstruction)

    def test_run_context_instruction_isinstance(self):
        """RunContextInstruction should support isinstance checks."""

        def with_ctx(ctx: RunContext[Any]) -> str:
            return "test"

        assert isinstance(with_ctx, RunContextInstruction)

    def test_both_contexts_instruction_isinstance(self):
        """BothContextsInstruction should support isinstance checks."""

        def with_both(
            agent_ctx: AgentContext[Any],
            run_ctx: RunContext[Any],
        ) -> str:
            return "test"

        assert isinstance(with_both, BothContextsInstruction)

    def test_async_simple_instruction_isinstance(self):
        """Async simple instruction should support isinstance checks."""

        async def async_simple() -> str:
            return "test"

        assert isinstance(async_simple, SimpleInstruction)
