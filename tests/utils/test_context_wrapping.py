"""Test context wrapping utility for instruction functions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock, Mock


if TYPE_CHECKING:
    from collections.abc import Awaitable

from pydantic_ai import RunContext

from agentpool.agents.context import AgentContext


class TestWrapInstruction:
    """Test wrap_instruction utility."""

    def _create_mock_run_context(self, deps: Any = None) -> RunContext[Any]:
        """Helper to create a mock RunContext with all required params."""
        mock_model = Mock()
        mock_usage = Mock()
        mock_messages: list[Any] = []
        return RunContext(
            deps=deps,
            model=mock_model,
            usage=mock_usage,
            messages=mock_messages,
        )

    async def test_sync_function_no_context(self):
        """Test wrapping a sync function with no context."""

        def simple() -> str:
            return "Be helpful"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(simple)
        run_ctx = self._create_mock_run_context()
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "Be helpful"

    async def test_async_function_no_context(self):
        """Test wrapping an async function with no context."""

        async def simple_async() -> str:
            return "Be helpful and async"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(simple_async)
        run_ctx = self._create_mock_run_context()
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "Be helpful and async"

    async def test_function_with_agent_context(self):
        """Test wrapping a function that expects AgentContext."""

        def with_agent(ctx: AgentContext) -> str:
            return f"Tool: {ctx.tool_name or 'none'}"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(with_agent)

        # Create a mock AgentContext that only needs tool_name attribute
        mock_agent_ctx = MagicMock(spec=AgentContext)
        mock_agent_ctx.tool_name = None

        run_ctx = self._create_mock_run_context(deps=mock_agent_ctx)
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "Tool: none"

    async def test_async_function_with_agent_context(self):
        """Test wrapping an async function that expects AgentContext."""

        async def with_agent_async(ctx: AgentContext) -> str:
            return f"Tool (async): {ctx.tool_name or 'none'}"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(with_agent_async)

        # Create a mock AgentContext
        mock_agent_ctx = MagicMock(spec=AgentContext)
        mock_agent_ctx.tool_name = None

        run_ctx = self._create_mock_run_context(deps=mock_agent_ctx)
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "Tool (async): none"

    async def test_function_with_run_context(self):
        """Test wrapping a function that expects RunContext."""

        def with_run(ctx: RunContext) -> str:
            return f"RunContext: {ctx.deps or 'none'}"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(with_run)
        run_ctx = self._create_mock_run_context()
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "RunContext: none"

    async def test_function_with_both_contexts(self):
        """Test wrapping a function that expects both AgentContext and RunContext."""

        def with_both(agent_ctx: AgentContext, run_ctx: RunContext) -> str:
            return f"Agent: {agent_ctx.tool_name or 'none'}, Run: {run_ctx.deps or 'none'}"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(with_both)

        # Create a mock AgentContext
        mock_agent_ctx = MagicMock(spec=AgentContext)
        mock_agent_ctx.tool_name = None

        run_ctx = self._create_mock_run_context(deps=mock_agent_ctx)
        result_str = await wrapped(run_ctx)
        # The result will contain to actual object representation
        assert "Agent: none" in result_str
        assert "Run:" in result_str
        assert "MagicMock" in result_str

    async def test_error_handling_with_fallback(self):
        """Test that errors are caught and fallback is returned."""

        def error_func() -> str:
            raise ValueError("This should be caught")

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(error_func, fallback="Fallback text")
        run_ctx = self._create_mock_run_context()
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == "Fallback text"

    async def test_default_fallback_empty_string(self):
        """Test that default fallback is empty string."""

        def error_func() -> str:
            raise ValueError("Error")

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(error_func)  # No fallback specified
        run_ctx = self._create_mock_run_context()
        result: Awaitable[str] = wrapped(run_ctx)
        assert await result == ""

    async def test_wrapped_preserves_function_name(self):
        """Test that wrapper preserves original function name."""

        def named_function() -> str:
            return "test"

        from agentpool.utils.context_wrapping import wrap_instruction

        wrapped = wrap_instruction(named_function)

        # functools.wraps preserves __name__
        assert wrapped.__name__ == "named_function"

    async def test_instruction_func_union(self):
        """Test that InstructionFunc union types are accepted."""

        def simple() -> str:
            return "simple"

        def with_agent(ctx: AgentContext) -> str:
            return "agent"

        def with_run(ctx: RunContext) -> str:
            return "run"

        def with_both(agent_ctx: AgentContext, run_ctx: RunContext) -> str:
            return "both"

        from agentpool.utils.context_wrapping import wrap_instruction

        # All should be accepted as InstructionFunc
        for func in [simple, with_agent, with_run, with_both]:
            wrapped = wrap_instruction(func)  # type: ignore[arg-type]
            run_ctx = self._create_mock_run_context()
            result: Awaitable[str] = wrapped(run_ctx)
            # Just verify it doesn't crash
            await result
