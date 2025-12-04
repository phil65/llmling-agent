"""Test context binding functionality in signatures utils."""

import inspect
from typing import TypeVar

import pytest

from llmling_agent.agent.context import AgentContext
from llmling_agent.utils.signatures import create_bound_callable


# Mock RunContext for testing
class MockRunContext:
    """Mock RunContext for testing."""

    def __init__(self, data: str = "run_context_data"):
        self.data = data


# Generic types for testing
T = TypeVar("T")


class GenericRunContext[T]:
    """Generic RunContext for testing generic type binding."""

    def __init__(self, deps: T, name: str = "generic_context"):
        self.deps = deps
        self.name = name


class MockDeps:
    """Mock dependencies for generic context testing."""

    def __init__(self, value: str = "deps_value"):
        self.value = value


# Test functions with different context parameter patterns
def sync_func_with_agent_ctx(agent_ctx: AgentContext, value: str) -> str:
    """Sync function that requires AgentContext."""
    return f"sync: {value} (ctx: {agent_ctx.__class__.__name__})"


async def async_func_with_agent_ctx(agent_ctx: AgentContext, value: str) -> str:
    """Async function that requires AgentContext."""
    return f"async: {value} (ctx: {agent_ctx.__class__.__name__})"


def func_with_run_ctx(run_ctx: MockRunContext, value: int) -> str:
    """Function that requires RunContext."""
    return f"run_ctx: {value} (data: {run_ctx.data})"


async def func_with_both_contexts(
    agent_ctx: AgentContext, run_ctx: MockRunContext, value: str
) -> str:
    """Function that requires both contexts."""
    return f"both: {value} (agent: {agent_ctx.__class__.__name__}, run: {run_ctx.data})"


def func_with_no_context(value: str) -> str:
    """Function that requires no context."""
    return f"no_ctx: {value}"


class TestContextBoundCallable:
    """Test context binding functionality."""

    @pytest.fixture
    def mock_agent_context(self):
        """Create a mock AgentContext for testing."""
        from llmling_agent.models.agents import AgentConfig
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        # Create minimal config objects
        config = AgentConfig(name="test_agent")
        NodeConfig()
        manifest = AgentsManifest()

        return AgentContext(node_name="test_agent", config=config, definition=manifest)

    @pytest.fixture
    def mock_run_context(self):
        """Create a mock RunContext for testing."""
        return MockRunContext("test_data")

    async def test_bind_agent_context_sync(self, mock_agent_context):
        """Test binding AgentContext to sync function."""
        bound_func = create_bound_callable(
            sync_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # Call without providing context
        result = await bound_func("test_value")
        assert result == "sync: test_value (ctx: AgentContext)"

        # Check signature was updated
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert param_names == ["value"]
        assert "agent_ctx" not in param_names

    async def test_bind_agent_context_async(self, mock_agent_context):
        """Test binding AgentContext to async function."""
        bound_func = create_bound_callable(
            async_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # Call without providing context
        result = await bound_func("test_value")
        assert result == "async: test_value (ctx: AgentContext)"

        # Check signature was updated
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert param_names == ["value"]

    async def test_no_context_needed(self):
        """Test function that doesn't need context binding."""
        bound_func = create_bound_callable(func_with_no_context, by_type={})

        # Should return a wrapper function (not the original)
        assert bound_func is not func_with_no_context  # type: ignore[comparison-overlap]

        # Should work normally
        result = await bound_func("test")
        assert result == "no_ctx: test"

    def test_context_provided_but_not_needed(self, mock_agent_context):
        """Test providing context to function that doesn't need it."""
        bound_func = create_bound_callable(
            func_with_no_context, by_type={AgentContext: mock_agent_context}
        )

        # Should return a wrapper function (not the original)
        assert bound_func is not func_with_no_context  # type: ignore[comparison-overlap]

    async def test_preserve_introspection_attributes(self, mock_agent_context):
        """Test that introspection attributes are preserved."""
        bound_func = create_bound_callable(
            sync_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # Check attributes are preserved
        assert bound_func.__name__ == "sync_func_with_agent_ctx"
        assert "Sync function that requires AgentContext" in (bound_func.__doc__ or "")
        assert hasattr(bound_func, "__signature__")

        # Check annotations are updated correctly
        annotations = getattr(bound_func, "__annotations__", {})
        assert "agent_ctx" not in annotations
        assert "value" in annotations
        assert annotations["value"] is str

    async def test_context_binding_with_positional_args(self, mock_agent_context):
        """Test context binding works with positional arguments."""
        bound_func = create_bound_callable(
            sync_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # Call with positional arg
        result = await bound_func("positional")
        assert result == "sync: positional (ctx: AgentContext)"

    async def test_context_binding_with_keyword_args(self, mock_agent_context):
        """Test context binding works with keyword arguments."""
        bound_func = create_bound_callable(
            sync_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # Call with keyword arg
        result = await bound_func(value="keyword")
        assert result == "sync: keyword (ctx: AgentContext)"

    def test_missing_context_when_needed(self):
        """Test behavior when context is needed but not provided."""
        bound_func = create_bound_callable(sync_func_with_agent_ctx, by_type={})

        # Should return a wrapper function (not the original)
        assert bound_func is not sync_func_with_agent_ctx  # type: ignore[comparison-overlap]


class TestCodeModeIntegration:
    """Integration tests for context binding in CodeMode."""

    @pytest.fixture
    def mock_agent_context(self):
        """Create a mock AgentContext for testing."""
        from llmling_agent.models.agents import AgentConfig
        from llmling_agent.models.manifest import AgentsManifest
        from llmling_agent_config.nodes import NodeConfig

        # Create minimal config objects
        config = AgentConfig(name="test_agent")
        NodeConfig()
        manifest = AgentsManifest()

        return AgentContext(node_name="test_agent", config=config, definition=manifest)

    async def test_fsspec_like_tool_binding(self, mock_agent_context):
        """Test binding context for FSSpec-like tools."""

        class MockFSSpecTool:
            async def _read_file(
                self, agent_ctx: AgentContext, path: str, encoding: str = "utf-8"
            ) -> dict[str, str]:
                """Mock FSSpec read file method."""
                return {"path": path, "encoding": encoding, "agent": agent_ctx.__class__.__name__}

        tool = MockFSSpecTool()
        bound_method = create_bound_callable(
            tool._read_file, by_type={AgentContext: mock_agent_context}
        )

        # Call the bound method
        result = await bound_method("/test/path.txt", encoding="utf-8")

        assert result["path"] == "/test/path.txt"
        assert result["encoding"] == "utf-8"
        assert result["agent"] == "AgentContext"

        # Check signature is correct
        sig = inspect.signature(bound_method)
        param_names = list(sig.parameters.keys())
        # Should exclude 'agent_ctx' but include other params
        assert "agent_ctx" not in param_names
        assert "path" in param_names
        assert "encoding" in param_names

    async def test_method_with_self_parameter(self, mock_agent_context):
        """Test binding context for instance methods with self parameter."""

        class TestTool:
            def __init__(self, name: str):
                self.name = name

            def process(self, agent_ctx: AgentContext, data: str) -> str:
                """Process data with context."""
                return f"{self.name}: {data} (ctx: {agent_ctx.__class__.__name__})"

        tool = TestTool("test_tool")
        bound_method = create_bound_callable(
            tool.process, by_type={AgentContext: mock_agent_context}
        )

        result = await bound_method("test_data")
        assert result == "test_tool: test_data (ctx: AgentContext)"

        # Check signature excludes agent_ctx but includes data
        sig = inspect.signature(bound_method)
        param_names = list(sig.parameters.keys())
        assert "agent_ctx" not in param_names
        assert "data" in param_names

    async def test_keyword_argument_conflicts(self, mock_agent_context):
        """Test that bound parameters are filtered from kwargs to avoid conflicts."""
        bound_func = create_bound_callable(
            sync_func_with_agent_ctx, by_type={AgentContext: mock_agent_context}
        )

        # This should work - bound parameter in kwargs gets filtered out
        result = await bound_func("test_value", agent_ctx="should_be_ignored")
        assert result == "sync: test_value (ctx: AgentContext)"

        # Regular kwargs should still work
        result = await bound_func(value="test_value")
        assert result == "sync: test_value (ctx: AgentContext)"

    async def test_binding_by_name_priority(self, mock_agent_context):
        """Test that binding by name takes priority over binding by type."""
        mock_run_context = MockRunContext("test_data")

        def test_func(ctx: AgentContext, value: str) -> str:
            return f"ctx: {ctx.__class__.__name__}, value: {value}"

        # Both by_name and by_type provide values - by_name should win
        bound_func = create_bound_callable(
            test_func,
            by_name={"ctx": mock_run_context},  # Different type but same name
            by_type={AgentContext: mock_agent_context},
        )

        result = await bound_func("test")
        assert "MockRunContext" in result  # Should use by_name value

    async def test_multiple_context_binding(self, mock_agent_context):
        """Test binding multiple parameters by different methods."""
        mock_run_context = MockRunContext("test_data")
        bound_func = create_bound_callable(
            func_with_both_contexts,
            by_name={"run_ctx": mock_run_context},
            by_type={AgentContext: mock_agent_context},
        )

        result = await bound_func("test_value")
        assert result == "both: test_value (agent: AgentContext, run: test_data)"

        # Check signature excludes both bound parameters
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert param_names == ["value"]
        assert "agent_ctx" not in param_names
        assert "run_ctx" not in param_names

    async def test_no_binding_preserves_function(self):
        """Test that functions without bindings still get wrapped properly."""
        bound_func = create_bound_callable(func_with_no_context)

        # Should work normally
        result = await bound_func("test")
        assert result == "no_ctx: test"

        # Signature should be unchanged
        sig = inspect.signature(bound_func)
        original_sig = inspect.signature(func_with_no_context)
        assert list(sig.parameters.keys()) == list(original_sig.parameters.keys())

    def test_invalid_callable_signature(self):
        """Test error handling for callables with uninspectable signatures."""
        from unittest.mock import Mock

        # Create a mock callable that raises on signature inspection
        bad_callable = Mock()
        bad_callable.__name__ = "bad_callable"

        # Mock inspect.signature to raise ValueError
        from unittest.mock import patch

        with (
            patch(
                "llmling_agent.utils.signatures.inspect.signature",
                side_effect=ValueError("no signature"),
            ),
            pytest.raises(ValueError, match="Cannot inspect signature"),
        ):
            create_bound_callable(bad_callable, by_type={str: "test"})

    async def test_bind_kwargs_disabled_by_default(self, mock_agent_context):
        """Test that keyword-only parameters are not bound when bind_kwargs=False (default)."""

        def func_with_kwonly(a: str, *, agent_ctx: AgentContext) -> str:
            return f"a: {a}, ctx: {agent_ctx.__class__.__name__}"

        # Default behavior - should not bind keyword-only parameter
        bound_func = create_bound_callable(
            func_with_kwonly, by_type={AgentContext: mock_agent_context}
        )

        # Signature should still include the keyword-only parameter
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "agent_ctx" in param_names
        assert "a" in param_names

        # Should still require the agent_ctx to be passed as kwarg
        result = await bound_func("test", agent_ctx=mock_agent_context)
        assert result == "a: test, ctx: AgentContext"

    async def test_bind_kwargs_enabled(self, mock_agent_context):
        """Test that keyword-only parameters are bound when bind_kwargs=True."""

        def func_with_kwonly(a: str, *, agent_ctx: AgentContext, other: str = "default") -> str:
            return f"a: {a}, ctx: {agent_ctx.__class__.__name__}, other: {other}"

        # Enable binding of keyword-only parameters
        bound_func = create_bound_callable(
            func_with_kwonly, by_type={AgentContext: mock_agent_context}, bind_kwargs=True
        )

        # Signature should exclude bound keyword-only parameter but keep unbound ones
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "agent_ctx" not in param_names  # Should be bound and removed
        assert "a" in param_names  # Regular param should remain
        assert "other" in param_names  # Unbound kwarg should remain

        # Should work without passing agent_ctx
        result = await bound_func("test")
        assert result == "a: test, ctx: AgentContext, other: default"

        # Should work with other kwargs
        result = await bound_func("test", other="custom")
        assert result == "a: test, ctx: AgentContext, other: custom"

    async def test_bind_kwargs_by_name_priority(self, mock_agent_context):
        """Test that by_name takes priority over by_type for keyword-only parameters."""
        mock_run_context = MockRunContext("test_data")

        def func_with_kwonly(a: str, *, ctx: AgentContext) -> str:
            return f"a: {a}, ctx: {ctx.__class__.__name__}"

        # Both by_name and by_type provide values - by_name should win
        bound_func = create_bound_callable(
            func_with_kwonly,
            by_name={"ctx": mock_run_context},  # Different type but same name
            by_type={AgentContext: mock_agent_context},
            bind_kwargs=True,
        )

        result = await bound_func("test")
        assert "MockRunContext" in result  # Should use by_name value

    async def test_mixed_binding_args_and_kwargs(self, mock_agent_context):
        """Test binding both positional and keyword-only parameters."""
        mock_run_context = MockRunContext("test_data")

        def mixed_func(agent_ctx: AgentContext, value: str, *, run_ctx: MockRunContext) -> str:
            return f"agent: {agent_ctx.__class__.__name__}, value: {value}, run: {run_ctx.data}"

        bound_func = create_bound_callable(
            mixed_func,
            by_type={AgentContext: mock_agent_context, MockRunContext: mock_run_context},
            bind_kwargs=True,
        )

        # Check signature excludes both bound parameters
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert param_names == ["value"]
        assert "agent_ctx" not in param_names
        assert "run_ctx" not in param_names

        result = await bound_func("test_value")
        assert result == "agent: AgentContext, value: test_value, run: test_data"

    async def test_remaining_args_handling(self, mock_agent_context):
        """Test that remaining positional args are properly passed through."""

        def func_with_extra_params(agent_ctx: AgentContext, a: int, b: str, c: float = 1.0) -> str:
            return f"ctx: {agent_ctx.__class__.__name__}, a: {a}, b: {b}, c: {c}"

        bound_func = create_bound_callable(
            func_with_extra_params, by_type={AgentContext: mock_agent_context}
        )

        # Call with exact args needed
        result = await bound_func(1, "test", 2.5)
        assert result == "ctx: AgentContext, a: 1, b: test, c: 2.5"

        # Call with fewer args (using default)
        result = await bound_func(1, "test")
        assert result == "ctx: AgentContext, a: 1, b: test, c: 1.0"

        # Test that extra args still get passed through to original function
        # (even though it will likely cause an error in the original function)
        with pytest.raises(TypeError):  # Too many positional args
            await bound_func(1, "test", 2.5, "extra")

    async def test_keyword_only_parameters_not_bound(self, mock_agent_context):
        """Test that keyword-only parameters are not bound by type."""

        def func_with_kwonly(a: str, *, agent_ctx: AgentContext) -> str:
            return f"a: {a}, ctx: {agent_ctx.__class__.__name__}"

        # This should not bind the keyword-only parameter
        bound_func = create_bound_callable(
            func_with_kwonly, by_type={AgentContext: mock_agent_context}
        )

        # Signature should still include the keyword-only parameter
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "agent_ctx" in param_names
        assert "a" in param_names

        # Should still require the agent_ctx to be passed as kwarg
        result = await bound_func("test", agent_ctx=mock_agent_context)
        assert result == "a: test, ctx: AgentContext"

    async def test_generic_type_exact_matching(self, mock_agent_context):
        """Test that exact generic type annotations work for binding."""
        deps = MockDeps("test_deps")
        generic_context = GenericRunContext(deps)

        def func_with_generic(ctx: GenericRunContext[MockDeps], value: str) -> str:
            return f"value: {value}, deps: {ctx.deps.value}, name: {ctx.name}"

        # Exact generic type matching should work
        bound_func = create_bound_callable(
            func_with_generic, by_type={GenericRunContext[MockDeps]: generic_context}
        )

        result = await bound_func("test")
        assert result == "value: test, deps: test_deps, name: generic_context"

        # Check signature excludes the bound generic parameter
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "ctx" not in param_names
        assert "value" in param_names

    async def test_generic_type_different_params_dont_match(self, mock_agent_context):
        """Test that different generic parameters still don't match each other."""
        deps = MockDeps("test_deps")
        generic_context = GenericRunContext(deps)

        def func_with_generic(ctx: GenericRunContext[str], value: str) -> str:
            return f"value: {value}, deps: {ctx.deps}"

        # Different generic parameter should NOT match
        bound_func = create_bound_callable(
            func_with_generic,
            by_type={GenericRunContext[MockDeps]: generic_context},  # Different type param
        )

        # Should not bind - signature should still include ctx
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "ctx" in param_names  # Should NOT be bound
        assert "value" in param_names

    async def test_different_generic_parameters_dont_match(self):
        """Test that different generic parameters don't match each other."""
        deps1 = MockDeps("deps1")
        deps2 = "string_deps"

        generic_context_mock = GenericRunContext(deps1)
        GenericRunContext(deps2)

        def func_expecting_str_context(ctx: GenericRunContext[str], value: str) -> str:
            return f"value: {value}, deps: {ctx.deps}"

        # MockDeps context should NOT match str context
        bound_func = create_bound_callable(
            func_expecting_str_context, by_type={GenericRunContext[MockDeps]: generic_context_mock}
        )

        # Should not bind - different type parameters
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "ctx" in param_names  # Should NOT be bound

    def test_generic_type_annotations_in_signature(self):
        """Test that generic type annotations are preserved correctly."""

        def func_with_generic(ctx: GenericRunContext[MockDeps], value: str) -> str:
            return "test"

        # Get original signature
        original_sig = inspect.signature(func_with_generic)
        ctx_param = original_sig.parameters["ctx"]

        # The annotation should be the generic type
        assert ctx_param.annotation == GenericRunContext[MockDeps]

    async def test_generic_origin_type_matching_now_works(self):
        """Test that origin types now match parameterized generic types."""
        deps = MockDeps("test_deps")
        generic_context = GenericRunContext(deps)

        def func_with_generic(ctx: GenericRunContext[MockDeps], value: str) -> str:
            return f"value: {value}, deps: {ctx.deps.value}, name: {ctx.name}"

        # Origin type should now match parameterized type
        bound_func = create_bound_callable(
            func_with_generic,
            by_type={GenericRunContext: generic_context},  # Origin type binding
        )

        result = await bound_func("test")
        assert result == "value: test, deps: test_deps, name: generic_context"

        # Check signature excludes the bound parameter
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert "ctx" not in param_names
        assert "value" in param_names

    async def test_exact_match_takes_priority_over_origin(self):
        """Test that exact generic matches take priority over origin matches."""
        deps = MockDeps("test_deps")
        exact_context = GenericRunContext(deps, "exact_match")
        origin_context = GenericRunContext(deps, "origin_match")

        def func_with_generic(ctx: GenericRunContext[MockDeps], value: str) -> str:
            return f"value: {value}, name: {ctx.name}"

        # Both exact and origin types provided - exact should win
        bound_func = create_bound_callable(
            func_with_generic,
            by_type={
                GenericRunContext[MockDeps]: exact_context,  # Exact match
                GenericRunContext: origin_context,  # Origin match
            },
        )

        result = await bound_func("test")
        assert result == "value: test, name: exact_match"  # Should use exact match

    async def test_mixed_generic_and_concrete_binding(self, mock_agent_context):
        """Test binding both generic and concrete types together."""
        deps = MockDeps("test_deps")
        generic_context = GenericRunContext(deps)

        def mixed_func(
            agent_ctx: AgentContext,
            generic_ctx: GenericRunContext[MockDeps],
            value: str,
        ) -> str:
            return f"agent: {agent_ctx.__class__.__name__}, generic: {generic_ctx.name}, value: {value}"  # noqa: E501

        bound_func = create_bound_callable(
            mixed_func,
            by_type={
                AgentContext: mock_agent_context,  # Concrete type
                GenericRunContext: generic_context,  # Origin type for generic
            },
        )

        result = await bound_func("test")
        assert result == "agent: AgentContext, generic: generic_context, value: test"

        # Check signature excludes both bound parameters
        sig = inspect.signature(bound_func)
        param_names = list(sig.parameters.keys())
        assert param_names == ["value"]

    async def test_practical_runcontext_generic_example(self, mock_agent_context):
        """Test a practical example with RunContext-style generic usage."""

        # Simulate a common pattern where RunContext[T] holds dependencies
        class DatabaseConn:
            def query(self, sql: str) -> str:
                return f"Result for: {sql}"

        class MyDeps:
            def __init__(self):
                self.db = DatabaseConn()
                self.config = {"debug": True}

        deps = MyDeps()
        run_context = GenericRunContext(deps, "run_context")

        def tool_function(
            agent_ctx: AgentContext, run_ctx: GenericRunContext[MyDeps], query: str
        ) -> str:
            """A typical tool function that needs both contexts."""
            result = run_ctx.deps.db.query(query)
            debug = run_ctx.deps.config["debug"]
            return f"Agent: {agent_ctx.node_name}, Debug: {debug}, Query result: {result}"

        # Bind both contexts using origin type for the generic one
        bound_tool = create_bound_callable(
            tool_function,
            by_type={
                AgentContext: mock_agent_context,
                GenericRunContext: run_context,  # Origin type matches any GenericRunContext[T]
            },
        )

        # Should work with just the query parameter
        result = await bound_tool("SELECT * FROM users")
        expected = "Agent: test_agent, Debug: True, Query result: Result for: SELECT * FROM users"
        assert result == expected

        # Signature should only contain the query parameter
        sig = inspect.signature(bound_tool)
        assert list(sig.parameters.keys()) == ["query"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
