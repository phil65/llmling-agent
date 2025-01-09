from __future__ import annotations

from llmling_agent.running.discovery import agent_function


async def test_basic_decoration():
    """Test basic function decoration."""

    @agent_function
    async def test_func():
        return "test"

    assert hasattr(test_func, "_agent_function")
    metadata = test_func._agent_function
    assert metadata.name == "test_func"
    assert not metadata.depends_on
    assert not metadata.default_inputs


async def test_decoration_with_args():
    """Test decoration with arguments."""

    @agent_function(depends_on="other_func")
    async def test_func():
        return "test"

    metadata = test_func._agent_function
    assert metadata.depends_on == ["other_func"]


async def test_multiple_dependencies():
    """Test multiple dependencies."""

    @agent_function(depends_on=["func1", "func2"])
    async def test_func():
        return "test"

    metadata = test_func._agent_function  # type: ignore
    assert metadata.depends_on == ["func1", "func2"]


async def test_default_inputs():
    """Test default input extraction."""

    @agent_function
    async def test_func(
        required: str,
        optional: int = 42,
        another: str = "default",
    ):
        return "test"

    metadata = test_func._agent_function  # type: ignore
    assert metadata.default_inputs == {
        "optional": 42,
        "another": "default",
    }


async def test_function_name():
    """Test function name extraction."""

    def outer():
        @agent_function
        async def inner():
            return "test"

        return inner

    func = outer()
    metadata = func._agent_function
    assert metadata.name == "inner"
