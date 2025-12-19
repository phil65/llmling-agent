"""Testing utilities for end-to-end ACP testing.

This module provides a lightweight test harness for running end-to-end tests
against the llmling-agent ACP server. It uses ACPAgent as the client, connecting
to a llmling-agent serve-acp subprocess.

Example:
    ```python
    async def test_basic_prompt():
        async with acp_test_session("tests/fixtures/simple.yml") as agent:
            result = await agent.run("Say hello")
            assert result.content

    async def test_filesystem_tool():
        async with acp_test_session(
            "tests/fixtures/with_tools.yml",
            file_access=True,
            terminal_access=True,
        ) as agent:
            result = await agent.run("List files in the current directory")
            assert "pyproject.toml" in result.content
    ```
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from evented.configs import EventConfig

    from llmling_agent.agents.acp_agent import ACPAgent
    from llmling_agent.common_types import BuiltinEventHandlerType, IndividualEventHandler


@asynccontextmanager
async def acp_test_session(
    config: str | Path | None = None,
    *,
    file_access: bool = True,
    terminal_access: bool = True,
    providers: list[str] | None = None,
    debug_messages: bool = False,
    debug_file: str | None = None,
    debug_commands: bool = False,
    agent: str | None = None,
    load_skills: bool = False,
    cwd: str | Path | None = None,
    event_configs: Sequence[EventConfig] | None = None,
    event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
) -> AsyncIterator[ACPAgent[Any]]:
    """Create an end-to-end ACP test session using llmling-agent as server.

    This context manager starts an ACPAgent connected to a llmling-agent serve-acp
    subprocess, enabling full round-trip testing of the ACP protocol.

    Args:
        config: Path to agent configuration YAML file. If None, uses default config.
        file_access: Enable file system access for agents.
        terminal_access: Enable terminal access for agents.
        providers: Model providers to search for models.
        debug_messages: Save raw JSON-RPC messages to debug file.
        debug_file: File path for JSON-RPC debug messages.
        debug_commands: Enable debug slash commands for testing.
        agent: Name of specific agent to use from config.
        load_skills: Load client-side skills from .claude/skills directory.
        cwd: Working directory for the ACP server subprocess.
        event_configs: Event configurations for the agent.
        event_handlers: Event handlers for the agent (e.g., ["detailed"] for logging).

    Yields:
        ACPAgent instance connected to the test server.

    Example:
        ```python
        async def test_echo():
            async with acp_test_session("my_config.yml") as agent:
                result = await agent.run("Hello!")
                assert "Hello" in result.content
        ```
    """
    from llmling_agent.agents.acp_agent import ACPAgent

    # Build command line arguments
    args = ["run", "llmling-agent", "serve-acp"]

    if config is not None:
        args.extend(["--config", str(config)])

    if not file_access:
        args.append("--no-file-access")

    if not terminal_access:
        args.append("--no-terminal-access")

    if providers:
        for provider in providers:
            args.extend(["--model-provider", provider])

    if debug_messages:
        args.append("--debug-messages")

    if debug_file:
        args.extend(["--debug-file", debug_file])

    if debug_commands:
        args.append("--debug-commands")

    if agent:
        args.extend(["--agent", agent])

    if not load_skills:
        args.append("--no-skills")

    working_dir = str(cwd) if cwd else str(Path.cwd())

    async with ACPAgent(
        command="uv",
        args=args,
        cwd=working_dir,
        event_configs=event_configs,
        event_handlers=event_handlers,
    ) as acp_agent:
        yield acp_agent
