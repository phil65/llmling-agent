"""Snapshot tests for ACP agent CLI help output.

These tests capture the actual CLI help output of agent tools to detect
when command-line arguments and options change.

Run with: uv run pytest tests/test_acp_agents_snapshots.py -m acp_cli_snapshot
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING, get_args

import pytest

from agentpool.models.acp_agents.mcp_capable import MCPCapableACPAgentConfigTypes
from agentpool.models.acp_agents.non_mcp import RegularACPAgentConfigTypes


if TYPE_CHECKING:
    from agentpool.models.acp_agents.base import BaseACPAgentConfig


def get_concrete_agent_classes() -> list[type[BaseACPAgentConfig]]:
    """Extract all concrete agent classes from the MCP and non-MCP union types."""
    mcp_classes = list(get_args(MCPCapableACPAgentConfigTypes))
    regular_classes = list(get_args(RegularACPAgentConfigTypes))
    all_classes = mcp_classes + regular_classes
    return sorted(all_classes, key=lambda cls: cls.__name__)


def get_cli_help_output(agent_class: type[BaseACPAgentConfig]) -> str:
    """Capture actual CLI help output by executing the command.

    Args:
        agent_class: The agent configuration class to test.

    Returns:
        The help output from running the CLI command with --help.
    """
    import asyncio

    try:
        instance = agent_class()
    except Exception as e:  # noqa: BLE001
        return f"Error creating instance: {e}"

    command = instance.get_command()

    # Get base args (subcommands) - these define the actual command we use
    # e.g., for opencode, get_args returns ["acp", ...] so we want "opencode acp --help"
    try:
        base_args = asyncio.run(instance.get_args(prompt_manager=None))
        # Extract only subcommand args (before any flags like --cwd)
        subcommand_args = []
        for arg in base_args:
            if arg.startswith("-"):
                break
            subcommand_args.append(arg)
    except Exception:  # noqa: BLE001
        subcommand_args = []

    full_command = [command, *subcommand_args] if subcommand_args else [command]
    command_str = " ".join(full_command)

    lines = [
        f"=== {agent_class.__name__} ===",
        "",
        f"Command: {command_str}",
        "",
    ]

    # Try common help flags
    for help_flag in ["--help", "-h", "help"]:
        try:
            result = subprocess.run(
                [*full_command, help_flag],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

            # If command succeeded or gave help output (some tools exit with code 1 for --help)
            if result.returncode in (0, 1) and (result.stdout or result.stderr):
                output = result.stdout or result.stderr
                if output.strip():
                    lines.append("CLI Help Output:")
                    lines.append("-" * 70)
                    lines.append(output.strip())
                    return "\n".join(lines)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            continue

    # If we couldn't get help output, note that
    lines.append("CLI Help Output:")
    lines.append("-" * 70)
    lines.append(
        f"(Unable to capture help - command '{command}' not found or doesn't support --help)"
    )

    return "\n".join(lines)


@pytest.mark.acp_cli_snapshot
@pytest.mark.parametrize(
    "agent_class",
    get_concrete_agent_classes(),
    ids=[cls.__name__ for cls in get_concrete_agent_classes()],
)
def test_acp_agent_help_output(agent_class: type[BaseACPAgentConfig], snapshot):
    """Test that ACP agent CLI help output matches snapshot.

    This test captures the actual --help output from running each agent's CLI command.
    Changes in command-line options, flags, or arguments will trigger snapshot updates.

    To update snapshots after intentional changes:
        uv run pytest tests/test_acp_agents_snapshots.py -m acp_cli_snapshot --snapshot-update
    """
    help_output = get_cli_help_output(agent_class)
    assert help_output == snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-m", "acp_cli_snapshot", "-v"])
