"""Snapshot tests for ACP agent capabilities.

These tests capture the capabilities reported by ACP agents during initialization
to detect when agent capabilities change (MCP support, prompt types, etc.).

Run with: uv run pytest tests/test_acp_capabilities_snapshots.py -m acp_snapshot
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any, get_args

import pytest

from acp.client.protocol import Client
from acp.schema import InitializeRequest
from acp.schema.capabilities import ClientCapabilities, FileSystemCapability
from acp.schema.common import Implementation
from acp.stdio import spawn_agent_process
from agentpool.models.acp_agents.mcp_capable import MCPCapableACPAgentConfigTypes
from agentpool.models.acp_agents.non_mcp import RegularACPAgentConfigTypes


if TYPE_CHECKING:
    from agentpool.models.acp_agents.base import BaseACPAgentConfig


class MinimalClient(Client):
    """Minimal client implementation for capability probing."""

    async def read_text_file(self, params: Any) -> Any:
        raise NotImplementedError

    async def write_text_file(self, params: Any) -> Any:
        raise NotImplementedError

    async def create_terminal(self, params: Any) -> Any:
        raise NotImplementedError

    async def terminal_output(self, params: Any) -> Any:
        raise NotImplementedError

    async def wait_for_terminal_exit(self, params: Any) -> Any:
        raise NotImplementedError

    async def kill_terminal_command(self, params: Any) -> Any:
        raise NotImplementedError

    async def release_terminal(self, params: Any) -> Any:
        raise NotImplementedError

    async def request_permission(self, params: Any) -> Any:
        raise NotImplementedError


def get_concrete_agent_classes() -> list[type[BaseACPAgentConfig]]:
    """Extract all concrete agent classes from the MCP and non-MCP union types."""
    mcp_classes = list(get_args(MCPCapableACPAgentConfigTypes))
    regular_classes = list(get_args(RegularACPAgentConfigTypes))
    all_classes = mcp_classes + regular_classes
    return sorted(all_classes, key=lambda cls: cls.__name__)


# Default values for agents with required fields
AGENT_TEST_DEFAULTS: dict[str, dict[str, Any]] = {
    "FastAgentACPAgentConfig": {"model": "anthropic.claude-3-5-haiku-latest"},
}


async def get_agent_capabilities(agent_class: type[BaseACPAgentConfig]) -> dict[str, Any]:
    """Probe an ACP agent for its capabilities via the initialize handshake.

    Args:
        agent_class: The agent configuration class to test.

    Returns:
        Dict containing agent info and capabilities, or error info.
    """
    result: dict[str, Any] = {
        "agent_class": agent_class.__name__,
        "status": "unknown",
    }

    # Get test defaults for this agent class if any
    defaults = AGENT_TEST_DEFAULTS.get(agent_class.__name__, {})

    try:
        instance = agent_class(**defaults)
    except Exception as e:  # noqa: BLE001
        result["status"] = "config_error"
        result["error"] = f"Error creating instance: {e}"
        return result

    command = instance.get_command()
    try:
        args = await instance.get_args(prompt_manager=None)
    except Exception:  # noqa: BLE001
        args = []

    result["command"] = command
    result["args"] = args

    # Pass through environment variables (needed for API keys, etc.)
    env = dict(os.environ)

    try:
        async with asyncio.timeout(15):
            async with spawn_agent_process(
                lambda _: MinimalClient(),
                command,
                *args,
                env=env,
                log_stderr=True,
            ) as (conn, _process):
                init_request = InitializeRequest(
                    protocol_version=1,
                    client_info=Implementation(
                        title="Capability Probe",
                        name="agentpool-capability-test",
                        version="0.1.0",
                    ),
                    client_capabilities=ClientCapabilities(
                        terminal=True,
                        fs=FileSystemCapability(
                            read_text_file=True,
                            write_text_file=True,
                        ),
                    ),
                )

                response = await conn.initialize(init_request)

                result["status"] = "ok"
                result["protocol_version"] = response.protocol_version

                if response.agent_info:
                    result["agent_info"] = {
                        "name": response.agent_info.name,
                        "title": response.agent_info.title,
                        "version": response.agent_info.version,
                    }

                if response.agent_capabilities:
                    caps = response.agent_capabilities
                    result["capabilities"] = {
                        "load_session": caps.load_session,
                    }

                    if caps.mcp_capabilities:
                        result["capabilities"]["mcp"] = {
                            "stdio": True,  # Always true per spec
                            "http": caps.mcp_capabilities.http,
                            "sse": caps.mcp_capabilities.sse,
                        }

                    if caps.prompt_capabilities:
                        result["capabilities"]["prompt"] = {
                            "audio": caps.prompt_capabilities.audio,
                            "image": caps.prompt_capabilities.image,
                            "embedded_context": caps.prompt_capabilities.embedded_context,
                        }

                    if caps.session_capabilities:
                        session_caps = caps.session_capabilities
                        result["capabilities"]["session"] = {
                            "fork": session_caps.fork is not None,
                            "list": session_caps.list is not None,
                            "resume": session_caps.resume is not None,
                        }

                if response.auth_methods:
                    result["auth_methods"] = [
                        {"id": m.id, "name": m.name} for m in response.auth_methods
                    ]

    except FileNotFoundError:
        result["status"] = "not_installed"
        result["error"] = f"Command '{command}' not found"
    except TimeoutError:
        result["status"] = "timeout"
        result["error"] = "Initialization timed out (15s)"
    except Exception as e:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = str(e)

    return result


def format_capabilities_output(caps: dict[str, Any]) -> str:
    """Format capabilities dict as human-readable output for snapshot."""
    lines = [
        f"=== {caps['agent_class']} ===",
        "",
        f"Status: {caps['status']}",
    ]

    if "command" in caps:
        cmd_str = f"{caps['command']} {' '.join(caps.get('args', []))}"
        lines.append(f"Command: {cmd_str.strip()}")

    if caps["status"] == "ok":
        lines.append("")

        if "agent_info" in caps:
            info = caps["agent_info"]
            lines.append("Agent Info:")
            lines.append(f"  name: {info.get('name')}")
            lines.append(f"  title: {info.get('title')}")
            lines.append(f"  version: {info.get('version')}")

        lines.append("")
        lines.append(f"Protocol Version: {caps.get('protocol_version')}")

        if "capabilities" in caps:
            lines.append("")
            lines.append("Capabilities:")
            # Sort for consistent output
            lines.append(json.dumps(caps["capabilities"], indent=2, sort_keys=True))

        if "auth_methods" in caps:
            lines.append("")
            lines.append("Auth Methods:")
            lines.extend(f"  - {method['id']}: {method['name']}" for method in caps["auth_methods"])

    elif "error" in caps:
        lines.append(f"Error: {caps['error']}")

    return "\n".join(lines)


@pytest.mark.acp_snapshot
@pytest.mark.parametrize(
    "agent_class",
    get_concrete_agent_classes(),
    ids=[cls.__name__ for cls in get_concrete_agent_classes()],
)
def test_acp_agent_capabilities(agent_class: type[BaseACPAgentConfig], snapshot):
    """Test that ACP agent capabilities match snapshot.

    This test probes each agent via the ACP initialize handshake and captures
    the reported capabilities. Changes in MCP support, prompt capabilities,
    or other features will trigger snapshot updates.

    To update snapshots after intentional changes:
        uv run pytest tests/test_acp_capabilities_snapshots.py -m acp_snapshot --snapshot-update
    """
    caps = asyncio.run(get_agent_capabilities(agent_class))
    output = format_capabilities_output(caps)
    assert output == snapshot


if __name__ == "__main__":
    pytest.main([__file__, "-m", "acp_snapshot", "-v"])
