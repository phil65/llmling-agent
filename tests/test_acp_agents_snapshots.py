"""Snapshot tests for ACP agent CLI help output.

These tests capture the help output of all ACP agent configurations
to detect changes in command-line arguments and options.

Run with: uv run pytest tests/test_acp_agents_snapshots.py -m acp_cli_snapshot
"""

from __future__ import annotations

import inspect
from typing import get_args

import pytest

from llmling_agent.models.acp_agents import (
    ACPAgentConfig,
    AmpACPAgentConfig,
    AuggieACPAgentConfig,
    BaseACPAgentConfig,
    CagentACPAgentConfig,
    ClaudeACPAgentConfig,
    CodexACPAgentConfig,
    CursorACPAgentConfig,
    FastAgentACPAgentConfig,
    GeminiACPAgentConfig,
    GooseACPAgentConfig,
    KimiACPAgentConfig,
    MistralACPAgentConfig,
    OpenCodeACPAgentConfig,
    OpenHandsACPAgentConfig,
    StakpakACPAgentConfig,
    VTCodeACPAgentConfig,
)


# All concrete ACP agent config classes
ACP_AGENT_CLASSES = [
    ACPAgentConfig,
    ClaudeACPAgentConfig,
    GeminiACPAgentConfig,
    CodexACPAgentConfig,
    OpenCodeACPAgentConfig,
    GooseACPAgentConfig,
    MistralACPAgentConfig,
    OpenHandsACPAgentConfig,
    FastAgentACPAgentConfig,
    AmpACPAgentConfig,
    AuggieACPAgentConfig,
    CagentACPAgentConfig,
    KimiACPAgentConfig,
    StakpakACPAgentConfig,
    VTCodeACPAgentConfig,
    CursorACPAgentConfig,
]


def get_help_output(agent_class: type[BaseACPAgentConfig]) -> str:
    """Generate help-like output for an ACP agent configuration.

    Args:
        agent_class: The agent configuration class to inspect.

    Returns:
        A formatted string showing the command, arguments, and field information.
    """
    try:
        # Create instance with minimal required fields
        instance = agent_class()
    except Exception as e:  # noqa: BLE001
        return f"Error creating instance: {e}"

    lines = [
        f"=== {agent_class.__name__} ===",
        "",
        "Command:",
        f"  {instance.get_command()}",
        "",
        "Arguments:",
    ]

    try:
        args = instance.get_args()
        if args:
            for arg in args:
                lines.append(f"  {arg}")  # noqa: PERF401
        else:
            lines.append("  (no arguments)")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  Error getting args: {e}")

    lines.extend(["", "Fields:"])

    # Get all fields from the model
    for field_name, field_info in agent_class.model_fields.items():
        field_type = field_info.annotation
        default = field_info.default

        # Format type hint
        type_str = str(field_type)
        if hasattr(field_type, "__name__"):
            type_str = field_type.__name__  # pyright: ignore[reportOptionalMemberAccess]
        elif hasattr(field_type, "__origin__"):
            # Handle generic types
            origin = getattr(field_type, "__origin__", None)
            args = get_args(field_type)
            if origin and args:
                type_str = (
                    f"{getattr(origin, '__name__', origin)}[{', '.join(str(a) for a in args)}]"
                )

        # Format default value
        if default is None:
            default_str = "None"
        elif default == ...:
            default_str = "REQUIRED"
        elif callable(default):
            default_str = f"<factory: {default.__name__}>"
        else:
            default_str = repr(default)

        description = field_info.description or ""
        if description:
            description = f" - {description}"

        lines.append(f"  {field_name}: {type_str} = {default_str}{description}")

    # Add model providers if available
    lines.extend(["", "Model Providers:"])
    try:
        providers = instance.model_providers
        if providers:
            for provider in providers:
                lines.append(f"  - {provider}")  # noqa: PERF401
        else:
            lines.append("  (none)")
    except Exception as e:  # noqa: BLE001
        lines.append(f"  Error getting providers: {e}")

    return "\n".join(lines)


@pytest.mark.acp_cli_snapshot
@pytest.mark.parametrize(
    "agent_class",
    ACP_AGENT_CLASSES,
    ids=[cls.__name__ for cls in ACP_AGENT_CLASSES],
)
def test_acp_agent_help_output(agent_class: type[BaseACPAgentConfig], snapshot):
    """Test that ACP agent help output matches snapshot.

    This test captures the CLI interface of each ACP agent to detect
    when command-line arguments, options, or configuration fields change.

    To update snapshots after intentional changes:
        uv run pytest tests/test_acp_agents_snapshots.py -m acp_cli_snapshot --snapshot-update
    """
    help_output = get_help_output(agent_class)
    assert help_output == snapshot


@pytest.mark.acp_cli_snapshot
def test_all_agent_classes_covered():
    """Verify all ACP agent classes are included in tests.

    This test ensures we don't forget to add new agent classes to the test suite.
    """
    # Import the module and find all classes that inherit from BaseACPAgentConfig
    import llmling_agent.models.acp_agents as acp_module

    # Exclude abstract base classes
    abstract_classes = {BaseACPAgentConfig, acp_module.MCPCapableACPAgentConfig}

    found_classes = []
    for name, obj in inspect.getmembers(acp_module, inspect.isclass):
        if (
            issubclass(obj, BaseACPAgentConfig)
            and obj not in abstract_classes
            and not name.startswith("_")
            and name.endswith("ACPAgentConfig")
        ):
            found_classes.append(obj)

    # Sort for consistent output
    found_classes.sort(key=lambda cls: cls.__name__)
    tested_classes = sorted(ACP_AGENT_CLASSES, key=lambda cls: cls.__name__)

    found_names = [cls.__name__ for cls in found_classes]
    tested_names = [cls.__name__ for cls in tested_classes]

    missing = set(found_names) - set(tested_names)
    extra = set(tested_names) - set(found_names)

    result = ["Agent Class Coverage:", ""]
    result.append(f"Found: {len(found_names)} classes")
    result.append(f"Tested: {len(tested_names)} classes")
    result.append("")

    if missing:
        result.append("Missing from tests:")
        for name in sorted(missing):
            result.append(f"  - {name}")  # noqa: PERF401
        result.append("")

    if extra:
        result.append("Extra in tests (not found in module):")
        for name in sorted(extra):
            result.append(f"  - {name}")  # noqa: PERF401
        result.append("")

    result.append("All tested classes:")
    for name in tested_names:
        result.append(f"  ✓ {name}")  # noqa: PERF401

    "\n".join(result)

    assert not missing, f"Missing agent classes in tests: {missing}"
    assert not extra, f"Extra agent classes in tests that don't exist: {extra}"


if __name__ == "__main__":
    pytest.main([__file__, "-m", "acp_cli_snapshot", "-v"])
