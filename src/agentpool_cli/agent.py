"""Agent-related CLI commands."""

from __future__ import annotations

import shutil
from typing import Annotated

import typer as t

from agentpool_cli import agent_store, resolve_agent_config
from agentpool_cli.common import verbose_opt


agent_cli = t.Typer(help="Agent management commands", no_args_is_help=True)

NAME_HELP = "Name for the configuration (defaults to filename)"

INTERACTIVE_CMD = "--interactive/--no-interactive"
INTERACTIVE_HELP = "Use interactive configuration wizard"


@agent_cli.command("init")
def init_agent_config(
    output: Annotated[str, t.Argument(help="Path to write agent configuration file")],
    name: Annotated[str | None, t.Option("--name", "-n", help=NAME_HELP)] = None,
    interactive: Annotated[bool, t.Option(INTERACTIVE_CMD, help=INTERACTIVE_HELP)] = False,
) -> None:
    """Initialize a new agent configuration file.

    Creates and activates a new agent configuration. The configuration will be
    automatically registered and set as active.
    """
    from pathlib import Path

    if interactive:
        from agentpool.utils.inspection import validate_import

        validate_import("promptantic", "chat")
        from promptantic import ModelGenerator

        from agentpool import AgentsManifest

        generator = ModelGenerator()
        manifest = generator.populate(AgentsManifest)
        manifest.save(output)
    else:
        from agentpool import config_resources

        shutil.copy2(config_resources.AGENTS_TEMPLATE, output)

    config_name = name or Path(output).stem
    agent_store.add_config(config_name, output)
    agent_store.set_active(config_name)

    print(f"\nCreated and activated agent configuration {config_name!r}: {output}")
    print("\nTry these commands:")
    print("  agentpool list")
    print("  agentpool chat simple_agent")


@agent_cli.command("add")
def add_agent_file(
    name: Annotated[str, t.Argument(help="Name for the agent configuration file")],
    path: Annotated[str, t.Argument(help="Path to agent configuration file")],
    verbose: bool = verbose_opt,
) -> None:
    """Add a new agent configuration file."""
    try:
        agent_store.add_config(name, path)
        t.echo(f"Added agent configuration {name!r} -> {path}")
    except Exception as e:
        t.echo(f"Error adding configuration: {e}", err=True)
        raise t.Exit(1) from e


@agent_cli.command("set")
def set_active_file(
    name: Annotated[str, t.Argument(help="Name of agent configuration to set as active")],
    verbose: bool = verbose_opt,
) -> None:
    """Set the active agent configuration file."""
    try:
        agent_store.set_active(name)
        t.echo(f"Set {name!r} as active agent configuration")
    except Exception as e:
        t.echo(f"Error setting active configuration: {e}", err=True)
        raise t.Exit(1) from e


def list_configs() -> None:
    """List stored agent configurations."""
    configs = agent_store.list_configs()
    active = agent_store.get_active()
    active_name = active.name if active else None

    if not configs:
        t.echo("No configurations stored. Use 'agentpool add' to add one.")
        return

    # Format as markdown table
    t.echo("| Name | Path | Active |")
    t.echo("|------|------|--------|")
    for name, path in configs:
        marker = "✓" if name == active_name else ""
        t.echo(f"| {name} | {path} | {marker} |")


def list_agents(
    config_name: Annotated[
        str | None,
        t.Option(
            "-c",
            "--config",
            help="Name of agent configuration to list (defaults to active)",
        ),
    ] = None,
    verbose: bool = verbose_opt,
) -> None:
    """List agents from the active (or specified) configuration."""
    import yaml

    try:
        try:
            config_path = resolve_agent_config(config_name)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        # Parse YAML directly without loading full manifest
        with open(config_path) as f:
            data = yaml.safe_load(f)

        agents = data.get("agents", {})
        if not agents:
            t.echo("No agents found in configuration.")
            return

        # Format as markdown table
        t.echo("| Name | Type | Display Name | Description |")
        t.echo("|------|------|--------------|-------------|")
        for name, config in agents.items():
            agent_type = config.get("type", "native")
            display = config.get("display_name", name)
            desc = config.get("description", "")
            # Truncate long descriptions
            if len(desc) > 50:
                desc = desc[:47] + "..."
            t.echo(f"| {name} | {agent_type} | {display} | {desc} |")

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
