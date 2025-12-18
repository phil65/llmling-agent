"""Agent-related CLI commands."""

from __future__ import annotations

import shutil
from typing import Annotated

import typer as t

from llmling_agent.utils.inspection import validate_import
from llmling_agent_cli import agent_store, resolve_agent_config
from llmling_agent_cli.common import OutputFormat, format_output, output_format_opt, verbose_opt


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
        validate_import("promptantic", "chat")
        from promptantic import ModelGenerator

        from llmling_agent import AgentsManifest

        generator = ModelGenerator()
        manifest = generator.populate(AgentsManifest)
        manifest.save(output)
    else:
        from llmling_agent import config_resources

        shutil.copy2(config_resources.AGENTS_TEMPLATE, output)

    config_name = name or Path(output).stem
    agent_store.add_config(config_name, output)
    agent_store.set_active(config_name)

    print(f"\nCreated and activated agent configuration {config_name!r}: {output}")
    print("\nTry these commands:")
    print("  llmling-agent list")
    print("  llmling-agent chat simple_agent")


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


@agent_cli.command("list")
def list_agents(
    config_name: Annotated[
        str | None,
        t.Option(
            "-c",
            "--config",
            help="Name of agent configuration to list (defaults to active)",
        ),
    ] = None,
    output_format: OutputFormat = output_format_opt,
    verbose: bool = verbose_opt,
) -> None:
    """List agents from the active (or specified) configuration."""
    from llmling_agent import AgentsManifest

    try:
        try:
            config_path = resolve_agent_config(config_name)
        except ValueError as e:
            msg = str(e)
            raise t.BadParameter(msg) from e

        agent_def = AgentsManifest.from_file(config_path)
        # Set the name field from the dict key for each agent
        agents = [ag.model_copy(update={"name": n}) for n, ag in agent_def.agents.items()]
        format_output(agents, output_format)

    except t.Exit:
        raise
    except Exception as e:
        t.echo(f"Error: {e}", err=True)
        if verbose:
            import traceback

            t.echo(traceback.format_exc(), err=True)
        raise t.Exit(1) from e
