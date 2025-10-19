"""Built-in commands for LLMling agent."""

from __future__ import annotations


from llmling_agent_commands.agents import (
    create_agent_cmd,
    list_agents_cmd,
    show_agent_cmd,
    # switch_agent_cmd,
)
from llmling_agent_commands.connections import (
    connect_cmd,
    disconnect_cmd,
    connections_cmd,
    disconnect_all_cmd,
)
from llmling_agent_commands.env import edit_env_cmd, set_env_cmd
from llmling_agent_commands.models import set_model_cmd
from llmling_agent_commands.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent_commands.resources import (
    ListResourcesCommand,
    ShowResourceCommand,
    AddResourceCommand,
)
from llmling_agent_commands.session import ClearCommand, ResetCommand
from llmling_agent_commands.read import read_cmd
from llmling_agent_commands.tools import (
    DisableToolCommand,
    EnableToolCommand,
    ListToolsCommand,
    RegisterToolCommand,
    ShowToolCommand,
)
from llmling_agent_commands.workers import (
    AddWorkerCommand,
    RemoveWorkerCommand,
    ListWorkersCommand,
)
from llmling_agent_commands.utils import CopyClipboardCommand, EditAgentFileCommand
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slashed import BaseCommand, SlashedCommand


def get_agent_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get commands that operate primarily on a single agent."""
    return [
        # Session/History management
        ClearCommand,
        ResetCommand,
        CopyClipboardCommand,  # operates on current agent's history
        # Model/Environment
        set_model_cmd,
        set_env_cmd,
        # Tool management
        ListToolsCommand,
        ShowToolCommand,
        EnableToolCommand,
        DisableToolCommand,
        RegisterToolCommand,
        # Resource management
        ListResourcesCommand,
        ShowResourceCommand,
        AddResourceCommand,
        # Prompt management
        list_prompts_cmd,
        prompt_cmd,
        # Worker management (all from current agent's perspective)
        AddWorkerCommand,
        RemoveWorkerCommand,
        ListWorkersCommand,
        # Connection management (all from current agent's perspective)
        connect_cmd,  # "Connect THIS agent to another one"
        disconnect_cmd,  # "Disconnect THIS agent from another"
        connections_cmd,  # "Show THIS agent's connections"
        disconnect_all_cmd,  # "Disconnect THIS agent from all others"
        # Context/Content
        read_cmd,
    ]


def get_pool_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get commands that operate on multiple agents or the pool itself."""
    return [
        # Pool-level agent management
        create_agent_cmd,  # Creates new agent in pool
        list_agents_cmd,  # Shows all agents in pool
        show_agent_cmd,  # Shows config from pool's manifest
        # switch_agent_cmd,  # Changes active agent in pool
        # Pool configuration
        edit_env_cmd,  # Edits pool's environment config
        EditAgentFileCommand,  # Edits pool's manifest
    ]


def get_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get all built-in commands."""
    return [
        *get_agent_commands(),
        *get_pool_commands(),
    ]
