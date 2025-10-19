"""Built-in commands for LLMling agent."""

from __future__ import annotations


from llmling_agent_commands.agents import (
    CreateAgentCommand,
    ListAgentsCommand,
    ShowAgentCommand,
    # SwitchAgentCommand,
)
from llmling_agent_commands.connections import (
    ConnectCommand,
    DisconnectCommand,
    ListConnectionsCommand,
    DisconnectAllCommand,
)
from llmling_agent_commands.env import OpenEnvFileCommand, SetEnvCommand
from llmling_agent_commands.models import SetModelCommand
from llmling_agent_commands.prompts import ListPromptsCommand, ShowPromptCommand
from llmling_agent_commands.resources import (
    ListResourcesCommand,
    ShowResourceCommand,
    AddResourceCommand,
)
from llmling_agent_commands.session import ClearCommand, ResetCommand
from llmling_agent_commands.read import ReadCommand
from llmling_agent_commands.tools import (
    DisableToolCommand,
    EnableToolCommand,
    ListToolsCommand,
    RegisterToolCommand,
    ShowToolCommand,
    WriteToolCommand,
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
        SetModelCommand,
        SetEnvCommand,
        # Tool management
        ListToolsCommand,
        ShowToolCommand,
        EnableToolCommand,
        DisableToolCommand,
        RegisterToolCommand,
        WriteToolCommand,
        # Resource management
        ListResourcesCommand,
        ShowResourceCommand,
        AddResourceCommand,
        # Prompt management
        ListPromptsCommand,
        ShowPromptCommand,
        # Worker management (all from current agent's perspective)
        AddWorkerCommand,
        RemoveWorkerCommand,
        ListWorkersCommand,
        # Connection management (all from current agent's perspective)
        ConnectCommand,  # "Connect THIS agent to another one"
        DisconnectCommand,  # "Disconnect THIS agent from another"
        ListConnectionsCommand,  # "Show THIS agent's connections"
        DisconnectAllCommand,  # "Disconnect THIS agent from all others"
        # Context/Content
        ReadCommand,
    ]


def get_pool_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get commands that operate on multiple agents or the pool itself."""
    return [
        # Pool-level agent management
        CreateAgentCommand,  # Creates new agent in pool
        ListAgentsCommand,  # Shows all agents in pool
        ShowAgentCommand,  # Shows config from pool's manifest
        # SwitchAgentCommand,  # Changes active agent in pool
        # Pool configuration
        OpenEnvFileCommand,  # Edits pool's environment config
        EditAgentFileCommand,  # Edits pool's manifest
    ]


def get_commands() -> list[BaseCommand | type[SlashedCommand]]:
    """Get all built-in commands."""
    return [
        *get_agent_commands(),
        *get_pool_commands(),
    ]
