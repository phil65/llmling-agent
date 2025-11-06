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
)
from llmling_agent_commands.workers import (
    AddWorkerCommand,
    RemoveWorkerCommand,
    ListWorkersCommand,
)
from llmling_agent_commands.utils import CopyClipboardCommand, EditAgentFileCommand
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from slashed import BaseCommand, SlashedCommand


def get_agent_commands(**kwargs) -> Sequence[BaseCommand | type[SlashedCommand]]:
    """Get commands that operate primarily on a single agent."""
    command_map = {
        "enable_clear": ClearCommand,
        "enable_reset": ResetCommand,
        "enable_copy_clipboard": CopyClipboardCommand,
        "enable_set_model": SetModelCommand,
        "enable_list_tools": ListToolsCommand,
        "enable_show_tool": ShowToolCommand,
        "enable_enable_tool": EnableToolCommand,
        "enable_disable_tool": DisableToolCommand,
        "enable_register_tool": RegisterToolCommand,
        "enable_list_resources": ListResourcesCommand,
        "enable_show_resource": ShowResourceCommand,
        "enable_add_resource": AddResourceCommand,
        "enable_list_prompts": ListPromptsCommand,
        "enable_show_prompt": ShowPromptCommand,
        "enable_add_worker": AddWorkerCommand,
        "enable_remove_worker": RemoveWorkerCommand,
        "enable_list_workers": ListWorkersCommand,
        "enable_connect": ConnectCommand,
        "enable_disconnect": DisconnectCommand,
        "enable_list_connections": ListConnectionsCommand,
        "enable_disconnect_all": DisconnectAllCommand,
        "enable_read": ReadCommand,
    }

    commands = []
    for flag, command in command_map.items():
        if kwargs.get(flag, True):
            commands.append(command)

    return commands


def get_pool_commands(**kwargs) -> Sequence[BaseCommand | type[SlashedCommand]]:
    """Get commands that operate on multiple agents or the pool itself."""
    command_map = {
        "enable_create_agent": CreateAgentCommand,
        "enable_list_agents": ListAgentsCommand,
        "enable_show_agent": ShowAgentCommand,
        "enable_edit_agent_file": EditAgentFileCommand,
    }

    commands = []
    for flag, command in command_map.items():
        if kwargs.get(flag, True):
            commands.append(command)

    return commands


def get_commands(
    *,
    enable_clear: bool = True,
    enable_reset: bool = True,
    enable_copy_clipboard: bool = True,
    enable_set_model: bool = True,
    enable_list_tools: bool = True,
    enable_show_tool: bool = True,
    enable_enable_tool: bool = True,
    enable_disable_tool: bool = True,
    enable_register_tool: bool = True,
    enable_list_resources: bool = True,
    enable_show_resource: bool = True,
    enable_add_resource: bool = True,
    enable_list_prompts: bool = True,
    enable_show_prompt: bool = True,
    enable_add_worker: bool = True,
    enable_remove_worker: bool = True,
    enable_list_workers: bool = True,
    enable_connect: bool = True,
    enable_disconnect: bool = True,
    enable_list_connections: bool = True,
    enable_disconnect_all: bool = True,
    enable_read: bool = True,
    enable_create_agent: bool = True,
    enable_list_agents: bool = True,
    enable_show_agent: bool = True,
    enable_edit_agent_file: bool = True,
) -> list[BaseCommand | type[SlashedCommand]]:
    """Get all built-in commands."""
    agent_kwargs = {
        "enable_clear": enable_clear,
        "enable_reset": enable_reset,
        "enable_copy_clipboard": enable_copy_clipboard,
        "enable_set_model": enable_set_model,
        "enable_list_tools": enable_list_tools,
        "enable_show_tool": enable_show_tool,
        "enable_enable_tool": enable_enable_tool,
        "enable_disable_tool": enable_disable_tool,
        "enable_register_tool": enable_register_tool,
        "enable_list_resources": enable_list_resources,
        "enable_show_resource": enable_show_resource,
        "enable_add_resource": enable_add_resource,
        "enable_list_prompts": enable_list_prompts,
        "enable_show_prompt": enable_show_prompt,
        "enable_add_worker": enable_add_worker,
        "enable_remove_worker": enable_remove_worker,
        "enable_list_workers": enable_list_workers,
        "enable_connect": enable_connect,
        "enable_disconnect": enable_disconnect,
        "enable_list_connections": enable_list_connections,
        "enable_disconnect_all": enable_disconnect_all,
        "enable_read": enable_read,
    }
    pool_kwargs = {
        "enable_create_agent": enable_create_agent,
        "enable_list_agents": enable_list_agents,
        "enable_show_agent": enable_show_agent,
        "enable_edit_agent_file": enable_edit_agent_file,
    }

    return [
        *get_agent_commands(**agent_kwargs),
        *get_pool_commands(**pool_kwargs),
    ]
