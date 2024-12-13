"""Built-in commands for LLMling agent."""

from llmling_agent.commands.base import BaseCommand
from llmling_agent.commands.builtin.hello import hello_command
from llmling_agent.commands.builtin.help_cmd import help_cmd
from llmling_agent.commands.builtin.prompts import list_prompts_cmd, prompt_cmd
from llmling_agent.commands.builtin.agents import list_agents_cmd, switch_agent_cmd
from llmling_agent.commands.builtin.session import clear_cmd, reset_cmd
from llmling_agent.commands.builtin.utils import copy_clipboard_cmd, edit_env_cmd
from llmling_agent.commands.builtin.tools import (
    list_tools_cmd,
    tool_info_cmd,
    enable_tool_cmd,
    disable_tool_cmd,
)


def get_builtin_commands() -> list[BaseCommand]:
    """Get list of built-in commands."""
    return [
        hello_command,
        help_cmd,
        list_prompts_cmd,
        prompt_cmd,
        switch_agent_cmd,
        list_agents_cmd,
        clear_cmd,
        reset_cmd,
        copy_clipboard_cmd,
        edit_env_cmd,
        list_tools_cmd,
        tool_info_cmd,
        enable_tool_cmd,
        disable_tool_cmd,
    ]
