"""Built-in toolsets for agent capabilities."""

from __future__ import annotations

# Import factory functions for backward compatibility
from llmling_agent_toolsets.builtin.agent_management import create_agent_management_tools
from llmling_agent_toolsets.builtin.code import create_code_tools
from llmling_agent_toolsets.builtin.code_execution import create_code_execution_tools
from llmling_agent_toolsets.builtin.file_access import create_file_access_tools
from llmling_agent_toolsets.builtin.history import create_history_tools
from llmling_agent_toolsets.builtin.process_management import (
    create_process_management_tools,
)
from llmling_agent_toolsets.builtin.resource_access import create_resource_access_tools
from llmling_agent_toolsets.builtin.tool_management import create_tool_management_tools
from llmling_agent_toolsets.builtin.user_interaction import create_user_interaction_tools

# Import provider classes
from llmling_agent_toolsets.builtin.agent_management import AgentManagementTools
from llmling_agent_toolsets.builtin.code import CodeTools
from llmling_agent_toolsets.builtin.code_execution import CodeExecutionTools
from llmling_agent_toolsets.builtin.file_access import FileAccessTools
from llmling_agent_toolsets.builtin.history import HistoryTools
from llmling_agent_toolsets.builtin.integration import IntegrationTools
from llmling_agent_toolsets.builtin.process_management import ProcessManagementTools
from llmling_agent_toolsets.builtin.resource_access import ResourceAccessTools
from llmling_agent_toolsets.builtin.tool_management import ToolManagementTools
from llmling_agent_toolsets.builtin.user_interaction import UserInteractionTools


__all__ = [
    # Provider classes
    "AgentManagementTools",
    "CodeExecutionTools",
    "CodeTools",
    "FileAccessTools",
    "HistoryTools",
    "IntegrationTools",
    "ProcessManagementTools",
    "ResourceAccessTools",
    "ToolManagementTools",
    "UserInteractionTools",
    # Factory functions
    "create_agent_management_tools",
    "create_code_execution_tools",
    "create_code_tools",
    "create_file_access_tools",
    "create_history_tools",
    "create_process_management_tools",
    "create_resource_access_tools",
    "create_tool_management_tools",
    "create_user_interaction_tools",
]
