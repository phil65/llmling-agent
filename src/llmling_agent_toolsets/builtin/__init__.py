"""Built-in toolsets for agent capabilities."""

from __future__ import annotations

# Import factory functions for backward compatibility
from .factories import (
    create_agent_management_tools,
    create_code_execution_tools,
    create_code_tools,
    create_file_access_tools,
    create_history_tools,
    create_process_management_tools,
    create_resource_access_tools,
    create_tool_management_tools,
    create_user_interaction_tools,
)

# Import provider classes
from .agent_management import AgentManagementTools
from .code import CodeTools
from .code_execution import CodeExecutionTools
from .file_access import FileAccessTools
from .history import HistoryTools
from .integration import IntegrationTools
from .process_management import ProcessManagementTools
from .resource_access import ResourceAccessTools
from .tool_management import ToolManagementTools
from .user_interaction import UserInteractionTools


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
