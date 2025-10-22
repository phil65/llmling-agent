"""Built-in toolsets for agent capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.resource_providers.static import StaticResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from llmling import RuntimeConfig

    from llmling_agent.tools.skills import SkillsRegistry


def create_agent_management_tools() -> list[Tool]:
    """Create tools for agent and team management operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.delegate_to,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.list_available_agents,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.list_available_teams,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.create_worker_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.spawn_delegate,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.add_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.add_team,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.ask_agent,
            source="builtin",
            category="other",
        ),
        Tool.from_callable(
            capability_tools.connect_nodes,
            source="builtin",
            category="other",
        ),
    ]


def create_file_access_tools() -> list[Tool]:
    """Create tools for file and directory access operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.read_file,
            source="builtin",
            category="read",
        ),
        Tool.from_callable(
            capability_tools.list_directory,
            source="builtin",
            category="search",
        ),
    ]


def create_code_execution_tools() -> list[Tool]:
    """Create tools for code execution operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.execute_python,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.execute_command,
            source="builtin",
            category="execute",
        ),
    ]


def create_process_management_tools() -> list[Tool]:
    """Create tools for process management operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.start_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.get_process_output,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.wait_for_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.kill_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.release_process,
            source="builtin",
            category="execute",
        ),
        Tool.from_callable(
            capability_tools.list_processes,
            source="builtin",
            category="search",
        ),
    ]


def create_resource_access_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for resource access operations."""
    tools: list[Tool] = []

    # Resource tools require runtime
    if runtime:
        tools.extend([
            Tool.from_callable(
                runtime.load_resource,
                source="builtin",
                category="read",
            ),
            Tool.from_callable(
                runtime.get_resources,
                source="builtin",
                category="search",
            ),
        ])

    return tools


def create_tool_management_tools(runtime: RuntimeConfig | None = None) -> list[Tool]:
    """Create tools for tool management operations."""
    tools: list[Tool] = []

    # Tool management requires runtime
    if runtime:
        tools.extend([
            Tool.from_callable(
                runtime.register_tool,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                runtime.register_code_tool,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                runtime.install_package,
                source="builtin",
                category="execute",
            ),
        ])

    return tools


def create_user_interaction_tools() -> list[Tool]:
    """Create tools for user interaction operations."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.ask_user,
            source="builtin",
            category="other",
        ),
    ]


def create_history_tools() -> list[Tool]:
    """Create tools for history and statistics access."""
    from llmling_agent_tools import capability_tools

    return [
        Tool.from_callable(
            capability_tools.search_history,
            source="builtin",
            category="search",
        ),
        Tool.from_callable(
            capability_tools.show_statistics,
            source="builtin",
            category="read",
        ),
    ]


# Provider factory functions
class AgentManagementTools(StaticResourceProvider):
    """Provider for agent management tools."""

    def __init__(self, name: str = "agent_management"):
        super().__init__(name=name, tools=create_agent_management_tools())


class FileAccessTools(StaticResourceProvider):
    """Provider for file access tools."""

    def __init__(self, name: str = "file_access"):
        super().__init__(name=name, tools=create_file_access_tools())


class CodeExecutionTools(StaticResourceProvider):
    """Provider for code execution tools."""

    def __init__(self, name: str = "code_execution"):
        super().__init__(name=name, tools=create_code_execution_tools())


class ProcessManagementTools(StaticResourceProvider):
    """Provider for process management tools."""

    def __init__(self, name: str = "process_management"):
        super().__init__(name=name, tools=create_process_management_tools())


class ResourceAccessTools(StaticResourceProvider):
    """Provider for resource access tools."""

    def __init__(
        self, name: str = "resource_access", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_resource_access_tools(runtime))


class ToolManagementTools(StaticResourceProvider):
    """Provider for tool management tools."""

    def __init__(
        self, name: str = "tool_management", runtime: RuntimeConfig | None = None
    ):
        super().__init__(name=name, tools=create_tool_management_tools(runtime))


class UserInteractionTools(StaticResourceProvider):
    """Provider for user interaction tools."""

    def __init__(self, name: str = "user_interaction"):
        super().__init__(name=name, tools=create_user_interaction_tools())


class HistoryTools(StaticResourceProvider):
    """Provider for history tools."""

    def __init__(self, name: str = "history"):
        super().__init__(name=name, tools=create_history_tools())


class IntegrationTools(ResourceProvider):
    """Provider for integration tools."""

    def __init__(
        self, name: str = "integrations", skills_registry: SkillsRegistry | None = None
    ):
        super().__init__(name)
        self.skills_registry = skills_registry

    async def get_tools(self) -> list[Tool]:
        """Get integration tools with dynamic skill tool."""
        from llmling_agent_tools import capability_tools

        tools = [
            Tool.from_callable(
                capability_tools.add_local_mcp_server,
                source="builtin",
                category="other",
            ),
            Tool.from_callable(
                capability_tools.add_remote_mcp_server,
                source="builtin",
                category="other",
            ),
        ]

        # Add skill loading tool if registry is available
        if self.skills_registry:
            await self.skills_registry.discover_skills()

            # Create skill tool with dynamic description including available skills
            base_desc = """Load a Claude Code Skill and return its instructions.

This tool provides access to Claude Code Skills - specialized workflows and techniques
for handling specific types of tasks. When you need to use a skill, call this tool
with the skill name.

Available skills:"""

            if self.skills_registry.is_empty:
                description = base_desc + "\n(No skills found in configured directories)"
            else:
                skills_list = []
                for skill_name in self.skills_registry.list_items():
                    skill = self.skills_registry.get(skill_name)
                    skills_list.append(f"- {skill.name}: {skill.description}")
                description = base_desc + "\n" + "\n".join(skills_list)

            skill_tool = Tool.from_callable(
                capability_tools.load_skill,
                source="builtin",
                category="read",
                description_override=description,
            )
            tools.append(skill_tool)

        return tools
