"""Resource provider for agent capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

from llmling_agent.resource_providers.base import ResourceProvider
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import Callable

    from llmling import RuntimeConfig

    from llmling_agent.config.capabilities import Capabilities
    from llmling_agent.tools.skills import SkillsRegistry


class CapabilitiesResourceProvider(ResourceProvider):
    """Provides tools based on agent capabilities."""

    def __init__(
        self,
        capabilities: Capabilities,
        runtime: RuntimeConfig | None = None,
        name: str = "capability_tools",
        skills_registry: SkillsRegistry | None = None,
    ):
        super().__init__(name)
        self.capabilities = capabilities
        self.runtime = runtime
        self.skills_registry: SkillsRegistry | None = skills_registry

    async def get_tools(self) -> list[Tool]:  # noqa: PLR0915
        """Get all tools enabled by current capabilities."""
        tools: list[Tool] = []

        # Resource tools (require runtime)
        if self.runtime:
            if self.capabilities.can_load_resources:
                tool = Tool.from_callable(
                    self.runtime.load_resource,
                    source="builtin",
                    requires_capability="can_load_resources",
                    category="read",
                )
                tools.append(tool)
            if self.capabilities.can_list_resources:
                tool = Tool.from_callable(
                    self.runtime.get_resources,
                    source="builtin",
                    requires_capability="can_list_resources",
                    category="search",
                )
                tools.append(tool)
            if self.capabilities.can_register_tools:
                tool = Tool.from_callable(
                    self.runtime.register_tool,
                    source="builtin",
                    requires_capability="can_register_tools",
                    category="other",
                )
                tools.append(tool)
            if self.capabilities.can_register_code:
                tool = Tool.from_callable(
                    self.runtime.register_code_tool,
                    source="builtin",
                    requires_capability="can_register_code",
                    category="other",
                )
                tools.append(tool)
            if self.capabilities.can_install_packages:
                tool = Tool.from_callable(
                    self.runtime.install_package,
                    source="builtin",
                    requires_capability="can_install_packages",
                    category="execute",
                )
                tools.append(tool)

        # Agent/team tools
        from llmling_agent_tools import capability_tools

        if self.capabilities.can_create_workers:
            tool = Tool.from_callable(
                capability_tools.create_worker_agent,
                source="builtin",
                requires_capability="can_create_workers",
                category="other",
            )
            tools.append(tool)
        if self.capabilities.can_create_delegates:
            tool = Tool.from_callable(
                capability_tools.spawn_delegate,
                source="builtin",
                requires_capability="can_create_delegates",
                category="other",
            )
            tools.append(tool)
        if self.capabilities.can_list_agents:
            tool = Tool.from_callable(
                capability_tools.list_available_agents,
                source="builtin",
                requires_capability="can_list_agents",
                category="search",
            )
            tools.append(tool)
        if self.capabilities.can_list_teams:
            tool = Tool.from_callable(
                capability_tools.list_available_teams,
                source="builtin",
                requires_capability="can_list_teams",
                category="search",
            )
            tools.append(tool)
        if self.capabilities.can_delegate_tasks:
            tool = Tool.from_callable(
                capability_tools.delegate_to,
                source="builtin",
                requires_capability="can_delegate_tasks",
                category="other",
            )
            tools.append(tool)

        # History and stats tools
        if self.capabilities.history_access != "none":
            tool = Tool.from_callable(
                capability_tools.search_history,
                source="builtin",
                requires_capability="history_access",
                category="search",
            )
            tools.append(tool)
            tool = Tool.from_callable(
                capability_tools.show_statistics,
                source="builtin",
                requires_capability="history_access",
                category="read",
            )
            tools.append(tool)

        # Agent/team management
        if self.capabilities.can_add_agents:
            tool = Tool.from_callable(
                capability_tools.add_agent,
                source="builtin",
                requires_capability="can_add_agents",
                category="other",
            )
            tools.append(tool)
        if self.capabilities.can_add_teams:
            tool = Tool.from_callable(
                capability_tools.add_team,
                source="builtin",
                requires_capability="can_add_teams",
                category="other",
            )
            tools.append(tool)

        if self.capabilities.can_connect_nodes:
            tool = Tool.from_callable(
                capability_tools.connect_nodes,
                source="builtin",
                requires_capability="can_can_connect_nodes",
                category="other",
            )
            tools.append(tool)

        if self.capabilities.can_ask_agents:
            tool = Tool.from_callable(
                capability_tools.ask_agent,
                source="builtin",
                requires_capability="can_ask_agents",
                category="other",
            )
            tools.append(tool)
        if self.capabilities.can_read_files:
            tool = Tool.from_callable(
                capability_tools.read_file,
                source="builtin",
                requires_capability="can_read_files",
                category="read",
            )
            tools.append(tool)
        if self.capabilities.can_list_directories:
            tool = Tool.from_callable(
                capability_tools.list_directory,
                source="builtin",
                requires_capability="can_list_directories",
                category="search",
            )
            tools.append(tool)

        # Execution tools
        if self.capabilities.can_execute_code:
            tool = Tool.from_callable(
                capability_tools.execute_python,
                source="builtin",
                requires_capability="can_execute_code",
                category="execute",
            )
            tools.append(tool)
        if self.capabilities.can_execute_commands:
            tool = Tool.from_callable(
                capability_tools.execute_command,
                source="builtin",
                requires_capability="can_execute_commands",
                category="execute",
            )
            tools.append(tool)

        # Process management tools
        if self.capabilities.can_manage_processes:
            process_tools: list[Callable] = [
                capability_tools.start_process,
                capability_tools.get_process_output,
                capability_tools.wait_for_process,
                capability_tools.kill_process,
                capability_tools.release_process,
                capability_tools.list_processes,
            ]
            for tool_func in process_tools:
                tool = Tool.from_callable(
                    tool_func,
                    source="builtin",
                    requires_capability="can_manage_processes",
                    category="execute",
                )
                tools.append(tool)

        # User interaction tools
        if self.capabilities.can_ask_user:
            tool = Tool.from_callable(
                capability_tools.ask_user,
                source="builtin",
                requires_capability="can_ask_user",
                category="other",
            )
            tools.append(tool)

        # MCP server management tools
        if self.capabilities.can_add_mcp_servers:
            tool = Tool.from_callable(
                capability_tools.add_local_mcp_server,
                source="builtin",
                requires_capability="can_add_mcp_servers",
                category="other",
            )
            tools.append(tool)
            tool = Tool.from_callable(
                capability_tools.add_remote_mcp_server,
                source="builtin",
                requires_capability="can_add_mcp_servers",
                category="other",
            )
            tools.append(tool)

        # Skill management tools
        if self.capabilities.can_load_skills:
            tool = await self._create_dynamic_skill_tool()
            tools.append(tool)

        return tools

    async def _create_dynamic_skill_tool(self) -> Tool:
        """Create skill tool with dynamic description including available skills."""
        # Use the agent's skills registry
        registry = self.skills_registry
        if registry:
            await registry.discover_skills()

        # Generate dynamic description
        base_desc = """Load a Claude Code Skill and return its instructions.

This tool provides access to Claude Code Skills - specialized workflows and techniques
for handling specific types of tasks. When you need to use a skill, call this tool
with the skill name.

Available skills:"""

        if not registry or registry.is_empty:
            description = base_desc + "\n(No skills found in configured directories)"
        else:
            skills_list = []
            for skill_name in registry.list_items():
                skill = registry.get(skill_name)
                skills_list.append(f"- {skill.name}: {skill.description}")
            description = base_desc + "\n" + "\n".join(skills_list)

        # Create tool with dynamic description
        from llmling_agent_tools import capability_tools

        return Tool.from_callable(
            capability_tools.load_skill,
            source="builtin",
            requires_capability="can_load_skills",
            category="read",
            description_override=description,
        )
