"""Models for agent configuration."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Literal
from uuid import UUID

from llmling_models.configs import AnyModelConfig  # noqa: TC002
from pydantic import ConfigDict, Field, model_validator
from pydantic_ai import UsageLimits  # noqa: TC002
from schemez import InlineSchemaDef
from tokonomics.model_names import ModelName  # noqa: TC002
from toprompt import render_prompt

from llmling_agent import log
from llmling_agent.common_types import EndStrategy  # noqa: TC001
from llmling_agent.prompts.prompts import BasePrompt, PromptMessage, StaticPrompt
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent.utils.importing import import_class
from llmling_agent_config.knowledge import Knowledge  # noqa: TC001
from llmling_agent_config.nodes import NodeConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001
from llmling_agent_config.session import MemoryConfig, SessionQuery
from llmling_agent_config.system_prompts import PromptConfig  # noqa: TC001
from llmling_agent_config.tools import BaseToolConfig, ToolConfig  # noqa: TC001
from llmling_agent_config.toolsets import ToolsetConfig  # noqa: TC001
from llmling_agent_config.workers import WorkerConfig  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent.tools.base import Tool


ToolConfirmationMode = Literal["always", "never", "per_tool"]
ToolMode = Literal["codemode"]

logger = log.get_logger(__name__)


# Claude Code model alias mapping
CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "sonnet": "anthropic:claude-sonnet-4-20250514",
    "opus": "anthropic:claude-opus-4-20250514",
    "haiku": "anthropic:claude-haiku-3-5-20241022",
}

# Claude Code permissionMode to ToolConfirmationMode mapping
PERMISSION_MODE_MAP: dict[str, ToolConfirmationMode] = {
    "default": "per_tool",
    "acceptEdits": "never",
    "bypassPermissions": "never",
    # 'plan' and 'ignore' don't map well, default to per_tool
}


def parse_agent_file(  # noqa: PLR0915
    file_path: str,
    *,
    skills_registry: Any | None = None,
) -> AgentConfig:
    """Parse agent markdown file to AgentConfig.

    Supports both Claude Code and OpenCode formats with auto-detection.
    Also supports local and remote paths via UPath.

    Format documentation:
    - Claude Code: https://code.claude.com/docs/en/sub-agents.md
    - OpenCode: https://raw.githubusercontent.com/sst/opencode/refs/heads/dev/packages/web/src/content/docs/agents.mdx

    Args:
        file_path: Path to .md file with YAML frontmatter (local or remote)
        skills_registry: Optional skills registry for loading skills

    Claude Code format:
        ```markdown
        ---
        name: agent-name
        description: When to use this agent
        tools: tool1, tool2  # comma-separated
        model: sonnet  # sonnet/opus/haiku/inherit
        permissionMode: default
        skills: skill1, skill2
        ---

        System prompt content.
        ```

    OpenCode format:
        ```markdown
        ---
        description: Agent description
        mode: subagent  # primary/subagent/all
        model: anthropic/claude-sonnet-4-20250514
        temperature: 0.1
        maxSteps: 5
        tools:
          write: false
          edit: false
        permission:
          edit: deny
          bash:
            "git diff": allow
        ---

        System prompt content.
        ```

    Additional llmling-agent fields in frontmatter are also supported.
    """
    from upathtools import to_upath
    import yamling

    path = to_upath(file_path)
    content = path.read_text("utf-8")

    # Extract YAML frontmatter
    frontmatter_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?", content, re.DOTALL)
    if not frontmatter_match:
        msg = f"No YAML frontmatter found in {file_path}"
        raise ValueError(msg)

    try:
        metadata = yamling.load_yaml(frontmatter_match.group(1))
    except yamling.YAMLError as e:
        msg = f"Invalid YAML frontmatter in {file_path}: {e}"
        raise ValueError(msg) from e

    if not isinstance(metadata, dict):
        msg = f"YAML frontmatter must be a dictionary in {file_path}"
        raise ValueError(msg)  # noqa: TRY004

    # Extract system prompt (everything after frontmatter)
    system_prompt = content[frontmatter_match.end() :].strip()

    # Build AgentConfig kwargs
    config_kwargs: dict[str, Any] = {}

    # Required field
    if description := metadata.get("description"):
        config_kwargs["description"] = description

    # Detect format and parse accordingly
    is_opencode = (
        any(key in metadata for key in ["mode", "temperature", "maxSteps", "disable"])
        or ("tools" in metadata and isinstance(metadata["tools"], dict))
        or ("permission" in metadata and isinstance(metadata["permission"], dict))
    )

    # Model handling
    if model := metadata.get("model"):
        if model == "inherit":
            pass  # Leave as None, will use default
        elif model in CLAUDE_MODEL_ALIASES:
            config_kwargs["model"] = CLAUDE_MODEL_ALIASES[model]
        else:
            config_kwargs["model"] = model

    # OpenCode-specific fields
    if is_opencode:
        # Temperature
        if temperature := metadata.get("temperature"):
            logger.debug(
                "OpenCode temperature %r in %s (not directly supported)", temperature, file_path
            )

        # MaxSteps
        if max_steps := metadata.get("maxSteps"):
            logger.debug(
                "OpenCode maxSteps %r in %s (not directly supported)", max_steps, file_path
            )

        # Disable
        if disable := metadata.get("disable"):
            logger.debug("OpenCode disable %r in %s (not directly supported)", disable, file_path)

        # Mode (primary/subagent/all) - we don't have direct equivalent
        if mode := metadata.get("mode"):
            logger.debug("OpenCode mode %r in %s (informational only)", mode, file_path)

        # Permission handling (granular per-tool)
        if permission := metadata.get("permission"):
            # For now, just check if edit permissions are restrictive
            edit_perm = permission.get("edit") if isinstance(permission, dict) else None
            if edit_perm in ("deny", "ask"):
                config_kwargs["requires_tool_confirmation"] = (
                    "always" if edit_perm == "ask" else "never"
                )
            logger.debug("OpenCode permission %r in %s (partial mapping)", permission, file_path)

        # TODO: OpenCode tools dict format (tool_name: bool) - needs toolset integration
        if (tools := metadata.get("tools")) and isinstance(tools, dict):
            logger.debug("OpenCode tools dict %r in %s (not yet supported)", tools, file_path)

    # Claude Code-specific fields
    else:
        # Permission mode mapping
        if permission_mode := metadata.get("permissionMode"):
            if mapped := PERMISSION_MODE_MAP.get(permission_mode):
                config_kwargs["requires_tool_confirmation"] = mapped
            else:
                logger.warning(
                    "Unknown permissionMode %r in %s, using default",
                    permission_mode,
                    file_path,
                )

        # TODO: Claude Code tools string format (comma-separated) - needs toolset integration
        if (tools := metadata.get("tools")) and isinstance(tools, str):
            logger.debug("Claude Code tools string %r in %s (not yet supported)", tools, file_path)

        # Skills handling (Claude Code only)
        if (skills_str := metadata.get("skills")) and skills_registry is not None:
            skill_names = [s.strip() for s in skills_str.split(",")]
            for skill_name in skill_names:
                if skill_name not in skills_registry:
                    logger.warning(
                        "Skill %r from %s not found in registry, ignoring",
                        skill_name,
                        file_path,
                    )

    # System prompt from markdown body
    if system_prompt:
        config_kwargs["system_prompts"] = [system_prompt]

    # Pass through any additional llmling-agent specific fields
    passthrough_fields = {
        "inherits",
        "toolsets",
        "session",
        "output_type",
        "retries",
        "output_retries",
        "end_strategy",
        "avatar",
        "config_file_path",
        "knowledge",
        "workers",
        "debug",
        "usage_limits",
        "tool_mode",
        "display_name",
        "triggers",
    }
    for field in passthrough_fields:
        if field in metadata:
            config_kwargs[field] = metadata[field]

    return AgentConfig(**config_kwargs)


class AgentConfig(NodeConfig):
    """Configuration for a single agent in the system.

    Defines an agent's complete configuration including its model, environment,
    and behavior settings.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/agent_configuration/
    """

    model_config = ConfigDict(
        json_schema_extra={
            "x-icon": "octicon:hubot-16",
            "x-doc-title": "Agent Configuration",
        }
    )

    inherits: str | None = Field(default=None, title="Inheritance source")
    """Name of agent config to inherit from"""

    model: str | ModelName | AnyModelConfig | None = Field(
        default=None,
        examples=["openai:gpt-5-nano"],
        title="Model configuration or name",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/model_configuration/"
        },
    )
    """The model to use for this agent. Can be either a simple model name
    string (e.g. 'openai:gpt-5') or a structured model definition.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/model_configuration/
    """

    tools: list[ToolConfig | str] = Field(
        default_factory=list,
        examples=[
            ["webbrowser:open", "builtins:print"],
            [
                {
                    "type": "import",
                    "import_path": "webbrowser:open",
                    "name": "web_browser",
                }
            ],
        ],
        title="Tool configurations",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/tool_configuration/"
        },
    )
    """A list of tools to register with this agent.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/tool_configuration/
    """

    toolsets: list[ToolsetConfig] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "openapi",
                    "spec": "https://api.example.com/openapi.json",
                    "namespace": "api",
                },
                {
                    "type": "file_access",
                },
                {
                    "type": "composio",
                    "user_id": "user123@example.com",
                    "toolsets": ["github", "slack"],
                },
            ],
        ],
        title="Toolset configurations",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/toolset_configuration/"
        },
    )
    """Toolset configurations for extensible tool collections.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/toolset_configuration/
    """

    session: str | SessionQuery | MemoryConfig | None = Field(
        default=None,
        examples=["main_session", "user_123"],
        title="Session configuration",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/session_configuration/"
        },
    )
    """Session configuration for conversation recovery.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/session_configuration/
    """

    output_type: str | StructuredResponseConfig | None = Field(
        default=None,
        examples=["json_response", "code_output"],
        title="Response type",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/response_configuration/"
        },
    )
    """Name of the response definition to use.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/response_configuration/
    """

    retries: int = Field(default=1, ge=0, examples=[1, 3], title="Model retries")
    """Number of retries for failed operations (maps to pydantic-ai's retries)"""

    output_retries: int | None = Field(
        default=None,
        examples=[1, 3],
        title="Output retries",
    )
    """Max retries for result validation"""

    end_strategy: EndStrategy = Field(
        default="early",
        examples=["early", "exhaust"],
        title="Tool execution strategy",
    )
    """The strategy for handling multiple tool calls when a final result is found"""

    avatar: str | None = Field(
        default=None,
        examples=["https://example.com/avatar.png", "/assets/robot.jpg"],
        title="Avatar image",
    )
    """URL or path to agent's avatar image"""

    system_prompts: Sequence[str | PromptConfig] = Field(
        default_factory=list,
        title="System prompts",
        examples=[["You are an AI assistant."]],
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/system_prompts_configuration/"
        },
    )
    """System prompts for the agent. Can be strings or structured prompt configs.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/system_prompts_configuration/
    """

    # context_sources: list[ContextSource] = Field(default_factory=list)
    # """Initial context sources to load"""

    config_file_path: str | None = Field(
        default=None,
        examples=["/path/to/config.yml", "configs/agent.yaml"],
        title="Configuration file path",
    )
    """Config file path for resolving environment."""

    knowledge: Knowledge | None = Field(
        default=None,
        title="Knowledge sources",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/knowledge_configuration/"
        },
    )
    """Knowledge sources for this agent.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/knowledge_configuration/
    """

    workers: list[WorkerConfig] = Field(
        default_factory=list,
        examples=[
            [
                {
                    "type": "agent",
                    "name": "web_agent",
                    "reset_history_on_run": True,
                }
            ],
            [
                {
                    "type": "team",
                    "name": "analysis_team",
                }
            ],
        ],
        title="Worker agents",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/worker_configuration/"
        },
    )
    """Worker agents which will be available as tools.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/worker_configuration/
    """

    requires_tool_confirmation: ToolConfirmationMode = Field(
        default="per_tool",
        examples=["always", "never", "per_tool"],
        title="Tool confirmation mode",
    )
    """How to handle tool confirmation:
    - "always": Always require confirmation for all tools
    - "never": Never require confirmation (ignore tool settings)
    - "per_tool": Use individual tool settings
    """

    debug: bool = Field(default=False, title="Debug mode")
    """Enable debug output for this agent."""

    usage_limits: UsageLimits | None = Field(default=None, title="Usage limits")
    """Usage limits for this agent."""

    tool_mode: ToolMode | None = Field(
        default=None,
        examples=["codemode"],
        title="Tool execution mode",
    )
    """Tool execution mode:
    - None: Default mode - tools are called directly
    - "codemode": Tools are wrapped in a Python execution environment
    """

    def is_structured(self) -> bool:
        """Check if this config defines a structured agent."""
        return self.output_type is not None

    @model_validator(mode="before")
    @classmethod
    def validate_output_type(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert result type and apply its settings."""
        output_type = data.get("output_type")
        if isinstance(output_type, dict):
            # Extract response-specific settings
            tool_name = output_type.pop("result_tool_name", None)
            tool_description = output_type.pop("result_tool_description", None)
            retries = output_type.pop("output_retries", None)

            # Convert remaining dict to ResponseDefinition
            if "type" not in output_type["response_schema"]:
                output_type["response_schema"]["type"] = "inline"
            data["output_type"]["response_schema"] = InlineSchemaDef(**output_type)

            # Apply extracted settings to agent config
            if tool_name:
                data["result_tool_name"] = tool_name
            if tool_description:
                data["result_tool_description"] = tool_description
            if retries is not None:
                data["output_retries"] = retries

        return data

    @model_validator(mode="before")
    @classmethod
    def handle_model_types(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Convert model inputs to appropriate format."""
        if isinstance((model := data.get("model")), str):
            data["model"] = {"type": "string", "identifier": model}
        return data

    def get_toolsets(self) -> list[ResourceProvider]:
        """Get all resource providers for this agent."""
        providers: list[ResourceProvider] = []

        # Add providers from toolsets
        for toolset_config in self.toolsets:
            try:
                provider = toolset_config.get_provider()
                providers.append(provider)
            except Exception as e:
                msg = "Failed to create provider for toolset"
                logger.exception(msg, toolset_config)
                raise ValueError(msg) from e

        return providers

    def get_tool_provider(self) -> ResourceProvider | None:
        """Get tool provider for this agent."""
        from llmling_agent.tools.base import Tool

        # Create provider for static tools
        if not self.tools:
            return None
        static_tools: list[Tool] = []
        for tool_config in self.tools:
            try:
                match tool_config:
                    case str():
                        if tool_config.startswith("crewai_tools"):
                            obj = import_class(tool_config)()
                            static_tools.append(Tool.from_crewai_tool(obj))
                        elif tool_config.startswith("langchain"):
                            obj = import_class(tool_config)()
                            static_tools.append(Tool.from_langchain_tool(obj))
                        else:
                            tool = Tool.from_callable(tool_config)
                            static_tools.append(tool)
                    case BaseToolConfig():
                        static_tools.append(tool_config.get_tool())
            except Exception:
                logger.exception("Failed to load tool", config=tool_config)
                continue

        return StaticResourceProvider(name="builtin", tools=static_tools)

    def get_session_config(self) -> MemoryConfig:
        """Get resolved memory configuration."""
        match self.session:
            case str() | UUID():
                return MemoryConfig(session=SessionQuery(name=str(self.session)))
            case SessionQuery():
                return MemoryConfig(session=self.session)
            case MemoryConfig():
                return self.session
            case None:
                return MemoryConfig()
            case _:
                msg = f"Invalid session configuration: {self.session}"
                raise ValueError(msg)

    def get_system_prompts(self) -> list[BasePrompt]:
        """Get all system prompts as BasePrompts."""
        from llmling_agent_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        prompts: list[BasePrompt] = []
        for prompt in self.system_prompts:
            match prompt:
                case str():
                    # Convert string to StaticPrompt
                    static_prompt = StaticPrompt(
                        name="system",
                        description="System prompt",
                        messages=[PromptMessage(role="system", content=prompt)],
                    )
                    prompts.append(static_prompt)
                case StaticPromptConfig(content=content):
                    # Convert StaticPromptConfig to StaticPrompt
                    static_prompt = StaticPrompt(
                        name="system",
                        description="System prompt",
                        messages=[PromptMessage(role="system", content=content)],
                    )
                    prompts.append(static_prompt)
                case FilePromptConfig(path=path):
                    # Load and convert file-based prompt

                    template_path = Path(path)
                    if not template_path.is_absolute() and self.config_file_path:
                        base_path = Path(self.config_file_path).parent
                        template_path = base_path / path

                    template_content = template_path.read_text("utf-8")
                    # Create a template-based prompt
                    # (for now as StaticPrompt with placeholder)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"File prompt: {path}",
                        messages=[PromptMessage(role="system", content=template_content)],
                    )
                    prompts.append(static_prompt)
                case LibraryPromptConfig(reference=reference):
                    # Create placeholder for library prompts (resolved by manifest)
                    msg = PromptMessage(role="system", content=f"[LIBRARY:{reference}]")
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"Library: {reference}",
                        messages=[msg],
                    )
                    prompts.append(static_prompt)
                case FunctionPromptConfig(arguments=arguments, function=function):
                    # Import and call the function to get prompt content
                    content = function(**arguments)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"Function prompt: {function}",
                        messages=[PromptMessage(role="system", content=content)],
                    )
                    prompts.append(static_prompt)
                case BasePrompt():
                    prompts.append(prompt)
        return prompts

    def render_system_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Render system prompts with context."""
        from llmling_agent_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        if not context:
            # Default context
            context = {"name": self.name, "id": 1, "model": self.model}

        rendered_prompts: list[str] = []
        for prompt in self.system_prompts:
            match prompt:
                case (str() as content) | StaticPromptConfig(content=content):
                    rendered_prompts.append(render_prompt(content, {"agent": context}))
                case FilePromptConfig(path=path, variables=variables):
                    # Load and render Jinja template from file

                    template_path = Path(path)
                    if not template_path.is_absolute() and self.config_file_path:
                        base_path = Path(self.config_file_path).parent
                        template_path = base_path / path

                    template_content = template_path.read_text("utf-8")
                    template_ctx = {"agent": context, **variables}
                    rendered_prompts.append(render_prompt(template_content, template_ctx))
                case LibraryPromptConfig(reference=reference):
                    # This will be handled by the manifest's get_agent method
                    # For now, just add a placeholder
                    rendered_prompts.append(f"[LIBRARY:{reference}]")
                case FunctionPromptConfig(function=function, arguments=arguments):
                    # Import and call the function to get prompt content
                    content = function(**arguments)
                    rendered_prompts.append(render_prompt(content, {"agent": context}))

        return rendered_prompts


if __name__ == "__main__":
    model = "openai:gpt-5-nano"
    agent_cfg = AgentConfig(name="test_agent", model=model, tools=["crewai_tools.BraveSearchTool"])
    print(agent_cfg)
