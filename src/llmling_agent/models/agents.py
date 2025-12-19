"""Models for agent configuration."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, assert_never
from uuid import UUID

from exxec.configs import ExecutionEnvironmentConfig  # noqa: TC002
from llmling_models.configs import AnyModelConfig  # noqa: TC002
from pydantic import ConfigDict, Field, model_validator
from pydantic_ai import UsageLimits  # noqa: TC002
from schemez import InlineSchemaDef
from tokonomics.model_names import ModelName  # noqa: TC002
from toprompt import render_prompt

from llmling_agent import log
from llmling_agent.common_types import EndStrategy  # noqa: TC001
from llmling_agent.prompts.prompts import PromptMessage, StaticPrompt
from llmling_agent.resource_providers import StaticResourceProvider
from llmling_agent_config.hooks import HooksConfig  # noqa: TC001
from llmling_agent_config.knowledge import Knowledge  # noqa: TC001
from llmling_agent_config.nodes import BaseAgentConfig
from llmling_agent_config.output_types import StructuredResponseConfig  # noqa: TC001
from llmling_agent_config.session import MemoryConfig, SessionQuery
from llmling_agent_config.system_prompts import PromptConfig  # noqa: TC001
from llmling_agent_config.tools import BaseToolConfig, ToolConfig  # noqa: TC001
from llmling_agent_config.toolsets import ToolsetConfig  # noqa: TC001
from llmling_agent_config.workers import WorkerConfig  # noqa: TC001


if TYPE_CHECKING:
    from llmling_agent.prompts.prompts import BasePrompt
    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent.tools.base import Tool


ToolMode = Literal["codemode"]
AutoCache = Literal["off", "5m", "1h"]

logger = log.get_logger(__name__)


class NativeAgentConfig(BaseAgentConfig):
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

    model: AnyModelConfig | ModelName | str | None = Field(
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

    output_retries: int | None = Field(default=None, examples=[1, 3], title="Output retries")
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
            [{"type": "agent", "name": "web_agent", "reset_history_on_run": True}],
            [{"type": "team", "name": "analysis_team"}],
        ],
        title="Worker agents",
        json_schema_extra={
            "documentation_url": "https://phil65.github.io/llmling-agent/YAML%20Configuration/worker_configuration/"
        },
    )
    """Worker agents which will be available as tools.

    Docs: https://phil65.github.io/llmling-agent/YAML%20Configuration/worker_configuration/
    """

    debug: bool = Field(default=False, title="Debug mode")
    """Enable debug output for this agent."""

    environment: ExecutionEnvironmentConfig | None = Field(
        default=None, title="Execution environment"
    )
    """Execution environment configuration for this agent."""

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

    auto_cache: AutoCache = Field(
        default="off",
        examples=["off", "5m", "1h"],
        title="Automatic caching",
    )
    """Automatic prompt caching configuration:
    - "off": No automatic caching
    - "5m": Add cache point with 5 minute TTL
    - "1h": Add cache point with 1 hour TTL
    """

    hooks: HooksConfig | None = Field(
        default=None,
        title="Lifecycle hooks",
    )
    """Hooks for intercepting and customizing agent behavior at key lifecycle points.

    Allows adding context, blocking operations, modifying inputs, or triggering
    side effects during session start/end, run execution, tool usage, and delegation.
    """

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
            case _ as unreachable:
                assert_never(unreachable)

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
                case (str() as content) | StaticPromptConfig(content=content):
                    # Convert string to StaticPrompt
                    msgs = [PromptMessage(role="system", content=content)]
                    static = StaticPrompt(name="system", description="System prompt", messages=msgs)
                    prompts.append(static)
                case FilePromptConfig(path=path):
                    template_path = Path(path)
                    if not template_path.is_absolute() and self.config_file_path:
                        base_path = Path(self.config_file_path).parent
                        template_path = base_path / path
                    template_content = template_path.read_text("utf-8")
                    # Create a template-based prompt (for now as StaticPrompt with placeholder)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"File prompt: {path}",
                        messages=[PromptMessage(role="system", content=template_content)],
                    )
                    prompts.append(static_prompt)
                case LibraryPromptConfig(reference=ref):
                    # Create placeholder for library prompts (resolved by manifest)
                    msg = PromptMessage(role="system", content=f"[LIBRARY:{ref}]")
                    static = StaticPrompt(name="system", description=f"Ref: {ref}", messages=[msg])
                    prompts.append(static)
                case FunctionPromptConfig(arguments=arguments, function=function):
                    # Import and call the function to get prompt content
                    content = function(**arguments)
                    static_prompt = StaticPrompt(
                        name="system",
                        description=f"Function prompt: {function}",
                        messages=[PromptMessage(role="system", content=content)],
                    )
                    prompts.append(static_prompt)
                case _ as unreachable:
                    assert_never(unreachable)
        return prompts

    def render_system_prompts(self, context: dict[str, Any] | None = None) -> list[str]:
        """Render system prompts with context."""
        from llmling_agent_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        context = context or {"name": self.name, "id": 1, "model": self.model}
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
    agent_cfg = NativeAgentConfig(name="test_agent", model=model)
    print(agent_cfg)
