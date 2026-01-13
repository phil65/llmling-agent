"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
import time
from typing import TYPE_CHECKING, Any, Self, TypedDict, TypeVar, overload
from uuid import uuid4

from anyenv import method_spawner
import logfire
from pydantic import ValidationError
from pydantic._internal import _typing_extra
from pydantic_ai import (
    Agent as PydanticAgent,
    BaseToolCallPart,
    CallToolsNode,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelRequestNode,
    PartStartEvent,
    RunContext,
    ToolReturnPart,
)
from pydantic_ai.models import Model

from agentpool.agents.base_agent import BaseAgent
from agentpool.agents.events import (
    RunStartedEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
)
from agentpool.agents.events.processors import FileTracker
from agentpool.agents.modes import ModeInfo
from agentpool.log import get_logger
from agentpool.messaging import ChatMessage, MessageHistory
from agentpool.prompts.convert import convert_prompts
from agentpool.storage import StorageManager
from agentpool.tools import Tool, ToolManager
from agentpool.tools.exceptions import ToolError
from agentpool.utils.inspection import get_argument_key
from agentpool.utils.pydantic_ai_helpers import safe_args_as_dict
from agentpool.utils.result_utils import to_type
from agentpool.utils.streams import merge_queue_into_iterator


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Coroutine, Sequence
    from types import TracebackType

    from exxec import ExecutionEnvironment
    from llmling_models_config import AnyModelConfig
    from pydantic_ai import UsageLimits, UserContent
    from pydantic_ai.builtin_tools import AbstractBuiltinTool
    from pydantic_ai.output import OutputSpec
    from pydantic_ai.settings import ModelSettings
    from slashed import BaseCommand
    from tokonomics.model_discovery import ProviderType
    from tokonomics.model_discovery.model_info import ModelInfo
    from toprompt import AnyPromptType
    from upathtools import JoinablePathLike

    from agentpool.agents import AgentContext
    from agentpool.agents.events import RichAgentStreamEvent
    from agentpool.agents.modes import ModeCategory
    from agentpool.common_types import (
        BuiltinEventHandlerType,
        EndStrategy,
        IndividualEventHandler,
        ModelType,
        ProcessorCallback,
        SessionIdType,
        ToolType,
    )
    from agentpool.delegation import AgentPool
    from agentpool.hooks import AgentHooks
    from agentpool.messaging import MessageNode
    from agentpool.models.agents import NativeAgentConfig, ToolMode
    from agentpool.models.manifest import AgentsManifest
    from agentpool.prompts.prompts import PromptType
    from agentpool.resource_providers import ResourceProvider
    from agentpool.tools.base import FunctionTool
    from agentpool.ui.base import InputProvider
    from agentpool_config.knowledge import Knowledge
    from agentpool_config.mcp_server import MCPServerConfig
    from agentpool_config.nodes import ToolConfirmationMode
    from agentpool_config.session import MemoryConfig, SessionQuery
    from agentpool_config.task import Job


logger = get_logger(__name__)
# OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
NoneType = type(None)

TResult = TypeVar("TResult")


def _extract_text_from_messages(
    messages: list[Any], include_interruption_note: bool = False
) -> str:
    """Extract text content from pydantic-ai messages.

    Args:
        messages: List of ModelRequest/ModelResponse messages
        include_interruption_note: Whether to append interruption notice

    Returns:
        Concatenated text content from all ModelResponse TextParts
    """
    from pydantic_ai.messages import ModelResponse, TextPart as PydanticTextPart

    content = "".join(
        part.content
        for msg in messages
        if isinstance(msg, ModelResponse)
        for part in msg.parts
        if isinstance(part, PydanticTextPart)
    )
    if include_interruption_note:
        if content:
            content += "\n\n"
        content += "[Request interrupted by user]"
    return content


class AgentKwargs(TypedDict, total=False):
    """Keyword arguments for configuring an Agent instance."""

    description: str | None
    model: ModelType
    system_prompt: str | Sequence[str]
    tools: Sequence[ToolType] | None
    toolsets: Sequence[ResourceProvider] | None
    mcp_servers: Sequence[str | MCPServerConfig] | None
    skills_paths: Sequence[JoinablePathLike] | None
    retries: int
    output_retries: int | None
    end_strategy: EndStrategy
    # context: AgentContext[Any] | None  # x
    session: SessionIdType | SessionQuery | MemoryConfig | bool | int
    input_provider: InputProvider | None
    event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None
    env: ExecutionEnvironment | None

    hooks: AgentHooks | None
    model_settings: ModelSettings | None
    usage_limits: UsageLimits | None
    providers: Sequence[ProviderType] | None


class Agent[TDeps = None, OutputDataT = str](BaseAgent[TDeps, OutputDataT]):
    """The main agent class.

    Generically typed with: Agent[Type of Dependencies, Type of Result]
    """

    def __init__(  # noqa: PLR0915
        # we dont use AgentKwargs here so that we can work with explicit ones in the ctor
        self,
        name: str = "agentpool",
        *,
        deps_type: type[TDeps] | None = None,
        model: ModelType,
        output_type: OutputSpec[OutputDataT] = str,  # type: ignore[assignment]
        # context: AgentContext[TDeps] | None = None,
        session: SessionIdType | SessionQuery | MemoryConfig | bool | int = None,
        system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
        description: str | None = None,
        display_name: str | None = None,
        tools: Sequence[ToolType] | None = None,
        toolsets: Sequence[ResourceProvider] | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        resources: Sequence[PromptType | str] = (),
        skills_paths: Sequence[JoinablePathLike] | None = None,
        retries: int = 1,
        output_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        input_provider: InputProvider | None = None,
        parallel_init: bool = True,
        model_settings: ModelSettings | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        agent_pool: AgentPool[Any] | None = None,
        tool_mode: ToolMode | None = None,
        knowledge: Knowledge | None = None,
        agent_config: NativeAgentConfig | None = None,
        env: ExecutionEnvironment | None = None,
        hooks: AgentHooks | None = None,
        tool_confirmation_mode: ToolConfirmationMode = "per_tool",
        builtin_tools: Sequence[AbstractBuiltinTool] | None = None,
        usage_limits: UsageLimits | None = None,
        providers: Sequence[ProviderType] | None = None,
        commands: Sequence[BaseCommand] | None = None,
    ) -> None:
        """Initialize agent.

        Args:
            name: Identifier for the agent (used for logging and lookups)
            deps_type: Type of dependencies to use
            model: The default model to use (defaults to GPT-5)
            output_type: The default output type to use (defaults to str)
            context: Agent context with configuration
            session: Memory configuration.
                - None: Default memory config
                - False: Disable message history (max_messages=0)
                - int: Max tokens for memory
                - str/UUID: Session identifier
                - MemoryConfig: Full memory configuration
                - MemoryProvider: Custom memory provider
                - SessionQuery: Session query

            system_prompt: System prompts for the agent
            description: Description of the Agent ("what it can do")
            display_name: Human-readable display name (falls back to name)
            tools: List of tools to register with the agent
            toolsets: List of toolset resource providers for the agent
            mcp_servers: MCP servers to connect to
            resources: Additional resources to load
            skills_paths: Local directories to search for agent-specific skills
            retries: Default number of retries for failed operations
            output_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            input_provider: Provider for human input (tool confirmation / HumanProviders)
            parallel_init: Whether to initialize resources in parallel
            model_settings: Settings for the AI model
            event_handlers: Sequence of event handlers to register with the agent
            agent_pool: AgentPool instance for managing agent resources
            tool_mode: Tool execution mode (None or "codemode")
            knowledge: Knowledge sources for this agent
            agent_config: Agent configuration
            env: Execution environment for code/command execution and filesystem access
            hooks: AgentHooks instance for intercepting agent behavior at run and tool events
            tool_confirmation_mode: Tool confirmation mode
            builtin_tools: PydanticAI builtin tools (WebSearchTool, CodeExecutionTool, etc.)
            usage_limits: Per-request usage limits (applied to each run() call independently,
                not cumulative across the session)
            providers: Model providers for model discovery (e.g., ["openai", "anthropic"]).
                Defaults to ["models.dev"] if not specified.
            commands: Slash commands
        """
        from llmling_models_config import StringModelConfig

        from agentpool.agents.interactions import Interactions
        from agentpool.agents.sys_prompts import SystemPrompts
        from agentpool.models.agents import NativeAgentConfig
        from agentpool.prompts.conversion_manager import ConversionManager
        from agentpool_commands.pool import CompactCommand
        from agentpool_config.session import MemoryConfig

        self.deps_type = deps_type
        self.model_settings = model_settings
        memory_cfg = (
            session if isinstance(session, MemoryConfig) else MemoryConfig.from_value(session)
        )
        # Collect MCP servers from config
        all_mcp_servers = list(mcp_servers) if mcp_servers else []
        if agent_config and agent_config.mcp_servers:
            all_mcp_servers.extend(agent_config.get_mcp_servers())
        # Add CompactCommand - only makes sense for Native Agent (has own history)
        # Other agents (ClaudeCode, ACP, AGUI) don't control their history directly
        all_commands = list(commands) if commands else []
        all_commands.append(CompactCommand())
        # Call base class with shared parameters
        super().__init__(
            name=name,
            description=description,
            display_name=display_name,
            enable_logging=memory_cfg.enable,
            mcp_servers=all_mcp_servers,
            agent_pool=agent_pool,
            event_configs=agent_config.triggers if agent_config else [],
            env=env,
            input_provider=input_provider,
            output_type=to_type(output_type),  # type: ignore[arg-type]
            tool_confirmation_mode=tool_confirmation_mode,
            event_handlers=event_handlers,
            commands=all_commands,
        )

        # Store config for context creation
        # Convert model to proper config type for NativeAgentConfig

        config_model: AnyModelConfig
        if isinstance(model, Model):
            config_model = StringModelConfig(
                identifier=model.model_name,
                **({"model_settings": model._settings} if model._settings else {}),
            )
        elif isinstance(model, str):
            config_model = StringModelConfig(
                identifier=model,
                **({"model_settings": model_settings} if model_settings else {}),
            )
        else:
            config_model = model
        self._agent_config = agent_config or NativeAgentConfig(name=name, model=config_model)
        # Store builtin tools for pydantic-ai
        self._builtin_tools = list(builtin_tools) if builtin_tools else []
        # Override tools with Agent-specific ToolManager (with tools and tool_mode)
        all_tools = list(tools or [])
        self.tools = ToolManager(all_tools, tool_mode=tool_mode)
        for toolset_provider in toolsets or []:
            self.tools.add_provider(toolset_provider)
        aggregating_provider = self.mcp.get_aggregating_provider()
        self.tools.add_provider(aggregating_provider)
        # Override conversation with Agent-specific MessageHistory (with storage, etc.)
        resources = list(resources)
        if knowledge:
            resources.extend(knowledge.get_resources())
        storage = agent_pool.storage if agent_pool else StorageManager(self._manifest.storage)
        self.conversation = MessageHistory(
            storage=storage,
            converter=ConversionManager(config=self._manifest.conversion),
            session_config=memory_cfg,
            resources=resources,
        )
        if isinstance(model, str):
            self._model, settings = self._resolve_model_string(model)
            if settings:
                self.model_settings = settings
        else:
            self._model = model
        self._retries = retries
        self._end_strategy: EndStrategy = end_strategy
        self._output_retries = output_retries
        self.parallel_init = parallel_init
        self.talk = Interactions(self)
        # Set up system prompts
        all_prompts: list[AnyPromptType] = []
        if isinstance(system_prompt, (list, tuple)):
            all_prompts.extend(system_prompt)
        elif system_prompt:
            all_prompts.append(system_prompt)
        self.sys_prompts = SystemPrompts(all_prompts, prompt_manager=self._manifest.prompt_manager)
        # Store hooks
        self.hooks = hooks
        # Store default usage limits
        self._default_usage_limits = usage_limits
        # Store providers for model discovery
        self._providers = list(providers) if providers else None

    def __repr__(self) -> str:
        desc = f", {self.description!r}" if self.description else ""
        return f"Agent({self.name!r}, model={self._model!r}{desc})"

    async def __prompt__(self) -> str:
        typ = self.__class__.__name__
        model = self.model_name or "default"
        parts = [f"Agent: {self.name}", f"Type: {typ}", f"Model: {model}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([await self.tools.__prompt__(), self.conversation.__prompt__()])
        return "\n".join(parts)

    @classmethod
    def from_config(  # noqa: PLR0915
        cls,
        config: NativeAgentConfig,
        *,
        name: str | None = None,
        manifest: AgentsManifest | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
        input_provider: InputProvider | None = None,
        agent_pool: AgentPool[Any] | None = None,
        deps_type: type[TDeps] | None = None,
    ) -> Self:
        """Create a native Agent from a config object.

        This is the preferred way to instantiate an Agent from configuration.
        Handles system prompt resolution, model resolution, toolsets setup, etc.

        Args:
            config: Native agent configuration
            name: Optional name override (used for manifest lookups, defaults to config.name)
            manifest: Optional manifest for resolving prompts, models, output types.
                     If not provided, uses agent_pool.manifest or creates empty one.
            event_handlers: Optional event handlers (merged with config handlers)
            input_provider: Optional input provider for user interactions
            agent_pool: Optional agent pool for coordination
            deps_type: Optional dependency type

        Returns:
            Configured Agent instance
        """
        from pathlib import Path

        from agentpool.models.manifest import AgentsManifest
        from agentpool.utils.result_utils import to_type
        from agentpool_config.system_prompts import (
            FilePromptConfig,
            FunctionPromptConfig,
            LibraryPromptConfig,
            StaticPromptConfig,
        )

        # Get manifest from pool or create empty one
        if manifest is None:
            manifest = agent_pool.manifest if agent_pool else AgentsManifest()

        # Use provided name, fall back to config.name, then default
        name = name or config.name or "agent"

        # Normalize system_prompt to a list for iteration
        sys_prompts: list[str] = []
        prompt_source = config.system_prompt
        if prompt_source is not None:
            prompts_to_process = (
                [prompt_source] if isinstance(prompt_source, str) else prompt_source
            )
            for prompt in prompts_to_process:
                match prompt:
                    case (str() as sys_prompt) | StaticPromptConfig(content=sys_prompt):
                        sys_prompts.append(sys_prompt)
                    case FilePromptConfig(path=path, variables=variables):
                        template_path = Path(path)
                        if not template_path.is_absolute() and config.config_file_path:
                            template_path = Path(config.config_file_path).parent / path
                        template_content = template_path.read_text("utf-8")
                        if variables:
                            from jinja2 import Template

                            template = Template(template_content)
                            content = template.render(**variables)
                        else:
                            content = template_content
                        sys_prompts.append(content)
                    case LibraryPromptConfig(reference=reference):
                        try:
                            content = manifest.prompt_manager.get.sync(reference)
                            sys_prompts.append(content)
                        except Exception as e:
                            msg = f"Failed to load library prompt {reference!r} for agent {name}"
                            logger.exception(msg)
                            raise ValueError(msg) from e
                    case FunctionPromptConfig(function=function, arguments=arguments):
                        content = function(**arguments)
                        sys_prompts.append(content)

        # Prepare toolsets list
        toolsets_list = config.get_toolsets()
        if config_tool_provider := config.get_tool_provider():
            toolsets_list.append(config_tool_provider)
        # Convert workers config to a toolset (backwards compatibility)
        if config.workers:
            from agentpool_toolsets.builtin.workers import WorkersTools

            workers_provider = WorkersTools(workers=list(config.workers), name="workers")
            toolsets_list.append(workers_provider)
        # Resolve output type
        agent_output_type = manifest.get_output_type(name) or str
        resolved_output_type = to_type(agent_output_type, manifest.responses)
        # Merge event handlers
        config_handlers = config.get_event_handlers()
        merged_handlers: list[IndividualEventHandler | BuiltinEventHandlerType] = [
            *config_handlers,
            *(event_handlers or []),
        ]
        # Resolve model
        resolved_model = manifest.resolve_model(config.model)
        model = resolved_model.get_model()
        model_settings = resolved_model.get_model_settings()
        # Extract builtin tools
        builtin_tools = config.get_builtin_tools()
        return cls(
            model=model,
            model_settings=model_settings,
            system_prompt=sys_prompts,
            name=name,
            display_name=config.display_name,
            deps_type=deps_type,
            env=config.environment.get_provider() if config.environment else None,
            description=config.description,
            retries=config.retries,
            session=config.get_session_config(),
            output_retries=config.output_retries,
            end_strategy=config.end_strategy,
            agent_config=config,
            input_provider=input_provider,
            output_type=resolved_output_type,  # type: ignore[arg-type]
            event_handlers=merged_handlers or None,
            agent_pool=agent_pool,
            tool_mode=config.tool_mode,
            knowledge=config.knowledge,
            toolsets=toolsets_list,
            hooks=config.hooks.get_agent_hooks() if config.hooks else None,
            tool_confirmation_mode=config.requires_tool_confirmation,
            builtin_tools=builtin_tools or None,
            usage_limits=config.usage_limits,
            providers=config.model_providers,
        )

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # Collect all coroutines that need to be run
            coros: list[Coroutine[Any, Any, Any]] = []
            coros.append(super().__aenter__())
            coros.extend(self.conversation.get_initialization_tasks())
            if self.parallel_init and coros:
                await asyncio.gather(*coros)
            else:
                for coro in coros:
                    await coro
        except Exception as e:
            msg = "Failed to initialize agent"
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit async context."""
        await super().__aexit__(exc_type, exc_val, exc_tb)

    @overload
    @classmethod
    def from_callback(
        cls,
        callback: Callable[..., Awaitable[TResult]],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Agent[None, TResult]: ...

    @overload
    @classmethod
    def from_callback(
        cls,
        callback: Callable[..., TResult],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Agent[None, TResult]: ...

    @classmethod
    def from_callback(
        cls,
        callback: ProcessorCallback[Any],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Agent[None, Any]:
        """Create an agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - must return str for pipeline compatibility
            name: Optional name for the agent
            kwargs: Additional arguments for agent
        """
        from llmling_models import function_to_model

        name = name or callback.__name__ or "processor"
        model = function_to_model(callback)
        output_type = _typing_extra.get_function_type_hints(callback).get("return")
        if (  # If async, unwrap from Awaitable
            output_type
            and hasattr(output_type, "__origin__")
            and output_type.__origin__ is Awaitable
        ):
            output_type = output_type.__args__[0]
        return Agent(model=model, name=name, output_type=output_type or str, **kwargs)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "agentpool"

    @name.setter
    def name(self, value: str) -> None:
        """Set agent name."""
        self._name = value

    def get_context(self, data: TDeps | None = None) -> AgentContext[TDeps]:  # type: ignore[override]
        """Create a new context for this agent.

        Args:
            data: Optional custom data to attach to the context

        Returns:
            A new AgentContext instance
        """
        from agentpool.agents import AgentContext

        return AgentContext(
            node=self,
            definition=self._manifest,
            config=self._agent_config,
            input_provider=self._input_provider,
            pool=self.agent_pool,
            data=data,
        )

    def _resolve_model_string(self, model: str) -> tuple[Model, ModelSettings | None]:
        """Resolve a model string, checking variants first.

        Args:
            model: Model identifier or variant name

        Returns:
            Tuple of (Model instance, ModelSettings or None)
            Settings are only returned for variants.
        """
        from llmling_models import infer_model

        # Check if it's a variant
        if self.agent_pool and model in self.agent_pool.manifest.model_variants:
            config = self.agent_pool.manifest.model_variants[model]
            return config.get_model(), config.get_model_settings()
        # Regular model string - no settings
        return infer_model(model), None

    def to_structured[NewOutputDataT](
        self,
        output_type: type[NewOutputDataT],
    ) -> Agent[TDeps, NewOutputDataT]:
        """Convert this agent to a structured agent.

        Warning: This method mutates the agent in place and breaks caching.
        Changing output type modifies tool definitions sent to the API.

        Args:
            output_type: Type for structured responses. Can be:
                - A Python type (Pydantic model)
            tool_name: Optional override for result tool name
            tool_description: Optional override for result tool description

        Returns:
            Self (same instance, not a copy)
        """
        self.log.debug("Setting result type", output_type=output_type)
        self._output_type = to_type(output_type)  # type: ignore[assignment]
        return self  # type: ignore

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format (provider:model_name)."""
        # Construct full model ID with provider prefix (e.g., "anthropic:claude-haiku-4-5")
        return f"{self._model.system}:{self._model.model_name}" if self._model else None

    def to_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        parent: Agent[Any, Any] | None = None,
        **_kwargs: Any,
    ) -> FunctionTool[OutputDataT]:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            description: Optional tool description override
            reset_history_on_run: Clear agent's history before each run
            pass_message_history: Pass parent's message history to agent
            parent: Optional parent agent for history/context sharing
        """

        async def wrapped_tool(prompt: str) -> Any:
            if pass_message_history and not parent:
                msg = "Parent agent required for message history sharing"
                raise ToolError(msg)

            if reset_history_on_run:
                await self.conversation.clear()

            history = None
            if pass_message_history and parent:
                history = parent.conversation.get_history()
                old = self.conversation.get_history()
                self.conversation.set_history(history)
            result = await self.run(prompt)
            if history:
                self.conversation.set_history(old)
            return result.data

        # Set the correct return annotation dynamically
        wrapped_tool.__annotations__ = {"prompt": str, "return": self._output_type or Any}

        normalized_name = self.name.replace("_", " ").title()
        docstring = f"Get expert answer from specialized agent: {normalized_name}"
        description = description or self.description
        if description:
            docstring = f"{docstring}\n\n{description}"
        tool_name = name or f"ask_{self.name}"
        wrapped_tool.__doc__ = docstring
        wrapped_tool.__name__ = tool_name

        return Tool.from_callable(
            wrapped_tool,
            name_override=tool_name,
            description_override=docstring,
            source="agent",
        )

    async def get_agentlet[AgentOutputType](
        self,
        model: ModelType | None,
        output_type: type[AgentOutputType] | None,
        input_provider: InputProvider | None = None,
    ) -> PydanticAgent[TDeps, AgentOutputType]:
        """Create pydantic-ai agent from current state."""
        # Monkey patch pydantic-ai to recognize AgentContext

        from agentpool.agents.tool_wrapping import wrap_tool

        tools = await self.tools.get_tools(state="enabled")
        final_type = to_type(output_type) if output_type not in [None, str] else self._output_type
        actual_model = model or self._model
        if isinstance(actual_model, str):
            model_, _settings = self._resolve_model_string(actual_model)
        else:
            model_ = actual_model
        agent = PydanticAgent(
            name=self.name,
            model=model_,
            model_settings=self.model_settings,
            instructions=await self.sys_prompts.format_system_prompt(self),
            retries=self._retries,
            end_strategy=self._end_strategy,
            output_retries=self._output_retries,
            deps_type=self.deps_type or NoneType,
            output_type=final_type,
            builtin_tools=self._builtin_tools,
        )

        base_context = self.get_context()
        context_for_tools = (
            base_context
            if input_provider is None
            else replace(base_context, input_provider=input_provider)
        )

        for tool in tools:
            wrapped = wrap_tool(tool, context_for_tools, hooks=self.hooks)
            if get_argument_key(wrapped, RunContext):
                logger.info("Registering tool: with context", tool_name=tool.name)
                agent.tool(wrapped)
            else:
                logger.info("Registering tool: no context", tool_name=tool.name)
                agent.tool_plain(wrapped)

        return agent  # type: ignore[return-value]

    async def _stream_events(  # noqa: PLR0915
        self,
        prompts: list[UserContent],
        *,
        user_msg: ChatMessage[Any],
        effective_parent_id: str | None,
        store_history: bool = True,
        message_id: str | None = None,
        conversation_id: str | None = None,
        parent_id: str | None = None,
        message_history: MessageHistory | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
        deps: TDeps | None = None,
        event_handlers: Sequence[IndividualEventHandler | BuiltinEventHandlerType] | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[OutputDataT]]:
        from anyenv import MultiEventHandler
        from pydantic_graph import End

        from agentpool.agents.events import resolve_event_handlers

        conversation = message_history if message_history is not None else self.conversation
        # Use provided event handlers or fall back to agent's handlers
        if event_handlers is not None:
            handler: MultiEventHandler[IndividualEventHandler] = MultiEventHandler(
                resolve_event_handlers(event_handlers)
            )
        else:
            handler = self.event_handler
        message_id = message_id or str(uuid4())
        run_id = str(uuid4())
        # Reset cancellation state
        self._cancelled = False
        # Initialize conversation_id on first run and log to storage
        # Conversation ID initialization handled by BaseAgent
        processed_prompts = prompts
        await self.message_received.emit(user_msg)
        start_time = time.perf_counter()
        history_list = conversation.get_history()
        pending_parts = conversation.get_pending_parts()
        # Execute pre-run hooks
        if self.hooks:
            pre_run_result = await self.hooks.run_pre_run_hooks(
                agent_name=self.name,
                prompt=user_msg.content
                if isinstance(user_msg.content, str)
                else str(user_msg.content),
                conversation_id=self.conversation_id,
            )
            if pre_run_result.get("decision") == "deny":
                reason = pre_run_result.get("reason", "Blocked by pre-run hook")
                msg = f"Run blocked: {reason}"
                raise RuntimeError(msg)

        assert self.conversation_id is not None  # Initialized by BaseAgent.run_stream()
        run_started = RunStartedEvent(
            thread_id=self.conversation_id, run_id=run_id, agent_name=self.name
        )
        await handler(None, run_started)
        yield run_started

        agentlet = await self.get_agentlet(None, self._output_type, input_provider)
        content = await convert_prompts(processed_prompts)
        response_msg: ChatMessage[Any] | None = None
        # Prepend pending context parts (content is already pydantic-ai format)
        converted = [*pending_parts, *content]
        history = [m for run in history_list for m in run.to_pydantic_ai()]
        # Track tool call starts to combine with results later
        pending_tcs: dict[str, BaseToolCallPart] = {}
        file_tracker = FileTracker()
        async with agentlet.iter(
            converted,
            deps=deps,  # type: ignore[arg-type]
            message_history=history,
            usage_limits=self._default_usage_limits,
        ) as agent_run:
            try:
                async for node in agent_run:
                    if self._cancelled:
                        self.log.info("Stream cancelled by user")
                        break
                    if isinstance(node, End):
                        break

                    # Stream events from model request node
                    if isinstance(node, ModelRequestNode):
                        async with (
                            node.stream(agent_run.ctx) as agent_stream,
                            merge_queue_into_iterator(
                                agent_stream,  # type: ignore[arg-type]
                                self._event_queue,
                            ) as merged,
                        ):
                            async for event in file_tracker(merged):
                                if self._cancelled:
                                    break
                                await handler(None, event)
                                yield event
                                combined = self._process_tool_event(event, pending_tcs, message_id)
                                if combined:
                                    await handler(None, combined)
                                    yield combined

                    # Stream events from tool call node
                    elif isinstance(node, CallToolsNode):
                        async with (
                            node.stream(agent_run.ctx) as tool_stream,
                            merge_queue_into_iterator(tool_stream, self._event_queue) as merged,
                        ):
                            async for event in file_tracker(merged):
                                if self._cancelled:
                                    break
                                await handler(None, event)
                                yield event
                                combined = self._process_tool_event(event, pending_tcs, message_id)
                                if combined:
                                    await handler(None, combined)
                                    yield combined
            except asyncio.CancelledError:
                self.log.info("Stream cancelled via task cancellation")
                self._cancelled = True

            # Build response message
            response_time = time.perf_counter() - start_time
            if self._cancelled:
                partial_content = _extract_text_from_messages(
                    agent_run.all_messages(), include_interruption_note=True
                )
                response_msg = ChatMessage(
                    content=partial_content,
                    role="assistant",
                    name=self.name,
                    message_id=message_id,
                    conversation_id=self.conversation_id,
                    parent_id=user_msg.message_id,
                    response_time=response_time,
                    finish_reason="stop",
                )
                complete_event = StreamCompleteEvent(message=response_msg)
                await handler(None, complete_event)
                yield complete_event
                return

            if agent_run.result:
                response_msg = await ChatMessage.from_run_result(
                    agent_run.result,
                    agent_name=self.name,
                    message_id=message_id,
                    conversation_id=self.conversation_id,
                    parent_id=user_msg.message_id,
                    response_time=response_time,
                    metadata=file_tracker.get_metadata(),
                )
            else:
                msg = "Stream completed without producing a result"
                raise RuntimeError(msg)

        # Execute post-run hooks
        if self.hooks:
            prompt_str = (
                user_msg.content if isinstance(user_msg.content, str) else str(user_msg.content)
            )
            await self.hooks.run_post_run_hooks(
                agent_name=self.name,
                prompt=prompt_str,
                result=response_msg.content,
                conversation_id=self.conversation_id,
            )

        # Send additional enriched completion event
        complete_event = StreamCompleteEvent(message=response_msg)
        await handler(None, complete_event)
        yield complete_event

    def _process_tool_event(
        self,
        event: RichAgentStreamEvent[Any],
        pending_tool_calls: dict[str, BaseToolCallPart],
        message_id: str,
    ) -> ToolCallCompleteEvent | None:
        """Process tool-related events and return combined event when complete.

        Args:
            event: The streaming event to process
            pending_tool_calls: Dict tracking in-progress tool calls by ID
            message_id: Message ID for the combined event

        Returns:
            ToolCallCompleteEvent if a tool call completed, None otherwise
        """
        match event:
            case PartStartEvent(part=BaseToolCallPart() as tool_part):
                pending_tool_calls[tool_part.tool_call_id] = tool_part
            case FunctionToolCallEvent(part=tool_part):
                pending_tool_calls[tool_part.tool_call_id] = tool_part
            case FunctionToolResultEvent(tool_call_id=call_id) as result_event:
                if call_info := pending_tool_calls.pop(call_id, None):
                    return ToolCallCompleteEvent(
                        tool_name=call_info.tool_name,
                        tool_call_id=call_id,
                        tool_input=safe_args_as_dict(call_info),
                        tool_result=result_event.result.content
                        if isinstance(result_event.result, ToolReturnPart)
                        else result_event.result,
                        agent_name=self.name,
                        message_id=message_id,
                    )
        return None

    @method_spawner
    async def run_job(
        self,
        job: Job[TDeps, str | None],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[OutputDataT]:
        """Execute a pre-defined task.

        Args:
            job: Job configuration to execute
            store_history: Whether the message exchange should be added to the
                           context window
            include_agent_tools: Whether to include agent tools
        Returns:
            Job execution result

        Raises:
            JobError: If task execution fails
            ValueError: If task configuration is invalid
        """
        from agentpool.tasks import JobError

        if job.required_dependency is not None:
            agent_ctx = self.get_context()
            if not isinstance(agent_ctx.data, job.required_dependency):
                msg = (
                    f"Agent dependencies ({type(agent_ctx.data)}) "
                    f"don't match job requirement ({job.required_dependency})"
                )
                raise JobError(msg)

        # Load task knowledge
        if job.knowledge:
            # Add knowledge sources to context
            for source in list(job.knowledge.paths):
                await self.conversation.load_context_source(source)
            for prompt in job.knowledge.prompts:
                await self.conversation.load_context_source(prompt)
        try:
            # Register task tools temporarily
            tools = job.get_tools()
            async with self.tools.temporary_tools(tools, exclusive=not include_agent_tools):
                # Execute job with job-specific tools
                return await self.run(await job.get_prompt(), store_history=store_history)

        except Exception as e:
            self.log.exception("Task execution failed", error=str(e))
            msg = f"Task execution failed: {e}"
            raise JobError(msg) from e

    def register_worker(
        self,
        worker: MessageNode[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
    ) -> Tool:
        """Register another agent as a worker tool."""
        return self.tools.register_worker(
            worker,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            parent=self if pass_message_history else None,
        )

    async def set_model(self, model: Model | str) -> None:
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        """
        if isinstance(model, str):
            self._model, settings = self._resolve_model_string(model)
            if settings:
                self.model_settings = settings
        else:
            self._model = model

    async def set_tool_confirmation_mode(self, mode: ToolConfirmationMode) -> None:
        """Set the tool confirmation mode for this agent.

        Args:
            mode: Tool confirmation mode:
                - "always": Always require confirmation for all tools
                - "never": Never require confirmation
                - "per_tool": Use individual tool settings
        """
        self.tool_confirmation_mode = mode
        self.log.info("Tool confirmation mode changed", mode=mode)

    @asynccontextmanager
    async def temporary_state[T](
        self,
        *,
        system_prompts: list[AnyPromptType] | None = None,
        output_type: type[T] | None = None,
        replace_prompts: bool = False,
        tools: list[ToolType] | None = None,
        replace_tools: bool = False,
        history: list[AnyPromptType] | SessionQuery | None = None,
        replace_history: bool = False,
        pause_routing: bool = False,
        model: ModelType | None = None,
    ) -> AsyncIterator[Self | Agent[T]]:
        """Temporarily modify agent state.

        Args:
            system_prompts: Temporary system prompts to use
            output_type: Temporary output type to use
            replace_prompts: Whether to replace existing prompts
            tools: Temporary tools to make available
            replace_tools: Whether to replace existing tools
            history: Conversation history (prompts or query)
            replace_history: Whether to replace existing history
            pause_routing: Whether to pause message routing
            model: Temporary model override
        """
        old_model = self._model
        old_settings = self.model_settings
        if output_type:
            old_type = self._output_type
            self.to_structured(output_type)
        async with AsyncExitStack() as stack:
            if system_prompts is not None:  # System prompts
                await stack.enter_async_context(
                    self.sys_prompts.temporary_prompt(system_prompts, exclusive=replace_prompts)
                )

            if tools is not None:  # Tools
                await stack.enter_async_context(
                    self.tools.temporary_tools(tools, exclusive=replace_tools)
                )

            if history is not None:  # History
                await stack.enter_async_context(
                    self.conversation.temporary_state(history, replace_history=replace_history)
                )

            if pause_routing:  # Routing
                await stack.enter_async_context(self.connections.paused_routing())

            if model is not None:  # Model
                if isinstance(model, str):
                    self._model, settings = self._resolve_model_string(model)
                    if settings:
                        self.model_settings = settings
                else:
                    self._model = model

            try:
                yield self
            finally:  # Restore model and settings
                if model is not None:
                    if old_model:
                        self._model = old_model
                    self.model_settings = old_settings
                if output_type:
                    self.to_structured(old_type)

    async def validate_against(
        self,
        prompt: str,
        criteria: type[OutputDataT],
        **kwargs: Any,
    ) -> bool:
        """Check if agent's response satisfies stricter criteria."""
        result = await self.run(prompt, **kwargs)
        try:
            criteria.model_validate(result.content.model_dump())  # type: ignore
        except ValidationError:
            return False
        else:
            return True

    async def get_available_models(self) -> list[ModelInfo] | None:
        """Get available models for this agent.

        Uses tokonomics model discovery to fetch models from configured providers.
        Defaults to openai, anthropic, and gemini if no providers specified.

        Returns:
            List of tokonomics ModelInfo, or None if discovery fails
        """
        from datetime import timedelta

        from tokonomics.model_discovery import get_all_models

        try:
            max_age = timedelta(days=200)
            return await get_all_models(
                providers=self._providers or ["models.dev"], max_age=max_age
            )
        except Exception:
            self.log.exception("Failed to discover models")
            return None

    async def get_modes(self) -> list[ModeCategory]:
        """Get available mode categories for this agent.

        Native agents expose permission modes and model selection.

        Returns:
            List of ModeCategory for permissions and models
        """
        from agentpool.agents.modes import ModeCategory, ModeInfo

        categories: list[ModeCategory] = []

        # Permission modes
        mode_id_map = {
            "per_tool": "default",
            "always": "default",
            "never": "acceptEdits",
        }
        current_id = mode_id_map.get(self.tool_confirmation_mode, "default")

        categories.append(
            ModeCategory(
                id="permissions",
                name="Permissions",
                available_modes=[
                    ModeInfo(
                        id="default",
                        name="Default",
                        description="Require confirmation for tools marked as needing it",
                        category_id="permissions",
                    ),
                    ModeInfo(
                        id="acceptEdits",
                        name="Accept Edits",
                        description="Auto-approve all tool calls without confirmation",
                        category_id="permissions",
                    ),
                ],
                current_mode_id=current_id,
                category="mode",
            )
        )

        # Model selection
        models = await self.get_available_models()
        if models:
            current_model = self.model_name or (models[0].id if models else "")
            categories.append(
                ModeCategory(
                    id="model",
                    name="Model",
                    available_modes=[
                        ModeInfo(
                            id=m.id,
                            name=m.name or m.id,
                            description=m.description or "",
                            category_id="model",
                        )
                        for m in models
                    ],
                    current_mode_id=current_model,
                    category="model",
                )
            )

        return categories

    async def set_mode(self, mode: ModeInfo | str, category_id: str | None = None) -> None:
        """Set a mode for this agent.

        Native agents support:
        - "permissions" category: default, acceptEdits
        - "model" category: any available model ID

        Args:
            mode: Mode to activate - ModeInfo object or mode ID string
            category_id: Category ID ("permissions" or "model")

        Raises:
            ValueError: If mode_id or category_id is invalid
        """
        # Extract mode_id and category from ModeInfo if provided
        if isinstance(mode, ModeInfo):
            mode_id = mode.id
            category_id = category_id or mode.category_id
        else:
            mode_id = mode

        # Default to permissions if no category specified
        if category_id is None:
            category_id = "permissions"

        if category_id == "permissions":
            # Map mode_id to confirmation mode
            mode_map: dict[str, ToolConfirmationMode] = {
                "default": "per_tool",
                "acceptEdits": "never",
            }
            if mode_id not in mode_map:
                msg = f"Unknown permission mode: {mode_id}. Available: {list(mode_map.keys())}"
                raise ValueError(msg)
            await self.set_tool_confirmation_mode(mode_map[mode_id])

        elif category_id == "model":
            # Validate model exists
            models = await self.get_available_models()
            if models:
                valid_ids = {m.id for m in models}
                if mode_id not in valid_ids:
                    msg = f"Unknown model: {mode_id}. Available: {valid_ids}"
                    raise ValueError(msg)
            # Set the model using set_model method
            await self.set_model(mode_id)
            self.log.info("Model changed", model=mode_id)

        else:
            msg = f"Unknown category: {category_id}. Available: permissions, model"
            raise ValueError(msg)


if __name__ == "__main__":
    import logging

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])
    sys_prompt = "Open browser with google,"
    _model = "openai:gpt-5-nano"

    async def handle_events(ctx: RunContext, event: Any) -> None:
        print(f"[EVENT] {type(event).__name__}: {event}")

    agent = Agent(model=_model, tools=["webbrowser.open"], event_handlers=[handle_events])
    result = agent.run.sync(sys_prompt)
