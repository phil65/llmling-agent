"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field
from os import PathLike
import time
from typing import TYPE_CHECKING, Any, Literal, Self, TypedDict, TypeVar, cast, overload
from uuid import uuid4

from anyenv import MultiEventHandler, method_spawner
from llmling import Config, RuntimeConfig, ToolError
import logfire
from psygnal import Signal
from upath import UPath

from llmling_agent.log import get_logger
from llmling_agent.messaging.messagenode import MessageNode
from llmling_agent.messaging.messages import ChatMessage, TokenCost
from llmling_agent.prompts.builtin_provider import RuntimePromptProvider
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.resource_providers.runtime import RuntimeResourceProvider
from llmling_agent.talk.stats import MessageStats
from llmling_agent.tools.base import Tool
from llmling_agent.tools.manager import ToolManager
from llmling_agent.utils.inspection import call_with_context, has_return_type
from llmling_agent.utils.now import get_now
from llmling_agent.utils.result_utils import to_type
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_config.session import MemoryConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Coroutine, Sequence
    from datetime import datetime
    import os
    from types import TracebackType

    from llmling.config.models import Resource
    from llmling.prompts import PromptType
    import PIL.Image
    from pydantic_ai.messages import AgentStreamEvent
    from toprompt import AnyPromptType
    from upath.types import JoinablePathLike

    from llmling_agent.agent import AgentContext, AnyAgent
    from llmling_agent.agent.conversation import ConversationManager
    from llmling_agent.agent.interactions import Interactions
    from llmling_agent.agent.structured import StructuredAgent
    from llmling_agent.common_types import (
        AgentName,
        EndStrategy,
        ModelType,
        SessionIdType,
        ToolType,
    )
    from llmling_agent.delegation.team import Team
    from llmling_agent.delegation.teamrun import TeamRun
    from llmling_agent.resource_providers.base import ResourceProvider
    from llmling_agent_config.mcp_server import MCPServerConfig
    from llmling_agent_config.providers import ProcessorCallback
    from llmling_agent_config.result_types import StructuredResponseConfig
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.task import Job
    from llmling_agent_input.base import InputProvider
    from llmling_agent_providers.base import UsageLimits

from pydantic_ai.agent import EventStreamHandler

from llmling_agent_providers.base import AgentProvider


AgentType = Literal["pydantic_ai", "human"] | AgentProvider | Callable[..., Any]

logger = get_logger(__name__)

TResult = TypeVar("TResult", default=str)


@dataclass(kw_only=True)
class StreamCompleteEvent:
    """Event indicating streaming is complete with final message."""

    message: ChatMessage[Any]
    """The final chat message with all metadata."""
    event_kind: Literal["stream_complete"] = "stream_complete"
    """Event type identifier."""


class AgentKwargs(TypedDict, total=False):
    """Keyword arguments for configuring an Agent instance."""

    # Core Identity
    provider: AgentType
    description: str | None

    # Model Configuration
    model: ModelType
    system_prompt: str | Sequence[str]
    # model_settings: dict[str, Any]

    # Runtime Environment
    runtime: RuntimeConfig | Config | JoinablePathLike | None
    tools: Sequence[ToolType | Tool] | None
    toolsets: Sequence[ResourceProvider] | None
    mcp_servers: Sequence[str | MCPServerConfig] | None

    # Execution Settings
    retries: int
    output_retries: int | None
    end_strategy: EndStrategy
    defer_model_check: bool

    # Context & State
    context: AgentContext[Any] | None  # x
    session: SessionIdType | SessionQuery | MemoryConfig | bool | int

    # Behavior Control
    input_provider: InputProvider | None
    debug: bool
    event_handlers: Sequence[EventStreamHandler] | None


class Agent[TDeps = None](MessageNode[TDeps, str]):
    """Agent for AI-powered interaction with LLMling resources and tools.

    Generically typed with: LLMLingAgent[Type of Dependencies, Type of Result]

    This agent integrates LLMling's resource system with PydanticAI's agent capabilities.
    It provides:
    - Access to resources through RuntimeConfig
    - Tool registration for resource operations
    - System prompt customization
    - Signals
    - Message history management
    - Database logging
    """

    @dataclass(frozen=True)
    class AgentReset:
        """Emitted when agent is reset."""

        agent_name: AgentName
        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=get_now)

    # this fixes weird mypy issue
    conversation: ConversationManager
    talk: Interactions
    model_changed = Signal(object)  # Model | None
    chunk_streamed = Signal(str, str)  # (chunk, message_id)
    run_failed = Signal(str, Exception)
    agent_reset = Signal(AgentReset)

    def __init__(  # noqa: PLR0915
        # we dont use AgentKwargs here so that we can work with explicit ones in the ctor
        self,
        name: str = "llmling-agent",
        provider: AgentType = "pydantic_ai",
        *,
        model: ModelType = None,
        runtime: RuntimeConfig | Config | JoinablePathLike | None = None,
        context: AgentContext[TDeps] | None = None,
        session: SessionIdType | SessionQuery | MemoryConfig | bool | int = None,
        system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
        description: str | None = None,
        tools: Sequence[ToolType | Tool] | None = None,
        toolsets: Sequence[ResourceProvider] | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        resources: Sequence[Resource | PromptType | str] = (),
        retries: int = 1,
        output_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        defer_model_check: bool = False,
        input_provider: InputProvider | None = None,
        parallel_init: bool = True,
        debug: bool = False,
        event_handlers: Sequence[EventStreamHandler] | None = None,
    ):
        """Initialize agent with runtime configuration.

        Args:
            name: Name of the agent for logging and identification
            provider: Agent type to use (ai: PydanticAIProvider, human: HumanProvider)
            model: The default model to use (defaults to GPT-5)
            runtime: Runtime configuration providing access to resources/tools
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
            tools: List of tools to register with the agent
            toolsets: List of toolset resource providers for the agent
            mcp_servers: MCP servers to connect to
            resources: Additional resources to load
            retries: Default number of retries for failed operations
            output_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            defer_model_check: Whether to defer model evaluation until first run
            input_provider: Provider for human input (tool confirmation / HumanProviders)
            parallel_init: Whether to initialize resources in parallel
            debug: Whether to enable debug mode
            event_handlers: Sequence of event handlers to register with the agent
        """
        from llmling_agent.agent import AgentContext
        from llmling_agent.agent.conversation import ConversationManager
        from llmling_agent.agent.interactions import Interactions
        from llmling_agent.agent.sys_prompts import SystemPrompts
        from llmling_agent_providers.base import AgentProvider

        self.task_manager = TaskManager()
        self._infinite = False
        # save some stuff for asnyc init
        self._owns_runtime = False
        # prepare context
        ctx = context or AgentContext[TDeps].create_default(
            name,
            input_provider=input_provider,
        )
        self._context = ctx
        memory_cfg = (
            session
            if isinstance(session, MemoryConfig)
            else MemoryConfig.from_value(session)
        )
        super().__init__(
            name=name,
            context=ctx,
            description=description,
            enable_logging=memory_cfg.enable,
            mcp_servers=mcp_servers,
        )
        # Initialize runtime
        match runtime:
            case None:
                ctx.runtime = RuntimeConfig.from_config(Config())
            case Config() | str() | PathLike() | UPath():
                ctx.runtime = RuntimeConfig.from_config(runtime)
            case RuntimeConfig():
                ctx.runtime = runtime
            case _:
                msg = f"Invalid runtime type: {type(runtime)}"
                raise TypeError(msg)

        runtime_provider = RuntimePromptProvider(ctx.runtime)
        ctx.definition.prompt_manager.providers["runtime"] = runtime_provider
        # Initialize tool manager
        self.event_handler = MultiEventHandler[EventStreamHandler](event_handlers)
        all_tools = list(tools or [])
        self.tools = ToolManager(all_tools)
        self.tools.add_provider(self.mcp)
        if builtin_tools := ctx.config.get_tool_provider():
            self.tools.add_provider(builtin_tools)

        # Add toolset providers
        if toolsets:
            for toolset_provider in toolsets:
                self.tools.add_provider(toolset_provider)

        # Initialize conversation manager
        resources = list(resources)
        if ctx.config.knowledge:
            resources.extend(ctx.config.knowledge.get_resources())
        self.conversation = ConversationManager(self, memory_cfg, resources=resources)
        # Initialize provider
        match provider:
            case "pydantic_ai":
                from llmling_agent_providers.pydanticai import PydanticAIProvider

                if model and not isinstance(model, str):
                    from pydantic_ai import models

                    assert isinstance(model, models.Model)
                self._provider: AgentProvider = PydanticAIProvider(
                    model=model,
                    retries=retries,
                    end_strategy=end_strategy,
                    output_retries=output_retries,
                    defer_model_check=defer_model_check,
                    debug=debug,
                    context=ctx,
                )
            case "human":
                from llmling_agent_providers.human import HumanProvider

                self._provider = HumanProvider(name=name, debug=debug, context=ctx)
            case Callable():
                from llmling_agent_providers.callback import CallbackProvider

                self._provider = CallbackProvider(
                    provider, name=name, debug=debug, context=ctx
                )
            case AgentProvider():
                self._provider = provider
                self._provider.context = ctx
            case _:
                msg = f"Invalid agent type: {type}"
                raise ValueError(msg)

        # Initialize skills registry
        from llmling_agent.tools.skills import SkillsRegistry

        self.skills_registry = SkillsRegistry()

        if ctx and ctx.definition:
            from llmling_agent.observability import registry

            registry.configure_observability(ctx.definition.observability)

        # init variables
        self._debug = debug
        self._result_type: type | None = None
        self.parallel_init = parallel_init
        self.name = name
        self._background_task: asyncio.Task[Any] | None = None

        # Forward provider signals
        self._provider.model_changed.connect(self.model_changed)
        self._provider.tool_used.connect(self.tool_used)
        self._provider.model_changed.connect(self.model_changed)

        self.talk = Interactions(self)

        # Set up system prompts
        config_prompts = ctx.config.system_prompts if ctx else []
        all_prompts: list[AnyPromptType] = list(config_prompts)
        if isinstance(system_prompt, list):
            all_prompts.extend(system_prompt)
        else:
            all_prompts.append(system_prompt)
        self.sys_prompts = SystemPrompts(all_prompts, context=ctx)

    def __repr__(self) -> str:
        desc = f", {self.description!r}" if self.description else ""
        tools = f", tools={len(self.tools)}" if self.tools else ""
        return f"Agent({self.name!r}, provider={self._provider.NAME!r}{desc}{tools})"

    def __prompt__(self) -> str:
        typ = self._provider.__class__.__name__
        model = self.model_name or "default"
        parts = [f"Agent: {self.name}", f"Type: {typ}", f"Model: {model}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([self.tools.__prompt__(), self.conversation.__prompt__()])

        return "\n".join(parts)

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # Collect all coroutines that need to be run
            coros: list[Coroutine[Any, Any, Any]] = []

            # Runtime initialization if needed
            runtime_ref = self.context.runtime
            if runtime_ref and not runtime_ref._initialized:
                self._owns_runtime = True
                coros.append(runtime_ref.__aenter__())

            # Events initialization
            coros.append(super().__aenter__())

            # Get conversation init tasks directly
            coros.extend(self.conversation.get_initialization_tasks())

            # Execute coroutines either in parallel or sequentially
            if self.parallel_init and coros:
                await asyncio.gather(*coros)
            else:
                for coro in coros:
                    await coro
            if runtime_ref:
                self.tools.add_provider(RuntimeResourceProvider(runtime_ref))
            for provider in self.context.config.get_toolsets():
                self.tools.add_provider(provider)
        except Exception as e:
            # Clean up in reverse order
            if self._owns_runtime and runtime_ref and self.context.runtime == runtime_ref:
                await runtime_ref.__aexit__(type(e), e, e.__traceback__)
            msg = "Failed to initialize agent"
            raise RuntimeError(msg) from e
        else:
            return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        """Exit async context."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
        try:
            await self.mcp.__aexit__(exc_type, exc_val, exc_tb)
        finally:
            if self._owns_runtime and self.context.runtime:
                self.tools.remove_provider("runtime")
                await self.context.runtime.__aexit__(exc_type, exc_val, exc_tb)
            # for provider in await self.context.config.get_toolsets():
            #     self.tools.remove_provider(provider.name)

    @overload
    def __and__(
        self, other: Agent[TDeps] | StructuredAgent[TDeps, Any]
    ) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: Team[TDeps]) -> Team[TDeps]: ...

    @overload
    def __and__(self, other: ProcessorCallback[Any]) -> Team[TDeps]: ...

    def __and__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> Team[Any]:
        """Create agent group using | operator.

        Example:
            group = analyzer & planner & executor  # Create group of 3
            group = analyzer & existing_group  # Add to existing group
        """
        from llmling_agent.agent import StructuredAgent
        from llmling_agent.delegation.team import Team

        match other:
            case Team():
                return Team([self, *other.agents])
            case Callable():
                if callable(other):
                    if has_return_type(other, str):
                        agent_2 = Agent.from_callback(other)
                    else:
                        agent_2 = StructuredAgent.from_callback(other)
                agent_2.context.pool = self.context.pool
                return Team([self, agent_2])
            case MessageNode():
                return Team([self, other])
            case _:
                msg = f"Invalid agent type: {type(other)}"
                raise ValueError(msg)

    @overload
    def __or__(self, other: MessageNode[TDeps, Any]) -> TeamRun[TDeps, Any]: ...

    @overload
    def __or__[TOtherDeps](
        self,
        other: MessageNode[TOtherDeps, Any],
    ) -> TeamRun[Any, Any]: ...

    @overload
    def __or__(self, other: ProcessorCallback[Any]) -> TeamRun[Any, Any]: ...

    def __or__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> TeamRun:
        # Create new execution with sequential mode (for piping)
        from llmling_agent import StructuredAgent, TeamRun

        if callable(other):
            if has_return_type(other, str):
                other = Agent.from_callback(other)
            else:
                other = StructuredAgent.from_callback(other)
            other.context.pool = self.context.pool

        return TeamRun([self, other])

    @classmethod
    def from_callback(
        cls,
        callback: ProcessorCallback[str],
        *,
        name: str | None = None,
        debug: bool = False,
        **kwargs: Any,
    ) -> Agent[None]:
        """Create an agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - must return str for pipeline compatibility
            name: Optional name for the agent
            debug: Whether to enable debug mode
            kwargs: Additional arguments for agent
        """
        from llmling_agent_providers.callback import CallbackProvider

        name = name or getattr(callback, "__name__", "processor")
        name = name or "processor"
        provider = CallbackProvider(callback, name=name)
        return Agent[None](provider=provider, name=name, debug=debug, **kwargs)

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @name.setter
    def name(self, value: str):
        self._provider.name = value
        self._name = value

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get agent context."""
        return self._context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set agent context and propagate to provider."""
        self._provider.context = value
        self.mcp.context = value
        self._context = value

    def set_result_type(
        self,
        result_type: type[TResult] | str | StructuredResponseConfig | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Set or update the result type for this agent.

        Args:
            result_type: New result type, can be:
                - A Python type for validation
                - Name of a response definition
                - Response definition instance
                - None to reset to unstructured mode
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description
        """
        logger.debug("Setting result type to: %s for %r", result_type, self.name)
        self._result_type = to_type(result_type)

    @property
    def provider(self) -> AgentProvider:
        """Get the underlying provider."""
        return self._provider

    @provider.setter
    def provider(self, value: AgentType, model: ModelType = None):
        """Set the underlying provider."""
        from llmling_agent_providers.base import AgentProvider

        name = self.name
        debug = self._debug
        self._provider.model_changed.disconnect(self.model_changed)
        self._provider.tool_used.disconnect(self.tool_used)
        self._provider.model_changed.disconnect(self.model_changed)
        match value:
            case AgentProvider():
                self._provider = value
            case "pydantic_ai":
                from llmling_agent_providers.pydanticai import PydanticAIProvider

                self._provider = PydanticAIProvider(model=model, name=name, debug=debug)
            case "human":
                from llmling_agent_providers.human import HumanProvider

                self._provider = HumanProvider(name=name, debug=debug)
            case Callable():
                from llmling_agent_providers.callback import CallbackProvider

                self._provider = CallbackProvider(value, name=name, debug=debug)
            case _:
                msg = f"Invalid agent type: {type}"
                raise ValueError(msg)
        self._provider.model_changed.connect(self.model_changed)
        self._provider.tool_used.connect(self.tool_used)
        self._provider.model_changed.connect(self.model_changed)
        self._provider.context = self._context  # pyright: ignore[reportAttributeAccessIssue]

    @overload
    def to_structured(
        self,
        result_type: None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Self: ...

    @overload
    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | StructuredResponseConfig,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult]: ...

    def to_structured[TResult](
        self,
        result_type: type[TResult] | str | StructuredResponseConfig | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> StructuredAgent[TDeps, TResult] | Self:
        """Convert this agent to a structured agent.

        If result_type is None, returns self unchanged (no wrapping).
        Otherwise creates a StructuredAgent wrapper.

        Args:
            result_type: Type for structured responses. Can be:
                - A Python type (Pydantic model)
                - Name of response definition from context
                - Complete response definition
                - None to skip wrapping
            tool_name: Optional override for result tool name
            tool_description: Optional override for result tool description

        Returns:
            Either StructuredAgent wrapper or self unchanged
        from llmling_agent.agent import StructuredAgent
        """
        if result_type is None:
            return self

        from llmling_agent.agent import StructuredAgent

        return StructuredAgent(
            self,
            result_type=result_type,
            tool_name=tool_name,
            tool_description=tool_description,
        )

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self.task_manager._pending_tasks or self._background_task)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return self._provider.model_name

    def to_tool(
        self,
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
        parent: AnyAgent[Any, Any] | None = None,
    ) -> Tool:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            reset_history_on_run: Clear agent's history before each run
            pass_message_history: Pass parent's message history to agent
            share_context: Whether to pass parent's context/deps
            parent: Optional parent agent for history/context sharing
        """
        tool_name = name or f"ask_{self.name}"

        async def wrapped_tool(prompt: str) -> str:
            if pass_message_history and not parent:
                msg = "Parent agent required for message history sharing"
                raise ToolError(msg)

            if reset_history_on_run:
                self.conversation.clear()

            history = None
            if pass_message_history and parent:
                history = parent.conversation.get_history()
                old = self.conversation.get_history()
                self.conversation.set_history(history)
            result = await self.run(prompt, result_type=self._result_type)
            if history:
                self.conversation.set_history(old)
            return result.data

        normalized_name = self.name.replace("_", " ").title()
        docstring = f"Get expert answer from specialized agent: {normalized_name}"
        if self.description:
            docstring = f"{docstring}\n\n{self.description}"

        wrapped_tool.__doc__ = docstring
        wrapped_tool.__name__ = tool_name

        return Tool.from_callable(
            wrapped_tool,
            name_override=tool_name,
            description_override=docstring,
        )

    @logfire.instrument("Calling Agent.run: {prompts}:")
    async def _run(
        self,
        *prompts: AnyPromptType | PIL.Image.Image | os.PathLike[str] | ChatMessage[Any],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        tool_choice: str | list[str] | None = None,
        usage_limits: UsageLimits | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[ChatMessage[Any]] | None = None,
        wait_for_connections: bool | None = None,
    ) -> ChatMessage[TResult]:
        """Run agent with prompt and get response.

        Args:
            prompts: User query or instruction
            result_type: Optional type for structured responses
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                            context window
            tool_choice: Filter tool choice by name
            usage_limits: Optional usage limits for the model
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            conversation_id: Optional conversation id for the returned message.
            messages: Optional list of messages to replace the conversation history
            wait_for_connections: Whether to wait for connected agents to complete

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        """Run agent with prompt and get response."""
        message_id = message_id or str(uuid4())
        tools = await self.tools.get_tools(state="enabled", names=tool_choice)
        self.set_result_type(result_type)
        start_time = time.perf_counter()
        sys_prompt = await self.sys_prompts.format_system_prompt(self)

        message_history = (
            messages if messages is not None else self.conversation.get_history()
        )
        try:
            result = await self._provider.generate_response(
                *await convert_prompts(prompts),
                message_id=message_id,
                message_history=message_history,
                tools=tools,
                result_type=result_type,
                usage_limits=usage_limits,
                model=model,
                system_prompt=sys_prompt,
            )
        except Exception as e:
            logger.exception("Agent run failed")
            self.run_failed.emit("Agent run failed", e)
            raise
        else:
            response_msg = ChatMessage[TResult](
                content=result.content,
                role="assistant",
                name=self.name,
                model=result.model_name,
                message_id=message_id,
                conversation_id=conversation_id,
                tool_calls=result.tool_calls,
                cost_info=result.cost_and_usage,
                response_time=time.perf_counter() - start_time,
                provider_extra=result.provider_extra or {},
            )
            if self._debug:
                import devtools

                devtools.debug(response_msg)
            return response_msg

    async def run_stream(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        tool_choice: str | list[str] | None = None,
        store_history: bool = True,
        usage_limits: UsageLimits | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[ChatMessage[Any]] | None = None,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[AgentStreamEvent | StreamCompleteEvent]:
        """Run agent with prompt and get a streaming response.

        Args:
            prompt: User query or instruction
            result_type: Optional type for structured responses
            model: Optional model override
            tool_choice: Filter tool choice by name
            store_history: Whether the message exchange should be added to the
                           context window
            usage_limits: Optional usage limits for the model
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            conversation_id: Optional conversation id for the returned message.
            messages: Optional list of messages to replace the conversation history
            wait_for_connections: Whether to wait for connected agents to complete
        Returns:
            An async iterator yielding streaming events with final message embedded.

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        message_id = message_id or str(uuid4())
        user_msg, prompts = await self.pre_run(*prompt)
        self.set_result_type(result_type)
        start_time = time.perf_counter()
        sys_prompt = await self.sys_prompts.format_system_prompt(self)
        tools = await self.tools.get_tools(state="enabled", names=tool_choice)
        message_history = (
            messages if messages is not None else self.conversation.get_history()
        )
        try:
            # Collect chunks for final message construction
            chunks = []
            usage = None
            model_name = None

            # Stream events directly from provider
            async for event in self._provider.stream_events(
                *prompts,
                message_id=message_id,
                message_history=message_history,
                result_type=result_type,
                model=model,
                tools=tools,
                usage_limits=usage_limits,
                system_prompt=sys_prompt,
            ):
                from pydantic_ai.messages import PartDeltaEvent, TextPartDelta
                from pydantic_ai.run import AgentRunResultEvent

                # Pass through PydanticAI events and collect chunks
                match event:
                    case PartDeltaEvent(delta=TextPartDelta(content_delta=delta)):
                        chunks.append(delta)
                        self.chunk_streamed.emit(delta, message_id)
                        yield event  # Pass through original event
                    case AgentRunResultEvent(result=result):
                        usage = result.usage()
                        model_name = result.response.model_name
                        # Don't yield AgentRunResultEvent, we'll send our own final event
                    case _:
                        yield event  # Pass through other events

            # Build final chat message
            cost_info = None
            if model_name and usage and model_name != "test":
                cost_info = await TokenCost.from_usage(usage, model_name)

            response_msg = ChatMessage[TResult](
                content=cast(TResult, "".join(chunks)),  # type: ignore
                role="assistant",
                name=self.name,
                model=model_name,
                message_id=message_id,
                conversation_id=user_msg.conversation_id,
                cost_info=cost_info,
                response_time=time.perf_counter() - start_time,
            )

            # Yield final event with embedded message
            yield StreamCompleteEvent(message=response_msg)

            # Post-processing
            self.message_sent.emit(response_msg)
            if store_history:
                self.conversation.add_chat_messages([user_msg, response_msg])
            await self.connections.route_message(
                response_msg,
                wait=wait_for_connections,
            )

        except Exception as e:
            logger.exception("Agent stream failed")
            self.run_failed.emit("Agent stream failed", e)
            raise

    async def run_iter(
        self,
        *prompt_groups: Sequence[AnyPromptType | PIL.Image.Image | os.PathLike[str]],
        result_type: type[TResult] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[ChatMessage[TResult]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially
            result_type: Optional type for structured responses
            model: Optional model override
            store_history: Whether to store in conversation history
            wait_for_connections: Whether to wait for connected agents

        Yields:
            Response messages in sequence

        Example:
            questions = [
                ["What is your name?"],
                ["How old are you?", image1],
                ["Describe this image", image2],
            ]
            async for response in agent.run_iter(*questions):
                print(response.content)
        """
        for prompts in prompt_groups:
            response = await self.run(
                *prompts,
                result_type=result_type,
                model=model,
                store_history=store_history,
                wait_for_connections=wait_for_connections,
            )
            yield response  # pyright: ignore

    @method_spawner
    async def run_job(
        self,
        job: Job[TDeps, str | None],
        *,
        store_history: bool = True,
        include_agent_tools: bool = True,
    ) -> ChatMessage[str]:
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
        from llmling_agent.tasks import JobError

        if job.required_dependency is not None:  # noqa: SIM102
            if not isinstance(self.context.data, job.required_dependency):
                msg = (
                    f"Agent dependencies ({type(self.context.data)}) "
                    f"don't match job requirement ({job.required_dependency})"
                )
                raise JobError(msg)

        # Load task knowledge
        if job.knowledge:
            # Add knowledge sources to context
            resources: list[Resource | str] = list(job.knowledge.paths) + list(
                job.knowledge.resources
            )
            for source in resources:
                await self.conversation.load_context_source(source)
            for prompt in job.knowledge.prompts:
                await self.conversation.load_context_source(prompt)
        try:
            # Register task tools temporarily
            tools = job.get_tools()
            with self.tools.temporary_tools(tools, exclusive=not include_agent_tools):
                # Execute job with job-specific tools
                return await self.run(await job.get_prompt(), store_history=store_history)

        except Exception as e:
            msg = f"Task execution failed: {e}"
            logger.exception(msg)
            raise JobError(msg) from e

    async def run_in_background(
        self,
        *prompt: AnyPromptType | PIL.Image.Image | os.PathLike[str],
        max_count: int | None = None,
        interval: float = 1.0,
        block: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[TResult] | None:
        """Run agent continuously in background with prompt or dynamic prompt function.

        Args:
            prompt: Static prompt or function that generates prompts
            max_count: Maximum number of runs (None = infinite)
            interval: Seconds between runs
            block: Whether to block until completion
            **kwargs: Arguments passed to run()
        """
        self._infinite = max_count is None

        async def _continuous():
            count = 0
            msg = "%s: Starting continuous run (max_count=%s, interval=%s) for %r"
            logger.debug(msg, self.name, max_count, interval, self.name)
            latest = None
            while max_count is None or count < max_count:
                try:
                    current_prompts = [
                        call_with_context(p, self.context, **kwargs) if callable(p) else p
                        for p in prompt
                    ]
                    msg = "%s: Generated prompt #%d: %s"
                    logger.debug(msg, self.name, count, current_prompts)

                    latest = await self.run(current_prompts, **kwargs)
                    msg = "%s: Run continous result #%d"
                    logger.debug(msg, self.name, count)

                    count += 1
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    logger.debug("%s: Continuous run cancelled", self.name)
                    break
                except Exception:
                    logger.exception("%s: Background run failed", self.name)
                    await asyncio.sleep(interval)
            msg = "%s: Continuous run completed after %d iterations"
            logger.debug(msg, self.name, count)
            return latest

        # Cancel any existing background task
        await self.stop()
        task = asyncio.create_task(_continuous(), name=f"background_{self.name}")
        if block:
            try:
                return await task  # type: ignore
            finally:
                if not task.done():
                    task.cancel()
        else:
            logger.debug("%s: Started background task %s", self.name, task.get_name())
            self._background_task = task
            return None

    async def stop(self):
        """Stop continuous execution if running."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            await self._background_task
            self._background_task = None

    async def wait(self) -> ChatMessage[TResult]:
        """Wait for background execution to complete."""
        if not self._background_task:
            msg = "No background task running"
            raise RuntimeError(msg)
        if self._infinite:
            msg = "Cannot wait on infinite execution"
            raise RuntimeError(msg)
        try:
            return await self._background_task
        finally:
            self._background_task = None

    def clear_history(self):
        """Clear both internal and pydantic-ai history."""
        self._logger.clear_state()
        self.conversation.clear()
        logger.debug("Cleared history and reset tool state")

    async def share(
        self,
        target: AnyAgent[TDeps, Any],
        *,
        tools: list[str] | None = None,
        resources: list[str] | None = None,
        history: bool | int | None = None,  # bool or number of messages
        token_limit: int | None = None,
    ):
        """Share capabilities and knowledge with another agent.

        Args:
            target: Agent to share with
            tools: List of tool names to share
            resources: List of resource names to share
            history: Share conversation history:
                    - True: Share full history
                    - int: Number of most recent messages to share
                    - None: Don't share history
            token_limit: Optional max tokens for history

        Raises:
            ValueError: If requested items don't exist
            RuntimeError: If runtime not available for resources
        """
        # Share tools if requested
        for name in tools or []:
            if tool := self.tools.get(name):
                meta = {"shared_from": self.name}
                target.tools.register_tool(tool.callable, metadata=meta)
            else:
                msg = f"Tool not found: {name}"
                raise ValueError(msg)

        # Share resources if requested
        if resources:
            if not self.runtime:
                msg = "No runtime available for sharing resources"
                raise RuntimeError(msg)
            for name in resources:
                if resource := self.runtime.get_resource(name):
                    await target.conversation.load_context_source(resource)  # type: ignore
                else:
                    msg = f"Resource not found: {name}"
                    raise ValueError(msg)

        # Share history if requested
        if history:
            history_text = await self.conversation.format_history(
                max_tokens=token_limit,
                num_messages=history if isinstance(history, int) else None,
            )
            target.conversation.add_context_message(
                history_text, source=self.name, metadata={"type": "shared_history"}
            )

    def register_worker(
        self,
        worker: MessageNode[Any, Any],
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        share_context: bool = False,
    ) -> Tool:
        """Register another agent as a worker tool."""
        return self.tools.register_worker(
            worker,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            share_context=share_context,
            parent=self if (pass_message_history or share_context) else None,
        )

    def set_model(self, model: ModelType):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        Emits:
            model_changed signal with the new model
        """
        self._provider.set_model(model)

    async def reset(self):
        """Reset agent state (conversation history and tool states)."""
        old_tools = await self.tools.list_tools()
        self.conversation.clear()
        self.tools.reset_states()
        new_tools = await self.tools.list_tools()

        event = self.AgentReset(
            agent_name=self.name,
            previous_tools=old_tools,
            new_tools=new_tools,
        )
        self.agent_reset.emit(event)

    @property
    def runtime(self) -> RuntimeConfig:
        """Get runtime configuration from context."""
        assert self.context.runtime
        return self.context.runtime

    @runtime.setter
    def runtime(self, value: RuntimeConfig):
        """Set runtime configuration and update context."""
        self.context.runtime = value

    @property
    def stats(self) -> MessageStats:
        return MessageStats(messages=self._logger.message_history)

    @asynccontextmanager
    async def temporary_state(
        self,
        *,
        system_prompts: list[AnyPromptType] | None = None,
        replace_prompts: bool = False,
        tools: list[ToolType] | None = None,
        replace_tools: bool = False,
        history: list[AnyPromptType] | SessionQuery | None = None,
        replace_history: bool = False,
        pause_routing: bool = False,
        model: ModelType | None = None,
        provider: AgentProvider | None = None,
    ) -> AsyncIterator[Self]:
        """Temporarily modify agent state.

        Args:
            system_prompts: Temporary system prompts to use
            replace_prompts: Whether to replace existing prompts
            tools: Temporary tools to make available
            replace_tools: Whether to replace existing tools
            history: Conversation history (prompts or query)
            replace_history: Whether to replace existing history
            pause_routing: Whether to pause message routing
            model: Temporary model override
            provider: Temporary provider override
        """
        old_model = self._provider.model if hasattr(self._provider, "model") else None  # pyright: ignore
        old_provider = self._provider

        async with AsyncExitStack() as stack:
            # System prompts (async)
            if system_prompts is not None:
                await stack.enter_async_context(
                    self.sys_prompts.temporary_prompt(
                        system_prompts, exclusive=replace_prompts
                    )
                )

            # Tools (sync)
            if tools is not None:
                stack.enter_context(
                    self.tools.temporary_tools(tools, exclusive=replace_tools)
                )

            # History (async)
            if history is not None:
                await stack.enter_async_context(
                    self.conversation.temporary_state(
                        history, replace_history=replace_history
                    )
                )

            # Routing (async)
            if pause_routing:
                await stack.enter_async_context(self.connections.paused_routing())

            # Model/Provider
            if provider is not None:
                self._provider = provider
            elif model is not None:
                self._provider.set_model(model)

            try:
                yield self
            finally:
                # Restore model/provider
                if provider is not None:
                    self._provider = old_provider
                elif model is not None and old_model:
                    self._provider.set_model(old_model)


if __name__ == "__main__":
    import logging

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])
    sys_prompt = "Open browser with google,"
    _model = "openai:gpt-5-nano"

    async def main():
        async with Agent(model=_model, tools=["webbrowser.open"]) as agent:
            agent.tool_used.connect(print)
            async for chunk in agent.run_stream(sys_prompt):
                print(chunk)

    asyncio.run(main())
