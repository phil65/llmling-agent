"""The main Agent. Can do all sort of crazy things."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Self,
    TypedDict,
    get_type_hints,
    overload,
)
from uuid import uuid4

from anyenv import MultiEventHandler, method_spawner
from llmling_models import function_to_model, infer_model
import logfire
from psygnal import Signal
from pydantic import ValidationError
from pydantic_ai import (
    Agent as PydanticAgent,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartStartEvent,
    RunContext,
    ToolCallPart,
    ToolReturnPart,
    models,
)
from pydantic_ai.models import Model

from llmling_agent.agent.events import (
    RichAgentStreamEvent,
    StreamCompleteEvent,
    ToolCallCompleteEvent,
    ToolCallProgressEvent,
    create_queuing_progress_handler,
)
from llmling_agent.common_types import IndividualEventHandler
from llmling_agent.log import get_logger
from llmling_agent.messaging import ChatMessage, MessageNode
from llmling_agent.prompts.convert import convert_prompts
from llmling_agent.talk.stats import MessageStats
from llmling_agent.tasks import JobError
from llmling_agent.tools import SkillsRegistry, Tool, ToolManager
from llmling_agent.tools.exceptions import ToolError
from llmling_agent.utils.inspection import call_with_context, get_argument_key
from llmling_agent.utils.now import get_now
from llmling_agent.utils.result_utils import to_type
from llmling_agent.utils.streams import merge_queue_into_iterator
from llmling_agent.utils.tasks import TaskManager
from llmling_agent_config.session import MemoryConfig


if TYPE_CHECKING:
    from collections.abc import AsyncIterable, AsyncIterator, Coroutine, Sequence
    from datetime import datetime
    from types import TracebackType

    from pydantic_ai import AgentStreamEvent, UsageLimits
    from pydantic_ai.output import OutputSpec
    from toprompt import AnyPromptType

    from llmling_agent.agent import AgentContext
    from llmling_agent.agent.events import RichAgentStreamEvent
    from llmling_agent.common_types import (
        AgentName,
        EndStrategy,
        ModelType,
        ProcessorCallback,
        PromptCompatible,
        SessionIdType,
        ToolType,
    )
    from llmling_agent.delegation import Team, TeamRun
    from llmling_agent.prompts.prompts import PromptType
    from llmling_agent.resource_providers import ResourceProvider
    from llmling_agent.ui.base import InputProvider
    from llmling_agent_config.mcp_server import MCPServerConfig
    from llmling_agent_config.output_types import StructuredResponseConfig
    from llmling_agent_config.session import SessionQuery
    from llmling_agent_config.task import Job


logger = get_logger(__name__)
# OutputDataT = TypeVar('OutputDataT', default=str, covariant=True)
NoneType = type(None)


class AgentKwargs(TypedDict, total=False):
    """Keyword arguments for configuring an Agent instance."""

    description: str | None
    model: ModelType
    system_prompt: str | Sequence[str]
    tools: Sequence[ToolType | Tool] | None
    toolsets: Sequence[ResourceProvider] | None
    mcp_servers: Sequence[str | MCPServerConfig] | None
    retries: int
    output_retries: int | None
    end_strategy: EndStrategy
    context: AgentContext[Any] | None  # x
    session: SessionIdType | SessionQuery | MemoryConfig | bool | int
    input_provider: InputProvider | None
    debug: bool
    event_handlers: Sequence[IndividualEventHandler] | None


class Agent[TDeps = None, OutputDataT = str](MessageNode[TDeps, OutputDataT]):
    """The main agent class.

    Generically typed with: LLMLingAgent[Type of Dependencies, Type of Result]
    """

    @dataclass(frozen=True)
    class AgentReset:
        """Emitted when agent is reset."""

        agent_name: AgentName
        previous_tools: dict[str, bool]
        new_tools: dict[str, bool]
        timestamp: datetime = field(default_factory=get_now)

    run_failed = Signal(str, Exception)
    agent_reset = Signal(AgentReset)

    def __init__(
        # we dont use AgentKwargs here so that we can work with explicit ones in the ctor
        self,
        name: str = "llmling-agent",
        *,
        deps_type: type[TDeps] | None = None,
        model: ModelType = None,
        output_type: OutputSpec[OutputDataT] | StructuredResponseConfig | str = str,  # type: ignore[assignment]
        context: AgentContext[TDeps] | None = None,
        session: SessionIdType | SessionQuery | MemoryConfig | bool | int = None,
        system_prompt: AnyPromptType | Sequence[AnyPromptType] = (),
        description: str | None = None,
        tools: Sequence[ToolType | Tool] | None = None,
        toolsets: Sequence[ResourceProvider] | None = None,
        mcp_servers: Sequence[str | MCPServerConfig] | None = None,
        resources: Sequence[PromptType | str] = (),
        retries: int = 1,
        output_retries: int | None = None,
        end_strategy: EndStrategy = "early",
        input_provider: InputProvider | None = None,
        parallel_init: bool = True,
        debug: bool = False,
        event_handlers: Sequence[IndividualEventHandler] | None = None,
    ):
        """Initialize agent.

        Args:
            name: Name of the agent for logging and identification
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
            tools: List of tools to register with the agent
            toolsets: List of toolset resource providers for the agent
            mcp_servers: MCP servers to connect to
            resources: Additional resources to load
            retries: Default number of retries for failed operations
            output_retries: Max retries for result validation (defaults to retries)
            end_strategy: Strategy for handling tool calls that are requested alongside
                          a final result
            input_provider: Provider for human input (tool confirmation / HumanProviders)
            parallel_init: Whether to initialize resources in parallel
            debug: Whether to enable debug mode
            event_handlers: Sequence of event handlers to register with the agent
        """
        from llmling_agent.agent import AgentContext
        from llmling_agent.agent.conversation import ConversationManager
        from llmling_agent.agent.interactions import Interactions
        from llmling_agent.agent.sys_prompts import SystemPrompts
        from llmling_agent.observability import registry

        self.task_manager = TaskManager()
        self._infinite = False
        # save some stuff for asnyc init
        self.deps_type = deps_type
        # prepare context
        ctx = context or AgentContext[TDeps].create_default(
            name,
            input_provider=input_provider,
        )
        self._context = ctx
        # TODO: use set_output_type with tool_name / description?
        self._output_type = to_type(output_type, ctx.definition.responses)
        memory_cfg = (
            session
            if isinstance(session, MemoryConfig)
            else MemoryConfig.from_value(session)
        )
        # Initialize progress queue before super().__init__()
        self._progress_queue = asyncio.Queue[ToolCallProgressEvent]()
        super().__init__(
            name=name,
            context=ctx,
            description=description,
            enable_logging=memory_cfg.enable,
            mcp_servers=mcp_servers,
            progress_handler=create_queuing_progress_handler(self._progress_queue),
        )

        # Initialize tool manager
        self.event_handler = MultiEventHandler[IndividualEventHandler](event_handlers)
        all_tools = list(tools or [])
        self.tools = ToolManager(all_tools)

        # MCP manager will be initialized in __aenter__ and providers added there
        if builtin_tools := ctx.config.get_tool_provider():
            self.tools.add_provider(builtin_tools)

        for toolset_provider in toolsets or []:
            self.tools.add_provider(toolset_provider)

        # Initialize conversation manager
        resources = list(resources)
        if ctx.config.knowledge:
            resources.extend(ctx.config.knowledge.get_resources())
        self.conversation = ConversationManager(self, memory_cfg, resources=resources)

        # Store pydantic-ai configuration
        if model and not isinstance(model, str):
            assert isinstance(model, models.Model)
        self._model = model
        self._retries = retries
        self._end_strategy: EndStrategy = end_strategy
        self._output_retries = output_retries

        self.skills_registry = SkillsRegistry()

        if ctx and ctx.definition:
            registry.configure_observability(ctx.definition.observability)

        # init variables
        self._debug = debug
        self.parallel_init = parallel_init
        self._name = name
        self._background_task: asyncio.Task[Any] | None = None

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
        return f"Agent({self.name!r}, model={self._model!r}{desc})"

    async def __prompt__(self) -> str:
        typ = self.__class__.__name__
        model = self.model_name or "default"
        parts = [f"Agent: {self.name}", f"Type: {typ}", f"Model: {model}"]
        if self.description:
            parts.append(f"Description: {self.description}")
        parts.extend([await self.tools.__prompt__(), self.conversation.__prompt__()])
        return "\n".join(parts)

    async def __aenter__(self) -> Self:
        """Enter async context and set up MCP servers."""
        try:
            # Collect all coroutines that need to be run
            coros: list[Coroutine[Any, Any, Any]] = []

            coros.append(super().__aenter__())
            coros.extend(self.conversation.get_initialization_tasks())

            # Execute coroutines either in parallel or sequentially
            if self.parallel_init and coros:
                await asyncio.gather(*coros)
            else:
                for coro in coros:
                    await coro

            # Add MCP aggregating provider after manager is initialized
            aggregating_provider = self.mcp.get_aggregating_provider()
            self.tools.add_provider(aggregating_provider)

            for toolset_provider in self.context.config.get_toolsets():
                self.tools.add_provider(toolset_provider)
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
    ):
        """Exit async context."""
        await super().__aexit__(exc_type, exc_val, exc_tb)
        # MCP manager cleanup is handled by MessageEmitter's __aexit__

    @overload
    def __and__(  # if other doesnt define deps, we take the agents one
        self, other: ProcessorCallback[Any] | Team[TDeps] | Agent[TDeps, Any]
    ) -> Team[TDeps]: ...

    @overload
    def __and__(  # otherwise, we dont know and deps is Any
        self, other: ProcessorCallback[Any] | Team[Any] | Agent[Any, Any]
    ) -> Team[Any]: ...

    def __and__(self, other: MessageNode[Any, Any] | ProcessorCallback[Any]) -> Team[Any]:
        """Create sequential team using & operator.

        Example:
            group = analyzer & planner & executor  # Create group of 3
            group = analyzer & existing_group  # Add to existing group
        """
        from llmling_agent.delegation.team import Team

        match other:
            case Team():
                return Team([self, *other.agents])
            case Callable():
                agent_2 = Agent.from_callback(other)
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
        from llmling_agent import TeamRun

        if callable(other):
            other = Agent.from_callback(other)
            other.context.pool = self.context.pool

        return TeamRun([self, other])

    @classmethod
    def from_callback[TResult](
        cls,
        callback: ProcessorCallback[TResult],
        *,
        name: str | None = None,
        **kwargs: Any,
    ) -> Agent[None, TResult]:
        """Create an agent from a processing callback.

        Args:
            callback: Function to process messages. Can be:
                - sync or async
                - with or without context
                - must return str for pipeline compatibility
            name: Optional name for the agent
            kwargs: Additional arguments for agent
        """
        name = name or callback.__name__ or "processor"

        model = function_to_model(callback)
        # Get return type from signature for validation
        hints = get_type_hints(callback)
        return_type = hints.get("return")

        # If async, unwrap from Awaitable
        if (
            return_type
            and hasattr(return_type, "__origin__")
            and return_type.__origin__ is Awaitable
        ):
            return_type = return_type.__args__[0]
        return Agent(
            model=model,
            name=name,
            output_type=return_type or str,
            **kwargs,
        )

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name or "llmling-agent"

    @property
    def context(self) -> AgentContext[TDeps]:
        """Get agent context."""
        return self._context

    @context.setter
    def context(self, value: AgentContext[TDeps]):
        """Set agent context and propagate to provider."""
        self._context = value
        self.mcp.context = value

    def set_output_type(
        self,
        output_type: type | str | StructuredResponseConfig | None,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ):
        """Set or update the result type for this agent.

        Args:
            output_type: New result type, can be:
                - A Python type for validation
                - Name of a response definition
                - Response definition instance
                - None to reset to unstructured mode
            tool_name: Optional override for tool name
            tool_description: Optional override for tool description
        """
        self.log.debug("Setting result type", output_type=output_type)
        self._output_type = to_type(output_type)

    def to_structured[NewOutputDataT](
        self,
        output_type: type[NewOutputDataT] | str | StructuredResponseConfig,
        *,
        tool_name: str | None = None,
        tool_description: str | None = None,
    ) -> Agent[TDeps, NewOutputDataT] | Self:
        """Convert this agent to a structured agent.

        Args:
            output_type: Type for structured responses. Can be:
                - A Python type (Pydantic model)
                - Name of response definition from context
                - Complete response definition
            tool_name: Optional override for result tool name
            tool_description: Optional override for result tool description

        Returns:
            Typed Agent
        """
        self.set_output_type(output_type)  # type: ignore
        return self

    def is_busy(self) -> bool:
        """Check if agent is currently processing tasks."""
        return bool(self.task_manager._pending_tasks or self._background_task)

    @property
    def model_name(self) -> str | None:
        """Get the model name in a consistent format."""
        return self._model.model_name if isinstance(self._model, Model) else self._model

    def to_tool(
        self,
        *,
        name: str | None = None,
        reset_history_on_run: bool = True,
        pass_message_history: bool = False,
        parent: Agent[Any, Any] | None = None,
    ) -> Tool[OutputDataT]:
        """Create a tool from this agent.

        Args:
            name: Optional tool name override
            reset_history_on_run: Clear agent's history before each run
            pass_message_history: Pass parent's message history to agent
            parent: Optional parent agent for history/context sharing
        """
        tool_name = name or f"ask_{self.name}"
        # Create wrapper function with correct return type annotation
        output_type = self._output_type or Any

        async def wrapped_tool(prompt: str):
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
            result = await self.run(prompt)
            if history:
                self.conversation.set_history(old)
            return result.data

        # Set the correct return annotation dynamically
        wrapped_tool.__annotations__ = {"prompt": str, "return": output_type}

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
            source="agent",
        )

    async def get_agentlet(
        self,
        tools: list[Tool],
        model: ModelType = None,
        output_type: type[Any] = str,
        deps_type: type[Any] | None = None,
        input_provider: InputProvider | None = None,
    ):
        """Create pydantic-ai agent from current state."""
        # Monkey patch pydantic-ai to recognize AgentContext

        from llmling_agent.agent.tool_wrapping import wrap_tool

        actual_model = model or self._model
        model_ = (
            infer_model(actual_model) if isinstance(actual_model, str) else actual_model
        )

        agent = PydanticAgent(
            name=self.name,
            model=model_,
            instructions=await self.sys_prompts.format_system_prompt(self),
            retries=self._retries,
            end_strategy=self._end_strategy,
            output_retries=self._output_retries,
            deps_type=deps_type or NoneType,
            output_type=output_type,
        )

        # If input_provider override is provided, create modified context
        context_for_tools = (
            self.context
            if input_provider is None
            else replace(self.context, input_provider=input_provider)
        )

        for tool in tools:
            wrapped = wrap_tool(tool, context_for_tools)
            if get_argument_key(wrapped, RunContext):
                agent.tool(wrapped)
            else:
                agent.tool_plain(wrapped)

        return agent

    async def _run(
        self,
        *prompts: PromptCompatible | ChatMessage[Any],
        output_type: type[OutputDataT] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        tool_choice: str | list[str] | None = None,
        usage_limits: UsageLimits | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[ChatMessage[Any]] | None = None,
        deps: TDeps | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
    ) -> ChatMessage[OutputDataT]:
        """Run agent with prompt and get response.

        Args:
            prompts: User query or instruction
            output_type: Optional type for structured responses
            model: Optional model override
            store_history: Whether the message exchange should be added to the
                            context window
            tool_choice: Filter tool choice by name
            usage_limits: Optional usage limits for the model
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            conversation_id: Optional conversation id for the returned message.
            messages: Optional list of messages to replace the conversation history
            deps: Optional dependencies for the agent
            input_provider: Optional input provider for the agent
            wait_for_connections: Whether to wait for connected agents to complete

        Returns:
            Result containing response and run information

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        """Run agent with prompt and get response."""
        message_id = message_id or str(uuid4())
        tools = await self.tools.get_tools(state="enabled", names=tool_choice)
        final_type = to_type(output_type) if output_type else self._output_type
        start_time = time.perf_counter()

        message_history = (
            messages if messages is not None else self.conversation.get_history()
        )
        try:
            # Create pydantic-ai agent for this run
            agentlet = await self.get_agentlet(
                tools, model, final_type, self.deps_type, input_provider
            )
            converted_prompts = await convert_prompts(prompts)

            # Merge internal and external event handlers like the old provider did
            async def event_distributor(
                ctx: RunContext, events: AsyncIterable[AgentStreamEvent]
            ):
                async for event in events:
                    for handler in self.event_handler._wrapped_handlers:
                        await handler(ctx, event)

            result = await agentlet.run(
                [
                    i if isinstance(i, str) else i.to_pydantic_ai()
                    for i in converted_prompts
                ],
                deps=deps,
                message_history=[
                    m for run in message_history for m in run.to_pydantic_ai()
                ],
                output_type=final_type or str,
                usage_limits=usage_limits,
                event_stream_handler=event_distributor,
            )

            response_time = time.perf_counter() - start_time
            response_msg = await ChatMessage.from_run_result(
                result,
                agent_name=self.name,
                message_id=message_id,
                conversation_id=conversation_id,
                response_time=response_time,
            )

        except Exception as e:
            self.log.exception("Agent run failed")
            self.run_failed.emit("Agent run failed", e)
            raise
        else:
            return response_msg

    @method_spawner
    async def run_stream(
        self,
        *prompt: PromptCompatible,
        output_type: type[OutputDataT] | None = None,
        model: ModelType = None,
        tool_choice: str | list[str] | None = None,
        store_history: bool = True,
        usage_limits: UsageLimits | None = None,
        message_id: str | None = None,
        conversation_id: str | None = None,
        messages: list[ChatMessage[Any]] | None = None,
        input_provider: InputProvider | None = None,
        wait_for_connections: bool | None = None,
        deps: TDeps | None = None,
    ) -> AsyncIterator[RichAgentStreamEvent[OutputDataT]]:
        """Run agent with prompt and get a streaming response.

        Args:
            prompt: User query or instruction
            output_type: Optional type for structured responses
            model: Optional model override
            tool_choice: Filter tool choice by name
            store_history: Whether the message exchange should be added to the
                           context window
            usage_limits: Optional usage limits for the model
            message_id: Optional message id for the returned message.
                        Automatically generated if not provided.
            conversation_id: Optional conversation id for the returned message.
            messages: Optional list of messages to replace the conversation history
            input_provider: Optional input provider for the agent
            wait_for_connections: Whether to wait for connected agents to complete
            deps: Optional dependencies for the agent
        Returns:
            An async iterator yielding streaming events with final message embedded.

        Raises:
            UnexpectedModelBehavior: If the model fails or behaves unexpectedly
        """
        message_id = message_id or str(uuid4())
        user_msg, prompts = await self.pre_run(*prompt)
        final_type = to_type(output_type) if output_type else self._output_type
        start_time = time.perf_counter()
        tools = await self.tools.get_tools(state="enabled", names=tool_choice)
        message_history = (
            messages if messages is not None else self.conversation.get_history()
        )
        try:
            agentlet = await self.get_agentlet(
                tools, model, final_type, self.deps_type, input_provider
            )
            content = await convert_prompts(prompts)
            # Initialize variables for final response
            response_msg = None
            # Create tool dict for signal emission
            converted = [i if isinstance(i, str) else i.to_pydantic_ai() for i in content]
            stream_events = agentlet.run_stream_events(
                converted,
                deps=deps,
                message_history=[
                    m for run in message_history for m in run.to_pydantic_ai()
                ],
                output_type=final_type or str,
                usage_limits=usage_limits,
            )

            # Stream events through merge_queue for progress events
            async with merge_queue_into_iterator(
                stream_events, self._progress_queue
            ) as events:
                # Track tool call starts to combine with results later
                pending_tcs: dict[str, ToolCallPart] = {}
                async for event in events:
                    # Process events and emit signals
                    yield event
                    match event:
                        case (
                            PartStartEvent(part=ToolCallPart() as tool_part)
                            | FunctionToolCallEvent(part=tool_part)
                        ):
                            # Store tool call start info for later combination with result
                            pending_tcs[tool_part.tool_call_id] = tool_part
                        case (
                            FunctionToolResultEvent(tool_call_id=call_id) as result_event
                        ):
                            # Check if we have a pending tool call to combine with
                            if call_info := pending_tcs.pop(call_id, None):
                                # Create and yield combined event
                                combined_event = ToolCallCompleteEvent(
                                    tool_name=call_info.tool_name,
                                    tool_call_id=call_id,
                                    tool_input=call_info.args_as_dict(),
                                    tool_result=result_event.result.content
                                    if isinstance(result_event.result, ToolReturnPart)
                                    else result_event.result,
                                    agent_name=self.name,
                                    message_id=message_id,
                                )
                                yield combined_event
                        case AgentRunResultEvent():
                            # Capture final result data
                            # Build final response message
                            response_time = time.perf_counter() - start_time
                            response_msg = await ChatMessage.from_run_result(
                                event.result,
                                agent_name=self.name,
                                message_id=message_id,
                                conversation_id=conversation_id,
                                response_time=response_time,
                            )

            # Send additional enriched completion event
            assert response_msg
            yield StreamCompleteEvent(message=response_msg)
            self.message_sent.emit(response_msg)
            await self.log_message(response_msg)
            if store_history:
                self.conversation.add_chat_messages([user_msg, response_msg])
            await self.connections.route_message(
                response_msg,
                wait=wait_for_connections,
            )

        except Exception as e:
            self.log.exception("Agent stream failed")
            self.run_failed.emit("Agent stream failed", e)
            raise

    async def run_iter(
        self,
        *prompt_groups: Sequence[PromptCompatible],
        output_type: type[OutputDataT] | None = None,
        model: ModelType = None,
        store_history: bool = True,
        wait_for_connections: bool | None = None,
    ) -> AsyncIterator[ChatMessage[OutputDataT]]:
        """Run agent sequentially on multiple prompt groups.

        Args:
            prompt_groups: Groups of prompts to process sequentially
            output_type: Optional type for structured responses
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
                output_type=output_type,
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
            for source in list(job.knowledge.paths):
                await self.conversation.load_context_source(source)
            for prompt in job.knowledge.prompts:
                await self.conversation.load_context_source(prompt)
        try:
            # Register task tools temporarily
            tools = job.get_tools()
            async with self.tools.temporary_tools(
                tools, exclusive=not include_agent_tools
            ):
                # Execute job with job-specific tools
                return await self.run(await job.get_prompt(), store_history=store_history)

        except Exception as e:
            self.log.exception("Task execution failed", error=str(e))
            msg = f"Task execution failed: {e}"
            raise JobError(msg) from e

    async def run_in_background(
        self,
        *prompt: PromptCompatible,
        max_count: int | None = None,
        interval: float = 1.0,
        block: bool = False,
        **kwargs: Any,
    ) -> ChatMessage[OutputDataT] | None:
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
            self.log.debug(
                "Starting continuous run", max_count=max_count, interval=interval
            )
            latest = None
            while max_count is None or count < max_count:
                try:
                    current_prompts = [
                        call_with_context(p, self.context, **kwargs) if callable(p) else p
                        for p in prompt
                    ]
                    self.log.debug("Generated prompt", iteration=count)
                    latest = await self.run(current_prompts, **kwargs)
                    self.log.debug("Run continuous result", iteration=count)

                    count += 1
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    self.log.debug("Continuous run cancelled")
                    break
                except Exception:
                    self.log.exception("Background run failed")
                    await asyncio.sleep(interval)
            self.log.debug("Continuous run completed", iterations=count)
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
            self.log.debug("Started background task", task_name=task.get_name())
            self._background_task = task
            return None

    async def stop(self):
        """Stop continuous execution if running."""
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            await self._background_task
            self._background_task = None

    async def wait(self) -> ChatMessage[OutputDataT]:
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

    async def share(
        self,
        target: Agent[TDeps, Any],
        *,
        tools: list[str] | None = None,
        history: bool | int | None = None,  # bool or number of messages
        token_limit: int | None = None,
    ):
        """Share capabilities and knowledge with another agent.

        Args:
            target: Agent to share with
            tools: List of tool names to share
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
            if tool := await self.tools.get_tool(name):
                meta = {"shared_from": self.name}
                target.tools.register_tool(tool.callable, metadata=meta)
            else:
                msg = f"Tool not found: {name}"
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
    ) -> Tool:
        """Register another agent as a worker tool."""
        return self.tools.register_worker(
            worker,
            name=name,
            reset_history_on_run=reset_history_on_run,
            pass_message_history=pass_message_history,
            parent=self if pass_message_history else None,
        )

    def set_model(self, model: ModelType):
        """Set the model for this agent.

        Args:
            model: New model to use (name or instance)

        """
        self._model = model

    async def reset(self):
        """Reset agent state (conversation history and tool states)."""
        old_tools = await self.tools.list_tools()
        self.conversation.clear()
        await self.tools.reset_states()
        new_tools = await self.tools.list_tools()

        event = self.AgentReset(
            agent_name=self.name,
            previous_tools=old_tools,
            new_tools=new_tools,
        )
        self.agent_reset.emit(event)

    async def get_stats(self) -> MessageStats:
        """Get message statistics (async version)."""
        messages = await self.get_message_history()
        return MessageStats(messages=messages)

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
        if output_type:
            old_type = self._output_type
            self.set_output_type(output_type)
        async with AsyncExitStack() as stack:
            if system_prompts is not None:  # System prompts
                await stack.enter_async_context(
                    self.sys_prompts.temporary_prompt(
                        system_prompts, exclusive=replace_prompts
                    )
                )

            if tools is not None:  # Tools
                await stack.enter_async_context(
                    self.tools.temporary_tools(tools, exclusive=replace_tools)
                )

            if history is not None:  # History
                await stack.enter_async_context(
                    self.conversation.temporary_state(
                        history, replace_history=replace_history
                    )
                )

            if pause_routing:  # Routing
                await stack.enter_async_context(self.connections.paused_routing())

            elif model is not None:  # Model
                self._model = model

            try:
                yield self
            finally:  # Restore model
                if model is not None and old_model:
                    self._model = old_model
                if output_type:
                    self.set_output_type(old_type)

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


if __name__ == "__main__":
    import logging

    logfire.configure()
    logfire.instrument_pydantic_ai()
    logging.basicConfig(handlers=[logfire.LogfireLoggingHandler()])
    sys_prompt = "Open browser with google,"
    _model = "openai:gpt-5-nano"

    async def handle_events(ctx, event):
        print(f"[EVENT] {type(event).__name__}: {event}")

    agent = Agent(model=_model, tools=["webbrowser.open"], event_handlers=[handle_events])
    result = agent.run.sync(sys_prompt)
