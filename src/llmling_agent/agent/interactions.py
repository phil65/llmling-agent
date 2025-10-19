"""Agent interaction patterns."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from schemez import Schema, create_constructor_schema

from llmling_agent.log import get_logger
from llmling_agent.messaging.messages import ChatMessage
from llmling_agent.tools.base import Tool


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Sequence

    from toprompt import AnyPromptType

    from llmling_agent import AgentPool, MessageNode
    from llmling_agent.agent import AnyAgent
    from llmling_agent.delegation.base_team import BaseTeam


ExtractionMode = Literal["structured", "tool_calls"]

type EndCondition = Callable[[list[ChatMessage[Any]], ChatMessage[Any]], bool]

logger = get_logger(__name__)


class LLMPick(Schema):
    """Decision format for LLM response."""

    selection: str  # The label/name of the selected option
    reason: str


class Pick[T](Schema):
    """Type-safe decision with original object."""

    selection: T
    reason: str


class LLMMultiPick(Schema):
    """Multiple selection format for LLM response."""

    selections: list[str]  # Labels of selected options
    reason: str


class MultiPick[T](Schema):
    """Type-safe multiple selection with original objects."""

    selections: list[T]
    reason: str


def get_label(item: Any) -> str:
    """Get label for an item to use in selection.

    Args:
        item: Item to get label for

    Returns:
        Label to use for selection

    Strategy:
        - strings stay as-is
        - types use __name__
        - others use __repr__ for unique identifiable string
    """
    from llmling_agent import MessageNode

    match item:
        case str():
            return item
        case type():
            return item.__name__
        case MessageNode():
            return item.name or "unnamed_agent"
        case _:
            return repr(item)


class Interactions[TDeps, TResult]:
    """Manages agent communication patterns."""

    def __init__(self, agent: AnyAgent[TDeps, TResult]):
        self.agent = agent

    async def conversation(
        self,
        other: MessageNode[Any, Any],
        initial_message: AnyPromptType,
        *,
        max_rounds: int | None = None,
        end_condition: Callable[[list[ChatMessage[Any]], ChatMessage[Any]], bool]
        | None = None,
        store_history: bool = True,
    ) -> AsyncIterator[ChatMessage[Any]]:
        """Maintain conversation between two agents.

        Args:
            other: Agent to converse with
            initial_message: Message to start conversation with
            max_rounds: Optional maximum number of exchanges
            end_condition: Optional predicate to check for conversation end
            store_history: Whether to store in conversation history

        Yields:
            Messages from both agents in conversation order
        """
        rounds = 0
        messages: list[ChatMessage[Any]] = []
        current_message = initial_message
        current_node: MessageNode[Any, Any] = self.agent

        while True:
            if max_rounds and rounds >= max_rounds:
                logger.debug("Conversation ended: max rounds (%d) reached", max_rounds)
                return

            response = await current_node.run(
                current_message, store_history=store_history
            )
            messages.append(response)
            yield response

            if end_condition and end_condition(messages, response):
                logger.debug("Conversation ended: end condition met")
                return

            # Switch agents for next round
            current_node = other if current_node == self.agent else self.agent
            current_message = response.content
            rounds += 1

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick[T: AnyPromptType](
        self,
        selections: Mapping[str, T],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]: ...

    @overload
    async def pick(
        self,
        selections: AgentPool,
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[AnyAgent[Any, Any]]: ...

    @overload
    async def pick(
        self,
        selections: BaseTeam[TDeps, Any],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[MessageNode[TDeps, Any]]: ...

    async def pick[T](
        self,
        selections: Sequence[T] | Mapping[str, T] | AgentPool | BaseTeam[TDeps, Any],
        task: str,
        prompt: AnyPromptType | None = None,
    ) -> Pick[T]:
        """Pick from available options with reasoning.

        Args:
            selections: What to pick from:
                - Sequence of items (auto-labeled)
                - Dict mapping labels to items
                - AgentPool
                - Team
            task: Task/decision description
            prompt: Optional custom selection prompt

        Returns:
            Decision with selected item and reasoning

        Raises:
            ValueError: If no choices available or invalid selection
        """
        # Get items and create label mapping
        from toprompt import to_prompt

        from llmling_agent import AgentPool
        from llmling_agent.delegation.base_team import BaseTeam

        match selections:
            case dict():
                label_map = selections
                items: list[Any] = list(selections.values())
            case BaseTeam():
                items = list(selections.agents)
                label_map = {get_label(item): item for item in items}
            case AgentPool():
                items = list(selections.agents.values())
                label_map = {get_label(item): item for item in items}
            case _:
                items = list(selections)
                label_map = {get_label(item): item for item in items}

        if not items:
            msg = "No choices available"
            raise ValueError(msg)

        # Get descriptions for all items
        descriptions = []
        for label, item in label_map.items():
            item_desc = await to_prompt(item)
            descriptions.append(f"{label}:\n{item_desc}")

        default_prompt = f"""Task/Decision: {task}

Available options:
{"-" * 40}
{"\n\n".join(descriptions)}
{"-" * 40}

Select ONE option by its exact label."""

        # Get LLM's string-based decision
        result = await self.agent.to_structured(LLMPick).run(prompt or default_prompt)

        # Convert to type-safe decision
        if result.content.selection not in label_map:
            msg = f"Invalid selection: {result.content.selection}"
            raise ValueError(msg)

        selected = cast(T, label_map[result.content.selection])
        return Pick(selection=selected, reason=result.content.reason)

    @overload
    async def pick_multiple[T: AnyPromptType](
        self,
        selections: Sequence[T],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]: ...

    @overload
    async def pick_multiple[T: AnyPromptType](
        self,
        selections: Mapping[str, T],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]: ...

    @overload
    async def pick_multiple(
        self,
        selections: BaseTeam[TDeps, Any],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[MessageNode[TDeps, Any]]: ...

    @overload
    async def pick_multiple(
        self,
        selections: AgentPool,
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[AnyAgent[Any, Any]]: ...

    async def pick_multiple[T](
        self,
        selections: Sequence[T] | Mapping[str, T] | AgentPool | BaseTeam[TDeps, Any],
        task: str,
        *,
        min_picks: int = 1,
        max_picks: int | None = None,
        prompt: AnyPromptType | None = None,
    ) -> MultiPick[T]:
        """Pick multiple options from available choices.

        Args:
            selections: What to pick from
            task: Task/decision description
            min_picks: Minimum number of selections required
            max_picks: Maximum number of selections (None for unlimited)
            prompt: Optional custom selection prompt
        """
        from toprompt import to_prompt

        from llmling_agent import AgentPool
        from llmling_agent.delegation.base_team import BaseTeam

        match selections:
            case Mapping():
                label_map = selections
                items: list[Any] = list(selections.values())
            case BaseTeam():
                items = list(selections.agents)
                label_map = {get_label(item): item for item in items}
            case AgentPool():
                items = list(selections.agents.values())
                label_map = {get_label(item): item for item in items}
            case _:
                items = list(selections)
                label_map = {get_label(item): item for item in items}

        if not items:
            msg = "No choices available"
            raise ValueError(msg)

        if max_picks is not None and max_picks < min_picks:
            msg = f"max_picks ({max_picks}) cannot be less than min_picks ({min_picks})"
            raise ValueError(msg)

        descriptions = []
        for label, item in label_map.items():
            item_desc = await to_prompt(item)
            descriptions.append(f"{label}:\n{item_desc}")

        picks_info = (
            f"Select between {min_picks} and {max_picks}"
            if max_picks is not None
            else f"Select at least {min_picks}"
        )

        default_prompt = f"""Task/Decision: {task}

Available options:
{"-" * 40}
{"\n\n".join(descriptions)}
{"-" * 40}

{picks_info} options by their exact labels.
List your selections, one per line, followed by your reasoning."""

        result = await self.agent.to_structured(LLMMultiPick).run(
            prompt or default_prompt
        )

        # Validate selections
        invalid = [s for s in result.content.selections if s not in label_map]
        if invalid:
            msg = f"Invalid selections: {', '.join(invalid)}"
            raise ValueError(msg)
        num_picks = len(result.content.selections)
        if num_picks < min_picks:
            msg = f"Too few selections: got {num_picks}, need {min_picks}"
            raise ValueError(msg)

        if max_picks and num_picks > max_picks:
            msg = f"Too many selections: got {num_picks}, max {max_picks}"
            raise ValueError(msg)

        selected = [cast(T, label_map[label]) for label in result.content.selections]
        return MultiPick(selections=selected, reason=result.content.reason)

    async def extract[T](
        self,
        text: str,
        as_type: type[T],
        *,
        mode: ExtractionMode = "structured",
        prompt: AnyPromptType | None = None,
        include_tools: bool = False,
    ) -> T:
        """Extract single instance of type from text.

        Args:
            text: Text to extract from
            as_type: Type to extract
            mode: Extraction approach:
                - "structured": Use Pydantic models (more robust)
                - "tool_calls": Use tool calls (more flexible)
            prompt: Optional custom prompt
            include_tools: Whether to include other tools (tool_calls mode only)
        """
        # Create model for single instance
        item_model = Schema.for_class_ctor(as_type)

        # Create extraction prompt
        final_prompt = prompt or f"Extract {as_type.__name__} from: {text}"
        schema_obj = create_constructor_schema(as_type)
        schema = schema_obj.model_dump_openai()["function"]

        if mode == "structured":

            class Extraction(Schema):
                instance: item_model  # type: ignore
                # explanation: str | None = None

            result = await self.agent.to_structured(Extraction).run(final_prompt)

            # Convert model instance to actual type
            return as_type(**result.content.instance.model_dump())  # type: ignore

        # Legacy tool-calls approach

        async def construct(**kwargs: Any) -> T:
            """Construct instance from extracted data."""
            return as_type(**kwargs)

        structured = self.agent.to_structured(item_model)
        tool = Tool.from_callable(
            construct,
            name_override=schema["name"],
            description_override=schema["description"],
            # schema_override=schema,
        )
        with structured.tools.temporary_tools(tool, exclusive=not include_tools):
            result = await structured.run(final_prompt)  # type: ignore
        return result.content  # type: ignore

    async def extract_multiple[T](
        self,
        text: str,
        as_type: type[T],
        *,
        mode: ExtractionMode = "structured",
        min_items: int = 1,
        max_items: int | None = None,
        prompt: AnyPromptType | None = None,
        include_tools: bool = False,
    ) -> list[T]:
        """Extract multiple instances of type from text.

        Args:
            text: Text to extract from
            as_type: Type to extract
            mode: Extraction approach:
                - "structured": Use Pydantic models (more robust)
                - "tool_calls": Use tool calls (more flexible)
            min_items: Minimum number of instances to extract
            max_items: Maximum number of instances (None=unlimited)
            prompt: Optional custom prompt
            include_tools: Whether to include other tools (tool_calls mode only)
        """
        item_model = Schema.for_class_ctor(as_type)

        instances: list[T] = []
        schema_obj = create_constructor_schema(as_type)
        final_prompt = prompt or "\n".join([
            f"Extract {as_type.__name__} instances from text.",
            # "Requirements:",
            # f"- Extract at least {min_items} instances",
            # f"- Extract at most {max_items} instances" if max_items else "",
            "\nText to analyze:",
            text,
        ])
        if mode == "structured":
            # Create model for individual instance

            class Extraction(Schema):
                instances: list[item_model]  # type: ignore
                # explanation: str | None = None

            result = await self.agent.to_structured(Extraction).run(final_prompt)

            # Validate counts
            num_instances = len(result.content.instances)
            if len(result.content.instances) < min_items:
                msg = f"Found only {num_instances} instances, need {min_items}"
                raise ValueError(msg)

            if max_items and num_instances > max_items:
                msg = f"Found {num_instances} instances, max is {max_items}"
                raise ValueError(msg)

            # Convert model instances to actual type
            return [
                as_type(
                    **instance.data  # type: ignore
                    if hasattr(instance, "data")
                    else instance.model_dump()  # type: ignore
                )
                for instance in result.content.instances
            ]

        # Legacy tool-calls approach

        async def add_instance(**kwargs: Any) -> str:
            """Add an extracted instance."""
            if max_items and len(instances) >= max_items:
                msg = f"Maximum number of items ({max_items}) reached"
                raise ValueError(msg)
            instance = as_type(**kwargs)
            instances.append(instance)
            return f"Added {instance}"

        add_instance.__annotations__ = schema_obj.get_annotations()
        add_instance.__signature__ = schema_obj.to_python_signature()  # type: ignore
        structured = self.agent.to_structured(item_model)
        with structured.tools.temporary_tools(add_instance, exclusive=not include_tools):
            # Create extraction prompt
            await structured.run(final_prompt)

        if len(instances) < min_items:
            msg = f"Found only {len(instances)} instances, need at least {min_items}"
            raise ValueError(msg)

        return instances
