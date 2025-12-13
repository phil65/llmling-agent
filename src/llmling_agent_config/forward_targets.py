"""Forward target models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema
from upathtools import to_upath

from llmling_agent_config.conditions import Condition


if TYPE_CHECKING:
    from pydantic_ai.models.function import FunctionModel
    from upathtools import UPath

    from llmling_agent.messaging import ChatMessage


ConnectionType = Literal["run", "context", "forward"]


class ConnectionConfig(Schema):
    """Base model for message forwarding targets."""

    model_config = ConfigDict(json_schema_extra={"title": "Connection Configuration"})

    type: str = Field(init=False)
    """Connection type."""

    wait_for_completion: bool = Field(default=True, title="Wait for completion")
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """

    queued: bool = Field(default=False, title="Enable message queueing")
    """Whether messages should be queued for manual processing."""

    queue_strategy: Literal["concat", "latest", "buffer"] = Field(
        default="latest",
        examples=["concat", "latest", "buffer"],
        title="Queue processing strategy",
    )
    """How to process queued messages."""

    priority: int = Field(default=0, examples=[0, 1, 5], title="Task priority")
    """Priority of the task. Lower = higher priority."""

    delay: timedelta | None = Field(default=None, title="Processing delay")
    """Delay before processing."""
    filter_condition: Condition | None = Field(
        default=None,
        title="Message filter condition",
        examples=[{"type": "word_match", "words": ["bad word"]}],
    )
    """When to filter messages (using Talk.when())."""

    stop_condition: Condition | None = Field(
        default=None,
        title="Connection stop condition",
        examples=[{"type": "cost", "max_cost": 0.1}],
    )
    """When to disconnect the connection."""

    exit_condition: Condition | None = Field(
        default=None,
        title="Application exit condition",
        examples=[{"type": "cost", "max_cost": 0.2}],
    )
    """When to exit the application (by raising SystemExit)."""

    transform: ImportString[Callable[[Any], Any | Awaitable[Any]]] | None = Field(
        default=None,
        examples=["mymodule.transform_message", "utils.filters:clean_content"],
        title="Message transform function",
    )
    """Optional function to transform messages before forwarding."""


class NodeConnectionConfig(ConnectionConfig):
    """Forward messages to another node.

    This configuration defines how messages should flow from one node to another,
    including:
    - Basic routing (which node, what type of connection)
    - Message queueing and processing strategies
    - Timing controls (priority, delay)
    - Execution behavior (wait for completion)
    """

    model_config = ConfigDict(json_schema_extra={"title": "Node Connection Configuration"})

    type: Literal["node"] = Field("node", init=False)
    """Connection to another node."""

    name: str = Field(
        examples=["output_agent", "processor", "notification_handler"],
        title="Target node name",
    )
    """Name of target agent."""

    connection_type: ConnectionType = Field(
        default="run",
        examples=["run", "context", "forward"],
        title="Connection type",
    )
    """How messages should be handled by the target agent:
    - run: Execute message as a new run
    - context: Add message to agent's context
    - forward: Forward message to agent's outbox
    """


DEFAULT_MESSAGE_TEMPLATE = """
[{{ message.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ message.name }}: {{ message.content }}
{%- if message.forwarded_from %}
(via: {{ message.forwarded_from|join(' -> ') }})
{%- endif %}
"""


class FileConnectionConfig(ConnectionConfig):
    """Write messages to a file using a template.

    The template receives the full message object for formatting.
    Available fields include:
    - timestamp: When the message was created
    - name: Name of the sender
    - content: Message content
    - role: Message role (user/assistant/system)
    - model: Model used (if any)
    - cost_info: Token usage and cost info
    - forwarded_from: Chain of message forwarding
    """

    model_config = ConfigDict(json_schema_extra={"title": "File Connection Configuration"})

    type: Literal["file"] = Field("file", init=False)
    """Connection to a file."""

    connection_type: Literal["run"] = Field("run", init=False, exclude=True)
    """Connection type (fixed to "run")"""

    path: str = Field(
        examples=["logs/messages.txt", "/var/log/agent-{date}.log", "output/{agent}.md"],
        title="Output file path",
    )
    """Path to output file. Supports variables: {date}, {time}, {agent}"""

    template: str = Field(
        default=DEFAULT_MESSAGE_TEMPLATE,
        examples=[
            DEFAULT_MESSAGE_TEMPLATE,
            "{{ message.content }}",
            "[{{ message.name }}]: {{ message.content }}",
        ],
        title="Message template",
    )
    """Jinja2 template for message formatting."""

    encoding: str = Field(
        default="utf-8",
        examples=["utf-8", "ascii", "latin1"],
        title="File encoding",
    )
    """File encoding to use."""

    def format_message(self, message: ChatMessage[Any]) -> str:
        """Format a message using the template."""
        from jinja2 import Template

        template = Template(self.template)
        return template.render(message=message)

    def resolve_path(self, context: dict[str, str]) -> UPath:
        """Resolve path template with context variables."""
        from llmling_agent.utils.now import get_now

        now = get_now()
        date = now.strftime("%Y-%m-%d")
        time_ = now.strftime("%H-%M-%S")
        variables = {"date": date, "time": time_, **context}
        return to_upath(self.path.format(**variables))

    def get_model(self) -> FunctionModel:
        """Get provider for file writing."""
        from jinja2 import Template
        from llmling_models import function_to_model

        path_obj = to_upath(self.path)
        template_obj = Template(self.template)

        async def write_message(message: str) -> str:
            formatted = template_obj.render(message=message)
            path_obj.write_text(formatted + "\n", encoding=self.encoding)
            return ""

        return function_to_model(write_message)


class CallableConnectionConfig(ConnectionConfig):
    """Forward messages to a callable.

    The callable can be either sync or async and should have the signature:
    def process_message(message: ChatMessage[Any], **kwargs) -> Any

    Any additional kwargs specified in the config will be passed to the callable.
    """

    model_config = ConfigDict(json_schema_extra={"title": "Callable Connection Configuration"})

    type: Literal["callable"] = Field("callable", init=False)
    """Connection to a callable imported from given import path."""

    callable: ImportString[Callable[..., Any]] = Field(
        examples=["mymodule.process_message", "handlers.notifications:send_email"],
        title="Callable import path",
    )
    """Import path to the message processing function."""

    connection_type: Literal["run"] = Field("run", init=False, exclude=True)
    """Connection type (fixed to "run")"""

    kw_args: dict[str, Any] = Field(default_factory=dict, title="Additional arguments")
    """Additional kwargs to pass to the callable."""

    async def process_message(self, message: ChatMessage[Any]) -> Any:
        """Process a message through the callable.

        Handles both sync and async callables transparently.
        """
        from llmling_agent.utils.inspection import execute

        return await execute(self.callable, message, **self.kw_args)

    def get_model(self) -> FunctionModel:
        """Get provider for callable."""
        from llmling_models import function_to_model

        return function_to_model(self.callable)


ForwardingTarget = Annotated[
    NodeConnectionConfig | FileConnectionConfig | CallableConnectionConfig,
    Field(discriminator="type"),
]
