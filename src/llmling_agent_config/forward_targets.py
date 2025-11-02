"""Forward target models."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, ImportString
from schemez import Schema
from upath import UPath

from llmling_agent_config.conditions import Condition


if TYPE_CHECKING:
    from llmling_agent.messaging.messages import ChatMessage
    from llmling_agent_providers.callback import CallbackProvider


ConnectionType = Literal["run", "context", "forward"]


class ConnectionConfig(Schema):
    """Base model for message forwarding targets."""

    type: str = Field(init=False)
    """Connection type."""

    wait_for_completion: bool = Field(True)
    """Whether to wait for the result before continuing.

    If True, message processing will wait for the target to complete.
    If False, message will be forwarded asynchronously.
    """

    queued: bool = False
    """Whether messages should be queued for manual processing."""

    queue_strategy: Literal["concat", "latest", "buffer"] = "latest"
    """How to process queued messages."""

    priority: int = 0
    """Priority of the task. Lower = higher priority."""

    delay: timedelta | None = None
    """Delay before processing."""

    filter_condition: Condition | None = None
    """When to filter messages (using Talk.when())."""

    stop_condition: Condition | None = None
    """When to disconnect the connection."""

    exit_condition: Condition | None = None
    """When to exit the application (by raising SystemExit)."""

    transform: ImportString[Callable[[Any], Any | Awaitable[Any]]] | None = None
    """Optional function to transform messages before forwarding."""

    model_config = ConfigDict(frozen=True)


class NodeConnectionConfig(ConnectionConfig):
    """Forward messages to another node.

    This configuration defines how messages should flow from one node to another,
    including:
    - Basic routing (which node, what type of connection)
    - Message queueing and processing strategies
    - Timing controls (priority, delay)
    - Execution behavior (wait for completion)
    """

    type: Literal["node"] = Field("node", init=False)
    """Connection to another node."""

    name: str
    """Name of target agent."""

    connection_type: ConnectionType = "run"
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
"""  # noqa: E501


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

    type: Literal["file"] = Field("file", init=False)
    """Connection to a file."""

    connection_type: Literal["run"] = Field("run", init=False, exclude=True)
    """Connection type (fixed to "run")"""

    path: str
    """Path to output file. Supports variables: {date}, {time}, {agent}"""

    template: str = DEFAULT_MESSAGE_TEMPLATE
    """Jinja2 template for message formatting."""

    encoding: str = "utf-8"
    """File encoding to use."""

    def format_message(self, message: ChatMessage[Any]) -> str:
        """Format a message using the template."""
        from jinja2 import Template

        template = Template(self.template)
        return template.render(message=message)

    def resolve_path(self, context: dict[str, str]) -> UPath:
        """Resolve path template with context variables."""
        from upath import UPath

        from llmling_agent.utils.now import get_now

        now = get_now()
        date = now.strftime("%Y-%m-%d")
        time_ = now.strftime("%H-%M-%S")
        variables = {"date": date, "time": time_, **context}
        return UPath(self.path.format(**variables))

    def get_provider(self) -> CallbackProvider:
        """Get provider for file writing."""
        from jinja2 import Template

        from llmling_agent_providers.callback import CallbackProvider

        path_obj = UPath(self.path)
        template_obj = Template(self.template)

        async def write_message(message: str) -> str:
            formatted = template_obj.render(message=message)
            path_obj.write_text(formatted + "\n", encoding=self.encoding)
            return ""

        name = f"file_writer_{path_obj.stem}"
        return CallbackProvider(write_message, name=name)


class CallableConnectionConfig(ConnectionConfig):
    """Forward messages to a callable.

    The callable can be either sync or async and should have the signature:
    def process_message(message: ChatMessage[Any], **kwargs) -> Any

    Any additional kwargs specified in the config will be passed to the callable.
    """

    type: Literal["callable"] = Field("callable", init=False)
    """Connection to a callable imported from given import path."""

    callable: ImportString[Callable[..., Any]]
    """Import path to the message processing function."""

    connection_type: Literal["run"] = Field("run", init=False, exclude=True)
    """Connection type (fixed to "run")"""

    kw_args: dict[str, Any] = Field(default_factory=dict)
    """Additional kwargs to pass to the callable."""

    async def process_message(self, message: ChatMessage[Any]) -> Any:
        """Process a message through the callable.

        Handles both sync and async callables transparently.
        """
        from llmling_agent.utils.inspection import execute

        return await execute(self.callable, message, **self.kw_args)

    def get_provider(self) -> CallbackProvider:
        """Get provider for callable."""
        from llmling_agent_providers.callback import CallbackProvider

        return CallbackProvider(self.callable, name=self.callable.__name__)


ForwardingTarget = Annotated[
    NodeConnectionConfig | FileConnectionConfig | CallableConnectionConfig,
    Field(discriminator="type"),
]
