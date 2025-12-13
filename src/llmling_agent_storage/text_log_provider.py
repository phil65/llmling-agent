"""Text-based storage provider with dynamic paths."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from upathtools import to_upath

from llmling_agent.log import get_logger
from llmling_agent.utils.now import get_now
from llmling_agent_storage.base import StorageProvider


if TYPE_CHECKING:
    from datetime import datetime

    from jinja2 import Template
    from upathtools import JoinablePathLike, UPath

    from llmling_agent.common_types import JsonValue
    from llmling_agent_config.storage import LogFormat, TextLogConfig


logger = get_logger(__name__)


CONVERSATIONS_TEMPLATE = """\
=== LLMling Agent Log ===

{%- for conv_id, conv in conversations.items() %}
=== Conversation {{ conv_id }} (agent: {{ conv.agent_name }}, started: {{ conv.start_time.strftime('%Y-%m-%d %H:%M:%S') }}) ===

{%- for msg in messages if msg.conversation_id == conv_id %}
[{{ msg.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ msg.sender }}{% if msg.model %} ({{ msg.model }}){% endif %}: {{ msg.content }}
{%- if msg.cost_info %}
Tokens: {{ msg.cost_info.token_usage.total }} (prompt: {{ msg.cost_info.token_usage.prompt }}, completion: {{ msg.cost_info.token_usage.completion }})
Cost: ${{ "%.4f"|format(msg.cost_info.total_cost) }}
{%- endif %}
{%- if msg.response_time %}
Response time: {{ "%.1f"|format(msg.response_time) }}s
{%- endif %}
{%- if msg.forwarded_from %}
Forwarded via: {{ msg.forwarded_from|join(' -> ') }}
{%- endif %}

{%- for tool in tool_calls if tool.message_id == msg.id %}
Tool Call: {{ tool.tool_name }}
Args: {{ tool.args|pprint }}
Result: {{ tool.result }}
{%- endfor %}
{%- endfor %}
{%- endfor %}

=== Commands ===
{%- for cmd in commands %}
[{{ cmd.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ cmd.agent_name }} ({{ cmd.session_id }}): {{ cmd.command }}
{%- endfor %}
"""  # noqa: E501

CHRONOLOGICAL_TEMPLATE = """\
=== LLMling Agent Log ===

{%- for entry in entries|sort(attribute="timestamp") %}
{%- if entry.type == "conversation_start" %}
=== Conversation {{ entry.conversation_id }} (agent: {{ entry.agent_name }}) started ===

{%- elif entry.type == "message" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] {{ entry.sender }}{% if entry.model %} ({{ entry.model }}){% endif %}: {{ entry.content }}
{%- if entry.cost_info %}
Tokens: {{ entry.cost_info.token_usage.total }} (prompt: {{ entry.cost_info.token_usage.prompt }}, completion: {{ entry.cost_info.token_usage.completion }})
Cost: ${{ "%.4f"|format(entry.cost_info.total_cost) }}
{%- endif %}
{%- if entry.response_time %}
Response time: {{ "%.1f"|format(entry.response_time) }}s
{%- endif %}
{%- if entry.forwarded_from %}
Forwarded via: {{ entry.forwarded_from|join(' -> ') }}
{%- endif %}

{%- elif entry.type == "tool_call" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] Tool Call: {{ entry.tool_name }}
Args: {{ entry.args|pprint }}
Result: {{ entry.result }}

{%- elif entry.type == "command" %}
[{{ entry.timestamp.strftime('%Y-%m-%d %H:%M:%S') }}] Command by {{ entry.agent_name }}: {{ entry.command }}

{%- endif %}
{%- endfor %}
"""  # noqa: E501


class TextLogProvider(StorageProvider):
    """Human-readable text log provider with dynamic paths.

    Available template variables:
    - now: datetime - Current timestamp
    - date: date - Current date
    - operation: str - Type of operation (message/conversation/tool_call/command)
    - conversation_id: str - ID of current conversation
    - agent_name: str - Name of the agent
    - content: str - Message content
    - role: str - Message role
    - model: str - Model name
    - session_id: str - Session ID
    - tool_name: str - Name of tool being called
    - command: str - Command being executed

    All variables default to empty string if not available for current operation.
    """

    TEMPLATES: ClassVar[dict[LogFormat, str]] = {
        "chronological": CHRONOLOGICAL_TEMPLATE,
        "conversations": CONVERSATIONS_TEMPLATE,
    }
    can_load_history = False

    def __init__(self, config: TextLogConfig) -> None:
        """Initialize text log provider."""
        from jinja2 import Environment, Undefined

        class EmptyStringUndefined(Undefined):
            """Return empty string for undefined variables."""

            def __str__(self) -> str:
                return ""

        super().__init__(config)
        self.encoding = config.encoding
        self.content_template = self._load_template(config.template)
        # Configure Jinja env with empty string for undefined
        env = Environment(undefined=EmptyStringUndefined, enable_async=True)
        self.path_template = env.from_string(config.path)

        self._entries: list[dict[str, Any]] = []

    def _load_template(
        self,
        template: LogFormat | JoinablePathLike | None,
    ) -> Template:
        """Load template from predefined or file."""
        from jinja2 import Template

        if template is None:
            template_str = self.TEMPLATES["chronological"]
        elif template in self.TEMPLATES:
            template_str = self.TEMPLATES[template]  # type: ignore
        else:  # Assume it's a path
            template_str = to_upath(template).read_text()
        return Template(template_str)

    def _get_base_context(self, operation: str) -> dict[str, Any]:
        """Get base context with defaults.

        Args:
            operation: Type of operation being logged

        Returns:
            Base context dict with defaults
        """
        # All other variables will default to empty string via EmptyStringUndefined
        return {"now": get_now(), "date": get_now().date(), "operation": operation}

    async def _get_path(self, operation: str, **context: Any) -> UPath:
        """Render path template with context.

        Args:
            operation: Type of operation being logged
            **context: Additional context variables

        Returns:
            Concrete path for current operation
        """
        # Combine base context with provided values
        path_context = self._get_base_context(operation)
        path_context.update(context)

        path = await self.path_template.render_async(**path_context)
        resolved_path = to_upath(path)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        return resolved_path

    async def log_message(
        self,
        *,
        conversation_id: str,
        message_id: str,
        content: str,
        role: str,
        name: str | None = None,
        cost_info: Any | None = None,
        model: str | None = None,
        response_time: float | None = None,
        forwarded_from: list[str] | None = None,
        provider_name: str | None = None,
        provider_response_id: str | None = None,
        messages: str | None = None,
        finish_reason: str | None = None,
    ) -> None:
        """Store message and update log."""
        entry = {
            "type": "message",
            "timestamp": get_now(),
            "conversation_id": conversation_id,
            "message_id": message_id,
            "content": content,
            "role": role,
            "agent_name": name,
            "model": model,
            "cost_info": cost_info,
            "response_time": response_time,
            "forwarded_from": forwarded_from,
            "provider_name": provider_name,
            "provider_response_id": provider_response_id,
            "messages": messages,
            "finish_reason": finish_reason,
        }
        self._entries.append(entry)

        path = await self._get_path("message", **entry)
        self._write(path)

    async def log_conversation(
        self,
        *,
        conversation_id: str,
        node_name: str,
        start_time: datetime | None = None,
    ) -> None:
        """Store conversation start."""
        entry = {
            "type": "conversation",
            "timestamp": start_time or get_now(),
            "conversation_id": conversation_id,
            "agent_name": node_name,
        }
        self._entries.append(entry)

        path = await self._get_path("conversation", **entry)
        self._write(path)

    async def log_command(
        self,
        *,
        agent_name: str,
        session_id: str,
        command: str,
        context_type: type | None = None,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        """Store command."""
        entry = {
            "type": "command",
            "timestamp": get_now(),
            "agent_name": agent_name,
            "session_id": session_id,
            "command": command,
            "context_type": context_type.__name__ if context_type else "",
            "metadata": metadata or {},
        }
        self._entries.append(entry)

        path = await self._get_path("command", **entry)
        self._write(path)

    def _write(self, path: UPath) -> None:
        """Write current state to file at given path."""
        context = {"entries": self._entries}
        try:
            text = self.content_template.render(context)
            path.write_text(text, encoding=self.encoding)
        except Exception as e:
            logger.exception("Failed to write to log file", path=path)
            msg = f"Failed to write to log file: {e}"
            raise RuntimeError(msg) from e
