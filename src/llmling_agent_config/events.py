"""Event sources for LLMling agent."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field, SecretStr
from schemez import Schema


if TYPE_CHECKING:
    from llmling_agent.messaging.events import ConnectionEventData


ConnectionEventType = Literal[
    "message_received",
    "message_processed",
    "message_forwarded",
    "queue_filled",
    "queue_triggered",
]

DEFAULT_TEMPLATE = """
{%- if include_timestamp %}at {{ timestamp }}{% endif %}
Event from {{ source }}:
{%- if include_metadata %}
Metadata:
{% for key, value in metadata.items() %}
{{ key }}: {{ value }}
{% endfor %}
{% endif %}
{{ content }}
"""


class EventSourceConfig(Schema):
    """Base configuration for event sources."""

    type: str = Field(init=False, title="Event source type")
    """Discriminator field for event source types."""

    name: str = Field(
        examples=["file_watcher", "api_webhook", "scheduler"],
        title="Event source name",
    )
    """Unique identifier for this event source."""

    enabled: bool = Field(default=True, title="Source enabled")
    """Whether this event source is active."""

    template: str = Field(
        default=DEFAULT_TEMPLATE,
        examples=[DEFAULT_TEMPLATE, "New event: {{ content }}", "{{ timestamp }}: {{ source }}"],
        title="Event template",
    )
    """Jinja2 template for formatting events."""

    include_metadata: bool = Field(default=True, title="Include metadata")
    """Control metadata visibility in template."""

    include_timestamp: bool = Field(default=True, title="Include timestamp")
    """Control timestamp visibility in template."""

    model_config = ConfigDict(frozen=True)


class FileWatchConfig(EventSourceConfig):
    """File watching event source."""

    type: Literal["file"] = Field("file", init=False)
    """File / folder content change events."""

    paths: list[str] = Field(
        examples=[["./src", "./docs"], ["/home/user/projects"], ["*.py", "config/*.yaml"]],
        title="Watch paths",
    )
    """Paths or patterns to watch for changes."""

    extensions: list[str] | None = Field(
        default=None,
        examples=[[".py", ".md"], [".js", ".ts", ".json"], [".yaml", ".yml"]],
        title="File extensions",
    )
    """File extensions to monitor (e.g. ['.py', '.md'])."""

    ignore_paths: list[str] | None = Field(
        default=None,
        examples=[["__pycache__", ".git"], ["node_modules"], ["*.tmp"]],
        title="Ignore patterns",
    )
    """Paths or patterns to ignore."""

    recursive: bool = Field(default=True, title="Recursive watching")
    """Whether to watch subdirectories."""

    debounce: int = Field(
        default=1600,
        gt=0,
        examples=[500, 1600, 3000],
        title="Debounce delay (ms)",
    )
    """Minimum time (ms) between trigger events."""


class WebhookConfig(EventSourceConfig):
    """Webhook event source."""

    type: Literal["webhook"] = Field("webhook", init=False)
    """webhook-based event."""

    port: int = Field(examples=[8080, 9000, 3000], title="Listen port", ge=1, lt=65536)
    """Port to listen on."""

    path: str = Field(
        examples=["/webhook", "/api/events", "/github-webhook"],
        title="URL path",
    )
    """URL path to handle requests."""

    secret: str | None = Field(
        default=None,
        examples=["webhook_secret_123", "github_webhook_key"],
        title="Validation secret",
    )
    """Optional secret for request validation."""


class TimeEventConfig(EventSourceConfig):
    """Time-based event source configuration."""

    type: Literal["time"] = Field("time", init=False)
    """Time event."""

    schedule: str = Field(
        examples=["0 9 * * 1-5", "*/30 * * * *", "0 0 * * 0"],
        title="Cron schedule",
    )
    """Cron expression for scheduling (e.g. '0 9 * * 1-5' for weekdays at 9am)"""

    prompt: str = Field(
        examples=["Daily status check", "Generate weekly report", "Check system health"],
        title="Trigger prompt",
    )
    """Prompt to send to the agent when the schedule triggers."""

    timezone: str | None = Field(
        default=None,
        examples=["UTC", "America/New_York", "Europe/London"],
        title="Schedule timezone",
    )
    """Timezone for schedule (defaults to system timezone)"""

    skip_missed: bool = Field(default=False, title="Skip missed executions")
    """Whether to skip executions missed while agent was inactive"""


class EmailConfig(EventSourceConfig):
    """Email event source configuration.

    Monitors an email inbox for new messages and converts them to events.
    """

    type: Literal["email"] = Field("email", init=False)
    """Email event."""

    host: str = Field(
        description="IMAP server hostname",
        examples=["imap.gmail.com", "outlook.office365.com", "imap.example.com"],
        title="IMAP server host",
    )
    """IMAP server hostname (e.g. 'imap.gmail.com')"""

    port: int = Field(
        default=993, ge=1, lt=65536, examples=[993, 143, 465], title="IMAP server port"
    )
    """Server port (defaults to 993 for IMAP SSL)"""

    username: str = Field(examples=["user@gmail.com", "admin@company.com"], title="Email username")
    """Email account username/address"""

    password: SecretStr = Field(title="Email password")
    """Account password or app-specific password"""

    folder: str = Field(default="INBOX", examples=["INBOX", "Sent", "Drafts"], title="Email folder")
    """Folder/mailbox to monitor"""

    ssl: bool = Field(default=True, title="Use SSL/TLS")
    """Whether to use SSL/TLS connection"""

    check_interval: int = Field(
        default=60,
        gt=0,
        description="Seconds between inbox checks",
        examples=[30, 60, 300],
        title="Check interval",
    )
    """How often to check for new emails (in seconds)"""

    mark_seen: bool = Field(default=True, title="Mark emails as seen")
    """Whether to mark processed emails as seen"""

    filters: dict[str, str] = Field(
        default_factory=dict,
        description="Email filtering criteria",
        title="Email filters",
    )
    """Filtering rules for emails (subject, from, etc)"""

    max_size: int | None = Field(
        default=None,
        description="Maximum email size in bytes",
        examples=[1048576, 5242880, 10485760],
        title="Maximum email size",
    )
    """Size limit for processed emails"""


class ConnectionTriggerConfig(EventSourceConfig):
    """Trigger config specifically for connection events."""

    type: Literal["connection"] = Field("connection", init=False)
    """Connection event trigger."""

    source: str | None = Field(
        default=None,
        examples=["main_agent", "web_scraper", "data_processor"],
        title="Connection source",
    )
    """Connection source name."""

    target: str | None = Field(
        default=None,
        examples=["output_handler", "notification_agent"],
        title="Connection target",
    )
    """Connection to trigger."""

    event: ConnectionEventType = Field(
        examples=["message_received", "message_processed", "queue_filled"],
        title="Event type",
    )
    """Event type to trigger on."""

    condition: ConnectionEventConditionType | None = Field(default=None, title="Event condition")
    """Condition-based filter for the event."""

    async def matches_event(self, event: ConnectionEventData[Any]) -> bool:
        """Check if this trigger matches the event."""
        # First check event type
        if event.event_type != self.event:
            return False

        # Check source/target filters
        if self.source and event.connection.source.name != self.source:
            return False
        if self.target and not any(t.name == self.target for t in event.connection.targets):
            return False

        # Check condition if any
        if self.condition:
            return await self.condition.check(event)

        return True


EventConfig = Annotated[
    FileWatchConfig | WebhookConfig | EmailConfig | TimeEventConfig | ConnectionTriggerConfig,
    Field(discriminator="type"),
]


class ConnectionEventCondition(Schema):
    """Base conditions specifically for connection events."""

    type: str = Field(
        init=False,
        title="Condition type",
    )

    async def check(self, event: ConnectionEventData[Any]) -> bool:
        raise NotImplementedError


class ConnectionContentCondition(ConnectionEventCondition):
    """Simple content matching for connection events."""

    type: Literal["content"] = Field("content", init=False)
    """Content-based trigger."""

    words: list[str] = Field(
        examples=[["error", "warning"], ["complete", "finished"], ["urgent"]],
        title="Match words",
    )
    """List of words to match."""

    mode: Literal["any", "all"] = Field(
        default="any",
        examples=["any", "all"],
        title="Matching mode",
    )
    """Matching mode."""

    async def check(self, event: ConnectionEventData[Any]) -> bool:
        if not event.message:
            return False
        text = str(event.message.content)
        return any(word in text for word in self.words)


class ConnectionJinja2Condition(ConnectionEventCondition):
    """Flexible Jinja2 condition for connection events."""

    type: Literal["jinja2"] = Field("jinja2", init=False)
    """Jinja2-based trigger configuration."""

    template: str = Field(
        examples=[
            "{{ event.message.content | length > 100 }}",
            "{{ 'urgent' in event.message.content }}",
        ],
        title="Jinja2 condition template",
    )
    """Jinja2-Template (needs to return a "boolean" string)."""

    async def check(self, event: ConnectionEventData[Any]) -> bool:
        from jinjarope import Environment

        env = Environment(enable_async=True)
        template = env.from_string(self.template)
        result = await template.render_async(event=event)
        return result.strip().lower() == "true"


ConnectionEventConditionType = Annotated[
    ConnectionContentCondition | ConnectionJinja2Condition,
    Field(discriminator="type"),
]
