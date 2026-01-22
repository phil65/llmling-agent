"""Utilities for database storage."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunUsage
from sqlalchemy import Column, and_
from sqlmodel import select

from agentpool.messaging import ChatMessage, TokenCost
from agentpool.storage import deserialize_messages
from agentpool_storage.models import ConversationData, MessageData
from agentpool_storage.sql_provider.models import Conversation


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlmodel.sql.expression import SelectOfScalar
    from tokonomics.toko_types import TokenUsage

    from agentpool_config.session import SessionQuery
    from agentpool_storage.sql_provider.models import Message


# Track migrated databases to skip redundant migration attempts
_migrated_databases: set[str] = set()


def aggregate_token_usage(
    messages: Sequence[Message | ChatMessage[str]],
) -> TokenUsage:
    """Sum up tokens from a sequence of messages."""
    from agentpool_storage.sql_provider.models import Message

    total = prompt = completion = 0
    for msg in messages:
        if isinstance(msg, Message):
            total += msg.total_tokens or 0
            prompt += msg.input_tokens or 0
            completion += msg.output_tokens or 0
        elif msg.cost_info:
            total += msg.cost_info.token_usage.total_tokens
            prompt += msg.cost_info.token_usage.input_tokens
            completion += msg.cost_info.token_usage.output_tokens
    return {"total": total, "prompt": prompt, "completion": completion}


def to_chat_message(db_message: Message) -> ChatMessage[str]:
    """Convert database message to ChatMessage."""
    cost_info = None
    if db_message.total_tokens is not None:
        cost_info = TokenCost(
            token_usage=RunUsage(
                input_tokens=db_message.input_tokens or 0,
                output_tokens=db_message.output_tokens or 0,
            ),
            total_cost=Decimal(db_message.cost or 0.0),
        )

    return ChatMessage[str](
        message_id=db_message.id,
        session_id=db_message.session_id,
        content=db_message.content,
        role=db_message.role,  # type: ignore
        name=db_message.name,
        model_name=db_message.model,
        cost_info=cost_info,
        response_time=db_message.response_time,
        timestamp=db_message.timestamp,
        provider_name=db_message.provider_name,
        provider_response_id=db_message.provider_response_id,
        messages=deserialize_messages(db_message.messages),
        finish_reason=db_message.finish_reason,  # type: ignore
    )


def run_alembic_migrations(connection: Any) -> None:
    """Run Alembic migrations using an existing database connection.

    This function is idempotent. It tracks which databases have been migrated
    in the current process to avoid redundant migration attempts.

    Args:
        connection: SQLAlchemy connection (sync)
    """
    from pathlib import Path

    from alembic.config import Config
    from alembic.runtime.environment import EnvironmentContext
    from alembic.script import ScriptDirectory
    from sqlmodel import SQLModel

    from agentpool.log import get_logger

    logger = get_logger(__name__)

    # Get database URL for tracking
    db_url = str(connection.engine.url)

    # Check if already migrated in this process (fast path)
    if db_url in _migrated_databases:
        logger.debug("Skipping migrations, already completed for this database")
        return

    # Find the migrations directory relative to this package
    migrations_dir = Path(__file__).parents[3] / "migrations"
    if not migrations_dir.exists():
        logger.warning("Migrations directory not found, skipping Alembic migrations")
        _migrated_databases.add(db_url)
        return

    alembic_cfg = Config()
    alembic_cfg.set_main_option("script_location", str(migrations_dir))
    alembic_cfg.attributes["connection"] = connection

    script = ScriptDirectory.from_config(alembic_cfg)

    def upgrade_fn(rev: str, context: EnvironmentContext) -> list:  # type: ignore[type-arg]
        return script._upgrade_revs("head", rev)

    with EnvironmentContext(
        alembic_cfg,
        script,
        fn=upgrade_fn,
        as_sql=False,
        starting_rev=None,
        destination_rev="head",
    ) as env_context:
        env_context.configure(
            connection=connection,
            target_metadata=SQLModel.metadata,
        )
        with env_context.begin_transaction():
            env_context.run_migrations()

    # Mark this database as migrated
    _migrated_databases.add(db_url)
    logger.debug("Migrations completed for database", db_url=db_url)


def parse_model_info(model: str | None) -> tuple[str | None, str | None]:
    """Parse model string into provider and name.

    Args:
        model: Full model string (e.g., "openai:gpt-5", "anthropic:claude-sonnet-4-0")

    Returns:
        Tuple of (provider, name)
    """
    if not model:
        return None, None

    # Try splitting by ':' or '/'
    parts = model.split(":") if ":" in model else model.split("/")

    if len(parts) == 2:  # noqa: PLR2004
        provider, name = parts
        return provider.lower(), name

    # No provider specified, try to infer
    name = parts[0]
    if name.startswith(("gpt-", "text-", "dall-e")):
        return "openai", name
    if name.startswith("claude"):
        return "anthropic", name
    if name.startswith(("llama", "mistral")):
        return "meta", name

    return None, name


def build_message_query(query: SessionQuery) -> SelectOfScalar[Any]:
    """Build SQLModel query from SessionQuery."""
    from agentpool_storage.sql_provider.models import Message

    stmt = select(Message).order_by(Message.timestamp)  # type: ignore

    conditions: list[Any] = []
    if query.name:
        conditions.append(Message.session_id == query.name)
    if query.agents:
        conditions.append(Column("name").in_(query.agents))
    if query.since and (cutoff := query.get_time_cutoff()):
        conditions.append(Message.timestamp >= cutoff)
    if query.until:
        conditions.append(Message.timestamp <= datetime.fromisoformat(query.until))
    if query.contains:
        conditions.append(Message.content.contains(query.contains))  # type: ignore
    if query.roles:
        conditions.append(Message.role.in_(query.roles))  # type: ignore

    if conditions:
        stmt = stmt.where(and_(*conditions))
    if query.limit:
        stmt = stmt.limit(query.limit)

    return stmt


def format_conversation(
    conv: Conversation | ConversationData,
    messages: Sequence[Message | ChatMessage[str]],
    *,
    include_tokens: bool = False,
    compact: bool = False,
) -> ConversationData:
    """Format SQL conversation model to ConversationData."""
    msgs = list(messages)
    if compact and len(msgs) > 1:
        msgs = [msgs[0], msgs[-1]]

    # Convert messages to ChatMessage format if needed
    chat_messages = [msg if isinstance(msg, ChatMessage) else to_chat_message(msg) for msg in msgs]

    # Convert Conversation to ConversationData format
    if isinstance(conv, Conversation):
        return ConversationData(
            id=conv.id,
            agent=conv.agent_name,
            title=conv.title,
            start_time=conv.start_time.isoformat(),
            messages=[
                MessageData(
                    role=msg.role,
                    content=msg.content,
                    timestamp=msg.timestamp.isoformat(),
                    model=msg.model_name,
                    name=msg.name,
                    token_usage={
                        "prompt": msg.usage.input_tokens,
                        "completion": msg.usage.output_tokens,
                        "total": msg.usage.total_tokens,
                    },
                    cost=float(msg.cost_info.total_cost) if msg.cost_info else None,
                    response_time=msg.response_time,
                    parent_id=msg.parent_id,
                )
                for msg in chat_messages
            ],
            token_usage=aggregate_token_usage(msgs) if include_tokens else None,
        )

    # If it's already ConversationData, update token_usage if needed
    if include_tokens and conv["token_usage"] is None:
        conv["token_usage"] = aggregate_token_usage(msgs)
    return conv
