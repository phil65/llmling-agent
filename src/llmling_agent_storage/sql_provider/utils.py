"""Utilities for database storage."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic_ai import RunUsage
from sqlalchemy import JSON, Column, and_, or_
from sqlalchemy.sql import expression
from sqlmodel import select

from llmling_agent.messaging import ChatMessage, TokenCost
from llmling_agent.storage import deserialize_messages
from llmling_agent_storage.models import ConversationData, MessageData
from llmling_agent_storage.sql_provider.models import Conversation


if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlmodel.sql.expression import SelectOfScalar
    from tokonomics.toko_types import TokenUsage

    from llmling_agent_config.session import SessionQuery
    from llmling_agent_storage.sql_provider.models import Message


def aggregate_token_usage(
    messages: Sequence[Message | ChatMessage[str]],
) -> TokenUsage:
    """Sum up tokens from a sequence of messages."""
    from llmling_agent_storage.sql_provider.models import Message

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
        conversation_id=db_message.conversation_id,
        content=db_message.content,
        role=db_message.role,  # type: ignore
        name=db_message.name,
        model_name=db_message.model,
        cost_info=cost_info,
        response_time=db_message.response_time,
        forwarded_from=db_message.forwarded_from or [],
        timestamp=db_message.timestamp,
        provider_name=db_message.provider_name,
        provider_response_id=db_message.provider_response_id,
        messages=deserialize_messages(db_message.messages),
        finish_reason=db_message.finish_reason,  # type: ignore
    )


def get_column_default(column: Any) -> str:
    """Get SQL DEFAULT clause for column."""
    if column.default is None:
        return ""
    if hasattr(column.default, "arg"):
        # Simple default value
        return f" DEFAULT {column.default.arg}"
    if hasattr(column.default, "sqltext"):
        # Computed default
        return f" DEFAULT {column.default.sqltext}"
    return ""


def auto_migrate_columns(sync_conn: Any, dialect: Any) -> None:
    """Automatically add missing columns to existing tables.

    Args:
        sync_conn: Synchronous database connection
        dialect: SQLAlchemy dialect for SQL type compilation
    """
    from sqlalchemy import inspect
    from sqlalchemy.sql import text
    from sqlmodel import SQLModel

    inspector = inspect(sync_conn)

    # For each table in our models
    for table_name, table in SQLModel.metadata.tables.items():
        existing = {col["name"] for col in inspector.get_columns(table_name)}

        # For each column in model that doesn't exist in DB
        for col in table.columns:
            if col.name not in existing:
                # Create ALTER TABLE statement based on column type
                type_sql = col.type.compile(dialect)
                nullable = "" if col.nullable else " NOT NULL"
                default = get_column_default(col)
                sql = (
                    f"ALTER TABLE {table_name} ADD COLUMN {col.name} {type_sql}{nullable}{default}"
                )
                sync_conn.execute(text(sql))


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
    from llmling_agent_storage.sql_provider.models import Message

    stmt = select(Message).order_by(Message.timestamp)  # type: ignore

    conditions: list[Any] = []
    if query.name:
        conditions.append(Message.conversation_id == query.name)
    if query.agents:
        agent_conditions = [Column("name").in_(query.agents)]
        if query.include_forwarded:
            agent_conditions.append(
                and_(
                    Column("forwarded_from").isnot(None),
                    expression.cast(Column("forwarded_from"), JSON).contains(list(query.agents)),  # type: ignore
                )
            )
        conditions.append(or_(*agent_conditions))
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
                )
                for msg in chat_messages
            ],
            token_usage=aggregate_token_usage(msgs) if include_tokens else None,
        )

    # If it's already ConversationData, update token_usage if needed
    if include_tokens and conv["token_usage"] is None:
        conv["token_usage"] = aggregate_token_usage(msgs)
    return conv
