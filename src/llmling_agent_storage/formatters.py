"""Formatting utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeGuard, assert_never

from rich.console import Console
from rich.markdown import Markdown


if TYPE_CHECKING:
    from llmling_agent_storage.models import ConversationData


def is_conversation_data(data: Any) -> TypeGuard[ConversationData]:
    """Type guard for ConversationData."""
    return (
        isinstance(data, dict)
        and "id" in data
        and "messages" in data
        and "agent" in data
        and "start_time" in data
    )


def format_output(
    data: ConversationData | list[ConversationData] | dict[str, Any],
    output_format: Literal["json", "yaml", "text"] = "text",
) -> str:
    """Format data for output in specified format.

    Args:
        data: Data to format
        output_format: Format to use (text/json/yaml)
    """
    import anyenv

    match output_format:
        case "json":
            return anyenv.dump_json(data, indent=True)
        case "yaml":
            import yamling

            return yamling.dump_yaml(data)
        case "text":
            console = Console(record=True)
            if is_conversation_data(data):
                # Single conversation
                _print_conversation(console, data)  # ty: ignore
            elif isinstance(data, list):
                # Multiple conversations
                for conv in data:
                    if is_conversation_data(conv):
                        _print_conversation(console, conv)
                        console.print()
            else:
                # At this point, data must be a stats dict
                stats_data: dict[str, Any] = data  # type: ignore
                _print_stats(console, stats_data)
            return console.export_text()
        case _ as unreachable:
            assert_never(unreachable)


def _print_conversation(console: Console, conv: ConversationData) -> None:
    """Print a conversation in text format."""
    console.print(f"\n[bold blue]Conversation {conv['id']}[/]")
    console.print(f"Agent: {conv['agent']}, Started: {conv['start_time']}\n")

    if token_usage := conv.get("token_usage"):
        console.print(
            "[dim]"
            f"Tokens: {token_usage['total']:,} total "
            f"({token_usage['prompt']:,} prompt, "
            f"{token_usage['completion']:,} completion)"
            "[/]"
        )
        console.print()

    for msg in conv["messages"]:
        role_color = "green" if msg["role"] == "assistant" else "yellow"
        text = f"[{role_color}]{msg['role'].title()}:[/] ({msg['timestamp']})"
        console.print(text)
        console.print(Markdown(msg["content"]))
        if msg.get("model"):
            console.print(f"[dim]Model: {msg['model']}[/]", highlight=False)
        console.print()


def _print_stats(console: Console, stats: dict[str, Any]) -> None:
    """Print statistics in text format."""
    if "period" in stats:
        console.print(f"\n[bold]Usage Statistics ({stats['period']})[/]")
        console.print(f"Grouped by: {stats.get('group_by', 'unknown')}\n")

    for entry in stats.get("entries", [stats]):
        console.print(f"[blue]{entry['name']}[/]")
        console.print(f"  Messages: {entry['messages']}")
        console.print(f"  Total tokens: {entry['total_tokens']:,}")
        if "models" in entry:
            console.print("  Models: " + ", ".join(entry["models"]))
        console.print()


def format_stats(
    stats: dict[str, dict[str, Any]],
    period: str,
    group_by: str,
) -> dict[str, Any]:
    """Format statistics for output.

    Args:
        stats: Raw statistics data
        period: Time period string (e.g. "1d")
        group_by: Grouping criterion used

    Returns:
        Formatted statistics ready for display
    """
    entries = [
        {
            "name": key,
            "messages": data["messages"],
            "total_tokens": data["total_tokens"],
            "models": sorted(data["models"]),
        }
        for key, data in stats.items()
    ]
    return {"period": period, "group_by": group_by, "entries": entries}
