"""Example usage of the Codex adapter."""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING, Any

from codex_adapter import CodexClient


if TYPE_CHECKING:
    from collections.abc import Callable


async def simple_chat() -> None:
    """Simple single-turn chat example."""
    print("=== Simple Chat Example ===\n")

    async with CodexClient() as client:
        # Start a thread
        response = await client.thread_start(cwd=".")
        thread_id = response.thread.id
        print(f"Started thread: {thread_id}\n")

        # Send a message
        message = "List the Python files in the current directory"
        print(f"> {message}\n")

        async for event in client.turn_stream(thread_id, message):
            # Print agent messages
            if event.event_type == "item/agentMessage/delta":
                print(event.get_text_delta(), end="", flush=True)

            # Print command outputs
            elif event.event_type == "item/commandExecution/outputDelta":
                delta = event.get_text_delta()
                if delta:
                    print(f"\n[Command output]\n{delta}", flush=True)

            # Handle completion
            elif event.event_type == "turn/completed":
                print("\n\n[Turn completed]")
                break

            # Handle errors
            elif event.event_type == "turn/error":
                from codex_adapter.models import TurnErrorData

                if isinstance(event.data, TurnErrorData):
                    print(f"\n\n[Error: {event.data.error}]", file=sys.stderr)
                break


async def multi_turn_chat() -> None:
    """Multi-turn conversation example."""
    print("=== Multi-Turn Chat Example ===\n")

    async with CodexClient() as client:
        response = await client.thread_start(
            cwd=".",
            model="gpt-5-codex",
        )
        thread_id = response.thread.id

        messages = [
            "What is the main purpose of this codebase?",
            "Show me the entry point file",
            "What dependencies does it use?",
        ]

        for i, message in enumerate(messages, 1):
            print(f"\n--- Turn {i} ---")
            print(f"> {message}\n")

            async for event in client.turn_stream(thread_id, message):
                if event.event_type == "item/agentMessage/delta":
                    print(event.get_text_delta(), end="", flush=True)
                elif event.event_type == "turn/completed":
                    print("\n")
                    break


async def model_override_example() -> None:
    """Example showing per-turn model override."""
    print("=== Model Override Example ===\n")

    async with CodexClient() as client:
        response = await client.thread_start(model="gpt-5-codex")
        thread_id = response.thread.id

        # First turn with default model
        print("Turn 1 (default model: gpt-5-codex)")
        print("> Write a hello world function\n")

        async for event in client.turn_stream(thread_id, "Write a hello world function"):
            if event.event_type == "item/agentMessage/delta":
                print(event.get_text_delta(), end="", flush=True)
            elif event.event_type == "turn/completed":
                print("\n")
                break

        # Second turn with different model
        print("\nTurn 2 (override to claude-opus-4, high effort)")
        print("> Now make it more elegant\n")

        async for event in client.turn_stream(
            thread_id,
            "Now make it more elegant",
            model="claude-opus-4",
            effort="high",
        ):
            if event.event_type == "item/agentMessage/delta":
                print(event.get_text_delta(), end="", flush=True)
            elif event.event_type == "turn/completed":
                print("\n")
                break


async def event_inspection_example() -> None:
    """Example showing detailed event inspection."""
    print("=== Event Inspection Example ===\n")

    async with CodexClient() as client:
        response = await client.thread_start(cwd=".")
        thread_id = response.thread.id

        async for event in client.turn_stream(thread_id, "What files are here?"):
            # Print all event types
            print(f"[{event.event_type}]", end=" ")

            # Show event-specific details
            if event.is_delta():
                text = event.get_text_delta()
                if text:
                    print(f"text: {text[:50]}...")
                else:
                    print(f"data: {event.data}")
            elif event.is_completed():
                # Get ID from different event types with proper type safety
                from codex_adapter.models import (
                    ItemCompletedData,
                    RawResponseItemCompletedData,
                    TurnCompletedData,
                )

                if isinstance(event.data, ItemCompletedData | RawResponseItemCompletedData):
                    print("✓ item")
                elif isinstance(event.data, TurnCompletedData):
                    print(f"✓ turn:{event.data.turn.id}")
                else:
                    print("✓")
            elif event.is_error():
                print(f"✗ {event.data}")
            else:
                print(event.data)

            if event.event_type == "turn/completed":
                break


async def main() -> None:
    """Run all examples."""
    examples: list[tuple[str, Callable[[], Any]]] = [
        # ("Simple Chat", simple_chat),
        # ("Multi-Turn Chat", multi_turn_chat),
        # ("Model Override", model_override_example),
        ("Event Inspection", event_inspection_example),
    ]

    if len(sys.argv) > 1:
        # Run specific example by number
        try:
            idx = int(sys.argv[1]) - 1
            if 0 <= idx < len(examples):
                name, func = examples[idx]
                print(f"Running: {name}\n")
                await func()
            else:
                print(f"Invalid example number. Choose 1-{len(examples)}")
        except ValueError:
            print("Usage: python example.py [example_number]")
    else:
        # Run all examples
        print("Codex Adapter Examples")
        print("=" * 50)
        print("Running all examples...\n")

        for i, (name, func) in enumerate(examples, 1):
            print(f"\n{'=' * 50}")
            print(f"Example {i}: {name}")
            print("=" * 50)
            await func()


if __name__ == "__main__":
    asyncio.run(main())
