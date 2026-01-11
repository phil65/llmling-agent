"""Example of the simplified PydanticAI-like structured response API."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

from codex_adapter import CodexClient


class FileInfo(BaseModel):
    """Information about a single file."""

    name: str = Field(description="File name")
    type: str = Field(description="File type/extension")
    size_kb: float | None = Field(default=None, description="File size in KB")


class FileListResponse(BaseModel):
    """Structured response for file listing."""

    files: list[FileInfo] = Field(description="List of files found")
    total_count: int = Field(description="Total number of files")
    summary: str = Field(description="Brief summary of findings")


async def simple_structured_example() -> None:
    """Demonstrate the simplified structured response API.

    This shows the PydanticAI-like approach where you just pass a type
    and get back a typed object automatically.
    """
    print("=" * 60)
    print("SIMPLE STRUCTURED RESPONSE API")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")
        print(f"\n✓ Started thread: {thread.id}\n")

        # One line to get a structured, typed response!
        result = await client.turn_stream_structured(
            thread.id,
            "List the Python files in the current directory.",
            FileListResponse,  # Just pass the type
        )

        # Result is automatically parsed and typed!
        print("\n" + "=" * 60)
        print("STRUCTURED RESULT (automatically parsed):")
        print("=" * 60)
        print(f"Total files: {result.total_count}")
        print(f"Summary: {result.summary}")
        print("\nFiles:")
        for file in result.files:
            size = f"{file.size_kb}KB" if file.size_kb else "unknown"
            print(f"  - {file.name} ({file.type}, {size})")

        # Full type safety - IDE autocomplete works!
        if result.total_count > 0:
            first_file = result.files[0]
            print(f"\n✓ First file: {first_file.name}")


async def comparison_example() -> None:
    """Show the difference between the old and new API."""
    print("\n" + "=" * 60)
    print("API COMPARISON")
    print("=" * 60)

    print("\n--- Old Way (Manual) ---")
    print("Code:")
    print("""
response_text = ""
async for event in client.turn_stream(
    thread.id,
    prompt,
    output_schema=FileListResponse,  # Generate schema
):
    if event.event_type == "item/agentMessage/delta":
        response_text += event.get_text_delta()
    elif event.event_type == "turn/completed":
        break

# Manually parse the response
result = FileListResponse.model_validate_json(response_text)
""")

    print("\n--- New Way (PydanticAI-like) ---")
    print("Code:")
    print("""
# One line - automatic schema and parsing!
result = await client.turn_stream_structured(
    thread.id,
    prompt,
    FileListResponse,
)
""")

    print("\n✓ Same result, much simpler API!")


async def main() -> None:
    """Run all examples."""
    await simple_structured_example()
    await comparison_example()

    print("\n" + "=" * 60)
    print("SIMPLE STRUCTURED API EXAMPLES COMPLETE")
    print("=" * 60)
    print("\n✓ Benefits:")
    print("  - One method call instead of manual streaming + parsing")
    print("  - Automatic schema generation from Pydantic type")
    print("  - Automatic JSON parsing and validation")
    print("  - Full type safety and IDE autocomplete")
    print("  - PydanticAI-like developer experience")


if __name__ == "__main__":
    asyncio.run(main())
