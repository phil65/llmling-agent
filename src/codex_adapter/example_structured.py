"""Example of using structured responses with Codex."""

from __future__ import annotations

import asyncio
import json

from codex_adapter import CodexClient


async def extract_structured_response() -> None:
    """Example: Extract structured data from agent response."""
    print("=" * 60)
    print("STRUCTURED RESPONSE EXAMPLE")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")
        print(f"\n✓ Started thread: {thread.id}\n")

        # Define JSON Schema for structured output
        schema = {
            "type": "object",
            "properties": {
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "type": {"type": "string"},
                            "size_kb": {"type": "number"},
                        },
                        "required": ["name", "type"],
                    },
                },
                "total_count": {"type": "number"},
                "summary": {"type": "string"},
            },
            "required": ["files", "total_count", "summary"],
            "additionalProperties": False,
        }

        print("Schema:")
        print(json.dumps(schema, indent=2))
        print("\nAsking: 'List Python files in current directory as structured JSON'\n")

        structured_content = ""
        async for event in client.turn_stream(
            thread.id,
            "List the Python files in the current directory. "
            "Return as JSON with: files (array of {name, type, size_kb}), "
            "total_count, and summary.",
            output_schema=schema,
        ):
            # Collect agent message
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                structured_content += delta
                print(delta, end="", flush=True)

            elif event.event_type == "turn/completed":
                print("\n\n✓ Turn completed")
                break

        # Parse the structured response
        if structured_content:
            try:
                # The response should be valid JSON matching our schema
                data = json.loads(structured_content)
                print("\n" + "=" * 60)
                print("PARSED STRUCTURED DATA:")
                print("=" * 60)
                print(json.dumps(data, indent=2))

                print("\n" + "=" * 60)
                print("ACCESSING FIELDS:")
                print("=" * 60)
                print(f"Total files: {data['total_count']}")
                print(f"Summary: {data['summary']}")
                print("\nFiles:")
                for file in data["files"]:
                    name = file["name"]
                    ftype = file["type"]
                    size = file.get("size_kb", "unknown")
                    print(f"  - {name} ({ftype}, {size}KB)")

            except json.JSONDecodeError as e:
                print(f"\n✗ Failed to parse JSON: {e}")
                print(f"Raw content:\n{structured_content}")


async def multiple_choice_response() -> None:
    """Example: Get a multiple choice answer."""
    print("\n" + "=" * 60)
    print("MULTIPLE CHOICE EXAMPLE")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")

        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "enum": ["python", "javascript", "rust", "go", "other"],
                },
                "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                "reasoning": {"type": "string"},
            },
            "required": ["answer", "confidence", "reasoning"],
        }

        print("\nAsking: 'What language is this project primarily written in?'\n")

        response_text = ""
        async for event in client.turn_stream(
            thread.id,
            "Looking at the files in this directory, what programming language "
            "is this project primarily written in? Choose from: python, javascript, "
            "rust, go, or other. Include your confidence level and brief reasoning.",
            output_schema=schema,
        ):
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                response_text += delta
                print(delta, end="", flush=True)
            elif event.event_type == "turn/completed":
                break

        if response_text:
            try:
                data = json.loads(response_text)
                print("\n\n✓ Structured response:")
                print(f"  Language: {data['answer']}")
                print(f"  Confidence: {data['confidence']}")
                print(f"  Reasoning: {data['reasoning']}")
            except json.JSONDecodeError:
                print(f"\n✗ Failed to parse: {response_text}")


async def classification_example() -> None:
    """Example: Classify code complexity."""
    print("\n" + "=" * 60)
    print("CLASSIFICATION EXAMPLE")
    print("=" * 60)

    async with CodexClient() as client:
        thread = await client.thread_start(cwd=".")

        schema = {
            "type": "object",
            "properties": {
                "complexity": {
                    "type": "string",
                    "enum": ["simple", "moderate", "complex", "very_complex"],
                },
                "metrics": {
                    "type": "object",
                    "properties": {
                        "files": {"type": "number"},
                        "lines_of_code": {"type": "number"},
                        "dependencies": {"type": "number"},
                    },
                },
                "main_technologies": {"type": "array", "items": {"type": "string"}},
                "assessment": {"type": "string"},
            },
            "required": ["complexity", "metrics", "main_technologies", "assessment"],
        }

        print("\nAsking: 'Assess the complexity of this codebase'\n")

        response_text = ""
        async for event in client.turn_stream(
            thread.id,
            "Analyze this codebase and classify its complexity. "
            "Provide metrics, main technologies used, and an assessment.",
            output_schema=schema,
        ):
            if event.event_type == "item/agentMessage/delta":
                delta = event.get_text_delta()
                response_text += delta
                print(delta, end="", flush=True)
            elif event.event_type == "turn/completed":
                break

        if response_text:
            try:
                data = json.loads(response_text)
                print("\n\n✓ Codebase Assessment:")
                print(f"  Complexity: {data['complexity']}")
                print(f"  Files: {data['metrics']['files']}")
                print(f"  Lines: {data['metrics']['lines_of_code']}")
                print(f"  Dependencies: {data['metrics']['dependencies']}")
                print(f"  Technologies: {', '.join(data['main_technologies'])}")
                print(f"  Assessment: {data['assessment']}")
            except json.JSONDecodeError:
                print(f"\n✗ Failed to parse: {response_text}")


async def main() -> None:
    """Run all structured response examples."""
    await extract_structured_response()
    await multiple_choice_response()
    await classification_example()

    print("\n" + "=" * 60)
    print("ALL STRUCTURED RESPONSE EXAMPLES COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
