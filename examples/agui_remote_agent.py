"""Example demonstrating remote AG-UI agent usage.

This example shows how to connect to and use a remote AG-UI protocol server.
"""

from __future__ import annotations

import asyncio

from llmling_agent.agents import AGUIAgent


async def basic_usage():
    """Basic AG-UI agent usage."""
    print("=== Basic Usage ===\n")

    async with AGUIAgent(
        endpoint="http://localhost:8000/agent/run",
        name="remote-assistant",
        timeout=30.0,
    ) as agent:
        # Simple question
        result = await agent.run("What is the capital of France?")
        print("Q: What is the capital of France?")
        print(f"A: {result.content}\n")

        # Get statistics
        stats = await agent.get_stats()
        print(f"Stats: {stats}\n")


async def streaming_usage():
    """Streaming AG-UI agent usage."""
    print("=== Streaming Usage ===\n")

    async with AGUIAgent(
        endpoint="http://localhost:8000/agent/run",
        name="streaming-assistant",
    ) as agent:
        print("Q: Tell me a short story about AI\n")
        print("A: ", end="", flush=True)

        # Stream the response
        async for event in agent.run_stream("Tell me a short story about AI"):
            # Print text deltas as they arrive
            from pydantic_ai import PartDeltaEvent
            from pydantic_ai.messages import TextPartDelta

            if isinstance(event, PartDeltaEvent) and isinstance(event.delta, TextPartDelta):
                print(event.delta.content_delta, end="", flush=True)

        print("\n")


async def tool_conversion():
    """Convert AG-UI agent to a callable tool."""
    print("=== Tool Conversion ===\n")

    async with AGUIAgent(
        endpoint="http://localhost:8000/agent/run",
        name="calculator-agent",
    ) as agent:
        # Convert to tool
        calculator_tool = agent.to_tool("A calculator agent that can solve math problems")

        # Use as a tool
        result = await calculator_tool("What is 157 * 89?")
        print(f"Calculator result: {result}\n")


async def multiple_requests():
    """Execute multiple requests with same agent."""
    print("=== Multiple Requests ===\n")

    async with AGUIAgent(
        endpoint="http://localhost:8000/agent/run",
        name="multi-request-agent",
    ) as agent:
        questions = [
            "What is 2+2?",
            "Name three primary colors",
            "What is the speed of light?",
        ]

        for i, question in enumerate(questions, 1):
            result = await agent.run(question)
            print(f"{i}. Q: {question}")
            print(f"   A: {result.content}\n")


async def with_custom_headers():
    """Use AG-UI agent with custom headers."""
    print("=== Custom Headers ===\n")

    headers = {
        "X-API-Key": "your-api-key-here",
        "X-User-ID": "user123",
    }

    async with AGUIAgent(
        endpoint="http://localhost:8000/agent/run",
        name="auth-agent",
        headers=headers,
    ) as agent:
        result = await agent.run("Hello with authentication")
        print(f"Response: {result.content}\n")


async def error_handling():
    """Demonstrate error handling."""
    print("=== Error Handling ===\n")

    try:
        async with AGUIAgent(
            endpoint="http://invalid-endpoint:9999/run",
            name="error-agent",
            timeout=5.0,
        ) as agent:
            await agent.run("This will fail")
    except (OSError, TimeoutError) as e:
        print(f"Caught expected error: {type(e).__name__}")
        print(f"Message: {e}\n")


async def main():
    """Run all examples."""
    examples = [
        ("Basic Usage", basic_usage),
        ("Streaming", streaming_usage),
        ("Tool Conversion", tool_conversion),
        ("Multiple Requests", multiple_requests),
        ("Custom Headers", with_custom_headers),
        ("Error Handling", error_handling),
    ]

    print("AG-UI Remote Agent Examples")
    print("=" * 50)
    print()

    for name, example_func in examples:
        try:
            await example_func()
        except (OSError, TimeoutError, RuntimeError) as e:
            print(f"Example '{name}' failed: {e}\n")

        # Small delay between examples
        await asyncio.sleep(0.5)


if __name__ == "__main__":
    asyncio.run(main())
