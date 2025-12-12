"""Example demonstrating AggregatingServer with multiple protocol servers.

This example shows how to:
1. Create an AgentPool with multiple agents
2. Set up multiple HTTP servers (AG-UI, A2A)
3. Run them in unified mode (single port) or separate mode (multiple ports)
"""

from __future__ import annotations

import asyncio

from llmling_agent import Agent, AgentPool
from llmling_agent_server import A2AServer, AggregatingServer, AGUIServer


async def main_unified() -> None:
    """Run AggregatingServer in unified HTTP mode (single port)."""

    # Create multiple agents with different specializations
    def research_callback(message: str) -> str:
        return f"Research: Analyzing your query about: {message}"

    def analysis_callback(message: str) -> str:
        return f"Analysis: Here's my detailed analysis of: {message}"

    def summary_callback(message: str) -> str:
        return f"Summary: In brief, regarding {message}..."

    # Create agents
    research_agent = Agent.from_callback(name="researcher", callback=research_callback)
    analysis_agent = Agent.from_callback(name="analyzer", callback=analysis_callback)
    summary_agent = Agent.from_callback(name="summarizer", callback=summary_callback)

    # Create agent pool and register agents
    pool = AgentPool()
    pool.register("researcher", research_agent)
    pool.register("analyzer", analysis_agent)
    pool.register("summarizer", summary_agent)

    # Create protocol servers (ports are ignored in unified mode)
    agui_server = AGUIServer(pool, name="agui-server")
    a2a_server = A2AServer(pool, name="a2a-server")

    # Create aggregating server in UNIFIED mode - single port for all!
    aggregating_server = AggregatingServer(
        pool,
        servers=[agui_server, a2a_server],
        name="unified-multi-protocol-server",
        unified_http=True,  # Enable unified mode
        unified_host="localhost",
        unified_port=8000,  # All servers on one port
    )

    print("Starting Aggregating Server in UNIFIED HTTP mode...")
    print(f"\nSingle endpoint: {aggregating_server.unified_base_url}")
    print("\nAll routes accessible at http://localhost:8000:")
    print("  AG-UI endpoints:")
    print("    - GET  /agui/           (list agents)")
    print("    - POST /agui/researcher")
    print("    - POST /agui/analyzer")
    print("    - POST /agui/summarizer")
    print("  A2A endpoints:")
    print("    - GET  /a2a/            (list agents)")
    print("    - POST /a2a/researcher")
    print("    - POST /a2a/analyzer")
    print("    - POST /a2a/summarizer")
    print("    - GET  /a2a/{agent}/.well-known/agent-card.json")
    print("  Root endpoint:")
    print("    - GET  /                (list all servers and routes)")

    print("\nPress Ctrl+C to stop the server\n")

    async with aggregating_server, aggregating_server.run_context():
        print("\nUnified server running!")
        print(f"Mode: {aggregating_server!r}")

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")


async def main_separate() -> None:
    """Run AggregatingServer in separate mode (multiple ports)."""

    def callback(message: str) -> str:
        return f"Response: {message}"

    agent = Agent.from_callback(name="helper", callback=callback)

    pool = AgentPool()
    pool.register("helper", agent)

    # Each server gets its own port in separate mode
    agui_server = AGUIServer(pool, host="localhost", port=8002, name="agui-server")
    a2a_server = A2AServer(pool, host="localhost", port=8001, name="a2a-server")

    # Separate mode (default) - each server on its own port
    aggregating_server = AggregatingServer(
        pool,
        servers=[agui_server, a2a_server],
        name="separate-multi-protocol-server",
        unified_http=False,  # Default - separate ports
    )

    print("Starting Aggregating Server in SEPARATE mode...")
    print(f"\nAG-UI Server: {agui_server.base_url}")
    print(f"A2A Server: {a2a_server.base_url}")

    print("\nPress Ctrl+C to stop all servers\n")

    async with aggregating_server, aggregating_server.run_context():
        print(f"\nServers running: {aggregating_server.running_server_count}")

        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down all servers...")


if __name__ == "__main__":
    """
    Example usage with curl (UNIFIED MODE):

    # List all servers and routes
    curl http://localhost:8000/

    # List agents via AG-UI
    curl http://localhost:8000/agui/

    # List agents via A2A
    curl http://localhost:8000/a2a/

    # Send request to researcher via AG-UI
    curl -X POST http://localhost:8000/agui/researcher \
      -H "Content-Type: application/json" \
      -d '{
        "messages": [
          {"role": "user", "content": "Research quantum computing"}
        ]
      }'

    # Get agent card via A2A
    curl http://localhost:8000/a2a/analyzer/.well-known/agent-card.json

    # Send request via A2A
    curl -X POST http://localhost:8000/a2a/analyzer \
      -H "Content-Type: application/json" \
      -d '{
        "jsonrpc": "2.0",
        "id": "1",
        "method": "message/send",
        "params": {
          "messages": [
            {
              "role": "user",
              "parts": [{"kind": "text", "text": "Analyze market trends"}]
            }
          ]
        }
      }'

    Benefits of unified mode:
    - Single port to expose/configure
    - Shared uvicorn instance (more efficient)
    - Simpler deployment
    - All protocols accessible from same base URL
    - Automatic route prefixing (/agui/, /a2a/, etc.)
    """
    # Run unified mode by default
    asyncio.run(main_unified())

    # To run separate mode instead:
    # asyncio.run(main_separate())
