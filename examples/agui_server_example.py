"""Example demonstrating AGUIServer with multiple agents.

This example shows how to:
1. Create an AgentPool with multiple agents
2. Set up an AGUIServer that exposes all agents
3. Access each agent via its own route
"""

from __future__ import annotations

import asyncio

from llmling_agent import Agent, AgentPool
from llmling_agent_server.agui_server import AGUIServer


async def main() -> None:
    """Run AGUIServer example with multiple agents."""

    # Create multiple agents with different capabilities
    def math_agent_callback(message: str) -> str:
        """Math specialist agent."""
        return f"Math Agent: Let me solve that math problem: {message}"

    def code_agent_callback(message: str) -> str:
        """Code specialist agent."""
        return f"Code Agent: Here's how to implement that: {message}"

    def writing_agent_callback(message: str) -> str:
        """Writing specialist agent."""
        return f"Writing Agent: Let me help you write: {message}"

    # Create agents
    math_agent = Agent.from_callback(
        name="math_specialist",
        callback=math_agent_callback,
    )

    code_agent = Agent.from_callback(
        name="code_specialist",
        callback=code_agent_callback,
    )

    writing_agent = Agent.from_callback(
        name="writing_specialist",
        callback=writing_agent_callback,
    )

    # Create agent pool and register agents
    pool = AgentPool()
    pool.register("math", math_agent)
    pool.register("code", code_agent)
    pool.register("writing", writing_agent)

    # Create AGUIServer
    server = AGUIServer(
        pool,
        host="localhost",
        port=8002,
        name="multi-agent-agui-server",
    )

    print("Starting AG-UI Server with multiple agents...")
    print(f"Server URL: {server.base_url}")
    print("\nAvailable agent endpoints:")
    for agent_name, url in server.list_routes().items():
        print(f"  - {agent_name}: POST {url}")
    print("\nAgent list endpoint:")
    print(f"  - GET {server.base_url}/")
    print("\nPress Ctrl+C to stop the server\n")

    # Run server
    async with server, server.run_context():
        # Server is now running and handling requests
        # Each agent is accessible at:
        #   POST http://localhost:8002/math
        #   POST http://localhost:8002/code
        #   POST http://localhost:8002/writing
        # List all agents:
        #   GET http://localhost:8002/

        try:
            # Keep server running
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("\nShutting down server...")


if __name__ == "__main__":
    """
    Example usage with curl:

    # List all available agents
    curl http://localhost:8002/

    # Send request to math agent
    curl -X POST http://localhost:8002/math \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'

    # Send request to code agent
    curl -X POST http://localhost:8002/code \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Write a hello world"}]}'

    # Send request to writing agent
    curl -X POST http://localhost:8002/writing \
      -H "Content-Type: application/json" \
      -d '{"messages": [{"role": "user", "content": "Write a haiku"}]}'
    """
    asyncio.run(main())
