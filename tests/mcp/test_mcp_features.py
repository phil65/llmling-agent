# /// script
# dependencies = ["llmling-agent"]
# ///

"""Demo: Agent using MCP server with code fixer (sampling + elicitation)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llmling_agent import Agent
from llmling_agent_config.mcp_server import StdioMCPServerConfig


async def main():
    """Demo MCP server with code fixer workflow."""
    print("🚀 Starting code fixer demo...")

    # Get server path
    server_path = Path(__file__).parent / "server.py"

    # Create MCP server config
    mcp_server = StdioMCPServerConfig(
        name="code_fixer_demo",
        command="uv",
        args=["run", str(server_path)],
    )

    # Create agent with MCP server
    agent = Agent(
        name="demo_agent",
        model="openai:gpt-5-nano",
        system_prompt="You are a helpful assistant with code fixing tools.",
        mcp_servers=[mcp_server],
    )

    async with agent:
        print(f"📋 Agent created with tools: {list(await agent.tools.get_tools())}")
        async for event in agent.run_stream(
            "Test the progress tool with a funny message"
        ):
            print(event)


if __name__ == "__main__":
    asyncio.run(main())
