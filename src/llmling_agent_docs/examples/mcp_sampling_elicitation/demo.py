# /// script
# dependencies = ["llmling-agent"]
# ///

"""Demo: Agent using MCP server with code fixer (sampling + elicitation)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llmling_agent import Agent
from llmling_agent_config.mcp_server import StdioMCPServerConfig


async def main() -> None:
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
        # Code with actual bugs
        buggy_code = 'prin("hello world"'

        print("\n" + "=" * 60)
        print("Demo: Code Fixer (Sampling + Elicitation)")
        print(f"Original code: {buggy_code}")
        print("=" * 60)

        result = await agent.run(
            f"Please use fix_code to analyze and fix this code: {buggy_code}"
        )
        print(f"\n✅ Agent response:\n{result.data}")

        print("\n✨ Code fixer demo completed!")


if __name__ == "__main__":
    asyncio.run(main())
