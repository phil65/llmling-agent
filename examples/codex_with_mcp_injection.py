"""Example: Injecting MCP servers into Codex agent programmatically.

This demonstrates how to inject MCP servers when initializing a Codex agent,
similar to how ToolManagerBridge creates in-process MCP servers for ACP agents.
"""

from __future__ import annotations

from codex_adapter import CodexClient, HttpMcpServer, StdioMcpServer


async def example_with_http_mcp_server():
    """Example: Inject an HTTP MCP server (like ToolManagerBridge creates)."""
    # Suppose we have a ToolManagerBridge running on port 8000
    # (In practice, you'd start the bridge first and get the URL)
    mcp_servers = {
        "agentpool-tools": HttpMcpServer(
            url="http://localhost:8000/mcp",
            # Optional: if the MCP server requires authentication
            bearer_token_env_var="AGENTPOOL_MCP_TOKEN",
        ),
    }

    async with CodexClient(mcp_servers=mcp_servers) as client:
        thread = await client.thread_start(cwd="/path/to/project")
        print(f"Started thread: {thread.thread.id}")

        # Now the Codex agent has access to all tools exposed by the MCP server
        async for event in client.turn_stream(
            thread.thread.id, "List available tools and show what they can do"
        ):
            if text := event.get_text_delta():
                print(text, end="", flush=True)
        print()


async def example_with_stdio_mcp_server():
    """Example: Inject a stdio-based MCP server."""
    mcp_servers = {
        "bash": StdioMcpServer(
            command="npx",
            args=["-y", "@openai/codex-shell-tool-mcp"],
        ),
    }

    async with CodexClient(mcp_servers=mcp_servers) as client:
        thread = await client.thread_start(cwd="/tmp")
        print(f"Started thread: {thread.thread.id}")

        async for event in client.turn_stream(thread.thread.id, "List files in current directory"):
            if text := event.get_text_delta():
                print(text, end="", flush=True)
        print()


async def example_with_multiple_mcp_servers():
    """Example: Inject multiple MCP servers at once."""
    mcp_servers = {
        # HTTP-based MCP server (e.g., ToolManagerBridge)
        "tools": HttpMcpServer(
            url="http://localhost:8000/mcp",
            http_headers={"X-Custom-Header": "value"},
        ),
        # Stdio-based MCP server
        "bash": StdioMcpServer(
            command="npx",
            args=["-y", "@openai/codex-shell-tool-mcp"],
            env={"DEBUG": "1"},  # Optional environment variables
        ),
        # Another HTTP MCP server (maybe from composio or another service)
        "composio": HttpMcpServer(
            url="https://api.composio.dev/mcp",
            bearer_token_env_var="COMPOSIO_API_KEY",
        ),
    }

    async with CodexClient(mcp_servers=mcp_servers) as client:
        thread = await client.thread_start(cwd="/path/to/project")
        print(f"Started thread with {len(mcp_servers)} MCP servers: {thread.thread.id}")

        # The agent now has access to tools from all three MCP servers
        async for event in client.turn_stream(
            thread.thread.id,
            "Show me all available tools and their sources",
        ):
            if text := event.get_text_delta():
                print(text, end="", flush=True)
        print()


async def example_integration_with_agentpool():
    """Example: How this would integrate with AgentPool's ToolManagerBridge.

    This is conceptual - shows how you'd use a ToolManagerBridge's MCP server
    config with a Codex agent.
    """
    # In AgentPool, you'd have something like:
    # bridge = ToolManagerBridge(node, config)
    # await bridge.start()
    # mcp_config = bridge.get_claude_mcp_server_config()

    # Simulating what that would return:
    mcp_config = {"agentpool-tools": {"type": "http", "url": "http://localhost:8765/mcp"}}

    # Convert to CodexClient format
    mcp_servers = {
        name: HttpMcpServer(url=config["url"])
        for name, config in mcp_config.items()
        if config["type"] == "http"
    }

    async with CodexClient(mcp_servers=mcp_servers) as client:
        thread = await client.thread_start()
        print(f"Codex agent with AgentPool tools: {thread.thread.id}")

        # Now the Codex agent can use tools from AgentPool!
        async for event in client.turn_stream(
            thread.thread.id,
            "Use the available tools to analyze this project",
        ):
            if text := event.get_text_delta():
                print(text, end="", flush=True)
        print()


if __name__ == "__main__":
    # Run the examples (comment out the ones you don't want to run)

    # Example 1: HTTP MCP server (like ToolManagerBridge)
    # asyncio.run(example_with_http_mcp_server())

    # Example 2: Stdio MCP server
    # asyncio.run(example_with_stdio_mcp_server())

    # Example 3: Multiple MCP servers
    # asyncio.run(example_with_multiple_mcp_servers())

    # Example 4: Integration with AgentPool
    # asyncio.run(example_integration_with_agentpool())

    print("Examples ready to run - uncomment the ones you want to try!")
