---
title: MCP Server integration
description: Model Context Protocol integration
icon: material/server
---

## Overview

LLMling integrates with MCP (Model Context Protocol) servers to provide additional tools and functionality to agents.

## Defining MCP Servers

### In YAML Configuration

```yaml
# In agent configuration
agents:
  code_assistant:
    description: "Helps with coding tasks"
    model: openai:gpt-5
    mcp_servers:
      # Simple string form (stdio)
      - "python -m myserver --debug"

      # Full configuration
      - type: stdio
        command: python
        args: ["-m", "codetools"]
        env:
          PYTHONPATH: src
          DEBUG: "1"

      # SSE server (coming soon)
      - type: sse
        url: "http://localhost:3001/events"
```

### In Agent Constructor

```python
from llmling_agent import Agent, StdioMCPServerConfig

# Create agent with MCP servers
agent = Agent(
    mcp_servers=[
        # String form
        "python -m myserver",

        # Full configuration
        StdioMCPServerConfig(
            command="python",
            args=["-m", "codetools"],
            env={"DEBUG": "1"}
        )
    ]
)
```

## Initialization

MCP servers are initialized when the agent enters its async context:

```python
async with Agent(mcp_servers=servers) as agent:
    # MCP servers are now connected
    # Tools are available
    await agent.run("Use MCP tools...")

# MCP servers are automatically cleaned up
```

The initialization process:

1. Creates MCP clients for each server
2. Establishes connections
3. Retrieves available tools
4. Registers tools with the agent

## Using MCP Tools

MCP tools become available like any other agent tool:

```python
async with Agent(mcp_servers=servers) as agent:
    # List available tools
    tools = await agent.tools.get_tools(source="mcp")

    # Use MCP tools in prompts
    result = await agent.run("""
        Analyze this code using the code_analyzer tool
        from the MCP server
    """)
```

Tools are automatically converted to LLMling's tool format with proper typing and documentation.

## Configuration Options

### StdioMCPServerConfig

```python
class StdioMCPServerConfig:
    """MCP server using stdio communication."""
    command: str
    """Command to execute"""

    args: list[str]
    """Command arguments"""

    env: dict[str, str] | None
    """Environment variables"""

    enabled: bool = True
    """Whether server is active"""
```

### SSEMCPServerConfig (Coming Soon)

```python
class SSEMCPServerConfig:
    """MCP server using SSE transport."""
    url: str
    """Server endpoint URL"""

    enabled: bool = True
    """Whether server is active"""
```

MCP integration allows agents to seamlessly use tools from external services while maintaining LLMling's type safety and resource management.
