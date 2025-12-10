---
title: MCP Servers (YAML)
description: MCP server integration with git tools
icon: material/file-code
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main_py.py`
- `main_yaml.py`



# AI-Human Interaction


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
mcp_servers:
  - "uvx mcp-server-git"

agents:
  picker:
    model: openai:gpt-5-nano
    description: Git commit history explorer
    system_prompts:
      - You are a specialist in looking up git commits using your tools from the current working directory.
    connections:
      - type: node
        name: analyzer

  analyzer:
    model: openai:gpt-5-nano
    description: Git commit analyzer
    system_prompts:
      - You are an expert in retrieving and returning information about a specific commit from the current working directory.

```


### `main_py.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///

"""Example: Two agents working together to explore git commit history."""

from __future__ import annotations

from llmling_agent import Agent, Team
from llmling_agent_docs.examples.utils import run


PICKER = """
You are a specialist in looking up git commits using your tools
from the current working directory."
"""
ANALYZER = """
You are an expert in retrieving and returning information
about a specific commit from the current working directoy."
"""

MODEL = "openai:gpt-5-nano"
SERVERS = ["uvx mcp-server-git"]


async def run_example() -> None:
    picker = Agent(model=MODEL, system_prompt=PICKER, mcp_servers=SERVERS)
    analyzer = Agent(model=MODEL, system_prompt=ANALYZER, mcp_servers=SERVERS)

    # Connect picker to analyzer
    picker >> analyzer

    # Register message handlers to see the messages
    picker.message_sent.connect(lambda msg: print(msg.format()))
    analyzer.message_sent.connect(lambda msg: print(msg.format()))
    # For MCP servers, we need async context.
    async with picker, analyzer:
        # Start the chain by asking picker for the latest commit
        await picker.run("Get the latest commit hash! ")

    # MCP servers also work on team level for all its members
    agent_without_mcp_server = Agent(model=MODEL, system_prompt=ANALYZER)
    team = Team([agent_without_mcp_server], mcp_servers=["uvx mcp-hn"])
    async with team:
        # this will show you the MCP server tools
        print(await agent_without_mcp_server.tools.get_tools())


if __name__ == "__main__":
    run(run_example())


"""
Output:

CommitPicker: The latest commit hash is **9bcd7718dbc33f16239d0522ca677ed75bac997b**.
CommitAnalyzer: The latest commit with hash **9bcd7718dbc33f16239d0522ca677ed75bac997b**
includes the following details:

- **Author:** Philipp Temminghoff
- **Date:** January 20, 2025, at 01:59:43 (local time)
- **Commit Message:** chore: docs

### Changes made:
...
"""

```


### `main_yaml.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///


"""Example demonstrating MCP server integration with git tools.

This example shows:
- Using MCP servers to provide git functionality to agents
- Agent connections through YAML configuration
- Message flow between connected agents
- Team-level MCP server configuration
"""

from __future__ import annotations

import os

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


PROMPT = "Get the latest commit hash!"

# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


async def run_example() -> None:
    """Run example using YAML configuration."""
    # Load config from YAML
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool(manifest) as pool:
        # Get agents (connections already set up from YAML)
        picker = pool.get_agent("picker")
        analyzer = pool.get_agent("analyzer")

        # Register handlers to see messages
        picker.message_sent.connect(lambda msg: print(msg.format()))
        analyzer.message_sent.connect(lambda msg: print(msg.format()))

        # Start the chain
        await picker.run(PROMPT)


if __name__ == "__main__":
    run(run_example())

```



This example demonstrates how AI agents can interact with humans:

- Using agent capabilities for human interaction
- Setting up a human agent in the pool
- Allowing AI to request human input when needed

## How It Works

1. We set up two agents:
   - An AI assistant with `can_ask_agents` capability
   - A human agent using the special "human" provider type

2. When the AI assistant encounters a question it can't answer:
   - It recognizes the need for human input
   - Uses its `can_ask_agents` capability to interact with the human agent
   - Incorporates the human's response into its answer

3. The conversation might look like this:
   ```
   Assistant: I need to check about Project DoomsDay's status. Let me ask the human.
   Human: Project DoomsDay is currently in Phase 2, with 60% completion.
   Assistant: Based on the human's input, Project DoomsDay is in Phase 2 and is 60% complete.
   ```

This demonstrates how to:

- Enable AI-human collaboration
- Control when AI can request human input
- Integrate human knowledge into AI responses

