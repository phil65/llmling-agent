---
title: MCP Sampling & Elicitation
description: MCP server with sampling tools
icon: material/lightning-bolt
---


## Try it in Pydantic Playground




## Files

- `demo.py`
- `server.py`



# MCP Sampling & Elicitation Example


### `demo.py`

```py
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

        result = await agent.run(f"Please use fix_code to analyze and fix this code: {buggy_code}")
        print(f"\n✅ Agent response:\n{result.data}")

        print("\n✨ Code fixer demo completed!")


if __name__ == "__main__":
    asyncio.run(main())

```


### `server.py`

```py
"""Compact FastMCP server demonstrating sampling and elicitation in one workflow."""

from fastmcp import Context, FastMCP
from mcp.types import ModelHint, ModelPreferences, TextContent


mcp = FastMCP("Code Fixer Server")


@mcp.tool
async def fix_code(ctx: Context, code: str) -> str:
    """Analyze code, ask user which issues to fix, then return improved code."""
    # Step 1: Use sampling to check if there are issues (yes/no)
    prefs = ModelPreferences(hints=[ModelHint(name="gpt-5-nano")])
    has_issues_result = await ctx.sample(
        f"Does this code have any syntax errors, bugs, or style issues?\n\n{code}\n\n"
        "Respond with only 'yes' or 'no'.",
        max_tokens=500,
        system_prompt="You are a code reviewer. Respond with only 'yes' or 'no'.",
        model_preferences=prefs,
    )

    assert isinstance(has_issues_result, TextContent)
    if has_issues_result.text.strip().lower() != "yes":
        return f"Code looks good! No issues found.\n\nOriginal code:\n{code}"

    # Step 2: Use elicitation to ask user whether to fix (boolean)
    prompt = "LLM found issues in your code. Should I fix them?"
    fix_request = await ctx.elicit(prompt, response_type=bool)  # type: ignore[arg-type]

    if fix_request.action != "accept" or not fix_request.data:
        return f"No changes made.\n\nOriginal code:\n{code}"

    # Step 3: Use sampling to generate fixed code
    fix_result = await ctx.sample(
        f"Fix all issues in this code:\n\n{code}",
        max_tokens=1000,
        system_prompt="You are a code fixer. Return only the corrected code.",
        model_preferences=prefs,
    )

    assert isinstance(fix_result, TextContent)
    fixed_code = fix_result.text
    return f"Code fixed!\n\nOriginal:\n{code}\n\nFixed:\n{fixed_code}"


if __name__ == "__main__":
    mcp.run()

```



This example demonstrates how to create and use a FastMCP server that combines **sampling** and **elicitation** in a single workflow.

## Overview

The example consists of:
- `server.py`: A compact MCP server with one comprehensive tool
- `demo.py`: Demo script showing agent interaction with the server

## The Code Fixer Tool

### `fix_code(code: str) -> str`
A single tool that demonstrates both MCP capabilities in one workflow:

1. **Sampling** (Server-side LLM): Analyzes the provided code for syntax errors, style issues, and improvements
2. **Elicitation** (Direct user interaction): Asks the user whether to proceed with fixing the identified issues  
3. **Sampling** (Server-side LLM): Generates the corrected code based on the analysis and user approval

**Input**: Code string (e.g., `print("hello world")` with typo)
**Output**: Analysis results and fixed code (if approved)

## Key Patterns

- **Server autonomy**: The server orchestrates a complex multi-step workflow internally
- **Direct user interaction**: Server asks user for decisions without going through the agent
- **Server-side intelligence**: Uses its own LLM for both analysis and code generation
- **Single tool interface**: Agent sees one simple tool, server handles complexity

## Running the Example

```bash
# Run the demo
uv run demo.py

# Or run server standalone
uv run server.py
```

The demo shows a complete workflow: code analysis → user confirmation → code fixing, all within one tool call.
