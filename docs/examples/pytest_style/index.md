---
title: PyTest-Style Functions
description: PyTest-style example patterns
icon: material/code-braces
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`



# PyTest-Style Agent Functions


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
# This example creates agents programmatically - no YAML config needed
agents: {}

```


### `main.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///


"""Example showing agent function discovery and execution.

This example demonstrates:
- Using agents as function decorators
- Automatic function discovery
- Dependency injection
- Execution order control
- Function result handling
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from llmling_agent.running import node_function, run_nodes_async
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


if TYPE_CHECKING:
    from llmling_agent import Agent

# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


DATA = """
Monthly Sales Data (2023):
Jan: $12,500
Feb: $15,300
Mar: $18,900
Apr: $14,200
May: $16,800
Jun: $21,500
"""


@node_function
async def analyze_data(analyzer: Agent) -> str:
    """First step: Analyze the data."""
    result = await analyzer.run(f"Analyze this sales data and identify trends:\n{DATA}")
    return result.data


@node_function(depends_on="analyze_data")
async def summarize_analysis(writer: Agent, analyze_data: str) -> str:
    """Second step: Create an executive summary."""
    prompt = f"Create a brief executive summary of this sales analysis:\n{analyze_data}"
    result = await writer.run(prompt)
    return result.data


async def run_example() -> None:
    """Run the analysis pipeline."""
    # Load config and run nodes
    config_path = get_config_path(None if is_pyodide() else __file__)
    results = await run_nodes_async(config_path, parallel=True)

    # Print results
    print("Analysis:", results["analyze_data"])
    print("Summary:", results["summarize_analysis"])


if __name__ == "__main__":
    run(run_example())

```



This example demonstrates a pytest-inspired way to work with agents:

- Using agents as function decorators
- Automatic function discovery
- Dependency injection
- Execution order control
- Function result handling


## How It Works

1. Functions are decorated with `@node_function`
2. Type hints specify which agent to inject (`analyzer: Agent`)
3. Dependencies are declared in the decorator (`depends_on="analyze_data"`)
4. Results from one function can be injected into another
5. All functions are discovered and executed in the correct order

Key Features:

- Automatic agent injection based on type hints
- Function dependency resolution
- Parallel execution where possible
- Results passed automatically between functions

This provides a clean, declarative way to orchestrate multi-agent workflows, similar to how pytest fixtures work.

