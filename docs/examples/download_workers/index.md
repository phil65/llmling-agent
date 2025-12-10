---
title: Download Workers
description: Using agents as tools for downloads
icon: material/download
---


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
agents:
  overseer:
    display_name: "Download Coordinator"
    description: "Coordinates downloads using registered worker tools"
    model: openai:gpt-5-nano
    system_prompts:
      - "You coordinate file downloads using the worker tools registered with you. Use all available worker tools to download the requested file."

  file_getter_1:
    display_name: "File Downloader Worker 1"
    description: "Downloads files from URLs"
    model: openai:gpt-5-nano
    system_prompts:
      - "You download files from URLs using the download_file tool."
    toolsets:
      - type: file_access

  file_getter_2:
    display_name: "File Downloader Worker 2"
    inherits: file_getter_1

```


### `main.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///


"""Example of using agents as tools for downloads.

This example demonstrates:
- Registering agents as tools
- Using worker tools for delegation
- Sequential vs parallel execution
"""

from __future__ import annotations

import os
import time
from typing import Any

from llmling_agent import Agent, AgentPool, AgentsManifest
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


PROMPT = "Download this file using both agent tools available to you: http://speedtest.tele2.net/10MB.zip"

# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


async def run_example() -> None:
    # Load config from YAML
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool(manifest) as pool:
        # Get the boss agent
        boss: Agent[Any, str] = pool.get_agent("overseer")

        # Create second downloader by cloning the first
        worker_1 = pool.get_agent("file_getter_1")
        worker_2 = pool.get_agent("file_getter_2")

        # Register both as worker tools
        boss.register_worker(worker_1)
        boss.register_worker(worker_2)

        print("Calling both tools:")
        start_time = time.time()
        result = await boss.run(PROMPT)
        duration = time.time() - start_time
        print(f"Sequential time: {duration:.2f} seconds")
        print(result.data)


if __name__ == "__main__":
    run(run_example())

```




## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`




