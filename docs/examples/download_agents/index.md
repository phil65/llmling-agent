---
title: Download Agents
description: Sequential vs parallel downloads comparison
icon: material/download
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`



# Multi-Agent Download System with Cheerleader


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
storage:
  # List of storage providers (can use multiple)
  providers:
    # Primary storage using SQLite
    - type: sql
      url: "sqlite:///history.db" # Database URL (SQLite, PostgreSQL, etc.)
    # Also output all messages, tool calls etc as a pretty printed text file
    - type: text_file
      path: "logs/chat.log"
      format: "chronological" # "chronological" or "conversations"
      template: "chronological"

agents:
  fan:
    display_name: "Async Agent Fan"
    description: "The #1 supporter of all agents!"
    toolsets:
      - type: agent_management # Need to know who to cheer for!
    model: openai:gpt-5-nano
    system_prompts:
      - |
        You are the MOST ENTHUSIASTIC async fan who runs in the background!
        Your job is to:
        1. Find all other agents using your tool (don't include yourself!)
        2. Cheer them on with over-the-top supportive messages considering the situation.
        3. Never stop believing in your team! 🎉
    tools:
      - llmling_agent_docs/examples.download_agents.main.cheer

  file_getter_1:
    display_name: "Mr. File Downloader"
    description: "Downloads files from URLs"
    model: openai:gpt-5-nano
    system_prompts:
      - "You have ONE job: use the download_file tool to download files."
    toolsets:
      - type: file_access

  overseer:
    display_name: "Download Coordinator"
    description: "Coordinates parallel downloads"
    model: openai:gpt-5-nano
    toolsets:
      - type: agent_management
    system_prompts:
      - |
        You coordinate file downloads using available agents. Your job is to:
        1. Check out the available agents and assign each of them the download task
        2. Report the EXACT download results from the agents including speeds and sizes

  file_getter_2:
    inherits: file_getter_1

```


### `main.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///


"""Example comparing sequential and parallel downloads with agents.

This example demonstrates:
- Continuous repetitive tasks
- Async parallel execution of LLM calls
- YAML config definitions
- Capability use: list other agents and delegate tasks
- Simple stateful callback mechanism using a class
- Storage providers: SQLite and pretty-printed text files
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import TYPE_CHECKING, Any

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent.agent.events import RichAgentStreamEvent
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


if TYPE_CHECKING:
    from pydantic_ai import RunContext


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")

FILE_URL = "http://speedtest.tele2.net/10MB.zip"
TEAM_PROMPT = f"Download this file: {FILE_URL}"
OVERSEER_PROMPT = f"""
Please coordinate downloading this file twice: {FILE_URL}

Delegate to file_getter_1 and file_getter_2. Report the results.
"""


def cheer(slogan: str) -> None:
    """🥳🎉 Cheer! Use this tool to show your apprreciation."""
    print(slogan)


@dataclass
class CheerProgress:
    """Class for tracking the progress of downloads and providing feedback."""

    def __init__(self) -> None:
        self.situation = "The team is assembling, ready to start the downloads!"

    def create_prompt(self) -> str:
        """Create a prompt for the fan based on current situation."""
        return f"Current situation: {self.situation}\nBe an enthusiastic and encouraging fan!"

    def update(self, situation: str) -> None:
        """Update the current situation and print it."""
        self.situation = situation
        print(situation)


async def run_example() -> None:
    # Load config from YAML
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)

    async def event_handler(ctx: RunContext, event: RichAgentStreamEvent[Any]) -> None:
        from llmling_agent.agent.events import ToolCallProgressEvent

        if isinstance(event, ToolCallProgressEvent):
            print(f"Progress: {event.progress}/{event.total} - {event.message}")

    async with AgentPool(
        manifest,
        event_handlers=[event_handler],
    ) as pool:
        # Get agents from the YAML config
        worker_1 = pool.get_agent("file_getter_1")
        worker_2 = pool.get_agent("file_getter_2")
        fan = pool.get_agent("fan")
        progress = CheerProgress()

        # Run fan in background with progress updates
        await fan.run_in_background(progress.create_prompt)

        # Sequential downloads
        progress.update("Sequential downloads starting - let's see how they do!")
        sequential_team = worker_1 | worker_2
        sequential = await sequential_team.execute(TEAM_PROMPT)
        progress.update(f"Downloads completed in {sequential.duration:.2f} secs!")

        # Parallel downloads
        parallel_team = worker_1 & worker_2
        parallel = await parallel_team.execute(TEAM_PROMPT)
        progress.update(f"Parallel downloads completed in {parallel.duration:.2f} secs!")

        # Overseer coordination
        overseer = pool.get_agent("overseer")
        result = await overseer.run(OVERSEER_PROMPT)
        progress.update(f"\nOverseer's report: {result.data}")

        await fan.stop()  # End of joy.


if __name__ == "__main__":
    run(run_example())

```



This example demonstrates several advanced features of LLMling-agent:

- Continuous repetitive tasks
- Async parallel execution of LLM calls
- YAML configuration with storage providers
- Capability usage (agent listing and task delegation)
- Stateful callback mechanism
- Multiple storage providers (SQLite + pretty-printed logs)


## How It Works

1. We set up a team of downloaders and a cheerleading fan
2. The fan runs continuously in the background, getting updates via callbacks
3. We test downloads in different modes:
   - Sequential (one after another)
   - Parallel (both at once)
   - Overseer-coordinated (using agent capabilities)
4. The fan cheers appropriately for each situation
5. All interactions are logged to both SQLite and pretty-printed text files

This demonstrates:

- Background tasks with continuous prompts
- Agent cloning
- Team operations (sequential vs parallel)
- Capability-based delegation
- Multi-provider storage

