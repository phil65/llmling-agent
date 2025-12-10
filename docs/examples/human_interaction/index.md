---
title: Human Interaction
description: AI-Human interaction patterns
icon: material/account
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`



# AI-Human Interaction


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
agents:
  assistant:
    model: openai:gpt-5-nano
    toolsets:
      - type: agent_management
    system_prompts:
      - |
        You are a helpful assistant. When you're not sure about something,
        don't hesitate to ask the human agent for guidance.

  human:
    model:
      type: input
    description: "A human who can provide answers"

```


### `main.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///


"""Example of AI-Human interaction using agent capabilities.

This example demonstrates:
- Using a human agent for interactive input
- AI agent querying human agent when unsure
- Using the can_ask_agents capability
"""

from __future__ import annotations

import os

from llmling_agent import AgentPool, AgentsManifest
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


QUESTION = """
What is the current status of Project DoomsDay?
This is crucial information that only a human would know.
If you don't know, ask the agent named "human".
"""


async def run_example() -> None:
    # Load config from YAML
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)

    async with AgentPool(manifest) as pool:
        # Get the assistant agent
        assistant = pool.get_agent("assistant")

        # Run interaction
        await assistant.run(QUESTION)

        # Print conversation history
        print(await assistant.conversation.format_history())


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

