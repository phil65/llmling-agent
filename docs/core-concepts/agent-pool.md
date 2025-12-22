---
title: Agent Pool
description: Agent pool management and features
icon: material/database
---

## Pool Basics

The Agent Pool is the central coordination point for multi-agent systems. It manages agent lifecycles, enables collaboration, and provides shared resources across agents.

### Central Registry

The pool acts as a central registry for all agents, managing their complete lifecycle from creation to cleanup:

- **Single Access Point**: All agents are accessed through the pool using their unique names
- **Lifecycle Management**: The pool handles async initialization and cleanup of agents
- **Manifest-Based**: Agent configurations are defined in YAML manifests
- **Dynamic Creation**: Agents can be created and cloned at runtime
- **Type Safety**: Pool can be typed with shared dependency type: `AgentPool[TDeps]`

Here's a typical pool setup with two agents:

```yaml
# agents.yml
agents:
  analyzer:
    model: openai:gpt-5
    description: "Analyzes input and extracts key information"
    system_prompt: "You analyze and summarize information precisely."

  planner:
    model: openai:gpt-5
    description: "Creates execution plans based on analysis"
    system_prompt: "You create detailed execution plans."
```

```python
from agentpool import AgentPool
from myapp.config import AppConfig  # Your dependency type

async def main():
    # Initialize pool with shared dependencies
    async with AgentPool[AppConfig]("agents.yml", shared_deps=app_config) as pool:
        # Get existing agent
        analyzer = pool.get_agent("analyzer")

        # Create new agent dynamically
        planner = await pool.add_agent(
            "dynamic_planner",
            model="openai:gpt-5",
            system_prompt="You plan next steps.",
        )
        # Use agents
        result = await analyzer.run("Analyze this text...")
        # create teams
        team = pool.create_team([analyzer, planner])
        await team.run("Process this task...")

if __name__ == "__main__":
    asyncio.run(main())
```

## Adding Agents to a Pool

The pool provides several ways to access and create agents, with a focus on type safety and dependency management.

### Getting Agents from Registry

The primary way to get agents is via the `get_agent()` method, which retrieves agents defined in the manifest:

```python
async with AgentPool[AppConfig](manifest_path) as pool:
    # Basic agent retrieval
    agent = pool.get_agent("analyzer")

    # With return type for structured output
    analyzer = pool.get_agent(
        "analyzer",
        return_type=AnalysisResult
    )

    # With custom dependencies
    agent = pool.get_agent(
        "analyzer",
        deps=custom_config
    )
```

!!! warning "Type Safety Best Practice"
    For best type safety, retrieve each agent by name only once and store the reference.
    Avoid getting the same agent multiple times with different dependency or return types,
    as this can lead to inconsistent typing.

    ```python
    # ✅ Good: Get once and reuse
    analyzer = pool.get_agent("analyzer")
    await analyzer.run("First task")
    await analyzer.run("Second task")

    # ❌ Avoid: Getting multiple times
    await pool.get_agent("analyzer").run("First task")
    await pool.get_agent("analyzer").run("Second task")
    ```
