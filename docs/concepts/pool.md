## Pool Basics

The Agent Pool is the central coordination point for multi-agent systems in LLMling. It manages agent lifecycles, enables collaboration, and provides shared resources across agents.

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
    system_prompts:
      - "You analyze and summarize information precisely."

  planner:
    model: openai:gpt-5
    description: "Creates execution plans based on analysis"
    system_prompts:
      - "You create detailed execution plans."
```

```python
from llmling_agent import AgentPool
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

if __name__ == "__main__":
    asyncio.run(main())
```

### Collaboration Hub

The pool enables and manages collaboration between agents:

- **Discovery**: Agents can find and interact with other pool members
- **Dependencies**: Type-safe sharing of dependencies between agents
- **Team Support**: Creation and management of agent teams
- **Message Routing**: Flexible routing of messages between agents

```python
async with AgentPool[AppConfig](manifest_path) as pool:
    # Create a team of agents
    team = pool.create_team(["analyzer", "planner"])

    # Set up message routing
    analyzer = pool.get_agent("analyzer")
    planner = pool.get_agent("planner")

    # Forward analyzer results to planner
    analyzer >> planner

    # Or using the team
    await team.run_sequential("Process this task...")
```

### Resource Sharing & Monitoring

The pool provides shared resources and monitoring capabilities:

- **Shared Storage**: Common storage configuration for all agents
- **Dependency Injection**: Pool-level dependencies available to all agents
- **Central Monitoring**: `pool_talk` provides a unified view of all pool communication

```python
# Monitor all pool messages
@pool.pool_talk.message_received.connect
def on_pool_message(message: ChatMessage):
    print(f"[{message.role}] {message.name}: {message.content}")

# Share dependencies
async with AgentPool[AppConfig](
    manifest_path,
    shared_deps=app_config
) as pool:
    # All agents have access to app_config
    agent = pool.get_agent("analyzer")
    assert agent.context.data == app_config

    # Monitor execution
    result = await agent.run("Analyze this...")
    # Messages appear in pool_talk
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
    Avoid getting the same agent multiple times with different dependencies or return types,
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

### Adding New Agents

The `add_agent()` method creates new agents dynamically:

```python
# Basic agent creation
agent = await pool.add_agent(
    "dynamic_agent",
    model="openai:gpt-5",
    system_prompt="You are a helpful assistant.",
)

# Structured agent with return type
planner = await pool.add_agent(
    "planner",
    output_type=PlanResult,
    model="openai:gpt-5",
    system_prompt="You create execution plans.",
)

```


### Creating Teams

The pool can also create teams from its registered agents:

```python
# Create team from agent names
team = pool.create_team(["analyzer", "planner", "executor"])

# With shared prompt
team = pool.create_team(
    ["researcher", "writer"],
    shared_prompt="Focus on technical accuracy."
)
```

The team creation is covered in detail in the Teams chapter.
