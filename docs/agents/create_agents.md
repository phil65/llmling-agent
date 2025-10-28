# Creating Agents

There are several ways to create and initialize agents. Since agents can have complex setup requirements (MCP servers, runtime configuration, etc.), proper initialization is important.

## Main Approach: AgentPool
The recommended way to create agents is through `AgentPool`:

```python
async with AgentPool("agents.yml") as pool:
    # Get agent
    agent = pool.get_agent("analyzer")

    # Get agent with dependencies
    agent = pool.get_agent("reviewer", deps=pr_context)

    # Get structured agent
    agent = pool.get_agent("validator", return_type=ValidationResult)
```

This ensures:

- Proper async initialization of all components
- MCP server setup
- Runtime configuration
- Agent interconnections
- Resource loading

## Direct Agent Creation
For simpler cases, agents can be created directly:

```python
# Manual instantiation (requires more setup)
agent = Agent("agent_name", model="my_model")
async with agent:
    result = await agent.run("Hello!")
```

## Advanced Pool Creation
`AgentPool` offers additional creation methods:

```python
# Create from manifest
manifest = AgentsManifest.from_file("agents.yml")
pool = AgentPool(manifest)

# Create with manual configuration
pool = AgentPool(manifest, connect_nodes=False)

# Create with custom input provider
pool = AgentPool(manifest, input_provider=my_input_provider)
```

## Importance of Async Initialization

Agents require proper async initialization for:

1. MCP server setup and tool registration
2. Runtime configuration loading
3. Resource initialization
4. Connection setup

Always use async context managers:
```python
# ❌ Wrong - missing async init
agent = Agent(...)
result = await agent.run("Hello")

# ❌ Limited: Works, but not everything is initialized
agent = Agent(...)
result = agent.run.sync("Hello")

# ✅ Correct - proper async initialization
async with Agent(...) as agent:
    result = await agent.run("Hello")
```

## Dynamic Agent Creation
Agents can be created dynamically in several ways:

### CLI Creation
The CLI can create agents on-the-fly:
```bash
# Add agent file
llmling-agent add reviewer.yml
```

### Ephemeral Agents via Capabilities
Agents with `can_create_delegates` capability can spawn temporary agents:

```yaml
agents:
  orchestrator:
    toolsets:
      - type: agent_management
    # ...

# In Python
await agent.spawn_delegate(
    task="Research this topic",
    system_prompt="You are a research specialist...",
    model="gpt-5",
    connect_back=True  # Send results back to creator
)
```

### Worker Agents
Agents can be registered as tools for other agents:

```python
# Register agent as tool
parent.register_worker(
    worker,
    name="research_tool",
    reset_history_on_run=True,
    pass_message_history=False
)
```

# In YAML config

```yaml
agents:
  parent:
    workers:
      - type: agent
        name: "researcher"
        reset_history_on_run: true
```

## Best Practices

1. Use `AgentPool` for managing multiple agents
2. Always use async context managers
3. Consider using templates for common agent types
