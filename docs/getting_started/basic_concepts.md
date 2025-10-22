# Basic Concepts

## Core Components

### Agents
An Agent in LLMling-agent is best thought of as an "agent framework" or "agent container" rather than an agent itself.
It provides the infrastructure and orchestration for agent-like behavior, but delegates the actual "thinking" to providers.
Currently LLMling-Agents are backed by 4 possible providers:

- AI Provider: Uses pydantic-ai and language models
- Human Provider: Gets responses through human input
- Callback Provider: Uses Python functions to process prompts to output

Key aspects:

- Manages infrastructure (tools, history, connections, events, resources)
- Handles message routing and tool execution
- Provides type safety and validation
- Integrates with storage and events
- Coordinates with other agents

The actual agent behavior (language model, human input, etc.) is pluggable via providers.

### Configuration and YAML

LLMling-agent excels at static definition of agents using YAML files and Pydantic models:

```yaml
# agents.yml (AgentsManifest)
agents:
  analyzer:    # AgentConfig
    analyzer:
        model: "openai:gpt-5"
        system_prompts: [...]
        toolsets: [...]
        environment: {...}
  planner:
    model: "anthropic:claude-sonnet-4-0"
    ...
```

Compared to other Frameworks, the YAML schema is a different beast and the capabilites to define agents statically are way more extensive.
It is possible to:

- Assign tools as well as special tools and capabilities
- Connect the agent to other agents with different "Connection types"
- Define and assign respone types for structured output in YAML
- Define and activate event triggers in YAML
- Set up (multiple) storage providers to write the conversations, tool calls, commands and much more to databases as well as files (pretty-printed or structured)
- Load previous conversations and even describe the Queries in the yaml file using simple syntax
- Assign agents to other agents for agent-as-a-tool-usage
- Assign agents to other agents as a resource (which gets evaluated on start. Also works nested to define pipeline-like patterns in easy ways)


The hierarchy is:

- **AgentsManifest**: Top-level configuration (YAML file)

- Defines available agents
- Sets up shared resources
- Configures storage providers
- Defines response types

- **AgentConfig**: Per-agent configuration (YAML section)

- Sets model/provider
- Defines capabilities
- Configures environment
- Sets up knowledge sources

All configuration is validated using Pydantic models, providing:

- Type safety
- Schema validation
- IDE support
- Clear error messages

### Providers
Providers implement the actual "agent behavior". The Agent class provides the framework, while providers handle the "thinking":

- **AI Provider**: Uses pydantic-ai and language models
- **Human Provider**: Gets responses through human input
- **Callback Provider**: Uses Python functions

### Pools
A Pool is a collection of agents that can:

- Share resources and knowledge
- Discover each other
- Communicate and delegate tasks
- Be monitored and supervised

Think of a pool as a workspace where agents can collaborate.

### Teams
Teams are dynamic groups of agents from a pool that work together on specific tasks. They support:

- Parallel execution
- Sequential processing
- Controlled communication
- Result aggregation

### Connections

Connections define how agents communicate. They include:

- Direct message forwarding
- Context sharing
- Task delegation
- Response awaiting

Connections can be:

- One-to-one
- One-to-many
- Temporary or permanent
- Conditional or unconditional
- Queued, accumulated, debounced, filtered
- Team-to-Team, Team-to-Callable, Team-to-Agent

### Tasks

Tasks are pre-defined operations that agents can execute. They include:

- Prompt templates
- Required tools
- Knowledge sources
- Expected result types

## Mental Model

### Message Flow

1. User/system sends message to agent (run call)
2. Agent processes via provider
3. Provider may use tools
4. Response is generated
5. Message gets returned and possibly also forwarded via connections into the connection layer.
6. Depending on the connection set up, we can start at step 2 again


## Key Patterns

### Component Setup
```python
# Create pool from manifest
async with AgentPool("agents.yml") as pool:
    # Get agent
    agent = pool.get_agent("analyzer")
    # Create team
    team = pool.create_team(["analyzer", "planner"])
    # Connect agents
    analyzer >> planner  # Forward results
```

### Message Processing
```python
# Basic usage
result = await agent.run("Analyze this code")

# Structured responses
result = await agent.to_structured(AnalysisResult).run("Analyze this code")

# Team execution
result = await team.run_parallel("Process this")
```
