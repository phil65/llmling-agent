# Basic Concepts

## Node Hierarchy

LLMling-agent uses a unified **node** abstraction for all message-processing entities. This enables seamless composition and communication between different types of agents.

### MessageNode

The base interface that all nodes implement. A node can receive messages, process them, and emit responses. This common interface allows uniform handling regardless of implementation.

### Agents

Native LLM-powered agents defined in your configuration:

```yaml
agents:
  assistant:
    model: openai:gpt-4
    system_prompts:
      - You are a helpful assistant.
```

Agents are the most flexible nodes - they support tools, structured output, conversation history, and all framework features.

### ACP Agents

External coding agents integrated via the [Agent Client Protocol](https://agentclientprotocol.com/). These wrap tools like Claude Code, Goose, Codex, or fast-agent as nodes:

```yaml
acp_agents:
  claude:
    type: claude
    cwd: /path/to/project
  goose:
    type: goose
```

ACP agents appear alongside native agents in your pool and can be connected, composed into teams, or used as tools - just like any other node.

### Teams

Groups of nodes that execute together:

- **Parallel teams** (`agent1 & agent2`): All nodes run simultaneously
- **Sequential pipelines** (`agent1 | agent2`): Output flows from one to the next

Teams themselves are nodes, enabling nested composition.

### Why This Matters

The unified node model means you can:

- Connect a native agent to an ACP agent
- Build a team mixing Claude Code with your custom agents
- Use any node as a tool for another node
- Apply the same monitoring, logging, and storage to all nodes

## Core Components

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
  planner:
    model: "anthropic:claude-sonnet-4-0"
    ...
```

Compared to other Frameworks, the YAML schema is a different beast and the capabilites to define agents statically are way more extensive.
It is possible to:

- Assign tools, toolsets, mcp servers
- Connect the agent to other agents with different "Connection types"
- Define and assign respone types for structured output in YAML
- Define and activate event triggers in YAML
- Set up (multiple) storage providers to write the conversations, tool calls, commands and much more to databases as well as files (pretty-printed or structured)
- Load previous conversations and even describe the Queries in the yaml file using simple syntax
- Assign agents to other agents for agent-as-a-tool-usage
- Assign agents to other agents as a resource (which gets evaluated on start. Also works nested to define pipeline-like patterns in easy ways)



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
