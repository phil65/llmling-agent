# LLMling-Agent

[![PyPI License](https://img.shields.io/pypi/l/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Package status](https://img.shields.io/pypi/status/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Monthly downloads](https://img.shields.io/pypi/dm/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Distribution format](https://img.shields.io/pypi/format/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Wheel availability](https://img.shields.io/pypi/wheel/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Python version](https://img.shields.io/pypi/pyversions/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Implementation](https://img.shields.io/pypi/implementation/llmling-agent.svg)](https://pypi.org/project/llmling-agent/)
[![Releases](https://img.shields.io/github/downloads/phil65/llmling-agent/total.svg)](https://github.com/phil65/llmling-agent/releases)
[![Github Contributors](https://img.shields.io/github/contributors/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/graphs/contributors)
[![Github Discussions](https://img.shields.io/github/discussions/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/discussions)
[![Github Forks](https://img.shields.io/github/forks/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/forks)
[![Github Issues](https://img.shields.io/github/issues/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/issues)
[![Github Watchers](https://img.shields.io/github/watchers/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/watchers)
[![Github Stars](https://img.shields.io/github/stars/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/stars)
[![Github last commit](https://img.shields.io/github/last-commit/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/commits)
[![Github release date](https://img.shields.io/github/release-date/phil65/llmling-agent)](https://github.com/phil65/llmling-agent/releases)
[![Github language count](https://img.shields.io/github/languages/count/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Github commits this year](https://img.shields.io/github/commit-activity/y/phil65/llmling-agent)](https://github.com/phil65/llmling-agent)
[![Package status](https://codecov.io/gh/phil65/llmling-agent/branch/main/graph/badge.svg)](https://codecov.io/gh/phil65/llmling-agent/)
[![PyUp](https://pyup.io/repos/github/phil65/llmling-agent/shield.svg)](https://pyup.io/repos/github/phil65/llmling-agent/)

### [Read the documentation!](https://phil65.github.io/llmling-agent/)

# 🚀 Getting Started

LLMling Agent is a framework for creating and managing LLM-powered agents. It integrates with LLMling's resource system and provides structured interactions with language models.

## ✨ Features

- 🔄 Modern python written from ground up with Python 3.13
- 🤝 Integrate multiple external ACP agents (Claude Code, Codex, Goose, etc.), AGUI Agents as well as native Pydantic-AI based agents into a single pool where they can cooperate, interact and delegate.
- 🛡️ Complete (multi-)agent pool setup via YAML files including extensive JSON schema to help with creating configurations.
- 🔌 Extensive MCP support including elicitation, sampling, progress reporting, multi-modality, including bridging to ACP / AG-UI protocols.
- 🛜 Comletely UPath backed. All file operations (& Code execution) by agents are abstrated in a way that agents can operate directly on remote sources without having to install anything on the remote.
- 🎤 Streaming TTS- support for all Agents
- 📚 Improved aider-based RepoMap implementation for code exploration
- 📂 Composable virtual filesystems for agents
- 📝 CodeMode support

## Quick Start

### Agent Client Protocol (ACP)

The fastest way to start chatting with an AI:

llmling-agent supports the Agent Client Protocol for seamless integration with desktop applications and IDEs. Run your agents as ACP servers to enable bidirectional communication, session management, and file operations through JSON-RPC 2.0 over stdio.

*The recommended client is Zed IDE (& soon Toad, a python client based on Textual)*

```bash
# Start ACP server
llmling-agent[default,coding] serve-acp [path/to/config.yml]
```

Compatible with ACP-enabled Clients like Zed. See the [ACP Integration documentation](https://phil65.github.io/llmling-agent/advanced/acp_integration/) for setup instructions.

Run `/help` in the chat to see what commands are at your disposal.

### YAML configuration

While you can define agents with 3 lines of YAML (or competely programmatic or via CLI),
you can also create agents as well as their connections, agent tasks, storage providers and much more via YAML.
This is the extended version

```yaml
# agents.yml
agents:
  analyzer:
    name: "Code Analyzer"  # Display name
    inherits: "base_agent"  # Optional parent config to inherit from
    description: "Code analysis specialist"
    debug: false
    retries: 1  # Number of retries for failed operations
    model:  # Model configuration
      type: "fallback"  # Lot of special "meta-models" included out of the box!
      models:  # Try models in sequence
        - "openai:gpt-5"
        - "openai:gpt-5-nano"
        - "anthropic:claude-sonnet-4-0"
    # Structured output
    output_type:
      type: "inline"  # or "import" for Python types
      fields:
        severity:
          type: "str"
          description: "Issue severity"
        issues:
          type: "list[str]"
          description: "Found issues"

    # Core behavior
    system_prompts:
      - "You analyze code for potential issues and improvements."

    # Session & History
    session:
      name: "analysis_session"
      since: "1h"  # Only load messages from last hour
      roles: ["user", "assistant"]  # Only specific message types

    # Toolsets (available tool groups)
    toolsets:
      - type: agent_management  # Enables delegation
      - type: resource_access   # Enables resource loading


    # Knowledge sources
    knowledge:
      paths: ["docs/**/*.md"]  # Glob patterns for files
      resources:
        - type: "repository"
          url: "https://github.com/user/repo"
      prompts:
        - type: "file"
          path: "prompts/analysis.txt"

    # MCP Server integration
    mcp_servers:
      - type: "stdio"
        command: "python"
        args: ["-m", "mcp_server"]
        env:
          DEBUG: "1"
      - "python -m other_server"  # shorthand syntax

    # Worker agents (specialists)
    workers:
      - type: agent
        name: "formatter"
        reset_history_on_run: true
        pass_message_history: false
      - "linter"  # shorthand syntax

    # Message forwarding
    connections:
      - type: node
        name: "reporter"
        connection_type: "run"  # "run" | "context" | "forward"
        priority: 1
        queued: true
        queue_strategy: "latest"
        transform: "my_module.transform_func"
        wait_for_completion: true
        filter_condition:  # When to forward messages
          type: "word_match"
          words: ["error", "warning"]
          case_sensitive: false
        stop_condition:  # When to disconnect
          type: "message_count"
          max_messages: 100
          count_mode: "total"  # or "per_agent"
        exit_condition:  # When to exit application
          type: "cost_limit"
          max_cost: 10.0
    # Event triggers
    triggers:
      - type: "file"
        name: "code_change"
        paths: ["src/**/*.py"]
        extensions: [".py"]
        debounce: 1000  # ms
teams:
  # Complex workflows via YAML
  full_pipeline:
    mode: sequential
    members:
      - analyzer
      - planner
    connections:
      - type: node
        name: final_reviewer
        wait_for_completion: true
      - type: file
        path: "reports/{date}_workflow.txt"
# Response type definitions
responses:
  AnalysisResult:
    response_schema:
      type: "inline"
      description: "Code analysis result format"
      fields:
        severity: {type: "str"}
        issues: {type: "list[str]"}

  ComplexResult:
    type: "import"
    import_path: "myapp.types.ComplexResult"

# Storage configuration
storage:
  providers:
    - type: "sql"
      url: "sqlite:///history.db"
      pool_size: 5
    - type: "text_file"
      path: "logs/chat.log"
      format: "chronological"
  log_messages: true
  log_conversations: true
  log_commands: true

# Pre-defined jobs
jobs:
  analyze_code:
    name: "Code Analysis"
    description: "Analyze code quality"
    prompt: "Analyze this code: {code}"
    required_return_type: "AnalysisResult"
    knowledge:
      paths: ["src/**/*.py"]
    tools: ["analyze_complexity", "run_linter"]
```

You can use an Agents manifest in multiple ways:



- Run it using the CLI

```bash
llmling-agent run --config agents.yml my_agent "Some prompt"
```

- Start *watch mode* and only react to triggers

```bash
llmling-agent watch --config agents.yml
```


### Agent Pool: Multi-Agent Coordination

The `AgentPool` allows multiple agents to work together on tasks, including external ACP-enabled agents like Claude Code, Codex, or Goose. Here's a practical example of parallel file downloading:

```yaml
# agents.yml
agents:
  file_getter:
    model: openai:gpt-5-mini
    toolsets:
      - type: file_access  # includes download_file, read_file, list_directory
    system_prompts:
      - |
        You are a download specialist. Just use the download_file tool
        and report its results. No explanations needed.

  overseer:
    toolsets:
      - type: agent_management  # Enables delegation and agent discovery tools
    model: openai:gpt-5-mini
    system_prompts:
      - |
        You coordinate downloads using available agents.
        1. Check out the available agents and assign each of them the download task
        2. Report the results.

```

Programmatic Usage:

```python
from llmling_agent.delegation import AgentPool

async def main():
    async with AgentPool("agents.yml") as pool:
        # first we create two agents based on the file_getter template
        file_getter_1 = pool.get_agent("file_getter")
        file_getter_2 = pool.get_agent("file_getter")
        # then we form a team and execute the task
        team = file_getter_1 & file_getter_2
        responses = await team.run_parallel("Download https://example.com/file.zip")

        # Or let a coordinator orchestrate using his capabilities.
        coordinator = pool.get_agent("coordinator")
        result = await overseer.run(
            "Download https://example.com/file.zip by delegating to all workers available!"
        )
```

#### External ACP Agents

You can also integrate external ACP-enabled agents into your pool via YAML configuration:

```yaml
# agents.yml
acp_agents:
  claude:
    type: claude
    display_name: "Claude Code"
    description: "Claude Code through ACP"
  goose:
    type: goose
    display_name: "Goose"
    description: "Block's Goose agent through ACP"

agents:
  coordinator:
    model: openai:gpt-5-mini
    toolsets:
      - type: agent_management  # Enables delegation to ACP agents
```

```python
async with AgentPool("agents.yml") as pool:
    # Access external ACP agents just like regular agents
    claude = pool.get_agent("claude")
    result = await claude.run("Refactor this code")
```

See the [ACP Integration documentation](https://phil65.github.io/llmling-agent/advanced/acp_integration/#external-acp-agents) for supported agents and configuration options.


The framework provides three types of message nodes:

1. **Agents**: Individual LLM-powered actors
```python
# Single agent processing
analyzer = pool.get_agent("analyzer")
result = await analyzer.run("analyze this")
```

2. **Teams**: Groups for parallel execution
```python
# Create team using & operator
team = analyzer & planner & executor
results = await team.run("handle this task")
```

3. **TeamRuns**: Sequential execution chains
```python
# Create chain using | operator
chain = analyzer | planner | executor
results = await chain.run("process in sequence")
```

The beauty of this system is that these nodes are completely composable:

```python

def process_text(text: str) -> str:
    return text.upper()

# Nested structures work naturally
team_1 = analyzer & planner  # Team
team_2 = validator & reporter  # Another team
chain = team_1 | process_text | team_2  # Teams and Callables in a chain

# Complex workflows become intuitive
(analyzer & planner) | validator  # Team followed by validator
team_1 | (team_2 & agent_3)  # Chain with parallel components

# Every node has the same core interface
async for message in node.run_iter("prompt"):
    print(message.content)

# Monitoring works the same for all types
print(f"Messages: {node.stats.message_count}")
print(f"Cost: ${node.stats.total_cost:.2f}")
```
(note: the operator overloading is just syntactic sugar. In general, teams should be created
using `pool.create_team()` / `pool.create_team_run()` or `agent/team.connect_to()`)
)

All message nodes support the same execution patterns:
```python
# Single execution
result = await node.run("prompt")

# Streaming
async for event in node.run_stream("prompt"):
    print(event)


# Nested teams work naturally
team_1 = analyzer & planner  # First team
team_2 = validator & reporter  # Second team
parallel_team = Team([team_1, agent_3, team_2])  # Team containing teams!

# This means you can create sophisticated structures:
result = await parallel_team.run("analyze this")  # Will execute:
# - team_1 (analyzer & planner) in parallel
# - agent_3 in parallel
# - team_2 (validator & reporter) in parallel

# And still use all the standard patterns:
async for msg in parallel_team.run_iter("prompt"):
    print(msg.content)

# With full monitoring functionality:
print(f"Total cost: ${parallel_team.stats.total_cost:.2f}")

```

This unified system makes it easy to:
- Build complex workflows
- Monitor message flow
- Compose nodes in any combination
- Use consistent patterns across all node types

Each message in the system carries content, metadata, and execution information, providing a consistent interface across all types of interactions. See [Message System](docs/core-concepts/messages.md) for details.


### Advanced Connection Features

Connections between agents are highly configurable and support various patterns:

```python
# Basic connection in shorthand form.
connection = agent_a >> agent_b  # Forward all messages

# Extended setup: Queued connection (manual processing)
connection = agent_a.connect_to(
    agent_b,
    queued=True,
    queue_strategy="latest",  # or "concat", "buffer"
)
# messages can queue up now
await connection.trigger(optional_additional_prompt)  # Process queued messages sequentially

# Filtered connection (example: filter by keyword):
connection = agent_a.connect_to(
    agent_b,
    filter_condition=lambda ctx: "keyword" in ctx.message.content,
)

# Conditional disconnection (example: disconnect after cost limit):
connection = agent_a.connect_to(
    agent_b,
    filter_condition=lambda ctx: ctx.stats.total_cost > 1.0,
)

# Message transformations
async def transform_message(message: str) -> str:
    return f"Transformed: {message}"

connection = agent_a.connect_to(agent_b, transform=transform_message)

# Connection statistics
print(f"Messages processed: {connection.stats.message_count}")
print(f"Total tokens: {connection.stats.token_count}")
print(f"Total cost: ${connection.stats.total_cost:.2f}")
```

The two basic programmatic patterns of this librry are:

1. Tree-like workflows (hierarchical):
```python
# Can be modeled purely with teams/chains using & and |
team_a = agent1 & agent2  # Parallel branch 1
team_b = agent3 & agent4  # Parallel branch 2
chain = preprocessor | team_a | postprocessor  # Sequential with team
nested = Team([chain, team_b])  # Hierarchical nesting
```

2. DAG (Directed Acyclic Graph) workflows:
```python
# Needs explicit signal connections for non-tree patterns
analyzer = Agent("analyzer")
planner = Agent("planner")
executor = Agent("executor")
validator = Agent("validator")

# Can't model this with just teams - need explicit connections
analyzer.connect_to(planner)
analyzer.connect_to(executor)  # Same source to multiple targets
planner.connect_to(validator)
executor.connect_to(validator) # Multiple sources to same target
validator.connect_to(executor) # Cyclic connections
```

BOTH connection types can be set up for BOTH teams and agents intiuiviely in the YAML file.

You can also use LLMling-models for more sophisticated human-in-the-loop integration:
- Remote human operators via network
- Hybrid human-AI workflows
- Input streaming support
- Custom UI integration


### Multi-Modal Support

Handle images and PDFs alongside text (depends on provider / model support)

```python
from llmling_agent import Agent

async with Agent(...) as agent:
    result = await agent.run("What's in this image?", pathlib.Path("image.jpg"))
    result = await agent.run("What's in this PDF?", pathlib.Path("document.pdf"))
```

### Command System

Extensive slash commands available when used via ACP:

```bash
/list-tools              # Show available tools
/enable-tool tool_name   # Enable specific tool
/connect other_agent     # Forward results
/model gpt-5            # Switch models
/history search "query"  # Search conversation
/stats                   # Show usage statistics
```

### Storage & Analytics

All interaction is tracked using (multiple) configurable storage providers.
Information can get fetched  via CLI.


```bash
# View recent conversations
llmling-agent history show
llmling-agent history show --period 24h  # Last 24 hours
llmling-agent history show --query "database"  # Search content

# View usage statistics
llmling-agent history stats  # Basic stats
llmling-agent history stats --group-by model  # Model usage
llmling-agent history stats --group-by day    # Daily breakdown
```

## 📚 MkDocs Integration

In combination with [MkNodes](https://github.com/phil65/mknodes) and the [MkDocs plugin](https://github.com/phil65/mkdocs_mknodes),
you can easily generate static documentation for websites with a few lines of code.

```python

@nav.route.page("Feature XYZ", icon="oui:documentation", hide="toc")
def gen_docs(page: mk.MkPage):
    """Generate docs using agents."""
    agent = Agent(model="openai:gpt-5-nano")
    page += mk.MkAdmonition("MkNodes includes all kinds of Markdown objects to generate docs!")
    source_code = load_source_code_from_folder(...)
    page += mk.MkCode() # if you want to display source code
    result = agent.run.sync("Describle Feature XYZ in MkDocs compatible markdown including examples.", content)
    page += result.content
```

### [Read the documentation for further info!](https://phil65.github.io/llmling-agent/)
