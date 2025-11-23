# Agent Manifest

The agent manifest is a YAML file that defines your complete agent setup at the top level.
The config part is powered by [Pydantic](https://docs.pydantic.dev/latest/) and provides excellent validation
and IDE support for YAML linters by providing an extensive, detailed schema.

## Top-Level Structure

Here's the complete manifest structure with all available top-level sections:

```yaml
# Agent definitions
agents:
  analyzer:
    # See Agent Configuration documentation
    model: "openai:gpt-5"
    description: "Code analysis specialist"
    # ... full agent config
  
  planner:
    model: "openai:gpt-5-nano" 
    # ... additional agent configs

# Team definitions for multi-agent workflows
teams:
  full_pipeline:
    mode: sequential
    members:
      - analyzer
      - planner
    connections:
      - type: node
        name: final_reviewer
        wait_for_completion: true

# Shared response type definitions
responses:
  AnalysisResult:
    response_schema:
      type: "inline"
      fields:
        severity:
          type: "str"
          description: "Issue severity"
  CodeMetrics:
    type: "import"
    import_path: "myapp.types.CodeMetrics"

# Storage and logging configuration
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

# Observability configuration
observability:
  providers:
    - type: "opentelemetry"
      endpoint: "http://localhost:4317"

# Document conversion settings
conversion:
  audio:
    provider: "whisper"
  video:
    provider: "ffmpeg"

# Global MCP server configurations
mcp_servers:
  - type: "stdio"
    command: "python"
    args: ["-m", "mcp_server"]
  - "python -m other_server"  # shorthand syntax

# Pool server for external access
pool_server:
  type: "stdio"
  host: "localhost"
  port: 8080

# Global prompt library
prompts:
  library_path: "prompts/"
  auto_reload: true

# Global command shortcuts
commands:
  check_disk: "df -h"
  analyze:
    type: "static"
    content: "Analyze the current situation"

# Pre-defined reusable tasks
jobs:
  analyze_code:
    prompt: "Analyze this code: {code}"
    output_type: "AnalysisResult"
    knowledge:
      paths: ["src/**/*.py"]
    tools:
      - "analyze_complexity"
      - import_path: "myapp.tools.analyze_security"

# Resource definitions (filesystems, APIs, etc.)
resources:
  docs: "file://./docs"
  api:
    type: "source"
    uri: "https://api.example.com"
    cached: true
```

## Top-Level Sections

### `agents`
Dictionary of individual agent configurations. Each key is an agent identifier, and the value is the complete agent configuration.

See [Agent Configuration](agent_config.md) for detailed agent setup options.

### `teams`  
Dictionary of team configurations for multi-agent workflows. Teams can run agents in parallel or sequence.

See [Team Configuration](team_config.md) for team setup and coordination.

### `responses`
Dictionary of shared response type definitions that can be referenced by agents. Supports both inline schema definitions and imported Python types.

See [Response Configuration](response_config.md) for structured output setup.

### `storage`
Configuration for how agent interactions are stored and logged. Supports multiple storage providers.

See [Storage Configuration](storage_config.md) for persistence and logging options.

### `observability`
Configuration for monitoring and telemetry collection.

### `conversion`
Settings for document and media conversion capabilities.

### `mcp_servers`
Global Model Context Protocol server configurations that provide tools and resources to agents.

See [MCP Configuration](mcp_config.md) for server setup and integration.

### `pool_server`
Configuration for the pool server that exposes agent functionality to external clients.

### `prompts`
Global prompt library configuration and management.

See [Prompt Configuration](prompt_config.md) for prompt organization.

### `commands`
Global command shortcuts that can be used across agents for prompt injection.

### `jobs`
Pre-defined reusable tasks with templates, knowledge requirements, and expected outputs.

See [Task Configuration](task_config.md) for job definitions.

### `resources`
Global resource definitions for filesystems, APIs, and data sources that agents can access.

## Configuration Inheritance

The manifest supports YAML inheritance using the `INHERIT` key at the top level, similar to MkDocs:

```yaml
# base-config.yml
storage:
  log_messages: true
agents:
  base_agent:
    model: "openai:gpt-5"
    retries: 2

# main-config.yml  
INHERIT: base-config.yml
agents:
  my_agent:
    inherits: base_agent
    description: "Specialized agent"
```

LLMling-Agent supports UPaths (universal-pathlib) for inheritance, allowing remote configurations.

## Schema Validation

You can get IDE linter support by adding this line at the top of your YAML:

```yaml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
```

!!! note
    Versioned config files will arrive soon for better stability!

## Usage

Load a manifest in your code:

```python
from llmling_agent import AgentPool

async with AgentPool("agents.yml") as pool:
    agent = pool.get_agent("analyzer")
    result = await agent.run("Analyze this code...")
```

## Next Steps

- [Agent Configuration](agent_config.md) for individual agent setup
- [Team Configuration](team_config.md) for multi-agent workflows  
- [Storage Configuration](storage_config.md) for persistence options
- [Response Configuration](response_config.md) for structured outputs