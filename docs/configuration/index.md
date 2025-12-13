---
title: Agent Manifest
description: Complete manifest structure and organization
icon: material/file-code
order: 2
---

The agent manifest is a YAML file that defines your complete agent setup at the top level.
The config part is powered by [Pydantic](https://docs.pydantic.dev/latest/) and provides excellent validation
and IDE support for YAML linters by providing an extensive, detailed schema.

## Top-Level Structure

Here's the complete manifest structure with all available top-level sections:

/// mknodes
{{ "llmling_agent.models.manifest.AgentsManifest" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", as_listitem=False) }}
///

## Top-Level Sections

### `agents`

Dictionary of individual agent configurations. Each key is an agent identifier, and the value is the complete agent configuration.

See [Agent Configuration](./agent.md) for detailed agent setup options.

### `teams`  

Dictionary of team configurations for multi-agent workflows. Teams can run agents in parallel or sequence.

See [Team Configuration](./team.md) for team setup and coordination.

### `responses`

Dictionary of shared response type definitions that can be referenced by agents. Supports both inline schema definitions and imported Python types.

See [Response Configuration](./responses.md) for structured output setup.

### `storage`

Configuration for how agent interactions are stored and logged. Supports multiple storage providers.

See [Storage Configuration](./storage.md) for persistence and logging options.

### `observability`

Configuration for monitoring and telemetry collection.

### `conversion`

Settings for document and media conversion capabilities.

### `mcp_servers`

Global Model Context Protocol server configurations that provide tools and resources to agents.

See [MCP Configuration](./mcp.md) for server setup and integration.

### `pool_server`

Configuration for the pool server that exposes agent functionality to external clients.

### `prompts`

Global prompt library configuration and management.

See [Prompt Configuration](./prompts.md) for prompt organization.

### `commands`

Global command shortcuts that can be used across agents for prompt injection.

### `jobs`

Pre-defined reusable tasks with templates, knowledge requirements, and expected outputs.

See [Task Configuration](./tasks.md) for job definitions.

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
