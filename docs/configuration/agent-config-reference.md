---
title: Agent Config Reference
description: Complete agent configuration reference
icon: material/file-document
---

# Agent Configuration

Agent configurations are defined in YAML files and consist of three main sections: agents, responses, and roles.

## Agents Section

Complete example of an agent configuration:

```yaml
agents:
  web_assistant:                   # Name of the agent
    description: "Helps with web tasks"  # Optional description
    model: openai:gpt-5           # Model to use
    tools:
      open_browser:
        import_path: webbrowser.open
        description: "Opens URLs in browser"
    system_prompts:
      - "You are a web assistant."
      - "Use open_browser to open URLs."
    retries: 2                   # Number of retries for failed
```

## Field Reference

| Field Name | Description |
|------------|-------------|
| [`name`](agent.md#name) | Identifier for the agent (set from dict key, not from YAML) |
| [`config_file_path`](agent.md#config-file-path) | Config file path for resolving relative paths |
| [`display_name`](agent.md#display-name) | Human-readable display name for the agent |
| [`description`](agent.md#description) | Optional description of the agent |
| [`triggers`](event-sources.md) | Event sources that activate this agent |
| [`connections`](connections.md) | Targets to forward results to |
| [`mcp_servers`](mcp.md) | List of MCP server configurations |
| [`input_provider`](agent.md#input-provider) | Provider for human-input-handling |
| [`event_handlers`](observability.md#event-handlers) | Event handlers for processing agent stream events |
| [`inherits`](inheritance.md) | Name of agent config to inherit from |
| [`model`](model.md) | The model to use for this agent |
| [`tools`](tools.md) | A list of tools to register with this agent |
| [`toolsets`](toolsets.md) | Toolset configurations for extensible tool collections |
| [`session`](session.md) | Session configuration for conversation recovery |
| [`output_type`](responses.md) | Name of the response definition to use |
| [`retries`](agent.md#retries) | Number of retries for failed operations |
| [`output_retries`](agent.md#output-retries) | Max retries for result validation |
| [`end_strategy`](agent.md#end-strategy) | The strategy for handling multiple tool calls when a final result is found |
| [`avatar`](agent.md#avatar) | URL or path to agent's avatar image |
| [`system_prompts`](prompts.md) | System prompts for the agent |
| [`knowledge`](knowledge.md) | Knowledge sources for this agent |
| [`workers`](worker.md) | Worker agents which will be available as tools |
| [`requires_tool_confirmation`](agent.md#tool-confirmation) | How to handle tool confirmation (always/never/per_tool) |
| [`debug`](agent.md#debug) | Enable debug output for this agent |
| [`environment`](execution-environments.md) | Execution environment configuration for this agent |
| [`usage_limits`](agent.md#usage-limits) | Usage limits for this agent |
| [`tool_mode`](agent.md#tool-mode) | Tool execution mode (None/codemode) |
| [`auto_cache`](agent.md#auto-cache) | Automatic prompt caching configuration |
