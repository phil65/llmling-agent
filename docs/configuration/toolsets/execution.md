---
title: Execution Environment Toolset
description: Execute code and commands
icon: material/console
---

# Execution Environment Toolset

The Execution Environment toolset provides tools for executing code and shell commands within a configured environment.

## Basic Usage

```yaml
agents:
  coder:
    toolsets:
      - type: execution
        environment:
          type: local
```

## Environment Types

The toolset supports various execution environments:

### Local

Execute on the local machine:

```yaml
toolsets:
  - type: execution
    environment:
      type: local
      cwd: /workspace
```

### Docker

Execute in a Docker container:

```yaml
toolsets:
  - type: execution
    environment:
      type: docker
      image: python:3.12
      volumes:
        /workspace: /app
```

### Remote

Execute on remote machines via SSH or other protocols.

## Available Tools

| Tool | Description |
|------|-------------|
| `execute_code` | Execute code in a specific language |
| `execute_command` | Run shell commands |
| `start_process` | Start a background process |
| `get_process_output` | Get output from a running process |
| `wait_for_process` | Wait for process completion |
| `kill_process` | Terminate a running process |

## Tool Selection

You can limit which tools are exposed:

```yaml
toolsets:
  - type: execution
    environment:
      type: local
    tools:
      - execute_command
      - execute_code
```

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.ExecutionEnvironmentToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///

## Security Considerations

- Use Docker or sandboxed environments for untrusted code
- Limit available tools to only what's needed
- Set appropriate working directories and permissions
