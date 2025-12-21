---
title: Commands
description: Slash commands for agent interaction
icon: material/console
---

# Commands

AgentPool provides a consistent command system across both CLI and web interfaces. Commands are prefixed with `/` and can take arguments and options.

## Essential Commands

### Session Management

```bash
/exit
```

Exit the current chat session cleanly.

### Tool Management

```bash
# List available tools
/list-tools

# Enable/disable specific tools
/enable-tool <name>
/disable-tool <name>

# Show detailed tool information
/tool-info <name>

# Register new tools
/register-tool <import_path> [--name custom_name] [--description "Tool description"]
Example: /register-tool json.dumps --name format_json --description "Format JSON data"

# Write and register tools directly in chat
/write-tool
```

### Resource Management

```bash
# List available resources
/list-resources

# Show resource content
/show-resource <name> [--param1 value1] [--param2 value2]
Example: /show-resource config.yml
Example: /show-resource template --date today
```

### Agent Configuration

```bash
# Show current agent configuration
/show-agent

# Switch to different agent
/switch-agent <name>

# List available agents
/list-agents

# Change model
/set-model <model>
Example: /set-model gpt-5
```

## Additional Commands

### Environment

```bash
# Change environment file
/set-env <path>

# Open environment configuration
/open-env-file
```

### History and Stats

```bash
# Clear chat history
/clear

# Reset session state
/reset

# Search conversation history
/search-history [query] [--hours N] [--limit N]

# Show usage statistics
/show-statistics [--group-by model|agent|hour|day] [--hours N]
```

### Utility Commands

```bash
# Copy last assistant message
/copy-clipboard

# Open agent configuration file
/open-agent-file

# Show help
/help [command]
```

## Command Categories

### Core Operations

- `/help` - Show available commands or help for specific command
- `/exit` - Exit chat session
- `/clear` - Clear chat history
- `/reset` - Reset session state

### Tool Management

- `/list-tools` - List available tools
- `/enable-tool` - Enable a tool
- `/disable-tool` - Disable a tool
- `/tool-info` - Show tool details
- `/register-tool` - Register new tool
- `/write-tool` - Create new tool interactively

### Resource Management

- `/list-resources` - List available resources
- `/show-resource` - Show resource content

### Agent Management

- `/show-agent` - Show current configuration
- `/switch-agent` - Switch to different agent
- `/list-agents` - List available agents
- `/set-model` - Change model

### Environment

- `/set-env` - Change environment file
- `/open-env-file` - Open environment config

### History and Analysis

- `/search-history` - Search conversation history
- `/show-statistics` - Show usage statistics

### Utilities

- `/copy-clipboard` - Copy last response
- `/open-agent-file` - Open config file

## Command Line Conventions

- Commands start with forward slash (`/`)
- Arguments can be positional or named:

  ```bash
  /command arg1 arg2              # Positional
  /command --name value           # Named
  /command arg --name "value"     # Mixed
  ```

- Arguments with spaces must be quoted
- Options are prefixed with `--`

## Command Output

Commands provide feedback in different ways:

- Direct responses (success/failure messages)
- Formatted data (tables, JSON, YAML)
- System messages in chat
- Status updates

> **Note**: Some commands may be restricted based on agent capabilities and role configuration.
