# LLMling Agent CLI

The LLMling Agent CLI provides a comprehensive set of commands to manage and interact with AI agents.
It's designed around the concept of an "active agent file" - a YAML configuration that defines your agents and their settings.
This will avoid the need to pass the config file path each time you want to run a command.

## Active Agent File

The CLI maintains an "active agent file" setting which determines which agents are available for commands like `run`, `chat`, or `watch`.
You can:

- Set the active file with `llmling-agent set <path>`
- Add new agent files with `llmling-agent add <path>`
- List available files with `llmling-agent list`

Most commands will use the currently active agent file by default, but can be overridden with the `--config` option.

## Available Commands

### Agent Management

- `add` - Register a new agent configuration file
- `set` - Set the active configuration file
- `list` - Show available agent configurations

### Interaction

- `chat` - Start an interactive chat session with an agent
- `run` - Execute a one-off command with an agent
- `task` - Run pre-defined tasks from your configuration

### Monitoring & History

- `watch` - Monitor file changes and run agents on updates
- `history` - View and manage conversation history
  - `history list` - Show recent conversations
  - `history show` - Display conversation details
  - `history clear` - Clear conversation history

### User Interface
- `launch` - Start the web interface
- `web` - Web interface related commands (requires `llmling-agent[ui]`)

### Setup
- `quickstart` - Create a new agent configuration with guided setup

## Working with Agent Files

A typical workflow might look like:

1. Create a new agent configuration:
   ```bash
   llmling-agent quickstart
   ```

2. Set it as the active configuration:
   ```bash
   llmling-agent set agents.yml
   ```

3. Start chatting with an agent:
   ```bash
   llmling-agent chat analyzer
   ```

The active agent file is stored in your user configuration and persists between sessions.
You can have multiple agent configurations and switch between them as needed.

## Configuration Files

Agent configurations are YAML files that define:

- Available agents and their capabilities
- System prompts and knowledge sources
- Tool configurations
- Response types
- And more

Example:

```yaml
agents:
  analyzer:
    name: "Text Analyzer"
    model: openai:gpt-5
    description: "Analyzes text and provides structured output"
    toolsets:
      - type: file_access
```

See the [Configuration Guide](../config/index.md) for detailed information about agent configuration.
