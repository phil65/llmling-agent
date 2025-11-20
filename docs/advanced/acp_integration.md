# ACP Integration

## What is ACP?

The Agent Client Protocol (ACP) is a standardized JSON-RPC 2.0 protocol that enables communication between code editors and AI agents. It allows llmling-agent to integrate seamlessly with desktop applications and IDEs that support the protocol.

ACP provides:
- Bidirectional communication between editor and agent
- Session management and conversation history
- File system operations with permission handling
- Terminal integration for command execution
- Support for multiple agents with mode switching

## Installation & Setup

Using uvx for one-off usage:

```bash
uvx --python 3.13 llmling-agent[default]@latest serve-acp --help
```

## CLI Usage

### Basic Commands

Start an ACP server from a configuration file:

```bash
llmling-agent serve-acp agents.yml
```


### Available Options

- `--file-access/--no-file-access`: Enable file system operations (default: enabled)
- `--terminal-access/--no-terminal-access`: Enable terminal integration (default: enabled)
- `--session-support/--no-session-support`: Enable session loading (default: enabled)
- `--model-provider`: Specify model providers to search (can be repeated)
- `--show-messages`: Show message activity in logs
- `--log-level`: Set logging level (debug, info, warning, error)

## IDE Configuration

### Zed Editor

Add this configuration to your Zed `settings.json`:

```json
{
  "agent_servers": {
    "LLMling": {
      "command": "uvx",
      "args": [
        "--python",
        "3.13",
        "llmling-agent[default]@latest",
        "serve-acp",
        "https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/src/llmling_agent_docs/examples/pick_experts/config.yml", # <- insert your agent config here
        "--model-provider",
        "openai"
      ],
      "env": {
        "OPENAI_API_KEY": "your-api-key-here"
      }
    }
  }
}
```

This configuration:

- Uses uvx to run the latest version without local installation
- Points to a remote configuration file with multiple expert agents
- Specifies OpenAI as the model provider
- Sets the required API key via environment variables

### Other IDEs

For IDEs that support ACP, the general pattern is:
1. Set the command to `llmling-agent` (or `uvx llmling-agent[default]@latest`)
2. Add `serve-acp` as the first argument
3. Specify your configuration file path
4. Add any desired CLI options
5. Set required environment variables (API keys, etc.)

## Multi-Agent Modes

When your configuration includes multiple agents, the IDE will show a mode selector allowing users to switch between different agents mid-conversation.

Example configuration with multiple agents:

```yaml
agents:
  code_reviewer:
    name: "Code Reviewer"
    model: "openai:gpt-4"
    system_prompt: "You are an expert code reviewer..."

  documentation_writer:
    name: "Documentation Writer"
    model: "anthropic:gpt-5-nano"
    system_prompt: "You are a technical documentation expert..."

```

Each agent appears as a separate "mode" in the IDE interface, allowing users to:

- Switch between specialized agents for different tasks
- Maintain separate conversation contexts per agent
- Access agent-specific capabilities and tools

## Configuration

### Remote Configurations

You can reference remote configuration files directly:

```bash
llmling-agent serve-acp https://example.com/config.yml
```


### Provider Selection

Limit which providers are searched for models:

```bash
llmling-agent serve-acp config.yml --model-provider openai --model-provider anthropic
```

If not provider is passed, OpenRouter is used.

Available model providers:

- `anthropic`
- `groq`
- `mistral`
- `openai`
- `openrouter`
- `github`
- `copilot`
- `cerebras`
- `gemini`
- `cohere`
- `deepseek`
- `requesty`
- `xai`
- `comet`
- `novita`
- `vercel-gateway`
- `chutes`
- `cortecs`
- `azure`
- `fireworks-ai`
- `ollama`
- `moonshotai`
- `zai`

The providers depend on environment variables being set, like `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, etc.
