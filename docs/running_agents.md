# Running Agents

This guide covers different ways to interact with LLMling agents.

## Command Line Interface

The `llmling-agent` command provides several subcommands:

```bash
# List available commands
llmling-agent --help

# Manage agent configurations
llmling-agent add my-agents agents.yml   # Add configuration
llmling-agent list                       # List available agents
llmling-agent set my-agents             # Set active configuration

# Run agents
llmling-agent run my-agent "What's the system status?"  # Single query
llmling-agent chat my-agent                             # Interactive chat
llmling-agent run my-agent -p check_system              # Use predefined prompt

# Multiple agents
llmling-agent run "agent1,agent2" "Analyze this"        # Run multiple agents

# View history
llmling-agent history show                              # Show recent conversations
llmling-agent history stats --group-by model            # Show usage statistics
```

Advanced CLI options:
```bash
# Override model
llmling-agent run my-agent --model openai:gpt-5-mini "Complex query"

# Output formatting
llmling-agent run my-agent --output-format json "Query"

# Environment selection
llmling-agent run my-agent --environment prod.yml "Query"

# Debug mode
llmling-agent chat my-agent --debug
```

## Interactive Chat Sessions

Chat sessions provide an interactive interface with enhanced features:

```bash
$ llmling-agent chat my-agent
Started chat with my-agent
Available tools: 5 (5 enabled)

You: What tools can you use?
Assistant: I can use these tools:
- read_file: Read file contents
- analyze_code: Run code analysis
- get_stats: Get system statistics
```

## Web Interface

Launch the web interface for a graphical experience:

```bash
# Start web interface
llmling-agent launch

# Custom server settings
llmling-agent launch --host 0.0.0.0 --port 8000

# Create shareable link
llmling-agent launch --share
```

The web interface provides:
- File upload for configurations
- Agent selection and management
- Real-time chat with streaming responses
- Tool state visualization
- Conversation history
- Debug logging

## Programmatic Usage

Use agents in your Python code:

```python
from llmling_agent import Agent

# Basic usage
async with Agent(...) as agent:

    # Basic query
    result = await agent.run("What's the system status?")
    print(result.data)

    # Streaming responses
    async for event in agent.run_stream("Long analysis..."):
        print(chunk)  # pydantic-ai event

    # Managing tools
    await agent.tools.enable_tool("search")  # Enable specific tool
    await agent.tools.disable_tool("edit")   # Disable specific tool
    print(await agent.tools.list_tools())    # See all tools and their states

    # Working with conversation history
    history = agent.conversation.get_history()  # Get current history
    agent.conversation.clear()  # Clear history

    # Add context for next message
    await agent.conversation.add_context_from_file("data.txt")

```

For more details, see:
- [CLI Reference](https://phil65.github.io/llmling-agent/cli-reference.html)
- [Web Interface Guide](https://phil65.github.io/llmling-agent/web-interface.html)
- [API Documentation](https://phil65.github.io/llmling-agent/api-reference.html)
