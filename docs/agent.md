# Agent

LLMling-agent provides an agent implementation based on pydantic-ai that integrates with the [LLMling](https://gitub.com/phil65/llmling) resource and tool system. The agent can be used standalone or as part of a larger application.

## Basic Usage

config.yml
``` yaml
tools:
  open_browser:
    import_path: "webbrowser.open"
  analyze:
    import_path: "module.some_tool_for_analysis"

resources:
  llmling_agent_manual:
    type: path
    path: "https://raw.githubusercontent.com/phil65/llmling_agent/refs/heads/main/README.md"
    description: "LLMling Agent Manual"
```


Create and use an agent:

```python
from llmling import RuntimeConfig
from llmling_agent import Agent
from pydantic import BaseModel


    # Create agent with string output. It will have all resources and tools available from the config.
async with Agent(
        runtime="config.yml",
        model="openai:gpt-5",
        system_prompt="You are a helpful assistant."
    ) as basic_agent:
    await basic_agent.run("Open google for me.")  # Uses tool to open browser
    # Create agent with structured output


    # Define return type
    class Analysis(BaseModel):
        summary: str
        suggestions: list[str]


    typed_agent = Agent[Any, Analysis](
        runtime,
        output_type=Analysis,
        model="openai:gpt-5",
        system_prompt=[
            "You are a code analysis assistant.",
            "Always provide structured results.",
        ]
    )
    result = await typed_agent.run("Analyze this code.")
    print(result.data.summary)         # Typed access
    print(result.data.suggestions)     # Type-checked
```

## Agent Configuration

The agent can be configured with various options:

```python
agent = Agent(
    runtime,
    # Model settings
    model="openai:gpt-5",            # Model to use
    output_type=Analysis,            # Optional result type

    # Prompt configuration
    system_prompt=[                  # Static system prompts
        "You are an assistant.",
        "Be concise and clear.",
    ],
    name="code-assistant",          # Agent name

    # Execution settings
    retries=3,                      # Max retries
)
```

## Running the Agent

Different ways to run the agent:

```python
# Basic run
result = await agent.run("Analyze this code.")

# Stream responses
async for event in agent.run_stream("Analyze this."):
    print(event)

# Synchronous operation (convenience wrapper)
result = agent.run.sync("Quick question")
```
