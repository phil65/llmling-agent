---
title: Basic Agent
description: Core Agent implementation and features
icon: material/account
---

## Core Interface

### Running Queries

The agent provides three main ways to execute queries:

```python
# Basic async run
result = await agent.run(
    "What is 2+2?",
    output_type=int,  # Optional type for structured responses (and a generic type)
    deps=my_deps,     # Optional dependencies
    model="gpt-5-mini"     # Optional model override
)

# Streaming responses
async for event in agent.run_stream("Count to 10"):
    print(chunk)  # Pydantic-AI event

# Synchronous wrapper (convenience)
result = agent.run.sync("Hello!")
```

### Conversation Management

The agent maintains conversation history and context through its `conversation` property:

```python
# Access conversation manager
agent.conversation.add_context_message("Important context")
history = agent.conversation.get_history()
agent.conversation.clear()
```

### Tool Management

Tools are managed through the `tools` property:

```python
# Register a tool
agent.tools.register_tool(my_tool)
tools = await agent.tools.get_tools()
```

## Signals

The agent emits various signals that can be connected to:

```python
# Message signals
agent.message_sent.connect(handle_message)
agent.message_received.connect(handle_message)
```

## Continuous Operation

Agents can run continuously:

```python
# Run with static prompt
await agent.run_in_background(
    "Monitor the system",
    interval=60,
    max_count=10
)

# Run with dynamic prompt
def get_prompt(ctx):
    return f"Check status of {ctx.data.system}"

await agent.run_in_background(get_prompt)
```

## Agents with output types

```python
from agentpool import Agent
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: float
    topics: list[str]

# Create structured agent directly
agent = Agent(base_agent, output_type=AnalysisResult)

```

## YAML Configuration

When defining structured agents in YAML, use the `output_type` field to specify either an inline response definition or reference a shared one:

```yaml
agents:
  analyzer:
    provider:
       type: pydantic_ai
       model: openai:gpt-5
    output_type: AnalysisResult  # Reference shared definition
    system_prompts:
      - You analyze text and provide structured results.

  validator:
    provider:
       type: pydantic_ai
       model: openai:gpt-5
    output_type:  # Inline definition
      type: inline
      fields:
        is_valid:
          type: bool
          description: "Whether the input is valid"
        issues:
          type: list[str]
          description: "List of validation issues"

responses:
  AnalysisResult:
    response_schema:
      type: inline
      description: "Text analysis result"
      fields:
        sentiment:
          type: float
          description: "Sentiment score between -1 and 1"
        topics:
          type: list[str]
          description: "Main topics discussed"
```

## Important Note on Usage Patterns

There are two distinct ways to use structured agents, which should not be mixed:

### Programmatic Usage (Type-Safe)

```python
class AnalysisResult(BaseModel):
    sentiment: float
    topics: list[str]

agent = Agent(..., output_type=AnalysisResult)
```

### Declarative Usage (YAML Configuration)

The usage of Agents with structured outputs in pure-YAML workflows is still in its infancy.
You can do it, but in the end there is no real "structured communication" happening
yet. If an Agent gets a BaseModel as its input, its getting formatted in a
Human/LLM-friendly way and processed as text input.

!!! warning
    Never mix these patterns by referencing manifest response definitions in programmatic code,
    as this breaks type safety.
    Always use concrete Python types when working programmatically with structured agents.
