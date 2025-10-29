# StructuredAgent

The `StructuredAgent` is a typed wrapper for Agent that enforces structured output validation. It provides type safety and validation while maintaining all capabilities of the base Agent.

## Overview

```python
from llmling_agent import Agent, StructuredAgent
from pydantic import BaseModel

class AnalysisResult(BaseModel):
    sentiment: float
    topics: list[str]

# Create structured agent directly
agent = Agent(base_agent, output_type=AnalysisResult)

```

## Key Features

- Type-safe responses with Pydantic validation
- Full access to base Agent capabilities
- Generic typing for dependencies and result type
- Easy conversion through `to_structured()`

## Relationship to Agent

Unlike a traditional inheritance relationship, StructuredAgent acts as a container that wraps and enhances a base Agent. This design:

- Maintains clean separation of concerns
- Allows dynamic conversion between structured/unstructured
- Preserves the base Agent's interface while adding type safety


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


```

### Declarative Usage (YAML Configuration)

The usage of StructuredAgents in pure-YAML workflows is still in its infancy.
You can do it, but in the end there is no real "structured communication" happening
yet. If an Agent gets a BaseModel as its input, its getting formatted in a
Human/LLM-friendly way and processed as text input.

!!! warning
    Never mix these patterns by referencing manifest response definitions in programmatic code,
    as this breaks type safety.
    Always use concrete Python types when working programmatically with structured agents.

## Example scenarios

### 1. Type-safe Task Execution
```python
# Task requires specific return type
task = Task[None, AnalysisResult](...)
agent = base_agent.to_structured(AnalysisResult)
result = await task.execute(agent)
```

## Type Parameters

- `TDeps`: Type of dependencies (defaults to None)
- `TResult`: Type of structured output

## Implementation Details

Unlike the base Agent, StructuredAgent:

- Enforces result validation
- Provides type hints for responses
- Manages conversion between structured/unstructured data
