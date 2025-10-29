# Callables as First-Class Citizens

In LLMling, regular Python functions (callables) are treated as first-class citizens and can be used interchangeably with agents in most contexts. This enables seamless integration of custom processing functions into agent workflows.

## Direct Agent Creation

You can create agents directly from callables, preserving their type information:

```python
# Simple untyped function becomes Agent[None, str]
def process(message: str) -> str:
    return message.upper()

agent = Agent.from_callback(process)

# Typed function becomes Agent[None, ResultType]
def analyze(message: str) -> AnalysisResult:
    return AnalysisResult(sentiment=0.8, topics=["AI"])

agent = Agent.from_callback(analyze)
```

## Automatic Conversion in Workflows

Callables are automatically converted to agents when used in:

### Teams (using `&`)
```python
def analyze(text: str) -> AnalysisResult:
    return AnalysisResult(...)

def summarize(text: str) -> str:
    return "Summary: " + text

# Both functions become agents in the team
team = analyzer & analyze & summarize
```

### Pipelines (using `|`)
```python
# Functions become agents in the pipeline
pipeline = agent | analyze | summarize
```

### Connections (using `>>`)
```python
# Function becomes agent when used as target
agent >> analyze
```

## Type Preservation

The return type of typed callables is preserved when converting to agents:

```python
class AnalysisResult(BaseModel):
    sentiment: float
    topics: list[str]

def analyze(text: str) -> AnalysisResult:
    return AnalysisResult(sentiment=0.5, topics=["tech"])

agent1 = Agent(analyze, output_type=AnalysisResult)

# Type is preserved in teams/pipelines
team = base_agent & analyze  # analyze becomes Agent[None, AnalysisResult]
```

## Context Awareness

Functions can optionally accept agent context:

```python
def process(ctx: AgentContext, message: str) -> str:
    # Access agent capabilities, configuration, etc.
    return f"Processed by {ctx.node_name}: {message}"

# Context is automatically injected
agent = Agent(provider=process)
```

This seamless integration of callables allows you to:

- Mix and match agents with regular functions
- Create lightweight processing steps without full agent overhead
- Preserve type safety throughout the workflow
- Gradually convert functions to full agents as needed


## Callables for prompts

LLMling-Agent also allows to pass Callables for system and user prompts which can get re-evaluted
for each run.

```python

def my_system_prompt(ctx: AgentContext) -> str:  # context optional
    return "You are an AI assistant."

agent = Agent(system_prompts=[my_system_prompt])

agent.run("Hello, how are you?")

# or:

agent.run(my_user_prompt)
```
