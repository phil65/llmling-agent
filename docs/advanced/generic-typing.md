---
title: Generic Typing
description: Advanced type system features
icon: lucide/braces
---

# Generic Typing in LLMling Agent

## Origins and Evolution

LLMling Agent's type system builds upon PydanticAI's foundation while extending it for multi-agent scenarios.
While PydanticAI provides a solid base for single-agent operations with typed dependencies and responses,
LLMling adds typing support for agent pools, teams, and inter-agent communication.

Key differences from PydanticAI:

- Pool-level dependency typing
- Team-level type preservation
- Generic typing for agent communication

## Core Generic Types

### Dependencies (TDeps)

LLMling inherits PydanticAI's dependency system to provide type-safe context for tool execution.
Dependencies are defined at agent construction time and flow through the entire tool system.
For detailed understanding of the dependency system, refer to PydanticAI's documentation.

### Agent Types

```python
class Agent[TDeps, TResult]:
    """Base agent with typed dependencies."""


# Base agent creation and access
agent1 = pool.get_agent("name1")  # Agent[GlobalDeps]
agent2 = pool.get_agent("name2", deps=CustomDeps())  # Agent[CustomDeps]
structured = pool.get_agent("name3", return_type=AnalysisResult)  # Agent[GlobalDeps, AnalysisResult]

# Team creation through & operator
team1 = agent1 & agent2  # Team[GlobalDeps] NOT POSSIBLE - falls back to Any
team2 = agent1 & structured  # Team[Any] - mixed Agents

# Explicit team creation
team3 = Team([agent1, agent1.clone()])  # Team[GlobalDeps] - same deps preserved
team4 = Team([agent1, agent2])  # Team[Any] - mixed deps
```

### Pool Typing

```python
pool = AgentPool[GlobalDeps]()

# Different ways to get agents
basic = pool.get_agent("name")  # Agent[GlobalDeps]
basic_2 = pool.get_agent("name")  # Agent[GlobalDeps]
custom = pool.get_agent("name", deps=CustomDeps())  # Agent[CustomDeps]
structured = pool.get_agent("name", return_type=AnalysisResult)  # Agent[GlobalDeps, AnalysisResult]
custom_structured = pool.get_agent(
    "name",
    deps=CustomDeps(),
    return_type=AnalysisResult
)  # Agent[CustomDeps, AnalysisResult]

# Group creation preserves types when possible
group1 = pool.create_team([basic, basic_2])  # Team[GlobalDeps]
group2 = pool.create_team([basic, custom])  # Team[Any]
```

Unlike PydanticAI's unified Agent class, LLMling splits these concerns into two classes.
This separation more has historical reasons since pydantic-ai didnt support per-run return types.
It might get re-evaluated in the future, but right now I cant spot any bigger downsides doing it this way.

The bridge between YAML configuration and code is the `get_agent` method, which enables type-safe agent creation:

```python
# Type-safe programmatic usage
agent = pool.get_agent("name", return_type=MyResponseType)
# Results in Agent[MyDeps, MyResponseType]
```

For type safety, programmatic usage with explicit return_type is preferred over YAML response type definitions.
You will still get an Agent with a response type when you assign a return_type via YAML,
but the linter will only recoginize the return type a as
a BaseModel since the type is generated dynamically.

## Multi-Agent Type System

### Pool Typing

```python
class AgentPool[TSharedDeps]:
    """Pool with optional shared dependencies."""
```
The pool can provide shared dependencies to its agents while still allowing individual agents to specify their own dependency types through `get_agent`.

### Team Type Management

```python
# Same dependency type - preserved through the team
team = Team[MyDeps]([agent1, agent2])  # all Agent[MyDeps]

# Mixed dependency types - must use Any due to type system limitations
team = Team[Any]([agent1: Agent[Deps1], agent2: Agent[Deps2]])
```

Type information is preserved when all team members share the same dependency type. When mixing agents with different dependency types,
we must fall back to Any due to the limitations of the type system in representing mixed dependencies.

### Communication Types
```python
class Talk[TTransmittedData]:
    """Type-safe communication channel between agents."""
```

The Talk class is generically typed over the transmitted message content, providing IDE support for type checking agent communications.
While currently primarily serving type hint purposes, this typing lays the groundwork for future "Structured Connections" - a system for type-safe,
structured communication between agents. (We already do this now already, but since our Agent currently takes any BaseModel input and converts it to LLM-readable form,
this doesnt yet guarantee "real" type-safe connections based on specific BaseModels subclasses)


### And more!

Response types, small decision / result dataclasses and much more should also preserve types to maxium extent.
This should provide great help static type checkers.
