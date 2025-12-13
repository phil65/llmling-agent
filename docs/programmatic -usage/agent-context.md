---
title: Agent Context
description: Agent context and state management
icon: material/folder-cog
---

## Overview

AgentContext provides a powerful way to access agent capabilities and dependencies in your tools:

1. Type-safe dependency injection
2. Access to shared resources and other agents
3. Capability validation and control
4. Storage and runtime integration

## Basic Usage

Tools can request context injection by adding an `AgentContext` parameter:

```python
async def my_tool(ctx: AgentContext[TDeps], arg: str) -> str:
    """Tool with access to context and dependencies."""
    # Access pool functionality
    if ctx.pool:
        helper = ctx.pool.get_agent("helper")
        await helper.run(arg)

    # Access typed dependencies
    config = ctx.data  # Type: TDeps
    return f"Processed {arg} with {config}"
```

## Type-Safe Dependencies

AgentContext enables type-safe dependency injection through its generic parameter:

```python
# Define dependencies
class AppConfig:
    api_key: str
    endpoint: str

# Create agent with typed context
agent = Agent[AppConfig](
    name="my_agent",
    deps=AppConfig(api_key="123", endpoint="api.example.com")
)
```

## Capabilities and Validation

Context enables tool validation through capabilities:

```python
async def execute_code(ctx: AgentContext, code: str) -> str:
    """Tool that requires specific capability."""
    if not ctx.capabilities.can_execute_code:
        raise ToolError("Missing required capability")

    # Execute code...
    return "Code executed"
```

!!! note "Provider Compatibility"
    When using the pydantic-ai provider, tools can alternatively request `RunContext[AgentContext]`
    for compatibility with pydantic-ai's context system. Both context types provide access to the
    same functionality.
