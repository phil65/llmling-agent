---
title: Tools and Tool Management
description: Tool management and registration
icon: material/tools
---

# Tool Manager in LLMling

## Overview

The `ToolManager` is a central component that manages the lifecycle and availability of tools for agents. It handles:
- Tool registration and validation
- Tool enabling/disabling
- Tool filtering and sorting
- MCP server integration
- Temporary tool contexts
- Tool state management

## Core Responsibilities

### 1. Tool Registration and Access

```python
from llmling_agent.tools import ToolManager



# Register a simple function as tool
agent.tools.register_tool(
    my_function,
    name_override="custom_name",     # Optional name override
    description_override="Does X",    # Optional description
    enabled=True,                    # Initial state
    source="source",                # Tool source
    requires_confirmation=False,     # Whether to confirm execution
    metadata={"custom": "data"}      # Additional metadata
)

# Get available tools
tools = await agent.tools.get_tools(state="enabled")  # filter options
# Get individual tool info
tool_info = manager["tool_name"]  # Dict-like access
```

### 2. Temporary Tool Contexts

One of the most powerful features is the ability to temporarily modify available tools:

```python
# Temporarily add tools
async with agent.tools.temporary_tools([tool1, tool2]) as temp_tools:
    # tool1 and tool2 are available in addition to the agents tools
    ...

# Temporarily add tools and disable all others
async with agent.tools.temporary_tools([special_tool], exclusive=True):
    # Only special_tool is available here
    ...

# Original tool states are restored after context exit
```

## Tool Registration Methods

### Basic Function Registration

```python
def my_tool(text: str) -> str:
    """Process some text.

    Args:
        text: Text to process
    """
    return text.upper()

# Register as tool
tool_info = manager.register_tool(my_tool)
tool_info = manager.register_tool("path.to.callable")

@agent.tools.tool
def some_tool(text: str) -> str:
    ...
```

### Worker Agent Registration

```python
# Register another agent as a tool
tool_info = manager.register_worker(
    worker_agent,
    name="worker_tool",                  # Optional name override
    reset_history_on_run=True,           # Clear history between runs
    pass_message_history=False,          # Share conversation history
)
```

### MCP Server Integration

```python
# Setup MCP server tools
await agent.tools.setup_mcp_servers([
    StdioMCPServerConfig(command="python", args=["-m", "my_server"]),
    SSEMCPServerConfig(url="http://localhost:3001/events")
])
```
Note: By default the tool manager is already initialzed with MCP server tools from the config when the agent itself enters the async context


## Tool Information

Each registered tool provides rich metadata through `Tool`:

```python
tool_info = manager["tool_name"]

print(tool_info.name)                 # Tool name
print(tool_info.description)          # Tool description
print(tool_info.enabled)              # Current state
print(tool_info.source)               # Where tool came from
print(tool_info.requires_confirmation)  # Whether to confirm
print(tool_info.metadata)             # Custom metadata
print(tool_info.parameters)           # Parameter information
```


### Capability-Based Tools

```python
# Register tool with capability requirement
manager.register_tool(
    sensitive_operation,
    requires_confirmation=True
)
```
