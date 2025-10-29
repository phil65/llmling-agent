# Base Agent

The basic Agent class is our core implementation of an Agent.
What sets this agent apart is that it has a concept of pluggable "providers". A provider can either be:

- An LLM ( currently backed by Pydantic-AI)
- A simple Python Callable which gets the prompt as well as the context object as input
- A Human! Right now, its simple, but providers will also be the "default" UI layer.
- A FastAPI server as a provider is also planned

In contrast to pydantic-ai, the Base Agent is only generic over its dependency type. Its only generic over its return type on a method scope. (using return_type=)
For an agent which is generic over both dependency and result type, see the StructuredAgent documentation.

Theres also much more to discover: Powerful options to manage agent state (like ToolManagers), MCP server support, Agent events, the list goes on!

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

# Tool and model signals
agent.tool_used.connect(handle_tool)
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

## Other Features

- `register_worker`: Turn an agent into a tool for another agent
- `to_tool`: Create a callable tool from the agent
- `set_model`: Change the model dynamically
- Various task execution and chain management methods

All methods maintain proper typing and integrate seamlessly with the rest of the LLMling ecosystem.
