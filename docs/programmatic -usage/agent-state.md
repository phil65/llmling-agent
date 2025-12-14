---
title: Agent State
description: Agent state management
icon: material/database
---

# Managing Agent State in LLMling

An agent maintains several types of state that influence its behavior. While some state is configured during initialization through YAML or constructor arguments, it is also possible to inspect and modify it during runtime.

## Core State Components

- **System Prompts**: Define the agent's role and behavior guidelines
- **Tools**: Available functions and capabilities
- **Memory/History**: Past conversation context
- **Model**: The used LLM model
- **Connections**: Active connections, to whom the own results get forwarded to.

## Runtime State Management

```python
async with Agent(...) as agent:
    # Modify system prompts by reference
    agent.sys_prompts.add("technical_style")
    # Register tools
    agent.tools.register_tool(my_tool)
    # Change model
    agent.set_model("openai:gpt-5-nano")
    # Clear conversation history
    agent.conversation.clear()
    # Add context messages
    agent.conversation.add_context_message(
        "Important background information",
        source="knowledge_base"
    )
```

## Temporary State Changes

LLMling's context managers allow temporary state modifications that automatically restore the original state:

```python
async with agent.temporary_state(
    system_prompts=["Be very concise"],  # Temporary prompts
    replace_prompts=True,  # Replace instead of append
    tools=[callable_1, callable_2],  # Temporary tools
    replace_tools=True,  # Replace existing tools
    history=["Previous relevant chat"],  # Temporary history
    replace_history=True,  # Replace existing history
    pause_routing=True,  # Pause message routing
    model="openai:gpt-5",  # Temporary model
) as modified_agent:
    result = await modified_agent.run("Summarize this.")
    # Original state is restored after the block
```

## Memory Management

Control message history size and persistence:

```python
# No message history
agent = Agent(session=False)
# Keep max 1000 tokens in context window
agent = Agent(session=1000)
# Complex memory configuration
agent = Agent(session=MemoryConfig(
    max_messages=5,  # Keep last 5 messages
    max_tokens=1000,  # And stay under 1000 tokens
    enable=True      # Enable history logging
))

# Recover previous session
agent = Agent(session="previous_chat_id")
agent = Agent(session=SessionQuery(
    name="previous_chat",
    since="1h",  # Last hour's messages
    roles={"user", "assistant"}
))
```

## Safe State Sharing

When working with multiple agents, state can be shared in a controlled way:

```python
await source_agent.share(
    target_agent,
    tools=["useful_tool"],     # Share specific tools
    resources=["knowledge"],    # Share specific resources
    history=5,                 # Share last 5 messages
    token_limit=1000           # Limit shared history size
)
```

All these mechanisms ensure that agent state can be precisely controlled while maintaining clean separation between temporary and permanent changes. The context managers are particularly useful for running agents with modified behavior without affecting their base configuration.
