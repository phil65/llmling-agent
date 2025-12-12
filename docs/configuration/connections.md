---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/forward_targets.py
title: Connections Configuration
description: Message routing and forwarding configuration
icon: material/arrow-right
---

Connections define how messages are automatically forwarded from agents to other destinations. They enable agent pipelines, parallel processing, and message routing patterns.

## Overview

Connections allow you to:

- Forward messages between agents for sequential processing
- Save messages to files with custom formatting
- Process messages through custom Python functions
- Create agent pipelines and workflows
- Implement complex message routing patterns

LLMling-Agent supports three connection types:

- **Node Connection**: Forward to another agent, team, or node
- **File Connection**: Save messages to files with templating
- **Callable Connection**: Process through Python functions

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.forward_targets.ForwardingTarget" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Connection Types

### Node Connection

Forward messages to other agents or teams:

```yaml
agents:
  analyzer:
    connections:
      - type: node
        name: "summarizer"
        connection_type: "run"
        wait_for_completion: true
```

Connection types:

- `run`: Execute target agent with the message
- `queue`: Add message to target's queue
- `forward`: Simple message forwarding

### File Connection

Save messages to files with Jinja2 templating:

```yaml
agents:
  logger:
    connections:
      - type: file
        path: "logs/{date}/{agent}.txt"
        template: "[{timestamp}] {name}: {content}"
        encoding: "utf-8"
```

### Callable Connection

Process messages through Python functions:

```yaml
agents:
  processor:
    connections:
      - type: callable
        callable: "myapp.handlers:process_message"
        kw_args:
          format: "json"
```

## Message Flow Control

### Wait for Completion

Control whether to wait for the connection to complete:

```yaml
connections:
  - type: node
    name: "worker"
    wait_for_completion: false  # Async, fire-and-forget
```

### Priority

Control processing order when multiple messages are queued:

```yaml
connections:
  - type: node
    name: "worker"
    priority: 1  # Lower numbers = higher priority
```

### Delay

Add delays before processing:

```yaml
connections:
  - type: node
    name: "worker"
    delay: 5.0  # Wait 5 seconds before forwarding
```

## Conditions

Apply conditions to control when messages are forwarded:

```yaml
connections:
  - type: node
    name: "expensive_agent"
    filter_condition:
      type: word_match
      words: ["urgent", "important"]
    stop_condition:
      type: cost_limit
      max_cost: 1.0
```

See [Conditions Configuration](conditions.md) for details on available condition types.

## Queue Management

Control message queueing behavior:

```yaml
connections:
  - type: node
    name: "worker"
    connection_type: "queue"
    queue_max_size: 100
    queue_trigger_size: 10  # Process when queue reaches this size
```

## Configuration Notes

- Connections are evaluated in the order they are defined
- Multiple connections can be active simultaneously
- File paths support variable substitution (date, agent name, etc.)
- Callable connections can be sync or async functions
- Queue connections allow batching messages for efficiency
- Conditions provide fine-grained control over message flow
- Priority affects order of processing when multiple messages are queued
