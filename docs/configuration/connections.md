---
title: Connections & Message Forwarding
description: Message routing and connection configuration
icon: material/arrow-right
---

# Connections

Connections define how agents automatically forward their messages to other agents or destinations.
They allow creating agent pipelines and communication patterns.

## Target Types

### Agent Forwarding
Forward messages to another agent:
```yaml
agents:
  analyzer:
    connections:
      - type: node
        name: "summarizer"              # Target agent name
        connection_type: "run"          # How to handle messages
        wait_for_completion: true       # Wait for target to finish
```

### File Output
Save messages to files with customizable formatting:
```yaml
agents:
  logger:
    connections:
      - type: "file"
        path: "logs/{date}/{agent}.txt"  # Supports variables
        wait_for_completion: true        # Wait for write to complete
        encoding: "utf-8"                # File encoding
        template: |                      # Custom format template
          [{timestamp}] {name}: {content}
```

### Callable Processing
Process messages through a Python function:
```yaml
agents:
  processor:
    connections:
      - type: "callable"
        callable: "myapp.process.handle_message"  # Import path
        kw_args:                                 # Additional kwargs
          format: "json"
          validate: true
```

## Connection Types
The `connection_type` field determines how messages are processed:

### "run"
Executes the message as a new prompt in the target agent. The target will process it as if it received a new user request:
```yaml
connections:
  - type: node
    name: "summarizer"
    connection_type: "run"          # Process as new prompt
```

### "context"
Adds the message to target's conversation context. This provides background information without triggering immediate processing:
```yaml
connections:
  - type: node
    name: "writer"
    connection_type: "context"      # Add as context
```

### "forward"
Simply forwards the message to the target's outbox. Useful for observation or logging:
```yaml
connections:
  - type: node
    name: "logger"
    connection_type: "forward"      # Pass through
```

## Shared Configuration Options

All target types support these configuration options:

### Basic Control
```yaml
connections:
  - type: node  # or "file" or "callable"
    wait_for_completion: true    # Wait for target to complete before continuing
    queued: false               # Queue messages for batch processing
    queue_strategy: "latest"    # How to handle queued messages:
                               # - "latest": Keep only most recent
                               # - "concat": Combine all messages
                               # - "buffer": Process all individually
    priority: 0                # Task priority (lower = higher priority)
    delay: "5s"               # Wait before processing message
```

### Message Flow Control
Control when and how messages are processed:
```yaml
connections:
  - type: node
    name: "expensive_model"

    # Transform messages before processing
    transform: "myapp.transform.process"

    # Only pass messages meeting condition
    filter_condition:
      type: "word_match"
      words: ["analyze"]

    # Disconnect when condition met
    stop_condition:
      type: "cost_limit"
      max_cost: 0.50

    # Exit entire process when condition met
    exit_condition:
      type: "token_threshold"
      max_tokens: 10000
```

### Queue Processing
When `queued: true`, messages are held for batch processing:

- `queue_strategy: "latest"`: Only keeps and processes the most recent message
- `queue_strategy: "concat"`: Combines all queued messages into one
- `queue_strategy: "buffer"`: Processes all messages individually in order

### Wait States
The `wait_for_completion` option affects process flow:

- `true`: Source agent waits for target to complete processing
- `false`: Source continues immediately after forwarding

This is important for:
- Maintaining message order
- Ensuring sequential processing
- Managing dependencies between agents

## Complex Example
Multiple connections showing different configurations:
```yaml
agents:
  analyzer:
    connections:
      # Standard processing with conditions
      - type: node
        name: "summarizer"  # agent name
        connection_type: "run"
        wait_for_completion: true
        filter_condition:
          type: "word_match"
          words: ["summarize"]
        stop_condition:
          type: "message_count"
          max_messages: 10
        exit_condition:
          type: "cost_limit"
          max_cost: 5.0

      # Queued processing with transform
      - type: "callable"
        callable: "myapp.log.process_message"
        queued: true
        queue_strategy: "concat"
        transform: "myapp.transform.preprocess"
        kw_args:
          format: "json"

      # Async file logging
      - type: "file"
        path: "logs/{date}/analysis_{time}.txt"
        wait_for_completion: false
        template: |
          Time: {timestamp}
          Agent: {name}
          Content: {content}
          Cost: ${cost_info.total_cost}

      # Context addition with delay
      - type: node
        name: "reviewer"
        connection_type: "context"
        wait_for_completion: false
        delay: "1m"
        priority: 2
```

## Path Variables
Available variables for file paths:
- `{date}`: Current date (YYYY-MM-DD)
- `{time}`: Current time (HH-MM-SS)
- `{agent}`: Name of the source agent

Example:
```yaml
connections:
  - type: "file"
    path: "logs/{date}/{agent}_{time}.txt"
```
