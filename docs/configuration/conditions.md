---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/conditions.py
title: Conditions Configuration
description: Conditional logic for message flow and lifecycle control
icon: material/source-branch
---

Conditions control message flow, connection lifecycle, and process termination. They provide fine-grained control over when messages are processed, when connections stop, and when the system should exit.

## Overview

Conditions can be used at three control levels:

- **Filter Condition**: Controls which messages pass through a connection
- **Stop Condition**: Triggers disconnection of a specific connection
- **Exit Condition**: Stops the entire process (raises SystemExit)

LLMling-Agent supports various condition types:

- **Word Match**: Check for specific words or phrases in messages
- **Message Count**: Control based on number of messages processed
- **Time**: Control based on elapsed time duration
- **Token Threshold**: Monitor and limit token usage
- **Cost Limit**: Control based on accumulated costs
- **Callable**: Custom Python functions for complex logic
- **Jinja2**: Template-based conditions with full context access
- **AND/OR**: Composite conditions combining multiple checks

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.conditions.Condition" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Control Levels

### Filter Condition

Controls which messages pass through the connection:

```yaml
connections:
  - type: node
    name: summarizer
    filter_condition:
      type: word_match
      words: ["summarize", "summary"]
      mode: "any"
```

### Stop Condition

Triggers disconnection of this specific connection:

```yaml
connections:
  - type: node
    name: expensive_model
    stop_condition:
      type: cost_limit
      max_cost: 1.0
```

### Exit Condition

Stops the entire process by raising SystemExit:

```yaml
connections:
  - type: node
    name: critical_processor
    exit_condition:
      type: token_threshold
      max_tokens: 10000
```

## Composite Conditions

Combine multiple conditions using AND/OR logic:

```yaml
# All conditions must be met
filter_condition:
  type: and
  conditions:
    - type: word_match
      words: ["important"]
    - type: cost_limit
      max_cost: 1.0

# Any condition can be met
stop_condition:
  type: or
  conditions:
    - type: message_count
      max_messages: 10
    - type: time
      duration: 300
```

## Context Access

All conditions have access to `EventContext` which provides:

- `message`: Current ChatMessage being processed
- `target`: MessageNode receiving the message
- `stats`: Connection statistics (message count, tokens, cost, timing)
- `registry`: All named connections
- `talk`: Current connection object

This context is available in Jinja2 templates and callable conditions.

## Best Practices

### Condition Hierarchy

- Use **filter conditions** for routine message control
- Use **stop conditions** for graceful connection termination
- Reserve **exit conditions** for critical system-wide issues

### Cost Control

```yaml
connections:
  - type: node
    name: expensive_agent
    stop_condition:
      type: cost_limit
      max_cost: 1.0
    exit_condition:
      type: cost_limit
      max_cost: 5.0
```

### Safety

Combine multiple safeguards:

```yaml
exit_condition:
  type: or
  conditions:
    - type: cost_limit
      max_cost: 10.0
    - type: token_threshold
      max_tokens: 50000
    - type: time
      duration: 3600
```

## Configuration Notes

- Conditions are evaluated for each message
- Simple conditions (word_match, message_count) have minimal overhead
- Callable and Jinja2 conditions support async operations
- Composite conditions (AND/OR) evaluate sub-conditions efficiently
- Statistics are accumulated throughout the connection lifetime
- Cost tracking requires model configurations with pricing information
