---
title: Conditions
description: Condition configurations
icon: material/source-branch
---

# Connection Conditions

LLMling offers a powerful condition system for controlling message flow, connection lifecycle, and process termination. These conditions can be defined both in YAML configuration and programmatically.

## Condition Types

### Word Match

Checks for specific words or phrases in messages:

```yaml
filter_condition:
  type: word_match
  words: ["analyze", "process"]    # Words to match
  case_sensitive: false           # Ignore case
  mode: "any"                     # Match any word ("all" for all words)
```

### Message Count

Controls based on number of messages:

```yaml
filter_condition:
  type: message_count
  max_messages: 5                 # Maximum messages to allow
  count_mode: "total"            # Count all messages
  # or "per_agent" to count separately for each agent
```

### Time Based

Controls based on elapsed time:

```yaml
filter_condition:
  type: time
  duration: 300                  # Maximum time in seconds
```

### Token Threshold

Monitors token usage:

```yaml
filter_condition:
  type: token_threshold
  max_tokens: 1000              # Maximum tokens allowed
  count_type: "total"          # Count all tokens
  # or "prompt"/"completion" for specific token types
```

### Cost Limit

Controls based on accumulated costs:

```yaml
filter_condition:
  type: cost_limit
  max_cost: 0.50               # Maximum cost in USD
```

## Composite Conditions

Combine multiple conditions using AND/OR logic:

### AND Condition

All conditions must be met:

```yaml
filter_condition:
  type: and
  conditions:
    - type: word_match
      words: ["important"]
    - type: cost_limit
      max_cost: 1.0
```

### OR Condition

Any condition can be met:

```yaml
filter_condition:
  type: or
  conditions:
    - type: message_count
      max_messages: 10
    - type: time
      duration: 300
```

## Control Levels

LLMling provides three levels of control through conditions:

### 1. Filter Condition

Controls which messages pass through the connection:

```yaml
connections:
  - type: node
    name: summarizer
    filter_condition:
      type: word_match
      words: ["summarize"]
```

Programmatically:

```python
talk.when(lambda msg: "summarize" in msg.content)
```

### 2. Stop Condition

Triggers disconnection of this specific connection:
```yaml
connections:
  - type: node
    name: expensive_model
    stop_condition:
      type: cost_limit
      max_cost: 0.50
```

Programmatically:

```python
agent.connect_to(
    other,
    stop_condition=lambda ctx: ctx.message.metadata.get("complete", False)
)
```

### 3. Exit Condition

Stops the entire process by raising SystemExit:

```yaml
connections:
  - type: node
    name: critical_processor
    exit_condition:
      type: token_threshold
      max_tokens: 10000
```

Programmatically:

```python
agent.connect_to(
    other,
    exit_condition=lambda msg: msg.metadata.get("emergency", False)
)
```

## Available Statistics

Conditions have access to connection statistics through TalkStats:

```python
@dataclass(frozen=True)
class TalkStats:
    message_count: int          # Total messages processed
    token_count: int           # Total tokens used
    total_cost: float          # Total cost in USD
    byte_count: int           # Total message size
    start_time: datetime      # When connection started
    last_message_time: datetime | None
```


### Jinja2 Template
Flexible conditions using Jinja2 templates with full context access:
```yaml
filter_condition:
  type: jinja2
  template: # your jinja temlate
```

You have access to the context of the current connection through `ctx`.
return `True` to pass the condition.

## Context Access

All conditions receive an EventContext with complete state access:

```python
@dataclass(frozen=True)
class EventContext:
    """Context for condition checks."""
    message: ChatMessage[Any]     # Current message
    target: MessageNode          # Target receiving message
    stats: TalkStats            # Connection statistics
    registry: ConnectionRegistry # All named connections
    talk: Talk                  # Current connection
```

## Custom Conditions

Create custom conditions using callable functions:

```python
async def check_rate_limit(message: ChatMessage[Any]) -> bool:
    """Custom condition checking external rate limit."""
    rate = await redis.get_rate(message.name)
    return rate < MAX_RATE

# Use as any condition type
agent.connect_to(
    other,
    filter_condition=check_rate_limit  # or stop_condition/exit_condition
)
```

## Best Practices

### 1. Condition Hierarchy
- Use filter conditions for routine message control
- Use stop conditions for graceful connection termination
- Reserve exit conditions for critical system-wide issues

### 2. Cost Control
```yaml
connections:
  - type: node
    name: gpt5_mini_agent
    # Stop this connection if too expensive
    stop_condition:
      type: cost_limit
      max_cost: 1.0
    # Exit process on extreme cost
    exit_condition:
      type: cost_limit
      max_cost: 5.0
```

### 3. Performance
- Simple conditions for high-frequency filtering
- Composite conditions for complex logic
- Async conditions for external checks

### 4. Safety
```yaml
connections:
  - type: node
    name: assistant
    # Monitor token usage
    filter_condition:
      type: token_threshold
      max_tokens: 1000
    # Emergency exit on excessive cost
    exit_condition:
      type: or
      conditions:
        - type: cost_limit
          max_cost: 10.0
        - type: token_threshold
          max_tokens: 50000
```

## Complete Example

```yaml
agents:
  analyzer:
    connections:
      - type: node
        name: expensive_processor
        connection_type: run

        # Only process relevant messages
        filter_condition:
          type: and
          conditions:
            - type: word_match
              words: ["analyze", "process"]
            - type: token_threshold
              max_tokens: 500
              count_type: "prompt"

        # Stop connection when cost limit reached
        stop_condition:
          type: or
          conditions:
            - type: cost_limit
              max_cost: 1.0
            - type: message_count
              max_messages: 50

        # Emergency exit conditions
        exit_condition:
          type: or
          conditions:
            - type: cost_limit
              max_cost: 5.0
            - type: token_threshold
              max_tokens: 10000
              count_type: "total"
```
