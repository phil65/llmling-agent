# Signal System

LLMling-agent uses signals to enable observation and reaction to various events throughout the system. These signals are particularly useful for building UIs, monitoring agent behavior, and implementing custom event handling.

## Registry Events

Several components provide a standard set of signals for monitoring changes to their collections. These signals are accessible through the `events` property:

```python
# Available on any BaseRegistry
events.adding = Signal(object)                    # (key, ) - Before addition
events.added = Signal(object, object)             # (key, value) - After addition
events.changing = Signal(object)                  # (key, ) - Before change
events.changed = Signal(object, object, object)   # (key, old_value, value) - After change
events.removing = Signal(object)                  # (key, ) - Before removal
events.removed = Signal(object, object)           # (key, value) - After removal
```

### Components using BaseRegistry

#### AgentPool
Monitors agent additions and removals:
```python
def on_agent_added(name: str, agent: Agent):
    print(f"New agent {name} added to pool")

pool.events.added.connect(on_agent_added)
```

#### ToolManager
Tracks tool registration and modifications:
```python
def on_tool_changed(name: str, old_info: Tool, new_info: Tool):
    print(f"Tool {name} configuration changed")

agent.tools.events.changed.connect(on_tool_changed)
```

## Agent Signals

The main Agent class provides signals for monitoring core agent activities:

```python
message_received = Signal(ChatMessage[str])  # New message received
message_sent = Signal(ChatMessage)           # Agent sent a message
run_failed = Signal(str, Exception)         # Run operation failed
agent_reset = Signal(AgentReset)            # Agent state was reset
```

## Conversation Manager Signals

Signals for conversation state changes:

```python
history_cleared = Signal(HistoryCleared)  # Chat history was cleared
```

## Talk/Connection Signals

Signals for agent communication:

```python
message_received = Signal(ChatMessage)   # Original message
message_forwarded = Signal(ChatMessage)  # After transformation
node_connected = Signal(object)         # Message connection established
connection_added = Signal(Talk)          # New connection created
```

## Common Use Cases

### Building UIs
```python
class ConsoleUI:
    def __init__(self, agent: Agent):
        # Core agent activity
        agent.message_sent.connect(self.display_message)

        # Tool management
        agent.tools.events.added.connect(self.update_tool_display)
        agent.tools.events.changed.connect(self.refresh_tool_status)
```

### Monitoring and Logging
```python
def setup_monitoring(pool: AgentPool):
    # Monitor agent lifecycle
    pool.events.added.connect(lambda name, agent: logger.info("Agent added: %s", name))
    pool.events.removed.connect(lambda name, agent: logger.info("Agent removed: %s", name))


```

### Provider Model Changes
```python
def on_model_change(new_model):
    # Update provider-specific settings for new model
    update_configuration(new_model)

```
