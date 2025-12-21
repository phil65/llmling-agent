---
title: Events & Triggers
description: Event system and handlers
icon: material/broadcast
---

Events allow agents to respond to external changes and automate actions based on various triggers.
AgentPool supports different types of triggers that can activate agents automatically.

- File system changes
- Webhook calls
- Incoming emails
- Time-based triggers
- User interface actions
- Function calls (monitor functions using a decorator and get event each time it is called)

## Basic Concepts

Events flow through the system in this order:

1. Event Source detects change
2. Event Manager processes event
3. Agent receives and handles event
4. Optional: Agent forwards event results through connections

## Event Types

### File Watch Events

Monitor file system changes and trigger agent actions:ä

```yaml
agents:
  code_monitor:
    triggers:
      - type: "file"
        name: "python_watcher"        # Unique identifier
        enabled: true                 # Can be disabled without removal
        paths: ["src/**/*.py"]        # Glob patterns to watch
        extensions: [".py"]           # Optional file type filter
        ignore_paths:                 # Optional ignore patterns
          - "**/__pycache__"
          - "**/.git"
        recursive: true               # Watch subdirectories
        debounce: 1600               # Minimum ms between triggers
```

### Webhook Events

Listen for HTTP requests:

```yaml
agents:
  api_handler:
    triggers:
      - type: "webhook"
        name: "github_webhook"
        enabled: true
        port: 8000
        path: "/github"
        secret: "${WEBHOOK_SECRET}"   # Optional validation secret
```

## Event Configuration

### Common Properties

All event types share these base properties:

```yaml
triggers:
  - name: "my_trigger"               # Unique identifier
    enabled: true                    # Whether trigger is active
    knowledge:                       # Optional knowledge to load
      paths: ["context/*.md"]        # Files to load as context
      resources:                     # LLMling resources
        - type: "cli"
          command: "git status"
      prompts:                       # Context prompts
        - "Consider this background information..."
```

### Multiple Triggers

Agents can have multiple triggers of different types:

```yaml
agents:
  project_assistant:
    triggers:
      # Watch for code changes
      - type: "file"
        name: "code_watcher"
        paths: ["src/**/*.py"]
        extensions: [".py"]

      # Listen for GitHub webhooks
      - type: "webhook"
        name: "github_events"
        port: 8000
        path: "/github"

      # Manual review trigger
      - type: "manual"
        name: "review_code"
        prompt: "Review latest changes"
```

## Event Handling

### In Configuration

Configure how agents handle events through system prompts:

```yaml
agents:
  file_monitor:
    system_prompts:
      - |
        You monitor file changes and analyze their impact.
        When receiving file change events:
        1. Check file type and content
        2. Assess impact of changes
        3. Recommend actions if needed
    triggers:
      - type: "file"
        name: "config_watch"
        paths: ["config/*.yml"]
```

## Event Manager

Each agent has an event manager (`agent.events`) that handles:

- Event source lifecycle
- Event processing
- Callback management

```python
# Access event manager
manager = agent.events

# Add custom callback
async def on_event(event: EventData):
    print(f"Event received: {event}")

manager.add_callback(on_event)

# Remove callback
manager.remove_callback(on_event)
```

Events can be handled in two ways:

1. **Automatic Handling**: Events are automatically converted to agent runs using their `to_prompt()` method
2. **Custom Callbacks**: Custom event handlers for more control

#### Custom Event Handlers

Register custom callbacks to handle events:

```python
async def handle_events(event: EventData):
    match event:
        case FileEventData(type="modified", path=p) if p.endswith('.py'):
            await agent.run(f"Python file modified: {p}")
        case WebhookEventData(path="/github"):
            await agent.run(f"GitHub webhook received: {event.data}")

# Register callback
agent.events.add_callback(handle_events)

# Remove callback
agent.events.remove_callback(handle_events)
```

## CLI Usage

Run agents in event-watching mode to handle events:

```bash
# Start watching with default configuration
agentpool watch agents.yml

# With specific log level
agentpool watch agents.yml --log-level DEBUG
```

The agents will:

1. Start monitoring configured event sources
2. Handle events based on their configuration
3. Run until interrupted (Ctrl+C)

## Event Data

Each event type provides specific data:

### FileEvent

```python
@dataclass(frozen=True)
class FileEventData(EventData):
    path: str           # Path to affected file
    type: ChangeType    # "added" | "modified" | "deleted"
    timestamp: datetime # When event occurred
    source: str        # Trigger name
```

### WebhookEvent

```python
@dataclass(frozen=True)
class WebhookEventData(EventData):
    path: str          # Request path
    method: str        # HTTP method
    data: dict        # Request data
    timestamp: datetime
    source: str
```

## Function Monitoring

The EventManager provides two powerful decorators for monitoring function execution and creating periodic tasks:

### Track Function Calls

Monitor any function and get events when it's called:

```python
agent = Agent(...)

@agent.events.track("search_executed")
async def search_docs(query: str) -> list[Doc]:
    results = await search(query)
    return results  # Result becomes event content

# Track with additional metadata
@agent.events.track(
    "user_action",
    category="authentication",
    severity="high"
)
def user_login(username: str) -> bool:
    return auth.login(username)
```

The decorator:

- Creates an event for each function call
- Includes function result as event content
- Adds timing and error information
- Supports both sync and async functions
- Can include custom metadata

### Poll Functions

Execute functions periodically and handle their results as events:

```python
agent = Agent(...)

# Check every hour
@agent.events.poll("system_stats", hours=1)
async def check_system() -> SystemInfo:
    stats = await get_system_stats()
    return stats

# Check every 30 minutes with metadata
@agent.events.poll(
    "database_status",
    minutes=30,
    metadata={"critical": True}
)
def check_database() -> DBStatus:
    return db.get_status()
```

The decorator:

- Executes function at specified intervals
- Converts results to events
- Handles both sync and async functions
- Can include custom metadata
- Supports flexible time intervals

## Common Patterns

### Code Review Automation

```yaml
agents:
  code_reviewer:
    triggers:
      - type: "file"
        name: "pr_watch"
        paths: ["src/**/*.py"]
    system_prompts:
      - "You review Python code changes and provide feedback."
```

### Configuration Monitor

```yaml
agents:
  config_monitor:
    triggers:
      - type: "file"
        name: "config_watch"
        paths: ["config/*.yml"]
        debounce: 5000  # 5 second delay
```

### Mixed Handling

```python
# Custom handling for specific cases
async def special_handler(event: EventData):
    if isinstance(event, FileEventData) and event.path.endswith('.secret'):
        await handle_secret_file(event.path)

# Register custom handler but keep auto-handling
agent.events.add_callback(special_handler)
# Auto-handling will still process other events
```
