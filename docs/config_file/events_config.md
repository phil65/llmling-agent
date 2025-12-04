---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/events.py
---

# Events Configuration

Events (or "triggers") define automated activations of agents based on various sources. They allow agents to respond to:
- File system changes
- Webhook calls
- Email notifications
- Time-based schedules
- Connection events between agents

## Common Properties

All event sources share these base properties:

```yaml
triggers:
  - name: "my_trigger"                   # Unique identifier
    type: "file"                         # Event type (required)
    enabled: true                        # Whether trigger is active
    template: <a_jinja_template_string>  # Optional event formatting
    include_metadata: true               # Include event metadata
    include_timestamp: true              # Include event timestamp
```

## File Watch Events

Monitor file system changes:

```yaml
triggers:
  - type: "file"
    name: "python_watcher"
    paths: ["src/**/*.py"]           # Paths to watch (required)
    extensions: [".py"]              # Optional file type filter
    ignore_paths: ["**/__pycache__"] # Optional ignore patterns
    recursive: true                  # Watch subdirectories
    debounce: 1600                   # Minimum ms between triggers
```

## Webhook Events

Listen for HTTP requests:

```yaml
triggers:
  - type: "webhook"
    name: "github_webhook"
    port: 8000                       # Port to listen on (required)
    path: "/github"                  # URL path to handle requests (required)
    secret: "${WEBHOOK_SECRET}"      # Optional validation secret
```

## Time Events

Schedule regular agent actions:

```yaml
triggers:
  - type: "time"
    name: "daily_report"
    schedule: "0 9 * * 1-5"          # Cron expression (required)
    prompt: "Generate daily report"  # Prompt to send (required)
    timezone: "UTC"                  # Optional timezone (defaults to system)
    skip_missed: false               # Whether to skip missed executions
```

## Email Events

Monitor email inbox:

```yaml
triggers:
  - type: "email"
    name: "support_inbox"
    host: "imap.gmail.com"          # IMAP server hostname (required)
    port: 993                       # Server port (993 for SSL)
    username: "support@domain.com"  # Email account username (required)
    password: "${EMAIL_PASSWORD}"   # Account password (required)
    folder: "INBOX"                 # Mailbox to monitor
    ssl: true                       # Use SSL connection
    check_interval: 60              # Seconds between checks
    mark_seen: true                 # Mark processed emails as seen
    filters:                        # Optional email filters
      from: "important@client.com"
      subject: "urgent"
    max_size: 1048576              # Max email size in bytes
```

## Connection Trigger Events

Monitor events between connections:

```yaml
triggers:
  - type: "connection"
    name: "process_completed"
    source: "analyzer"              # Optional: source agent name
    target: "summarizer"            # Optional: target agent name
    event: "message_processed"      # Required: event type to trigger on
    condition:                      # Optional: condition to filter events
      type: "content"
      words: ["complete", "finished"]
      mode: "any"                   # "any" or "all"
```

### Connection Event Types

Available connection event types:
- `message_received`: Triggered when a message is received
- `message_processed`: Triggered when a message is processed
- `message_forwarded`: Triggered when a message is forwarded
- `queue_filled`: Triggered when a message queue is filled
- `queue_triggered`: Triggered when a queued message is processed

### Connection Event Conditions

You can filter connection events with conditions:

#### Content Condition

```yaml
condition:
  type: "content"
  words: ["important", "urgent"]
  mode: "any"  # Match any word (or "all" to require all words)
```


## Multiple Events

Agents can have multiple events of different types:

```yaml
agents:
  project_assistant:
    triggers:
      # Watch for code changes
      - type: "file"
        name: "code_watcher"
        paths: ["src/**/*.py"]

      # Daily schedule
      - type: "time"
        name: "daily_summary"
        schedule: "0 9 * * 1-5"
        prompt: "Summarize yesterday's changes"

      # Monitor support inbox
      - type: "email"
        name: "support"
        host: "imap.gmail.com"
        username: "${EMAIL_USER}"
        password: "${EMAIL_PASS}"

      # Listen for webhooks
      - type: "webhook"
        name: "github_events"
        port: 8000
        path: "/github"
        
      # Monitor connections
      - type: "connection"
        name: "process_completed" 
        event: "message_processed"
        source: "analyzer"
```

## Event Processing

1. Event source detects change and creates event
2. Event's core content is extracted
3. Template wraps content with optional metadata/timestamp
4. Formatted event is sent to agent for processing
5. Agent processes event as a new request or chat message