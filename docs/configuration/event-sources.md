---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/events.py
title: Event Sources
description: Event source configuration for triggering agent actions
icon: material/bell-alert
---

Event sources define external triggers that can start agent tasks or workflows. They allow agents to respond to file changes, webhooks, scheduled events, and connection events.

## Overview

LLMling-Agent supports multiple event source types that enable reactive agent workflows:

- **File Watch**: Trigger on file system changes
- **Webhook**: HTTP endpoint for external triggers  
- **Email**: Email-based triggers
- **Time**: Scheduled/periodic execution
- **Connection**: Trigger on agent connection events

Event sources are typically used with task configurations to create event-driven agent systems.

## Configuration Reference

### Event Sources

/// mknodes
{{ "llmling_agent_config.events.EventConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

### Connection Event Conditions

Connection events can be filtered using conditions:

/// mknodes
{{ "llmling_agent_config.events.ConnectionEventConditionType" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Usage with Tasks

Event sources are configured with tasks to create reactive workflows:

```yaml
tasks:
  monitor_logs:
    agent: log_analyzer
    event_source:
      type: file_watch
      paths: ["logs/*.log"]
      recursive: false
    parameters:
      analysis_level: "detailed"

  scheduled_backup:
    agent: backup_agent
    event_source:
      type: time
      schedule: "0 2 * * *"  # 2 AM daily
```

## Best Practices

### File Watching

- Use specific glob patterns to minimize events
- Set appropriate `ignore_patterns` to exclude temporary files
- Consider `recursive: false` for better performance if subdirectories aren't needed

### Webhooks

- Always use authentication for production webhooks
- Validate webhook payloads in your agent logic
- Use environment variables for sensitive tokens

### Email Events

- Use dedicated email accounts for automation
- Configure appropriate filters to avoid processing spam
- Consider rate limiting for high-volume scenarios

### Time Events

- Use cron syntax for recurring schedules
- Test schedules with online cron expression validators
- Consider timezone implications for scheduled tasks

### Connection Events

- Use specific source/target filters to reduce noise
- Combine with conditions for fine-grained control
- Monitor event frequency to avoid performance issues

## Configuration Notes

- Event sources can be shared across multiple tasks
- File watch and webhook events require the system to be running
- Email events poll at configurable intervals
- Connection events are processed in real-time
- Time events use system timezone unless specified
- All event sources support async operation
