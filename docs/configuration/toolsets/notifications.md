---
title: Notifications Toolset
description: Send notifications via various channels
icon: material/bell
---

# Notifications Toolset

Send notifications through various channels like email, Slack, or webhooks.

## Basic Usage

```yaml
agents:
  notifier:
    toolsets:
      - type: notifications
        email:
          smtp_host: smtp.gmail.com
          smtp_port: 587
          username: ${EMAIL_USER}
          password: ${EMAIL_PASS}
```

## Supported Channels

- Email (SMTP)
- Slack
- Webhooks
- Desktop notifications

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.NotificationsToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
