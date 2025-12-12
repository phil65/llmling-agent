---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/commands.py
title: Command Configuration
description: Slash command configuration
icon: material/slash-forward
---

# Command Configuration

Commands (slash commands) provide reusable prompt templates that can be invoked during conversations. They allow you to define frequently used prompts with parameters.

## Overview

Commands can be defined in three ways:

- **Static**: Inline command content with Jinja2 templating
- **File**: Load command content from external files
- **Callable**: Reference Python functions that generate command content

Commands are invoked with `/command-name` syntax and support parameter substitution.

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.commands.CommandConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Usage Notes

- Command names should be unique within an agent
- Commands support Jinja2 templating for dynamic content
- File-based commands can reference templates with variables
- Callable commands allow full programmatic control
- Commands are available in both CLI and web interfaces
- Parameters can be passed as key-value pairs when invoking commands
