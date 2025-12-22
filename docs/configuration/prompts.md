---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool_config/system_prompts.py
title: System Prompts
description: System prompt configuration and library
icon: material/text-box
---

## Overview

AgentPool's prompt library allows defining reusable system prompts that can be shared across agents. Prompts are defined in the `prompts` section of your configuration and can be referenced by name.

System prompts define agent behavior, personality, and methodology. You can configure prompts using:

- **Static prompts**: Inline text content
- **File prompts**: Load from external files with Jinja2 templating
- **Library prompts**: Reference shared prompts from the library
- **Function prompts**: Dynamically generate prompts using Python functions

## Basic Structure

```yaml
--8<-- "docs/configuration/prompts_example.yml"
```

## Prompt Categories

System prompts can be categorized by their purpose:

- **Role**: Define WHO the agent is (e.g., "expert developer", "data scientist")
- **Methodology**: Define HOW the agent works (e.g., "step-by-step", "analytical")
- **Tone**: Define communication STYLE (e.g., "professional", "friendly")
- **Format**: Define output STRUCTURE (e.g., "markdown", "structured")

## Configuration Reference

/// mknodes
{{ "agentpool_config.system_prompts.PromptConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Complete Example

```yaml
prompts:
  system_prompt:
    # Role definitions
    technical_expert:
      category: role
      content: |
        You are a technical expert specializing in:
        - Software development best practices
        - System architecture and design
        - Code review and quality assurance

    # Methodology definitions
    systematic:
      category: methodology
      content: |
        Follow this systematic approach:
        1. Understand requirements fully
        2. Break down complex problems
        3. Apply best practices consistently
        4. Validate results thoroughly

    # Tone definitions
    professional:
      category: tone
      content: |
        Maintain professional communication:
        - Use formal, precise language
        - Be respectful and constructive
        - Provide clear explanations

agents:
  senior_dev:
    model: gpt-4
    system_prompt:
      - "Specialize in Python and TypeScript development."
      - type: library
        reference: technical_expert
      - type: library
        reference: systematic
      - type: library
        reference: professional
      - type: file
        path: "prompts/coding_style.j2"
```

## Organization Best Practices

### File Structure

Keep prompts organized in separate files:

```yaml
# prompts/roles.yml
prompts:
  system_prompt:
    technical_expert:
      category: role
      content: ...

# prompts/styles.yml
prompts:
  system_prompt:
    professional:
      category: tone
      content: ...

# agents.yml
INHERIT:
  - prompts/roles.yml
  - prompts/styles.yml

agents:
  my_agent:
    system_prompt:
      - type: library
        reference: technical_expert
      - type: library
        reference: professional
```
