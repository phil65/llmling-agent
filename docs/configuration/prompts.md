---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/llmling_agent_config/system_prompts.py
title: System Prompts Configuration
description: System prompt configuration and library
icon: material/text-box
---

# System Prompts Configuration

## Overview

LLMling's prompt library allows defining reusable system prompts that can be shared across agents. Prompts are defined in the `prompts` section of your configuration and can be referenced by name.

System prompts define agent behavior, personality, and methodology. You can configure prompts using:

- **Static prompts**: Inline text content
- **File prompts**: Load from external files with Jinja2 templating
- **Library prompts**: Reference shared prompts from the library
- **Function prompts**: Dynamically generate prompts using Python functions

## Basic Structure

```yaml
prompts:
  # Define reusable system prompts
  system_prompts:
    expert_analyst:
      content: |
        You are an expert data analyst.
        Focus on finding patterns and insights.
      category: role

    step_by_step:
      content: |
        Break tasks into sequential steps.
        Explain each step thoroughly.
      category: methodology

# Using prompts in agents
agents:
  analyst:
    system_prompts:
      # Direct string prompts
      - "You help with analysis."

      # Reference library prompts
      - type: library
        reference: expert_analyst
      - type: library
        reference: step_by_step
```

## Prompt Categories

System prompts can be categorized by their purpose:

- **Role**: Define WHO the agent is (e.g., "expert developer", "data scientist")
- **Methodology**: Define HOW the agent works (e.g., "step-by-step", "analytical")
- **Tone**: Define communication STYLE (e.g., "professional", "friendly")
- **Format**: Define output STRUCTURE (e.g., "markdown", "structured")

## Configuration Reference

/// mknodes
{{ "llmling_agent_config.system_prompts.PromptConfig" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Complete Example

```yaml
prompts:
  system_prompts:
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
    system_prompts:
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
  system_prompts:
    technical_expert:
      category: role
      content: ...

# prompts/styles.yml
prompts:
  system_prompts:
    professional:
      category: tone
      content: ...

# agents.yml
INHERIT:
  - prompts/roles.yml
  - prompts/styles.yml

agents:
  my_agent:
    system_prompts:
      - type: library
        reference: technical_expert
      - type: library
        reference: professional
```

### Naming Conventions

Use clear, descriptive names:

- **Roles**: `expert_analyst`, `code_reviewer`, `technical_writer`
- **Methodologies**: `step_by_step`, `analytical`, `iterative`
- **Tones**: `professional`, `friendly`, `formal`, `casual`
- **Formats**: `markdown`, `structured`, `bullet_points`
