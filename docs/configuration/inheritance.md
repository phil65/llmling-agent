---
title: Inheritance
description: Configuration inheritance system
icon: material/source-merge
---

AgentPool supports inheritance both for individual agents and entire YAML files, making configurations more reusable and maintainable.

## Agent Inheritance

Agents can inherit configuration from other agents using the `inherits` field:

```yaml
agents:
  # Base agent configuration
  base_assistant:
    model: openai:gpt-5
    system_prompt: "You are a helpful assistant."
    toolsets:
      - type: resource_access

  # Specialized agent inheriting base config
  code_assistant:
    inherits: base_assistant  # Inherit from base
    description: "Specializes in code review"
    system_prompt: "Focus on code quality and best practices."
    toolsets:  # Extends base toolsets
      - type: code_execution

  # Another specialized agent
  docs_assistant:
    inherits: base_assistant
    description: "Specializes in documentation"
    system_prompt: "Focus on clear documentation."
```

Child agents:

- Inherit all fields from parent
- Can override any inherited field
- Can add new fields
- System prompts are combined

## YAML File Inheritance

Using Yamling's inheritance system, entire YAML files can inherit from other files:


=== "Base config"

    ```yaml title="base.yml"
    agents:
      base_agent:
        model: openai:gpt-5
        toolsets:
          - type: resource_access
    
    storage:
      providers:
        - type: sql
          url: sqlite:///history.db
    ```

=== "Specialized config"

    ```yaml title="agents.yml"
    INHERIT: base.yml  # Inherit entire base configuration
    
    agents:
      specialized_agent:
        inherits: base_agent
        description: "Specialized version"
    ```


### Remote File Inheritance

Yamling supports UPath, allowing inheritance from remote files:

```yaml
# Inherit from remote sources
INHERIT:
  - base.yml
  - https://example.com/base_config.yml
  - s3://my-bucket/configs/agents.yml
  - git+https://github.com/org/repo/config.yml
```

## Inheritance Resolution

### Agent Inheritance

1. Start with parent configuration
2. Recursively resolve parent inheritances
3. Apply child configuration:
   - Override simple fields
   - Merge lists (e.g., system_prompts)
   - Update dictionaries (e.g., capabilities)

### YAML File Inheritance

1. Load all inherited files in order
2. Merge configurations:
   - Later files override earlier ones
   - Lists and dictionaries are merged
   - Complex fields use smart merging

Inheritance in AgentPool helps maintain DRY (Don't Repeat Yourself) configurations while allowing for flexible specialization at both the agent and file level.
