---
title: Knowledge System
description: Knowledge management and access
icon: material/book
---

## Overview

The Knowledge system in LLMling provides a way to give agents access to information sources. These can be files, resources, or prompts that provide context for the agent's operations.
The Resource types

## Structure

```python
class Knowledge:
    """Collection of context sources for an agent."""

    paths: list[str]
    """Quick access to files and URLs."""

    resources: list[Resource]
    """Rich resource definitions supporting:
    - PathResource: Complex file patterns, watching
    - TextResource: Raw content
    - CLIResource: Command output
    - RepositoryResource: Git repos
    - SourceResource: Python source
    - CallableResource: Function results
    """

    prompts: list[PromptType]
    """Prompts for dynamic content generation:
    - StaticPrompt: Fixed message templates
    - DynamicPrompt: Python function-based
    - FilePrompt: File-based with template support
    """

    convert_to_markdown: bool = False
    """Whether to convert content to markdown when possible."""
```

## YAML Configuration

```yaml
agents:
  researcher:
    # Knowledge configuration
    knowledge:
      # Simple paths
      paths:
        - "docs/**/*.md"
        - "https://api.example.com/docs"

      # Rich resources
      resources:
        - type: path
          path: "src/**/*.py"
          watch: true

        - type: text
          content: "Important context..."

        - type: cli
          command: "git log"

      # Context prompts
      prompts:
        - "Consider these guidelines..."
        - type: file
          path: "prompts/analysis.txt"
```

## Initialization

Knowledge sources are automatically loaded when the agent enters its async context:

```python
async with Agent(knowledge=knowledge) as agent:
    # Knowledge is now loaded and available in context
    await agent.run("Use the documentation...")
```

Knowledge initialization happens in parallel with other async setup (MCP servers, runtime) unless `parallel_init=False` is set.

!!! info
    This part will undergo significant refactor in the future and is subject to change. There will be adapters for LangChain's resources, and more.
