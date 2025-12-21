---
title: Semantic Memory Toolset
description: Vector-based semantic memory
icon: material/brain
---

# Semantic Memory Toolset

Vector-based semantic memory for storing and retrieving information based on meaning.

## Basic Usage

```yaml
agents:
  my_agent:
    toolsets:
      - type: semantic_memory
```

## Features

- Store information with semantic embeddings
- Retrieve by similarity search
- Persistent memory across sessions

## Configuration Reference

/// mknodes
{{ "agentpool_config.toolsets.SemanticMemoryToolsetConfig" | schema_to_markdown(display_mode="yaml", header_style="pymdownx", wrapped_in="toolsets") }}
///
