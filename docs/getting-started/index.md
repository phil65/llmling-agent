---
title: Installation
description: Installation instructions for LLMling-Agent
---

# Installation

## Basic Installation

Simple installation

```bash
uv tool install llmling-agent[default]
```

Multiple extras

```bash
uv tool install llmling-agent[default, coding]
```

## Available Extras

/// mknodes
{{ "extras"| MkDependencyGroups }}
///


### One-Line ACP Setup

No installation needed - run directly with uvx:

```bash
uvx --python 3.13 llmling-agent[default]@latest serve-acp 

# or

uvx --python 3.13 llmling-agent[default]@latest serve-acp path/to/agents.yml
```
