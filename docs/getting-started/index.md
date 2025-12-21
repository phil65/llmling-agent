---
title: Installation
description: Installation instructions for AgentPool
order: 1
---

# Installation

## Basic Installation

Simple installation

```bash
uv tool install agentpool[default]
```

Multiple extras

```bash
uv tool install agentpool[default, coding]
```

## Available Extras

/// mknodes
{{ "extras"| MkDependencyGroups }}
///


### One-Line ACP Setup

No installation needed - run directly with uvx:

```bash
uvx --python 3.13 agentpool[default]@latest serve-acp 

# or

uvx --python 3.13 agentpool[default]@latest serve-acp path/to/agents.yml
```
