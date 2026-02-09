---
sync:
  agent: doc_sync_agent
  dependencies:
    - src/agentpool_config/loaders.py
title: Resources
description: Resource loader configuration
icon: material/file-document
---

Resources provide dynamic content that can be accessed by agents during execution. They allow agents to fetch data from various sources like files, CLI commands, source code, and external systems.

## Overview

AgentPool supports multiple resource types:

- **Path**: Load content from file paths with pattern matching
- **Text**: Static text content with optional templating
- **CLI**: Execute command-line tools and capture output
- **Source**: Extract source code from Python modules and classes
- **LangChain**: Integration with LangChain document loaders
- **Callable**: Custom Python functions that return content

Resources are loaded on-demand when agents request them, supporting parameterization for dynamic content generation.

## Configuration Reference

/// mknodes
{{ "agentpool_config.loaders.Resource" | union_to_markdown(display_mode="yaml", header_style="pymdownx") }}
///

## Key Features

- **On-demand loading**: Resources are loaded only when requested
- **Parameterization**: Pass parameters to resources for dynamic content
- **Caching**: Optional caching to improve performance
- **Pattern matching**: Use glob patterns to load multiple files
- **Content processing**: Transform and filter content before delivery
- **Access control**: Restrict resource access through capabilities

## Use Cases

- **Documentation**: Provide agents with access to project documentation
- **Code analysis**: Give agents access to source code for review or modification
- **Data access**: Load configuration files, datasets, or API responses
- **Dynamic content**: Generate content based on current state or parameters
- **External integration**: Fetch data from external systems and tools

## Configuration Notes

- Resources can be defined at manifest level (global) or agent level (local)
- Path resources support glob patterns for batch loading
- CLI resources execute in the system shell with security considerations
- Source resources automatically extract docstrings and type hints
- LangChain resources leverage the extensive LangChain loader ecosystem
- Callable resources provide maximum flexibility for custom logic

## Dynamic Instructions from Resource Providers

ResourceProviders can now provide dynamic instructions that are re-evaluated on each agent run. This allows providers to generate context-aware instructions based on runtime state.

### How It Works

ResourceProviders can implement the `get_instructions()` method to return instruction functions:

```python
from agentpool.resource_providers import ResourceProvider
from agentpool.prompts.instructions import InstructionFunc
from agentpool.agents.context import AgentContext

class MyProvider(ResourceProvider):
    async def get_instructions(self) -> list[InstructionFunc]:
        """Return dynamic instruction functions."""
        return [
            self._get_static_instruction,      # No context
            self._get_context_instruction,     # With AgentContext
        ]

    def _get_static_instruction(self) -> str:
        """Instruction without context access."""
        return "Always be helpful."

    async def _get_context_instruction(self, ctx: AgentContext) -> str:
        """Instruction with context access."""
        return f"Agent: {ctx.name}, Model: {ctx.model_name}"
```

### YAML Configuration

Configure providers to provide instructions using the `instructions` field:

```yaml
agents:
  my_agent:
    type: native
    model: openai:gpt-4o
    toolsets:
      - type: custom
        import_path: myapp.providers.MyProvider
        name: my_provider

    # Add provider-based instructions
    instructions:
      - type: provider
        ref: my_provider
```

### Instruction Function Types

Instruction functions can accept different context types:

- **No context**: `() -> str`
- **AgentContext only**: `(AgentContext) -> str`
- **RunContext only**: `(RunContext) -> str`
- **Both contexts**: `(AgentContext, RunContext) -> str`

```python
# No context
def simple() -> str:
    return "Be helpful."

# AgentContext only
async def with_agent(ctx: AgentContext) -> str:
    return f"Agent: {ctx.name}"

# RunContext only
async def with_run(ctx: RunContext) -> str:
    return f"Model: {ctx.model.model_name}"

# Both contexts
async def with_both(agent_ctx: AgentContext, run_ctx: RunContext) -> str:
    return f"Agent {agent_ctx.name} using {run_ctx.model.model_name}"
```

### Benefits

- **Context-aware**: Instructions adapt to runtime state (conversation history, tools used, etc.)
- **Per-run re-evaluation**: Unlike static prompts, dynamic instructions regenerate on each run
- **Provider integration**: Toolsets and other providers can inject their own contextual instructions
- **Flexible context access**: Choose what context you need (AgentContext, RunContext, or both)

### Error Handling

If an instruction function fails:
- Error is logged with context
- Agent initialization continues
- Failed instruction is skipped (uses empty string fallback)

### See Also

- [Dynamic Instructions Example](../../examples/dynamic-instructions/)
- [ResourceProvider Base Class](../api/resource_providers.md)
- [Instruction Types](../api/instructions.md)
