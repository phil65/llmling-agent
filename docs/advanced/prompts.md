---
title: Prompts
description: Advanced prompt handling
icon: material/comment-multiple
---

# Prompts in AgentPool

AgentPool provides a flexible system for handling different types of prompts, allowing seamless conversion of various data types into prompt text.

## Prompt Types

The following types can be used as prompts:

- Strings: Plain text prompts
- Objects implementing `__prompt__`: Dynamic content providers
- Types implementing `__prompt_type__`: Type-level information
- Pydantic models: Structured data
- Base prompts: AgentPool prompt types (Static, Dynamic, File)
- Dictionaries: Key-value representations
- Lists/Tuples: Sequences of prompts
- Callables: Functions returning prompts
- Coroutines: Async functions returning prompts

## Prompt Protocols

### `__prompt__` Protocol

The `__prompt__` protocol allows objects to provide LLM-friendly descriptions of themselves. When implementing this method, follow these guidelines:

### Best Practices

**Be Descriptive Yet Concise**

```python
class DataAnalysis:
    def __prompt__(self) -> str:
        return f"Analysis of {self.dataset}: {self.key_findings[:200]}..."
        # ✓ Includes context and main content
        # ✓ Truncates long content
```

**Format for LLM Understanding**

```python
class APIEndpoint:
    def __prompt__(self) -> str:
        return f"""
        REST Endpoint: {self.path}
        Method: {self.method}
        Purpose: {self.description}
        """.strip()
        # ✓ Clear structure
        # ✓ Key information first
        # ✓ Human-readable format

```

### `__prompt_type__` Protocol

Types can implement the `__prompt_type__` classmethod to provide type-level information:

```python
class OutputType:
    @classmethod
    def __prompt_type__(cls) -> str:
        return "A structured response with fields: ..."
```

## Recursive Resolution

Prompts are resolved recursively:

```python
await agent.run([
    "Basic text",
    data_source,        # __prompt__ called
    OutputType,       # __prompt_type__ called
    {"key": "value"},   # Formatted as key-value
    my_prompt_func,     # Called to get result
])
```

All components are combined into a single coherent prompt.

## Supported in AgentPool

The prompt system is used throughout AgentPool:

- Agent.run() accepts any prompt type
- Agent instances implement `__prompt__`
- Context loading supports all prompt types
- Tasks can use any prompt type
- Knowledge sources can be any prompt type

For more examples and detailed API documentation, see the API reference.
