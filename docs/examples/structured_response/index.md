---
title: Structured Responses
description: Structured response type examples
icon: material/code-json
hide:
  - toc
---

# Structured Responses

/// mknodes
{{ ['docs/examples/structured_response/main.py', 'docs/examples/structured_response/config.yml'] | pydantic_playground }}
///

: Python vs YAML
This example demonstrates two ways to define structured responses in LLMling-agent:

- Using Python Pydantic models
- Using YAML response definitions
- Type validation and constraints
- Agent integration with structured outputs


## How It Works

1. Python-defined Responses:

- Use Pydantic models
- Full IDE support and type checking
- Best for programmatic use
- Inline field documentation

2. YAML-defined Responses:

- Define in configuration
- Include validation constraints
- Best for configuration-driven workflows
- Self-documenting fields

Example Output:

This demonstrates:

- Two ways to define structured outputs
- Validation and constraints
- Integration with type system
- Trade-offs between approaches
