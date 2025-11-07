# Structured Responses: Python vs YAML

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
```
Python-defined Response:
Main point: User expresses enthusiasm for new feature
Is positive: true

YAML-defined Response:
Sentiment: positive
Confidence: 0.95
Mood: excited
```

This demonstrates:

- Two ways to define structured outputs
- Validation and constraints
- Integration with type system
- Trade-offs between approaches
