---
title: Structured Responses
description: Structured response type examples
icon: material/code-json
---


## Try it in Pydantic Playground




## Files

- `config.yml`
- `main.py`



# Structured Responses: Python vs YAML


### `config.yml`

```yml
# yaml-language-server: $schema=https://raw.githubusercontent.com/phil65/llmling-agent/refs/heads/main/schema/config-schema.json
responses:
  YamlResult:
    response_schema:
      type: inline
      description: "Sentiment analysis result"
      fields:
        sentiment:
          type: str
          description: "Overall sentiment"
          constraints:
            enum: ["positive", "negative", "neutral"]
        confidence:
          type: float
          description: "Confidence score"
          constraints:
            ge: 0.0
            le: 1.0
        mood:
          type: str
          description: "Detected mood"
          constraints:
            min_length: 3
            max_length: 20

agents:
  yaml_analyzer:
    display_name: "YAML-defined Analyzer"
    model: openai:gpt-5-mini
    system_prompts:
      - |
        Analyze text for sentiment and mood.
        Always respond with a structured response containing:
        - sentiment (positive/negative/neutral)
        - confidence (0-1)
        - mood (descriptive word)
    output_type: YamlResult # Use YAML-defined type

  python_analyzer:
    display_name: "Python-defined Analyzer"
    model: openai:gpt-5-mini
    system_prompts:
      - |
        Analyze text and extract key points.
        Always structure your response with:
        - main_point (clear summary)
        - support_points (list of evidence)
        - confidence_level (0-100)

```


### `main.py`

```py
# /// script
# dependencies = ["llmling-agent"]
# ///

"""Example of structured responses defined both in code and YAML."""

import os

from schemez import Schema

from llmling_agent import Agent, AgentPool, AgentsManifest
from llmling_agent_docs.examples.utils import get_config_path, is_pyodide, run


# set your OpenAI API key here
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "your_api_key_here")


class PythonResult(Schema):
    """Structured response defined in Python."""

    main_point: str
    is_positive: bool


async def run_example() -> None:
    """Show both ways of defining structured responses."""
    # Example 1: Python-defined structure
    agent = Agent(
        model="openai:gpt-5-mini",
        system_prompt="Summarize text in a structured way.",
        output_type=PythonResult,
    )
    async with agent as summarizer:
        result = await summarizer.run("I love this new feature!")
        summary = result.data
        print("\nPython-defined Response:")
        print(f"Main point: {summary.main_point}")
        print(f"Is positive: {summary.is_positive}")

    # Example 2: YAML-defined structure
    # NOTE: this is not recommended for programmatic usage and is just a demo. Use this
    # only for complete YAML workflows, otherwise your linter wont like what you are doin.
    config_path = get_config_path(None if is_pyodide() else __file__)
    manifest = AgentsManifest.from_file(config_path)
    async with AgentPool(manifest) as pool:
        analyzer = pool.get_agent("analyzer")
        result_2 = await analyzer.run("I'm really excited about this project!")
        analysis = result_2.data
        print("\nYAML-defined Response:")
        # Type checkers cant deal with dynamically generated Models, so we have to
        # git-ignore
        print(f"Sentiment: {analysis.sentiment}")  # type: ignore
        print(f"Confidence: {analysis.confidence:.2f}")  # type: ignore
        print(f"Mood: {analysis.mood}")  # type: ignore


if __name__ == "__main__":
    run(run_example())

```



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

